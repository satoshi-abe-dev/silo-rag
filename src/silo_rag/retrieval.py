"""Node C: hybrid retrieval (BM25 + vectors) with LLM reranking.

Normalized BM25 and vector scores are blended by `config.retrieval.vector_weight`,
then the chat model reranks the candidates.

No cross-encoder or morphological analyzer: everything runs on the single local LLM
(LM Studio), and BM25 uses a dependency-free tokenizer (character bigrams + ASCII words).
The BM25 index is rebuilt from ChromaDB on every `search()` call; fine for a corpus of
tens to hundreds of chunks.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .config import CHROMA_DIR, COLLECTION_NAME, load_config
from .ingest import Chunk
from .llm_client import LLMClient, LLMConnectionError

# --- Tokenizer ---------------------------------------------------------------

# A run of ASCII alphanumerics (model numbers, English words) is one token.
_ASCII_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def tokenize(text: str) -> list[str]:
    """Tokenize without a morphological analyzer.

    ASCII alphanumeric runs become single lowercase tokens; other non-space runs
    (kanji, kana, symbols) become character bigrams. Word boundaries are approximate,
    but queries go through the same function, so BM25 matching still works.
    """
    tokens: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch.isascii() and ch.isalnum():
            m = _ASCII_TOKEN_RE.match(text, i)
            if m:
                tokens.append(m.group(0).lower())
                i = m.end()
                continue
        if ch.isspace():
            i += 1
            continue
        j = i
        while j < n and not text[j].isspace() and not (text[j].isascii() and text[j].isalnum()):
            j += 1
        span = text[i:j]
        if len(span) == 1:
            tokens.append(span)
        else:
            tokens.extend(span[k : k + 2] for k in range(len(span) - 1))
        i = j
    return tokens


# --- Public interface ----------------------------------------------------------


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float  # combined normalized score; higher is more relevant


def search(
    client: LLMClient,
    query: str,
    *,
    top_k: int | None = None,
    filters: dict[str, str] | None = None,
    history: list[tuple[str, str]] | None = None,
    rewrite_query: bool | None = None,
) -> list[ScoredChunk]:
    """Run hybrid search (BM25 + vectors), blend the scores, then rerank with the LLM.

    top_k: number of results; defaults to config.retrieval.top_k_final.
    filters: exact-match metadata filters, ANDed (e.g. {"dept": "..."}).
    history: prior (question, answer) turns; if given, the query is rewritten into a
        standalone one so follow-up questions retrieve well.
    rewrite_query: if True, have the LLM turn the question into a search query first
        (plan_query), dropping self-introductions and request phrasing and resolving
        references from history. Falls back to the history rewrite on failure.
        Defaults to config.retrieval.rewrite_query.
    """
    config = load_config()
    resolved_top_k = top_k if top_k is not None else config.retrieval.top_k_final
    top_k_candidates = config.retrieval.top_k_candidates
    vector_weight = config.retrieval.vector_weight
    use_rewrite = config.retrieval.rewrite_query if rewrite_query is None else rewrite_query
    planned = plan_query(client, query, history)[0] if use_rewrite else None
    if planned is not None:
        query = planned
    elif history:
        query = _resolve_query(client, query, history)

    import chromadb

    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        collection = chroma_client.get_collection(COLLECTION_NAME)
    except Exception as exc:
        raise RuntimeError(
            f"ChromaDBコレクション「{COLLECTION_NAME}」が見つかりません（{CHROMA_DIR}）。"
            "先にingestを実行してください。"
        ) from exc

    where = _build_where(filters)

    # BM25 scans the whole filtered collection, so no hits means an empty corpus;
    # return early and skip the pointless embedding request.
    bm25_scores, bm25_chunks = _bm25_search(collection, query, top_k_candidates, where)
    if not bm25_chunks:
        return []

    if vector_weight <= 0.0:
        # BM25-only mode: skip vector search so it works even without an embedding model loaded.
        vector_scores: dict[str, float] = {}
        vector_chunks: dict[str, Chunk] = {}
    else:
        vector_scores, vector_chunks = _vector_search(collection, client, query, top_k_candidates, where)

    if vector_weight >= 1.0:
        # Vector-only mode: use only vector candidates. BM25-only candidates (v=0) would tie
        # with the lowest vector hit (also 0 after min-max) and, coming first in insertion
        # order, could push real vector results out of the cutoff.
        chunk_map: dict[str, Chunk] = dict(vector_chunks)
        bm25_scores = {}
    else:
        chunk_map = {**bm25_chunks, **vector_chunks}

    bm25_norm = _normalize_bm25(bm25_scores)
    vector_norm = _normalize_minmax(vector_scores)

    combined: list[tuple[str, float]] = []
    for chunk_id in chunk_map:
        # A chunk missing from one side scores 0 on that side.
        b = bm25_norm.get(chunk_id, 0.0)
        v = vector_norm.get(chunk_id, 0.0)
        score = (1 - vector_weight) * b + vector_weight * v
        combined.append((chunk_id, score))
    combined.sort(key=lambda pair: pair[1], reverse=True)
    combined = combined[:top_k_candidates]

    candidates = [ScoredChunk(chunk=chunk_map[chunk_id], score=score) for chunk_id, score in combined]
    reranked = _llm_rerank(client, query, candidates)
    return reranked[:resolved_top_k]


# --- History-aware query rewrite ------------------------------------------------

_MAX_HISTORY_TURNS = 3
_HISTORY_ANSWER_TRUNCATE = 200

_QUERY_REWRITE_SYSTEM_PROMPT = (
    "あなたは検索クエリの書き換え役です。会話履歴と、それに続く新しい質問を読み、"
    "履歴を見なくても意味が伝わる独立した検索クエリ1文に書き換えてください。"
    "新しい質問がそれ単体で意味が通る場合（履歴に依存しない新しい話題の場合）は、"
    "書き換えずそのまま返してください。"
    "出力は書き換え後の質問文のみとし、説明や前置き・引用符は一切含めないでください。"
)


def _build_query_rewrite_prompt(question: str, history: list[tuple[str, str]]) -> str:
    lines = ["会話履歴:"]
    for q, a in history[-_MAX_HISTORY_TURNS:]:
        truncated = a if len(a) <= _HISTORY_ANSWER_TRUNCATE else a[:_HISTORY_ANSWER_TRUNCATE] + "…"
        lines.append(f"Q: {q}")
        lines.append(f"A: {truncated}")
    lines.append("")
    lines.append(f"新しい質問: {question}")
    return "\n".join(lines)


def _resolve_query(client: LLMClient, question: str, history: list[tuple[str, str]]) -> str:
    """Rewrite a follow-up question into a standalone search query using the history.

    Questions like "tell me more about that" have too few search terms on their own.
    Uses temperature 0 for stable output. On LLM failure, returns the original question
    so search still runs.
    """
    if not history:
        return question
    prompt = _build_query_rewrite_prompt(question, history)
    try:
        rewritten = client.chat(_QUERY_REWRITE_SYSTEM_PROMPT, prompt, temperature=0.0)
    except LLMConnectionError:
        return question
    return rewritten.strip() or question


# --- Query planning (turn a question into a search query) ----------------------


def build_history_lines(history: list[tuple[str, str]] | None) -> list[str]:
    """Format the last few history turns as prompt lines (empty if no history)."""
    if not history:
        return []
    lines = ["会話履歴（指示語の解決にのみ使う）:"]
    for q, a in history[-_MAX_HISTORY_TURNS:]:
        truncated = a if len(a) <= _HISTORY_ANSWER_TRUNCATE else a[:_HISTORY_ANSWER_TRUNCATE] + "…"
        lines.append(f"Q: {q}")
        lines.append(f"A: {truncated}")
    lines.append("")
    return lines


PLAN_SYSTEM_PROMPT = (
    "あなたは社内ナレッジ検索の検索クエリを作る係です。"
    "質問を読み、社内のプロジェクト知見レポートを検索するための検索クエリを1文だけ作ってください。"
    "質問者が名乗っている所属部署名や、「教えてください」「ありますか」などの依頼の言い回しは、"
    "クエリに含めないでください（探したいのは他部署を含む全部署の事例であり、検索に関係の無い言葉が"
    "混ざるとキーワード検索が散るため）。"
    "探したい事例のテーマ・施策の種類・失敗パターンなど、事例の中身を表す言葉だけで書いてください。"
    "会話履歴が付いている場合は、質問中の指示語（「それ」「さっきの」等）の解決にだけ使ってください。"
    "出力は検索クエリの文のみとし、説明や前置き・引用符は一切含めないでください。"
)


def build_plan_prompt(question: str, history: list[tuple[str, str]] | None) -> str:
    return "\n".join([*build_history_lines(history), f"質問: {question}", "", "検索クエリ:"])


def parse_query_response(raw: str) -> str | None:
    """Return the first non-empty line with surrounding quotes stripped, or None."""
    for line in raw.splitlines():
        query = line.strip().strip("「」\"'").strip()
        if query:
            return query
    return None


def plan_query(
    client: LLMClient, question: str, history: list[tuple[str, str]] | None = None
) -> tuple[str | None, str | None]:
    """Have the LLM turn a question into a search query; returns (query, failure_reason).

    Self-introductions and request phrasing in questions scatter BM25 bigram matches.
    Never raises: returns (None, reason) on failure so callers can search with the raw question.
    """
    prompt = build_plan_prompt(question, history)
    try:
        raw = client.chat(PLAN_SYSTEM_PROMPT, prompt, temperature=0.0)
    except LLMConnectionError:
        return None, "LLM呼び出しに失敗"
    query = parse_query_response(raw)
    if query is None:
        return None, "応答が空"
    return query, None


# --- BM25 candidates -----------------------------------------------------------


def _build_where(filters: dict[str, str] | None) -> dict | None:
    """Convert filters to a ChromaDB where clause.

    Some ChromaDB versions reject multi-key dicts, so multiple keys are wrapped in $and.
    """
    if not filters:
        return None
    if len(filters) == 1:
        return dict(filters)
    return {"$and": [{k: v} for k, v in filters.items()]}


def _clean_metadata(raw: dict | None) -> dict[str, str]:
    """Coerce loosely typed ChromaDB metadata to dict[str, str]."""
    return {k: str(v) for k, v in (raw or {}).items()}


def _bm25_search(
    collection,
    query: str,
    top_n: int,
    where: dict | None,
) -> tuple[dict[str, float], dict[str, Chunk]]:
    """Score every chunk in the filtered collection with BM25 and return the top_n."""
    # BM25Plus, not BM25Okapi: Okapi's IDF goes negative for terms in over half the documents,
    # and on a small corpus its epsilon * average_idf floor can stay negative too, ranking
    # matches below non-matches. Small filtered corpora and common bigrams make this realistic.
    from rank_bm25 import BM25Plus

    got = collection.get(where=where, include=["documents", "metadatas"])
    ids: list[str] = got.get("ids") or []
    if not ids:
        return {}, {}
    documents: list[str] = got.get("documents") or []
    metadatas: list[dict] = got.get("metadatas") or []

    tokenized_corpus = [tokenize(doc or "") for doc in documents]
    bm25 = BM25Plus(tokenized_corpus)
    raw_scores = bm25.get_scores(tokenize(query))

    ranked_indices = sorted(range(len(ids)), key=lambda i: raw_scores[i], reverse=True)[:top_n]
    scores = {ids[i]: float(raw_scores[i]) for i in ranked_indices}
    chunk_map = {
        ids[i]: Chunk(chunk_id=ids[i], text=documents[i] or "", metadata=_clean_metadata(metadatas[i]))
        for i in ranked_indices
    }
    return scores, chunk_map


# --- Vector candidates ---------------------------------------------------------


def _vector_search(
    collection,
    client: LLMClient,
    query: str,
    top_n: int,
    where: dict | None,
) -> tuple[dict[str, float], dict[str, Chunk]]:
    """Embed the query and return the top_n ChromaDB vector matches."""
    embeddings = client.embed([query])
    if not embeddings:
        return {}, {}

    result = collection.query(
        query_embeddings=embeddings,
        n_results=top_n,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    ids: list[str] = (result.get("ids") or [[]])[0]
    if not ids:
        return {}, {}
    documents: list[str] = (result.get("documents") or [[]])[0]
    metadatas: list[dict] = (result.get("metadatas") or [[]])[0]
    distances: list[float] = (result.get("distances") or [[]])[0]

    # Negate distances so higher means more relevant.
    scores = {chunk_id: -float(dist) for chunk_id, dist in zip(ids, distances, strict=True)}
    chunk_map = {
        chunk_id: Chunk(chunk_id=chunk_id, text=documents[i] or "", metadata=_clean_metadata(metadatas[i]))
        for i, chunk_id in enumerate(ids)
    }
    return scores, chunk_map


# --- Score normalization -------------------------------------------------------


def _normalize_minmax(scores: dict[str, float]) -> dict[str, float]:
    """Min-max normalize to [0, 1]; if all values are equal, every score becomes 1.0."""
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return dict.fromkeys(scores, 1.0)
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def _normalize_bm25(scores: dict[str, float]) -> dict[str, float]:
    """Normalize BM25 scores, keeping all zeros when nothing matched.

    Plain min-max would lift an all-zero (no keyword hit) set to 1.0.
    """
    if not scores:
        return {}
    values = list(scores.values())
    hi = max(values)
    if hi <= 0:
        return dict.fromkeys(scores, 0.0)
    return _normalize_minmax(scores)


# --- LLM reranking -------------------------------------------------------------

_RERANK_TEXT_TRUNCATE = 300

_RERANK_SYSTEM_PROMPT = (
    "あなたは社内向けプロジェクト知見レポート検索システムのリランカーです。"
    "与えられた質問と候補チャンクの一覧を読み、質問への関連度が高い順に候補番号を並べ替えてください。"
    "出力は候補番号のJSON配列のみとし、説明文やコードブロック記法など余計な文字列は一切含めないでください。"
    '例: [3, 1, 5]'
)


def _build_rerank_prompt(query: str, candidates: list[ScoredChunk]) -> str:
    lines = [f"質問: {query}", "", "候補チャンク一覧:"]
    for i, scored in enumerate(candidates, start=1):
        meta = scored.chunk.metadata
        header = (
            f"[{i}] report_id={meta.get('report_id', '?')} "
            f"section={meta.get('section', '?')} dept={meta.get('dept', '?')}"
        )
        text = scored.chunk.text
        if len(text) > _RERANK_TEXT_TRUNCATE:
            text = text[:_RERANK_TEXT_TRUNCATE] + "…"
        lines.append(header)
        lines.append(text)
        lines.append("")
    lines.append(f"以上{len(candidates)}件の番号を、質問への関連度が高い順にJSON配列で出力してください。")
    return "\n".join(lines)


def _parse_rerank_response(raw: str, n_candidates: int) -> list[int] | None:
    """Extract the JSON array of candidate numbers from the model response.

    Returns None if no array is found, the JSON is broken, or no number is usable;
    the caller then falls back to the blended-score order.
    """
    match = re.search(r"\[[^\[\]]*\]", raw, re.DOTALL)
    if match is None:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list):
        return None

    order: list[int] = []
    seen: set[int] = set()
    for item in parsed:
        try:
            idx = int(item)
        except (TypeError, ValueError):
            continue
        if 1 <= idx <= n_candidates and idx not in seen:
            seen.add(idx)
            order.append(idx)
    return order or None


def _llm_rerank(client: LLMClient, query: str, candidates: list[ScoredChunk]) -> list[ScoredChunk]:
    """Rerank candidates with the LLM, falling back to blended-score order on failure."""
    if not candidates:
        return []

    prompt = _build_rerank_prompt(query, candidates)
    try:
        raw = client.chat(_RERANK_SYSTEM_PROMPT, prompt, temperature=0.0)
    except LLMConnectionError:
        # Intentional fallback: still return results in blended-score order.
        # Only LLMConnectionError is caught.
        return candidates

    order = _parse_rerank_response(raw, len(candidates))
    if order is None:
        # Unparseable response: same intentional fallback.
        return candidates

    reranked = [candidates[i - 1] for i in order]
    # Append any candidates the model omitted, in original order, so none are dropped.
    included = set(order)
    reranked.extend(scored for i, scored in enumerate(candidates, start=1) if i not in included)
    return reranked


def rerank(client: LLMClient, query: str, candidates: list[ScoredChunk]) -> list[ScoredChunk]:
    """Rerank candidates gathered outside search() (e.g. merged results in agent.py).

    Same reranking as search(); returns the input order on failure.
    """
    return _llm_rerank(client, query, candidates)
