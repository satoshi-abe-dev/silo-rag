"""ハイブリッド検索 + LLMリランキング（DAGノードC）。

BM25（キーワード検索）とベクトル類似度検索をそれぞれ独立に走らせ、正規化したスコアを
`config.retrieval.vector_weight` で重み付け合成して候補集合を作る。その後、候補を
LM Studio経由のチャットモデルにリランキングさせて最終順位を決める。

cross-encoderやsentence-transformers等の追加重量級依存は使わない（「ローカルLLM1本
（LM Studio）で完結させる」という本プロジェクトの方針のため）。同様に、形態素解析器
（mecab/janome/sudachi等）も使わず、依存なしの簡易トークナイザ（文字2-gram + ASCII語）
でBM25を成立させる。

データ規模がポートフォリオ相当（数十〜数百チャンク）であることを前提に、BM25インデックス
は`search()`呼び出しのたびにChromaDBコレクションから読み直して作り直す（永続化・キャッシュ
は行わない）。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .config import CHROMA_DIR, COLLECTION_NAME, load_config
from .ingest import Chunk
from .llm_client import LLMClient, LLMConnectionError

# --- トークナイザ ------------------------------------------------------------

# ASCII英数字（型番・英単語など）は連続した塊をそのまま1トークンとして扱う。
_ASCII_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def tokenize(text: str) -> list[str]:
    """形態素解析なしの簡易トークナイザ。

    - ASCII英数字の連続（型番・英単語等）はそのまま1トークンにする。
    - それ以外（漢字・かな・記号等）の連続範囲は文字2-gramに分解する。
      形態素解析器に依存しないための実用的な妥協。単語境界を厳密には捉えないが、
      クエリ側にも同じ関数を通すためBM25の一致判定としては十分機能する。
    - 空白はトークンの区切りとしてのみ使い、トークン自体には含めない。
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
        # 非ASCII・非空白の連続範囲（漢字・かな・カナ・全角記号など）をまとめて2-gram化する。
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


# --- 公開インターフェース -----------------------------------------------------


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float  # 正規化後の統合スコア（高いほど関連度が高い）


def search(
    client: LLMClient,
    query: str,
    *,
    top_k: int | None = None,
    filters: dict[str, str] | None = None,
    history: list[tuple[str, str]] | None = None,
) -> list[ScoredChunk]:
    """ハイブリッド検索（BM25＋ベクトル）→ スコア統合 → LLMリランキングを行う。

    top_k: 最終的に返す件数。Noneならconfig.retrieval.top_k_finalを使う。
    filters: メタデータの完全一致フィルタ（例: {"dept": "マーケティング部"}）。複数キー指定時はAND条件。
    history: 直前までの会話（質問, 回答本文）のリスト。指定すると、検索前にqueryを
        履歴込みで独立した検索クエリに書き換える（フォローアップ質問での検索精度向上のため）。
    """
    config = load_config()
    resolved_top_k = top_k if top_k is not None else config.retrieval.top_k_final
    top_k_candidates = config.retrieval.top_k_candidates
    vector_weight = config.retrieval.vector_weight
    query = _resolve_query(client, query, history) if history else query

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

    # BM25側はfilters適用後のコレクション全件をcollection.get()で取得するため、ここが0件なら
    # 「（フィルタ後の）対象コーパスが空」を意味する。ベクトル検索をしても結果は必ず0件になる
    # ため、無駄な埋め込みリクエスト（LM Studioへの依存）を発生させずに早期returnする。
    bm25_scores, bm25_chunks = _bm25_search(collection, query, top_k_candidates, where)
    if not bm25_chunks:
        return []

    if vector_weight <= 0.0:
        # config.retrieval.vector_weight=0.0は「BM25のみ」を意味する設定。この場合は
        # 埋め込みモデルへの依存自体を発生させないよう、ベクトル検索を呼び出さない
        # （呼ぶと埋め込みモデル未ロード時にBM25のみのモードまで失敗してしまう）。
        vector_scores: dict[str, float] = {}
        vector_chunks: dict[str, Chunk] = {}
    else:
        vector_scores, vector_chunks = _vector_search(collection, client, query, top_k_candidates, where)

    if vector_weight >= 1.0:
        # ベクトルのみを意味する設定。ここでbm25_chunksを候補集合に混ぜると、ベクトル側に
        # 出てこないBM25専用候補が「v=0（不在）」のままベクトル最下位候補（min-max正規化で
        # 同じく0点になりうる）と同点になり、安定ソート＋件数制限のときにbm25_chunksが
        # 挿入順で先に来るぶん本来のベクトル候補を押し出してしまうことがある。ベクトルのみの
        # 意図を守るため、候補集合はベクトル結果だけに絞る（bm25_scoresは
        # 「対象コーパスが空か」の判定にのみ使い、スコア統合には使わない）。
        chunk_map: dict[str, Chunk] = dict(vector_chunks)
        bm25_scores = {}
    else:
        chunk_map = {**bm25_chunks, **vector_chunks}

    bm25_norm = _normalize_bm25(bm25_scores)
    vector_norm = _normalize_minmax(vector_scores)

    combined: list[tuple[str, float]] = []
    for chunk_id in chunk_map:
        # 片方の候補集合にしか出てこないチャンクは、出ていない側のスコアを0として扱う。
        b = bm25_norm.get(chunk_id, 0.0)
        v = vector_norm.get(chunk_id, 0.0)
        score = (1 - vector_weight) * b + vector_weight * v
        combined.append((chunk_id, score))
    combined.sort(key=lambda pair: pair[1], reverse=True)
    combined = combined[:top_k_candidates]

    candidates = [ScoredChunk(chunk=chunk_map[chunk_id], score=score) for chunk_id, score in combined]
    reranked = _llm_rerank(client, query, candidates)
    return reranked[:resolved_top_k]


# --- 会話履歴を踏まえたクエリ書き換え -------------------------------------------

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
    """会話履歴を踏まえて、質問を検索エンジンにかけられる独立したクエリに書き換える。

    「それについてもう少し詳しく」のような指示語頼みの質問は、そのままBM25/ベクトル検索に
    かけても検索語が乏しく空振りする。LLMに履歴込みで読ませ、履歴が無くても意味が通る
    検索クエリに書き換えさせる（rerankと同じ理由でtemperature=0.0にし、書き換えのブレを抑える）。

    LLM呼び出しに失敗した場合は書き換えを諦め、元の質問文をそのまま返す
    （検索自体は動かしたいので、ここでの失敗によって検索全体を止めない）。
    """
    if not history:
        return question
    prompt = _build_query_rewrite_prompt(question, history)
    try:
        rewritten = client.chat(_QUERY_REWRITE_SYSTEM_PROMPT, prompt, temperature=0.0)
    except LLMConnectionError:
        return question
    return rewritten.strip() or question


# --- BM25候補検索 -------------------------------------------------------------


def _build_where(filters: dict[str, str] | None) -> dict | None:
    """filtersをChromaDBのwhere句に変換する。

    ChromaDBのwhere句は複数キーを単純な辞書として渡すことを許さないバージョンがあるため、
    2キー以上のときは明示的に$andでまとめる。
    """
    if not filters:
        return None
    if len(filters) == 1:
        return dict(filters)
    return {"$and": [{k: v} for k, v in filters.items()]}


def _clean_metadata(raw: dict | None) -> dict[str, str]:
    """ChromaDBから返るメタデータ（値の型が緩い）をdict[str, str]に揃える。"""
    return {k: str(v) for k, v in (raw or {}).items()}


def _bm25_search(
    collection,
    query: str,
    top_n: int,
    where: dict | None,
) -> tuple[dict[str, float], dict[str, Chunk]]:
    """コレクション（filters適用後）全件に対してBM25を計算し、上位top_n件を返す。"""
    # BM25Okapiではなく意図的にBM25Plusを使う。BM25Okapiは「半数以上の文書に出現する語」の
    # IDFが負になり得て、その際のフロア処理（epsilon * average_idf）もコーパスが小さいと
    # 平均IDF自体が負に転ぶため負のままになりうる。結果、マッチした文書がノーマッチの文書
    # （スコア0）より下位に沈むという転倒が起こる。本プロジェクトの想定コーパス（数十〜数百
    # チャンク、部署フィルタでさらに小さくなる）や文字2-gramトークナイザ（頻出2-gramが半数
    # 以上の文書に出現しやすい）では現実的に起こりうるため、IDFが常に正であるBM25Plusを使う。
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


# --- ベクトル候補検索 ----------------------------------------------------------


def _vector_search(
    collection,
    client: LLMClient,
    query: str,
    top_n: int,
    where: dict | None,
) -> tuple[dict[str, float], dict[str, Chunk]]:
    """クエリを埋め込み、ChromaDBのベクトル類似度検索で上位top_n件を返す。"""
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

    # 距離は小さいほど類似（=関連度が高い）なので符号を反転し、「大きいほど良い」に揃える。
    scores = {chunk_id: -float(dist) for chunk_id, dist in zip(ids, distances, strict=True)}
    chunk_map = {
        chunk_id: Chunk(chunk_id=chunk_id, text=documents[i] or "", metadata=_clean_metadata(metadatas[i]))
        for i, chunk_id in enumerate(ids)
    }
    return scores, chunk_map


# --- スコア正規化 --------------------------------------------------------------


def _normalize_minmax(scores: dict[str, float]) -> dict[str, float]:
    """min-max正規化で[0,1]に写像する。全件同値なら1.0（＝差がない＝等しく採用）とする。"""
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return dict.fromkeys(scores, 1.0)
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def _normalize_bm25(scores: dict[str, float]) -> dict[str, float]:
    """BM25スコア用の正規化。

    BM25スコアは0以上で「0=一致なし」という意味を持つため、通常のmin-max正規化を
    そのまま使うと「候補全員が0点（＝誰もキーワードにヒットしていない）」場合に
    全員を1.0（最高評価）へ底上げしてしまう。それを避け、最大値が0以下（＝ヒットなし）
    のときは全員0.0のままにする。
    """
    if not scores:
        return {}
    values = list(scores.values())
    hi = max(values)
    if hi <= 0:
        return dict.fromkeys(scores, 0.0)
    return _normalize_minmax(scores)


# --- LLMリランキング -----------------------------------------------------------

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
    """モデル応答から候補番号のJSON配列を抜き出す。

    失敗時（配列が見つからない・JSONとして壊れている・使える番号が1つもない）はNoneを返す。
    呼び出し側はこれを「意図的なフォールバック」の合図として使い、統合スコア順にフォールバックする。
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
    """候補チャンクをLLMにリランキングさせる。失敗時は統合スコア順にフォールバックする。"""
    if not candidates:
        return []

    prompt = _build_rerank_prompt(query, candidates)
    try:
        raw = client.chat(_RERANK_SYSTEM_PROMPT, prompt, temperature=0.0)
    except LLMConnectionError:
        # LLM呼び出し自体の失敗（サーバー未起動・タイムアウト等）。検索結果自体は返したいので、
        # リランキング前の統合スコア順にフォールバックする（意図的な代替パス。黙って例外を
        # 握りつぶしているわけではなく、LLMConnectionErrorのみを明示的に捕捉している）。
        return candidates

    order = _parse_rerank_response(raw, len(candidates))
    if order is None:
        # JSON配列の抽出・解釈に失敗した場合の意図的なフォールバック。上と同様。
        return candidates

    reranked = [candidates[i - 1] for i in order]
    # モデルが一部の番号しか返さなかった場合、漏れた候補を元の順序のまま末尾に補う
    # （黙って取りこぼさない）。
    included = set(order)
    reranked.extend(scored for i, scored in enumerate(candidates, start=1) if i not in included)
    return reranked
