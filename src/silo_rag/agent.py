"""Node G: iterative-retrieval RAG agent (LangGraph).

Calls search() (node C) and answer_question() (node D) only through their public functions,
so C and D stay independent and the module graph remains a DAG; the loop exists only at runtime.

Runtime graph:

    START → retrieve → grade ─(sufficient, or attempt limit reached)→ select → generate → END
               ↑         └─(insufficient, under limit)→ rewrite ─(rewrite failed)→ select
               └──────────(new query)──────────┘

With first_query="rewrite", a plan node builds the first query from the question; "raw" searches
with the question as-is. select re-ranks the whole pool against the original question only after
multiple searches (a single search() already re-ranks).

The LLM decides sufficiency and the next query; code caps the loop (config.agent.max_attempts).
Tool calling is unreliable on ~7B local models, so this is not a free-form ReAct agent: the LLM
answers in fixed formats (one word / one query) that code parses strictly.

Failures in the auxiliary grade/rewrite steps never stop the run; we fall through to answering with
the chunks at hand. Failures in answer_question itself propagate to the caller.

LangGraph sends nothing externally unless LangSmith env vars (LANGSMITH_TRACING etc.) are set,
and all LLM calls go through the local LLMClient.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from .config import load_config
from .generation import Answer, answer_question
from .llm_client import LLMClient, LLMConnectionError
from .retrieval import ScoredChunk, build_history_lines, parse_query_response, plan_query, rerank, search

# Max chunk chars shown to the grader; the gist is enough to judge.
_GRADE_CHUNK_TRUNCATE = 300


class AgentState(TypedDict):
    question: str
    history: list[tuple[str, str]] | None
    filters: dict[str, str] | None
    tried_queries: list[str]  # prevents re-running the same query
    results_by_attempt: list[list[ScoredChunk]]  # newest attempt first
    scored_chunks: list[ScoredChunk]  # results_by_attempt flattened by _merge_chunks
    attempts: int
    verdict: Literal["sufficient", "insufficient"] | None
    next_query: str | None
    answer: Answer | None
    trace: list[str]  # shows what the agent did, for the UI and eval


@dataclass
class AgentResult:
    answer: Answer
    scored_chunks: list[ScoredChunk]  # chunks passed to generation, by relevance
    tried_queries: list[str]
    attempts: int
    trace: list[str]


# --- Accumulating results -------------------------------------------------------


def _merge_chunks(results_by_attempt: list[list[ScoredChunk]]) -> list[ScoredChunk]:
    """Interleave attempts rank by rank (newest first within a rank), keeping the first duplicate.

    Simply prepending the latest results would push out all earlier evidence once the latest attempt
    returns top_k hits; interleaving keeps every attempt's top hits. Scores aren't used because
    search() normalizes them per query, so they aren't comparable across queries.
    """
    merged: list[ScoredChunk] = []
    seen: set[str] = set()
    longest = max((len(r) for r in results_by_attempt), default=0)
    for rank in range(longest):
        for results in results_by_attempt:
            if rank >= len(results):
                continue
            sc = results[rank]
            if sc.chunk.chunk_id in seen:
                continue
            seen.add(sc.chunk.chunk_id)
            merged.append(sc)
    return merged


# --- Grading (is the evidence sufficient?) ---------------------------------------

_GRADE_PROMPT_HEAD = (
    "あなたは社内ナレッジ検索の検索結果を点検する係です。"
    "質問と検索で見つかった資料の抜粋を読み、質問に答えるための根拠が資料に含まれているかを判定してください。"
    "会話履歴が付いている場合は、質問中の指示語（「それ」「さっきの」等）が何を指すかの解決にだけ使ってください。"
)
_GRADE_PROMPT_TAIL = "出力は SUFFICIENT か INSUFFICIENT のどちらか1語のみとし、説明は一切含めないでください。"

# Grading strictness (config.agent.grade_mode). Both had drawbacks on real data with a 7B model:
#   strict:  8 of 9 "insufficient"; nearly always hits the limit (~2x time, 4-5x LLM calls),
#            but sometimes finds answers the baseline misses.
#   lenient: says "sufficient" quickly, so often matches the baseline.
# The grader can't know about better unseen documents, so compare both with eval.py.
_GRADE_SYSTEM_PROMPTS: dict[str, str] = {
    "strict": (
        _GRADE_PROMPT_HEAD
        + "根拠が含まれていれば SUFFICIENT、含まれていなければ INSUFFICIENT と答えてください。"
        + _GRADE_PROMPT_TAIL
    ),
    "lenient": (
        _GRADE_PROMPT_HEAD
        + "資料は、質問と全く同じ案件・部署・プロジェクト種別である必要はありません。"
        + "テーマや失敗パターンが似ていて参考になる事例（他部署の事例を含む）が"
        + "1件でも含まれていれば SUFFICIENT、参考になる事例が1件も無ければ INSUFFICIENT と答えてください。"
        + _GRADE_PROMPT_TAIL
    ),
}
GRADE_MODES = tuple(_GRADE_SYSTEM_PROMPTS)
FIRST_QUERY_MODES = ("raw", "rewrite")


def _build_grade_prompt(
    question: str, chunks: list[ScoredChunk], history: list[tuple[str, str]] | None = None
) -> str:
    # Follow-up questions can't be graded without history, so pass it like the other steps do.
    lines = [*build_history_lines(history), f"質問: {question}", "", "検索で見つかった資料の抜粋:"]
    for i, sc in enumerate(chunks, start=1):
        meta = sc.chunk.metadata
        text = sc.chunk.text
        if len(text) > _GRADE_CHUNK_TRUNCATE:
            text = text[:_GRADE_CHUNK_TRUNCATE] + "…"
        lines.append(
            f"[{i}] report_id={meta.get('report_id', '不明')} / 部署={meta.get('dept', '不明')} / "
            f"セクション={meta.get('section', '不明')}"
        )
        lines.append(text)
    lines.append("")
    lines.append("判定(SUFFICIENT か INSUFFICIENT の1語のみ):")
    return "\n".join(lines)


def _parse_grade_response(raw: str) -> Literal["sufficient", "insufficient"] | None:
    """Parse only an exact SUFFICIENT/INSUFFICIENT reply; None otherwise.

    Exact match (after trimming whitespace and a trailing period), since a substring search would
    find "SUFFICIENT" inside "INSUFFICIENT".
    """
    word = raw.strip().rstrip("。.").strip().upper()
    if word == "SUFFICIENT":
        return "sufficient"
    if word == "INSUFFICIENT":
        return "insufficient"
    return None


# --- Query rewriting --------------------------------------------------------------

_REWRITE_SYSTEM_PROMPT = (
    "あなたは社内ナレッジ検索の検索クエリを考える係です。"
    "これまでのクエリでは質問に答える根拠が見つかりませんでした。"
    "これまでと違う切り口（言い換え・上位概念・関連する失敗パターン・他部署で使われそうな用語など）で、"
    "新しい検索クエリを1文だけ作ってください。"
    "質問者が自分の所属部署を名乗っていても、その部署名はクエリに含めないでください"
    "（探したいのは他部署を含む全部署の事例であり、部署名を入れると質問者の部署の資料に検索が偏るため）。"
    "テーマ・施策の種類・失敗パターンなど、事例の中身を表す言葉で探してください。"
    "出力は検索クエリの文のみとし、説明や前置き・引用符は一切含めないでください。"
)


def _build_rewrite_prompt(
    question: str, tried_queries: list[str], history: list[tuple[str, str]] | None
) -> str:
    lines = build_history_lines(history)
    lines.append(f"質問: {question}")
    lines.append("")
    lines.append("これまでに試したクエリ:")
    lines.extend(f"- {q}" for q in tried_queries)
    lines.append("")
    lines.append("新しい検索クエリ:")
    return "\n".join(lines)


def _normalize_query(query: str) -> str:
    return "".join(query.split()).lower()


def _build_rerank_query(question: str, history: list[tuple[str, str]] | None) -> str:
    """Query for the final re-rank; prepends the previous question so follow-ups make sense."""
    if not history:
        return question
    return f"{history[-1][0]}（に続けて）{question}"


# --- Graph ------------------------------------------------------------------------


def build_graph(
    client: LLMClient,
    *,
    max_attempts: int,
    top_k: int,
    grade_mode: str = "lenient",
    first_query: str = "raw",
):
    """Build and compile the agent graph.

    client is captured by closures, not stored in State, because LLMClient isn't serializable.
    grade_mode: "strict" or "lenient" (see _GRADE_SYSTEM_PROMPTS).
    first_query: "raw" searches with the question; "rewrite" has the LLM build the first query.
    """
    if grade_mode not in _GRADE_SYSTEM_PROMPTS:
        raise ValueError(f"grade_modeは{GRADE_MODES}のいずれかを指定してください: {grade_mode!r}")
    if first_query not in FIRST_QUERY_MODES:
        raise ValueError(f"first_queryは{FIRST_QUERY_MODES}のいずれかを指定してください: {first_query!r}")
    grade_system_prompt = _GRADE_SYSTEM_PROMPTS[grade_mode]

    def plan(state: AgentState) -> dict:
        # Raw questions carry noise (self-introductions, polite phrasing) that scatters character-bigram
        # BM25. On failure, next_query stays None and retrieve searches with the raw question.
        query, failure = plan_query(client, state["question"], state["history"])
        if query is None:
            return {"trace": [*state["trace"], f"クエリ作成: {failure}。質問のまま検索"]}
        return {"next_query": query, "trace": [*state["trace"], f"クエリ作成: 「{query}」"]}

    def retrieve(state: AgentState) -> dict:
        attempts = state["attempts"] + 1
        if state["next_query"] is None and state["attempts"] == 0:
            # Same as the baseline: search() folds any history into a standalone query.
            query = state["question"]
            latest = search(
                client,
                query,
                top_k=top_k,
                filters=state["filters"],
                history=state["history"],
                rewrite_query=False,
            )
        else:
            # plan/rewrite already used the history to build this query.
            query = state["next_query"] or state["question"]
            latest = search(client, query, top_k=top_k, filters=state["filters"], rewrite_query=False)
        results_by_attempt = [latest, *state["results_by_attempt"]]
        return {
            "attempts": attempts,
            "tried_queries": [*state["tried_queries"], query],
            "results_by_attempt": results_by_attempt,
            "scored_chunks": _merge_chunks(results_by_attempt),
            "next_query": None,
            "trace": [*state["trace"], f"検索{attempts}回目「{query}」→ {len(latest)}件"],
        }

    def grade(state: AgentState) -> dict:
        if state["attempts"] >= max_attempts:
            # No retries left, so the verdict wouldn't change anything; skip the LLM call.
            return {"verdict": None, "trace": [*state["trace"], "判定: 省略（検索の上限に達したため）"]}
        chunks = state["scored_chunks"][:top_k]
        if not chunks:
            # No evidence at all; insufficient without asking the LLM.
            return {"verdict": "insufficient", "trace": [*state["trace"], "判定: 不十分（検索結果0件）"]}
        prompt = _build_grade_prompt(state["question"], chunks, state["history"])
        try:
            raw = client.chat(grade_system_prompt, prompt, temperature=0.0)
        except LLMConnectionError:
            return {
                "verdict": "sufficient",
                "trace": [*state["trace"], "判定: LLM呼び出しに失敗。十分とみなした"],
            }
        verdict = _parse_grade_response(raw)
        if verdict is None:
            # Treat malformed replies as sufficient so they don't trigger pointless retries.
            return {
                "verdict": "sufficient",
                "trace": [*state["trace"], "判定: 応答を解釈できず。十分とみなした"],
            }
        label = "十分" if verdict == "sufficient" else "不十分"
        return {"verdict": verdict, "trace": [*state["trace"], f"判定: {label}"]}

    def route_after_grade(state: AgentState) -> Literal["rewrite", "select"]:
        if state["verdict"] == "sufficient":
            return "select"
        if state["attempts"] >= max_attempts:
            return "select"
        return "rewrite"

    def rewrite(state: AgentState) -> dict:
        prompt = _build_rewrite_prompt(state["question"], state["tried_queries"], state["history"])
        try:
            raw = client.chat(_REWRITE_SYSTEM_PROMPT, prompt, temperature=0.0)
        except LLMConnectionError:
            return {"next_query": None, "trace": [*state["trace"], "書き直し: LLM呼び出しに失敗。打ち切り"]}
        query = parse_query_response(raw)
        if query is None:
            return {"next_query": None, "trace": [*state["trace"], "書き直し: 応答が空。打ち切り"]}
        if _normalize_query(query) in {_normalize_query(q) for q in state["tried_queries"]}:
            # Re-running the same query would return the same results.
            return {
                "next_query": None,
                "trace": [*state["trace"], f"書き直し: 「{query}」は試行済み。打ち切り"],
            }
        return {"next_query": query, "trace": [*state["trace"], f"書き直し: 「{query}」"]}

    def route_after_rewrite(state: AgentState) -> Literal["retrieve", "select"]:
        return "retrieve" if state["next_query"] else "select"

    def select(state: AgentState) -> dict:
        # A single search() has already re-ranked against the question.
        if state["attempts"] <= 1:
            return {}
        # Cutting the interleaved list at top_k keeps only each attempt's top hits and drops the
        # original query's runners-up (this lost answers the baseline found on real data), so
        # re-rank the whole pool against the original question; generate trims to top_k.
        query = _build_rerank_query(state["question"], state["history"])
        candidates = state["scored_chunks"]
        reranked = rerank(client, query, candidates)
        return {
            "scored_chunks": reranked,
            "trace": [*state["trace"], f"最終選定: 集めた{len(candidates)}件を元の質問で並べ直し"],
        }

    def generate(state: AgentState) -> dict:
        chunks = [sc.chunk for sc in state["scored_chunks"][:top_k]]
        answer = answer_question(client, state["question"], chunks, history=state["history"])
        return {"answer": answer, "trace": [*state["trace"], "回答生成"]}

    graph = StateGraph(AgentState)
    graph.add_node("plan", plan)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade", grade)
    graph.add_node("rewrite", rewrite)
    graph.add_node("select", select)
    graph.add_node("generate", generate)
    graph.add_edge(START, "plan" if first_query == "rewrite" else "retrieve")
    graph.add_edge("plan", "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", route_after_grade, {"rewrite": "rewrite", "select": "select"})
    graph.add_conditional_edges("rewrite", route_after_rewrite, {"retrieve": "retrieve", "select": "select"})
    graph.add_edge("select", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


def run_agent(
    client: LLMClient,
    question: str,
    *,
    filters: dict[str, str] | None = None,
    history: list[tuple[str, str]] | None = None,
    top_k: int | None = None,
    max_attempts: int | None = None,
    grade_mode: str | None = None,
    first_query: str | None = None,
) -> AgentResult:
    """Run the agent for one question. Other args match search() and answer_question().

    top_k: chunks used for grading and answering (default: config.retrieval.top_k_final).
    max_attempts: max searches, including the first (default: config.agent.max_attempts).
    grade_mode: "strict" or "lenient" (default: config.agent.grade_mode).
    first_query: "raw" or "rewrite" (default: config.agent.first_query).
    """
    config = load_config()
    resolved_top_k = top_k if top_k is not None else config.retrieval.top_k_final
    resolved_max = max_attempts if max_attempts is not None else config.agent.max_attempts
    resolved_max = max(1, resolved_max)

    resolved_mode = grade_mode if grade_mode is not None else config.agent.grade_mode

    resolved_first = first_query if first_query is not None else config.agent.first_query

    app = build_graph(
        client,
        max_attempts=resolved_max,
        top_k=resolved_top_k,
        grade_mode=resolved_mode,
        first_query=resolved_first,
    )
    initial: AgentState = {
        "question": question,
        "history": history,
        "filters": filters,
        "tried_queries": [],
        "results_by_attempt": [],
        "scored_chunks": [],
        "attempts": 0,
        "verdict": None,
        "next_query": None,
        "answer": None,
        "trace": [],
    }
    # Max steps: plan 1 + retrieve/grade max_attempts each + rewrite max_attempts-1 + select/generate 1 each
    # = 3*max_attempts+2. A tight LangGraph recursion limit is a backstop against infinite loops.
    final = app.invoke(initial, config={"recursion_limit": 3 * resolved_max + 3})
    return AgentResult(
        answer=final["answer"],
        scored_chunks=final["scored_chunks"][:resolved_top_k],
        tried_queries=final["tried_queries"],
        attempts=final["attempts"],
        trace=final["trace"],
    )
