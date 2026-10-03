"""検索を繰り返すRAGエージェント（DAGノードG、LangGraph）。

retrieval.pyの`search()`（ノードC）とgeneration.pyの`answer_question()`（ノードD）を
公開関数として呼ぶだけで、C・Dの中身には手を入れない。C・Dは互いに依存しないまま、
両方を知っているのはこのモジュールだけ（evalがC・Dを両方呼ぶのと同じ形）なので、
モジュール間の依存関係はDAGのまま保たれる。ループはこのノードの内側（実行時の処理の流れ）にだけある。

実行時のグラフ:

    START → retrieve → grade ─(十分、または不十分でも上限到達)→ select → generate → END
               ↑         └─(不十分・上限未満)→ rewrite ─(書き直し失敗)→ select
               └──────────(新しいクエリ)──────────┘

first_query="rewrite" のときは、START直後にplanノードが質問から検索クエリを作ってから検索する
（"raw"なら質問をそのまま検索に使う）。

selectは、複数回検索した場合だけ、集めた候補全体を元の質問でLLMに並べ直させる
（retrieval.rerank()。1回だけならsearch()内で並べ直し済みなので何もしない）。

「根拠が十分か」「次にどんなクエリで探すか」はLLMが判断し、「何回まで繰り返すか」は
config.agent.max_attemptsでコードが決める。

7B程度のローカルモデルではツール呼び出し（function calling）が安定しないため、
ReAct型の自由なエージェントにはしない。LLMには「SUFFICIENT／INSUFFICIENTの1語」
「検索クエリ1文」のように決まった形式だけを答えさせ、コード側で厳密に解釈する
（eval.pyの_judge_answerと同じ考え方）。

失敗時の方針: 判定・書き直しという補助的なLLM判断の失敗（接続失敗・形式の崩れ）では
処理を止めず、手元のチャンクで回答生成に進む（retrieval.pyの_resolve_query/_llm_rerankと
同じ方針）。回答生成そのもの（answer_question）の失敗は、従来どおり呼び出し元に伝える。

LangGraphは、LangSmith用の環境変数（LANGSMITH_TRACING等）を設定しない限り外部へ何も送信しない。
LLMの呼び出しはすべて既存のLLMClient経由（ローカルのLM Studio等）で、外部送信ゼロの方針は変わらない。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from .config import load_config
from .generation import Answer, answer_question
from .llm_client import LLMClient, LLMConnectionError
from .retrieval import ScoredChunk, build_history_lines, parse_query_response, plan_query, rerank, search

# 判定プロンプトに載せるチャンク本文の最大文字数（判定には要点が分かれば足りる）。
_GRADE_CHUNK_TRUNCATE = 300


class AgentState(TypedDict):
    question: str
    history: list[tuple[str, str]] | None
    filters: dict[str, str] | None
    tried_queries: list[str]  # 同じクエリでの再検索を防ぐ
    results_by_attempt: list[list[ScoredChunk]]  # 各回の検索結果（新しい回が先頭）
    scored_chunks: list[ScoredChunk]  # results_by_attemptを_merge_chunksで1列に並べたもの
    attempts: int
    verdict: Literal["sufficient", "insufficient"] | None
    next_query: str | None
    answer: Answer | None
    trace: list[str]  # UI・evalで「エージェントが何をしたか」を見せるためのログ


@dataclass
class AgentResult:
    answer: Answer
    scored_chunks: list[ScoredChunk]  # 回答生成に渡したチャンク（関連度順）
    tried_queries: list[str]
    attempts: int
    trace: list[str]


# --- 検索結果の蓄積 -------------------------------------------------------------


def _merge_chunks(results_by_attempt: list[list[ScoredChunk]]) -> list[ScoredChunk]:
    """各回の検索結果を、順位ごとに交互に並べて1列にする（重複は先に出た方だけ残す）。

    results_by_attemptは新しい回が先頭。各回の1位→各回の2位→…の順に並べ、同じ順位の中では
    新しい回を優先する（再検索するのは「前回までの結果では不十分」と判定されたときだけのため）。

    最新の結果を単純に先頭へ積むと、最新の回がtop_k件を返した時点で前回までの根拠がすべて
    押し出され、複数回の検索で補い合う根拠を蓄積できない（codexレビューで指摘され、修正した）。
    交互に並べれば、どの回の上位の根拠も残る。
    スコアで混ぜないのは、retrieval.search()のスコアがクエリごとに正規化されていて、
    別クエリの結果同士では比較できないため（最終順位もLLMリランキングで決まっている）。
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


# --- 判定（根拠は十分か） ---------------------------------------------------------

_GRADE_PROMPT_HEAD = (
    "あなたは社内ナレッジ検索の検索結果を点検する係です。"
    "質問と検索で見つかった資料の抜粋を読み、質問に答えるための根拠が資料に含まれているかを判定してください。"
    "会話履歴が付いている場合は、質問中の指示語（「それ」「さっきの」等）が何を指すかの解決にだけ使ってください。"
)
_GRADE_PROMPT_TAIL = "出力は SUFFICIENT か INSUFFICIENT のどちらか1語のみとし、説明は一切含めないでください。"

# 判定の厳しさ（config.agent.grade_mode）。実データ（7Bモデル）では、どちらにも弱点があった:
#   strict:  9回中8回が「不十分」。ほぼ毎回上限まで再検索し、時間は約2倍・LLM呼び出しは4〜5倍。
#            代わりに、baselineでは拾えない正解を再検索で拾えることがある。
#   lenient: すぐ「十分」と判定するため、多くの質問でbaselineと同じ結果になる。
# 判定するLLMは「まだ見ていない、もっと良い資料があるか」を知りようがないため、どちらが良いかは
# プロンプトの言い回しだけでは決まらない。eval.pyで両方を計測して比較する。
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
    # フォローアップ質問（「その費用は？」等）は、履歴が無いと何についての質問か判定できない
    # （codexレビューで指摘され、修正した。検索・書き直し・回答生成と同じく履歴を渡す）。
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
    """応答が「SUFFICIENT／INSUFFICIENTの1語ちょうど」の場合だけ解釈する。

    部分一致で探すと、"INSUFFICIENT"の中の"SUFFICIENT"を誤って拾ってしまうため、
    前後の空白・末尾の句点を除いたうえでの完全一致にする。
    """
    word = raw.strip().rstrip("。.").strip().upper()
    if word == "SUFFICIENT":
        return "sufficient"
    if word == "INSUFFICIENT":
        return "insufficient"
    return None


# --- クエリの書き直し ------------------------------------------------------------

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
    """最終選定の並べ直しに使うクエリ。

    retrieval.rerank()は質問文を1つしか受け取れないため、フォローアップ質問（「その費用は？」等）
    では直前の質問を添えて、何についての質問かが分かるようにする。
    """
    if not history:
        return question
    return f"{history[-1][0]}（に続けて）{question}"


# --- グラフ ---------------------------------------------------------------------


def build_graph(
    client: LLMClient,
    *,
    max_attempts: int,
    top_k: int,
    grade_mode: str = "lenient",
    first_query: str = "raw",
):
    """エージェントのグラフを組み立てて、コンパイル済みのものを返す。

    clientはクロージャで各ノードに渡す（LLMClientはシリアライズできないため、Stateには入れない）。
    grade_mode: 判定の厳しさ。"strict" か "lenient"（_GRADE_SYSTEM_PROMPTSのコメント参照）。
    first_query: 1回目の検索に使うクエリ。"raw"は質問そのまま、"rewrite"は質問からLLMが作ったクエリ。
    """
    if grade_mode not in _GRADE_SYSTEM_PROMPTS:
        raise ValueError(f"grade_modeは{GRADE_MODES}のいずれかを指定してください: {grade_mode!r}")
    if first_query not in FIRST_QUERY_MODES:
        raise ValueError(f"first_queryは{FIRST_QUERY_MODES}のいずれかを指定してください: {first_query!r}")
    grade_system_prompt = _GRADE_SYSTEM_PROMPTS[grade_mode]

    def plan(state: AgentState) -> dict:
        # 1回目の検索クエリを、質問から作る。質問の文には、名乗りや依頼の言い回しなど検索に関係の無い
        # 言葉が多く混ざり、BM25（文字2-gram）が散りやすい。失敗したら、質問そのままで検索する
        # （retrieveがnext_query=Noneを見て従来の初回の動きに戻る）。
        query, failure = plan_query(client, state["question"], state["history"])
        if query is None:
            return {"trace": [*state["trace"], f"クエリ作成: {failure}。質問のまま検索"]}
        return {"next_query": query, "trace": [*state["trace"], f"クエリ作成: 「{query}」"]}

    def retrieve(state: AgentState) -> dict:
        attempts = state["attempts"] + 1
        if state["next_query"] is None and state["attempts"] == 0:
            # 質問そのままで検索する。会話履歴があれば、search()側が履歴込みの
            # 独立したクエリに書き換えてから検索する（従来のbaselineと同じ挙動）。
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
            # planノード・rewriteノードが、履歴も踏まえてクエリを作っているので、historyは渡さない。
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
            # もう再検索できないので、判定の結果は何も変えない（どちらでも回答生成に進む）。LLMを呼ばない。
            return {"verdict": None, "trace": [*state["trace"], "判定: 省略（検索の上限に達したため）"]}
        chunks = state["scored_chunks"][:top_k]
        if not chunks:
            # 判定するまでもなく根拠が無い。LLMを呼ばずに「不十分」とする。
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
            # 形式が崩れた応答で検索を無駄に繰り返させないよう、「十分」とみなして先へ進む。
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
            # 同じクエリで検索し直しても結果は変わらない。
            return {
                "next_query": None,
                "trace": [*state["trace"], f"書き直し: 「{query}」は試行済み。打ち切り"],
            }
        return {"next_query": query, "trace": [*state["trace"], f"書き直し: 「{query}」"]}

    def route_after_rewrite(state: AgentState) -> Literal["retrieve", "select"]:
        return "retrieve" if state["next_query"] else "select"

    def select(state: AgentState) -> dict:
        # 1回しか検索していなければ、search()内ですでに質問で並べ直し済み。
        if state["attempts"] <= 1:
            return {}
        # 複数回検索した場合、_merge_chunksの交互の並びのままtop_kで切ると、回数が増えるほど
        # 各回の上位しか残らず、元の質問での検索（たいてい最も信頼できる）の2位以下が押し出される
        # （実データで、baselineでは拾えていた正解がこれで落ちた）。集めた候補全体を元の質問で
        # 並べ直してから、generateでtop_k件に絞る。
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
    """エージェントを1問ぶん実行する。引数の意味はretrieval.search()・generation.answer_question()と同じ。

    top_k: 判定・回答生成に使うチャンク数。Noneならconfig.retrieval.top_k_finalを使う。
    max_attempts: 検索の上限回数（初回を含む）。Noneならconfig.agent.max_attemptsを使う。
    grade_mode: 判定の厳しさ（"strict"／"lenient"）。Noneならconfig.agent.grade_modeを使う。
    first_query: 1回目の検索クエリ（"raw"＝質問そのまま／"rewrite"＝質問からLLMが作る）。
        Noneならconfig.agent.first_queryを使う。
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
    # 最大ステップ数は plan（rewrite-firstのとき）が1回、retrieve・grade が各max_attempts回、rewriteが
    # max_attempts-1回、select・generateが各1回で 3*max_attempts+2。max_attemptsでループは必ず止まるが、
    # 万一の無限ループに備えてLangGraph側のステップ上限も、必要数ぎりぎりで掛けておく。
    final = app.invoke(initial, config={"recursion_limit": 3 * resolved_max + 3})
    return AgentResult(
        answer=final["answer"],
        scored_chunks=final["scored_chunks"][:resolved_top_k],
        tried_queries=final["tried_queries"],
        attempts=final["attempts"],
        trace=final["trace"],
    )
