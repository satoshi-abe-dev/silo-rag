"""Node E: evaluation.

Runs the full RAG pipeline over the gold QA pairs in `data/eval/qa_pairs.json` (from datagen) and
measures retrieval (hit rate, Recall@k, MRR) and answer quality (LLM-as-judge, gold citation rate).

Cross-department and same-department questions are reported separately, since cross-department
search is the project's core value proposition.

`--pipeline agent` answers with the LangGraph agent (node G); `--pipeline langchain` uses LangChain's
stock `create_agent` (node H). For langchain, retrieval metrics count every chunk the tool returned,
so more searches score higher; citation rate and judge compare fairly. Time, LLM calls and searches
per question are recorded too, since the agents pay for their accuracy in cost.
`--compare` merges several result files into one Markdown table.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import EVAL_DIR, load_config
from .generation import Answer, answer_question
from .llm_client import LLMClient
from .retrieval import ScoredChunk, search


@dataclass
class QAResult:
    qa_id: str
    question: str
    cross_dept: bool
    gold_report_ids: list[str]
    retrieved_report_ids: list[str]  # by relevance, deduplicated per report_id
    hit: bool  # at least one gold report was retrieved
    recall: float  # fraction of gold reports retrieved (0.0-1.0)
    reciprocal_rank: float
    answer_text: str
    cited_gold: bool
    judge_score: int | None
    # Cost metrics. llm_calls counts chat calls to reach the answer (not embeddings or judging).
    attempts: int = 1  # number of searches (always 1 for baseline)
    llm_calls: int = 0
    elapsed_sec: float = 0.0


def _load_qa_pairs(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"{path} が見つかりません。先に datagen を実行してください。")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not data:
        raise SystemExit(f"{path} にQAペアがありません。先に datagen を実行してください。")
    return data


def _dedup_report_ids(chunks: list[ScoredChunk]) -> list[str]:
    """Deduplicate by report_id, keeping relevance order."""
    seen: list[str] = []
    for sc in chunks:
        rid = sc.chunk.metadata.get("report_id")
        if rid and rid not in seen:
            seen.append(rid)
    return seen


def _reciprocal_rank(retrieved_ids: list[str], gold_ids: set[str]) -> float:
    for i, rid in enumerate(retrieved_ids, start=1):
        if rid in gold_ids:
            return 1.0 / i
    return 0.0


_JUDGE_SYSTEM = (
    "あなたは社内ナレッジ検索RAGシステムの回答品質を評価する審査員です。"
    "質問・正解の根拠(gold evidence)・生成された回答を読み、回答が質問に的確に答え、"
    "根拠に沿った具体的な内容になっているかを1(不十分)〜5(優れている)の5段階で評価してください。"
    "出力は1〜5の整数1文字のみとし、説明文は一切含めないでください。"
)


PIPELINES = ("baseline", "agent", "langchain")


class _CountingClient:
    """Wrap an LLMClient and count chat calls; everything else is delegated."""

    def __init__(self, inner: LLMClient):
        self._inner = inner
        self.chat_calls = 0

    def chat(self, *args, **kwargs) -> str:
        self.chat_calls += 1
        return self._inner.chat(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _judge_answer(
    client: LLMClient,
    question: str,
    gold_evidence: list[str],
    answer_text: str,
    *,
    model: str | None = None,
) -> int | None:
    """Simple LLM-as-judge returning 1-5, or None if the reply is unparsable.

    None is excluded from the average. LLM errors also return None so a server failure
    doesn't abort the whole evaluation.
    """
    evidence_text = "\n".join(f"- {e}" for e in gold_evidence) or "（なし）"
    user = (
        f"質問: {question}\n\n"
        f"正解の根拠（参考情報）:\n{evidence_text}\n\n"
        f"生成された回答:\n{answer_text}\n\n"
        "評価(1〜5の整数のみ):"
    )
    try:
        # Don't set a small max_tokens: reasoning models can spend it all on thinking, hitting
        # LLMClient's empty-reply error and turning every score into None.
        raw = client.chat(_JUDGE_SYSTEM, user, model=model, temperature=0.0)
    except Exception:
        return None
    # Require exactly one digit 1-5: searching for `[1-5]` would pick up the "1" in "10"
    # or digits inside "RPT-014".
    stripped = raw.strip()
    return int(stripped) if stripped in {"1", "2", "3", "4", "5"} else None


def _answer_baseline(
    client, question: str, top_k: int | None, rewrite_query: bool | None
) -> tuple[list[ScoredChunk], Answer, int]:
    scored = search(client, question, top_k=top_k, rewrite_query=rewrite_query)
    answer = answer_question(client, question, [sc.chunk for sc in scored])
    return scored, answer, 1


def run_eval(
    client: LLMClient,
    qa_pairs: list[dict],
    top_k: int | None = None,
    *,
    pipeline: str = "baseline",
    grade_mode: str | None = None,
    first_query: str | None = None,
    max_attempts: int | None = None,
    judge_model: str | None = None,
    rewrite_query: bool | None = None,
) -> list[QAResult]:
    """Answer and score every QA pair with the given pipeline.

    pipeline: "baseline" (search + answer_question), "agent" (agent.run_agent) or
        "langchain" (run_langchain_agent, tool-calling stock agent).
    grade_mode / first_query: agent only; None uses config.agent.
    max_attempts: search cap for agent/langchain; None uses config.agent.max_attempts (agent)
        or DEFAULT_MAX_SEARCHES (langchain).
    judge_model: None uses config.ai.llm_model. Fix it when comparing answer models, or judge
        scores aren't comparable.
    rewrite_query: baseline only (agents write their own queries); None uses
        config.retrieval.rewrite_query.
    """
    if pipeline not in PIPELINES:
        raise ValueError(f"pipelineは{PIPELINES}のいずれかを指定してください: {pipeline!r}")
    # Lazy imports: langgraph/langchain are optional extras (.[agent] / .[langchain]).
    if pipeline == "agent":
        from .agent import run_agent
    elif pipeline == "langchain":
        from .langchain_adapter import run_langchain_agent

    results: list[QAResult] = []
    for qa in qa_pairs:
        gold_refs = qa["gold_references"]
        gold_ids = {r["report_id"] for r in gold_refs}
        gold_evidence = [r["evidence"] for r in gold_refs]

        counting = _CountingClient(client)
        extra_llm_calls = 0  # calls that bypass `counting` (the langchain agent's own model)
        started = time.perf_counter()
        if pipeline == "langchain":
            lc_result = run_langchain_agent(counting, qa["question"], top_k=top_k, max_searches=max_attempts)
            scored, answer, attempts = lc_result.scored_chunks, lc_result.answer, lc_result.searches
            extra_llm_calls = lc_result.llm_calls
        elif pipeline == "agent":
            # _CountingClient duck-types LLMClient but isn't one, hence the ignore.
            agent_result = run_agent(
                counting,  # type: ignore[arg-type]
                qa["question"],
                top_k=top_k,
                grade_mode=grade_mode,
                first_query=first_query,
                max_attempts=max_attempts,
            )
            scored, answer, attempts = agent_result.scored_chunks, agent_result.answer, agent_result.attempts
        else:
            scored, answer, attempts = _answer_baseline(counting, qa["question"], top_k, rewrite_query)
        elapsed = time.perf_counter() - started

        retrieved_report_ids = _dedup_report_ids(scored)
        matched_gold = gold_ids & set(retrieved_report_ids)
        hit = bool(matched_gold)
        recall = len(matched_gold) / len(gold_ids) if gold_ids else 0.0
        rr = _reciprocal_rank(retrieved_report_ids, gold_ids)

        # answer.citations lists every retrieved chunk, so check the answer text itself for gold
        # report_ids (the generation prompt has the model cite report_ids inline).
        cited_gold = any(rid in answer.text for rid in gold_ids)
        judge_score = _judge_answer(client, qa["question"], gold_evidence, answer.text, model=judge_model)

        result = QAResult(
            qa_id=qa["qa_id"],
            question=qa["question"],
            cross_dept=bool(qa.get("cross_dept", False)),
            gold_report_ids=sorted(gold_ids),
            retrieved_report_ids=retrieved_report_ids,
            hit=hit,
            recall=recall,
            reciprocal_rank=rr,
            answer_text=answer.text,
            cited_gold=cited_gold,
            judge_score=judge_score,
            attempts=attempts,
            llm_calls=counting.chat_calls + extra_llm_calls,
            elapsed_sec=elapsed,
        )
        results.append(result)
        print(
            f"{qa['qa_id']}: hit={hit} rr={rr:.2f} cited_gold={cited_gold} judge={judge_score} "
            f"attempts={attempts} llm_calls={counting.chat_calls + extra_llm_calls} {elapsed:.1f}s"
        )
    return results


def _summarize_subset(results: list[QAResult]) -> dict:
    n = len(results)
    if n == 0:
        return {
            "n": 0,
            "hit_rate": None,
            "recall_at_k": None,
            "mrr": None,
            "citation_rate": None,
            "avg_judge_score": None,
            "judge_score_count": 0,
            "avg_attempts": None,
            "avg_llm_calls": None,
            "avg_elapsed_sec": None,
        }
    judge_scores = [r.judge_score for r in results if r.judge_score is not None]
    return {
        "n": n,
        # hit_rate: share of questions with at least one gold report retrieved.
        # recall_at_k: mean fraction of gold reports retrieved; stricter when a question has
        # several gold reports and only some are found.
        "hit_rate": sum(r.hit for r in results) / n,
        "recall_at_k": sum(r.recall for r in results) / n,
        "mrr": sum(r.reciprocal_rank for r in results) / n,
        "citation_rate": sum(r.cited_gold for r in results) / n,
        "avg_judge_score": statistics.mean(judge_scores) if judge_scores else None,
        "judge_score_count": len(judge_scores),
        "avg_attempts": sum(r.attempts for r in results) / n,
        "avg_llm_calls": sum(r.llm_calls for r in results) / n,
        "avg_elapsed_sec": sum(r.elapsed_sec for r in results) / n,
    }


def summarize(results: list[QAResult]) -> dict:
    return {
        "overall": _summarize_subset(results),
        "cross_dept": _summarize_subset([r for r in results if r.cross_dept]),
        "same_dept": _summarize_subset([r for r in results if not r.cross_dept]),
    }


_COMPARE_COLUMNS: list[tuple[str, str, str]] = [
    # (header, summary subset, metric key)
    ("hit_rate", "overall", "hit_rate"),
    ("recall@k", "overall", "recall_at_k"),
    ("MRR", "overall", "mrr"),
    ("引用率", "overall", "citation_rate"),
    ("judge", "overall", "avg_judge_score"),
    ("横断hit", "cross_dept", "hit_rate"),
    ("自部署hit", "same_dept", "hit_rate"),
    ("検索回数", "overall", "avg_attempts"),
    ("LLM回数", "overall", "avg_llm_calls"),
    ("秒/問", "overall", "avg_elapsed_sec"),
]


def _positive_int(text: str) -> int:
    """argparse type: accept only integers >= 1 rather than silently clamping 0 or negatives."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"整数を指定してください: {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"1以上の整数を指定してください: {value}")
    return value


def _resolve_max_attempts(pipeline: str, requested: int | None, configured: int) -> int | None:
    """Resolve the search cap; fall back to config (agent) or the default (langchain) only on None.

    Checks `is None` rather than `requested or configured`, which would drop an explicit 0.
    Baseline never repeats searches, so it gets None.
    """
    if pipeline == "baseline":
        return None
    if requested is not None:
        return requested
    if pipeline == "langchain":
        from .langchain_adapter import DEFAULT_MAX_SEARCHES

        return DEFAULT_MAX_SEARCHES
    return configured


def _default_result_name(
    pipeline: str,
    grade_mode: str | None,
    first_query: str | None,
    max_attempts: int | None,
    rewrite_query: bool = False,
) -> str:
    """Default --out name; encodes the run settings so different runs don't overwrite each other."""
    if pipeline == "baseline":
        # Keep rewrite_query runs from overwriting the plain eval_results.json.
        return "eval_results_baseline_rewrite.json" if rewrite_query else "eval_results.json"
    if pipeline == "langchain":
        return f"eval_results_langchain_max{max_attempts}.json"
    rewrite = "_rewrite" if first_query == "rewrite" else ""
    return f"eval_results_agent_{grade_mode}{rewrite}_max{max_attempts}.json"


def _run_label(payload: dict, fallback: str) -> str:
    run = payload.get("run")
    if not run:
        return fallback  # older result files without run metadata
    detail: list[str] = []
    if run.get("grade_mode"):
        detail.append(run["grade_mode"])
    if run.get("first_query") == "rewrite":
        detail.append("rewrite-first")  # raw runs get no marker
    if run.get("rewrite_query"):
        detail.append("rewrite-query")  # baseline run where search() rewrote the query
    if run.get("max_attempts") is not None:
        # Always shown so runs with different caps stay distinguishable.
        detail.append(f"max={run['max_attempts']}")
    label = run["pipeline"] + (f"({','.join(detail)})" if detail else "")
    return f"{label} / {run['llm_model']}"


def format_comparison(payloads: list[tuple[str, dict]]) -> str:
    """Render (file name, payload) pairs as a Markdown comparison table."""
    header = "| 実行 | " + " | ".join(col for col, _, _ in _COMPARE_COLUMNS) + " |"
    lines = [header, "|" + " --- |" * (len(_COMPARE_COLUMNS) + 1)]
    for name, payload in payloads:
        summary = payload.get("summary", {})
        cells = []
        for _, subset, key in _COMPARE_COLUMNS:
            value = summary.get(subset, {}).get(key)
            cells.append("-" if value is None else f"{value:.2f}")
        lines.append(f"| {_run_label(payload, name)} | " + " | ".join(cells) + " |")
    judge_models = {p["run"]["judge_model"] for _, p in payloads if p.get("run")}
    if len(judge_models) > 1:
        lines.append("")
        lines.append(
            f"注意: 採点モデルが実行ごとに異なるため、judgeの値は比べられない（{sorted(judge_models)}）。"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="RAGパイプラインの評価（検索精度・回答品質）")
    parser.add_argument("--qa-file", type=Path, default=EVAL_DIR / "qa_pairs.json")
    parser.add_argument("--top-k", type=int, default=None, help="Noneならconfig.retrieval.top_k_finalを使う")
    parser.add_argument("--pipeline", choices=list(PIPELINES), default="baseline")
    parser.add_argument(
        "--grade-mode",
        choices=["strict", "lenient"],
        default=None,
        help="agentの判定の厳しさ。Noneならconfig.agent.grade_mode",
    )
    parser.add_argument(
        "--first-query",
        choices=["raw", "rewrite"],
        default=None,
        help="agentの1回目の検索クエリ（質問そのまま／質問からLLMが作る）。Noneならconfig.agent.first_query",
    )
    parser.add_argument(
        "--max-attempts",
        type=_positive_int,
        default=None,
        help="agent・langchainの検索の上限回数。Noneなら、agentはconfig.agent.max_attempts、langchainは3回",
    )
    parser.add_argument(
        "--rewrite-query",
        action="store_true",
        help="baselineのとき、検索前に質問から検索クエリを作る（retrieval.plan_query）。"
        "指定しなければconfig.retrieval.rewrite_query。agent・langchainでは指定できない",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="LLM-as-judgeの採点モデル。Noneならconfig.ai.llm_model。回答モデルを変えて比べるときは固定する",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="既定は eval_results.json（baseline）、eval_results_agent_<grade_mode>[_rewrite]_max<N>.json"
        "（agent）、eval_results_langchain_max<N>.json（langchain）",
    )
    parser.add_argument(
        "--compare",
        type=Path,
        nargs="+",
        metavar="RESULT_JSON",
        help="評価は実行せず、指定した結果ファイル群の比較表（Markdown）を表示する",
    )
    args = parser.parse_args()

    if args.compare:
        payloads = [(path.stem, json.loads(path.read_text(encoding="utf-8"))) for path in args.compare]
        print(format_comparison(payloads))
        return

    if args.rewrite_query and args.pipeline != "baseline":
        parser.error("--rewrite-queryはbaselineのときだけ指定できます（agentは--first-queryで選びます）")

    qa_pairs = _load_qa_pairs(args.qa_file)
    config = load_config()
    is_agent = args.pipeline == "agent"
    grade_mode = (args.grade_mode or config.agent.grade_mode) if is_agent else None
    first_query = (args.first_query or config.agent.first_query) if is_agent else None
    max_attempts = _resolve_max_attempts(args.pipeline, args.max_attempts, config.agent.max_attempts)
    judge_model = args.judge_model or config.ai.llm_model
    # Baseline only. Record the effective value so runs enabled via config are labeled too.
    is_baseline = args.pipeline == "baseline"
    rewrite_query = (args.rewrite_query or config.retrieval.rewrite_query) if is_baseline else None
    out = args.out
    if out is None:
        out = EVAL_DIR / _default_result_name(
            args.pipeline, grade_mode, first_query, max_attempts, bool(rewrite_query)
        )

    with LLMClient(config.ai) as client:
        if not client.ping():
            raise SystemExit(
                f"LM Studio ({config.ai.base_url}) に接続できません。起動してモデルをロードしてください。"
            )
        results = run_eval(
            client,
            qa_pairs,
            top_k=args.top_k,
            pipeline=args.pipeline,
            grade_mode=grade_mode,
            first_query=first_query,
            max_attempts=max_attempts,
            judge_model=judge_model,
            rewrite_query=rewrite_query,
        )

    summary = summarize(results)
    print("\n=== summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    out.parent.mkdir(parents=True, exist_ok=True)
    run = {
        "pipeline": args.pipeline,
        "llm_model": config.ai.llm_model,
        "judge_model": judge_model,
        "grade_mode": grade_mode,
        "first_query": first_query,
        "max_attempts": max_attempts,
        "rewrite_query": rewrite_query,
        "top_k": args.top_k if args.top_k is not None else config.retrieval.top_k_final,
    }
    payload = {"run": run, "summary": summary, "results": [asdict(r) for r in results]}
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n結果を書き出しました: {out}")


if __name__ == "__main__":
    main()
