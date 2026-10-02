"""評価スクリプト（DAGノードE）。

`data/eval/qa_pairs.json`（datagenノードが生成したgold-standard QAペア）に対して、
retrieval→generationのRAGパイプライン全体を実行し、検索精度（Recall@k, MRR）と
回答品質（簡易LLM-as-judge, 正解引用率）を測定する。

部署をまたいだ設問（cross_dept=true）と自部署内の設問を分けて集計する。本プロジェクトの
核となる価値提案（部署間の情報共有が不十分でも横断検索できること）が実際に機能しているか
を、この内訳で確認できるようにするため。

`--pipeline agent` を指定すると、retrieval→generationを直接呼ぶ代わりにLangGraphエージェント
（agent.py、DAGノードG）で回答し、同じ指標で比較できる。精度に加えて、1問あたりの時間・
LLM呼び出し回数・検索回数も記録する（エージェントはその分のコストを払うため）。
`--compare` で、複数の結果ファイルを1つの比較表（Markdown）にまとめて表示できる。
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
    retrieved_report_ids: list[str]  # 関連度順、report_id単位で重複排除
    hit: bool  # gold_report_idsのうち1件以上を検索できたか
    recall: float  # gold_report_idsのうち検索できた割合（0.0〜1.0）
    reciprocal_rank: float
    answer_text: str
    cited_gold: bool
    judge_score: int | None
    # コスト指標。llm_callsは回答までのchat呼び出し回数（埋め込み・judgeの採点は含まない）。
    attempts: int = 1  # 検索回数（baselineは常に1）
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
    """関連度順（chunksの並び順）を保ったまま、report_id単位で重複排除する。"""
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


class _CountingClient:
    """LLMClientを包み、chatの呼び出し回数を数える（それ以外の属性は本物にそのまま任せる）。"""

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
    """簡易LLM-as-judge。応答が解釈できない場合はNone（judge_score_countから除外され、
    平均スコアの計算対象にもならない。サーバー障害等で評価プロセス全体を止めないため、
    ここでのLLM呼び出し失敗は握りつぶしてNoneを返す）。"""
    evidence_text = "\n".join(f"- {e}" for e in gold_evidence) or "（なし）"
    user = (
        f"質問: {question}\n\n"
        f"正解の根拠（参考情報）:\n{evidence_text}\n\n"
        f"生成された回答:\n{answer_text}\n\n"
        "評価(1〜5の整数のみ):"
    )
    try:
        # max_tokensは指定しない（config.ai.max_tokensのデフォルトに従う）。推論モデルは
        # 可視の回答を書く前に思考トークンを消費するため、ここで小さい値を決め打ちすると
        # 思考だけで使い切り、LLMClient側の「本文が空」検出でエラーになり得る
        # （すべての評価がNoneになりかねない）。
        raw = client.chat(_JUDGE_SYSTEM, user, model=model, temperature=0.0)
    except Exception:
        return None
    # 応答を1〜5の整数「1文字だけ」として厳密に検証する。単純に`[1-5]`を本文中から
    # 検索すると、"10"の"1"や"RPT-014"内の数字を誤って拾ってしまう
    # （境界を付けた正規表現でも「本当に1桁だけか」までは保証できないため、
    # ここでは「前後の空白を除いたら1〜5の1文字ちょうど」という厳密な一致にする）。
    stripped = raw.strip()
    return int(stripped) if stripped in {"1", "2", "3", "4", "5"} else None


def _answer_baseline(client, question: str, top_k: int | None) -> tuple[list[ScoredChunk], Answer, int]:
    scored = search(client, question, top_k=top_k)
    answer = answer_question(client, question, [sc.chunk for sc in scored])
    return scored, answer, 1


def run_eval(
    client: LLMClient,
    qa_pairs: list[dict],
    top_k: int | None = None,
    *,
    pipeline: str = "baseline",
    grade_mode: str | None = None,
    judge_model: str | None = None,
) -> list[QAResult]:
    """pipeline: "baseline"（search→answer_questionを直接呼ぶ）か "agent"（agent.run_agent）。
    grade_mode: agentの判定の厳しさ。Noneならconfig.agent.grade_mode。
    judge_model: LLM-as-judgeの採点に使うモデル。Noneならconfig.ai.llm_model（回答と同じモデル）。
        回答モデルを変えて比較するときは、採点モデルを固定しないとjudgeスコアが比べられない。
    """
    if pipeline not in ("baseline", "agent"):
        raise ValueError(f"pipelineは'baseline'か'agent'を指定してください: {pipeline!r}")
    if pipeline == "agent":
        # langgraphは任意依存（.[agent]）。baselineだけ使う環境ではimportしない。
        from .agent import run_agent

    results: list[QAResult] = []
    for qa in qa_pairs:
        gold_refs = qa["gold_references"]
        gold_ids = {r["report_id"] for r in gold_refs}
        gold_evidence = [r["evidence"] for r in gold_refs]

        counting = _CountingClient(client)
        started = time.perf_counter()
        if pipeline == "agent":
            agent_result = run_agent(counting, qa["question"], top_k=top_k, grade_mode=grade_mode)  # type: ignore[arg-type]
            scored, answer, attempts = agent_result.scored_chunks, agent_result.answer, agent_result.attempts
        else:
            scored, answer, attempts = _answer_baseline(counting, qa["question"], top_k)
        elapsed = time.perf_counter() - started

        retrieved_report_ids = _dedup_report_ids(scored)
        matched_gold = gold_ids & set(retrieved_report_ids)
        hit = bool(matched_gold)
        recall = len(matched_gold) / len(gold_ids) if gold_ids else 0.0
        rr = _reciprocal_rank(retrieved_report_ids, gold_ids)

        # answer.citationsは検索されたchunks全件から機械的に作られるため、実際に生成された
        # 回答本文がgoldレポートIDを引用しているかは別途、本文への部分一致で判定する
        # （generation.pyのプロンプトはreport_idをそのまま本文に埋め込ませる設計）。
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
            llm_calls=counting.chat_calls,
            elapsed_sec=elapsed,
        )
        results.append(result)
        print(
            f"{qa['qa_id']}: hit={hit} rr={rr:.2f} cited_gold={cited_gold} judge={judge_score} "
            f"attempts={attempts} llm_calls={counting.chat_calls} {elapsed:.1f}s"
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
        # hit_rate: gold_report_idsのうち1件以上を検索できた設問の割合。
        # recall_at_k: 各設問のgold_report_idsのうち検索できた割合（0.0〜1.0）の平均。
        # goldが複数件ある設問で一部しか拾えていない場合、hit_rateは1.0のままでも
        # recall_at_kは1.0未満になり、より厳密な指標になる。
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
    # (見出し, summaryのサブセット, 指標キー)
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


def _run_label(payload: dict, fallback: str) -> str:
    run = payload.get("run")
    if not run:
        return fallback  # runメタデータが無い（この機能より前の）結果ファイル
    label = run["pipeline"]
    if run.get("grade_mode"):
        label += f"({run['grade_mode']})"
    return f"{label} / {run['llm_model']}"


def format_comparison(payloads: list[tuple[str, dict]]) -> str:
    """複数の評価結果（(ファイル名, 中身)のリスト）を、Markdownの比較表にする。"""
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
    parser.add_argument("--pipeline", choices=["baseline", "agent"], default="baseline")
    parser.add_argument(
        "--grade-mode",
        choices=["strict", "lenient"],
        default=None,
        help="agentの判定の厳しさ。Noneならconfig.agent.grade_mode",
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
        help="既定はbaselineならeval_results.json、agentならeval_results_agent_<grade_mode>.json",
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

    qa_pairs = _load_qa_pairs(args.qa_file)
    config = load_config()
    grade_mode = (args.grade_mode or config.agent.grade_mode) if args.pipeline == "agent" else None
    judge_model = args.judge_model or config.ai.llm_model
    out = args.out
    if out is None:
        name = "eval_results.json" if args.pipeline == "baseline" else f"eval_results_agent_{grade_mode}.json"
        out = EVAL_DIR / name

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
            judge_model=judge_model,
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
        "max_attempts": config.agent.max_attempts if args.pipeline == "agent" else None,
        "top_k": args.top_k if args.top_k is not None else config.retrieval.top_k_final,
    }
    payload = {"run": run, "summary": summary, "results": [asdict(r) for r in results]}
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n結果を書き出しました: {out}")


if __name__ == "__main__":
    main()
