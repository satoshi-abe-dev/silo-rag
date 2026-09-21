"""評価スクリプト（DAGノードE）。

`data/eval/qa_pairs.json`（datagenノードが生成したgold-standard QAペア）に対して、
retrieval→generationのRAGパイプライン全体を実行し、検索精度（Recall@k, MRR）と
回答品質（簡易LLM-as-judge, 正解引用率）を測定する。

部署をまたいだ設問（cross_dept=true）と自部署内の設問を分けて集計する。本プロジェクトの
核となる価値提案（部署間の情報共有が不十分でも横断検索できること）が実際に機能しているか
を、この内訳で確認できるようにするため。
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import EVAL_DIR, load_config
from .generation import answer_question
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


def _judge_answer(client: LLMClient, question: str, gold_evidence: list[str], answer_text: str) -> int | None:
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
        # max_tokensは指定しない（config.server.max_tokensのデフォルトに従う）。推論モデルは
        # 可視の回答を書く前に思考トークンを消費するため、ここで小さい値を決め打ちすると
        # 思考だけで使い切り、LLMClient側の「本文が空」検出でエラーになり得る
        # （すべての評価がNoneになりかねない）。
        raw = client.chat(_JUDGE_SYSTEM, user, temperature=0.0)
    except Exception:
        return None
    # 応答を1〜5の整数「1文字だけ」として厳密に検証する。単純に`[1-5]`を本文中から
    # 検索すると、"10"の"1"や"RPT-014"内の数字を誤って拾ってしまう
    # （境界を付けた正規表現でも「本当に1桁だけか」までは保証できないため、
    # ここでは「前後の空白を除いたら1〜5の1文字ちょうど」という厳密な一致にする）。
    stripped = raw.strip()
    return int(stripped) if stripped in {"1", "2", "3", "4", "5"} else None


def run_eval(client: LLMClient, qa_pairs: list[dict], top_k: int | None = None) -> list[QAResult]:
    results: list[QAResult] = []
    for qa in qa_pairs:
        gold_refs = qa["gold_references"]
        gold_ids = {r["report_id"] for r in gold_refs}
        gold_evidence = [r["evidence"] for r in gold_refs]

        scored = search(client, qa["question"], top_k=top_k)
        retrieved_report_ids = _dedup_report_ids(scored)
        matched_gold = gold_ids & set(retrieved_report_ids)
        hit = bool(matched_gold)
        recall = len(matched_gold) / len(gold_ids) if gold_ids else 0.0
        rr = _reciprocal_rank(retrieved_report_ids, gold_ids)

        chunks = [sc.chunk for sc in scored]
        answer = answer_question(client, qa["question"], chunks)
        # answer.citationsは検索されたchunks全件から機械的に作られるため、実際に生成された
        # 回答本文がgoldレポートIDを引用しているかは別途、本文への部分一致で判定する
        # （generation.pyのプロンプトはreport_idをそのまま本文に埋め込ませる設計）。
        cited_gold = any(rid in answer.text for rid in gold_ids)
        judge_score = _judge_answer(client, qa["question"], gold_evidence, answer.text)

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
        )
        results.append(result)
        print(f"{qa['qa_id']}: hit={hit} rr={rr:.2f} cited_gold={cited_gold} judge={judge_score}")
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
    }


def summarize(results: list[QAResult]) -> dict:
    return {
        "overall": _summarize_subset(results),
        "cross_dept": _summarize_subset([r for r in results if r.cross_dept]),
        "same_dept": _summarize_subset([r for r in results if not r.cross_dept]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="RAGパイプラインの評価（検索精度・回答品質）")
    parser.add_argument("--qa-file", type=Path, default=EVAL_DIR / "qa_pairs.json")
    parser.add_argument("--top-k", type=int, default=None, help="Noneならconfig.retrieval.top_k_finalを使う")
    parser.add_argument("--out", type=Path, default=EVAL_DIR / "eval_results.json")
    args = parser.parse_args()

    qa_pairs = _load_qa_pairs(args.qa_file)
    config = load_config()
    with LLMClient(config.server) as client:
        if not client.ping():
            raise SystemExit(
                f"LM Studio ({config.server.base_url}) に接続できません。起動してモデルをロードしてください。"
            )
        results = run_eval(client, qa_pairs, top_k=args.top_k)

    summary = summarize(results)
    print("\n=== summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary, "results": [asdict(r) for r in results]}
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n結果を書き出しました: {args.out}")


if __name__ == "__main__":
    main()
