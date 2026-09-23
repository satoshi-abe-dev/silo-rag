"""eval（DAGノードE）の純粋ロジックのテスト。LLM・検索は使わない。"""

from __future__ import annotations

from silo_rag.eval import QAResult, _dedup_report_ids, _reciprocal_rank, _summarize_subset, summarize
from silo_rag.ingest import Chunk
from silo_rag.retrieval import ScoredChunk


def _scored(report_id: str, score: float = 1.0) -> ScoredChunk:
    chunk = Chunk(chunk_id=f"{report_id}::x", text="t", metadata={"report_id": report_id})
    return ScoredChunk(chunk=chunk, score=score)


def test_dedup_report_ids_preserves_order_and_dedups():
    chunks = [_scored("RPT-002"), _scored("RPT-001"), _scored("RPT-002")]
    assert _dedup_report_ids(chunks) == ["RPT-002", "RPT-001"]


def test_reciprocal_rank_found_at_position():
    assert _reciprocal_rank(["RPT-003", "RPT-001", "RPT-002"], {"RPT-001"}) == 0.5


def test_reciprocal_rank_not_found():
    assert _reciprocal_rank(["RPT-003", "RPT-004"], {"RPT-001"}) == 0.0


def _result(
    qa_id: str, cross_dept: bool, hit: bool, recall: float, rr: float, cited: bool, judge
) -> QAResult:
    return QAResult(
        qa_id=qa_id,
        question="q",
        cross_dept=cross_dept,
        gold_report_ids=["RPT-001"],
        retrieved_report_ids=["RPT-001"] if hit else ["RPT-999"],
        hit=hit,
        recall=recall,
        reciprocal_rank=rr,
        answer_text="a",
        cited_gold=cited,
        judge_score=judge,
    )


def test_summarize_subset_empty_returns_none_metrics():
    summary = _summarize_subset([])
    assert summary["n"] == 0
    assert summary["hit_rate"] is None
    assert summary["recall_at_k"] is None
    assert summary["judge_score_count"] == 0


def test_summarize_subset_distinguishes_hit_rate_from_recall_at_k():
    """退行テスト: 以前、goldが複数件ある設問で一部しか拾えていなくても
    recall_at_kが1.0（hit_rateと同じ値）になってしまうバグがあった。"""
    results = [
        _result("QA-001", cross_dept=False, hit=True, recall=1.0, rr=1.0, cited=True, judge=5),
        _result("QA-002", cross_dept=False, hit=True, recall=0.5, rr=1.0, cited=True, judge=3),
    ]
    summary = _summarize_subset(results)
    assert summary["hit_rate"] == 1.0  # 両方とも1件以上は当たっている
    assert summary["recall_at_k"] == 0.75  # (1.0 + 0.5) / 2 、hit_rateとは異なる値になる


def test_summarize_subset_ignores_none_judge_scores_in_average():
    results = [
        _result("QA-001", cross_dept=False, hit=True, recall=1.0, rr=1.0, cited=True, judge=4),
        _result("QA-002", cross_dept=False, hit=True, recall=1.0, rr=1.0, cited=True, judge=None),
    ]
    summary = _summarize_subset(results)
    assert summary["avg_judge_score"] == 4.0
    assert summary["judge_score_count"] == 1


def test_summarize_splits_cross_dept_and_same_dept():
    results = [
        _result("QA-001", cross_dept=True, hit=True, recall=1.0, rr=1.0, cited=True, judge=4),
        _result("QA-002", cross_dept=False, hit=False, recall=0.0, rr=0.0, cited=False, judge=2),
    ]
    summary = summarize(results)
    assert summary["overall"]["n"] == 2
    assert summary["cross_dept"]["n"] == 1
    assert summary["cross_dept"]["hit_rate"] == 1.0
    assert summary["same_dept"]["n"] == 1
    assert summary["same_dept"]["hit_rate"] == 0.0
