"""eval（DAGノードE）のテスト。LLM・検索は使わない（必要な箇所はフェイクに差し替える）。"""

from __future__ import annotations

import pytest

import silo_rag.eval as eval_module
from silo_rag.eval import (
    QAResult,
    _dedup_report_ids,
    _reciprocal_rank,
    _summarize_subset,
    format_comparison,
    run_eval,
    summarize,
)
from silo_rag.generation import Answer
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


# --- パイプライン切り替え・コスト指標・比較表 ------------------------------------------


class _JudgeClient:
    """回答生成はフェイクに差し替えるので、ここに来るchatはjudgeの採点だけ。"""

    def __init__(self):
        self.models: list[str | None] = []

    def chat(self, system, user, *, model=None, **kwargs):
        self.models.append(model)
        return "4"


_QA = [
    {
        "qa_id": "QA-001",
        "question": "質問",
        "cross_dept": True,
        "gold_references": [{"report_id": "RPT-001", "dept": "営業推進部", "evidence": "根拠"}],
    }
]


def test_run_eval_baseline_counts_llm_calls_but_not_judge(monkeypatch):
    def fake_search(client, question, *, top_k=None):
        client.chat("rerank", "x")  # 検索内のLLM呼び出し（リランキング）の代わり
        return [_scored("RPT-001")]

    def fake_answer(client, question, chunks):
        client.chat("answer", "x")
        return Answer(text="RPT-001が参考になります", citations=[])

    monkeypatch.setattr(eval_module, "search", fake_search)
    monkeypatch.setattr(eval_module, "answer_question", fake_answer)
    judge = _JudgeClient()

    (result,) = run_eval(judge, _QA, judge_model="judge-32b")

    assert result.hit and result.cited_gold
    assert result.attempts == 1
    assert result.llm_calls == 2  # 検索1回＋回答1回。judgeの採点は数えない
    assert result.judge_score == 4
    assert judge.models[-1] == "judge-32b"


def test_run_eval_agent_uses_agent_result(monkeypatch):
    import silo_rag.agent as agent_module
    from silo_rag.agent import AgentResult

    captured = {}

    def fake_run_agent(client, question, *, top_k=None, grade_mode=None, first_query=None):
        captured["grade_mode"] = grade_mode
        captured["first_query"] = first_query
        for _ in range(5):
            client.chat("agent", "x")
        return AgentResult(
            answer=Answer(text="RPT-001より", citations=[]),
            scored_chunks=[_scored("RPT-009"), _scored("RPT-001")],
            tried_queries=["質問", "別のクエリ"],
            attempts=2,
            trace=[],
        )

    monkeypatch.setattr(agent_module, "run_agent", fake_run_agent)

    (result,) = run_eval(_JudgeClient(), _QA, pipeline="agent", grade_mode="strict", first_query="rewrite")

    assert captured["grade_mode"] == "strict"
    assert captured["first_query"] == "rewrite"
    assert result.attempts == 2
    assert result.llm_calls == 5
    assert result.retrieved_report_ids == ["RPT-009", "RPT-001"]
    assert result.reciprocal_rank == 0.5


def test_run_eval_rejects_unknown_pipeline():
    with pytest.raises(ValueError):
        run_eval(_JudgeClient(), _QA, pipeline="magic")


def test_summarize_subset_includes_cost_metrics():
    results = [
        _result("QA-001", cross_dept=False, hit=True, recall=1.0, rr=1.0, cited=True, judge=4),
        _result("QA-002", cross_dept=False, hit=True, recall=1.0, rr=1.0, cited=True, judge=4),
    ]
    results[0].attempts, results[0].llm_calls, results[0].elapsed_sec = 1, 3, 10.0
    results[1].attempts, results[1].llm_calls, results[1].elapsed_sec = 3, 9, 20.0
    summary = _summarize_subset(results)
    assert summary["avg_attempts"] == 2.0
    assert summary["avg_llm_calls"] == 6.0
    assert summary["avg_elapsed_sec"] == 15.0


def _payload(pipeline, grade_mode, llm_model, judge_model, hit_rate):
    return {
        "run": {
            "pipeline": pipeline,
            "grade_mode": grade_mode,
            "llm_model": llm_model,
            "judge_model": judge_model,
        },
        "summary": {"overall": {"hit_rate": hit_rate}, "cross_dept": {}, "same_dept": {}},
    }


def test_format_comparison_labels_runs_and_fills_missing_metrics():
    table = format_comparison(
        [
            ("a", _payload("baseline", None, "qwen-7b", "qwen-32b", 0.4)),
            ("b", _payload("agent", "strict", "qwen-7b", "qwen-32b", 0.6)),
            ("old_file", {"summary": {"overall": {"hit_rate": 0.5}}}),
        ]
    )
    lines = table.splitlines()
    assert lines[2].startswith("| baseline / qwen-7b | 0.40 |")
    assert lines[3].startswith("| agent(strict) / qwen-7b | 0.60 |")
    assert lines[4].startswith("| old_file | 0.50 | - |")
    assert "注意" not in table


def test_format_comparison_warns_when_judge_models_differ():
    table = format_comparison(
        [
            ("a", _payload("baseline", None, "qwen-7b", "qwen-7b", 0.4)),
            ("b", _payload("baseline", None, "qwen-32b", "qwen-32b", 0.5)),
        ]
    )
    assert "採点モデルが実行ごとに異なる" in table


def test_run_eval_langchain_adds_agent_llm_calls_and_uses_searches(monkeypatch):
    import silo_rag.langchain_adapter as adapter
    from silo_rag.langchain_adapter import LangChainAgentResult

    def fake_run(client, question, *, top_k=None):
        client.chat("rerank", "x")  # 検索内のリランキング（LLMClient経由。countingに数えられる）
        return LangChainAgentResult(
            answer=Answer(text="RPT-001を参考に", citations=[]),
            scored_chunks=[_scored("RPT-009"), _scored("RPT-001")],
            searches=2,
            requested_searches=2,
            llm_calls=3,  # エージェント自身の判断。countingを通らないので、別に足される
        )

    monkeypatch.setattr(adapter, "run_langchain_agent", fake_run)

    (result,) = run_eval(_JudgeClient(), _QA, pipeline="langchain")

    assert result.attempts == 2
    assert result.llm_calls == 4  # リランキング1回＋エージェント3回
    assert result.retrieved_report_ids == ["RPT-009", "RPT-001"]
    assert result.cited_gold


def test_format_comparison_marks_rewrite_first_runs():
    payload = _payload("agent", "strict", "qwen-7b", "qwen-7b", 0.7)
    payload["run"]["first_query"] = "rewrite"
    raw = _payload("agent", "strict", "qwen-7b", "qwen-7b", 0.6)
    raw["run"]["first_query"] = "raw"

    lines = format_comparison([("a", raw), ("b", payload)]).splitlines()

    assert lines[2].startswith("| agent(strict) / qwen-7b |")
    assert lines[3].startswith("| agent(strict,rewrite-first) / qwen-7b |")


def test_format_comparison_marks_single_search_runs():
    payload = _payload("agent", "lenient", "qwen-7b", "qwen-7b", 0.7)
    payload["run"]["first_query"] = "rewrite"
    payload["run"]["max_attempts"] = 1

    lines = format_comparison([("a", payload)]).splitlines()

    assert lines[2].startswith("| agent(lenient,rewrite-first,1-search) / qwen-7b |")
