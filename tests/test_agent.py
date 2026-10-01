"""agent（DAGノードG、LangGraph）のテスト。ChromaDB・LLMは使わない。

search()とanswer_question()は差し替え、LLMClientは「判定」「書き直し」の応答を
台本どおりに返すフェイクにして、グラフの分岐とループの止まり方だけを検証する。
"""

from __future__ import annotations

import pytest

import silo_rag.agent as agent_module
from silo_rag.agent import (
    _GRADE_SYSTEM_PROMPTS,
    _REWRITE_SYSTEM_PROMPT,
    _build_rerank_query,
    _merge_chunks,
    _parse_grade_response,
    _parse_rewrite_response,
    run_agent,
)
from silo_rag.generation import Answer
from silo_rag.ingest import Chunk
from silo_rag.llm_client import LLMConnectionError
from silo_rag.retrieval import ScoredChunk


def _sc(chunk_id: str, score: float = 1.0) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(chunk_id=chunk_id, text=f"{chunk_id}の本文", metadata={"report_id": chunk_id}),
        score=score,
    )


class _ScriptedAgentClient:
    """判定・書き直しの応答を、それぞれ台本どおりに順番に返すフェイク。
    台本の要素がLLMConnectionErrorのインスタンスなら、それを送出する。"""

    def __init__(self, grades: list[object] | None = None, rewrites: list[object] | None = None):
        self.grades = list(grades or [])
        self.rewrites = list(rewrites or [])
        self.grade_calls = 0
        self.rewrite_calls = 0
        self.grade_prompts: list[str] = []

    def chat(self, system: str, user: str, **kwargs) -> str:
        if system in _GRADE_SYSTEM_PROMPTS.values():
            self.grade_calls += 1
            self.grade_prompts.append(user)
            item = self.grades.pop(0)
        elif system == _REWRITE_SYSTEM_PROMPT:
            self.rewrite_calls += 1
            item = self.rewrites.pop(0)
        else:
            raise AssertionError(f"想定外のプロンプト: {system[:30]}")
        if isinstance(item, Exception):
            raise item
        return str(item)


@pytest.fixture
def fake_pipeline(monkeypatch):
    """search()・answer_question()を差し替え、呼び出し内容を記録する。

    results: search()が呼ばれるたびに順番に返す検索結果。足りなくなったら空リストを返す。
    """

    class _Recorder:
        def __init__(self):
            self.results: list[list[ScoredChunk]] = []
            self.search_calls: list[dict] = []
            self.rerank_calls: list[dict] = []
            # rerank()の並べ替え方。既定は入力の順序のまま返す。
            self.rerank_order = lambda candidates: list(candidates)
            self.generated_with: list[str] | None = None

    rec = _Recorder()

    def fake_search(client, query, *, top_k=None, filters=None, history=None):
        rec.search_calls.append({"query": query, "top_k": top_k, "filters": filters, "history": history})
        return rec.results.pop(0) if rec.results else []

    def fake_answer_question(client, question, chunks, history=None):
        rec.generated_with = [c.chunk_id for c in chunks]
        return Answer(text="回答", citations=[])

    def fake_rerank(client, query, candidates):
        rec.rerank_calls.append({"query": query, "ids": [sc.chunk.chunk_id for sc in candidates]})
        return rec.rerank_order(candidates)

    monkeypatch.setattr(agent_module, "search", fake_search)
    monkeypatch.setattr(agent_module, "rerank", fake_rerank)
    monkeypatch.setattr(agent_module, "answer_question", fake_answer_question)
    return rec


# --- グラフの分岐とループ ----------------------------------------------------------


def test_sufficient_on_first_try_skips_rewrite(fake_pipeline):
    fake_pipeline.results = [[_sc("a"), _sc("b")]]
    client = _ScriptedAgentClient(grades=["SUFFICIENT"])

    result = run_agent(client, "質問", top_k=5, max_attempts=3)

    assert len(fake_pipeline.search_calls) == 1
    assert client.grade_calls == 1
    assert client.rewrite_calls == 0
    assert result.attempts == 1
    assert fake_pipeline.generated_with == ["a", "b"]
    assert result.answer.text == "回答"


def test_insufficient_retries_with_rewritten_query(fake_pipeline):
    fake_pipeline.results = [[_sc("a")], [_sc("b")]]
    client = _ScriptedAgentClient(grades=["INSUFFICIENT", "SUFFICIENT"], rewrites=["別の切り口のクエリ"])

    result = run_agent(client, "質問", top_k=5, max_attempts=3)

    assert [c["query"] for c in fake_pipeline.search_calls] == ["質問", "別の切り口のクエリ"]
    assert result.tried_queries == ["質問", "別の切り口のクエリ"]
    assert result.attempts == 2
    # 最新の検索結果が優先され、前回の結果は後ろに残る。
    assert fake_pipeline.generated_with == ["b", "a"]


def test_loop_stops_at_max_attempts(fake_pipeline):
    fake_pipeline.results = [[_sc("a")], [_sc("b")], [_sc("c")]]
    client = _ScriptedAgentClient(grades=["INSUFFICIENT"] * 3, rewrites=["クエリ2", "クエリ3"])

    result = run_agent(client, "質問", top_k=5, max_attempts=3)

    assert len(fake_pipeline.search_calls) == 3
    assert client.rewrite_calls == 2  # 上限に達した後は書き直さない
    assert result.attempts == 3
    assert result.answer is not None  # 不十分のままでも、手元のチャンクで回答する


def test_max_attempts_one_never_rewrites(fake_pipeline):
    fake_pipeline.results = [[_sc("a")]]
    client = _ScriptedAgentClient(grades=["INSUFFICIENT"])

    result = run_agent(client, "質問", top_k=5, max_attempts=1)

    assert client.rewrite_calls == 0
    assert result.attempts == 1


def test_max_attempts_below_one_is_clamped_to_one(fake_pipeline):
    fake_pipeline.results = [[_sc("a")]]
    client = _ScriptedAgentClient(grades=["INSUFFICIENT"])

    result = run_agent(client, "質問", top_k=5, max_attempts=0)

    assert len(fake_pipeline.search_calls) == 1
    assert result.attempts == 1


def test_unparsable_grade_is_treated_as_sufficient(fake_pipeline):
    fake_pipeline.results = [[_sc("a")]]
    client = _ScriptedAgentClient(grades=["たぶん足りると思います"])

    result = run_agent(client, "質問", top_k=5, max_attempts=3)

    assert client.rewrite_calls == 0
    assert result.attempts == 1
    assert any("解釈できず" in t for t in result.trace)


def test_grade_connection_error_is_treated_as_sufficient(fake_pipeline):
    fake_pipeline.results = [[_sc("a")]]
    client = _ScriptedAgentClient(grades=[LLMConnectionError("接続失敗")])

    result = run_agent(client, "質問", top_k=5, max_attempts=3)

    assert client.rewrite_calls == 0
    assert result.answer is not None


def test_rewrite_connection_error_stops_loop(fake_pipeline):
    fake_pipeline.results = [[_sc("a")]]
    client = _ScriptedAgentClient(grades=["INSUFFICIENT"], rewrites=[LLMConnectionError("接続失敗")])

    result = run_agent(client, "質問", top_k=5, max_attempts=3)

    assert len(fake_pipeline.search_calls) == 1
    assert result.answer is not None


def test_rewrite_repeating_tried_query_stops_loop(fake_pipeline):
    fake_pipeline.results = [[_sc("a")]]
    # 空白・大小文字の違いだけなら「同じクエリ」とみなす。
    client = _ScriptedAgentClient(grades=["INSUFFICIENT"], rewrites=["KPI 改善"])

    result = run_agent(client, "kpi改善", top_k=5, max_attempts=3)

    assert len(fake_pipeline.search_calls) == 1
    assert any("試行済み" in t for t in result.trace)


def test_empty_results_skip_grade_llm_call(fake_pipeline):
    fake_pipeline.results = [[], [_sc("a")]]
    client = _ScriptedAgentClient(grades=["SUFFICIENT"], rewrites=["別のクエリ"])

    result = run_agent(client, "質問", top_k=5, max_attempts=3)

    assert client.grade_calls == 1  # 0件の回はLLMを呼ばずに「不十分」とする
    assert result.attempts == 2
    assert fake_pipeline.generated_with == ["a"]


def test_history_and_filters_are_passed_correctly(fake_pipeline):
    fake_pipeline.results = [[_sc("a")], [_sc("b")]]
    client = _ScriptedAgentClient(grades=["INSUFFICIENT", "SUFFICIENT"], rewrites=["別のクエリ"])
    history = [("前の質問", "前の回答")]
    filters = {"dept": "営業部"}

    run_agent(client, "それについて詳しく", filters=filters, history=history, top_k=5, max_attempts=3)

    first, second = fake_pipeline.search_calls
    # 初回はsearch()側の履歴込みクエリ書き換えに任せる。2回目以降はrewrite済みなので渡さない。
    assert first["history"] == history
    assert second["history"] is None
    assert first["filters"] == second["filters"] == filters


def test_retry_keeps_earlier_evidence_when_latest_fills_top_k(fake_pipeline):
    # 再検索がtop_k件ちょうどを返しても、前回の上位の根拠が押し出されないこと
    # （codexレビューで指摘されたケース）。
    fake_pipeline.results = [[_sc("first-a"), _sc("first-b")], [_sc("second-a"), _sc("second-b")]]
    client = _ScriptedAgentClient(grades=["INSUFFICIENT", "SUFFICIENT"], rewrites=["別のクエリ"])

    run_agent(client, "質問", top_k=2, max_attempts=2)

    assert fake_pipeline.generated_with == ["second-a", "first-a"]
    # 2回目の判定にも、前回の根拠が含まれている。
    assert "first-a" in client.grade_prompts[1]


def test_grade_prompt_includes_history_for_follow_up_question(fake_pipeline):
    fake_pipeline.results = [[_sc("a")]]
    client = _ScriptedAgentClient(grades=["SUFFICIENT"])
    history = [("新商品ローンチキャンペーンの失敗事例は？", "RPT-001が参考になります。")]

    run_agent(client, "その費用は？", history=history, top_k=5, max_attempts=3)

    prompt = client.grade_prompts[0]
    assert "Q: 新商品ローンチキャンペーンの失敗事例は？" in prompt
    assert "質問: その費用は？" in prompt


def test_grade_prompt_without_history_has_no_history_block(fake_pipeline):
    fake_pipeline.results = [[_sc("a")]]
    client = _ScriptedAgentClient(grades=["SUFFICIENT"])

    run_agent(client, "質問", top_k=5, max_attempts=3)

    assert "会話履歴" not in client.grade_prompts[0]


def test_single_search_skips_final_rerank(fake_pipeline):
    # 1回だけならsearch()内で並べ直し済みなので、最終選定でLLMを呼ばない。
    fake_pipeline.results = [[_sc("a")]]
    client = _ScriptedAgentClient(grades=["SUFFICIENT"])

    run_agent(client, "質問", top_k=5, max_attempts=3)

    assert fake_pipeline.rerank_calls == []


def test_final_rerank_sees_all_collected_chunks_and_decides_order(fake_pipeline):
    # 交互の並びのままtop_kで切ると落ちる「1回目の2位」も、最終選定の候補には入ること
    # （実データでbaselineの正解がこれで落ちたケース）。
    fake_pipeline.results = [[_sc("a1"), _sc("a2")], [_sc("b1"), _sc("b2")], [_sc("c1"), _sc("c2")]]
    fake_pipeline.rerank_order = lambda cands: sorted(cands, key=lambda sc: sc.chunk.chunk_id != "a2")
    client = _ScriptedAgentClient(grades=["INSUFFICIENT"] * 3, rewrites=["クエリ2", "クエリ3"])

    result = run_agent(client, "元の質問", top_k=2, max_attempts=3)

    (call,) = fake_pipeline.rerank_calls
    assert call["query"] == "元の質問"
    assert sorted(call["ids"]) == ["a1", "a2", "b1", "b2", "c1", "c2"]
    assert fake_pipeline.generated_with[0] == "a2"
    assert result.scored_chunks[0].chunk.chunk_id == "a2"
    assert any("最終選定" in t for t in result.trace)


def test_build_rerank_query_adds_previous_question_for_follow_up():
    assert _build_rerank_query("質問", None) == "質問"
    query = _build_rerank_query("その費用は？", [("古い質問", "回答"), ("新商品の失敗事例は？", "回答")])
    assert "新商品の失敗事例は？" in query
    assert "その費用は？" in query
    assert "古い質問" not in query


def test_top_k_limits_chunks_passed_to_generation(fake_pipeline):
    fake_pipeline.results = [[_sc("a"), _sc("b"), _sc("c")]]
    client = _ScriptedAgentClient(grades=["SUFFICIENT"])

    result = run_agent(client, "質問", top_k=2, max_attempts=3)

    assert fake_pipeline.generated_with == ["a", "b"]
    assert [sc.chunk.chunk_id for sc in result.scored_chunks] == ["a", "b"]


# --- 純粋関数 --------------------------------------------------------------------


def test_merge_chunks_interleaves_by_rank_latest_first():
    # 新しい回が先頭。各回の1位→各回の2位→…の順に並ぶ。
    merged = _merge_chunks([[_sc("c"), _sc("d")], [_sc("a"), _sc("b")]])
    assert [sc.chunk.chunk_id for sc in merged] == ["c", "a", "d", "b"]


def test_merge_chunks_dedups_keeping_first_occurrence():
    merged = _merge_chunks([[_sc("c"), _sc("a")], [_sc("a"), _sc("b")]])
    assert [sc.chunk.chunk_id for sc in merged] == ["c", "a", "b"]


def test_merge_chunks_handles_uneven_and_empty_results():
    merged = _merge_chunks([[], [_sc("a"), _sc("b"), _sc("c")], [_sc("x")]])
    assert [sc.chunk.chunk_id for sc in merged] == ["a", "x", "b", "c"]
    assert _merge_chunks([]) == []


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("SUFFICIENT", "sufficient"),
        ("INSUFFICIENT", "insufficient"),
        ("  sufficient。\n", "sufficient"),
        ("Insufficient.", "insufficient"),
        # 部分一致で拾わない（"INSUFFICIENT"の中の"SUFFICIENT"等）。
        ("SUFFICIENT because the context covers it", None),
        ("答え: INSUFFICIENT", None),
        ("", None),
    ],
)
def test_parse_grade_response(raw, expected):
    assert _parse_grade_response(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("新商品ローンチの失敗事例", "新商品ローンチの失敗事例"),
        ("「新商品ローンチの失敗事例」", "新商品ローンチの失敗事例"),
        ("\n\n  クエリ  \n説明文", "クエリ"),
        ("   \n", None),
    ],
)
def test_parse_rewrite_response(raw, expected):
    assert _parse_rewrite_response(raw) == expected


@pytest.mark.parametrize("mode", ["strict", "lenient"])
def test_grade_mode_selects_system_prompt(fake_pipeline, mode):
    fake_pipeline.results = [[_sc("a")]]
    seen: list[str] = []

    class _Client(_ScriptedAgentClient):
        def chat(self, system, user, **kwargs):
            seen.append(system)
            return super().chat(system, user, **kwargs)

    run_agent(_Client(grades=["SUFFICIENT"]), "質問", top_k=5, max_attempts=3, grade_mode=mode)

    assert seen == [_GRADE_SYSTEM_PROMPTS[mode]]


def test_unknown_grade_mode_raises(fake_pipeline):
    with pytest.raises(ValueError):
        run_agent(_ScriptedAgentClient(), "質問", top_k=5, max_attempts=3, grade_mode="medium")
