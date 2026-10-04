"""Tests for langchain_adapter (DAG node H): retriever, tool output, search counting and caps.

No ChromaDB or real LLM: search() is stubbed and the agent model replays scripted messages.
"""

from __future__ import annotations

from itertools import count

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import Field

import silo_rag.langchain_adapter as adapter
from silo_rag.ingest import Chunk
from silo_rag.langchain_adapter import (
    _NO_ANSWER,
    TOOL_NAME,
    SiloRetriever,
    _message_text,
    build_retriever_tool,
    run_langchain_agent,
)
from silo_rag.retrieval import ScoredChunk


def _sc(
    chunk_id: str, report_id: str = "RPT-001", dept: str = "営業推進部", score: float = 0.9
) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(
            chunk_id=chunk_id,
            text=f"{chunk_id}の本文",
            metadata={"report_id": report_id, "dept": dept, "section": "教訓"},
        ),
        score=score,
    )


def _call_tool(query: str, call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": TOOL_NAME, "args": {"query": query}, "id": call_id}])


class _FakeToolModel(GenericFakeChatModel):
    """Replays scripted messages; bind_tools is a no-op and inputs are recorded in `seen`."""

    seen: list = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, *args, **kwargs):
        self.seen.append(list(messages))
        return super()._generate(messages, *args, **kwargs)


def _model(*responses) -> _FakeToolModel:
    return _FakeToolModel(messages=iter(responses), seen=[])


@pytest.fixture
def fake_search(monkeypatch):
    calls: list[dict] = []
    results: list[list[ScoredChunk]] = []

    def _search(client, query, *, top_k=None, filters=None, history=None, rewrite_query=None):
        calls.append(
            {
                "query": query,
                "top_k": top_k,
                "filters": filters,
                "history": history,
                "rewrite_query": rewrite_query,
            }
        )
        return results.pop(0) if results else []

    monkeypatch.setattr(adapter, "search", _search)
    return calls, results


# --- SiloRetriever ----------------------------------------------------------------


def test_retriever_converts_chunks_to_documents_and_records(fake_search):
    calls, results = fake_search
    results.append([_sc("c1", "RPT-001", "営業推進部", 0.9), _sc("c2", "RPT-002", "商品開発部", 0.5)])
    retriever = SiloRetriever(client=object(), filters={"dept": "営業推進部"}, top_k=2)

    docs = retriever.invoke("検索クエリ")

    assert [d.page_content for d in docs] == ["c1の本文", "c2の本文"]
    assert docs[0].metadata["report_id"] == "RPT-001"
    assert docs[1].metadata["dept"] == "商品開発部"
    assert docs[0].metadata["chunk_id"] == "c1"
    assert docs[0].metadata["score"] == 0.9
    # The LLM already wrote the query, so search-side rewriting is always off, even if configured on.
    assert calls == [
        {
            "query": "検索クエリ",
            "top_k": 2,
            "filters": {"dept": "営業推進部"},
            "history": None,
            "rewrite_query": False,
        }
    ]
    assert retriever.search_calls == 1
    assert [sc.chunk.chunk_id for sc in retriever.retrieved] == ["c1", "c2"]


def test_retriever_fills_missing_metadata_with_defaults(fake_search):
    _, results = fake_search
    results.append([ScoredChunk(chunk=Chunk(chunk_id="x", text="本文", metadata={}), score=0.1)])

    (doc,) = SiloRetriever(client=object()).invoke("q")

    assert doc.metadata["report_id"] == "不明"
    assert doc.metadata["dept"] == "不明"
    assert doc.metadata["section"] == "不明"


def test_tool_output_includes_citation_header(fake_search):
    _, results = fake_search
    results.append([_sc("c1", "RPT-014", "マーケティング部")])
    tool = build_retriever_tool(SiloRetriever(client=object()))

    assert tool.name == TOOL_NAME
    output = tool.invoke({"query": "q"})

    assert "report_id=RPT-014" in output
    assert "部署=マーケティング部" in output
    assert "c1の本文" in output


def test_retriever_runs_concurrent_searches_one_at_a_time(monkeypatch):
    # Parallel tool calls must not overlap search(): ChromaDB and LM Studio can't handle concurrency.
    import threading
    import time

    active = 0
    max_active = 0
    guard = threading.Lock()

    def slow_search(client, query, **kwargs):
        nonlocal active, max_active
        with guard:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        with guard:
            active -= 1
        return [_sc(f"c-{query}")]

    monkeypatch.setattr(adapter, "search", slow_search)
    retriever = SiloRetriever(client=object())

    threads = [threading.Thread(target=retriever.invoke, args=(f"q{i}",)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert max_active == 1
    assert retriever.search_calls == 4
    assert len(retriever.retrieved) == 4


def test_retriever_stops_searching_after_max_calls(fake_search):
    calls, results = fake_search
    results.extend([[_sc(f"c{i}")] for i in range(5)])
    retriever = SiloRetriever(client=object(), max_calls=2)

    outputs = [retriever.invoke(f"q{i}") for i in range(5)]

    assert len(calls) == 2  # no search from the third call on
    assert [len(o) for o in outputs] == [1, 1, 0, 0, 0]
    assert retriever.search_calls == 2
    assert retriever.search_requests == 5
    assert len(retriever.retrieved) == 2


# --- run_langchain_agent ---------------------------------------------------------


def test_agent_searches_once_then_answers(fake_search):
    calls, results = fake_search
    results.append([_sc("c1", "RPT-001"), _sc("c2", "RPT-002", "商品開発部")])
    model = _model(_call_tool("失敗事例", "t1"), AIMessage(content="RPT-001が参考になります"))

    result = run_langchain_agent(object(), "質問", top_k=5, max_searches=3, model=model)

    assert result.answer.text == "RPT-001が参考になります"
    assert result.searches == 1
    assert result.llm_calls == 2  # one tool-call decision + the final answer
    assert [sc.chunk.chunk_id for sc in result.scored_chunks] == ["c1", "c2"]
    assert {c.report_id for c in result.answer.citations} == {"RPT-001", "RPT-002"}
    assert [c["query"] for c in calls] == ["失敗事例"]
    assert calls[0]["top_k"] == 5


def test_agent_answering_without_search_reports_zero_searches(fake_search):
    # Small models may skip the tool entirely: zero searches, an answer without evidence.
    model = _model(AIMessage(content="検索せずに答えました"))

    result = run_langchain_agent(object(), "質問", top_k=5, max_searches=3, model=model)

    assert result.searches == 0
    assert result.llm_calls == 1
    assert result.scored_chunks == []
    assert result.answer.citations == []


def test_agent_caps_parallel_tool_calls_within_one_response(fake_search):
    # One response can request many parallel searches (~50 on real data); recursion_limit doesn't
    # catch that, so the retriever's own cap must.
    calls, results = fake_search
    results.extend([[_sc(f"c{i}")] for i in range(10)])
    many = AIMessage(
        content="",
        tool_calls=[{"name": TOOL_NAME, "args": {"query": f"q{i}"}, "id": f"t{i}"} for i in range(6)],
    )
    model = _model(many, AIMessage(content="回答"))

    result = run_langchain_agent(object(), "質問", top_k=5, max_searches=3, model=model)

    assert result.searches == 3
    assert result.requested_searches == 6
    assert len(calls) == 3
    assert result.answer.text == "回答"


@pytest.mark.parametrize("searches", [1, 2, 3])
def test_agent_returns_valid_answer_after_exactly_max_searches(fake_search, searches):
    # Regression (codex review): with a tight step limit, using exactly max_searches raised
    # GraphRecursionError after the answer and discarded it.
    _, results = fake_search
    results.extend([[_sc(f"c{i}")] for i in range(searches)])
    model = _model(*[_call_tool(f"q{i}", f"t{i}") for i in range(searches)], AIMessage(content="有効な回答"))

    result = run_langchain_agent(object(), "質問", top_k=5, max_searches=searches, model=model)

    assert result.answer.text == "有効な回答"
    assert result.searches == searches
    assert result.llm_calls == searches + 1


def test_agent_counts_actual_model_calls_when_it_loops_on_an_unknown_tool(fake_search):
    # Regression: llm_calls was estimated as searches + 1, under-reporting a model that keeps
    # calling an invalid tool. It must be the actual count.
    wrong = (
        AIMessage(content="", tool_calls=[{"name": "wrong_tool", "args": {"query": "q"}, "id": f"t{i}"}])
        for i in count()
    )
    model = _FakeToolModel(messages=wrong, seen=[])

    result = run_langchain_agent(object(), "質問", top_k=5, max_searches=3, model=model)

    assert result.searches == 0
    assert result.answer.text == _NO_ANSWER
    assert result.llm_calls == len(model.seen)
    assert result.llm_calls > 1


def test_agent_dedups_chunks_across_searches(fake_search):
    _, results = fake_search
    results.extend([[_sc("c1"), _sc("c2")], [_sc("c2"), _sc("c3")]])
    model = _model(_call_tool("a", "t1"), _call_tool("b", "t2"), AIMessage(content="回答"))

    result = run_langchain_agent(object(), "質問", top_k=5, max_searches=3, model=model)

    assert result.searches == 2
    assert [sc.chunk.chunk_id for sc in result.scored_chunks] == ["c1", "c2", "c3"]


def test_agent_stops_at_max_searches(fake_search):
    _, results = fake_search
    results.extend([[_sc(f"c{i}")] for i in range(10)])
    endless = (_call_tool(f"q{i}", f"t{i}") for i in count())
    model = _FakeToolModel(messages=endless, seen=[])

    result = run_langchain_agent(object(), "質問", top_k=5, max_searches=2, model=model)

    assert result.searches == 2  # the third search never runs
    assert result.answer.text == _NO_ANSWER
    assert len(result.scored_chunks) == 2


def test_agent_passes_history_as_messages(fake_search):
    model = _model(AIMessage(content="回答"))
    history = [
        ("古い質問1", "回答1"),
        ("古い質問2", "回答2"),
        ("古い質問3", "回答3"),
        ("直前の質問", "直前の回答"),
    ]

    run_langchain_agent(object(), "それについて詳しく", history=history, top_k=5, max_searches=3, model=model)

    (seen,) = model.seen
    human_texts = [m.content for m in seen if isinstance(m, HumanMessage)]
    # Only the last 3 of 4 turns are passed.
    assert human_texts == ["古い質問2", "古い質問3", "直前の質問", "それについて詳しく"]


def test_agent_counts_only_new_messages_not_history(fake_search):
    model = _model(AIMessage(content="回答"))

    result = run_langchain_agent(object(), "質問", history=[("q", "a")], top_k=5, max_searches=3, model=model)

    assert result.llm_calls == 1


def test_agent_max_searches_below_one_is_clamped(fake_search):
    _, results = fake_search
    results.append([_sc("c1")])
    model = _model(_call_tool("q", "t1"), AIMessage(content="回答"))

    result = run_langchain_agent(object(), "質問", top_k=5, max_searches=0, model=model)

    assert result.searches == 1


# --- Pure functions --------------------------------------------------------------


def test_message_text_handles_string_and_block_content():
    assert _message_text(AIMessage(content="  回答  ")) == "回答"
    blocks = [{"type": "text", "text": "前半"}, {"type": "text", "text": "後半"}]
    assert _message_text(AIMessage(content=blocks)) == "前半後半"
