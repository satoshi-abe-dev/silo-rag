"""langchain_adapter（DAGノードH、LangChain連携）のテスト。ChromaDB・実LLMは使わない。

search()は差し替え、エージェントのLLMは「決めた順にメッセージを返す」フェイクにして、
Retrieverの変換・ツール出力・エージェントの検索回数の数え方・上限での停止を検証する。
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
    """決めた順にメッセージを返すフェイク。bind_tools（ツールをモデルに渡す処理）は何もしない。
    モデルに渡されたメッセージ列は、seenに記録する。"""

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

    def _search(client, query, *, top_k=None, filters=None, history=None):
        calls.append({"query": query, "top_k": top_k, "filters": filters, "history": history})
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
    assert calls == [{"query": "検索クエリ", "top_k": 2, "filters": {"dept": "営業推進部"}, "history": None}]
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


# --- run_langchain_agent ---------------------------------------------------------


def test_agent_searches_once_then_answers(fake_search):
    calls, results = fake_search
    results.append([_sc("c1", "RPT-001"), _sc("c2", "RPT-002", "商品開発部")])
    model = _model(_call_tool("失敗事例", "t1"), AIMessage(content="RPT-001が参考になります"))

    result = run_langchain_agent(object(), "質問", top_k=5, max_searches=3, model=model)

    assert result.answer.text == "RPT-001が参考になります"
    assert result.searches == 1
    assert result.llm_calls == 2  # ツール呼び出しの判断1回＋最終回答1回
    assert [sc.chunk.chunk_id for sc in result.scored_chunks] == ["c1", "c2"]
    assert {c.report_id for c in result.answer.citations} == {"RPT-001", "RPT-002"}
    assert [c["query"] for c in calls] == ["失敗事例"]
    assert calls[0]["top_k"] == 5


def test_agent_answering_without_search_reports_zero_searches(fake_search):
    # ツール呼び出しをしないモデル（小さなモデルで起きうる）。検索0回で、根拠なしの回答になる。
    model = _model(AIMessage(content="検索せずに答えました"))

    result = run_langchain_agent(object(), "質問", top_k=5, max_searches=3, model=model)

    assert result.searches == 0
    assert result.llm_calls == 1
    assert result.scored_chunks == []
    assert result.answer.citations == []


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

    assert result.searches == 2  # 3回目の検索は実行されない
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
    # 4往復の履歴のうち、直近3往復だけが渡る。
    assert human_texts == ["古い質問2", "古い質問3", "直前の質問", "それについて詳しく"]


def test_agent_counts_only_new_messages_not_history(fake_search):
    # 履歴のAIMessageをllm_callsに数えない。
    model = _model(AIMessage(content="回答"))

    result = run_langchain_agent(object(), "質問", history=[("q", "a")], top_k=5, max_searches=3, model=model)

    assert result.llm_calls == 1


def test_agent_max_searches_below_one_is_clamped(fake_search):
    _, results = fake_search
    results.append([_sc("c1")])
    model = _model(_call_tool("q", "t1"), AIMessage(content="回答"))

    result = run_langchain_agent(object(), "質問", top_k=5, max_searches=0, model=model)

    assert result.searches == 1


# --- 純粋関数 --------------------------------------------------------------------


def test_message_text_handles_string_and_block_content():
    assert _message_text(AIMessage(content="  回答  ")) == "回答"
    blocks = [{"type": "text", "text": "前半"}, {"type": "text", "text": "後半"}]
    assert _message_text(AIMessage(content=blocks)) == "前半後半"
