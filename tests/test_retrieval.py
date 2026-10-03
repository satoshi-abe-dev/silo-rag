"""retrieval（DAGノードC）の純粋ロジックのテスト。ChromaDB・LLMは使わない
（tokenize/スコア正規化/フィルタ変換/リランク応答パースはいずれも外部サービス不要）。"""

from __future__ import annotations

import pytest

import silo_rag.retrieval as retrieval
from silo_rag.config import Config
from silo_rag.ingest import Chunk
from silo_rag.llm_client import LLMConnectionError
from silo_rag.retrieval import (
    PLAN_SYSTEM_PROMPT,
    ScoredChunk,
    _build_query_rewrite_prompt,
    _build_where,
    _normalize_bm25,
    _normalize_minmax,
    _parse_rerank_response,
    _resolve_query,
    build_plan_prompt,
    plan_query,
    search,
    tokenize,
)


def test_tokenize_ascii_kept_whole():
    assert tokenize("RPT-014") == ["rpt-014"]


def test_tokenize_japanese_uses_bigrams():
    assert tokenize("解析") == ["解析"]  # 2文字なので1つの2-gram
    assert tokenize("解析目的") == ["解析", "析目", "目的"]


def test_tokenize_mixed_ascii_and_japanese():
    # ASCII語（KPI）はそのまま1トークン、続く日本語部分（で確認）はbigram化される。
    assert tokenize("KPIで確認") == ["kpi", "で確", "確認"]


def test_tokenize_whitespace_is_separator_not_token():
    tokens = tokenize("abc def")
    assert "abc" in tokens
    assert "def" in tokens
    assert " " not in tokens
    assert "" not in tokens


def test_normalize_minmax_basic():
    result = _normalize_minmax({"a": 0.0, "b": 5.0, "c": 10.0})
    assert result["a"] == 0.0
    assert result["b"] == 0.5
    assert result["c"] == 1.0


def test_normalize_minmax_all_equal_returns_one():
    result = _normalize_minmax({"a": 3.0, "b": 3.0})
    assert result == {"a": 1.0, "b": 1.0}


def test_normalize_minmax_empty():
    assert _normalize_minmax({}) == {}


def test_normalize_bm25_all_nonpositive_stays_zero():
    # 全員ヒットなし（BM25スコアが0以下）のとき、min-maxで全員1.0に底上げされてはいけない。
    result = _normalize_bm25({"a": 0.0, "b": 0.0})
    assert result == {"a": 0.0, "b": 0.0}


def test_normalize_bm25_normal_case_uses_minmax():
    result = _normalize_bm25({"a": 1.0, "b": 3.0})
    assert result["a"] == 0.0
    assert result["b"] == 1.0


def test_build_where_none_for_no_filters():
    assert _build_where(None) is None
    assert _build_where({}) is None


def test_build_where_single_key_is_plain_dict():
    assert _build_where({"dept": "マーケティング部"}) == {"dept": "マーケティング部"}


def test_build_where_multi_key_uses_and():
    where = _build_where({"dept": "マーケティング部", "project_type": "新規事業立ち上げ"})
    assert where == {
        "$and": [
            {"dept": "マーケティング部"},
            {"project_type": "新規事業立ち上げ"},
        ]
    }


def _candidates(n: int) -> list[ScoredChunk]:
    return [
        ScoredChunk(chunk=Chunk(chunk_id=f"c{i}", text=f"text{i}", metadata={}), score=0.0)
        for i in range(1, n + 1)
    ]


def test_parse_rerank_response_valid_json():
    order = _parse_rerank_response("[3, 1, 2]", 3)
    assert order == [3, 1, 2]


def test_parse_rerank_response_ignores_out_of_range_and_duplicates():
    order = _parse_rerank_response("[3, 3, 99, 0, 1]", 3)
    assert order == [3, 1]  # 範囲外(99, 0)は無視、重複(3)は1回だけ


def test_parse_rerank_response_returns_none_on_garbage():
    assert _parse_rerank_response("そんな候補はありません", 3) is None


def test_parse_rerank_response_returns_none_on_broken_json():
    assert _parse_rerank_response("[1, 2,", 3) is None


def test_parse_rerank_response_extracts_array_from_surrounding_text():
    order = _parse_rerank_response("回答: [2, 1] です。", 2)
    assert order == [2, 1]


def test_build_query_rewrite_prompt_includes_history_and_question():
    prompt = _build_query_rewrite_prompt(
        "それについてもう少し詳しく", [("マーケの施策で参考事例は？", "RPT-001が参考になります。")]
    )
    assert "Q: マーケの施策で参考事例は？" in prompt
    assert "A: RPT-001が参考になります。" in prompt
    assert "新しい質問: それについてもう少し詳しく" in prompt


class _StubLLMClient:
    def __init__(self, response: str | None = None, raises: bool = False):
        self.response = response
        self.raises = raises
        self.calls = 0

    def chat(self, system: str, user: str, **kwargs) -> str:
        self.calls += 1
        if self.raises:
            raise LLMConnectionError("接続失敗")
        return self.response


def test_resolve_query_without_history_skips_llm_call():
    client = _StubLLMClient(response="呼ばれたら困る")
    assert _resolve_query(client, "質問", []) == "質問"
    assert client.calls == 0


def test_resolve_query_returns_rewritten_query():
    client = _StubLLMClient(response="マーケティング部の新商品ローンチキャンペーンの失敗事例")
    result = _resolve_query(client, "それについてもう少し詳しく", [("元の質問", "元の回答")])
    assert result == "マーケティング部の新商品ローンチキャンペーンの失敗事例"
    assert client.calls == 1


def test_resolve_query_falls_back_to_original_on_llm_failure():
    client = _StubLLMClient(raises=True)
    assert _resolve_query(client, "元の質問", [("前の質問", "前の回答")]) == "元の質問"


# --- 検索クエリの作成（plan_query）と、search()のrewrite_query ----------------------------


def test_build_plan_prompt_includes_question_and_history():
    prompt = build_plan_prompt("その費用は？", [("新商品の施策は？", "RPT-001です。")])
    assert "質問: その費用は？" in prompt
    assert "Q: 新商品の施策は？" in prompt
    assert prompt.endswith("検索クエリ:")


def test_plan_query_returns_the_query_the_llm_wrote():
    client = _StubLLMClient(response="「新商品ローンチの失敗事例」\n説明文")
    assert plan_query(client, "営業推進部です。新商品ローンチで失敗した事例を教えてください。") == (
        "新商品ローンチの失敗事例",
        None,
    )
    assert client.calls == 1


def test_plan_query_reports_llm_failure_without_raising():
    assert plan_query(_StubLLMClient(raises=True), "質問") == (None, "LLM呼び出しに失敗")


def test_plan_query_reports_empty_response():
    assert plan_query(_StubLLMClient(response="  \n"), "質問") == (None, "応答が空")


class _EmptyCorpus:
    """search()が早期に終わるよう、BM25の候補が0件になる偽のChromaDB。使われたクエリを記録する。"""

    def __init__(self):
        self.bm25_queries: list[str] = []


@pytest.fixture
def empty_search(monkeypatch):
    """ChromaDB・埋め込みを使わずに、search()が最初のBM25検索に渡すクエリだけを調べる。"""
    import chromadb

    corpus = _EmptyCorpus()

    class _FakeChroma:
        def __init__(self, path):
            pass

        def get_collection(self, name):
            return object()

    def fake_bm25(collection, query, top_k, where):
        corpus.bm25_queries.append(query)
        return {}, {}

    monkeypatch.setattr(chromadb, "PersistentClient", _FakeChroma)
    monkeypatch.setattr(retrieval, "_bm25_search", fake_bm25)

    def set_config(rewrite_query: bool) -> None:
        config = Config()
        config.retrieval.rewrite_query = rewrite_query
        monkeypatch.setattr(retrieval, "load_config", lambda: config)

    set_config(False)
    corpus.set_config = set_config  # type: ignore[attr-defined]
    return corpus


class _ScriptedClient:
    """システムプロンプトで、plan・履歴での書き換えの応答を出し分ける。"""

    def __init__(self, plan: str | None = "計画されたクエリ", resolve: str = "履歴で解決したクエリ"):
        self.plan = plan
        self.resolve = resolve
        self.plan_calls = 0
        self.resolve_calls = 0

    def chat(self, system: str, user: str, **kwargs) -> str:
        if system == PLAN_SYSTEM_PROMPT:
            self.plan_calls += 1
            if self.plan is None:
                raise LLMConnectionError("接続失敗")
            return self.plan
        self.resolve_calls += 1
        return self.resolve


def test_search_uses_the_question_as_is_by_default(empty_search):
    client = _ScriptedClient()
    search(client, "元の質問")
    assert empty_search.bm25_queries == ["元の質問"]
    assert (client.plan_calls, client.resolve_calls) == (0, 0)


def test_search_rewrites_the_query_when_asked(empty_search):
    client = _ScriptedClient()
    search(client, "元の質問", rewrite_query=True)
    assert empty_search.bm25_queries == ["計画されたクエリ"]
    assert (client.plan_calls, client.resolve_calls) == (1, 0)


def test_search_follows_the_config_when_rewrite_query_is_not_given(empty_search):
    empty_search.set_config(True)
    search(_ScriptedClient(), "元の質問")
    assert empty_search.bm25_queries == ["計画されたクエリ"]


def test_search_argument_overrides_the_config(empty_search):
    empty_search.set_config(True)
    client = _ScriptedClient()
    search(client, "元の質問", rewrite_query=False)
    assert empty_search.bm25_queries == ["元の質問"]
    assert client.plan_calls == 0


def test_search_falls_back_to_the_question_when_planning_fails(empty_search):
    search(_ScriptedClient(plan=None), "元の質問", rewrite_query=True)
    assert empty_search.bm25_queries == ["元の質問"]


def test_search_falls_back_to_history_rewrite_when_planning_fails(empty_search):
    client = _ScriptedClient(plan=None)
    search(client, "それは？", history=[("前の質問", "前の回答")], rewrite_query=True)
    assert empty_search.bm25_queries == ["履歴で解決したクエリ"]


def test_search_with_history_and_plan_uses_one_llm_call(empty_search):
    # 履歴の指示語の解決は、planのプロンプトの中で行うので、書き換えのLLM呼び出しは重ねない。
    client = _ScriptedClient()
    search(client, "それは？", history=[("前の質問", "前の回答")], rewrite_query=True)
    assert empty_search.bm25_queries == ["計画されたクエリ"]
    assert (client.plan_calls, client.resolve_calls) == (1, 0)
