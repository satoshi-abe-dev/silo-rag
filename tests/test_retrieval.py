"""retrieval（DAGノードC）の純粋ロジックのテスト。ChromaDB・LLMは使わない
（tokenize/スコア正規化/フィルタ変換/リランク応答パースはいずれも外部サービス不要）。"""

from __future__ import annotations

from silo_rag.ingest import Chunk
from silo_rag.llm_client import LLMConnectionError
from silo_rag.retrieval import (
    ScoredChunk,
    _build_query_rewrite_prompt,
    _build_where,
    _normalize_bm25,
    _normalize_minmax,
    _parse_rerank_response,
    _resolve_query,
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
