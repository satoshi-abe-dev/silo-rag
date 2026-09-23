"""generation（DAGノードD）の純粋ロジックのテスト。LLMは使わない
（空チャンク時にLLMを一切呼ばないことも、フェイクのclientで検証する）。"""

from __future__ import annotations

from cae_rag.generation import Citation, _build_citations, _build_context_block, answer_question
from cae_rag.ingest import Chunk


def _chunk(report_id: str, section: str, dept: str, text: str = "本文") -> Chunk:
    return Chunk(
        chunk_id=f"{report_id}::{section}",
        text=text,
        metadata={"report_id": report_id, "section": section, "dept": dept},
    )


def test_build_citations_dedup_by_report_and_section():
    chunks = [
        _chunk("RPT-001", "解析目的", "ボディ設計部"),
        _chunk("RPT-001", "解析目的", "ボディ設計部"),  # 完全な重複 -> 1件に畳まれる
        _chunk("RPT-001", "結果サマリー", "ボディ設計部"),  # 同一レポートの別セクション -> 別件
        _chunk("RPT-002", "解析目的", "シャシー設計部"),
    ]
    citations = _build_citations(chunks)
    assert citations == [
        Citation(report_id="RPT-001", section="解析目的", dept="ボディ設計部"),
        Citation(report_id="RPT-001", section="結果サマリー", dept="ボディ設計部"),
        Citation(report_id="RPT-002", section="解析目的", dept="シャシー設計部"),
    ]


def test_build_citations_empty_for_no_chunks():
    assert _build_citations([]) == []


def test_build_context_block_includes_metadata_header():
    chunk = _chunk(
        "RPT-014", "トラブルシューティング・教訓", "パワートレイン設計部", text="共振が問題になった。"
    )
    block = _build_context_block(1, chunk)
    assert "RPT-014" in block
    assert "パワートレイン設計部" in block
    assert "トラブルシューティング・教訓" in block
    assert "共振が問題になった。" in block


class _NeverCallLLMClient:
    """chunksが空のときにLLMへ問い合わせないことを確認するためのフェイク。
    .chatが呼ばれたらテストを失敗させる。"""

    def chat(self, *args, **kwargs):
        raise AssertionError("chunksが空なのにLLMが呼び出された")


def test_answer_question_with_no_chunks_skips_llm_call():
    answer = answer_question(_NeverCallLLMClient(), "質問", [])
    assert answer.citations == []
    assert answer.text  # 何らかの「見つかりませんでした」系メッセージが返る
