"""generation（DAGノードD）の純粋ロジックのテスト。LLMは使わない
（空チャンク時にLLMを一切呼ばないことも、フェイクのclientで検証する）。"""

from __future__ import annotations

from silo_rag.generation import (
    Citation,
    _build_citations,
    _build_context_block,
    _build_history_block,
    _build_prompt,
    answer_question,
)
from silo_rag.ingest import Chunk


def _chunk(report_id: str, section: str, dept: str, text: str = "本文") -> Chunk:
    return Chunk(
        chunk_id=f"{report_id}::{section}",
        text=text,
        metadata={"report_id": report_id, "section": section, "dept": dept},
    )


def test_build_citations_dedup_by_report_and_section():
    chunks = [
        _chunk("RPT-001", "プロジェクト目的", "マーケティング部"),
        _chunk("RPT-001", "プロジェクト目的", "マーケティング部"),  # 完全な重複 -> 1件に畳まれる
        _chunk("RPT-001", "成果サマリー", "マーケティング部"),  # 同一レポートの別セクション -> 別件
        _chunk("RPT-002", "プロジェクト目的", "営業推進部"),
    ]
    citations = _build_citations(chunks)
    assert citations == [
        Citation(report_id="RPT-001", section="プロジェクト目的", dept="マーケティング部"),
        Citation(report_id="RPT-001", section="成果サマリー", dept="マーケティング部"),
        Citation(report_id="RPT-002", section="プロジェクト目的", dept="営業推進部"),
    ]


def test_build_citations_empty_for_no_chunks():
    assert _build_citations([]) == []


def test_build_context_block_includes_metadata_header():
    chunk = _chunk(
        "RPT-014", "教訓・つまずいたポイント", "経営企画部", text="関係部署への説明不足が問題になった。"
    )
    block = _build_context_block(1, chunk)
    assert "RPT-014" in block
    assert "経営企画部" in block
    assert "教訓・つまずいたポイント" in block
    assert "関係部署への説明不足が問題になった。" in block


class _NeverCallLLMClient:
    """chunksが空のときにLLMへ問い合わせないことを確認するためのフェイク。
    .chatが呼ばれたらテストを失敗させる。"""

    def chat(self, *args, **kwargs):
        raise AssertionError("chunksが空なのにLLMが呼び出された")


def test_answer_question_with_no_chunks_skips_llm_call():
    answer = answer_question(_NeverCallLLMClient(), "質問", [])
    assert answer.citations == []
    assert answer.text  # 何らかの「見つかりませんでした」系メッセージが返る


def test_build_history_block_empty_for_no_history():
    assert _build_history_block(None) == ""
    assert _build_history_block([]) == ""


def test_build_history_block_includes_qa_pairs():
    block = _build_history_block([("マーケの施策で参考事例は？", "RPT-001の事例が参考になります。")])
    assert "Q: マーケの施策で参考事例は？" in block
    assert "A: RPT-001の事例が参考になります。" in block


def test_build_history_block_truncates_long_answers():
    long_answer = "あ" * 300
    block = _build_history_block([("質問", long_answer)])
    assert "あ" * 200 + "…" in block
    assert "あ" * 201 not in block


def test_build_history_block_caps_to_last_n_turns():
    history = [(f"質問{i}", f"回答{i}") for i in range(1, 6)]  # 5往復
    block = _build_history_block(history)
    assert "質問1" not in block  # 直近3往復だけが残る
    assert "質問2" not in block
    assert "質問3" in block
    assert "質問5" in block


class _RecordingLLMClient:
    """.chatに渡された(system, user)を記録するだけのフェイク。"""

    def __init__(self, response: str = "回答本文"):
        self.response = response
        self.calls: list[tuple[str, str]] = []

    def chat(self, system: str, user: str, **kwargs) -> str:
        self.calls.append((system, user))
        return self.response


def test_build_prompt_places_history_before_question():
    chunks = [_chunk("RPT-001", "教訓・つまずいたポイント", "マーケティング部", text="本文")]
    _, user = _build_prompt("それについてもう少し詳しく", chunks, [("元の質問", "元の回答")])
    assert user.index("Q: 元の質問") < user.index("# 質問")


def test_answer_question_passes_history_into_prompt():
    client = _RecordingLLMClient()
    chunks = [_chunk("RPT-001", "教訓・つまずいたポイント", "マーケティング部", text="本文")]
    answer_question(client, "それについてもう少し詳しく", chunks, history=[("元の質問", "元の回答")])
    assert len(client.calls) == 1
    _, user = client.calls[0]
    assert "元の質問" in user
    assert "元の回答" in user


def test_answer_question_without_history_omits_history_block():
    client = _RecordingLLMClient()
    chunks = [_chunk("RPT-001", "教訓・つまずいたポイント", "マーケティング部", text="本文")]
    answer_question(client, "質問", chunks)
    _, user = client.calls[0]
    assert "これまでの会話" not in user
