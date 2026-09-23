"""ingest（DAGノードB）の純粋ロジックのテスト。ChromaDB・LLMは使わない。"""

from __future__ import annotations

import pytest

from fem_rag.ingest import (
    RESULT_IMAGE_SECTION,
    _caption_image,
    _parse_pdf_text,
    build_chunks,
    parse_frontmatter,
    split_into_sections,
)
from fem_rag.llm_client import LLMConnectionError

SAMPLE_REPORT = """---
report_id: RPT-001
dept: マーケティング部
project_type: 新規事業立ち上げ
subject: 新商品ローンチキャンペーン
method: アジャイル（スクラム）
resourcing: 既存メンバーのみで対応
author: 担当者A
date: 2023-01-01
---
# 新商品ローンチキャンペーン 新規事業立ち上げ 振り返りレポート（RPT-001）

## プロジェクト目的
新商品の認知拡大と初動売上の確保を目的とした振り返り。

## 対象領域・テーマ
新商品ローンチキャンペーン。マーケティング部主管の施策。

## 推進体制
週次スクラムで進行、関係部署は都度共有会で連携。
"""


def test_parse_frontmatter_roundtrip():
    meta, body = parse_frontmatter(SAMPLE_REPORT)
    assert meta["report_id"] == "RPT-001"
    assert meta["dept"] == "マーケティング部"
    assert meta["project_type"] == "新規事業立ち上げ"
    assert body.startswith("# 新商品ローンチキャンペーン")


def test_parse_frontmatter_missing_raises():
    with pytest.raises(ValueError):
        parse_frontmatter("見出しだけで始まる本文\n本文本文")


def test_split_into_sections_basic():
    _, body = parse_frontmatter(SAMPLE_REPORT)
    sections = split_into_sections(body)
    headings = [h for h, _ in sections]
    assert headings == ["プロジェクト目的", "対象領域・テーマ", "推進体制"]
    assert "認知拡大" in dict(sections)["プロジェクト目的"]


def test_split_into_sections_ignores_text_before_first_heading():
    body = "タイトル行など見出し前のテキスト\n\n## セクションA\n本文A"
    sections = split_into_sections(body)
    assert len(sections) == 1
    assert sections[0] == ("セクションA", "本文A")


def test_split_into_sections_empty_when_no_headings():
    assert split_into_sections("見出しが一つもない本文だけ") == []


def test_build_chunks_ids_and_metadata(tmp_path):
    report_path = tmp_path / "RPT-001.md"
    report_path.write_text(SAMPLE_REPORT, encoding="utf-8")

    chunks = build_chunks(report_path)

    assert len(chunks) == 3
    ids = [c.chunk_id for c in chunks]
    assert ids == [
        "RPT-001::プロジェクト目的",
        "RPT-001::対象領域・テーマ",
        "RPT-001::推進体制",
    ]
    # 各チャンクのメタデータには、フロントマターの値 + section が乗っている
    for c in chunks:
        assert c.metadata["report_id"] == "RPT-001"
        assert c.metadata["dept"] == "マーケティング部"
        assert c.metadata["section"] in {"プロジェクト目的", "対象領域・テーマ", "推進体制"}
        assert c.text.startswith(f"【{c.metadata['section']}】")


SAMPLE_REPORT_WITH_IMAGE = f"""---
report_id: RPT-002
dept: マーケティング部
project_type: 新規事業立ち上げ
subject: 新商品ローンチキャンペーン
method: アジャイル（スクラム）
resourcing: 既存メンバーのみで対応
author: 担当者A
date: 2023-01-01
---
# タイトル

## プロジェクト目的
本文。

## {RESULT_IMAGE_SECTION}
成果の説明文。

![成果画像](RPT-002.png)
"""


class _FakeVLMClient:
    """describe_imageだけを持つフェイク。実際のLM Studio接続は使わない。"""

    def __init__(self, caption: str | None = None, raise_error: bool = False):
        self._caption = caption
        self._raise_error = raise_error
        self.calls = 0

    def describe_image(self, image: bytes, prompt: str) -> str:
        self.calls += 1
        if self._raise_error:
            raise LLMConnectionError("VLM未ロード")
        assert self._caption is not None
        return self._caption


def test_build_chunks_reads_sibling_image_and_strips_markdown_syntax(tmp_path):
    report_path = tmp_path / "RPT-002.md"
    report_path.write_text(SAMPLE_REPORT_WITH_IMAGE, encoding="utf-8")
    (tmp_path / "RPT-002.png").write_bytes(b"fake-png-bytes")

    chunks = build_chunks(report_path)  # client=None -> キャプション化しない

    result_chunk = next(c for c in chunks if c.metadata["section"] == RESULT_IMAGE_SECTION)
    assert "[結果画像の説明]" not in result_chunk.text
    assert "![成果画像]" not in result_chunk.text  # 画像記法は本文チャンクから取り除かれる
    assert "成果の説明文。" in result_chunk.text


def test_build_chunks_merges_vlm_caption_into_result_section(tmp_path):
    report_path = tmp_path / "RPT-002.md"
    report_path.write_text(SAMPLE_REPORT_WITH_IMAGE, encoding="utf-8")
    (tmp_path / "RPT-002.png").write_bytes(b"fake-png-bytes")

    client = _FakeVLMClient(caption="施策後にKPIが向上している。")
    chunks = build_chunks(report_path, client)

    result_chunk = next(c for c in chunks if c.metadata["section"] == RESULT_IMAGE_SECTION)
    assert client.calls == 1
    assert "[結果画像の説明] 施策後にKPIが向上している。" in result_chunk.text


def test_build_chunks_skips_captioning_when_no_image_present(tmp_path):
    report_path = tmp_path / "RPT-001.md"
    report_path.write_text(SAMPLE_REPORT, encoding="utf-8")

    client = _FakeVLMClient(caption="呼ばれないはず")
    build_chunks(report_path, client)

    assert client.calls == 0  # 画像が無いレポートではVLMを呼ばない


def test_caption_image_returns_none_on_llm_connection_error():
    client = _FakeVLMClient(raise_error=True)
    assert _caption_image(client, b"data") is None


def test_parse_pdf_text_basic():
    text = "report_id: RPT-041\ndept: 経営企画部\n\n## プロジェクト目的\n本文A\n\n## 成果サマリー\n本文B\n"
    meta, sections = _parse_pdf_text(text)
    assert meta == {"report_id": "RPT-041", "dept": "経営企画部"}
    assert sections == [("プロジェクト目的", "本文A"), ("成果サマリー", "本文B")]


def test_parse_pdf_text_does_not_depend_on_blank_line_separator():
    """退行テスト: 実際にreportlabで生成したPDFをpypdfで抽出すると、drawStringを
    呼ばなかった空行がテキストに残らず、メタデータとセクションの区切りの空行が
    消えることがある（60件中9件のPDFでゼロセクションになる実例が見つかった）。
    空行の有無に依存しない実装になっていることを確認する。"""
    text = "report_id: RPT-041\ndept: 経営企画部\n## プロジェクト目的\n本文A\n## 成果サマリー\n本文B\n"
    meta, sections = _parse_pdf_text(text)
    assert meta == {"report_id": "RPT-041", "dept": "経営企画部"}
    assert sections == [("プロジェクト目的", "本文A"), ("成果サマリー", "本文B")]


def test_parse_pdf_text_empty_metadata_line_is_harmless():
    # メタデータ部分に複数の空行が挟まっても問題なく無視される。
    text = "report_id: RPT-001\n\n\ndept: マーケティング部\n## プロジェクト目的\n本文\n"
    meta, sections = _parse_pdf_text(text)
    assert meta == {"report_id": "RPT-001", "dept": "マーケティング部"}
    assert sections == [("プロジェクト目的", "本文")]
