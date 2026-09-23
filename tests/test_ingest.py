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
dept: ボディ設計部
analysis_type: 静解析（線形）
part: フロントドアパネル
solver: Abaqus
material: 高張力鋼板(980MPa級)
author: 担当者A
date: 2023-01-01
---
# フロントドアパネル 静解析（線形） 解析レポート（RPT-001）

## 解析目的
衝突安全性能を確認するための静解析。

## 対象部品・製品カテゴリ
フロントドアパネル。ボディ系部品。

## メッシュ設定
四角形シェル要素、平均寸法5mm。
"""


def test_parse_frontmatter_roundtrip():
    meta, body = parse_frontmatter(SAMPLE_REPORT)
    assert meta["report_id"] == "RPT-001"
    assert meta["dept"] == "ボディ設計部"
    assert meta["analysis_type"] == "静解析（線形）"
    assert body.startswith("# フロントドアパネル")


def test_parse_frontmatter_missing_raises():
    with pytest.raises(ValueError):
        parse_frontmatter("見出しだけで始まる本文\n本文本文")


def test_split_into_sections_basic():
    _, body = parse_frontmatter(SAMPLE_REPORT)
    sections = split_into_sections(body)
    headings = [h for h, _ in sections]
    assert headings == ["解析目的", "対象部品・製品カテゴリ", "メッシュ設定"]
    assert "衝突安全性能" in dict(sections)["解析目的"]


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
        "RPT-001::解析目的",
        "RPT-001::対象部品・製品カテゴリ",
        "RPT-001::メッシュ設定",
    ]
    # 各チャンクのメタデータには、フロントマターの値 + section が乗っている
    for c in chunks:
        assert c.metadata["report_id"] == "RPT-001"
        assert c.metadata["dept"] == "ボディ設計部"
        assert c.metadata["section"] in {"解析目的", "対象部品・製品カテゴリ", "メッシュ設定"}
        assert c.text.startswith(f"【{c.metadata['section']}】")


SAMPLE_REPORT_WITH_IMAGE = f"""---
report_id: RPT-002
dept: ボディ設計部
analysis_type: 静解析（線形）
part: フロントドアパネル
solver: Abaqus
material: 高張力鋼板(980MPa級)
author: 担当者A
date: 2023-01-01
---
# タイトル

## 解析目的
本文。

## {RESULT_IMAGE_SECTION}
結果の説明文。

![結果画像](RPT-002.png)
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
    assert "![結果画像]" not in result_chunk.text  # 画像記法は本文チャンクから取り除かれる
    assert "結果の説明文。" in result_chunk.text


def test_build_chunks_merges_vlm_caption_into_result_section(tmp_path):
    report_path = tmp_path / "RPT-002.md"
    report_path.write_text(SAMPLE_REPORT_WITH_IMAGE, encoding="utf-8")
    (tmp_path / "RPT-002.png").write_bytes(b"fake-png-bytes")

    client = _FakeVLMClient(caption="応力が端部に集中している。")
    chunks = build_chunks(report_path, client)

    result_chunk = next(c for c in chunks if c.metadata["section"] == RESULT_IMAGE_SECTION)
    assert client.calls == 1
    assert "[結果画像の説明] 応力が端部に集中している。" in result_chunk.text


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
    text = "report_id: RPT-041\ndept: パワートレイン設計部\n\n## 解析目的\n本文A\n\n## 結果サマリー\n本文B\n"
    meta, sections = _parse_pdf_text(text)
    assert meta == {"report_id": "RPT-041", "dept": "パワートレイン設計部"}
    assert sections == [("解析目的", "本文A"), ("結果サマリー", "本文B")]


def test_parse_pdf_text_does_not_depend_on_blank_line_separator():
    """退行テスト: 実際にreportlabで生成したPDFをpypdfで抽出すると、drawStringを
    呼ばなかった空行がテキストに残らず、メタデータとセクションの区切りの空行が
    消えることがある（60件中9件のPDFでゼロセクションになる実例が見つかった）。
    空行の有無に依存しない実装になっていることを確認する。"""
    text = "report_id: RPT-041\ndept: パワートレイン設計部\n## 解析目的\n本文A\n## 結果サマリー\n本文B\n"
    meta, sections = _parse_pdf_text(text)
    assert meta == {"report_id": "RPT-041", "dept": "パワートレイン設計部"}
    assert sections == [("解析目的", "本文A"), ("結果サマリー", "本文B")]


def test_parse_pdf_text_empty_metadata_line_is_harmless():
    # メタデータ部分に複数の空行が挟まっても問題なく無視される。
    text = "report_id: RPT-001\n\n\ndept: ボディ設計部\n## 解析目的\n本文\n"
    meta, sections = _parse_pdf_text(text)
    assert meta == {"report_id": "RPT-001", "dept": "ボディ設計部"}
    assert sections == [("解析目的", "本文")]
