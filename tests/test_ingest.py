"""ingest（DAGノードB）の純粋ロジックのテスト。ChromaDB・LLMは使わない。"""

from __future__ import annotations

import pytest

from cae_rag.ingest import build_chunks, parse_frontmatter, split_into_sections

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
