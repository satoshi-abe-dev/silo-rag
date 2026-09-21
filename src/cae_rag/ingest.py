"""Ingestion パイプライン（DAGノードB）。

`data/synth_reports/*.md`（datagenノードが生成したダミー解析レポート）を読み込み、
セクション単位でチャンキングし、ローカル埋め込みモデルでベクトル化してChromaDBに格納する。

メタデータ（部署・解析種別・部品・ソルバー・日付等）は、datagenが各レポート冒頭に書き出す
YAML風フロントマター（`---`で囲まれた`key: value`行）から直接読み取る。自由記述文からの
あいまいな抽出は行わない（レポート形式を自分たちで決められるため、構造化ヘッダーの方が
確実で、実務の社内文書でもよくあるパターンでもある）。

チャンキングはMarkdownの `## ` 見出し単位（セクション境界を尊重し、単純な文字数分割はしない）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .config import CHROMA_DIR, COLLECTION_NAME, SYNTH_REPORTS_DIR, load_config
from .llm_client import LLMClient

_FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
_SECTION_RE = re.compile(r"^## +(.+?)\s*$", re.MULTILINE)


@dataclass
class Chunk:
    chunk_id: str
    text: str
    metadata: dict[str, str]


def parse_frontmatter(raw: str) -> tuple[dict[str, str], str]:
    """先頭の `---` フロントマターを辞書として取り出し、残りの本文と一緒に返す。"""
    m = _FRONTMATTER_RE.match(raw)
    if not m:
        raise ValueError("フロントマター（--- ... ---）が見つかりません。datagenで生成したファイルか確認してください。")
    meta: dict[str, str] = {}
    for line in m.group(1).splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    body = raw[m.end() :]
    return meta, body


def split_into_sections(body: str) -> list[tuple[str, str]]:
    """`## 見出し` ごとにMarkdown本文を分割する。見出し前のタイトル行は無視する。"""
    matches = list(_SECTION_RE.finditer(body))
    sections: list[tuple[str, str]] = []
    for i, m in enumerate(matches):
        heading = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        text = body[start:end].strip()
        if text:
            sections.append((heading, text))
    return sections


def build_chunks(report_path: Path) -> list[Chunk]:
    raw = report_path.read_text(encoding="utf-8")
    meta, body = parse_frontmatter(raw)
    report_id = meta.get("report_id", report_path.stem)
    chunks: list[Chunk] = []
    for heading, text in split_into_sections(body):
        chunk_id = f"{report_id}::{heading}"
        chunk_text = f"【{heading}】\n{text}"
        chunk_meta = {**meta, "section": heading}
        chunks.append(Chunk(chunk_id=chunk_id, text=chunk_text, metadata=chunk_meta))
    return chunks


def load_all_chunks(reports_dir: Path = SYNTH_REPORTS_DIR) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(reports_dir.glob("*.md")):
        chunks.extend(build_chunks(path))
    return chunks


def ingest(reports_dir: Path = SYNTH_REPORTS_DIR, chroma_dir: Path = CHROMA_DIR) -> int:
    """レポートを読み込み、埋め込みを計算してChromaDBに格納する。格納したチャンク数を返す。"""
    import chromadb

    chunks = load_all_chunks(reports_dir)
    if not chunks:
        raise SystemExit(f"{reports_dir} にMarkdownレポートが見つかりません。先にdatagenを実行してください。")

    config = load_config()
    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))

    with LLMClient(config.llm) as llm_client:
        if not llm_client.ping():
            raise SystemExit(
                f"LM Studio ({config.llm.base_url}) に接続できません。起動してモデルをロードしてください。"
            )

        # 新しいコレクションを一時名で組み立て、全チャンクの埋め込みが成功してから
        # 旧コレクションと入れ替える。途中で失敗しても既存の検索用インデックスを壊さない。
        # list_collections()の返り値の形（Collectionオブジェクト/名前の文字列）は
        # chromadbのバージョンによって異なるため、存在チェックはせず削除を試みて無視する。
        building_name = f"{COLLECTION_NAME}__building"
        _delete_collection_if_exists(client, building_name)
        building = client.create_collection(building_name)

        try:
            # 大量チャンクでも1リクエストに収まるよう、適度なバッチサイズで埋め込みを取得する。
            batch_size = 32
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i : i + batch_size]
                embeddings = llm_client.embed([c.text for c in batch])
                building.add(
                    ids=[c.chunk_id for c in batch],
                    embeddings=embeddings,
                    documents=[c.text for c in batch],
                    metadatas=[c.metadata for c in batch],
                )
                print(f"embedded {min(i + batch_size, len(chunks))}/{len(chunks)} chunks")
        except Exception:
            _delete_collection_if_exists(client, building_name)
            raise

        _delete_collection_if_exists(client, COLLECTION_NAME)
        building.modify(name=COLLECTION_NAME)

    return len(chunks)


def _delete_collection_if_exists(client, name: str) -> None:
    try:
        client.delete_collection(name)
    except Exception:
        pass


def main() -> None:
    count = ingest()
    print(f"ingest完了: {count} チャンクを {CHROMA_DIR} に格納しました。")


if __name__ == "__main__":
    main()
