"""Check that Mermaid diagrams in the README and docs don't get clipped on GitHub.

Mermaid can't wrap unspaced Japanese, so a label line wider than ~200px (about 13 full-width
chars) is clipped at the node edge. Each <br/>-separated label line must fit 12 full-width chars,
and every diagram must start with the init directive (padding and wrapping width).
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
README_FILES = ["README_ja.md", "README_en.md"]
# Include docs with diagrams too, so moving a diagram out of the README keeps it checked.
DIAGRAM_FILES = README_FILES + sorted(
    f"docs/{p.name}"
    for p in (REPO_ROOT / "docs").glob("*.md")
    if "```mermaid" in p.read_text(encoding="utf-8")
)
MAX_LINE_WIDTH = 24  # 12 full-width chars (full-width = 2, half-width = 1)

# Node labels: ID["…"] / ID[…] / ID((…)). Subgraph titles and edge labels are excluded.
_NODE_LABEL_RE = re.compile(r'\["([^"]+)"\]|\[([^\]\["]+)\]|\(\(([^()]+)\)\)')


def _display_width(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def _mermaid_blocks(path: Path) -> list[str]:
    return re.findall(r"```mermaid\n(.*?)```", path.read_text(encoding="utf-8"), re.S)


@pytest.mark.parametrize("name", README_FILES)
def test_readme_has_diagrams(name):
    blocks = _mermaid_blocks(REPO_ROOT / name)
    assert blocks, f"{name} にMermaidの図が見つからない（このテストが空振りしている）"


@pytest.mark.parametrize("name", DIAGRAM_FILES)
def test_every_diagram_starts_with_the_init_directive(name):
    for i, block in enumerate(_mermaid_blocks(REPO_ROOT / name), 1):
        assert block.startswith("%%{init:"), (
            f"{name} の{i}番目の図が、余白・折り返し幅の設定（%%{{init: …}}%%）で始まっていない"
        )


@pytest.mark.parametrize("name", DIAGRAM_FILES)
def test_no_node_label_line_is_too_wide(name):
    too_wide: list[str] = []
    for i, block in enumerate(_mermaid_blocks(REPO_ROOT / name), 1):
        for line in block.splitlines():
            if line.lstrip().startswith("subgraph"):
                continue
            for match in _NODE_LABEL_RE.finditer(line):
                label = next(group for group in match.groups() if group)
                for part in re.split(r"<br\s*/?>", label):
                    width = _display_width(part)
                    if width > MAX_LINE_WIDTH:
                        too_wide.append(f"{name} の{i}番目の図: 幅{width} 「{part}」")
    assert not too_wide, (
        "文字切れの恐れがある、長すぎるラベルの行がある（<br/>で短く分けること）:\n" + "\n".join(too_wide)
    )


def test_the_width_check_catches_the_labels_that_were_clipped_before():
    # Labels that were actually clipped (node G flow diagram) must be flagged.
    assert _display_width("select: 複数回検索したときだけ、集めた候補を元の質問で並べ直す") > MAX_LINE_WIDTH
    assert _display_width("rewrite: 別の切り口でクエリを作る") > MAX_LINE_WIDTH
    assert _display_width("（Dの answer_question）") <= MAX_LINE_WIDTH
