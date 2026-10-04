"""Keep the README and docs free of condescending, bossy or blunt phrasing.

The docs are written as plain statements of fact. Phrases that tell the reader how to read,
lecture them ("just do X"), or dismiss things bluntly have slipped in before, so they are
listed here and checked on every CI run. Fenced code blocks are skipped.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DOC_FILES = [
    "README_ja.md",
    "README_en.md",
    *sorted(f"docs/{p.name}" for p in (REPO_ROOT / "docs").glob("*.md")),
]

# Phrase -> why it is not allowed. Japanese phrases are matched as substrings.
BANNED_JA = {
    "てほしい": "asks the reader to do something ('read it as ...'); state the fact instead",
    "でほしい": "asks the reader to do something ('read it as ...'); state the fact instead",
    "すればよい": "lecturing ('you just need to'); state the step instead",
    "すれば良い": "lecturing ('you just need to'); state the step instead",
    "当てにならない": "blunt and dismissive; say what may not carry over",
    "言うまでもない": "condescending ('needless to say')",
    "知っての通り": "condescending ('as you know')",
    "当然": "condescending ('of course')",
    "もちろん": "condescending ('of course')",
    "ちゃんと": "casual and patronizing",
    "しっかり": "casual and patronizing",
    "べきだ": "preachy ('you should'); state the reason instead",
    "べきである": "preachy ('you should'); state the reason instead",
}
# English phrases are matched case-insensitively on word boundaries.
BANNED_EN = {
    "do not rely": "blunt and dismissive; say what may not carry over",
    "don't rely": "blunt and dismissive; say what may not carry over",
    "read the numbers": "tells the reader how to read; state the fact instead",
    "read the values": "tells the reader how to read; state the fact instead",
    "read the results": "tells the reader how to read; state the fact instead",
    "keep them apart": "an order to the reader; describe what the text does",
    "obviously": "condescending",
    "simply": "condescending ('simply do X')",
    "of course": "condescending",
    "needless to say": "condescending",
    "as you know": "condescending",
    "you should": "preachy; state the reason instead",
    "just run": "lecturing ('just do X')",
}


def _prose_lines(path: Path) -> list[tuple[int, str]]:
    lines, in_code = [], False
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.lstrip().startswith("```"):
            in_code = not in_code
            continue
        if not in_code:
            lines.append((number, line))
    return lines


@pytest.mark.parametrize("name", DOC_FILES)
def test_docs_avoid_condescending_or_bossy_phrasing(name):
    found: list[str] = []
    for number, line in _prose_lines(REPO_ROOT / name):
        for phrase, why in BANNED_JA.items():
            if phrase in line:
                found.append(f"{name}:{number}: 「{phrase}」 ({why})")
        for phrase, why in BANNED_EN.items():
            if re.search(rf"\b{re.escape(phrase)}\b", line, re.IGNORECASE):
                found.append(f"{name}:{number}: '{phrase}' ({why})")
    assert not found, "Rephrase as plain statements:\n" + "\n".join(found)


def test_the_check_catches_phrases_that_slipped_in_before():
    # Phrases that were actually in the docs and had to be fixed.
    assert "でほしい" in "「傾向」として読んでほしい"
    assert "当てにならない" in "READMEの精度の数字は、当てにならない"
    assert re.search(r"\bdo not rely\b", "Do not rely on the accuracy figures", re.IGNORECASE)
    assert re.search(r"\bread the numbers\b", 'Read the numbers as a "tendency"', re.IGNORECASE)
