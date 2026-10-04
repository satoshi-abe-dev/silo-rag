"""Node D: answer generation with citations.

Answers from already-retrieved chunks only, via a local OpenAI-compatible LLM. Does no retrieval
and doesn't depend on retrieval.py, so both can be built and tested independently.

The core value is surfacing other departments' past cases, so answers name each source's department
to make own-department vs. other-department cases obvious. No company names, real or fictional.
"""

from __future__ import annotations

from dataclasses import dataclass

from .ingest import Chunk
from .llm_client import LLMClient

_NO_CONTEXT_ANSWER = "該当する事例が見つかりませんでした。"

_MAX_HISTORY_TURNS = 3
_HISTORY_ANSWER_TRUNCATE = 200


@dataclass
class Citation:
    report_id: str
    section: str
    dept: str


@dataclass
class Answer:
    text: str
    citations: list[Citation]


def _build_context_block(index: int, chunk: Chunk) -> str:
    """Format one chunk for the prompt, with a metadata header so the LLM doesn't mix up sources."""
    meta = chunk.metadata
    header = (
        f"[出典{index}] report_id={meta.get('report_id', '不明')} / "
        f"部署={meta.get('dept', '不明')} / "
        f"テーマ={meta.get('subject', '不明')} / "
        f"プロジェクト種別={meta.get('project_type', '不明')} / "
        f"セクション={meta.get('section', '不明')}"
    )
    return f"{header}\n{chunk.text}"


def _build_history_block(history: list[tuple[str, str]] | None) -> str:
    """Format recent history as a prompt block; empty string if there is none."""
    if not history:
        return ""
    lines = ["# これまでの会話（指示語の解決にのみ使う。事実の根拠にはしない）"]
    for q, a in history[-_MAX_HISTORY_TURNS:]:
        truncated = a if len(a) <= _HISTORY_ANSWER_TRUNCATE else a[:_HISTORY_ANSWER_TRUNCATE] + "…"
        lines.append(f"Q: {q}")
        lines.append(f"A: {truncated}")
    return "\n".join(lines) + "\n\n"


def _build_prompt(
    question: str, chunks: list[Chunk], history: list[tuple[str, str]] | None = None
) -> tuple[str, str]:
    system = (
        "あなたは部署横断のプロジェクト知見・教訓に関する社内ナレッジ検索アシスタントです。"
        "以下の方針を厳守してください。\n"
        "- 回答は必ず、与えられた「出典」コンテキストに書かれている内容のみに基づいて作成してください。"
        "コンテキストに書かれていない事実を推測・創作しないでください。\n"
        "- 会話履歴が付いている場合、指示語（「それ」「さっきの」等）の意味はそこから解決して"
        "構いません。ただし事実の根拠には使わず、あくまで出典コンテキストだけを根拠にしてください"
        "（履歴中の過去の回答は、LLMが生成した文章であり事実とは限らないため）。\n"
        "- 出典は、質問と全く同じ案件やプロジェクト種別である必要はありません。"
        "テーマや失敗パターンが似ている他部署の関連事例であれば、それが他部署の事例である旨を"
        "明記した上で、積極的に参考情報として回答に含めてください。これが本アシスタントの"
        "核となる価値です。\n"
        "- 「該当する事例が見つかりません」と答えるのは、(a) 出典コンテキストの内容が質問と"
        "本当に無関係な場合、または (b) 質問文自体が部署横断のプロジェクト知見・教訓についての"
        "検索質問になっていない場合（雑談・挨拶・アシスタント自身や会話そのものへの言及など）に"
        "限ってください。\n"
        "- 一方、出典コンテキストに関連する内容はあるものの、質問に確信を持って答えるだけの"
        "情報が無い場合（例:「部署はいくつありますか」「参照している資料は何件ありますか」"
        "「最も多いのはどれですか」のような、個々のレポートの内容ではなく検索結果全体の件数・"
        "網羅性・集計を問う質問）は、「該当する事例が見つかりません」ではなく「分かりません」と"
        "答えてください。その際、本システムは関連度の高いレポートを検索して提示する仕組みであり、"
        "レポート全体の総数や集計値を正確に把握しているわけではない旨を簡潔に添えてください。\n"
        "- 回答文中で根拠にした出典は、対応するreport_idを使って"
        "「（RPT-014より）」のように本文中に引用してください。\n"
        "- 質問文に質問者自身の所属部署が明記されている場合、その部署と異なる部署の出典を"
        "引用するときは、必ず部署名を添えて「（RPT-014, マーケティング部の事例）」のように示し、"
        "自部署の事例ではなく他部署の事例であることが読み手に分かるようにしてください。"
        "質問者の部署が不明な場合でも、引用ごとに出典の部署名を添えると親切です。\n"
        "- 実在・架空を問わず、企業名やブランド名は一切書かないでください。\n"
        "- 日本語で、簡潔かつ具体的に（実施条件・つまずいたポイント・教訓など実務に役立つ点を中心に）"
        "回答してください。"
    )

    context_text = "\n\n---\n\n".join(_build_context_block(i, c) for i, c in enumerate(chunks, start=1))
    history_block = _build_history_block(history)
    user = f"""{history_block}# 質問
{question}

# 出典コンテキスト（この内容のみを根拠にしてください）
{context_text}

上記の出典コンテキストだけに基づいて、質問に回答してください。"""
    return system, user


def build_citations(chunks: list[Chunk]) -> list[Citation]:
    """Build citations deterministically from chunk metadata, deduplicated by (report_id, section).

    Built from metadata rather than parsed from LLM output, so they stay accurate even when the
    LLM's inline citations are sloppy. Deduping by report_id alone would lose which section was cited.
    """
    seen: dict[tuple[str, str], Citation] = {}
    for chunk in chunks:
        meta = chunk.metadata
        report_id = meta.get("report_id", "不明")
        section = meta.get("section", "不明")
        dept = meta.get("dept", "不明")
        key = (report_id, section)
        if key not in seen:
            seen[key] = Citation(report_id=report_id, section=section, dept=dept)
    return list(seen.values())


def answer_question(
    client: LLMClient,
    question: str,
    chunks: list[Chunk],
    history: list[tuple[str, str]] | None = None,
) -> Answer:
    """Generate a cited answer from retrieved chunks (ordered by relevance).

    With no chunks, returns a "nothing found" answer without calling the LLM, to avoid hallucination.
    history: prior (question, answer) pairs, used only to resolve references like "that one",
        never as factual evidence (enforced by the prompt).
    LLMConnectionError from LLMClient.chat propagates to the caller.
    """
    if not chunks:
        return Answer(text=_NO_CONTEXT_ANSWER, citations=[])

    system, user = _build_prompt(question, chunks, history)
    text = client.chat(system, user)
    citations = build_citations(chunks)
    return Answer(text=text, citations=citations)
