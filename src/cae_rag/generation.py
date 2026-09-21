"""引用付き回答生成（DAGノードD）。

質問と、すでに検索済みのコンテキストチャンク（`retrieval.py` が返すもの）を受け取り、
ローカルLLM（LM Studio等、OpenAI互換API）でチャンクの内容だけに基づいた回答を生成する。

このモジュール自身は検索を一切行わない（retrieval.pyへは依存しない）。検索と生成を
分離することで、retrieval.pyとgeneration.pyを独立して実装・テストできるようにしている
（プラン「開発ワークフロー」節のDAG並列実装方針を参照）。

本プロジェクトの価値提案の核は「部署間で情報共有が統一されていないなかで、他部署の
過去事例を横断的に見つけられること」（プランContext節参照）。そのため回答生成では、
各引用元がどの部署の事例かを本文中に明示させ、読み手が「自部署の事例か・他部署の事例か」
を一目で区別できるようにする。

社名は実在・架空を問わず一切出さない（プラン全体の前提）。
"""

from __future__ import annotations

from dataclasses import dataclass

from .ingest import Chunk
from .llm_client import LLMClient

_NO_CONTEXT_ANSWER = "該当する事例が見つかりませんでした。"


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
    """1チャンク分をLLMへの提示用テキストに整形する。

    出典を混同されないよう、report_id/部署/部品/解析種別/セクションをヘッダーとして
    チャンク本文の前に明示し、チャンクごとに区切り線で区切る。
    """
    meta = chunk.metadata
    header = (
        f"[出典{index}] report_id={meta.get('report_id', '不明')} / "
        f"部署={meta.get('dept', '不明')} / "
        f"部品={meta.get('part', '不明')} / "
        f"解析種別={meta.get('analysis_type', '不明')} / "
        f"セクション={meta.get('section', '不明')}"
    )
    return f"{header}\n{chunk.text}"


def _build_prompt(question: str, chunks: list[Chunk]) -> tuple[str, str]:
    system = (
        "あなたは自動車部品の構造解析(FEM/CAE)に関する社内ナレッジ検索アシスタントです。"
        "以下の方針を厳守してください。\n"
        "- 回答は必ず、与えられた「出典」コンテキストに書かれている内容のみに基づいて作成してください。"
        "コンテキストに書かれていない事実を推測・創作しないでください。\n"
        "- コンテキストだけでは答えられない場合は、素直に「分かりません」「該当する事例が見つかりません」"
        "と答えてください。\n"
        "- 回答文中で根拠にした出典は、対応するreport_idを使って"
        "「（RPT-014より）」のように本文中に引用してください。\n"
        "- 質問文に質問者自身の所属部署が明記されている場合、その部署と異なる部署の出典を"
        "引用するときは、必ず部署名を添えて「（RPT-014, ボディ設計部の事例）」のように示し、"
        "自部署の事例ではなく他部署の事例であることが読み手に分かるようにしてください。"
        "質問者の部署が不明な場合でも、引用ごとに出典の部署名を添えると親切です。\n"
        "- 実在・架空を問わず、企業名やブランド名は一切書かないでください。\n"
        "- 日本語で、簡潔かつ具体的に（解析条件・トラブル・教訓など実務に役立つ点を中心に）回答してください。"
    )

    context_text = "\n\n---\n\n".join(_build_context_block(i, c) for i, c in enumerate(chunks, start=1))
    user = f"""# 質問
{question}

# 出典コンテキスト（この内容のみを根拠にしてください）
{context_text}

上記の出典コンテキストだけに基づいて、質問に回答してください。"""
    return system, user


def _build_citations(chunks: list[Chunk]) -> list[Citation]:
    """入力チャンクのメタデータから、決定的に引用リストを組み立てる。

    LLMに引用リストを列挙させるのではなく、ここでチャンクのメタデータから直接構築する
    （LLMの引用表記が不完全でも、citationsフィールドは常に正確であることを保証するため）。

    1つのレポートが複数セクション（＝複数チャンク）から引用されることがあるため、
    report_idだけで丸めてしまうとUI上で「どのセクションが根拠か」が失われる。
    一方で同じ(report_id, section)の組が複数回渡された場合（検索結果の重複等）に
    同じCitationを繰り返し表示するのは冗長。そこで (report_id, section) の組で
    重複排除しつつ、異なるsectionはそれぞれ別のCitationとして残す。
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
) -> Answer:
    """質問と検索済みチャンクから、引用付きの回答を生成する。

    chunks は関連度順にすでに検索済みのコンテキスト（retrieval.search()の結果を想定）。
    このモジュールは検索を行わない。chunksが空の場合はLLMを呼ばず、該当事例なしの
    回答を返す（コンテキストなしでLLMに回答させるとハルシネーションの原因になるため）。

    LLMClient.chatが送出するLLMConnectionErrorはここで捕まえず、呼び出し元に伝播させる。
    """
    if not chunks:
        return Answer(text=_NO_CONTEXT_ANSWER, citations=[])

    system, user = _build_prompt(question, chunks)
    text = client.chat(system, user)
    citations = _build_citations(chunks)
    return Answer(text=text, citations=citations)
