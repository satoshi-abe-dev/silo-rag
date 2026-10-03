"""LangChain連携（DAGノードH、オプション）。

1. `SiloRetriever`: 自前のハイブリッド検索（retrieval.search()）を、LangChainのRetriever
   （`langchain_core.retrievers.BaseRetriever`）の規格に合わせて公開する。検索の中身（BM25＋ベクトル＋
   LLMリランキング）はそのままで、LangChainで作られた他のチェーンやエージェントから部品として使える。
2. `run_langchain_agent`: 上のRetrieverを検索ツールとして、LangChain既製の`create_agent`
   （LLMがツール呼び出しで検索を行う）に渡して回答する。自作のLangGraphエージェント（agent.py、ノードG）との
   比較用。Gは「判定は決まった形式の1語」という設計でツール呼び出しを避けているのに対し、こちらはLLMの
   ツール呼び出し（function calling）に頼る。小さなローカルモデルでそれが成り立つかを測るための比較対象。

search()とanswer_question()の公開関数だけを使い、C・Dの中身には手を入れない（Gと同じ形）。
LLM呼び出しは、検索内のリランキングなどは既存のLLMClient、エージェント自身の判断はChatOpenAI経由で、
どちらもローカルのLM Studio等（config.ai.base_url）に向かう。外部送信ゼロの方針は変わらない
（LangSmith用の環境変数を設定しない限り、LangChainも外部へ何も送信しない）。

注意: 既製エージェントが使うツール呼び出しは、LM Studioでツール対応のモデル（モデル一覧に
ツールのアイコンが付くもの）でしか動かない。未対応のモデルを指定するとAPIがエラーを返す。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

from langchain.agents import create_agent
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import create_retriever_tool
from langgraph.errors import GraphRecursionError
from pydantic import ConfigDict, Field, PrivateAttr

from .config import load_config
from .generation import Answer, build_citations
from .retrieval import ScoredChunk, search

TOOL_NAME = "search_reports"
# 既製エージェントの検索の上限回数（既定）。自作エージェントの既定（config.agent.max_attempts）とは独立にして
# いる。自作側の既定を1（再検索しない）にしても、既製エージェントには複数回の検索を許す
# （測定した条件と、LLMが自分で検索回数を決めるという既製エージェントの前提を保つため）。
DEFAULT_MAX_SEARCHES = 3
_MAX_HISTORY_TURNS = 3

_NO_ANSWER = "検索の上限回数内に回答に至りませんでした。"

_SYSTEM_PROMPT = (
    "あなたは部署横断のプロジェクト知見・教訓に関する社内ナレッジ検索アシスタントです。"
    "以下の方針を厳守してください。\n"
    f"- 質問に答える前に、必ず{TOOL_NAME}ツールで検索してください。"
    "結果が足りなければ、クエリを変えて複数回検索して構いません。\n"
    "- 回答は必ず、検索結果に書かれている内容のみに基づいて作成してください。"
    "書かれていない事実を推測・創作しないでください。\n"
    "- 質問と全く同じ案件やプロジェクト種別である必要はありません。"
    "テーマや失敗パターンが似ている他部署の関連事例は、他部署の事例である旨を明記した上で、"
    "参考情報として回答に含めてください。\n"
    "- 根拠にした出典は、「（RPT-014, マーケティング部の事例）」のように、report_idと部署名を"
    "本文中に引用してください。\n"
    "- 検索結果が質問と本当に無関係な場合は、「該当する事例が見つかりませんでした。」と答えてください。\n"
    "- 実在・架空を問わず、企業名やブランド名は一切書かないでください。\n"
    "- 日本語で、簡潔かつ具体的に（実施条件・つまずいたポイント・教訓など実務に役立つ点を中心に）"
    "回答してください。"
)

_TOOL_DESCRIPTION = (
    "部署横断のプロジェクト知見・教訓レポートを検索する。引数queryに、探したい内容を表す"
    "検索クエリ（日本語の文）を渡す。出典のreport_id・部署・セクション付きで、関連度の高い順に結果が返る。"
)

# ツールの出力（LLMに見せるテキスト）の整形。回答で出典を引用できるよう、ヘッダーにreport_id・部署を入れる。
_DOCUMENT_PROMPT = PromptTemplate.from_template(
    "[出典 report_id={report_id} / 部署={dept} / セクション={section}]\n{page_content}"
)


class _ModelCallCounter(BaseCallbackHandler):
    """エージェント自身のLLM（チャットモデル）の呼び出し回数を、実際に数える。

    メッセージ数や検索回数からの推測だと、無効なツール呼び出しの繰り返しや、上限での打ち切りの途中で
    実際の回数とずれる（codexレビューで指摘され、修正した）。
    """

    def __init__(self) -> None:
        self.count = 0

    def on_chat_model_start(self, serialized: Any, messages: Any, **kwargs: Any) -> None:
        self.count += 1


class SiloRetriever(BaseRetriever):
    """自前のハイブリッド検索（retrieval.search()）を、LangChainのRetrieverとして公開する。

    `retrieved`・`search_calls`には、呼ばれた履歴を記録する（エージェントがどのチャンクを
    見たか、何回検索したかを、評価やUIで後から調べるため）。

    検索は1つずつ順番に実行する。create_agentのLLMは、1回の応答で検索ツールを複数同時に呼ぶこと
    （並列ツール呼び出し）があり、LangGraphはそれを別スレッドで同時に実行する。しかしsearch()は
    1スレッドでの利用を前提にしていて、同時に走らせるとChromaDB（同じフォルダの同時オープン）と
    LM Studio（埋め込みの同時リクエスト → HTTP 500）の両方で失敗する（実データのQA-005で再現）。

    max_callsを指定すると、検索を実行する回数をその回数までに制限する。上限を超えた呼び出しは
    検索せずに空の結果を返す（search_requestsには数える）。LangGraphのステップ数の上限
    （recursion_limit）は「モデルが何回考え直すか」しか制限できず、1回の応答の中の並列ツール呼び出しには
    効かない。実際、7Bモデルが1回の応答で検索ツールを約50回同時に呼ぼうとした（QA-005）。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: Any  # LLMClient（search()内のリランキング・クエリ書き換え・埋め込みに使う）
    filters: dict[str, str] | None = None
    top_k: int | None = None
    max_calls: int | None = None
    retrieved: list[ScoredChunk] = Field(default_factory=list)
    search_calls: int = 0  # 実際に検索した回数
    search_requests: int = 0  # 検索を要求された回数（上限超過で実行しなかった分を含む）
    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    def _get_relevant_documents(self, query: str, *, run_manager: Any = None) -> list[Document]:
        with self._lock:
            self.search_requests += 1
            if self.max_calls is not None and self.search_calls >= self.max_calls:
                return []
            # ツールに渡るクエリは、エージェントのLLMがすでに作ったものなので、検索側で書き直さない。
            scored = search(self.client, query, top_k=self.top_k, filters=self.filters, rewrite_query=False)
            self.search_calls += 1
            self.retrieved.extend(scored)
        return [_to_document(sc) for sc in scored]


def _to_document(sc: ScoredChunk) -> Document:
    meta = sc.chunk.metadata
    return Document(
        page_content=sc.chunk.text,
        metadata={
            **meta,
            # ツール出力のテンプレートが参照するキーは、欠けていても落ちないよう既定値を入れておく。
            "report_id": meta.get("report_id", "不明"),
            "dept": meta.get("dept", "不明"),
            "section": meta.get("section", "不明"),
            "chunk_id": sc.chunk.chunk_id,
            "score": sc.score,
        },
    )


def build_retriever_tool(retriever: SiloRetriever):
    """RetrieverをLangChainの「検索ツール」にする（LLMがツール呼び出しで使えるようにする）。"""
    return create_retriever_tool(
        retriever,
        TOOL_NAME,
        _TOOL_DESCRIPTION,
        document_prompt=_DOCUMENT_PROMPT,
    )


@dataclass
class LangChainAgentResult:
    answer: Answer
    scored_chunks: list[ScoredChunk]  # 検索ツールが返したチャンク（呼ばれた順・重複排除）
    searches: int  # 実際に検索した回数（上限max_searches以下）
    requested_searches: int  # エージェントが検索を要求した回数（上限超過で実行しなかった分を含む）
    llm_calls: int  # エージェント自身の判断のLLM呼び出し回数（検索内のリランキング等は含まない）


def _build_chat_model() -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    ai = load_config().ai
    return ChatOpenAI(
        model=ai.llm_model,
        base_url=ai.base_url,
        api_key=ai.api_key,  # type: ignore[arg-type]
        temperature=0.2,
        max_completion_tokens=ai.max_tokens,
        timeout=ai.timeout,
    )


def _message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content.strip()
    # 一部のモデルは、内容をブロックのリストで返す。テキストのブロックだけをつなぐ。
    parts = [b.get("text", "") if isinstance(b, dict) else str(b) for b in content]
    return "".join(parts).strip()


def _dedup_chunks(scored: list[ScoredChunk]) -> list[ScoredChunk]:
    seen: set[str] = set()
    unique: list[ScoredChunk] = []
    for sc in scored:
        if sc.chunk.chunk_id in seen:
            continue
        seen.add(sc.chunk.chunk_id)
        unique.append(sc)
    return unique


def run_langchain_agent(
    client: Any,
    question: str,
    *,
    filters: dict[str, str] | None = None,
    history: list[tuple[str, str]] | None = None,
    top_k: int | None = None,
    max_searches: int | None = None,
    model: BaseChatModel | None = None,
) -> LangChainAgentResult:
    """LangChain既製のエージェント（create_agent）で1問ぶん回答する。

    client: 検索内のLLM処理（リランキング・埋め込み）に使うLLMClient。
    top_k: 1回の検索で返すチャンク数。Noneならconfig.retrieval.top_k_final。
    max_searches: 検索の上限回数。NoneならDEFAULT_MAX_SEARCHES（3回）。
    model: エージェントのLLM。Noneならconfig.aiのローカルサーバーに向けたChatOpenAIを作る
        （テストでフェイクを差し込むための引数）。
    """
    resolved_max = max(1, max_searches if max_searches is not None else DEFAULT_MAX_SEARCHES)
    retriever = SiloRetriever(client=client, filters=filters, top_k=top_k, max_calls=resolved_max)
    agent = create_agent(
        model or _build_chat_model(),
        tools=[build_retriever_tool(retriever)],
        system_prompt=_SYSTEM_PROMPT,
    )

    messages: list[AnyMessage] = []
    for q, a in (history or [])[-_MAX_HISTORY_TURNS:]:
        messages.extend([HumanMessage(q), AIMessage(a)])
    messages.append(HumanMessage(question))

    counter = _ModelCallCounter()
    try:
        # 検索1回につき「モデル→ツール」の2ステップ、最後にモデルの回答が1ステップで、検索が上限ちょうどまで
        # 続いても回答まで届くよう、ステップ数の上限には余裕を持たせる（ぎりぎりの値にすると、有効な回答が
        # 出ているのに打ち切ってしまう）。検索の回数そのものはRetrieverの上限で止める。この上限は、
        # 上限を超えた検索を要求し続ける（またはツール名を間違え続ける）ループを止めるための安全装置。
        # 下の型チェック警告（call-overload）は、create_agentの入力型が
        # mypyのoverload解決に通らないため抑えている。
        result = agent.invoke(  # type: ignore[call-overload]
            {"messages": messages},
            config=RunnableConfig(recursion_limit=2 * resolved_max + 3, callbacks=[counter]),
        )
    except GraphRecursionError:
        chunks = _dedup_chunks(retriever.retrieved)
        return LangChainAgentResult(
            answer=Answer(text=_NO_ANSWER, citations=build_citations([sc.chunk for sc in chunks])),
            scored_chunks=chunks,
            searches=retriever.search_calls,
            requested_searches=retriever.search_requests,
            llm_calls=counter.count,
        )

    new_messages = result["messages"][len(messages) :]
    chunks = _dedup_chunks(retriever.retrieved)
    final = next((m for m in reversed(new_messages) if isinstance(m, AIMessage)), None)
    return LangChainAgentResult(
        answer=Answer(
            text=_message_text(final) if final is not None else "",
            citations=build_citations([sc.chunk for sc in chunks]),
        ),
        scored_chunks=chunks,
        searches=retriever.search_calls,
        requested_searches=retriever.search_requests,
        llm_calls=counter.count,
    )
