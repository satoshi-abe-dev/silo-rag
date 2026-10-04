"""Node H: LangChain integration (optional).

1. `SiloRetriever` exposes the hybrid search (retrieval.search(): BM25 + vectors + LLM rerank)
   as a LangChain `BaseRetriever`, so other LangChain chains and agents can reuse it.
2. `run_langchain_agent` hands that retriever as a tool to LangChain's stock `create_agent`.
   It is a baseline for the LangGraph agent (node G): G avoids tool calling by having the LLM
   answer in fixed one-word verdicts, while this relies on function calling, so it tests
   whether small local models can handle that.

Like G, it only uses the public search()/answer_question() API of nodes C/D.
All LLM calls (LLMClient inside search, ChatOpenAI for the agent) go to the local server at
config.ai.base_url; nothing leaves the machine unless LangSmith env vars are set.

Note: tool calling only works with LM Studio models that support tools (tool icon in the
model list); other models make the API return an error.
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
# Default search cap, independent of config.agent.max_attempts: even if node G defaults to one
# search, the stock agent keeps choosing its own search count, matching the measured setup.
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

# Tool output format; the header carries report_id and dept so the answer can cite them.
_DOCUMENT_PROMPT = PromptTemplate.from_template(
    "[出典 report_id={report_id} / 部署={dept} / セクション={section}]\n{page_content}"
)


class _ModelCallCounter(BaseCallbackHandler):
    """Count the agent's own chat model calls directly.

    Inferring from message or search counts drifted on invalid tool-call retries and
    cap cutoffs (caught in codex review).
    """

    def __init__(self) -> None:
        self.count = 0

    def on_chat_model_start(self, serialized: Any, messages: Any, **kwargs: Any) -> None:
        self.count += 1


class SiloRetriever(BaseRetriever):
    """Expose retrieval.search() as a LangChain Retriever.

    `retrieved` and the counters record every call so eval and the UI can see what the agent read.

    Searches run one at a time under a lock: LangGraph runs parallel tool calls in threads, but
    search() is single-threaded and concurrent calls broke both ChromaDB (same directory opened
    twice) and LM Studio (concurrent embeddings -> HTTP 500), reproduced on QA-005.

    max_calls caps executed searches; extra calls return no results but count in search_requests.
    recursion_limit can't do this, since it doesn't cover parallel calls within one reply
    (a 7B model tried ~50 at once on QA-005).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: Any  # LLMClient used inside search() (rerank, query rewrite, embeddings)
    filters: dict[str, str] | None = None
    top_k: int | None = None
    max_calls: int | None = None
    retrieved: list[ScoredChunk] = Field(default_factory=list)
    search_calls: int = 0  # searches actually run
    search_requests: int = 0  # searches requested, including ones skipped over the cap
    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    def _get_relevant_documents(self, query: str, *, run_manager: Any = None) -> list[Document]:
        with self._lock:
            self.search_requests += 1
            if self.max_calls is not None and self.search_calls >= self.max_calls:
                return []
            # The agent's LLM already wrote this query, so don't rewrite it again.
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
            # Defaults for keys _DOCUMENT_PROMPT needs, so missing metadata doesn't crash it.
            "report_id": meta.get("report_id", "不明"),
            "dept": meta.get("dept", "不明"),
            "section": meta.get("section", "不明"),
            "chunk_id": sc.chunk.chunk_id,
            "score": sc.score,
        },
    )


def build_retriever_tool(retriever: SiloRetriever):
    """Wrap the retriever as a tool the LLM can call."""
    return create_retriever_tool(
        retriever,
        TOOL_NAME,
        _TOOL_DESCRIPTION,
        document_prompt=_DOCUMENT_PROMPT,
    )


@dataclass
class LangChainAgentResult:
    answer: Answer
    scored_chunks: list[ScoredChunk]  # chunks returned by the tool, in call order, deduplicated
    searches: int  # searches actually run (<= max_searches)
    requested_searches: int  # includes searches skipped over the cap
    llm_calls: int  # agent model calls only (not reranking etc. inside search)


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
    # Some models return a list of content blocks; join their text.
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
    """Answer one question with LangChain's stock create_agent.

    client: LLMClient for the LLM work inside search (rerank, embeddings).
    top_k: chunks per search; None uses config.retrieval.top_k_final.
    max_searches: search cap; None uses DEFAULT_MAX_SEARCHES (3).
    model: agent LLM; None builds a ChatOpenAI for config.ai's local server (tests inject a fake).
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
        # Each search takes two steps (model -> tool) plus one for the final answer; leave headroom
        # so a run that uses every search still reaches its answer. The retriever caps searches;
        # this limit is only a safety net for loops that keep calling over the cap or a wrong tool.
        # call-overload: mypy can't resolve create_agent's input type overloads.
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
