"""Node F: Streamlit UI (chat plus cited past cases).

The first layer that combines search() and answer_question(), which are independent in the DAG.
Agent mode uses run_agent() (node G) instead and shows the agent's trace under the answer.

Run: streamlit run src/silo_rag/app.py --server.address localhost
(Without --server.address localhost, other machines on the network can open it.)
`streamlit run` executes this file directly, so use absolute imports (needs `pip install -e .`).
"""

from __future__ import annotations

import importlib.util

import streamlit as st

from silo_rag.config import load_config
from silo_rag.datagen import DEPARTMENTS, PROJECT_TYPES
from silo_rag.generation import Answer, answer_question
from silo_rag.llm_client import LLMClient
from silo_rag.retrieval import ScoredChunk, search

st.set_page_config(page_title="部署横断ナレッジ検索", page_icon="🔧")


@st.cache_resource
def get_client() -> LLMClient:
    config = load_config()
    return LLMClient(config.ai)


def _filter_widgets() -> dict[str, str]:
    """Sidebar filters; unselected keys are omitted (search everything)."""
    st.sidebar.header("絞り込み（任意）")
    dept = st.sidebar.selectbox("作成部署", ["(指定なし・全部署横断)", *sorted(DEPARTMENTS.keys())])
    project_type = st.sidebar.selectbox("プロジェクト種別", ["(指定なし)", *PROJECT_TYPES])

    filters: dict[str, str] = {}
    if dept != "(指定なし・全部署横断)":
        filters["dept"] = dept
    if project_type != "(指定なし)":
        filters["project_type"] = project_type
    return filters


_MODE_BASELINE = "通常（1回検索）"
_MODE_AGENT = "エージェント（LangGraph・検索クエリを作ってから検索）"


def _mode_widget() -> str:
    """Answer-mode selector; baseline only if the optional langgraph extra (.[agent]) is missing."""
    st.sidebar.header("回答方式")
    if importlib.util.find_spec("langgraph") is None:
        st.sidebar.caption('エージェントを使うには `pip install -e ".[agent]"` が必要です。')
        return _MODE_BASELINE
    return st.sidebar.radio("方式", [_MODE_BASELINE, _MODE_AGENT], label_visibility="collapsed")


def _render_trace(trace: list[str] | None) -> None:
    if not trace:
        return
    with st.expander("エージェントの動き"):
        for line in trace:
            st.text(line)


def _render_citations(answer: Answer, scored_chunks: list[ScoredChunk]) -> None:
    if not answer.citations:
        return
    text_lookup = {
        (sc.chunk.metadata.get("report_id"), sc.chunk.metadata.get("section")): sc.chunk.text
        for sc in scored_chunks
    }
    st.markdown("**参照した過去事例:**")
    for c in answer.citations:
        with st.expander(f"{c.report_id} ／ {c.section} ／ {c.dept}"):
            st.text(text_lookup.get((c.report_id, c.section), "（本文を取得できませんでした）"))


def main() -> None:
    st.title("🔧 部署横断ナレッジ検索")
    st.caption(
        "部署横断のプロジェクト知見・教訓を、部署間の情報共有が"
        "不十分な状況でも横断的に検索・再利用できるようにするデモです。"
    )

    client = get_client()
    if not client.ping():
        st.error(
            f"LM Studio ({load_config().ai.base_url}) に接続できません。"
            "起動してモデルをロードしてから再読み込みしてください。"
        )
        st.stop()

    mode = _mode_widget()
    filters = _filter_widgets()

    if "history" not in st.session_state:
        st.session_state.history = []

    for question, answer, scored, past_trace in st.session_state.history:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            st.write(answer.text)
            _render_trace(past_trace)
            _render_citations(answer, scored)

    # st.chat_input submits on Enter, including the Enter that confirms IME conversion.
    # A text_area + submit button form keeps Enter as a newline.
    with st.form("question_form", clear_on_submit=True):
        question = st.text_area(
            "質問を入力してください（例: 新商品の販促キャンペーンで、他部署の失敗事例を知りたい）",
            height=80,
        )
        submitted = st.form_submit_button("送信")

    if not submitted or not question.strip():
        return
    question = question.strip()

    # The form clears on submit and history only shows after rerun, so echo the question now
    # or it vanishes from the screen while processing.
    with st.chat_message("user"):
        st.write(question)

    # History lets both retrieval and generation resolve follow-up references.
    history = [(q, a.text) for q, a, _, _ in st.session_state.history] or None

    trace: list[str] | None = None
    with st.spinner("検索・回答生成中..."):
        try:
            if mode == _MODE_AGENT:
                from silo_rag.agent import run_agent

                result = run_agent(client, question, filters=filters or None, history=history)
                answer, scored, trace = result.answer, result.scored_chunks, result.trace
            else:
                scored = search(client, question, filters=filters or None, history=history)
                answer = answer_question(client, question, [sc.chunk for sc in scored], history=history)
        except RuntimeError as exc:
            # Setup errors (e.g. no ChromaDB collection because ingest wasn't run);
            # LLMConnectionError is also a RuntimeError, so it's shown here too.
            st.error(str(exc))
            return

    st.session_state.history.append((question, answer, scored, trace))
    st.rerun()


if __name__ == "__main__":
    main()
