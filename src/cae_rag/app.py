"""Streamlit UI（DAGノードF）。

質問応答チャット + 引用元（類似事例）一覧を表示する、社内ナレッジ検索RAGのデモUI。
retrieval.search() と generation.answer_question() をこの層で初めて組み合わせる
（両モジュールはDAG上お互いに依存しないよう独立実装されているため）。

起動方法: streamlit run src/cae_rag/app.py
（`streamlit run` はファイルを直接実行するため、他モジュールのような相対import
（`from .config import ...`）ではなく絶対import（`from cae_rag.config import ...`）を使う。
`pip install -e .` 済みであれば、cwdによらず絶対importで解決できる。）
"""

from __future__ import annotations

import streamlit as st

from cae_rag.config import load_config
from cae_rag.datagen import ANALYSIS_TYPES, DEPARTMENTS
from cae_rag.generation import Answer, answer_question
from cae_rag.llm_client import LLMClient, LLMConnectionError
from cae_rag.retrieval import ScoredChunk, search

st.set_page_config(page_title="CAE解析ナレッジ検索", page_icon="🔧")


@st.cache_resource
def get_client() -> LLMClient:
    config = load_config()
    return LLMClient(config.ai)


def _filter_widgets() -> dict[str, str]:
    """サイドバーの絞り込みUI。未選択のキーはfiltersに含めない（＝全件対象）。"""
    st.sidebar.header("絞り込み（任意）")
    dept = st.sidebar.selectbox("作成部署", ["(指定なし・全部署横断)", *sorted(DEPARTMENTS.keys())])
    analysis_type = st.sidebar.selectbox("解析種別", ["(指定なし)", *ANALYSIS_TYPES])

    filters: dict[str, str] = {}
    if dept != "(指定なし・全部署横断)":
        filters["dept"] = dept
    if analysis_type != "(指定なし)":
        filters["analysis_type"] = analysis_type
    return filters


def _render_citations(answer: Answer, scored_chunks: list[ScoredChunk]) -> None:
    if not answer.citations:
        return
    # 引用（report_id, section）に対応する本文スニペットを、検索結果チャンクから引く。
    text_lookup = {
        (sc.chunk.metadata.get("report_id"), sc.chunk.metadata.get("section")): sc.chunk.text
        for sc in scored_chunks
    }
    st.markdown("**参照した過去事例:**")
    for c in answer.citations:
        with st.expander(f"{c.report_id} ／ {c.section} ／ {c.dept}"):
            st.text(text_lookup.get((c.report_id, c.section), "（本文を取得できませんでした）"))


def main() -> None:
    st.title("🔧 CAE解析ナレッジ検索アシスタント")
    st.caption(
        "自動車部品の構造解析(FEM)に関する社内の過去事例を、部署間の情報共有が"
        "不十分な状況でも横断的に検索・再利用できるようにするデモです。"
    )

    client = get_client()
    if not client.ping():
        st.error(
            f"LM Studio ({load_config().ai.base_url}) に接続できません。"
            "起動してモデルをロードしてから再読み込みしてください。"
        )
        st.stop()

    filters = _filter_widgets()

    if "history" not in st.session_state:
        st.session_state.history: list[tuple[str, Answer, list[ScoredChunk]]] = []

    for question, answer, scored in st.session_state.history:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            st.write(answer.text)
            _render_citations(answer, scored)

    question = st.chat_input("質問を入力してください（例: アルミ製ブラケットの固有値解析で共振を避けたい）")
    if not question:
        return

    with st.chat_message("user"):
        st.write(question)

    with st.spinner("検索・回答生成中..."):
        try:
            scored = search(client, question, filters=filters or None)
        except RuntimeError as exc:
            # ChromaDBコレクション未作成（ingest未実行）などの構成エラー。
            st.error(str(exc))
            return
        chunks = [sc.chunk for sc in scored]
        try:
            answer = answer_question(client, question, chunks)
        except LLMConnectionError as exc:
            st.error(str(exc))
            return

    with st.chat_message("assistant"):
        st.write(answer.text)
        _render_citations(answer, scored)

    st.session_state.history.append((question, answer, scored))


if __name__ == "__main__":
    main()
