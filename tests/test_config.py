"""設定（config.py）のテスト。TOMLファイルは使わず、既定値と環境変数の上書きだけを見る。"""

from __future__ import annotations

import pytest

from silo_rag.config import AgentConfig, load_config


def test_agent_defaults_are_the_best_measured_small_model_setting():
    # README「ノードG」の評価（7Bで最良）に合わせた既定値。変えるときは、評価とREADMEも直すこと。
    defaults = AgentConfig()

    assert defaults.first_query == "rewrite"
    assert defaults.max_attempts == 1
    assert defaults.grade_mode == "lenient"


def test_agent_settings_can_be_overridden_by_environment(monkeypatch, tmp_path):
    empty_toml = tmp_path / "empty.toml"
    empty_toml.write_text("", encoding="utf-8")
    monkeypatch.setenv("SILORAG_AGENT_FIRST_QUERY", "raw")
    monkeypatch.setenv("SILORAG_AGENT_MAX_ATTEMPTS", "3")
    monkeypatch.setenv("SILORAG_AGENT_GRADE_MODE", "strict")

    agent = load_config(empty_toml).agent

    assert (agent.first_query, agent.max_attempts, agent.grade_mode) == ("raw", 3, "strict")


@pytest.mark.parametrize("env_value", ["1", "3"])
def test_langchain_search_cap_ignores_the_agent_max_attempts_setting(monkeypatch, env_value):
    # 自作エージェントの既定を1にしても、既製エージェントの検索上限（3回）は変わらないこと。
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage

    import silo_rag.langchain_adapter as adapter
    from silo_rag.ingest import Chunk
    from silo_rag.langchain_adapter import DEFAULT_MAX_SEARCHES, TOOL_NAME, run_langchain_agent
    from silo_rag.retrieval import ScoredChunk

    monkeypatch.setenv("SILORAG_AGENT_MAX_ATTEMPTS", env_value)
    monkeypatch.setattr(
        adapter,
        "search",
        lambda client, query, **kw: [
            ScoredChunk(chunk=Chunk(chunk_id=f"c-{query}", text="t", metadata={}), score=1.0)
        ],
    )

    class _Model(GenericFakeChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    def call(i):
        return AIMessage(
            content="", tool_calls=[{"name": TOOL_NAME, "args": {"query": f"q{i}"}, "id": f"t{i}"}]
        )

    model = _Model(messages=iter([call(0), call(1), call(2), AIMessage(content="回答")]))

    result = run_langchain_agent(object(), "質問", model=model)

    assert DEFAULT_MAX_SEARCHES == 3
    assert result.searches == 3
    assert result.answer.text == "回答"
