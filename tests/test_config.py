"""Tests for config.py: defaults and environment/TOML overrides."""

from __future__ import annotations

import pytest

from silo_rag.config import AgentConfig, RetrievalConfig, load_config


def test_agent_defaults_are_the_best_measured_small_model_setting():
    # Matches docs/agent_evaluation_*.md (best on 7B); update that doc and the README if changed.
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


def test_rewrite_query_is_off_by_default():
    # Helped on 7B but showed no gain on 32B, so the default searches with the raw question.
    assert RetrievalConfig().rewrite_query is False


@pytest.mark.parametrize(
    ("env_value", "expected"),
    [("true", True), ("1", True), ("On", True), ("false", False), ("0", False), ("off", False)],
)
def test_rewrite_query_can_be_set_by_environment(monkeypatch, tmp_path, env_value, expected):
    # Env values are strings; "false" must not be truthy.
    empty_toml = tmp_path / "empty.toml"
    empty_toml.write_text("", encoding="utf-8")
    monkeypatch.setenv("SILORAG_RETRIEVAL_REWRITE_QUERY", env_value)

    assert load_config(empty_toml).retrieval.rewrite_query is expected


def test_rewrite_query_can_be_set_in_toml(tmp_path):
    toml = tmp_path / "config.toml"
    toml.write_text("[retrieval]\nrewrite_query = true\n", encoding="utf-8")

    assert load_config(toml).retrieval.rewrite_query is True


@pytest.mark.parametrize("env_value", ["1", "3"])
def test_langchain_search_cap_ignores_the_agent_max_attempts_setting(monkeypatch, env_value):
    # The custom agent's max_attempts must not change the LangChain agent's search cap (3).
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
