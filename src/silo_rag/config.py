"""Configuration loading.

Precedence (highest first):
    1. Environment variables (SILORAG_ prefix)
    2. TOML file (config.toml, else config.example.toml)
    3. Defaults in code

Uses stdlib tomllib; same approach as the meeting-minutes project's config.py.
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Callable
from dataclasses import dataclass, field, fields
from pathlib import Path


def _find_repo_root() -> Path:
    """Find the repo root by walking up to pyproject.toml.

    Works with an editable install (the supported setup). If not found (e.g. a
    non-editable install into site-packages), falls back to two levels up; unsupported.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return here.parents[2]


REPO_ROOT = _find_repo_root()
DATA_DIR = REPO_ROOT / "data"
SYNTH_REPORTS_DIR = DATA_DIR / "synth_reports"
EVAL_DIR = DATA_DIR / "eval"
CHROMA_DIR = DATA_DIR / "chroma_db"
# Kept independent of the package/repo name so a rename doesn't orphan existing local indexes.
COLLECTION_NAME = "reports"


@dataclass
class AIConfig:
    """Local model server (e.g. LM Studio) connection settings and the three models used on it.

    All models share one base_url; using separate servers per role would require splitting this class.
    """

    base_url: str = "http://localhost:1234/v1"
    api_key: str = "local-no-key"
    # Chat/generation model.
    llm_model: str = "qwen2.5-7b-instruct"
    # Embedding model; must be loaded in LM Studio.
    embed_model: str = "text-embedding-nomic-embed-text-v1.5"
    # Vision model for image captions (e.g. Qwen2.5-VL / Qwen3-VL). If not loaded, ingest
    # just logs and skips images; nothing else is affected.
    vlm_model: str = "qwen2.5-vl-7b-instruct"
    timeout: float = 300.0
    max_tokens: int = 2048


@dataclass
class RetrievalConfig:
    # Hybrid score blend: 0.0 = BM25 only, 1.0 = vectors only.
    vector_weight: float = 0.5
    top_k_candidates: int = 20
    top_k_final: int = 5
    # Have the LLM build a search query before searching (retrieval.plan_query). Off by default:
    # it helped on the 15-question eval (7B) but showed no gain on 45 questions with 7B or 32B,
    # and costs one extra LLM call per question. See docs/agent_evaluation_en.md.
    rewrite_query: bool = False


@dataclass
class AgentConfig:
    # Defaults (max_attempts=1, first_query="rewrite") scored best on the 15-question 7B eval.
    # Retry loops added nothing on top of query planning with 7B or 32B; with 32B the agent matched
    # the standard pipeline and lost one cross-department question, so larger models use the
    # standard pipeline. See docs/agent_evaluation_en.md.
    #
    # Max searches by the LangGraph agent (agent.py), including the first; 1 = no retry.
    # The LLM judges whether evidence suffices, but code caps the retries.
    max_attempts: int = 1
    # Strictness of the evidence check: "strict" or "lenient" (see agent.py _GRADE_SYSTEM_PROMPTS).
    grade_mode: str = "lenient"
    # First-search query: "raw" = the question as is, "rewrite" = LLM-built query (agent.py plan
    # node), since self-introductions and request phrasing scatter keyword search.
    first_query: str = "rewrite"


@dataclass
class Config:
    ai: AIConfig = field(default_factory=AIConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)


def _to_bool(text: str) -> bool:
    """Parse an env var string as a bool (bool("false") would be True)."""
    value = text.strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off", ""):
        return False
    raise ValueError(f"真偽値として解釈できません: {text!r}")


_ENV_MAP: dict[str, tuple[str, str, Callable[[str], object]]] = {
    "SILORAG_AI_BASE_URL": ("ai", "base_url", str),
    "SILORAG_AI_API_KEY": ("ai", "api_key", str),
    "SILORAG_AI_LLM_MODEL": ("ai", "llm_model", str),
    "SILORAG_AI_EMBED_MODEL": ("ai", "embed_model", str),
    "SILORAG_AI_VLM_MODEL": ("ai", "vlm_model", str),
    "SILORAG_AI_TIMEOUT": ("ai", "timeout", float),
    "SILORAG_AI_MAX_TOKENS": ("ai", "max_tokens", int),
    "SILORAG_RETRIEVAL_VECTOR_WEIGHT": ("retrieval", "vector_weight", float),
    "SILORAG_RETRIEVAL_TOP_K_CANDIDATES": ("retrieval", "top_k_candidates", int),
    "SILORAG_RETRIEVAL_TOP_K_FINAL": ("retrieval", "top_k_final", int),
    "SILORAG_RETRIEVAL_REWRITE_QUERY": ("retrieval", "rewrite_query", _to_bool),
    "SILORAG_AGENT_MAX_ATTEMPTS": ("agent", "max_attempts", int),
    "SILORAG_AGENT_GRADE_MODE": ("agent", "grade_mode", str),
    "SILORAG_AGENT_FIRST_QUERY": ("agent", "first_query", str),
}

_SECTION_TYPES = {
    "ai": AIConfig,
    "retrieval": RetrievalConfig,
    "agent": AgentConfig,
}


def default_config_path() -> Path | None:
    """Pick the TOML file: config.toml, then config.example.toml, else None."""
    for name in ("config.toml", "config.example.toml"):
        p = REPO_ROOT / name
        if p.is_file():
            return p
    return None


def _build_section(section_cls: type, raw: dict) -> object:
    """Build a dataclass section from a dict, ignoring unknown keys and loosely coercing types."""
    known = {f.name: f for f in fields(section_cls)}
    kwargs = {}
    for key, value in raw.items():
        if key not in known:
            continue
        target_type = known[key].type
        try:
            if target_type in ("int", int):
                value = int(value)
            elif target_type in ("float", float):
                value = float(value)
            elif target_type in ("str", str):
                value = str(value)
        except (TypeError, ValueError):
            pass
        kwargs[key] = value
    return section_cls(**kwargs)


def load_config(path: str | os.PathLike | None = None) -> Config:
    """Load the configuration.

    path: TOML path; defaults to default_config_path().
    """
    toml_path: Path | None
    if path is not None:
        toml_path = Path(path)
        if not toml_path.is_file():
            raise FileNotFoundError(f"設定ファイルが見つかりません: {toml_path}")
    else:
        toml_path = default_config_path()

    data: dict = {}
    if toml_path is not None:
        with open(toml_path, "rb") as f:
            data = tomllib.load(f)

    # A legacy section name ([llm], or the interim [server]) would silently fall back to all
    # defaults, since a missing "ai" key raises nothing. Warn about either.
    legacy_sections = [name for name in ("llm", "server") if name in data]
    if legacy_sections and "ai" not in data:
        print(
            f"警告: config.tomlの{[f'[{n}]' for n in legacy_sections]}セクションは[ai]にリネームされました。"
            "config.example.tomlを参照して[ai]に書き換えてください"
            "（このままだとconfig.tomlの内容は無視され、コード内のデフォルト値が使われます）。"
        )

    # Likewise, a leftover "model" key (now "llm_model") would be silently ignored.
    ai_section = data.get("ai") or data.get("server") or data.get("llm") or {}
    if isinstance(ai_section, dict) and "model" in ai_section and "llm_model" not in ai_section:
        print(
            "警告: config.tomlの`model`キーは`llm_model`にリネームされました。"
            "config.example.tomlを参照して書き換えてください"
            "（このままだと指定したモデル名は無視され、コード内のデフォルト値が使われます）。"
        )

    sections: dict[str, object] = {}
    for name, cls in _SECTION_TYPES.items():
        sections[name] = _build_section(cls, data.get(name, {}) or {})

    for env_name, (section, key, caster) in _ENV_MAP.items():
        if env_name not in os.environ:
            continue
        raw = os.environ[env_name]
        try:
            casted = caster(raw)
        except (TypeError, ValueError):
            casted = raw
        setattr(sections[section], key, casted)

    return Config(**sections)  # type: ignore[arg-type]
