"""設定の読み込み。

優先順位（強い順）:
    1. 環境変数（FEMRAG_ プレフィックス）
    2. TOML ファイル（既定は config.toml、無ければ config.example.toml）
    3. コード内のデフォルト値

TOML は標準ライブラリ tomllib（Python 3.11+）で読む。追加依存なし。
meeting-minutes プロジェクト（src/meeting_minutes/model/config.py）と同じ考え方を踏襲している。
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Callable
from dataclasses import dataclass, field, fields
from pathlib import Path


def _find_repo_root() -> Path:
    """リポジトリのルートを探す。

    `pip install -e .`（本プロジェクトが前提とする導入方法）ならこのファイルは
    リポジトリ内の src/fem_rag/config.py のまま残るので、pyproject.toml を目印に
    上へ辿れば見つかる。万一 `pip install .`（非editable）のように site-packages に
    コピーされていて見つからない場合は、通常のsrcレイアウトの相対位置（2つ上）に
    フォールバックする（その場合の動作は保証しない。本プロジェクトはeditable
    インストールまたはリポジトリ直下からの実行のみを想定している）。
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return here.parents[2]


# リポジトリのルート。
REPO_ROOT = _find_repo_root()
DATA_DIR = REPO_ROOT / "data"
SYNTH_REPORTS_DIR = DATA_DIR / "synth_reports"
EVAL_DIR = DATA_DIR / "eval"
CHROMA_DIR = DATA_DIR / "chroma_db"
COLLECTION_NAME = "fem_reports"


@dataclass
class AIConfig:
    """LM Studio等、ローカルモデルサーバーへの接続設定＋そこから使う3種類のモデル指定。

    base_url/api_key/timeout/max_tokensは接続・リクエストの設定、
    llm_model/embed_model/vlm_modelは「同じサーバーのどのモデルを使うか」の指定
    （役割ごとに名前を揃えてある）。どれも同じサーバー（同じbase_url）へのリクエストなので
    1つのセクションにまとめている（役割ごとにサーバー自体を分けたい場合は、この
    dataclass自体を分割する必要がある）。
    """

    base_url: str = "http://localhost:1234/v1"
    api_key: str = "local-no-key"
    # チャット/生成用モデル。
    llm_model: str = "qwen2.5-7b-instruct"
    # 埋め込み用モデル。LM Studio に埋め込みモデルをロードしておく必要がある。
    embed_model: str = "text-embedding-nomic-embed-text-v1.5"
    # 画像説明（VLM）用モデル。LM Studio にvisionモデルをロードしておく必要がある
    # （例: Qwen2.5-VL / Qwen3-VL系）。未ロードでも他機能には影響しない
    # （画像キャプション取得に失敗した場合はログを出して該当画像をスキップするのみ）。
    vlm_model: str = "qwen2.5-vl-7b-instruct"
    timeout: float = 300.0
    max_tokens: int = 2048


@dataclass
class RetrievalConfig:
    # ハイブリッド検索でのスコア統合比率（0.0=BM25のみ, 1.0=ベクトルのみ）
    vector_weight: float = 0.5
    top_k_candidates: int = 20
    top_k_final: int = 5


@dataclass
class Config:
    ai: AIConfig = field(default_factory=AIConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)


_ENV_MAP: dict[str, tuple[str, str, Callable[[str], object]]] = {
    "FEMRAG_AI_BASE_URL": ("ai", "base_url", str),
    "FEMRAG_AI_API_KEY": ("ai", "api_key", str),
    "FEMRAG_AI_LLM_MODEL": ("ai", "llm_model", str),
    "FEMRAG_AI_EMBED_MODEL": ("ai", "embed_model", str),
    "FEMRAG_AI_VLM_MODEL": ("ai", "vlm_model", str),
    "FEMRAG_AI_TIMEOUT": ("ai", "timeout", float),
    "FEMRAG_AI_MAX_TOKENS": ("ai", "max_tokens", int),
    "FEMRAG_RETRIEVAL_VECTOR_WEIGHT": ("retrieval", "vector_weight", float),
    "FEMRAG_RETRIEVAL_TOP_K_CANDIDATES": ("retrieval", "top_k_candidates", int),
    "FEMRAG_RETRIEVAL_TOP_K_FINAL": ("retrieval", "top_k_final", int),
}

_SECTION_TYPES = {
    "ai": AIConfig,
    "retrieval": RetrievalConfig,
}


def default_config_path() -> Path | None:
    """使う TOML を決める。config.toml > config.example.toml > なし。"""
    for name in ("config.toml", "config.example.toml"):
        p = REPO_ROOT / name
        if p.is_file():
            return p
    return None


def _build_section(section_cls: type, raw: dict) -> object:
    """辞書から dataclass セクションを作る。未知キーは無視し、型は緩く合わせる。"""
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
    """設定を読み込む。

    path: TOML のパス。None なら default_config_path() を使う。
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

    # 旧セクション名（"[ai]"へのリネーム前）が残っていると、設定が黙って全部
    # デフォルト値に戻ってしまう（"ai"キーが無いだけなので例外にならない）。
    # "[llm]"（最初期の名前）と"[server]"（"[ai]"に決める前に一時的に案内した名前）の
    # どちらが残っていても気づけるよう、両方チェックする。
    legacy_sections = [name for name in ("llm", "server") if name in data]
    if legacy_sections and "ai" not in data:
        print(
            f"警告: config.tomlの{[f'[{n}]' for n in legacy_sections]}セクションは[ai]にリネームされました。"
            "config.example.tomlを参照して[ai]に書き換えてください"
            "（このままだとconfig.tomlの内容は無視され、コード内のデフォルト値が使われます）。"
        )

    # セクション名は[ai]に直しても、中の"model"キー（現在は"llm_model"）を
    # リネームし忘れると同様に黙って無視される。こちらも個別に警告する。
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
