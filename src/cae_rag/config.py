"""設定の読み込み。

優先順位（強い順）:
    1. 環境変数（CAERAG_ プレフィックス）
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
    リポジトリ内の src/cae_rag/config.py のまま残るので、pyproject.toml を目印に
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
COLLECTION_NAME = "cae_reports"


@dataclass
class ServerConfig:
    """LM Studio等、ローカルモデルサーバーへの接続設定＋そこから使う3種類のモデル指定。

    base_url/api_key/timeout/max_tokensは接続・リクエストの設定、
    model/embed_model/vlm_modelは「同じサーバーのどのモデルを使うか」の指定。
    どれも同じサーバー（同じbase_url）へのリクエストなので1つのセクションにまとめている
    （役割ごとにサーバー自体を分けたい場合は、このdataclass自体を分割する必要がある）。
    """

    base_url: str = "http://localhost:1234/v1"
    api_key: str = "local-no-key"
    # チャット/生成用モデル。
    model: str = "qwen2.5-7b-instruct"
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
    server: ServerConfig = field(default_factory=ServerConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)


_ENV_MAP: dict[str, tuple[str, str, Callable[[str], object]]] = {
    "CAERAG_SERVER_BASE_URL": ("server", "base_url", str),
    "CAERAG_SERVER_API_KEY": ("server", "api_key", str),
    "CAERAG_SERVER_MODEL": ("server", "model", str),
    "CAERAG_SERVER_EMBED_MODEL": ("server", "embed_model", str),
    "CAERAG_SERVER_VLM_MODEL": ("server", "vlm_model", str),
    "CAERAG_SERVER_TIMEOUT": ("server", "timeout", float),
    "CAERAG_SERVER_MAX_TOKENS": ("server", "max_tokens", int),
    "CAERAG_RETRIEVAL_VECTOR_WEIGHT": ("retrieval", "vector_weight", float),
    "CAERAG_RETRIEVAL_TOP_K_CANDIDATES": ("retrieval", "top_k_candidates", int),
    "CAERAG_RETRIEVAL_TOP_K_FINAL": ("retrieval", "top_k_final", int),
}

_SECTION_TYPES = {
    "server": ServerConfig,
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

    # 旧セクション名"[llm]"（"[server]"へのリネーム前）が残っていると、設定が
    # 黙って全部デフォルト値に戻ってしまう（"server"キーが無いだけなので例外にならない）。
    # 気づかないまま意図しないモデル・接続先で動いてしまうのを防ぐため警告する。
    if "llm" in data and "server" not in data:
        print(
            "警告: config.tomlの[llm]セクションは[server]にリネームされました。"
            "config.example.tomlを参照して[server]に書き換えてください"
            "（このままだとconfig.tomlの内容は無視され、コード内のデフォルト値が使われます）。"
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
