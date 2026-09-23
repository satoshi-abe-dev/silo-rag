#!/usr/bin/env bash
# UI起動前の下準備（合成データ生成 → 取り込み → 評価）を1コマンドでまとめて実行する。
# 個別ステップだけやり直したい場合は、README記載の各コマンドを直接叩けばよい。
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "=== 1/3: 合成データ生成 ==="
python -m silo_rag.datagen

echo "=== 2/3: チャンキング + 埋め込み + ChromaDBへの格納 ==="
python -m silo_rag.ingest

echo "=== 3/3: 評価 ==="
python -m silo_rag.eval

echo
echo "準備完了。次のコマンドでUIを起動できます:"
echo "  streamlit run src/silo_rag/app.py"
