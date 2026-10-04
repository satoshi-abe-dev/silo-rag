#!/usr/bin/env bash
# Prepare everything the UI needs in one command: generate data -> ingest -> evaluate.
# To redo a single step, run its command from the README directly.
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
echo "  streamlit run src/silo_rag/app.py --server.address localhost"
