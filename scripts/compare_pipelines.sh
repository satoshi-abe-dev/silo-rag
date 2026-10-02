#!/usr/bin/env bash
# baseline と エージェント（判定 strict／lenient）を同じ評価セットで回し、比較表を表示する。
#
# 使い方（仮想環境を有効化した状態で）:
#   bash scripts/compare_pipelines.sh                       # config.tomlのllm_modelで実行
#   bash scripts/compare_pipelines.sh qwen2.5-7b-instruct   # 回答モデルを指定して実行
#
# 結果は data/eval/compare/<モデル名>/ に書き出す。別モデルの結果と並べるには:
#   python -m silo_rag.eval --compare data/eval/compare/*/*.json
#
# judge（LLM-as-judge）の採点モデルは、回答モデルと同じにしている（LM Studioで2つの大きな
# モデルを1問ごとに切り替えると、ロードし直しで極端に遅くなるため）。そのため、judgeの値は
# 同じモデルの実行どうしでしか比べられない（比較表にも注意が出る）。検索精度（hit_rate等）は
# 採点モデルに依存しないので、モデルをまたいで比べられる。
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ $# -ge 1 ]]; then
  export SILORAG_AI_LLM_MODEL="$1"
fi

MODEL=$(python -c 'from silo_rag.config import load_config; print(load_config().ai.llm_model)')
SLUG=$(printf '%s' "$MODEL" | tr -c 'A-Za-z0-9._-' '_')
OUT="data/eval/compare/$SLUG"
mkdir -p "$OUT"
echo "回答モデル: $MODEL → $OUT"

echo "=== 1/3: baseline ==="
python -m silo_rag.eval --pipeline baseline --out "$OUT/baseline.json"

echo "=== 2/3: agent (strict) ==="
python -m silo_rag.eval --pipeline agent --grade-mode strict --out "$OUT/agent_strict.json"

echo "=== 3/3: agent (lenient) ==="
python -m silo_rag.eval --pipeline agent --grade-mode lenient --out "$OUT/agent_lenient.json"

echo
echo "=== 比較 ==="
python -m silo_rag.eval --compare "$OUT/baseline.json" "$OUT/agent_strict.json" "$OUT/agent_lenient.json"
