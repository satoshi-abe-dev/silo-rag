#!/usr/bin/env bash
# baseline と エージェント（判定 strict／lenient）を同じ評価セットで回し、比較表を表示する。
#
# 使い方（仮想環境を有効化した状態で）:
#   bash scripts/compare_pipelines.sh                       # config.tomlのllm_modelで実行
#   bash scripts/compare_pipelines.sh qwen2.5-7b-instruct   # 回答モデルを指定して実行
#   bash scripts/compare_pipelines.sh qwen2.5-7b-instruct --rewrite-first   # 1回目からクエリを作る版（3種）も追加
#
# 実行ごとに、エージェントの設定（first_query・max_attempts）は明示している。config.tomlや環境変数の
# 既定（クエリを作って1回検索する）に左右されず、毎回同じ条件で比べるため。
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

REWRITE_FIRST=0
ARGS=()
for arg in "$@"; do
  if [[ "$arg" == "--rewrite-first" ]]; then REWRITE_FIRST=1; else ARGS+=("$arg"); fi
done
if [[ ${#ARGS[@]} -ge 1 ]]; then
  export SILORAG_AI_LLM_MODEL="${ARGS[0]}"
fi

MODEL=$(python -c 'from silo_rag.config import load_config; print(load_config().ai.llm_model)')
SLUG=$(printf '%s' "$MODEL" | tr -c 'A-Za-z0-9._-' '_')
OUT="data/eval/compare/$SLUG"
mkdir -p "$OUT"
echo "回答モデル: $MODEL → $OUT"

echo "=== 1/3: baseline ==="
python -m silo_rag.eval --pipeline baseline --out "$OUT/baseline.json"

echo "=== 2/3: agent (strict) ==="
python -m silo_rag.eval --pipeline agent --grade-mode strict --first-query raw --max-attempts 3 \
  --out "$OUT/agent_strict.json"

echo "=== 3/3: agent (lenient) ==="
python -m silo_rag.eval --pipeline agent --grade-mode lenient --first-query raw --max-attempts 3 \
  --out "$OUT/agent_lenient.json"

RESULTS=("$OUT/baseline.json" "$OUT/agent_strict.json" "$OUT/agent_lenient.json")

if [[ $REWRITE_FIRST -eq 1 ]]; then
  echo "=== 追加1/3: agent (strict, rewrite-first, 3 searches) ==="
  python -m silo_rag.eval --pipeline agent --grade-mode strict --first-query rewrite --max-attempts 3 \
    --out "$OUT/agent_strict_rewrite.json"
  echo "=== 追加2/3: agent (lenient, rewrite-first, 3 searches) ==="
  python -m silo_rag.eval --pipeline agent --grade-mode lenient --first-query rewrite --max-attempts 3 \
    --out "$OUT/agent_lenient_rewrite.json"
  echo "=== 追加3/3: agent (lenient, rewrite-first, 1 search) ==="
  python -m silo_rag.eval --pipeline agent --grade-mode lenient --first-query rewrite --max-attempts 1 \
    --out "$OUT/agent_lenient_rewrite_1search.json"
  RESULTS+=("$OUT/agent_strict_rewrite.json" "$OUT/agent_lenient_rewrite.json"
            "$OUT/agent_lenient_rewrite_1search.json")
fi

echo
echo "=== 比較 ==="
python -m silo_rag.eval --compare "${RESULTS[@]}"
