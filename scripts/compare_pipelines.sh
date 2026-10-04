#!/usr/bin/env bash
# Run baseline and the agent (strict / lenient grading) on the same eval set and print a comparison.
#
# Usage (with the venv active):
#   bash scripts/compare_pipelines.sh                       # llm_model from config.toml
#   bash scripts/compare_pipelines.sh qwen2.5-7b-instruct   # pick the answer model
#   bash scripts/compare_pipelines.sh qwen2.5-7b-instruct --rewrite-first   # also run 3 rewrite-first variants
#
# Agent settings (first_query, max_attempts) are passed explicitly so config.toml or env defaults
# can't change the conditions between runs.
#
# Results go to data/eval/compare/<model>/. To compare across models:
#   python -m silo_rag.eval --compare data/eval/compare/*/*.json
#
# The judge uses the answer model, since swapping two large models per question in LM Studio is very
# slow. So judge scores only compare within one model; retrieval metrics compare across models.
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
