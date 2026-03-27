#!/usr/bin/env bash
# Reproduce all 5 tracked Milestone 2 runs.
# Run from repo root: bash scripts/run_all.sh
set -e

CONFIG_V1="configs/m2.yaml"
CONFIG_V2="configs/m2_capped.yaml"

echo "=== Run 1: Baseline threshold sweep on val ==="
python scripts/evaluate.py \
  --config $CONFIG_V1 \
  --split val \
  --sweep \
  --data-version v1 \
  --note "baseline val sweep"

echo "=== Run 2: Baseline selected-threshold eval on val (confusion matrix) ==="
THRESHOLD=$(python -c "import json; d=json.load(open('outputs/eval/val/selected_threshold.json')); print(d['threshold'])")
python scripts/evaluate.py \
  --config $CONFIG_V1 \
  --split val \
  --threshold $THRESHOLD \
  --data-version v1 \
  --note "baseline val selected threshold"

echo "=== Run 3: Baseline final report on test ==="
python scripts/evaluate.py \
  --config $CONFIG_V1 \
  --split test \
  --threshold $THRESHOLD \
  --data-version v1 \
  --note "baseline test final report"

echo "=== Generating v2 pairs (capped) ==="
python scripts/make_pairs.py --config $CONFIG_V2 --data-version v2

echo "=== Run 4: Post-change threshold sweep on val (capped pairs) ==="
python scripts/evaluate.py \
  --config $CONFIG_V2 \
  --split val \
  --sweep \
  --data-version v2 \
  --note "v2 capped val sweep"

THRESHOLD_V2=$(python -c "import json; d=json.load(open('outputs/eval_v2/val/selected_threshold.json')); print(d['threshold'])")

echo "=== Run 5: Post-change test report (capped pairs) ==="
python scripts/evaluate.py \
  --config $CONFIG_V2 \
  --split test \
  --threshold $THRESHOLD_V2 \
  --data-version v2 \
  --note "v2 capped test final report"

echo "=== All 5 runs complete. Results in outputs/runs/ ==="
