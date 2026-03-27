# Face Verification Pipeline — Milestone 2

## Project Overview

This project builds a reproducible face verification system on LFW. Given two face images, the pipeline produces a cosine-similarity score and a same/different-person binary decision. Milestone 2 adds a disciplined evaluation loop: threshold sweeps, run tracking, error analysis, a data-centric improvement, and tests.

## Milestone 2 Summary

**Baseline (v1):** Cosine similarity on flattened, L2-normalized pixel vectors. Threshold selected on the val split by maximizing balanced accuracy.

**Data-centric improvement (v2):** Capped overrepresented identities to at most 10 images (`max_images_per_identity=10`) so no single identity dominates the pair distribution. New pairs generated into `outputs/pairs_v2/`. Threshold re-selected on the same val split.

**Report:** `reports/milestone2_report.pdf`

---

## Repository Structure

```
repo_root/
├── configs/
│   ├── m1.yaml              # Milestone 1 config
│   ├── m2.yaml              # Milestone 2 baseline config
│   └── m2_capped.yaml       # Data-centric v2 config
├── src/
│   ├── similarity.py        # Vectorized cosine & Euclidean
│   ├── metrics.py           # ROC, confusion matrix, threshold selection
│   ├── validation.py        # Input/output validation checks
│   ├── tracking.py          # Run logging (JSONL + CSV)
│   ├── error_analysis.py    # Error slicing utilities
│   └── scoring.py           # Pair scorer (pixel cosine)
├── scripts/
│   ├── ingest_lfw.py        # LFW ingestion and split
│   ├── make_pairs.py        # Pair generation (v1 + v2 with capping)
│   ├── bench_similarity.py  # Loop vs vectorized benchmark
│   ├── evaluate.py          # Threshold sweep, metrics, run logging
│   └── run_all.sh           # Reproduce all 5 tracked runs
├── tests/
│   ├── conftest.py
│   ├── test_unit.py         # Unit tests
│   └── test_integration.py  # Small integration test
├── reports/
│   └── milestone2_report.pdf
├── outputs/                 # Generated artifacts (not committed)
│   ├── runs/                # runs.jsonl + runs_summary.csv
│   ├── eval/                # v1 sweep, ROC, confusion matrix, error analysis
│   └── eval_v2/             # v2 equivalents
└── data/                    # Dataset cache (not committed)
```

---

## How to Run

### 1. Environment setup
```bash
pip install -r requirements.txt
```

### 2. Ingest LFW and generate splits
```bash
python scripts/ingest_lfw.py --config configs/m2.yaml
```

### 3. Generate pairs (baseline v1)
```bash
python scripts/make_pairs.py --config configs/m2.yaml --data-version v1
```

### 4. Generate pairs (v2 capped — data-centric improvement)
```bash
python scripts/make_pairs.py --config configs/m2_capped.yaml --data-version v2
```

### 5. Run all 5 tracked runs (requires pairs already generated)
```bash
bash scripts/run_all.sh
```

Or run individual evaluations:
```bash
# Threshold sweep on val (baseline)
python scripts/evaluate.py --config configs/m2.yaml --split val --sweep --data-version v1

# Evaluate at a fixed threshold
python scripts/evaluate.py --config configs/m2.yaml --split test --threshold 0.72 --data-version v1
```

### 6. Run tests
```bash
pytest tests/ -v
```

---

## Artifact Locations

| Artifact | Path |
|---|---|
| Tracked runs (JSONL) | `outputs/runs/runs.jsonl` |
| Run summary (CSV) | `outputs/runs/runs_summary.csv` |
| ROC curve (baseline) | `outputs/eval/val/roc.png` |
| Confusion matrix (baseline) | `outputs/eval/val/confusion_matrix.png` |
| Selected threshold (baseline) | `outputs/eval/val/selected_threshold.json` |
| Error analysis (baseline) | `outputs/eval/val/error_analysis.json` |
| ROC curve (v2) | `outputs/eval_v2/val/roc.png` |
| Milestone 2 report | `reports/milestone2_report.pdf` |

---

## Threshold Selection Rule

Threshold is selected on the **val split** by maximizing **balanced accuracy** across a sweep of 100 evenly spaced thresholds. The rule is applied identically for both v1 and v2 runs. The locked threshold is then applied to the held-out **test split** for final reporting.

---

## Notes

- `evaluate.py` uses a mock scorer by default (reproducible, no dataset required). Pass `--use-real-scores` to use the pixel-cosine scorer with real TFDS data.
- The integration test (`tests/test_integration.py`) uses synthetic pairs and runs without the LFW dataset.
