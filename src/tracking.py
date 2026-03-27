"""
Lightweight run tracking — appends runs to a JSON-lines file and a CSV summary.
"""

import csv
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional


RUNS_JSONL = "outputs/runs/runs.jsonl"
RUNS_CSV = "outputs/runs/runs_summary.csv"

CSV_FIELDS = [
    "run_id", "timestamp", "commit", "config", "split", "data_version",
    "threshold", "accuracy", "balanced_accuracy", "f1", "tpr", "fpr",
    "tp", "fp", "fn", "tn", "note",
]


def _get_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _next_run_id(jsonl_path: str) -> str:
    n = 0
    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r") as f:
            n = sum(1 for line in f if line.strip())
    return f"run_{n + 1:03d}"


def log_run(
    metrics: Dict[str, Any],
    threshold: float,
    split: str,
    config_name: str,
    data_version: str,
    note: str = "",
    extra: Optional[Dict[str, Any]] = None,
    runs_dir: str = "outputs/runs",
) -> str:
    os.makedirs(runs_dir, exist_ok=True)
    jsonl_path = os.path.join(runs_dir, "runs.jsonl")
    csv_path = os.path.join(runs_dir, "runs_summary.csv")

    run_id = _next_run_id(jsonl_path)
    ts = datetime.now(timezone.utc).isoformat()
    commit = _get_commit()

    record = {
        "run_id": run_id,
        "timestamp": ts,
        "commit": commit,
        "config": config_name,
        "split": split,
        "data_version": data_version,
        "threshold": threshold,
        "note": note,
        **metrics,
    }
    if extra:
        record.update(extra)

    with open(jsonl_path, "a") as f:
        f.write(json.dumps(record) + "\n")

    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(record)

    print(f"[tracking] Logged {run_id} | split={split} | threshold={threshold:.4f} | bal_acc={metrics.get('balanced_accuracy', 'N/A'):.4f}")
    return run_id
