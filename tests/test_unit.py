"""
Unit tests for src/metrics.py, src/validation.py, src/tracking.py, src/error_analysis.py
"""

import csv
import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.metrics import (
    compute_confusion_matrix,
    compute_metrics_at_threshold,
    compute_roc,
    select_threshold_max_balanced_accuracy,
)
from src.validation import (
    validate_pair_file,
    validate_scores_match_pairs,
    validate_threshold,
    validate_config,
    validate_no_split_leakage,
)
from src.error_analysis import build_results_df, slice_false_positives, slice_false_negatives, summarize_slice


# ── metrics ──────────────────────────────────────────────────────────────────

def test_confusion_matrix_perfect():
    scores = np.array([0.9, 0.1, 0.8, 0.2])
    labels = np.array([1, 0, 1, 0])
    cm = compute_confusion_matrix(scores, labels, threshold=0.5)
    assert cm == {"tp": 2, "fp": 0, "fn": 0, "tn": 2}


def test_confusion_matrix_all_fp():
    scores = np.array([0.9, 0.9])
    labels = np.array([0, 0])
    cm = compute_confusion_matrix(scores, labels, threshold=0.5)
    assert cm["fp"] == 2 and cm["tp"] == 0 and cm["tn"] == 0 and cm["fn"] == 0


def test_metrics_accuracy():
    scores = np.array([0.9, 0.1, 0.8, 0.2])
    labels = np.array([1, 0, 1, 0])
    m = compute_metrics_at_threshold(scores, labels, threshold=0.5)
    assert m["accuracy"] == 1.0
    assert m["balanced_accuracy"] == 1.0
    assert m["f1"] == 1.0


def test_metrics_all_wrong():
    scores = np.array([0.1, 0.9])
    labels = np.array([1, 0])
    m = compute_metrics_at_threshold(scores, labels, threshold=0.5)
    assert m["tp"] == 0 and m["tn"] == 0


def test_roc_shape():
    scores = np.array([0.1, 0.4, 0.6, 0.9])
    labels = np.array([0, 0, 1, 1])
    thresholds = np.array([0.3, 0.5, 0.7])
    fprs, tprs, ts = compute_roc(scores, labels, thresholds)
    assert len(fprs) == len(thresholds)
    assert len(tprs) == len(thresholds)


def test_select_threshold():
    scores = np.array([0.2, 0.4, 0.6, 0.8])
    labels = np.array([0, 0, 1, 1])
    thresholds = np.linspace(0.1, 0.9, 50)
    t = select_threshold_max_balanced_accuracy(scores, labels, thresholds)
    assert 0.4 <= t <= 0.7


# ── validation ────────────────────────────────────────────────────────────────

def _make_pair_csv(rows, tmpdir, filename="pairs.csv"):
    path = os.path.join(tmpdir, filename)
    fieldnames = ["left_path", "right_path", "label", "split"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_validate_pair_file_valid(tmp_path):
    rows = [{"left_path": "a", "right_path": "b", "label": "1", "split": "val"}]
    path = _make_pair_csv(rows, str(tmp_path))
    errors = validate_pair_file(path)
    assert errors == []


def test_validate_pair_file_bad_label(tmp_path):
    rows = [{"left_path": "a", "right_path": "b", "label": "2", "split": "val"}]
    path = _make_pair_csv(rows, str(tmp_path))
    errors = validate_pair_file(path)
    assert any("label" in e for e in errors)


def test_validate_pair_file_bad_split(tmp_path):
    rows = [{"left_path": "a", "right_path": "b", "label": "1", "split": "holdout"}]
    path = _make_pair_csv(rows, str(tmp_path))
    errors = validate_pair_file(path)
    assert any("split" in e for e in errors)


def test_validate_pair_file_missing_column(tmp_path):
    path = os.path.join(str(tmp_path), "bad.csv")
    with open(path, "w") as f:
        f.write("left_path,label\na,1\n")
    errors = validate_pair_file(path)
    assert any("Missing columns" in e for e in errors)


def test_validate_scores_match():
    df = pd.DataFrame({"label": [0, 1, 0]})
    scores = np.array([0.1, 0.9, 0.2])
    assert validate_scores_match_pairs(scores, df) == []


def test_validate_scores_mismatch():
    df = pd.DataFrame({"label": [0, 1]})
    scores = np.array([0.1, 0.9, 0.5])
    errors = validate_scores_match_pairs(scores, df)
    assert len(errors) == 1


def test_validate_threshold_valid():
    assert validate_threshold(0.5) == []


def test_validate_threshold_out_of_range():
    errors = validate_threshold(2.0)
    assert len(errors) == 1


def test_validate_config_valid():
    cfg = {
        "seed": 42,
        "data": {},
        "split_policy": {"train": 0.7, "val": 0.15, "test": 0.15},
        "pairs": {},
        "manifest": {},
        "benchmark": {},
    }
    assert validate_config(cfg) == []


def test_validate_config_missing_key():
    cfg = {"seed": 42}
    errors = validate_config(cfg)
    assert len(errors) > 0


def test_validate_no_split_leakage_clean():
    splits = {"train": ["a", "b"], "val": ["c"], "test": ["d"]}
    assert validate_no_split_leakage(splits) == []


def test_validate_no_split_leakage_overlap():
    splits = {"train": ["a", "b"], "val": ["b", "c"]}
    errors = validate_no_split_leakage(splits)
    assert len(errors) == 1


# ── error analysis ─────────────────────────────────────────────────────────────

def _make_results_df():
    pairs = pd.DataFrame({
        "left_path": ["a", "b", "c", "d"],
        "right_path": ["e", "f", "g", "h"],
        "label": [1, 0, 1, 0],
        "split": ["val"] * 4,
    })
    scores = np.array([0.8, 0.7, 0.3, 0.2])
    return build_results_df(pairs, scores, threshold=0.5)


def test_build_results_df_columns():
    df = _make_results_df()
    assert "score" in df.columns
    assert "pred" in df.columns
    assert "error_type" in df.columns


def test_slice_false_positives():
    df = _make_results_df()
    fp = slice_false_positives(df)
    assert all(fp["error_type"] == "fp")


def test_slice_false_negatives():
    df = _make_results_df()
    fn = slice_false_negatives(df)
    assert all(fn["error_type"] == "fn")


def test_summarize_slice_empty():
    df = _make_results_df()
    empty = df[df["label"] == 99]
    summary = summarize_slice(empty, "test")
    assert summary["count"] == 0
