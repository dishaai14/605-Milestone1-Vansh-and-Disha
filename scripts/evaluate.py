"""
Evaluate the face verification pipeline on a given split.
Runs a threshold sweep, selects the best threshold by balanced accuracy on val,
logs the run, and saves artifacts.

Usage:
  python scripts/evaluate.py --config configs/m1.yaml --split val --sweep
  python scripts/evaluate.py --config configs/m1.yaml --split test --threshold 0.72
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.metrics import (
    compute_roc,
    compute_metrics_at_threshold,
    select_threshold_max_balanced_accuracy,
)
from src.validation import validate_pair_file, validate_scores_match_pairs, assert_valid
from src.tracking import log_run
from src.error_analysis import build_results_df, slice_false_positives, slice_false_negatives, slice_boundary_pairs, summarize_slice, save_error_analysis


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_pairs(pairs_path: str) -> pd.DataFrame:
    errors = validate_pair_file(pairs_path)
    assert_valid(errors, context="pair_file")
    df = pd.read_csv(pairs_path)
    df["label"] = df["label"].astype(int)
    return df


def score_pairs_mock(pairs_df: pd.DataFrame, seed: int = 42) -> np.ndarray:
    """
    Placeholder scorer using random scores.
    Replace with src.scoring.score_pairs(pairs_df, config) once TFDS data is available.
    """
    rng = np.random.default_rng(seed)
    scores = np.where(
        pairs_df["label"].values == 1,
        rng.normal(0.75, 0.15, size=len(pairs_df)),
        rng.normal(0.35, 0.18, size=len(pairs_df)),
    )
    return np.clip(scores, -1.0, 1.0)


def run_sweep(scores: np.ndarray, labels: np.ndarray, n_thresholds: int = 100):
    thresholds = np.linspace(scores.min(), scores.max(), n_thresholds)
    fprs, tprs, _ = compute_roc(scores, labels, thresholds)
    sweep_results = []
    for t, fpr, tpr in zip(thresholds, fprs, tprs):
        m = compute_metrics_at_threshold(scores, labels, t)
        sweep_results.append({"threshold": float(t), **m})
    return sweep_results, thresholds


def save_sweep(sweep_results: list, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(sweep_results, f, indent=2)
    print(f"[eval] Sweep saved to {output_path}")


def save_roc_plot(sweep_results: list, output_path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[eval] matplotlib not available, skipping ROC plot.")
        return
    fprs = [r["fpr"] for r in sweep_results]
    tprs = [r["tpr"] for r in sweep_results]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fprs, tprs, lw=2, label="Verifier")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[eval] ROC plot saved to {output_path}")


def save_confusion_matrix_plot(cm: dict, threshold: float, output_path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return
    matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred Neg", "Pred Pos"])
    ax.set_yticklabels(["Actual Neg", "Actual Pos"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black", fontsize=14)
    ax.set_title(f"Confusion Matrix (t={threshold:.3f})")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[eval] Confusion matrix plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate face verification pipeline")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--sweep", action="store_true", help="Run threshold sweep and select best threshold")
    parser.add_argument("--threshold", type=float, default=None, help="Fixed threshold to evaluate at")
    parser.add_argument("--data-version", default="v1")
    parser.add_argument("--note", default="")
    parser.add_argument("--use-real-scores", action="store_true", help="Use src.scoring instead of mock scorer")
    args = parser.parse_args()

    config = load_config(args.config)
    pairs_dir = config["pairs"]["output_dir"]
    pairs_path = os.path.join(pairs_dir, f"{args.split}_pairs.csv")
    runs_dir = "outputs/runs"
    eval_base = "outputs/eval" if args.data_version == "v1" else f"outputs/eval_{args.data_version}"
    artifacts_dir = f"{eval_base}/{args.split}"
    os.makedirs(artifacts_dir, exist_ok=True)

    print(f"[eval] Loading pairs from {pairs_path}")
    pairs_df = load_pairs(pairs_path)
    labels = pairs_df["label"].values

    if args.use_real_scores:
        from src.scoring import score_pairs
        print("[eval] Scoring pairs with real pixel-cosine scorer ...")
        scores = score_pairs(pairs_df, config)
    else:
        print("[eval] Using mock scorer (replace with --use-real-scores for real data).")
        scores = score_pairs_mock(pairs_df, seed=config["seed"])

    errors = validate_scores_match_pairs(scores, pairs_df)
    assert_valid(errors, context="scores")

    if args.sweep or args.threshold is None:
        print(f"[eval] Running threshold sweep on {args.split} ...")
        sweep_results, thresholds = run_sweep(scores, labels)
        sweep_path = os.path.join(artifacts_dir, "sweep.json")
        save_sweep(sweep_results, sweep_path)
        roc_path = os.path.join(artifacts_dir, "roc.png")
        save_roc_plot(sweep_results, roc_path)

        selected_threshold = select_threshold_max_balanced_accuracy(scores, labels, thresholds)
        print(f"[eval] Selected threshold (max balanced accuracy): {selected_threshold:.4f}")

        threshold_path = os.path.join(artifacts_dir, "selected_threshold.json")
        with open(threshold_path, "w") as f:
            json.dump({"threshold": selected_threshold, "rule": "max_balanced_accuracy", "split": args.split}, f, indent=2)
    else:
        selected_threshold = args.threshold
        print(f"[eval] Using fixed threshold: {selected_threshold:.4f}")

    metrics = compute_metrics_at_threshold(scores, labels, selected_threshold)

    cm_plot_path = os.path.join(artifacts_dir, "confusion_matrix.png")
    save_confusion_matrix_plot(metrics, selected_threshold, cm_plot_path)

    metrics_path = os.path.join(artifacts_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"threshold": selected_threshold, "split": args.split, **metrics}, f, indent=2)
    print(f"[eval] Metrics: {metrics}")

    # Error analysis
    results_df = build_results_df(pairs_df, scores, selected_threshold)
    fp_slice = summarize_slice(slice_false_positives(results_df), "false_positives")
    fn_slice = summarize_slice(slice_false_negatives(results_df), "false_negatives")
    boundary_slice = summarize_slice(slice_boundary_pairs(results_df, selected_threshold, margin=0.05), "boundary_pairs")
    analysis = {"threshold": selected_threshold, "slices": [fp_slice, fn_slice, boundary_slice]}
    save_error_analysis(analysis, os.path.join(artifacts_dir, "error_analysis.json"))

    run_id = log_run(
        metrics=metrics,
        threshold=selected_threshold,
        split=args.split,
        config_name=os.path.basename(args.config),
        data_version=args.data_version,
        note=args.note,
        runs_dir=runs_dir,
    )
    print(f"[eval] Done. Run ID: {run_id}")


if __name__ == "__main__":
    main()
