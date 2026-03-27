"""
Evaluation metrics for face verification.
"""

import numpy as np
from typing import Tuple, Dict


def compute_roc(scores: np.ndarray, labels: np.ndarray, thresholds: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    tprs, fprs = [], []
    for t in thresholds:
        preds = (scores >= t).astype(int)
        tp = np.sum((preds == 1) & (labels == 1))
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))
        tn = np.sum((preds == 0) & (labels == 0))
        tprs.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        fprs.append(fp / (fp + tn) if (fp + tn) > 0 else 0.0)
    return np.array(fprs), np.array(tprs), thresholds


def compute_confusion_matrix(scores: np.ndarray, labels: np.ndarray, threshold: float) -> Dict[str, int]:
    preds = (scores >= threshold).astype(int)
    tp = int(np.sum((preds == 1) & (labels == 1)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def compute_metrics_at_threshold(scores: np.ndarray, labels: np.ndarray, threshold: float) -> Dict[str, float]:
    cm = compute_confusion_matrix(scores, labels, threshold)
    tp, fp, fn, tn = cm["tp"], cm["fp"], cm["fn"], cm["tn"]
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    balanced_acc = (tpr + (1 - fpr)) / 2
    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tpr": tpr,
        "fpr": fpr,
        **cm,
    }


def select_threshold_max_balanced_accuracy(scores: np.ndarray, labels: np.ndarray, thresholds: np.ndarray) -> float:
    best_t, best_ba = thresholds[0], -1.0
    for t in thresholds:
        m = compute_metrics_at_threshold(scores, labels, t)
        if m["balanced_accuracy"] > best_ba:
            best_ba = m["balanced_accuracy"]
            best_t = t
    return float(best_t)
