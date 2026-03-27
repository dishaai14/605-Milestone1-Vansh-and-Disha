"""
Error analysis and slicing utilities for face verification evaluation.
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def build_results_df(pairs_df: pd.DataFrame, scores: np.ndarray, threshold: float) -> pd.DataFrame:
    df = pairs_df.copy()
    df["score"] = scores
    df["pred"] = (scores >= threshold).astype(int)
    df["correct"] = (df["pred"] == df["label"]).astype(int)
    df["error_type"] = "correct"
    df.loc[(df["pred"] == 1) & (df["label"] == 0), "error_type"] = "fp"
    df.loc[(df["pred"] == 0) & (df["label"] == 1), "error_type"] = "fn"
    return df


def slice_false_positives(results_df: pd.DataFrame) -> pd.DataFrame:
    return results_df[results_df["error_type"] == "fp"].copy()


def slice_false_negatives(results_df: pd.DataFrame) -> pd.DataFrame:
    return results_df[results_df["error_type"] == "fn"].copy()


def slice_boundary_pairs(results_df: pd.DataFrame, threshold: float, margin: float = 0.05) -> pd.DataFrame:
    return results_df[np.abs(results_df["score"] - threshold) <= margin].copy()


def slice_low_image_identities(results_df: pd.DataFrame, identity_counts: Dict[str, int], max_images: int = 3) -> pd.DataFrame:
    low_ids = {k for k, v in identity_counts.items() if v <= max_images}
    mask = results_df["left_identity"].isin(low_ids) | results_df["right_identity"].isin(low_ids)
    return results_df[mask].copy()


def summarize_slice(slice_df: pd.DataFrame, name: str) -> Dict:
    total = len(slice_df)
    if total == 0:
        return {"slice": name, "count": 0}
    return {
        "slice": name,
        "count": total,
        "error_types": slice_df["error_type"].value_counts().to_dict(),
        "mean_score": float(slice_df["score"].mean()),
        "std_score": float(slice_df["score"].std()),
        "label_dist": slice_df["label"].value_counts().to_dict(),
        "examples": slice_df.head(5)[["left_path", "right_path", "label", "score", "pred", "error_type"]].to_dict(orient="records"),
    }


def save_error_analysis(analysis: Dict, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"[error_analysis] Saved to {output_path}")
