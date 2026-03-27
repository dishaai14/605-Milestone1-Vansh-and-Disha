"""
Validation checks for pipeline inputs, pair files, and config values.
"""

import os
import csv
from typing import List, Dict, Any


REQUIRED_PAIR_COLUMNS = {"left_path", "right_path", "label", "split"}
VALID_LABELS = {0, 1, "0", "1"}
VALID_SPLITS = {"train", "val", "test"}


def validate_pair_file(csv_path: str, check_paths: bool = False) -> List[str]:
    errors = []
    if not os.path.exists(csv_path):
        return [f"Pair file not found: {csv_path}"]
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
        missing = REQUIRED_PAIR_COLUMNS - cols
        if missing:
            errors.append(f"Missing columns: {missing}")
            return errors
        for i, row in enumerate(reader):
            if row["label"] not in VALID_LABELS:
                errors.append(f"Row {i}: invalid label '{row['label']}'")
            if row["split"] not in VALID_SPLITS:
                errors.append(f"Row {i}: invalid split '{row['split']}'")
            if check_paths:
                for col in ("left_path", "right_path"):
                    if not os.path.exists(row[col]):
                        errors.append(f"Row {i}: path not found '{row[col]}'")
    return errors


def validate_scores_match_pairs(scores, pairs_df) -> List[str]:
    errors = []
    if len(scores) != len(pairs_df):
        errors.append(f"Score count {len(scores)} != pair count {len(pairs_df)}")
    return errors


def validate_threshold(threshold: float, score_min: float = -1.0, score_max: float = 1.0) -> List[str]:
    errors = []
    if not isinstance(threshold, (int, float)):
        errors.append(f"Threshold must be numeric, got {type(threshold)}")
    elif not (score_min <= threshold <= score_max):
        errors.append(f"Threshold {threshold} outside expected range [{score_min}, {score_max}]")
    return errors


def validate_config(config: Dict[str, Any]) -> List[str]:
    errors = []
    required_keys = ["seed", "data", "split_policy", "pairs", "manifest", "benchmark"]
    for k in required_keys:
        if k not in config:
            errors.append(f"Missing config key: {k}")
    if "split_policy" in config:
        sp = config["split_policy"]
        for s in ("train", "val", "test"):
            if s not in sp:
                errors.append(f"Missing split ratio: {s}")
        if all(s in sp for s in ("train", "val", "test")):
            total = sp["train"] + sp["val"] + sp["test"]
            if abs(total - 1.0) > 1e-6:
                errors.append(f"Split ratios sum to {total}, expected 1.0")
    return errors


def validate_no_split_leakage(splits: Dict[str, List]) -> List[str]:
    errors = []
    sets = {k: set(v) for k, v in splits.items()}
    split_names = list(sets.keys())
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            overlap = sets[split_names[i]] & sets[split_names[j]]
            if overlap:
                errors.append(f"Leakage between {split_names[i]} and {split_names[j]}: {len(overlap)} identities")
    return errors


def assert_valid(errors: List[str], context: str = "") -> None:
    if errors:
        prefix = f"[{context}] " if context else ""
        raise ValueError(f"{prefix}Validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
