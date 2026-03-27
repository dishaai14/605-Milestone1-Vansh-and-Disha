"""
Score face pairs using cosine similarity on raw pixel vectors (flattened + normalized).
This is the baseline representation — no learned embeddings yet.
"""

import numpy as np
import pandas as pd
import tensorflow_datasets as tfds
from typing import Dict, Tuple


def _load_lfw_records(config: dict) -> Dict[str, np.ndarray]:
    """Return {record_key: flattened_normalized_vector} for all LFW records."""
    cache_dir = config["data"]["cache_dir"]
    ds = tfds.load(config["data"]["tfds_name"], split="train", data_dir=cache_dir, shuffle_files=False)
    record_map = {}
    for i, record in enumerate(tfds.as_numpy(ds)):
        label = record["label"].decode("utf-8") if isinstance(record["label"], bytes) else str(record["label"])
        key = f"lfw_record_{label}_{i:06d}"
        img = record["image"].astype(np.float64).flatten()
        norm = np.linalg.norm(img)
        record_map[key] = img / norm if norm > 1e-10 else img
    return record_map


def score_pairs(pairs_df: pd.DataFrame, config: dict) -> np.ndarray:
    """
    Given a pairs DataFrame with left_path / right_path columns (record keys),
    returns cosine similarity scores for each pair.
    """
    from src.similarity import cosine_similarity_vectorized

    record_map = _load_lfw_records(config)
    keys = list(record_map.keys())
    key_set = set(keys)

    left_vecs, right_vecs = [], []
    for _, row in pairs_df.iterrows():
        lk, rk = row["left_path"], row["right_path"]
        lv = record_map.get(lk, np.zeros(1))
        rv = record_map.get(rk, np.zeros(1))
        left_vecs.append(lv)
        right_vecs.append(rv)

    # Pad to same length if needed (shouldn't happen with consistent pipeline)
    d = max(v.shape[0] for v in left_vecs + right_vecs)
    def pad(v):
        if v.shape[0] < d:
            return np.concatenate([v, np.zeros(d - v.shape[0])])
        return v

    a = np.stack([pad(v) for v in left_vecs])
    b = np.stack([pad(v) for v in right_vecs])
    return cosine_similarity_vectorized(a, b)
