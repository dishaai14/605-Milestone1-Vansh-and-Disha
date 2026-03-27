"""
Generate deterministic verification pairs (positive and negative) per split.
Reads the splits produced by ingest_lfw.py and writes CSV pair files.

Data-centric improvements over v1:
  - Cap overrepresented identities (max_images_per_identity) to reduce bias.
  - Filter identities with fewer than min_images_per_identity for positive pairs.
  - Record data_version in pair metadata for traceability.
"""

import argparse
import csv
import json
import os
import random
from collections import defaultdict

import yaml
import tensorflow_datasets as tfds


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_identity_map(config: dict, max_images_per_identity: int = None) -> dict:
    cache_dir = config["data"]["cache_dir"]
    ds = tfds.load(
        config["data"]["tfds_name"],
        split="train",
        data_dir=cache_dir,
        shuffle_files=False,
    )
    identity_map = defaultdict(list)
    for i, record in enumerate(tfds.as_numpy(ds)):
        label = record["label"].decode("utf-8") if isinstance(record["label"], bytes) else str(record["label"])
        identity_map[label].append(f"lfw_record_{label}_{i:06d}")
    for k in identity_map:
        identity_map[k].sort()
        if max_images_per_identity is not None:
            identity_map[k] = identity_map[k][:max_images_per_identity]
    return identity_map


def generate_pairs(
    identity_map: dict,
    id_set: list,
    n_pos: int,
    n_neg: int,
    seed: int,
    split_name: str,
    min_images_for_positive: int = 2,
) -> list:
    rng = random.Random(seed)
    rows = []
    id_set = sorted(id_set)

    eligible = [i for i in id_set if len(identity_map[i]) >= min_images_for_positive]
    pos_count = 0
    rng.shuffle(eligible)
    idx = 0
    while pos_count < n_pos and eligible:
        ident = eligible[idx % len(eligible)]
        imgs = identity_map[ident]
        a, b = rng.sample(imgs, 2)
        rows.append({"left_path": a, "right_path": b, "label": 1, "split": split_name})
        pos_count += 1
        idx += 1

    neg_count = 0
    while neg_count < n_neg:
        id_a, id_b = rng.sample(id_set, 2)
        img_a = rng.choice(identity_map[id_a])
        img_b = rng.choice(identity_map[id_b])
        rows.append({"left_path": img_a, "right_path": img_b, "label": 0, "split": split_name})
        neg_count += 1

    return rows


def main():
    parser = argparse.ArgumentParser(description="Generate verification pairs")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--data-version", default="v1", help="Data version tag for tracking (e.g. v1, v2)")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config["seed"]
    n_pos = config["pairs"]["num_positive_per_split"]
    n_neg = config["pairs"]["num_negative_per_split"]
    output_dir = config["pairs"]["output_dir"]
    manifest_path = config["manifest"]["output_path"]

    max_images = config["pairs"].get("max_images_per_identity", None)
    min_images = config["pairs"].get("min_images_for_positive", 2)

    os.makedirs(output_dir, exist_ok=True)

    splits_path = os.path.join(os.path.dirname(manifest_path), "splits.json")
    with open(splits_path, "r") as f:
        splits = json.load(f)

    print("[pairs] Building identity map from TFDS ...")
    identity_map = build_identity_map(config, max_images_per_identity=max_images)

    fieldnames = ["left_path", "right_path", "label", "split"]
    meta = {"data_version": args.data_version, "max_images_per_identity": max_images, "min_images_for_positive": min_images}

    for split_name, id_list in splits.items():
        split_seed = seed + hash(split_name) % 10000
        rows = generate_pairs(identity_map, id_list, n_pos, n_neg, split_seed, split_name, min_images_for_positive=min_images)

        out_path = os.path.join(output_dir, f"{split_name}_pairs.csv")
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"[pairs] {split_name}: {len(rows)} pairs -> {out_path}")

    meta_path = os.path.join(output_dir, "pairs_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[pairs] Metadata saved to {meta_path}")
    print("[pairs] Done.")


if __name__ == "__main__":
    main()
