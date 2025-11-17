#!/usr/bin/env python3
"""Generate per-box prior FD features from FD/theta maps.

This script converts pixel-level FD and theta maps into per-ground-truth box
features suitable for training losses that operate on oriented bounding boxes.

The script scans a YOLO-style oriented bounding box dataset with the following
structure:

- Images:            ``lxdata/test/images/train``
- Label files:       ``lxdata/test_fd/labels/train`` (YOLO OBB format with 4 points)
- FD/theta map files ``lxdata/test_fd/fd_theta_maps/train`` containing ``*_fd.npy``
  and ``*_theta.npy`` arrays with matching spatial resolution.

For each label file, the script reads the ground-truth oriented boxes, locates
box centers within the FD/theta maps, and samples a pair ``(theta_prior, fd_norm)``
from the pixel containing the center. The results are saved to
``lxdata/test_fd/train`` as ``*_prior_fd.npy`` arrays of shape ``[N, 2]`` where
``N`` is the number of boxes in the image.

Example usage::

    python tools/generate_prior_fd.py \
        --label-dir lxdata/test_fd/labels/train \
        --map-dir lxdata/test_fd/fd_theta_maps/train \
        --out-dir lxdata/test_fd/train

The generated ``*_prior_fd.npy`` files can be loaded with ``numpy.load`` and
aligned with the corresponding label rows for downstream processing.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Iterable
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate (theta_prior, fd_norm) features for each GT box.")
    parser.add_argument(
        "--label-dir",
        type=Path,
        default=Path("lxdata/test_fd/labels/train"),
        help="Directory containing YOLO OBB label files (*.txt).",
    )
    parser.add_argument(
        "--map-dir",
        type=Path,
        default=Path("lxdata/test_fd/fd_theta_maps/train"),
        help="Directory containing *_fd.npy and *_theta.npy maps.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("lxdata/test_fd/train"),
        help="Directory where *_prior_fd.npy files will be written.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("lxdata/test/images/train"),
        help="Optional image directory (unused but validated for completeness).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing *_prior_fd.npy files if present.",
    )
    return parser.parse_args()


def ensure_directories(*paths: Iterable[Path]) -> None:
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {path}")
        if not path.is_dir():
            raise NotADirectoryError(f"Expected directory but found: {path}")


def load_labels(label_path: Path) -> np.ndarray:
    data = np.loadtxt(label_path, dtype=np.float32, ndmin=2)
    if data.size == 0:
        return data.reshape(0, 9)
    if data.shape[1] != 9:
        raise ValueError(f"Label file {label_path} has {data.shape[1]} columns, expected 9.")
    return data


def sample_prior_features(labels: np.ndarray, fd_map: np.ndarray, theta_map: np.ndarray) -> np.ndarray:
    if labels.size == 0:
        return np.zeros((0, 2), dtype=np.float32)

    pts = labels[:, 1:].reshape(-1, 4, 2)
    H, W = fd_map.shape

    if theta_map.shape != (H, W):
        raise ValueError(
            f"FD map and theta map must have the same spatial dimensions, got {fd_map.shape} and {theta_map.shape}."
        )

    centers = pts.mean(axis=1)
    cx = np.clip(np.rint(centers[:, 0] * W), 0, W - 1).astype(np.int64)
    cy = np.clip(np.rint(centers[:, 1] * H), 0, H - 1).astype(np.int64)

    fd_values = fd_map[cy, cx].astype(np.float32)
    theta_values = theta_map[cy, cx].astype(np.float32)

    return np.stack([theta_values, fd_values], axis=1)


def generate_prior_fd(label_dir: Path, map_dir: Path, out_dir: Path, overwrite: bool = False) -> tuple[int, int]:
    os.makedirs(out_dir, exist_ok=True)

    label_files = sorted(p for p in label_dir.iterdir() if p.suffix == ".txt")
    processed = 0
    skipped = 0

    for label_path in label_files:
        stem = label_path.stem
        fd_path = map_dir / f"{stem}_fd.npy"
        theta_path = map_dir / f"{stem}_theta.npy"
        out_path = out_dir / f"{stem}_prior_fd.npy"

        if not fd_path.exists() or not theta_path.exists():
            print(f"[WARN] Missing map(s) for {stem}, skipping.")
            skipped += 1
            continue

        if out_path.exists() and not overwrite:
            print(f"[SKIP] {out_path} exists. Use --overwrite to regenerate.")
            skipped += 1
            continue

        labels = load_labels(label_path)
        fd_map = np.load(fd_path)
        theta_map = np.load(theta_path)

        prior_features = sample_prior_features(labels, fd_map, theta_map)
        np.save(out_path, prior_features)
        processed += 1
        print(f"[OK] Saved {out_path} with shape {prior_features.shape}.")

    return processed, skipped


def write_usage_file(out_dir: Path) -> None:
    usage_path = out_dir / "README_prior_fd_usage.txt"
    content = """Per-box prior FD feature files
===============================

Each ``*_prior_fd.npy`` file contains an array of shape ``[N, 2]`` where ``N`` is
identical to the number of rows in the matching YOLO OBB label file.

Column meanings::

    [0] -> theta_prior (radians, typically in the range (-pi/2, pi/2])
    [1] -> fd_norm     (normalized FD value in [0, 1])

Usage example in Python::

    import numpy as np

    prior_fd = np.load("example_prior_fd.npy")
    theta_prior = prior_fd[:, 0]
    fd_norm = prior_fd[:, 1]

The ordering of rows aligns exactly with the labels in the corresponding
``*.txt`` file, enabling direct pairing with ground-truth boxes during training.
"""
    with open(usage_path, "w", encoding="utf-8") as fh:
        fh.write(content)


def main() -> None:
    args = parse_args()
    ensure_directories(args.label_dir, args.map_dir)
    if args.image_dir:
        ensure_directories(args.image_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    processed, skipped = generate_prior_fd(
        label_dir=args.label_dir,
        map_dir=args.map_dir,
        out_dir=args.out_dir,
        overwrite=args.overwrite,
    )

    write_usage_file(args.out_dir)

    print(f"\nSummary: processed {processed} files, skipped {skipped}. Output directory: {args.out_dir}")
    print(f"Usage instructions saved to {args.out_dir / 'README_prior_fd_usage.txt'}")


if __name__ == "__main__":
    main()
