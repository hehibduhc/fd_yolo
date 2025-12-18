#!/usr/bin/env python3
"""Compute skeleton-based fractal dimension for OBB cracks.

This script processes oriented bounding box (OBB) annotations and corresponding
crack mask images to compute a skeleton-based fractal dimension (FD) for each
GT instance. Results are written to a matching output text file.
"""
from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np
from skimage import morphology
from skimage.morphology import skeletonize
from sklearn.linear_model import RANSACRegressor, TheilSenRegressor
from sklearn.metrics import r2_score

# ---------------------------- Geometry helpers -----------------------------


def normalize_theta(theta: float) -> float:
    """Normalize theta to radians.

    If |theta| > ~pi, interpret the input as degrees and convert.
    """
    if abs(theta) > 3.2:  # close to pi
        return math.radians(theta)
    return theta


def obb_to_polygon(cx: float, cy: float, w: float, h: float, theta: float) -> np.ndarray:
    """Convert OBB to 4-point polygon (float32).

    Points are returned in clockwise order starting from top-left after rotation.
    """
    theta = normalize_theta(theta)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    hw, hh = w / 2.0, h / 2.0
    corners = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]], dtype=np.float32)
    rotation = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)
    rotated = corners @ rotation.T
    rotated[:, 0] += cx
    rotated[:, 1] += cy
    return rotated


# ---------------------------- Mask utilities ------------------------------


def rasterize_polygon(polygon: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Rasterize polygon into a binary mask of given (H, W)."""
    h, w = shape
    poly_int = np.round(polygon).astype(np.int32)
    poly_int[:, 0] = np.clip(poly_int[:, 0], 0, w - 1)
    poly_int[:, 1] = np.clip(poly_int[:, 1], 0, h - 1)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly_int], 1)
    return mask.astype(bool)


def extract_obb_crack(mask: np.ndarray, polygon: np.ndarray, min_component: int = 20) -> np.ndarray:
    """Extract crack pixels within OBB polygon and clean small components."""
    if mask.dtype != np.uint8:
        mask_bin = (mask > 0).astype(np.uint8)
    else:
        mask_bin = (mask > 0).astype(np.uint8)
    poly_mask = rasterize_polygon(polygon, mask_bin.shape)
    clipped = (mask_bin.astype(bool) & poly_mask).astype(bool)
    if min_component > 0:
        clipped = morphology.remove_small_objects(clipped, min_size=min_component)
    return clipped


def crop_to_content(binary_mask: np.ndarray) -> np.ndarray:
    """Crop binary mask to minimal bounding box containing True pixels."""
    coords = np.argwhere(binary_mask)
    if coords.size == 0:
        return np.zeros((0, 0), dtype=bool)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return binary_mask[y0:y1, x0:x1]


# ------------------------ Fractal dimension logic -------------------------


def box_counting(skel: np.ndarray, size: int) -> int:
    """Count number of boxes of given size containing at least one pixel."""
    h, w = skel.shape
    pad_h = (size - h % size) % size
    pad_w = (size - w % size) % size
    if pad_h or pad_w:
        skel = np.pad(skel, ((0, pad_h), (0, pad_w)), mode="constant")
    new_h, new_w = skel.shape
    reshaped = skel.reshape(new_h // size, size, new_w // size, size)
    box_sum = reshaped.any(axis=(1, 3))
    return int(box_sum.sum())


def compute_fd_from_skeleton(skel: np.ndarray, random_state: int = 42) -> Tuple[float, int, float]:
    """Compute fractal dimension from skeleton mask.

    Returns (fd, num_scales_used, r2).
    """
    if skel.size == 0 or skel.sum() == 0:
        return -1.0, 0, 0.0

    skel = crop_to_content(skel)
    h, w = skel.shape
    min_dim = min(h, w)
    if min_dim < 2:
        return -1.0, 0, 0.0

    max_power = int(math.floor(math.log(min_dim, 2)))
    sizes = [2 ** p for p in range(1, max_power + 1)]
    if not sizes:
        return -1.0, 0, 0.0

    counts = []
    valid_sizes = []
    for s in sizes:
        c = box_counting(skel, s)
        if c > 1:  # ignore trivial counts
            valid_sizes.append(s)
            counts.append(c)

    if len(valid_sizes) < 2:
        return -1.0, 0, 0.0

    # Discard largest scales if saturation occurs
    while len(valid_sizes) > 2 and counts[-1] <= counts[-2]:
        valid_sizes.pop()
        counts.pop()

    # Discard smallest scale if too close to pixel noise
    if len(valid_sizes) > 2 and valid_sizes[0] <= 2:
        valid_sizes = valid_sizes[1:]
        counts = counts[1:]

    if len(valid_sizes) < 2:
        return -1.0, 0, 0.0

    log_sizes = np.log(valid_sizes).reshape(-1, 1)
    log_counts = np.log(counts)

    fd = -1.0
    r2_val = 0.0

    try:
        model = TheilSenRegressor(random_state=random_state, fit_intercept=True)
        model.fit(log_sizes, log_counts)
        pred = model.predict(log_sizes)
        r2_val = r2_score(log_counts, pred)
        fd = -model.coef_[0]
    except Exception as exc:  # noqa: BLE001
        logging.warning("Theil-Sen regression failed: %s", exc)
        try:
            ransac = RANSACRegressor(random_state=random_state)
            ransac.fit(log_sizes, log_counts)
            pred = ransac.predict(log_sizes)
            r2_val = r2_score(log_counts, pred)
            fd = -ransac.estimator_.coef_[0]
        except Exception as exc2:  # noqa: BLE001
            logging.warning("RANSAC regression failed: %s", exc2)
            coeffs = np.polyfit(log_sizes.flatten(), log_counts, 1)
            fd = -coeffs[0]
            pred = np.polyval(coeffs, log_sizes)
            r2_val = r2_score(log_counts, pred)

    return float(fd), len(valid_sizes), float(r2_val)


# ------------------------------ Main logic --------------------------------


def parse_label_file(label_path: Path) -> List[Tuple[str, float, float, float, float, float]]:
    """Parse label file lines into tuples."""
    entries = []
    with label_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 6:
                logging.warning("Skipping malformed line in %s: %s", label_path, line)
                continue
            cls = parts[0]
            cx, cy, w, h, theta = map(float, parts[1:6])
            entries.append((cls, cx, cy, w, h, theta))
    return entries


def read_mask(mask_path: Path) -> np.ndarray:
    """Read mask image as binary array (uint8 0/1)."""
    img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read mask: {mask_path}")
    return (img > 0).astype(np.uint8)


def process_label_file(
    label_path: Path,
    mask_dir: Path,
    out_dir: Path,
    min_component: int,
    min_skeleton_points: int,
) -> Tuple[int, int]:
    """Process a single label file. Returns (#instances, #neg1)."""
    base = label_path.stem
    mask_candidates = list(mask_dir.glob(base + ".*"))
    if not mask_candidates:
        raise FileNotFoundError(f"No mask found for {base} in {mask_dir}")
    mask_path = sorted(mask_candidates)[0]
    mask = read_mask(mask_path)
    entries = parse_label_file(label_path)
    out_lines: List[str] = []
    neg_count = 0
    for cls, cx, cy, w, h, theta in entries:
        polygon = obb_to_polygon(cx, cy, w, h, theta)
        clipped = extract_obb_crack(mask, polygon, min_component=min_component)
        if clipped.sum() == 0:
            fd, num_scales, r2_val = -1.0, 0, 0.0
            neg_count += 1
            logging.warning("No crack pixels inside OBB for %s", base)
        else:
            skel = skeletonize(clipped)
            if skel.sum() < min_skeleton_points:
                fd, num_scales, r2_val = -1.0, 0, 0.0
                neg_count += 1
                logging.warning(
                    "Insufficient skeleton pixels (%d) for %s", skel.sum(), base
                )
            else:
                fd, num_scales, r2_val = compute_fd_from_skeleton(skel)
                if fd < 0:
                    neg_count += 1
                    logging.warning(
                        "FD computation failed (scales=%d, r2=%.3f) for %s",
                        num_scales,
                        r2_val,
                        base,
                    )
        out_line = f"{cls} {cx:.4f} {cy:.4f} {w:.4f} {h:.4f} {theta:.6f} {fd:.6f}"
        out_lines.append(out_line)

    out_file = out_dir / label_path.name
    with out_file.open("w") as f:
        f.write("\n".join(out_lines) + "\n")
    return len(entries), neg_count


def process_all(labels_dir: Path, masks_dir: Path, out_dir: Path,
                min_component: int = 20, min_skeleton_points: int = 30) -> None:
    label_files = sorted(labels_dir.glob("*.txt"))
    total_files = 0
    total_instances = 0
    total_neg = 0
    out_dir.mkdir(parents=True, exist_ok=True)

    for label_path in label_files:
        total_files += 1
        instances, neg = process_label_file(
            label_path,
            masks_dir,
            out_dir,
            min_component=min_component,
            min_skeleton_points=min_skeleton_points,
        )
        total_instances += instances
        total_neg += neg

    logging.info(
        "Processed %d label files, %d instances, %d returned -1",
        total_files,
        total_instances,
        total_neg,
    )


# ------------------------------- CLI setup --------------------------------


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute skeleton-based fractal dimension for crack OBB labels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--masks", required=True, type=Path, help="Directory of mask images")
    parser.add_argument("--labels", required=True, type=Path, help="Directory of label txt files")
    parser.add_argument("--out", required=True, type=Path, help="Output directory for FD txt files")
    parser.add_argument("--min-component", type=int, default=20, help="Minimum component size to keep")
    parser.add_argument(
        "--min-skeleton-points",
        type=int,
        default=30,
        help="Minimum number of skeleton pixels required to compute FD",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    return parser


def main(args: Sequence[str] | None = None) -> None:
    parser = build_argparser()
    parsed = parser.parse_args(args=args)
    logging.basicConfig(
        level=getattr(logging, parsed.log_level),
        format="%(levelname)s:%(message)s",
    )
    np.random.seed(42)

    process_all(
        labels_dir=parsed.labels,
        masks_dir=parsed.masks,
        out_dir=parsed.out,
        min_component=parsed.min_component,
        min_skeleton_points=parsed.min_skeleton_points,
    )


if __name__ == "__main__":
    main()
