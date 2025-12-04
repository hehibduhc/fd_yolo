"""Compute fractal dimensions and geometry stats for GT boxes and write augmented labels.

This utility scans a directory of crack *mask* images and the corresponding
YOLO-style label files, computes the fractal dimension (FD) for the crack
region inside each ground-truth box, along with the aspect ratio (AR) and
orientation stability (local orientation variance) for the crack mask, and
writes a new label file with all three metrics appended per line. The
computation is performed directly on the provided binary/gray masks instead of
thresholding the raw RGB images, avoiding the additional errors introduced by
image preprocessing while keeping the "FD–几何形态" relationship measurable.

Supported label formats
-----------------------
- Polygon/quadrilateral: ``cls x1 y1 x2 y2 x3 y3 x4 y4`` (normalized or
  absolute pixels).
- Rotated box: ``cls cx cy w h theta`` where ``theta`` is in radians by
  default. Set ``--theta-in-deg`` if the angle is stored in degrees.
- Axis-aligned box: ``cls cx cy w h`` (theta assumed to be 0).

Example:
-------
python tools/compute_fd_from_labels.py \
    --masks path/to/masks \
    --labels path/to/labels \
    --output path/to/output_labels
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
from skimage.morphology import skeletonize
from sklearn.decomposition import PCA


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute FD, aspect ratio, and orientation stability for each GT box and append to labels",
    )
    parser.add_argument("--masks", type=Path, required=True, help="Directory containing crack mask images")
    parser.add_argument("--labels", type=Path, required=True, help="Directory containing label txt files")
    parser.add_argument("--output", type=Path, required=True, help="Directory to write augmented labels")
    parser.add_argument(
        "--theta-in-deg",
        action="store_true",
        help="Set if rotated-box theta values are stored in degrees instead of radians",
    )
    parser.add_argument(
        "--min-box-size",
        type=int,
        default=12,
        help="Skip FD calculation for boxes whose shorter side is below this pixel threshold",
    )
    return parser.parse_args()


# Geometry helpers ---------------------------------------------------------------------------------


def _maybe_denormalize(points: np.ndarray, width: int, height: int) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if pts.max() <= 1.5:  # Heuristic: coordinates are normalized to [0,1]
        if pts.ndim == 1 and pts.size == 4:
            scale = np.array([width, height, width, height], dtype=np.float32)
        else:
            scale = np.array([width, height], dtype=np.float32)
        pts = pts * scale
    return pts


def polygon_from_label(values: np.ndarray, img_w: int, img_h: int, theta_in_deg: bool) -> np.ndarray:
    if values.size == 8:  # Polygon/quadrilateral
        pts = values.reshape(-1, 2)
        pts = _maybe_denormalize(pts, img_w, img_h)
        return pts.astype(np.float32)

    if values.size in {4, 5, 6}:  # cx cy w h [theta]
        cx, cy, w, h = values[:4]
        if values.size == 6:
            theta_raw = values[4]
            theta_in = float(theta_raw) if theta_in_deg else math.degrees(float(theta_raw))
        elif values.size == 5:
            theta_raw = values[4]
            theta_in = float(theta_raw) if theta_in_deg else math.degrees(float(theta_raw))
        else:
            theta_in = 0.0

        cx, cy, w, h = _maybe_denormalize(np.array([cx, cy, w, h], dtype=np.float32), img_w, img_h)
        rect = ((float(cx), float(cy)), (float(w), float(h)), float(theta_in))
        pts = cv2.boxPoints(rect)
        return pts.astype(np.float32)

    raise ValueError(f"Unsupported label with {values.size} numeric values")


# Fractal dimension helpers ------------------------------------------------------------------------


def box_count_fractal_dimension(binary: np.ndarray) -> float:
    foreground = binary.astype(bool)
    if not foreground.any():
        return float("nan")

    h, w = foreground.shape
    max_scale = int(math.log2(min(h, w)))
    scales: list[int] = []
    counts: list[int] = []

    for exp in range(1, max_scale + 1):
        box = 2**exp
        tiles_y = math.ceil(h / box)
        tiles_x = math.ceil(w / box)
        count = 0
        for ty in range(tiles_y):
            y0, y1 = ty * box, min((ty + 1) * box, h)
            for tx in range(tiles_x):
                x0, x1 = tx * box, min((tx + 1) * box, w)
                if foreground[y0:y1, x0:x1].any():
                    count += 1
        if count > 0:
            scales.append(box)
            counts.append(count)

    if len(scales) < 2:
        return float("nan")

    log_scales = np.log(1.0 / np.array(scales, dtype=np.float32))
    log_counts = np.log(np.array(counts, dtype=np.float32))
    slope, _ = np.polyfit(log_scales, log_counts, 1)
    return float(slope)


def orientation_stability(mask: np.ndarray, sample_num: int = 200) -> float:
    """Estimate local orientation variance on a skeletonized mask."""
    sk = skeletonize(mask > 0)
    pts = np.argwhere(sk > 0)
    if len(pts) < 20:
        return 0.0

    idx = np.random.choice(len(pts), min(sample_num, len(pts)), replace=False)
    thetas: list[float] = []

    for i in idx:
        y, x = pts[i]
        # Neighborhood points within radius 5
        win = pts[np.linalg.norm(pts - np.array([y, x]), axis=1) < 5]
        if len(win) < 5:
            continue
        pca = PCA(n_components=2).fit(win)
        v = pca.components_[0]
        theta = math.atan2(v[1], v[0])
        thetas.append(theta)

    if len(thetas) == 0:
        return 0.0

    return float(np.var(thetas))


def compute_geometry_for_patch(
    mask_img: np.ndarray, polygon: np.ndarray, args: argparse.Namespace
) -> tuple[float, float, float]:
    """Return FD, AR, theta variance for a polygon patch."""
    poly_int = np.round(polygon).astype(np.int32)
    img_h, img_w = mask_img.shape[:2]

    poly_clipped = poly_int.copy()
    poly_clipped[:, 0] = np.clip(poly_clipped[:, 0], 0, img_w - 1)
    poly_clipped[:, 1] = np.clip(poly_clipped[:, 1], 0, img_h - 1)

    x, y, w, h = cv2.boundingRect(poly_clipped)
    if min(w, h) < args.min_box_size or w == 0 or h == 0:
        return float("nan"), float("nan"), 0.0

    ar = float(max(w, h) / min(w, h)) if min(w, h) > 0 else float("nan")

    patch = mask_img[y : y + h, x : x + w]
    mask = np.zeros((h, w), dtype=np.uint8)
    shifted = np.ascontiguousarray(poly_clipped - np.array([x, y], dtype=np.int32))
    cv2.fillPoly(mask, [shifted], 255)

    if mask.sum() == 0:
        return float("nan"), ar, 0.0

    masked = cv2.bitwise_and(patch, patch, mask=mask)
    _, binary = cv2.threshold(masked, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.bitwise_and(binary, mask)

    fd = box_count_fractal_dimension(binary > 0)
    theta_var = orientation_stability(binary)
    return fd, ar, theta_var


# IO pipeline --------------------------------------------------------------------------------------


def load_labels(label_path: Path) -> list[str]:
    text = label_path.read_text().strip().splitlines()
    return [line.strip() for line in text if line.strip()]


def parse_label_numbers(line: str) -> tuple[str, np.ndarray]:
    parts = line.split()
    if len(parts) < 5:
        raise ValueError(f"Invalid label line (needs >=5 entries): {line}")
    cls, coords = parts[0], np.array([float(x) for x in parts[1:]], dtype=np.float32)
    return cls, coords


def process_single_image(mask_path: Path, label_path: Path, args: argparse.Namespace) -> list[str]:
    mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        raise FileNotFoundError(f"Failed to read mask image: {mask_path}")

    h, w = mask_img.shape[:2]
    out_lines: list[str] = []

    for line in load_labels(label_path):
        _cls, coords = parse_label_numbers(line)
        polygon = polygon_from_label(coords, w, h, theta_in_deg=args.theta_in_deg)
        fd, ar, theta_var = compute_geometry_for_patch(mask_img, polygon, args)
        out_lines.append(f"{line} {fd:.6f} {ar:.6f} {theta_var:.6f}")

    return out_lines


# Entrypoint ---------------------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    label_files = sorted(Path(args.labels).glob("*.txt"))
    if not label_files:
        raise FileNotFoundError(f"No label files found in {args.labels}")

    for label_path in label_files:
        stem = label_path.stem
        mask_candidates = list(Path(args.masks).glob(f"{stem}.*"))
        if not mask_candidates:
            raise FileNotFoundError(f"No mask image found for label {label_path}")

        mask_path = mask_candidates[0]
        out_lines = process_single_image(mask_path, label_path, args)
        out_path = args.output / label_path.name
        out_path.write_text("\n".join(out_lines) + "\n")
        print(f"Processed {label_path.name}: {len(out_lines)} boxes")


if __name__ == "__main__":
    main()
