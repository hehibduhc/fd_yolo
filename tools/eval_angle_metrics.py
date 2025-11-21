"""Angle error evaluation for oriented bounding boxes (OBB).

This script follows the workflow described in the user instructions:

1.  Load YOLO-format ground-truth labels that store class followed by the
    quadrilateral coordinates ``x1 y1 x2 y2 x3 y3 x4 y4``.  The coordinates are
    converted to ``[cx, cy, w, h, theta]`` via
    :func:`ultralytics.utils.ops.xyxyxyxy2xywhr`.
2.  Load prediction files that contain ``cx cy w h theta conf cls`` per line.
3.  Match predictions to ground truths by greedy IoU (``IoU > 0.5``) while
    keeping a one-to-one assignment.
4.  Report the angular MAE and RMSE only on matched (true-positive) pairs.  The
    angular difference is wrapped to ``(-pi, pi]`` to respect the periodicity of
    rotation angles.

Usage example::

    python tools/eval_angle_metrics.py \
        --labels /path/to/val/labels \
        --preds /path/to/preds_a /path/to/preds_b \
        --iou-thresh 0.5 --conf-thresh 0.01

Adjust the paths to your dataset.  Prediction files must share the stem of the
label files (``0001.txt`` → ``0001.txt``).
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch

from ultralytics.utils.metrics import batch_probiou
from ultralytics.utils.ops import xyxyxyxy2xywhr


def _safe_loadtxt(path: Path, expected_cols: int) -> np.ndarray:
    """Load a text file into a 2D ``np.ndarray`` with ``expected_cols`` columns."""
    if not path.exists():
        return np.zeros((0, expected_cols), dtype=np.float32)

    try:
        data = np.loadtxt(path, dtype=np.float32, ndmin=2)
    except ValueError:  # Empty file
        return np.zeros((0, expected_cols), dtype=np.float32)

    if data.size == 0:
        return np.zeros((0, expected_cols), dtype=np.float32)

    return data


def load_gt_boxes(label_path: Path) -> np.ndarray:
    """Return ``[N, 6]`` arrays with ``cx, cy, w, h, theta, cls``."""
    arr = _safe_loadtxt(label_path, expected_cols=9)
    if arr.size == 0:
        return np.zeros((0, 6), dtype=np.float32)

    cls = arr[:, 0:1]
    quads = arr[:, 1:9]
    quads_tensor = torch.from_numpy(quads.reshape(-1, 8))
    xywhr = xyxyxyxy2xywhr(quads_tensor).numpy()
    return np.concatenate([xywhr, cls], axis=1)


def load_pred_boxes(pred_path: Path, conf_thres: float) -> np.ndarray:
    """Return predictions sorted by confidence (``[M, 7]``)."""
    arr = _safe_loadtxt(pred_path, expected_cols=7)
    if arr.size == 0:
        return np.zeros((0, 7), dtype=np.float32)

    arr = arr[arr[:, 5] >= conf_thres]
    if arr.size == 0:
        return np.zeros((0, 7), dtype=np.float32)

    order = np.argsort(-arr[:, 5])
    return arr[order]


def obb_iou_xywhr(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute IoU between two ``xywhr`` OBBs using probabilistic IoU."""
    if box1 is None or box2 is None:
        return 0.0

    b1 = np.asarray(box1[:5], dtype=np.float32).reshape(1, 5)
    b2 = np.asarray(box2[:5], dtype=np.float32).reshape(1, 5)
    return float(batch_probiou(b1, b2)[0, 0])


def match_image(gt_boxes: np.ndarray, pred_boxes: np.ndarray, iou_thresh: float) -> list[float]:
    """Return angle differences (radians) for matched prediction/GT pairs."""
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return []

    gt_used = np.zeros(len(gt_boxes), dtype=bool)
    matched_dtheta: list[float] = []

    for pred in pred_boxes:
        best_iou = 0.0
        best_idx = -1

        for idx, gt in enumerate(gt_boxes):
            if gt_used[idx]:
                continue

            if pred.shape[-1] == 7 and gt.shape[-1] == 6:
                cls_pred = pred[-1]
                cls_gt = gt[-1]
                if not math.isclose(cls_pred, cls_gt):
                    continue

            iou = obb_iou_xywhr(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx

        if best_idx >= 0 and best_iou >= iou_thresh:
            dtheta = pred[4] - gt_boxes[best_idx][4]
            dtheta = math.atan2(math.sin(dtheta), math.cos(dtheta))
            matched_dtheta.append(dtheta)
            gt_used[best_idx] = True

    return matched_dtheta


def eval_model(
    labels_dir: Path,
    preds_dir: Path,
    iou_thresh: float,
    conf_thres: float,
) -> tuple[int, int, list[float]]:
    """Evaluate a prediction directory and return (n_gt, n_pred, dtheta list)."""
    label_files = sorted(p for p in labels_dir.glob("*.txt"))
    all_dtheta: list[float] = []
    total_preds = 0
    total_gt = 0

    for label_path in label_files:
        pred_path = preds_dir / label_path.name
        gt_boxes = load_gt_boxes(label_path)
        pred_boxes = load_pred_boxes(pred_path, conf_thres=conf_thres)

        total_gt += len(gt_boxes)
        total_preds += len(pred_boxes)

        all_dtheta.extend(match_image(gt_boxes, pred_boxes, iou_thresh))

    return total_gt, total_preds, all_dtheta


def summarize(pred_name: str, total_gt: int, total_preds: int, dtheta: Iterable[float]) -> None:
    """Print summary metrics for one prediction directory."""
    dtheta = np.array(list(dtheta), dtype=np.float32)
    print(f"=== {pred_name} ===")
    print(f"Total GT: {total_gt}, Total predictions: {total_preds}, Matched pairs: {len(dtheta)}")
    if dtheta.size == 0:
        print("No matched pairs (IoU threshold too high or missing predictions).\n")
        return

    mae = float(np.mean(np.abs(dtheta)))
    rmse = float(np.sqrt(np.mean(np.square(dtheta))))
    print(f"Angle MAE: {mae:.6f} rad")
    print(f"Angle RMSE: {rmse:.6f} rad\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Angle error evaluation for OBB predictions")
    parser.add_argument("--labels", type=Path, required=True, help="Directory that contains GT label txt files")
    parser.add_argument(
        "--preds",
        type=Path,
        nargs="+",
        required=True,
        help="One or more directories with prediction txt files",
    )
    parser.add_argument("--iou-thresh", type=float, default=0.5, help="IoU threshold to accept a match")
    parser.add_argument("--conf-thresh", type=float, default=0.01, help="Confidence threshold for predictions")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    for pred_dir in args.preds:
        total_gt, total_preds, dtheta = eval_model(
            labels_dir=args.labels,
            preds_dir=pred_dir,
            iou_thresh=args.iou_thresh,
            conf_thres=args.conf_thresh,
        )
        summarize(pred_dir.name, total_gt, total_preds, dtheta)


if __name__ == "__main__":
    main()
