from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass

import cv2
import numpy as np

try:
    from skimage.morphology import thin

    SKIMAGE_OK = True
except Exception:
    SKIMAGE_OK = False


"""
usage:
single: python tools/fd_theta_preprocess.py path/to/image.jpg --out fd_theta_out
directory: python tools/fd_theta_preprocess.py path/to/images_dir --out fd_theta_out
with masks: python tools/fd_theta_preprocess.py mo_yolo_dataset/c9.jpg --mask mo_yolo_dataset/c9.png --fd box --orientation pca --save_viz --viz_alpha 0.6
visilization:python tools/fd_theta_preprocess.py path/to/img_or_dir --out fd_theta_out --save_viz --viz_alpha 0.6

``--mask`` may point to either a single mask file or a directory containing
per-image masks whose file names share the same stem as the corresponding
image (e.g., ``foo.jpg`` → ``foo.png``).
"""


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
MASK_EXTENSIONS = IMAGE_EXTENSIONS


@dataclass
class PreprocessConfig:
    window: int = 21
    stride: int = 8
    orientation_method: str = "structure"  # or "pca"
    fd_method: str = "dbc"  # or "box"
    smooth_sigma: float = 3.0
    normalize_mode: str = "linear"  # or "quantile"
    d_min: float = 1.1
    d_max: float = 2.0
    quantiles: tuple[float, float] = (0.01, 0.99)
    thresholds: tuple[float, float] = (0.3, 0.6)
    max_bridge: int = 7  # for simple morphology repair
    canny_low_high: tuple[int, int] = (50, 150)
    save_viz: bool = False
    viz_alpha: float = 0.5


def ensure_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ValueError("Unsupported image shape")


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 1:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def boxes_to_mask(shape: tuple[int, int], boxes_xyxy: list[tuple[int, int, int, int]]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for x1, y1, x2, y2 in boxes_xyxy:
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w - 1, int(x2)), min(h - 1, int(y2))
        mask[y1 : y2 + 1, x1 : x2 + 1] = 255
    return mask


def simple_crack_seg(gray: np.ndarray, roi_mask: np.ndarray | None, canny_low_high=(50, 150)) -> np.ndarray:
    # Edge-based segmentation with morphology cleanup
    edges = cv2.Canny(gray, canny_low_high[0], canny_low_high[1])
    if roi_mask is not None:
        edges = cv2.bitwise_and(edges, edges, mask=roi_mask)
    # Morphological closing to fill small gaps
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=2)
    # Thin to approximate crack centerlines as a segmentation proxy
    seg = closed.copy()
    # Slight dilation then threshold to make a foreground mask
    seg = cv2.dilate(seg, k, iterations=1)
    seg = (seg > 0).astype(np.uint8) * 255
    return seg


def foreground_from_annotations(
    image: np.ndarray,
    label_mask_path: str | None = None,
    yolo_boxes_path: str | None = None,
    cfg: PreprocessConfig | None = None,
) -> np.ndarray:
    gray = ensure_gray(image)
    roi = None
    if label_mask_path and os.path.isfile(label_mask_path):
        mask = cv2.imread(label_mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Failed to read mask: {label_mask_path}")
        mask = cv2.resize(mask, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
        return (mask > 0).astype(np.uint8) * 255

    if yolo_boxes_path and os.path.isfile(yolo_boxes_path):
        # Expect YOLO txt: class cx cy w h normalized
        h, w = gray.shape
        boxes: list[tuple[int, int, int, int]] = []
        with open(yolo_boxes_path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                _, cx, cy, bw, bh = map(float, parts[:5])
                x = int((cx - bw / 2) * w)
                y = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                boxes.append((x, y, x2, y2))
        roi = boxes_to_mask(gray.shape, boxes)

    # If only boxes or nothing: do simple segmentation
    canny = cfg.canny_low_high if cfg else (50, 150)
    return simple_crack_seg(gray, roi, canny)


def skeletonize_mask(mask: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    # Binarize
    bin_mask = (mask > 0).astype(np.uint8)
    # Basic cleanup
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, k, iterations=1)
    # Skeletonize
    if SKIMAGE_OK:
        skel = thin(bin_mask.astype(bool))
        skel = skel.astype(np.uint8)
    else:
        # Fallback: morphological skeletonization via OpenCV loop (approximate)
        skel = np.zeros_like(bin_mask)
        temp = np.zeros_like(bin_mask)
        eroded = bin_mask.copy()
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        while True:
            opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, element)
            temp = cv2.subtract(eroded, opened)
            skel = cv2.bitwise_or(skel, temp)
            eroded = cv2.erode(eroded, element)
            if cv2.countNonZero(eroded) == 0:
                break
    # Simple repair: close tiny gaps
    if cfg.max_bridge > 0:
        bridge_k = cv2.getStructuringElement(cv2.MORPH_RECT, (cfg.max_bridge, 1))
        skel = cv2.morphologyEx(skel, cv2.MORPH_CLOSE, bridge_k, iterations=1)
        bridge_k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, cfg.max_bridge))
        skel = cv2.morphologyEx(skel, cv2.MORPH_CLOSE, bridge_k2, iterations=1)
        skel = (skel > 0).astype(np.uint8)
    return skel


def orientation_structure_tensor(gray: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    gray_f = gray.astype(np.float32) / 255.0
    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    Jxx = gx * gx
    Jxy = gx * gy
    Jyy = gy * gy
    k = int(2 * round(3 * sigma) + 1)
    if k < 3:
        k = 3
    Jxx = cv2.GaussianBlur(Jxx, (k, k), sigma)
    Jxy = cv2.GaussianBlur(Jxy, (k, k), sigma)
    Jyy = cv2.GaussianBlur(Jyy, (k, k), sigma)
    # Orientation (radians) in [-pi/2, pi/2)
    theta = 0.5 * np.arctan2(2.0 * Jxy, (Jxx - Jyy) + 1e-12)
    return theta.astype(np.float32)


def orientation_pca_on_skeleton(skel: np.ndarray, window: int = 21, stride: int = 8) -> np.ndarray:
    h, w = skel.shape
    pad = window // 2
    padded = np.pad(skel, pad, mode="constant")
    theta_map = np.zeros((h, w), dtype=np.float32)
    weight = np.zeros((h, w), dtype=np.float32)
    ys = range(0, h, stride)
    xs = range(0, w, stride)
    for y in ys:
        for x in xs:
            win = padded[y : y + window, x : x + window]
            pts = np.column_stack(np.nonzero(win))
            if pts.shape[0] < 5:
                continue
            # PCA via covariance
            pts = pts.astype(np.float32)
            mean = pts.mean(axis=0, keepdims=True)
            cov = np.cov((pts - mean).T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            v = eigvecs[:, np.argmax(eigvals)]
            angle = math.atan2(float(v[0]), float(v[1]))  # y,x ordering to map to image coords
            # Fill a small block with the angle
            y0 = max(0, y - stride // 2)
            x0 = max(0, x - stride // 2)
            y1 = min(h, y + stride // 2 + 1)
            x1 = min(w, x + stride // 2 + 1)
            theta_map[y0:y1, x0:x1] += angle
            weight[y0:y1, x0:x1] += 1.0
    # Normalize
    mask = weight > 0
    theta_full = np.zeros_like(theta_map)
    theta_full[mask] = theta_map[mask] / weight[mask]
    # For gaps, fallback to nearest filled via interpolation using Gaussian blur
    theta_filled = cv2.GaussianBlur(theta_full, (0, 0), 2.0)
    theta_full[~mask] = theta_filled[~mask]
    # Wrap to [-pi/2, pi/2)
    theta_full = ((theta_full + np.pi / 2) % np.pi) - np.pi / 2
    return theta_full.astype(np.float32)


def _linear_regression_slope(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    A = np.vstack([x, np.ones_like(x)]).T
    # Least squares
    slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(slope)


def fd_box_counting_local(
    binary: np.ndarray, window: int = 33, stride: int = 8, scales: list[int] = (2, 4, 8, 16)
) -> np.ndarray:
    h, w = binary.shape
    pad = window // 2
    bin_pad = np.pad(binary, pad, mode="constant")
    D_map = np.full((h, w), np.nan, dtype=np.float32)
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            patch = bin_pad[y : y + window, x : x + window]
            if patch.size == 0:
                continue
            counts = []
            inv_eps = []
            for s in scales:
                # grid size s x s over the window
                gy, gx = s, s
                cell_h = window / gy
                cell_w = window / gx
                c = 0
                for i in range(s):
                    for j in range(s):
                        y0 = int(i * cell_h)
                        x0 = int(j * cell_w)
                        y1 = int(min(window, (i + 1) * cell_h))
                        x1 = int(min(window, (j + 1) * cell_w))
                        cell = patch[y0:y1, x0:x1]
                        if np.any(cell > 0):
                            c += 1
                if c > 0:
                    counts.append(c)
                    inv_eps.append(s)
            if len(counts) >= 2:
                yv = np.log(np.array(counts) + 1e-9)
                xv = np.log(np.array(inv_eps) + 1e-9)
                slope = _linear_regression_slope(xv, yv)
                D = max(1.0, min(2.5, slope))
            else:
                D = np.nan
            D_map[y : y + stride, x : x + stride] = D
    return D_map


def fd_differential_box_counting_local(
    gray: np.ndarray, window: int = 33, stride: int = 8, scales: list[int] = (2, 4, 8, 16)
) -> np.ndarray:
    # Normalize gray to [0, 255]
    g = ensure_gray(gray)
    g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    h, w = g.shape
    pad = window // 2
    gpad = np.pad(g, pad, mode="reflect")
    D_map = np.full((h, w), np.nan, dtype=np.float32)
    G = 256.0
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            patch = gpad[y : y + window, x : x + window]
            counts = []
            inv_eps = []
            for s in scales:
                # Partition into s x s grid; intensity axis partitioned proportionally
                Hs = window / s
                Ws = window / s
                math.ceil(G / s)
                nboxes = 0
                for i in range(s):
                    for j in range(s):
                        y0 = int(i * Hs)
                        x0 = int(j * Ws)
                        y1 = int(min(window, (i + 1) * Hs))
                        x1 = int(min(window, (j + 1) * Ws))
                        cell = patch[y0:y1, x0:x1]
                        gmin = int(cell.min())
                        gmax = int(cell.max())
                        kmin = math.floor((gmin * s) / G)
                        kmax = math.floor((gmax * s) / G)
                        nboxes += kmax - kmin + 1
                if nboxes > 0:
                    counts.append(nboxes)
                    inv_eps.append(s)
            if len(counts) >= 2:
                yv = np.log(np.array(counts, dtype=np.float64) + 1e-9)
                xv = np.log(np.array(inv_eps, dtype=np.float64) + 1e-9)
                slope = _linear_regression_slope(xv, yv)
                # DBC estimates FD in [2,3] for 3D intensity surface; map to ~[1,2]
                D = max(1.0, min(2.5, slope - 1.0))
            else:
                D = np.nan
            D_map[y : y + stride, x : x + stride] = D
    return D_map


def smooth_and_fill(D_map: np.ndarray, sigma: float) -> np.ndarray:
    # Fill NaNs with local mean via Gaussian
    mask = ~np.isfinite(D_map)
    filled = D_map.copy()
    if np.any(mask):
        # Replace NaN with 0, blur both image and mask, then divide
        tmp = filled.copy()
        tmp[mask] = 0.0
        ksize = max(3, int(2 * round(3 * sigma) + 1))
        blur_num = cv2.GaussianBlur(tmp, (ksize, ksize), sigma)
        blur_den = cv2.GaussianBlur((~mask).astype(np.float32), (ksize, ksize), sigma)
        with np.errstate(divide="ignore", invalid="ignore"):
            est = np.where(blur_den > 1e-6, blur_num / blur_den, 0)
        filled[mask] = est[mask]
    ksize = max(3, int(2 * round(3 * sigma) + 1))
    sm = cv2.GaussianBlur(filled.astype(np.float32), (ksize, ksize), sigma)
    return sm


def normalize_fd(D: np.ndarray, mode: str, d_min: float, d_max: float, q: tuple[float, float]) -> np.ndarray:
    if mode == "quantile":
        lo = np.nanquantile(D, q[0])
        hi = np.nanquantile(D, q[1])
        span = max(1e-6, hi - lo)
        Dn = (D - lo) / span
    else:
        span = max(1e-6, d_max - d_min)
        Dn = (D - d_min) / span
    Dn = np.clip(Dn, 0.0, 1.0)
    return Dn.astype(np.float32)


def colorize_fd_map(Dn: np.ndarray) -> np.ndarray:
    norm = np.nan_to_num(Dn, nan=0.0)
    norm = np.clip(norm, 0.0, 1.0)
    fd_uint8 = (norm * 255).astype(np.uint8)
    return cv2.applyColorMap(fd_uint8, cv2.COLORMAP_TURBO)


def colorize_theta_map(theta: np.ndarray, weight: np.ndarray | None = None) -> np.ndarray:
    theta = np.nan_to_num(theta, nan=0.0)
    theta_norm = ((theta + np.pi / 2.0) % np.pi) / np.pi
    hue = (theta_norm * 179).astype(np.uint8)
    sat = np.full_like(hue, 255, dtype=np.uint8)
    if weight is not None:
        w = np.nan_to_num(weight, nan=0.0)
        val = (np.clip(w, 0.0, 1.0) * 255).astype(np.uint8)
    else:
        val = np.full_like(hue, 255, dtype=np.uint8)
    hsv = cv2.merge([hue, sat, val])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def overlay_with_mask(base: np.ndarray, overlay: np.ndarray, mask: np.ndarray | None, alpha: float) -> np.ndarray:
    base_bgr = ensure_bgr(base).astype(np.float32)
    overlay = ensure_bgr(overlay).astype(np.float32)
    alpha = np.clip(alpha, 0.0, 1.0)
    if mask is not None:
        mask_f = (mask.astype(np.float32) / 255.0)[..., None]
        blended = base_bgr * (1.0 - alpha * mask_f) + overlay * (alpha * mask_f)
    else:
        blended = base_bgr * (1.0 - alpha) + overlay * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def save_visualizations(
    image: np.ndarray,
    fd_norm: np.ndarray,
    theta_map: np.ndarray,
    fg_mask: np.ndarray,
    out_dir: str,
    base: str,
    alpha: float,
):
    fd_color = colorize_fd_map(fd_norm)
    theta_color = colorize_theta_map(theta_map, weight=fd_norm)

    fd_img_path = os.path.join(out_dir, f"{base}_FD_map.png")
    theta_img_path = os.path.join(out_dir, f"{base}_theta_map.png")
    cv2.imwrite(fd_img_path, fd_color)
    cv2.imwrite(theta_img_path, theta_color)

    overlay_base = overlay_with_mask(image, fd_color, fg_mask, alpha)
    overlay_path = os.path.join(out_dir, f"{base}_FD_overlay.png")
    cv2.imwrite(overlay_path, overlay_base)

    theta_overlay = overlay_with_mask(image, theta_color, fg_mask, alpha)
    theta_overlay_path = os.path.join(out_dir, f"{base}_theta_overlay.png")
    cv2.imwrite(theta_overlay_path, theta_overlay)


def build_mask_lookup(mask_dir: str) -> dict[str, str]:
    lookup: dict[str, str] = {}
    try:
        entries = os.listdir(mask_dir)
    except FileNotFoundError:
        return lookup
    for name in entries:
        if not name.lower().endswith(MASK_EXTENSIONS):
            continue
        stem = os.path.splitext(name)[0].lower()
        lookup[stem] = os.path.join(mask_dir, name)
    return lookup


def preprocess_image(
    image_path: str,
    out_dir: str,
    label_mask_path: str | None,
    yolo_boxes_path: str | None,
    cfg: PreprocessConfig,
) -> tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    # 1) Foreground mask
    fg = foreground_from_annotations(img, label_mask_path, yolo_boxes_path, cfg)

    # 2) Skeletonization & light repair
    skel = skeletonize_mask(fg, cfg)

    # 3) Orientation field
    if cfg.orientation_method == "pca":
        theta = orientation_pca_on_skeleton(skel, window=cfg.window, stride=cfg.stride)
    else:
        gray = ensure_gray(img)
        theta = orientation_structure_tensor(gray, sigma=1.0)

    # 4) Fractal dimension map
    if cfg.fd_method == "box":
        D_local = fd_box_counting_local((fg > 0).astype(np.uint8), window=max(33, cfg.window | 1), stride=cfg.stride)
    else:
        D_local = fd_differential_box_counting_local(
            ensure_gray(img), window=max(33, cfg.window | 1), stride=cfg.stride
        )

    # 5) Smooth
    D_sm = smooth_and_fill(D_local, sigma=cfg.smooth_sigma)

    # 6) Normalize and thresholds
    Dn = normalize_fd(D_sm, cfg.normalize_mode, cfg.d_min, cfg.d_max, cfg.quantiles)

    base = os.path.splitext(os.path.basename(image_path))[0]
    fd_path = os.path.join(out_dir, f"{base}_FD_map.npy")
    th_path = os.path.join(out_dir, f"{base}_theta_map.npy")
    np.save(fd_path, Dn)
    np.save(th_path, theta.astype(np.float32))

    if cfg.save_viz:
        save_visualizations(ensure_bgr(img), Dn, theta, fg, out_dir, base, cfg.viz_alpha)
    return fd_path, th_path


def parse_boxes_json(box_json_path: str, image_shape: tuple[int, int]) -> str | None:
    # Convenience: accept a JSON with xyxy boxes and make a temporary YOLO-like TXT
    # JSON structure: [{"x1":int,"y1":int,"x2":int,"y2":int}, ...]
    try:
        with open(box_json_path, encoding="utf-8") as f:
            boxes = json.load(f)
    except Exception:
        return None
    h, w = image_shape
    lines = []
    for b in boxes:
        x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
        cx = ((x1 + x2) / 2) / w
        cy = ((y1 + y2) / 2) / h
        bw = abs(x2 - x1) / w
        bh = abs(y2 - y1) / h
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
    tmp_path = box_json_path + ".txt.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    return tmp_path


def main():
    ap = argparse.ArgumentParser(description="Offline preprocessing: FD and orientation maps")
    ap.add_argument("input", help="Image file or directory")
    ap.add_argument("--out", default="fd_theta_out", help="Output directory")
    ap.add_argument("--mask", default=None, help="Optional mask path (same size as image)")
    ap.add_argument("--yolo_boxes", default=None, help="Optional YOLO .txt boxes (normalized)")
    ap.add_argument("--boxes_json", default=None, help="Optional JSON with xyxy boxes (alternative)")
    ap.add_argument("--window", type=int, default=21)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--orientation", choices=["structure", "pca"], default="structure")
    ap.add_argument("--fd", choices=["dbc", "box"], default="dbc")
    ap.add_argument("--sigma", type=float, default=3.0, help="Smoothing sigma for FD map")
    ap.add_argument("--norm", choices=["linear", "quantile"], default="linear")
    ap.add_argument("--dmin", type=float, default=1.1)
    ap.add_argument("--dmax", type=float, default=2.0)
    ap.add_argument("--qlo", type=float, default=0.01)
    ap.add_argument("--qhi", type=float, default=0.99)
    ap.add_argument("--canny", type=int, nargs=2, default=[50, 150])
    ap.add_argument("--save_viz", action="store_true", help="Write PNG previews for FD/theta maps")
    ap.add_argument("--viz_alpha", type=float, default=0.5, help="Overlay strength for previews")
    args = ap.parse_args()

    cfg = PreprocessConfig(
        window=args.window,
        stride=args.stride,
        orientation_method=args.orientation,
        fd_method=args.fd,
        smooth_sigma=args.sigma,
        normalize_mode=args.norm,
        d_min=args.dmin,
        d_max=args.dmax,
        quantiles=(args.qlo, args.qhi),
        canny_low_high=tuple(args.canny),
        save_viz=args.save_viz,
        viz_alpha=args.viz_alpha,
    )

    inputs: list[str] = []
    if os.path.isdir(args.input):
        for name in os.listdir(args.input):
            if name.lower().endswith(IMAGE_EXTENSIONS):
                inputs.append(os.path.join(args.input, name))
    else:
        inputs.append(args.input)

    os.makedirs(args.out, exist_ok=True)

    mask_file = args.mask if args.mask and os.path.isfile(args.mask) else None
    mask_dir = args.mask if args.mask and os.path.isdir(args.mask) else None
    mask_lookup = build_mask_lookup(mask_dir) if mask_dir else {}

    for img_path in inputs:
        mask_path = mask_file
        if mask_dir:
            stem = os.path.splitext(os.path.basename(img_path))[0].lower()
            mask_path = mask_lookup.get(stem)
            if mask_path is None:
                print(f"Warning: mask for '{img_path}' not found in directory '{mask_dir}'")
        boxes_path = args.yolo_boxes

        if args.boxes_json and os.path.isfile(args.boxes_json) and not boxes_path:
            # Create temporary YOLO-like file from JSON
            dummy_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if dummy_img is not None:
                boxes_path = parse_boxes_json(args.boxes_json, dummy_img.shape)

        fd_p, th_p = preprocess_image(img_path, args.out, mask_path, boxes_path, cfg)
        print(f"Saved: {fd_p} | {th_p}")


if __name__ == "__main__":
    main()
