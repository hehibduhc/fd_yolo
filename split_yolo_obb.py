#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Split a YOLO OBB dataset into train/val (default 80/20).

Pairs images and labels by filename stem and places them into:
  <out>/images/train, <out>/images/val
  <out>/labels/train, <out>/labels/val

Usage:
  python split_yolo_obb.py \
      --images /path/to/images \
      --labels /path/to/yolo_dataset \
      --out /path/to/output_dataset \
      --val-ratio 0.2 \
      --mode copy      # or move | symlink
      --seed 42 \
      --write-data-yaml --nc 1 --names crack

Notes:
- Images are matched by stem against a .txt label of the same stem in the labels folder.
- Unpaired images or labels are reported and skipped.
- For symlink mode on Windows, you may need admin privileges or Developer Mode.
"""

import argparse
import random
import shutil
from pathlib import Path
from typing import List, Tuple, Dict

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def list_images(images_dir: Path) -> Dict[str, Path]:
    files = {}
    for p in images_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files[p.stem] = p
    return files

def list_labels(labels_dir: Path) -> Dict[str, Path]:
    files = {}
    for p in labels_dir.rglob("*.txt"):
        if p.is_file():
            files[p.stem] = p
    return files

def pair_images_labels(images: Dict[str, Path], labels: Dict[str, Path]) -> Tuple[List[Tuple[Path, Path]], List[Path], List[Path]]:
    pairs = []
    img_only = []
    lbl_only = []
    for stem, ip in images.items():
        if stem in labels:
            pairs.append((ip, labels[stem]))
        else:
            img_only.append(ip)
    for stem, lp in labels.items():
        if stem not in images:
            lbl_only.append(lp)
    return pairs, img_only, lbl_only

def split_pairs(pairs: List[Tuple[Path, Path]], val_ratio: float, seed: int) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
    rnd = random.Random(seed)
    rnd.shuffle(pairs)
    n_total = len(pairs)
    n_val = int(round(n_total * val_ratio))
    val = pairs[:n_val]
    train = pairs[n_val:]
    return train, val

def ensure_dirs(root: Path):
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        (root / sub).mkdir(parents=True, exist_ok=True)

def place_pair(pair: Tuple[Path, Path], dst_root: Path, split: str, mode: str):
    img_src, lbl_src = pair
    img_dst = dst_root / "images" / split / img_src.name
    lbl_dst = dst_root / "labels" / split / (lbl_src.name)
    img_dst.parent.mkdir(parents=True, exist_ok=True)
    lbl_dst.parent.mkdir(parents=True, exist_ok=True)

    if mode == "copy":
        shutil.copy2(img_src, img_dst)
        shutil.copy2(lbl_src, lbl_dst)
    elif mode == "move":
        shutil.move(str(img_src), str(img_dst))
        shutil.move(str(lbl_src), str(lbl_dst))
    elif mode == "symlink":
        if not img_dst.exists():
            img_dst.symlink_to(img_src.resolve())
        if not lbl_dst.exists():
            lbl_dst.symlink_to(lbl_src.resolve())
    else:
        raise ValueError(f"Unknown mode: {mode}")

def write_data_yaml(dst_root: Path, nc: int, names: List[str]):
    yaml_path = dst_root / "data.yaml"
    names_list = "[" + ", ".join(f"'{n}'" for n in names) + "]"
    content = (
        f"path: {dst_root.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: {nc}\n"
        f"names: {names_list}\n"
    )
    yaml_path.write_text(content, encoding="utf-8")
    return yaml_path

def parse_args():
    ap = argparse.ArgumentParser(description="Split YOLO OBB dataset into train/val.")
    ap.add_argument("--images", type=Path, required=True, help="Images root directory")
    ap.add_argument("--labels", type=Path, required=True, help="Labels root directory (YOLO .txt)")
    ap.add_argument("--out", type=Path, required=True, help="Output dataset root directory")
    ap.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio (default 0.2)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--mode", type=str, default="copy", choices=["copy", "move", "symlink"], help="How to place files")
    ap.add_argument("--write-data-yaml", action="store_true", help="Write minimal data.yaml for Ultralytics")
    ap.add_argument("--nc", type=int, default=1, help="Number of classes (for data.yaml)")
    ap.add_argument("--names", type=str, nargs="*", default=["crack"], help="Class names list (for data.yaml)")
    return ap.parse_args()

def main():
    args = parse_args()
    images = list_images(args.images)
    labels = list_labels(args.labels)
    pairs, img_only, lbl_only = pair_images_labels(images, labels)

    if not pairs:
        raise SystemExit("No paired image/label files found. Check your directories and filename matching.")

    train, val = split_pairs(pairs, args.val_ratio, args.seed)

    ensure_dirs(args.out)

    for ip, lp in train:
        place_pair((ip, lp), args.out, "train", args.mode)
    for ip, lp in val:
        place_pair((ip, lp), args.out, "val", args.mode)

    yaml_path = None
    if args.write_data_yaml:
        yaml_path = write_data_yaml(args.out, args.nc, args.names)

    print("=== Split Summary ===")
    print(f"Total paired: {len(pairs)}")
    print(f"Train: {len(train)}  Val: {len(val)}  (val_ratio={args.val_ratio})")
    if img_only:
        print(f"Unpaired images (skipped): {len(img_only)}")
    if lbl_only:
        print(f"Unpaired labels (skipped): {len(lbl_only)}")
    print(f"Mode: {args.mode}")
    print(f"Output root: {args.out}")
    if yaml_path:
        print(f"Wrote data.yaml: {yaml_path}")

if __name__ == "__main__":
    main()
