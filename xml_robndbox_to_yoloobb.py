#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from xml.etree import ElementTree as ET


def load_name_to_index(mapping_path: Path | None):
    if mapping_path is None:
        return None
    if not mapping_path.exists():
        raise FileNotFoundError(f"Mapping file not found: {mapping_path}")
    if mapping_path.suffix.lower() == ".json":
        return json.loads(mapping_path.read_text(encoding="utf-8"))
    names = [ln.strip() for ln in mapping_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return {name: i for i, name in enumerate(names)}


def read_image_size(root: ET.Element):
    size = root.find("size")
    if size is None:
        return None, None
    try:
        w = int(float(size.findtext("width", default="0")))
        h = int(float(size.findtext("height", default="0")))
        if w <= 0 or h <= 0:
            return None, None
        return w, h
    except Exception:
        return None, None


def parse_one_xml(
    xml_path: Path,
    name_to_index=None,
    default_unknown=-1,
    strict_unknown=False,
    angle_in_degrees_in_xml=False,
    reverse_angle_sign=False,
    normalize=False,
    default_wh=None,
    emit_degrees=False,
):
    try:
        tree = ET.parse(xml_path)
    except Exception as e:
        print(f"[WARN] Failed to parse XML: {xml_path} ({e})", file=sys.stderr)
        return []
    root = tree.getroot()
    W, H = read_image_size(root)
    if normalize and (W is None or H is None):
        if default_wh is None:
            print(
                f"[WARN] {xml_path.name}: <size> missing; normalization requested but no --default-wh provided. Skipping normalization.",
                file=sys.stderr,
            )
            normalize_local = False
        else:
            W, H = default_wh
            normalize_local = True
    else:
        normalize_local = normalize
    lines = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "unknown").strip()
        if name_to_index is None:
            cls_id = default_unknown
        else:
            if name not in name_to_index:
                if strict_unknown:
                    raise KeyError(f"Unknown class '{name}' in {xml_path}")
                cls_id = default_unknown
            else:
                cls_id = name_to_index[name]
        r = obj.find("robndbox")
        if r is None:
            continue
        try:
            cx = float(r.findtext("cx"))
            cy = float(r.findtext("cy"))
            w = float(r.findtext("w"))
            h = float(r.findtext("h"))
            ang = float(r.findtext("angle"))
        except Exception as e:
            print(f"[WARN] Skip invalid robndbox in {xml_path.name}: {e}", file=sys.stderr)
            continue
        if normalize_local and W and H:
            cx_out, cy_out, w_out, h_out = cx / W, cy / H, w / W, h / H
        else:
            cx_out, cy_out, w_out, h_out = cx, cy, w, h
        if angle_in_degrees_in_xml:
            ang = math.radians(ang)
        if reverse_angle_sign:
            ang = -ang
        r_out = math.degrees(ang) if emit_degrees else ang
        lines.append(f"{cls_id} {cx_out:.6f} {cy_out:.6f} {w_out:.6f} {h_out:.6f} {r_out:.6f}")
    return lines


def write_lines(dst_path: Path, lines):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_path.write_text("\n".join(lines), encoding="utf-8")


def convert_dir_xml_to_obb_txt(
    src_dir: Path,
    out_dir: Path,
    classes_path=None,
    angle_in_degrees_in_xml=False,
    reverse_angle_sign=False,
    normalize=False,
    default_wh=None,
    emit_degrees=False,
    mirror_subdirs=True,
    default_unknown=-1,
    strict_unknown=False,
    single_file=None,
):
    src_dir = src_dir.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    name_to_index = load_name_to_index(classes_path) if classes_path else None
    results = []
    merged = []
    for xml_path in sorted(src_dir.rglob("*.xml")):
        dst = (
            (out_dir / (xml_path.relative_to(src_dir).with_suffix(".txt")))
            if (not single_file and mirror_subdirs)
            else (out_dir / (xml_path.stem + ".txt") if not single_file else out_dir / single_file)
        )
        lines = parse_one_xml(
            xml_path,
            name_to_index,
            default_unknown,
            strict_unknown,
            angle_in_degrees_in_xml,
            reverse_angle_sign,
            normalize,
            default_wh,
            emit_degrees,
        )
        if single_file:
            merged.extend(lines)
        else:
            write_lines(dst, lines)
        results.append((xml_path, dst, len(lines)))
    if single_file:
        write_lines(out_dir / single_file, merged)
    return results


def build_argparser():
    p = argparse.ArgumentParser(description="Convert VOC XML <robndbox> to 'class_id x y w h r'")
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--classes", type=Path, default=None)
    p.add_argument("--angle-deg", action="store_true")
    p.add_argument("--reverse-angle", action="store_true")
    p.add_argument("--emit-deg", action="store_true")
    p.add_argument("--normalize", action="store_true")
    p.add_argument("--default-wh", type=int, nargs=2, metavar=("W", "H"), default=None)
    p.add_argument("--no-mirror", action="store_true")
    p.add_argument("--default-unknown", type=int, default=-1)
    p.add_argument("--strict-unknown", action="store_true")
    p.add_argument("--single-file", type=str, default=None)
    return p


def main():
    args = build_argparser().parse_args()
    default_wh = tuple(args.default_wh) if args.default_wh else None
    res = convert_dir_xml_to_obb_txt(
        src_dir=args.src,
        out_dir=args.out,
        classes_path=args.classes,
        angle_in_degrees_in_xml=args.angle_deg,
        reverse_angle_sign=args.reverse_angle,
        normalize=args.normalize,
        default_wh=default_wh,
        emit_degrees=args.emit_deg,
        mirror_subdirs=not args.no_mirror,
        default_unknown=args.default_unknown,
        strict_unknown=args.strict_unknown,
        single_file=args.single_file,
    )
    total_xml = len(res)
    total_objs = sum(n for _, _, n in res)
    print(f"Converted {total_xml} XML files, total {total_objs} objects.")
    for xml_path, dst_path, n in res[:10]:
        print(f"- {xml_path} -> {dst_path} ({n} objects)")
    if total_xml > 10:
        print(f"... ({total_xml - 10} more)")


if __name__ == "__main__":
    main()
