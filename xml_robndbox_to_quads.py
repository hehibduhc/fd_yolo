#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XML robndbox -> quads converter

Converts VOC-style XML files containing <robndbox>
(cx, cy, w, h, angle) into text files with lines:
"class_index x1 y1 x2 y2 x3 y3 x4 y4"

Features
--------
- Recursive scan of a source directory for *.xml
- Per-XML .txt output (mirrors source subdirs), or a single merged output file
- Class mapping via classes.txt (one name per line) OR mapping.json {"name": index}
- Angle in degrees or radians, and optional reverse sign (clockwise vs. counter-clockwise)
- Configurable default index for unknown classes, or strict mode to fail on unknowns
- Deterministic corner order TL -> TR -> BR -> BL (local rect), then rotated about (cx, cy)

Usage
-----
# Per-XML outputs, mirroring subdirs under out_dir
python xml_robndbox_to_quads.py \
  --src /path/to/xml_root \
  --out /path/to/out_dir \
  --classes /path/to/classes.txt \
  --angle-deg \
  --reverse-angle

# Single merged output file
python xml_robndbox_to_quads.py \
  --src /path/to/xml_root \
  --out /path/to/out_dir \
  --classes /path/to/mapping.json \
  --single-file merged.txt

Notes
-----
- Angle convention matters. If results look rotated the wrong way,
  try toggling --angle-deg and/or --reverse-angle.
- Output coordinates are in the same image coordinate system as the XML (y downwards typical for images).
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET

def load_name_to_index(mapping_path: Optional[Path]) -> Optional[Dict[str, int]]:
    """Load class mapping from a file.
    - .json -> {"name": index}
    - others -> plain text, one class name per line, index = line number
    """
    if mapping_path is None:
        return None
    if not mapping_path.exists():
        raise FileNotFoundError(f"Mapping file not found: {mapping_path}")
    if mapping_path.suffix.lower() == ".json":
        return json.loads(mapping_path.read_text(encoding="utf-8"))
    # plain text
    names = [ln.strip() for ln in mapping_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return {name: i for i, name in enumerate(names)}

def robndbox_to_quad(cx: float, cy: float, w: float, h: float, angle: float,
                     angle_in_degrees: bool=False, reverse_angle: bool=False) -> List[float]:
    """Return 8 numbers: TL(x,y), TR(x,y), BR(x,y), BL(x,y) from (cx,cy,w,h,angle)."""
    if angle_in_degrees:
        angle = math.radians(angle)
    if reverse_angle:
        angle = -angle
    cw, sw = math.cos(angle), math.sin(angle)
    dx, dy = w / 2.0, h / 2.0
    # local TL, TR, BR, BL (mathematical coords, then mapped to image by rotation and translation)
    pts = [(-dx, -dy), ( dx, -dy), ( dx,  dy), (-dx,  dy)]
    quad: List[float] = []
    for x, y in pts:
        xr = cx + x * cw - y * sw
        yr = cy + x * sw + y * cw
        quad.extend([xr, yr])
    return quad

def parse_one_xml(xml_path: Path,
                  name_to_index: Optional[Dict[str, int]] = None,
                  default_unknown: int = -1,
                  strict_unknown: bool = False,
                  angle_in_degrees: bool=False,
                  reverse_angle: bool=False) -> List[str]:
    """Parse one XML and return lines in 'class_index x1 y1 x2 y2 x3 y3 x4 y4' format."""
    try:
        tree = ET.parse(xml_path)
    except Exception as e:
        print(f"[WARN] Failed to parse XML: {xml_path} ({e})", file=sys.stderr)
        return []
    root = tree.getroot()
    lines: List[str] = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "unknown").strip()
        if name_to_index is None:
            idx = default_unknown
        else:
            if name not in name_to_index:
                if strict_unknown:
                    raise KeyError(f"Unknown class '{name}' in {xml_path}")
                idx = default_unknown
            else:
                idx = name_to_index[name]

        r = obj.find("robndbox")
        if r is None:
            # skip if not a rotated object
            continue
        try:
            cx = float(r.findtext("cx")); cy = float(r.findtext("cy"))
            w  = float(r.findtext("w"));  h  = float(r.findtext("h"))
            ang = float(r.findtext("angle"))
        except Exception as e:
            print(f"[WARN] Skip invalid robndbox in {xml_path.name}: {e}", file=sys.stderr)
            continue
        quad = robndbox_to_quad(cx, cy, w, h, ang, angle_in_degrees, reverse_angle)
        line = f"{idx} " + " ".join(f"{v:.6f}" for v in quad)
        lines.append(line)
    return lines

def write_lines(dst_path: Path, lines: List[str]) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_path.write_text("\n".join(lines), encoding="utf-8")

def convert_dir_xml_to_quad_txt(src_dir: Path,
                                out_dir: Path,
                                classes_path: Optional[Path] = None,
                                angle_in_degrees: bool=False,
                                reverse_angle: bool=False,
                                mirror_subdirs: bool=True,
                                default_unknown: int=-1,
                                strict_unknown: bool=False,
                                single_file: Optional[str]=None) -> List[Tuple[Path, Path, int]]:
    """Batch convert all *.xml in src_dir recursively to quad txt in out_dir.
       If single_file is provided, write all lines into out_dir/single_file.
       Returns a list of (xml_path, txt_path, num_lines)."""
    src_dir = src_dir.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    name_to_index = load_name_to_index(classes_path) if classes_path else None
    results: List[Tuple[Path, Path, int]] = []
    merged_lines: List[str] = []

    for xml_path in sorted(src_dir.rglob("*.xml")):
        if single_file:
            dst = out_dir / single_file
        else:
            if mirror_subdirs:
                rel = xml_path.relative_to(src_dir)
                dst = out_dir / rel.with_suffix(".txt")
            else:
                dst = out_dir / (xml_path.stem + ".txt")

        lines = parse_one_xml(xml_path, name_to_index, default_unknown, strict_unknown,
                              angle_in_degrees, reverse_angle)
        if single_file:
            merged_lines.extend(lines)
        else:
            write_lines(dst, lines)
        results.append((xml_path, dst, len(lines)))

    if single_file:
        write_lines(out_dir / single_file, merged_lines)
    return results

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Batch convert VOC XML with <robndbox> to quad txt files.")
    p.add_argument("--src", type=Path, required=True, help="Source root directory containing XML files")
    p.add_argument("--out", type=Path, required=True, help="Output directory")
    p.add_argument("--classes", type=Path, default=None,
                   help="Path to classes.txt (one name per line) or mapping.json (name->index). Optional")
    p.add_argument("--angle-deg", action="store_true",
                   help="Set if <angle> values are in DEGREES (default: radians)")
    p.add_argument("--reverse-angle", action="store_true",
                   help="Reverse the angle sign (use if your angles are clockwise)")
    p.add_argument("--no-mirror", action="store_true",
                   help="Do not mirror subdirectories; flatten outputs into --out")
    p.add_argument("--default-unknown", type=int, default=-1,
                   help="Class index to use for unmapped/unknown names (default: -1)")
    p.add_argument("--strict-unknown", action="store_true",
                   help="Fail on unknown class names (default: False)")
    p.add_argument("--single-file", type=str, default=None,
                   help="If set, write all outputs into a single file under --out/<single-file>")
    return p

def main():
    args = build_argparser().parse_args()
    res = convert_dir_xml_to_quad_txt(
        src_dir=args.src,
        out_dir=args.out,
        classes_path=args.classes,
        angle_in_degrees=args.angle_deg,
        reverse_angle=args.reverse_angle,
        mirror_subdirs=not args.no_mirror,
        default_unknown=args.default_unknown,
        strict_unknown=args.strict_unknown,
        single_file=args.single_file,
    )
    total_xml = len(res)
    total_objs = sum(n for _, _, n in res)
    print(f"Converted {total_xml} XML files, total {total_objs} objects.")
    # Show a few samples
    shown = 0
    for xml_path, dst_path, n in res[:10]:
        print(f"- {xml_path} -> {dst_path} ({n} objects)")
        shown += 1
    if total_xml > shown:
        print(f"... ({total_xml - shown} more)")

if __name__ == "__main__":
    main()
