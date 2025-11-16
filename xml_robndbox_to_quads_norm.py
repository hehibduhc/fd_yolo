#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, math, sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET

'''
usage:
python xml_robndbox_to_quads_norm.py \
  --src /path/to/xml_root \
  --out /path/to/out_dir \
  --classes /path/to/classes.txt \
  --normalize
需要xml文件是弧度，如果是度的话用以下代码
python xml_robndbox_to_quads_norm.py \
  --src /path/to/xml_root \
  --out /path/to/out_dir \
  --angle-deg --reverse-angle --normalize
'''

def load_name_to_index(mapping_path: Optional[Path]) -> Optional[Dict[str, int]]:
    if mapping_path is None:
        return None
    if not mapping_path.exists():
        raise FileNotFoundError(f"Mapping file not found: {mapping_path}")
    if mapping_path.suffix.lower() == ".json":
        return json.loads(mapping_path.read_text(encoding="utf-8"))
    names = [ln.strip() for ln in mapping_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return {name: i for i, name in enumerate(names)}

def read_image_size(root: ET.Element) -> Tuple[Optional[int], Optional[int]]:
    size = root.find("size")
    if size is None: return None, None
    try:
        w = int(float(size.findtext("width", default="0"))); h = int(float(size.findtext("height", default="0")))
        if w <= 0 or h <= 0: return None, None
        return w, h
    except Exception:
        return None, None

def robndbox_to_quad(cx: float, cy: float, w: float, h: float, angle: float,
                     angle_in_degrees: bool=False, reverse_angle: bool=False) -> List[float]:
    if angle_in_degrees: angle = math.radians(angle)
    if reverse_angle: angle = -angle
    cw, sw = math.cos(angle), math.sin(angle)
    dx, dy = w/2.0, h/2.0
    pts = [(-dx,-dy),(dx,-dy),(dx,dy),(-dx,dy)]
    quad: List[float] = []
    for x, y in pts:
        xr = cx + x*cw - y*sw
        yr = cy + x*sw + y*cw
        quad.extend([xr, yr])
    return quad

def parse_one_xml(xml_path: Path,
                  name_to_index: Optional[Dict[str, int]] = None,
                  default_unknown: int = -1,
                  strict_unknown: bool = False,
                  angle_in_degrees: bool=False,
                  reverse_angle: bool=False,
                  normalize: bool=False,
                  default_wh: Optional[Tuple[int,int]] = None) -> List[str]:
    try:
        tree = ET.parse(xml_path)
    except Exception as e:
        print(f"[WARN] Failed to parse XML: {xml_path} ({e})", file=sys.stderr)
        return []
    root = tree.getroot()
    W, H = read_image_size(root)
    if normalize and (W is None or H is None):
        if default_wh is None:
            print(f"[WARN] {xml_path.name}: <size> missing; normalization requested but no --default-wh provided. Skipping normalization.", file=sys.stderr)
            normalize_local = False
        else:
            W, H = default_wh
            normalize_local = True
    else:
        normalize_local = normalize

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
            continue
        try:
            cx = float(r.findtext("cx")); cy = float(r.findtext("cy"))
            w  = float(r.findtext("w"));  h  = float(r.findtext("h"))
            ang = float(r.findtext("angle"))
        except Exception as e:
            print(f"[WARN] Skip invalid robndbox in {xml_path.name}: {e}", file=sys.stderr)
            continue

        quad = robndbox_to_quad(cx, cy, w, h, ang, angle_in_degrees, reverse_angle)

        if normalize_local and W and H:
            quad = [ (v/W if i%2==0 else v/H) for i, v in enumerate(quad) ]

        lines.append(f"{idx} " + " ".join(f"{v:.6f}" for v in quad))
    return lines

def write_lines(dst_path: Path, lines: List[str]) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_path.write_text("\n".join(lines), encoding="utf-8")

def convert_dir_xml_to_quad_txt(src_dir: Path,
                                out_dir: Path,
                                classes_path: Optional[Path] = None,
                                angle_in_degrees: bool=False,
                                reverse_angle: bool=False,
                                normalize: bool=False,
                                default_wh: Optional[Tuple[int,int]] = None,
                                mirror_subdirs: bool=True,
                                default_unknown: int=-1,
                                strict_unknown: bool=False,
                                single_file: Optional[str]=None):
    src_dir = src_dir.resolve(); out_dir = out_dir.resolve(); out_dir.mkdir(parents=True, exist_ok=True)
    name_to_index = load_name_to_index(classes_path) if classes_path else None
    results = []; merged = []
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
                              angle_in_degrees, reverse_angle, normalize, default_wh)
        if single_file: merged.extend(lines)
        else: write_lines(dst, lines)
        results.append((xml_path, dst, len(lines)))
    if single_file: write_lines(out_dir / single_file, merged)
    return results

def build_argparser():
    p = argparse.ArgumentParser(description="Convert VOC XML <robndbox> to normalized quads: 'class_index x1 y1 x2 y2 x3 y3 x4 y4'")
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--classes", type=Path, default=None)
    p.add_argument("--angle-deg", action="store_true")
    p.add_argument("--reverse-angle", action="store_true")
    p.add_argument("--normalize", action="store_true")
    p.add_argument("--default-wh", type=int, nargs=2, metavar=("W","H"), default=None)
    p.add_argument("--no-mirror", action="store_true")
    p.add_argument("--default-unknown", type=int, default=-1)
    p.add_argument("--strict-unknown", action="store_true")
    p.add_argument("--single-file", type=str, default=None)
    return p

def main():
    args = build_argparser().parse_args()
    default_wh = tuple(args.default_wh) if args.default_wh else None
    res = convert_dir_xml_to_quad_txt(
        src_dir=args.src, out_dir=args.out, classes_path=args.classes,
        angle_in_degrees=args.angle_deg, reverse_angle=args.reverse_angle,
        normalize=args.normalize, default_wh=default_wh,
        mirror_subdirs=not args.no_mirror, default_unknown=args.default_unknown,
        strict_unknown=args.strict_unknown, single_file=args.single_file,
    )
    total_xml = len(res); total_objs = sum(n for _,_,n in res)
    print(f"Converted {total_xml} XML files, total {total_objs} objects.")
    for xml_path, dst_path, n in res[:10]:
        print(f"- {xml_path} -> {dst_path} ({n} objects)")
    if total_xml > 10: print(f"... ({total_xml-10} more)" )

if __name__ == "__main__":
    main()
