"""
Build a region JSON schema (bbox_norm + label_rect_pts) from template.json.

Grouping: one region per ``placement.zone``, union of element footprints in
label-normalised [0,1] coordinates. Does not read or modify ground_truth files
beyond the *template_json* path the caller opens.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Set, Tuple

# Canonical zone name → region name in config/regions conventions
_ZONE_TO_REGION_NAME: Dict[str, str] = {
    "left_identity": "left_identity",
    "center_barcode": "center_barcode",
    "center_address": "center_address",
    "right_compliance": "right_compliance",
    "right_patents": "right_patents",
    "right_legal_symbols": "right_legal_symbols",
    "right_date": "right_date_logo",
    "right_copy_count": "right_optional_strip",
    "far_right_strip": "far_right_strip",
}

_POSITION_RE = re.compile(
    r"relative\s+position\s*\(\s*([-0-9.Ee+]+)\s*,\s*([-0-9.Ee+]+)\s*\)",
    re.IGNORECASE,
)

_BARCODE_TYPES: Set[str] = {"barcode", "datamatrix"}
_TEXT_TYPES: Set[str] = {"text", "composite_block", "symbol_block"}
_IMAGE_TYPES: Set[str] = {"image"}


def map_zone_to_region_name(zone: str) -> str:
    z = (zone or "").strip()
    return _ZONE_TO_REGION_NAME.get(z, _sanitize_zone_name(z))


def _sanitize_zone_name(zone: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", zone.strip()).strip("_").lower()
    return s or "zone_unknown"


def _parse_label_rect_pts(raw: Any) -> Optional[Tuple[float, float, float, float]]:
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)) and len(raw) == 4:
        try:
            a, b, c, d = (float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]))
        except (TypeError, ValueError):
            return None
    elif isinstance(raw, dict):
        try:
            a = float(raw["x0"])
            b = float(raw["y0"])
            c = float(raw["x1"])
            d = float(raw["y1"])
        except (KeyError, TypeError, ValueError):
            return None
    else:
        return None
    if not (a < c and b < d):
        return None
    return (a, b, c, d)


def _label_mm(template: Dict[str, Any]) -> Tuple[float, float]:
    ls = template.get("label_size_mm") or {}
    try:
        w = float(ls.get("width_mm", 0.0))
        h = float(ls.get("height_mm", 0.0))
    except (TypeError, ValueError):
        w, h = 0.0, 0.0
    if w <= 0 or h <= 0:
        raise ValueError("template.json: label_size_mm.width_mm/height_mm must be positive")
    return w, h


def _element_mm_size(el: Dict[str, Any]) -> Tuple[float, float]:
    spec = el.get("size_spec") or {}
    w = spec.get("width_mm_approx", spec.get("width_mm", 4.0))
    h = spec.get("height_mm_approx", spec.get("height_mm", 2.0))
    try:
        fw, fh = float(w), float(h)
    except (TypeError, ValueError):
        fw, fh = 4.0, 2.0
    return max(fw, 0.1), max(fh, 0.1)


def _parse_position_notes(notes: Any) -> Optional[Tuple[float, float]]:
    if not isinstance(notes, str):
        return None
    m = _POSITION_RE.search(notes)
    if not m:
        return None
    try:
        return float(m.group(1)), float(m.group(2))
    except ValueError:
        return None


def _element_norm_rect(
    el: Dict[str, Any],
    label_w_mm: float,
    label_h_mm: float,
) -> Optional[Tuple[float, float, float, float]]:
    """
    Return (x, y, w, h) in label-normalised coordinates [0,1] or None if
    position cannot be inferred.
    """
    placement = el.get("placement") or {}
    notes = placement.get("position_notes")
    pos = _parse_position_notes(notes)
    if pos is None:
        return None
    rx, ry = pos
    w_mm, h_mm = _element_mm_size(el)
    w_norm = w_mm / label_w_mm
    h_norm = h_mm / label_h_mm
    align = str(placement.get("alignment", "left")).lower()

    if align == "center":
        x0 = rx - w_norm / 2.0
        y0 = ry - h_norm / 2.0
    elif align in ("right", "end"):
        x0 = rx - w_norm
        y0 = ry
    else:
        x0 = rx
        y0 = ry

    return (x0, y0, w_norm, h_norm)


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _union_norm_rects(rects: List[Tuple[float, float, float, float]]) -> Tuple[float, float, float, float]:
    """Union of axis-aligned rects as (x, y, w, h), clamped to label bounds."""
    if not rects:
        return (0.0, 0.0, 0.05, 0.05)
    x0 = min(r[0] for r in rects)
    y0 = min(r[1] for r in rects)
    x1 = max(r[0] + r[2] for r in rects)
    y1 = max(r[1] + r[3] for r in rects)
    x0, y0 = _clamp01(x0), _clamp01(y0)
    x1, y1 = _clamp01(x1), _clamp01(y1)
    if x1 <= x0:
        x1 = min(1.0, x0 + 0.05)
    if y1 <= y0:
        y1 = min(1.0, y0 + 0.05)
    w, h = x1 - x0, y1 - y0
    min_frac = 0.03
    if w < min_frac:
        pad = (min_frac - w) / 2.0
        x0 = _clamp01(x0 - pad)
        x1 = _clamp01(x0 + min_frac)
        w = x1 - x0
    if h < min_frac:
        pad = (min_frac - h) / 2.0
        y0 = _clamp01(y0 - pad)
        y1 = _clamp01(y0 + min_frac)
        h = y1 - y0
    return (x0, y0, w, h)


def _zone_region_type(elements: List[Dict[str, Any]], region_name: str) -> str:
    """Pick a single validator type for the unioned zone."""
    types = [str(e.get("type", "")).lower() for e in elements]
    has_barcode = any(t in _BARCODE_TYPES for t in types)
    has_text = any(t in _TEXT_TYPES or t == "composite_block" for t in types)

    if region_name == "center_barcode":
        return "barcode_block" if has_barcode else "text_block"
    if has_barcode:
        return "barcode_block"
    imgs = [e for e in elements if str(e.get("type", "")).lower() in _IMAGE_TYPES]
    if imgs:
        if all(not e.get("required", False) for e in imgs) and not has_text:
            return "optional_image"
        return "logo_block"
    if has_text:
        return "text_block"
    return "free_form_block"


def _zone_mandatory_weight(elements: List[Dict[str, Any]]) -> Tuple[bool, float]:
    if any(e.get("required") is True for e in elements):
        return True, 1.0
    return False, 0.0


def _region_sort_key(name: str) -> Tuple[int, str]:
    order = [
        "left_identity",
        "center_barcode",
        "center_address",
        "right_legal_symbols",
        "right_compliance",
        "right_patents",
        "right_date_logo",
        "right_optional_strip",
        "far_right_strip",
    ]
    try:
        return (order.index(name), name)
    except ValueError:
        return (len(order), name)


def build_region_schema_dict(
    template: Dict[str, Any],
    product_code: str,
    *,
    template_version: str = "",
) -> Dict[str, Any]:
    """
    Return a JSON-serialisable region config dict (product_code, page_size,
    optional label_rect_pts, regions with bbox_norm).
    """
    label_w_mm, label_h_mm = _label_mm(template)
    lr = _parse_label_rect_pts(template.get("label_rect_pts"))
    if lr is None:
        lr_list: Optional[List[float]] = None
    else:
        lr_list = [round(lr[0], 4), round(lr[1], 4), round(lr[2], 4), round(lr[3], 4)]

    # Canonical raster page_size: preserve label aspect ratio (wide landscape → large width).
    base_w = 1000
    page_h = max(1, int(round(base_w * (label_h_mm / label_w_mm))))
    page_w = base_w

    elements = template.get("elements")
    if not isinstance(elements, list):
        raise ValueError("template.json: 'elements' must be a list")

    by_zone: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    for el in elements:
        if not isinstance(el, dict):
            continue
        pl = el.get("placement") or {}
        zone = pl.get("zone")
        if not zone:
            continue
        by_zone[str(zone)].append(el)

    by_rname: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    for zone, els in by_zone.items():
        by_rname[map_zone_to_region_name(zone)].extend(els)

    regions_out: List[Dict[str, Any]] = []
    for rname in sorted(by_rname.keys(), key=_region_sort_key):
        els = by_rname[rname]
        rects: List[Tuple[float, float, float, float]] = []
        for el in els:
            r = _element_norm_rect(el, label_w_mm, label_h_mm)
            if r is not None:
                rects.append(r)
        if not rects:
            continue
        nx, ny, nw, nh = _union_norm_rects(rects)
        mandatory, weight = _zone_mandatory_weight(els)
        rtype = _zone_region_type(els, rname)
        regions_out.append({
            "name":       rname,
            "type":       rtype,
            "bbox_norm":  [round(nx, 4), round(ny, 4), round(nw, 4), round(nh, 4)],
            "mandatory":  mandatory,
            "weight":     weight,
        })

    if not any(r["mandatory"] for r in regions_out):
        raise ValueError(
            "Generated region schema has no mandatory region — check template elements."
        )

    out: Dict[str, Any] = {
        "product_code": product_code,
        "page_size": {"width": page_w, "height": page_h},
        "regions": regions_out,
        "_meta": {
            "generated_from": "template.json",
            "template_id":    template.get("template_id", ""),
            "family":         template.get("family", ""),
            "template_version": template_version,
        },
    }
    if lr_list is not None:
        out["label_rect_pts"] = lr_list
    return out


def load_template_json(path: Path) -> Dict[str, Any]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top-level JSON must be an object")
    return raw


def write_region_schema(path: Path, schema: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
