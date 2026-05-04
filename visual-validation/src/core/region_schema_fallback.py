"""
Coarse ``bbox_norm`` region schemas when ``template.json`` cannot drive the
precision generator (empty elements, missing ``label_size_mm``, etc.).

Uses ``label.pdf`` page dimensions (points) for aspect; optional
``label_rect_pts`` from the template is emitted only if it overlaps ``page.rect``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import fitz  # PyMuPDF
except ImportError as exc:  # pragma: no cover
    raise ImportError("PyMuPDF is required for region_schema_fallback") from exc

from src.core.region_schema_from_template import _parse_label_rect_pts


def _pdf_page_size_pts(pdf_path: Path) -> Tuple[float, float]:
    doc = fitz.open(str(pdf_path))
    try:
        r = doc[0].rect
        return float(r.width), float(r.height)
    finally:
        doc.close()


def _png_pixel_size(png_path: Path) -> Optional[Tuple[int, int]]:
    try:
        from PIL import Image
        with Image.open(png_path) as im:
            return im.width, im.height
    except Exception:
        return None


def _intersect_label_rect_page(
    lr: Tuple[float, float, float, float],
    Wp: float,
    Hp: float,
) -> Optional[Tuple[float, float, float, float]]:
    x0, y0, x1, y1 = lr
    ix0 = max(0.0, min(Wp, x0))
    iy0 = max(0.0, min(Hp, y0))
    ix1 = max(0.0, min(Wp, x1))
    iy1 = max(0.0, min(Hp, y1))
    if ix1 <= ix0 or iy1 <= iy0:
        return None
    return (ix0, iy0, ix1, iy1)


def _page_size_from_aspect(Wp: float, Hp: float) -> Tuple[int, int]:
    base_w = 1000
    if Wp <= 0 or Hp <= 0:
        return base_w, 1000
    h = max(1, int(round(base_w * (Hp / Wp))))
    return base_w, h


def _orientation_mode(template: Dict[str, Any], Wp: float, Hp: float) -> str:
    """Return ``landscape`` | ``portrait`` | ``square`` for layout choice."""
    raw = str(template.get("orientation", "") or "").strip().lower()
    if raw in ("landscape", "portrait", "square"):
        return raw
    if Hp <= 0 or Wp <= 0:
        return "landscape"
    ar = Wp / Hp
    if ar >= 1.15:
        return "landscape"
    if ar <= 0.87:
        return "portrait"
    return "square"


def _regions_landscape_strip() -> List[Dict[str, Any]]:
    return [
        {
            "name": "main_left",
            "type": "text_block",
            "bbox_norm": [0.0, 0.0, 0.26, 1.0],
            "mandatory": True,
            "weight": 1.0,
        },
        {
            "name": "main_middle",
            "type": "free_form_block",
            "bbox_norm": [0.26, 0.0, 0.28, 1.0],
            "mandatory": True,
            "weight": 1.0,
        },
        {
            "name": "barcode_or_symbol_area",
            "type": "barcode_block",
            "bbox_norm": [0.54, 0.0, 0.22, 1.0],
            "mandatory": True,
            "weight": 1.0,
        },
        {
            "name": "footer_or_right_area",
            "type": "logo_block",
            "bbox_norm": [0.76, 0.0, 0.14, 1.0],
            "mandatory": True,
            "weight": 1.0,
        },
        {
            "name": "optional_far_strip",
            "type": "optional_image",
            "bbox_norm": [0.90, 0.0, 0.10, 1.0],
            "mandatory": False,
            "weight": 0.0,
        },
    ]


def _regions_portrait_stack() -> List[Dict[str, Any]]:
    return [
        {
            "name": "top_identity",
            "type": "text_block",
            "bbox_norm": [0.0, 0.0, 1.0, 0.22],
            "mandatory": True,
            "weight": 1.0,
        },
        {
            "name": "main_middle",
            "type": "free_form_block",
            "bbox_norm": [0.0, 0.22, 1.0, 0.36],
            "mandatory": True,
            "weight": 1.0,
        },
        {
            "name": "barcode_or_symbol_area",
            "type": "barcode_block",
            "bbox_norm": [0.0, 0.48, 1.0, 0.27],
            "mandatory": True,
            "weight": 1.0,
        },
        {
            "name": "footer_or_right_area",
            "type": "text_block",
            "bbox_norm": [0.0, 0.72, 1.0, 0.20],
            "mandatory": True,
            "weight": 1.0,
        },
        {
            "name": "optional_bottom_strip",
            "type": "optional_image",
            "bbox_norm": [0.0, 0.92, 1.0, 0.08],
            "mandatory": False,
            "weight": 0.0,
        },
    ]


def build_coarse_fallback_schema_dict(
    product_code: str,
    *,
    template: Dict[str, Any],
    label_pdf_path: Path,
    label_png_path: Optional[Path],
    template_version: str,
) -> Dict[str, Any]:
    """
    Build a conservative ``bbox_norm`` schema from page / image aspect and
    template orientation hint.
    """
    Wp, Hp = _pdf_page_size_pts(label_pdf_path)
    if Wp <= 0 or Hp <= 0:
        raise ValueError(f"Invalid PDF page size from {label_pdf_path}")

    mode = _orientation_mode(template, Wp, Hp)
    if mode == "portrait" or mode == "square":
        regions = _regions_portrait_stack()
    else:
        regions = _regions_landscape_strip()

    page_w, page_h = _page_size_from_aspect(Wp, Hp)

    lr_pts = _parse_label_rect_pts(template.get("label_rect_pts"))
    lr_list: Optional[List[float]] = None
    if lr_pts is not None:
        inter = _intersect_label_rect_page(lr_pts, Wp, Hp)
        if inter is not None:
            lr_list = [round(inter[0], 4), round(inter[1], 4), round(inter[2], 4), round(inter[3], 4)]

    png_dims = _png_pixel_size(label_png_path) if label_png_path and label_png_path.is_file() else None

    out: Dict[str, Any] = {
        "product_code": product_code,
        "page_size": {"width": page_w, "height": page_h},
        "regions": regions,
        "_meta": {
            "generated_from": "coarse_fallback",
            "template_version": template_version,
            "orientation_mode": mode,
            "page_rect_pts": [0.0, 0.0, round(Wp, 4), round(Hp, 4)],
            "label_png_px": list(png_dims) if png_dims else None,
        },
    }
    if lr_list is not None:
        out["label_rect_pts"] = lr_list
    return out
