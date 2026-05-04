"""
Layout-only region validators: compare template vs candidate geometry and
structure without requiring matching text or barcode payload.

Used when ``--layout-only`` is enabled on the region report pipeline.
Legacy ``region_validators`` remain unchanged.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.core.layout_geometry import (
    center_delta_mm,
    edge_structure_score,
    iou_xywh,
    largest_ink_bbox,
    line_count_from_spans,
    pdf_boxes_shift_size_mm,
    region_image_xywh_to_pdf_rect,
    size_delta_mm,
    spans_overflow_region,
    union_dark_pixel_bbox,
    union_span_bbox,
)
from src.core.layout_pdf_vectors import vector_stats_for_region
from src.core.pdf_region_extractor import ExtractedRegion, PDFExtraction
from src.core.pdf_text_introspector import get_text_spans_for_extraction
from src.core.region_comparator import RegionScore
from src.core.region_loader import Region

logger = logging.getLogger(__name__)

METHOD_LAYOUT_TEXT: str = "layout_text_geometry"
METHOD_LAYOUT_BARCODE: str = "layout_barcode_geometry"
METHOD_LAYOUT_VISUAL_EDGES: str = "layout_visual_edges"
METHOD_LAYOUT_OPTIONAL: str = "layout_optional_skip"
METHOD_LAYOUT_LOGO: str = "layout_logo_visual"
METHOD_LAYOUT_IMAGE: str = "layout_image_geometry"
METHOD_LAYOUT_VECTOR: str = "layout_vector_geometry"


def _tol_mm(region: Region, default: float) -> float:
    raw = region.extras.get("tolerance_mm")
    if raw is None:
        return default
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return default
    return max(0.05, v)


def _label_rect_pts(ext: PDFExtraction) -> Tuple[float, float, float, float]:
    lr = ext.label_rect_pdf
    return (float(lr[0]), float(lr[1]), float(lr[2]), float(lr[3]))


def _score_from_layout_metrics(
    *,
    iou: float,
    shift_x_mm: float,
    shift_y_mm: float,
    dw_mm: float,
    dh_mm: float,
    tol_mm: float,
    missing: bool,
) -> Tuple[float, List[str], Dict[str, Any]]:
    notes: List[str] = []
    if missing:
        return 0.0, ["region_missing_content"], {"iou": round(iou, 4)}

    # Normalise penalties vs tolerance (larger shift → lower score)
    def pen(v: float) -> float:
        return min(1.0, abs(v) / max(tol_mm, 0.05))

    p_shift = 0.5 * pen(shift_x_mm) + 0.5 * pen(shift_y_mm)
    p_size = 0.5 * pen(dw_mm) + 0.5 * pen(dh_mm)
    combined = float(
        np.clip(0.4 * iou + 0.3 * (1.0 - p_shift) + 0.3 * (1.0 - p_size), 0.0, 1.0)
    )

    if shift_x_mm > tol_mm or shift_x_mm < -tol_mm:
        notes.append(f"shift_x_mm:{shift_x_mm:+.2f}")
    if shift_y_mm > tol_mm or shift_y_mm < -tol_mm:
        notes.append(f"shift_y_mm:{shift_y_mm:+.2f}")
    if abs(dw_mm) > tol_mm:
        notes.append(f"width_delta_mm:{dw_mm:+.2f}")
    if abs(dh_mm) > tol_mm:
        notes.append(f"height_delta_mm:{dh_mm:+.2f}")
    if iou < 0.35:
        notes.append(f"low_iou:{iou:.3f}")

    details = {
        "iou":               round(iou, 4),
        "shift_x_mm":      round(shift_x_mm, 3),
        "shift_y_mm":      round(shift_y_mm, 3),
        "width_delta_mm":  round(dw_mm, 3),
        "height_delta_mm": round(dh_mm, 3),
        "tolerance_mm":    tol_mm,
    }
    return combined, notes, details


def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _layout_text_block(
    region: Region,
    t_er: ExtractedRegion,
    c_er: ExtractedRegion,
    template_extr: PDFExtraction,
    candidate_extr: PDFExtraction,
    label_w_mm: float,
    label_h_mm: float,
    default_tol_mm: float,
) -> RegionScore:
    """
    Layout-only text: compare the candidate to the **configured region** and to
    template anchors (font size, line count, left/baseline shift). Candidate
    span union width vs template is **not** used — different copy may extend
    horizontally as long as it stays inside the region bbox.

    Optional ``region.extras`` (JSON region fields not consumed by the loader
    core schema are stored in ``extras``):

    * ``tolerance_mm`` — max allowed horizontal/vertical anchor shift (mm)
      vs template (default: pipeline ``default_tolerance_mm``).
    * ``overflow_epsilon_pt`` — PDF pt slack on each region edge when testing
      span containment (default ``0.75``).
    * ``font_size_tol_pt`` — minimum absolute pt tolerance; also compared to
      ``0.12 * template_median_size`` and ``0.6`` pt (default ``1.25``).
    * ``line_count_tolerance`` — allowed ``|candidate_lines - template_lines|``
      (default ``0``).
    """
    tol_mm = _tol_mm(region, default_tol_mm)
    lr = _label_rect_pts(template_extr)
    lr_w_pt = max(lr[2] - lr[0], 1e-6)
    lr_h_pt = max(lr[3] - lr[1], 1e-6)
    mm_per_pt_x = label_w_mm / lr_w_pt
    mm_per_pt_y = label_h_mm / lr_h_pt

    rx0, ry0, rx1, ry1 = (float(c_er.pdf_bbox[0]), float(c_er.pdf_bbox[1]),
                         float(c_er.pdf_bbox[2]), float(c_er.pdf_bbox[3]))
    eps_pt = float(region.extras.get("overflow_epsilon_pt", 0.75))
    font_tol_pt = float(region.extras.get("font_size_tol_pt", 1.25))
    line_tol = int(region.extras.get("line_count_tolerance", 0))

    spans_t = get_text_spans_for_extraction(template_extr, region)
    spans_c = get_text_spans_for_extraction(candidate_extr, region)

    if not spans_t and not spans_c:
        gt = cv2.cvtColor(t_er.image_bgr, cv2.COLOR_BGR2GRAY) if t_er.image_bgr.ndim == 3 else t_er.image_bgr
        gc = cv2.cvtColor(c_er.image_bgr, cv2.COLOR_BGR2GRAY) if c_er.image_bgr.ndim == 3 else c_er.image_bgr
        box_t = largest_ink_bbox(gt, dark_thresh=210)
        box_c = largest_ink_bbox(gc, dark_thresh=210)
        if box_t[2] < 1 or box_t[3] < 1:
            combined, notes, details = 0.0, ["template_region_empty"], {"iou": 0.0, "mode": "ink_fallback"}
            _stamp_layout_report(details, list(notes), combined, t_er, c_er)
            return _mk_score(region, combined, METHOD_LAYOUT_TEXT, notes, details, t_er)
        if box_c[2] < 1 or box_c[3] < 1:
            combined, notes, details = 0.0, ["candidate_region_empty"], {"iou": 0.0, "mode": "ink_fallback"}
            _stamp_layout_report(details, list(notes), combined, t_er, c_er)
            return _mk_score(region, combined, METHOD_LAYOUT_TEXT, notes, details, t_er)

        iou_v = iou_xywh(box_t, box_c)
        sx, sy = center_delta_mm(box_t, box_c, t_er.pdf_bbox, label_w_mm, label_h_mm, lr_w_pt, lr_h_pt)
        dw, dh = size_delta_mm(box_t, box_c, t_er.pdf_bbox, label_w_mm, label_h_mm, lr_w_pt, lr_h_pt)
        missing = (box_t[2] < 1 or box_t[3] < 1) or (box_c[2] < 1 or box_c[3] < 1)
        combined, notes, details = _score_from_layout_metrics(
            iou=iou_v, shift_x_mm=sx, shift_y_mm=sy,
            dw_mm=dw, dh_mm=dh, tol_mm=tol_mm,
            missing=missing,
        )
        details["mode"] = "ink_fallback"
        details["union_bbox_template_px"] = [round(v, 2) for v in box_t]
        details["union_bbox_candidate_px"] = [round(v, 2) for v in box_c]
        _stamp_layout_report(details, list(notes), combined, t_er, c_er)
        return _mk_score(region, combined, METHOD_LAYOUT_TEXT, notes, details, t_er)

    notes: List[str] = []
    details: Dict[str, Any] = {
        "mode":               "region_anchor",
        "region_pdf":         [round(rx0, 2), round(ry0, 2), round(rx1, 2), round(ry1, 2)],
        "tolerance_mm":       tol_mm,
        "overflow_epsilon_pt": eps_pt,
    }

    if spans_t and not spans_c:
        details["lines_template"] = line_count_from_spans(spans_t)
        details["lines_candidate"] = 0
        nmiss = ["candidate_text_missing"]
        _stamp_layout_report(details, nmiss, 0.0, t_er, c_er)
        return _mk_score(
            region, 0.0, METHOD_LAYOUT_TEXT,
            nmiss, details, t_er,
        )

    overflow, bad = spans_overflow_region(spans_c, rx0, ry0, rx1, ry1, eps_pt)
    if overflow and bad is not None:
        notes.append("text_overflow_outside_region")
        details["overflow_span_bbox_pdf"] = [round(v, 2) for v in bad]

    if spans_t:
        lt = line_count_from_spans(spans_t)
        lc = line_count_from_spans(spans_c)
        details["lines_template"] = lt
        details["lines_candidate"] = lc
        if abs(lc - lt) > line_tol:
            notes.append(f"line_count_mismatch:template={lt} candidate={lc}")

        sizes_t = [s.size_pt for s in spans_t if s.size_pt > 0.1]
        sizes_c = [s.size_pt for s in spans_c if s.size_pt > 0.1]
        mt = _median(sizes_t) if sizes_t else 0.0
        mc = _median(sizes_c) if sizes_c else 0.0
        details["font_size_template_pt"] = round(mt, 3)
        details["font_size_candidate_pt"] = round(mc, 3)
        if sizes_t and sizes_c:
            dyn_tol = max(font_tol_pt, 0.12 * mt, 0.6)
            if abs(mc - mt) > dyn_tol:
                notes.append(f"font_size_mismatch:delta_pt={abs(mc - mt):.2f}")

        left_t = min(s.bbox_pdf[0] for s in spans_t)
        left_c = min(s.bbox_pdf[0] for s in spans_c)
        base_t = _median([0.5 * (s.bbox_pdf[1] + s.bbox_pdf[3]) for s in spans_t])
        base_c = _median([0.5 * (s.bbox_pdf[1] + s.bbox_pdf[3]) for s in spans_c])
        delta_x_mm = (left_c - left_t) * mm_per_pt_x
        delta_y_mm = (base_c - base_t) * mm_per_pt_y
        details["anchor_shift_x_mm"] = round(delta_x_mm, 3)
        details["anchor_shift_y_mm"] = round(delta_y_mm, 3)
        if abs(delta_x_mm) > tol_mm:
            notes.append(f"horizontal_shift_mm:{delta_x_mm:+.2f}")
        if abs(delta_y_mm) > tol_mm:
            notes.append(f"vertical_shift_mm:{delta_y_mm:+.2f}")
    else:
        details["lines_template"] = 0
        details["lines_candidate"] = line_count_from_spans(spans_c)
        details["font_size_template_pt"] = None
        details["font_size_candidate_pt"] = round(
            _median([s.size_pt for s in spans_c if s.size_pt > 0.1]) or 0.0, 3,
        )

    box_c = union_span_bbox(spans_c)
    details["union_bbox_candidate_px"] = [round(v, 2) for v in box_c]
    if spans_t:
        details["union_bbox_template_px"] = [round(v, 2) for v in union_span_bbox(spans_t)]
    else:
        details["union_bbox_template_px"] = None

    combined = 0.0 if notes else 1.0
    _stamp_layout_report(details, notes, combined, t_er, c_er)
    return _mk_score(region, combined, METHOD_LAYOUT_TEXT, notes, details, t_er)


def _gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def _stamp_layout_report(
    details: Dict[str, Any],
    notes: List[str],
    combined: float,
    t_er: ExtractedRegion,
    c_er: ExtractedRegion,
) -> None:
    details["failure_reasons"] = list(notes)
    if notes:
        details["severity"] = "critical"
    elif combined < 1.0:
        details["severity"] = "warning"
    else:
        details["severity"] = "none"
    details["pdf_bbox_region_template"] = [round(float(x), 3) for x in t_er.pdf_bbox]
    details["pdf_bbox_region_candidate"] = [round(float(x), 3) for x in c_er.pdf_bbox]


def _layout_barcode_block(
    region: Region,
    t_er: ExtractedRegion,
    c_er: ExtractedRegion,
    template_extr: PDFExtraction,
    candidate_extr: PDFExtraction,
    label_w_mm: float,
    label_h_mm: float,
    default_tol_mm: float,
) -> RegionScore:
    """
    Barcode / DataMatrix layout: compare **outer ink extent** only (union of dark
    pixels). Internal module pattern differences do not affect bbox; position
    and size are checked in mm against tolerance.
    """
    tol_mm = _tol_mm(region, default_tol_mm * 0.8)
    lr = _label_rect_pts(template_extr)
    lr_w = max(lr[2] - lr[0], 1e-6)
    lr_h = max(lr[3] - lr[1], 1e-6)

    gt = _gray(t_er.image_bgr)
    gc = _gray(c_er.image_bgr)
    Ht, Wt = int(gt.shape[0]), int(gt.shape[1])
    Hc, Wc = int(gc.shape[0]), int(gc.shape[1])

    thresh = int(region.extras.get("barcode_dark_thresh", 180))
    min_px_t = int(region.extras.get("barcode_min_dark_pixels", max(8, int(0.0008 * Wt * Ht))))
    min_px_c = int(region.extras.get("barcode_min_dark_pixels", max(8, int(0.0008 * Wc * Hc))))

    box_t = union_dark_pixel_bbox(gt, dark_thresh=thresh, min_dark_pixels=min_px_t)
    box_c = union_dark_pixel_bbox(gc, dark_thresh=thresh, min_dark_pixels=min_px_c)

    pdf_t = region_image_xywh_to_pdf_rect(box_t, t_er.pdf_bbox, Wt, Ht)
    pdf_c = region_image_xywh_to_pdf_rect(box_c, c_er.pdf_bbox, Wc, Hc)

    notes: List[str] = []
    details: Dict[str, Any] = {
        "mode":                      "barcode_union_ink",
        "ink_bbox_template_px":      [round(v, 2) for v in box_t],
        "ink_bbox_candidate_px":     [round(v, 2) for v in box_c],
        "expected_object_bbox_pdf":  [round(v, 3) for v in pdf_t],
        "candidate_object_bbox_pdf": [round(v, 3) for v in pdf_c],
        "tolerance_mm":              tol_mm,
    }

    t_ok = box_t[2] >= 2.0 and box_t[3] >= 2.0
    c_ok = box_c[2] >= 2.0 and box_c[3] >= 2.0

    if not t_ok:
        notes.append("template_barcode_missing")
    if not c_ok:
        notes.append("candidate_barcode_missing")

    if t_ok and c_ok:
        sx, sy, dw, dh = pdf_boxes_shift_size_mm(
            pdf_t, pdf_c,
            label_w_mm=label_w_mm, label_h_mm=label_h_mm,
            label_rect_w_pt=lr_w, label_rect_h_pt=lr_h,
        )
        details["shift_x_mm"] = round(sx, 3)
        details["shift_y_mm"] = round(sy, 3)
        details["width_delta_mm"] = round(dw, 3)
        details["height_delta_mm"] = round(dh, 3)
        if abs(sx) > tol_mm:
            notes.append(f"barcode_shift_x_mm:{sx:+.2f}")
        if abs(sy) > tol_mm:
            notes.append(f"barcode_shift_y_mm:{sy:+.2f}")
        if abs(dw) > tol_mm:
            notes.append(f"barcode_width_delta_mm:{dw:+.2f}")
        if abs(dh) > tol_mm:
            notes.append(f"barcode_height_delta_mm:{dh:+.2f}")

    combined = 0.0 if notes else 1.0
    _stamp_layout_report(details, notes, combined, t_er, c_er)
    return _mk_score(region, combined, METHOD_LAYOUT_BARCODE, notes, details, t_er)


def _layout_logo_icon_block(
    region: Region,
    t_er: ExtractedRegion,
    c_er: ExtractedRegion,
    template_extr: PDFExtraction,
    candidate_extr: PDFExtraction,
    label_w_mm: float,
    label_h_mm: float,
    default_tol_mm: float,
) -> RegionScore:
    tol_mm = _tol_mm(region, default_tol_mm)
    lr = _label_rect_pts(template_extr)
    lr_w = max(lr[2] - lr[0], 1e-6)
    lr_h = max(lr[3] - lr[1], 1e-6)

    gt = _gray(t_er.image_bgr)
    gc = _gray(c_er.image_bgr)
    Ht, Wt = int(gt.shape[0]), int(gt.shape[1])
    Hc, Wc = int(gc.shape[0]), int(gc.shape[1])

    dark = int(region.extras.get("logo_dark_thresh", 218))
    minp_t = int(region.extras.get("logo_min_dark_pixels", max(32, int(0.003 * Wt * Ht))))
    minp_c = int(region.extras.get("logo_min_dark_pixels", max(32, int(0.003 * Wc * Hc))))

    box_t = union_dark_pixel_bbox(gt, dark_thresh=dark, min_dark_pixels=minp_t)
    box_c = union_dark_pixel_bbox(gc, dark_thresh=dark, min_dark_pixels=minp_c)
    pdf_t = region_image_xywh_to_pdf_rect(box_t, t_er.pdf_bbox, Wt, Ht)
    pdf_c = region_image_xywh_to_pdf_rect(box_c, c_er.pdf_bbox, Wc, Hc)

    notes: List[str] = []
    edge = edge_structure_score(gt, gc)
    edge_min = float(region.extras.get("logo_edge_min", 0.36))

    details: Dict[str, Any] = {
        "mode":                      "logo_union_ink_plus_edges",
        "edge_structure_score":      round(edge, 4),
        "logo_edge_min":             edge_min,
        "ink_bbox_template_px":      [round(v, 2) for v in box_t],
        "ink_bbox_candidate_px":     [round(v, 2) for v in box_c],
        "expected_object_bbox_pdf":  [round(v, 3) for v in pdf_t],
        "candidate_object_bbox_pdf": [round(v, 3) for v in pdf_c],
        "tolerance_mm":              tol_mm,
    }

    t_ok = box_t[2] >= 3.0 and box_t[3] >= 3.0
    c_ok = box_c[2] >= 3.0 and box_c[3] >= 3.0
    if not t_ok:
        notes.append("template_logo_missing")
    if not c_ok:
        notes.append("candidate_logo_missing")

    if t_ok and c_ok:
        sx, sy, dw, dh = pdf_boxes_shift_size_mm(
            pdf_t, pdf_c,
            label_w_mm=label_w_mm, label_h_mm=label_h_mm,
            label_rect_w_pt=lr_w, label_rect_h_pt=lr_h,
        )
        details["shift_x_mm"] = round(sx, 3)
        details["shift_y_mm"] = round(sy, 3)
        details["width_delta_mm"] = round(dw, 3)
        details["height_delta_mm"] = round(dh, 3)
        if abs(sx) > tol_mm:
            notes.append(f"logo_shift_x_mm:{sx:+.2f}")
        if abs(sy) > tol_mm:
            notes.append(f"logo_shift_y_mm:{sy:+.2f}")
        if abs(dw) > tol_mm:
            notes.append(f"logo_width_delta_mm:{dw:+.2f}")
        if abs(dh) > tol_mm:
            notes.append(f"logo_height_delta_mm:{dh:+.2f}")

    if not notes and edge < edge_min:
        notes.append("logo_visual_mismatch_low_edge_structure")

    combined = 0.0 if notes else 1.0
    _stamp_layout_report(details, notes, combined, t_er, c_er)
    return _mk_score(region, combined, METHOD_LAYOUT_LOGO, notes, details, t_er)


def _layout_image_placeholder_block(
    region: Region,
    t_er: ExtractedRegion,
    c_er: ExtractedRegion,
    template_extr: PDFExtraction,
    candidate_extr: PDFExtraction,
    label_w_mm: float,
    label_h_mm: float,
    default_tol_mm: float,
) -> RegionScore:
    """Image / placeholder: placement + size vs template; content optional by default."""
    tol_mm = _tol_mm(region, default_tol_mm)
    lr = _label_rect_pts(template_extr)
    lr_w = max(lr[2] - lr[0], 1e-6)
    lr_h = max(lr[3] - lr[1], 1e-6)

    gt = _gray(t_er.image_bgr)
    gc = _gray(c_er.image_bgr)
    Ht, Wt = int(gt.shape[0]), int(gt.shape[1])
    Hc, Wc = int(gc.shape[0]), int(gc.shape[1])

    dark = int(region.extras.get("image_dark_thresh", 235))
    minp_t = int(region.extras.get("image_min_dark_pixels", max(20, int(0.002 * Wt * Ht))))
    minp_c = int(region.extras.get("image_min_dark_pixels", max(20, int(0.002 * Wc * Hc))))

    box_t = union_dark_pixel_bbox(gt, dark_thresh=dark, min_dark_pixels=minp_t)
    box_c = union_dark_pixel_bbox(gc, dark_thresh=dark, min_dark_pixels=minp_c)
    pdf_t = region_image_xywh_to_pdf_rect(box_t, t_er.pdf_bbox, Wt, Ht)
    pdf_c = region_image_xywh_to_pdf_rect(box_c, c_er.pdf_bbox, Wc, Hc)

    notes: List[str] = []
    require_visual = bool(region.extras.get("require_visual_match", False))
    edge_min = float(region.extras.get("image_edge_min", 0.22))
    edge = edge_structure_score(gt, gc)

    details: Dict[str, Any] = {
        "mode":                      "image_placeholder_geometry",
        "require_visual_match":      require_visual,
        "edge_structure_score":      round(edge, 4),
        "ink_bbox_template_px":      [round(v, 2) for v in box_t],
        "ink_bbox_candidate_px":     [round(v, 2) for v in box_c],
        "expected_object_bbox_pdf":  [round(v, 3) for v in pdf_t],
        "candidate_object_bbox_pdf": [round(v, 3) for v in pdf_c],
        "tolerance_mm":              tol_mm,
    }

    t_ok = box_t[2] >= 3.0 and box_t[3] >= 3.0
    c_ok = box_c[2] >= 3.0 and box_c[3] >= 3.0
    if not t_ok:
        notes.append("template_image_missing")
    if not c_ok:
        notes.append("candidate_image_missing")

    if t_ok and c_ok:
        sx, sy, dw, dh = pdf_boxes_shift_size_mm(
            pdf_t, pdf_c,
            label_w_mm=label_w_mm, label_h_mm=label_h_mm,
            label_rect_w_pt=lr_w, label_rect_h_pt=lr_h,
        )
        details["shift_x_mm"] = round(sx, 3)
        details["shift_y_mm"] = round(sy, 3)
        details["width_delta_mm"] = round(dw, 3)
        details["height_delta_mm"] = round(dh, 3)
        if abs(sx) > tol_mm:
            notes.append(f"image_shift_x_mm:{sx:+.2f}")
        if abs(sy) > tol_mm:
            notes.append(f"image_shift_y_mm:{sy:+.2f}")
        if abs(dw) > tol_mm:
            notes.append(f"image_width_delta_mm:{dw:+.2f}")
        if abs(dh) > tol_mm:
            notes.append(f"image_height_delta_mm:{dh:+.2f}")

    if not notes and require_visual and edge < edge_min:
        notes.append("image_visual_mismatch_low_edge_structure")

    combined = 0.0 if notes else 1.0
    _stamp_layout_report(details, notes, combined, t_er, c_er)
    return _mk_score(region, combined, METHOD_LAYOUT_IMAGE, notes, details, t_er)


def _layout_vector_structure(
    region: Region,
    t_er: ExtractedRegion,
    c_er: ExtractedRegion,
    template_extr: PDFExtraction,
    candidate_extr: PDFExtraction,
    label_w_mm: float,
    label_h_mm: float,
    default_tol_mm: float,
) -> RegionScore:
    """
    Dividers / borders / table lines via ``get_drawings()`` union in PDF space,
    with raster edge fallback when no vector paths are recorded (flattened art).
    """
    tol_mm = _tol_mm(region, default_tol_mm * 0.9)
    lr = _label_rect_pts(template_extr)
    lr_w = max(lr[2] - lr[0], 1e-6)
    lr_h = max(lr[3] - lr[1], 1e-6)

    clip_t = tuple(float(x) for x in t_er.pdf_bbox)
    clip_c = tuple(float(x) for x in c_er.pdf_bbox)
    v_t = vector_stats_for_region(template_extr.pdf_path, template_extr.page_index, clip_t)
    v_c = vector_stats_for_region(candidate_extr.pdf_path, candidate_extr.page_index, clip_c)

    path_tol = int(region.extras.get("vector_path_count_tolerance", 3))
    notes: List[str] = []

    details: Dict[str, Any] = {
        "mode":                   "vector_union_pdf",
        "vector_paths_template":  v_t.path_count,
        "vector_paths_candidate": v_c.path_count,
        "vector_union_template_pdf": [round(v, 3) for v in v_t.union_pdf],
        "vector_union_candidate_pdf": [round(v, 3) for v in v_c.union_pdf],
        "tolerance_mm":           tol_mm,
    }

    ut, uc = v_t.union_pdf, v_c.union_pdf
    min_pt = float(region.extras.get("vector_min_union_pt", 0.08))
    valid_t = (ut[2] - ut[0]) >= min_pt and (ut[3] - ut[1]) >= min_pt
    valid_c = (uc[2] - uc[0]) >= min_pt and (uc[3] - uc[1]) >= min_pt

    def _raster_fallback() -> RegionScore:
        gt = _gray(t_er.image_bgr)
        gc = _gray(c_er.image_bgr)
        edge = edge_structure_score(gt, gc)
        edge_min = float(region.extras.get("vector_raster_edge_min", 0.36))
        details["mode"] = "vector_raster_edge_fallback"
        details["edge_structure_score"] = round(edge, 4)
        n2: List[str] = []
        if edge < edge_min:
            n2.append("line_art_mismatch_raster_fallback")
        comb = 0.0 if n2 else float(edge)
        _stamp_layout_report(details, n2, comb, t_er, c_er)
        return _mk_score(region, comb, METHOD_LAYOUT_VECTOR, n2, details, t_er)

    if v_t.path_count == 0 and v_c.path_count == 0:
        return _raster_fallback()

    if valid_t and not valid_c:
        notes.append("vector_art_missing_on_candidate")
    elif not valid_t and valid_c:
        notes.append("vector_extra_on_candidate_no_template_art")
    elif valid_t and valid_c:
        if abs(v_t.path_count - v_c.path_count) > path_tol and max(v_t.path_count, v_c.path_count) >= 2:
            notes.append(
                f"vector_path_count_mismatch:template={v_t.path_count} candidate={v_c.path_count}",
            )
        if not notes:
            sx, sy, dw, dh = pdf_boxes_shift_size_mm(
                ut, uc,
                label_w_mm=label_w_mm, label_h_mm=label_h_mm,
                label_rect_w_pt=lr_w, label_rect_h_pt=lr_h,
            )
            details["shift_x_mm"] = round(sx, 3)
            details["shift_y_mm"] = round(sy, 3)
            details["width_delta_mm"] = round(dw, 3)
            details["height_delta_mm"] = round(dh, 3)
            details["expected_object_bbox_pdf"] = [round(v, 3) for v in ut]
            details["candidate_object_bbox_pdf"] = [round(v, 3) for v in uc]
            if abs(sx) > tol_mm:
                notes.append(f"vector_shift_x_mm:{sx:+.2f}")
            if abs(sy) > tol_mm:
                notes.append(f"vector_shift_y_mm:{sy:+.2f}")
            if abs(dw) > tol_mm:
                notes.append(f"vector_width_delta_mm:{dw:+.2f}")
            if abs(dh) > tol_mm:
                notes.append(f"vector_height_delta_mm:{dh:+.2f}")
    else:
        return _raster_fallback()

    combined = 0.0 if notes else 1.0
    if not notes:
        details["expected_object_bbox_pdf"] = [round(v, 3) for v in ut]
        details["candidate_object_bbox_pdf"] = [round(v, 3) for v in uc]
    _stamp_layout_report(details, notes, combined, t_er, c_er)
    return _mk_score(region, combined, METHOD_LAYOUT_VECTOR, notes, details, t_er)


def _layout_visual_edges(
    region: Region,
    t_er: ExtractedRegion,
    c_er: ExtractedRegion,
    **_unused: Any,
) -> RegionScore:
    t = cv2.cvtColor(t_er.image_bgr, cv2.COLOR_BGR2GRAY) if t_er.image_bgr.ndim == 3 else t_er.image_bgr
    c = cv2.cvtColor(c_er.image_bgr, cv2.COLOR_BGR2GRAY) if c_er.image_bgr.ndim == 3 else c_er.image_bgr
    score = edge_structure_score(t, c)
    notes: List[str] = []
    if score < 0.45:
        notes.append("edge_structure_mismatch")
    details: Dict[str, Any] = {"edge_iou": round(score, 4), "mode": "edge_structure_only"}
    _stamp_layout_report(details, notes, score, t_er, c_er)
    return _mk_score(region, float(score), METHOD_LAYOUT_VISUAL_EDGES, notes, details, t_er)


def _layout_optional(
    region: Region,
    t_er: ExtractedRegion,
    c_er: ExtractedRegion,
    **_unused: Any,
) -> RegionScore:
    return _mk_score(
        region, 1.0, METHOD_LAYOUT_OPTIONAL,
        [], {"skipped": True, "candidate_absent": False}, t_er,
    )


def _layout_optional_candidate_absent(region: Region, t_er: ExtractedRegion) -> RegionScore:
    """Optional region not extracted on candidate — does not affect weighted score."""
    return _mk_score(
        region,
        1.0,
        METHOD_LAYOUT_OPTIONAL,
        ["optional_region_absent_on_candidate"],
        {"skipped": True, "candidate_absent": True},
        t_er,
    )


def _mk_score(
    region: Region,
    combined: float,
    method: str,
    notes: List[str],
    details: Dict[str, Any],
    t_er: Optional[ExtractedRegion],
) -> RegionScore:
    w = int(region.width) if t_er is None else int(t_er.region.width)
    h = int(region.height) if t_er is None else int(t_er.region.height)
    return RegionScore(
        region_name=region.name,
        mandatory=region.mandatory,
        weight=region.weight,
        ssim=float(combined),
        pixel=float(combined),
        edge=float(combined),
        combined=float(np.clip(combined, 0.0, 1.0)),
        width=w,
        height=h,
        notes=tuple(notes),
        region_type=region.type,
        method=method,
        details=details,
    )


_LayoutFn = Callable[..., RegionScore]

_REGISTRY: Dict[str, _LayoutFn] = {}


def _register(name: str, fn: _LayoutFn) -> None:
    _REGISTRY[name] = fn


_register("text_block", _layout_text_block)
_register("header", _layout_text_block)
_register("footer", _layout_text_block)
_register("barcode_block", _layout_barcode_block)
_register("datamatrix_block", _layout_barcode_block)
_register("logo_block", _layout_logo_icon_block)
_register("icon_block", _layout_logo_icon_block)
_register("image_block", _layout_image_placeholder_block)
_register("placeholder_block", _layout_image_placeholder_block)
_register("table_block", _layout_vector_structure)
_register("divider_line", _layout_vector_structure)
_register("border_box", _layout_vector_structure)
_register("free_form_block", _layout_visual_edges)


def validate_layout_extractions(
    template_extr: PDFExtraction,
    candidate_extr: PDFExtraction,
    *,
    label_width_mm: float,
    label_height_mm: float,
    default_tolerance_mm: float = 1.5,
) -> List[RegionScore]:
    """
    Pair regions by name; run layout-focused validators (geometry / edges).

    *label_width_mm* / *label_height_mm* come from template.json ``label_size_mm``
    (fallback to page rect mapping if unknown — caller should pass real values).
    """
    if label_width_mm <= 0 or label_height_mm <= 0:
        raise ValueError("label_width_mm and label_height_mm must be positive")

    t_by = template_extr.by_name()
    c_by = candidate_extr.by_name()
    t_names, c_names = set(t_by), set(c_by)
    extra_c = c_names - t_names
    if extra_c:
        raise ValueError(
            "candidate PDF extraction has unexpected region name(s) not present on "
            f"the template side: {sorted(extra_c)}. Check region config vs PDF pair."
        )
    missing_c = t_names - c_names
    for nm in missing_c:
        if t_by[nm].region.mandatory:
            raise ValueError(
                f"mandatory region {nm!r} is missing from candidate extraction — "
                "cannot compare layout on that page."
            )

    scores: List[RegionScore] = []
    for er_t in template_extr.extracted:
        region = er_t.region
        er_c = c_by.get(er_t.name)
        if er_c is None:
            if not region.mandatory:
                scores.append(_layout_optional_candidate_absent(region, er_t))
                continue
            raise ValueError(f"internal: mandatory region {er_t.name!r} missing on candidate")
        if not region.mandatory:
            scores.append(_layout_optional(region, er_t, er_c))
            continue
        fn = _REGISTRY.get(region.type)
        if fn is None:
            logger.warning(
                "layout mode: unknown region type %r — using edge structure",
                region.type,
            )
            fn = _layout_visual_edges
        scores.append(
            fn(
                region=region,
                t_er=er_t,
                c_er=er_c,
                template_extr=template_extr,
                candidate_extr=candidate_extr,
                label_w_mm=label_width_mm,
                label_h_mm=label_height_mm,
                default_tol_mm=default_tolerance_mm,
            )
        )
    return scores


def load_label_size_mm_from_template_json(path: Any) -> Tuple[float, float]:
    """Return (width_mm, height_mm) from ground-truth template.json."""
    p = Path(path)
    raw = json.loads(p.read_text(encoding="utf-8"))
    ls = raw.get("label_size_mm") or {}
    w = float(ls.get("width_mm", 0.0))
    h = float(ls.get("height_mm", 0.0))
    if w <= 0 or h <= 0:
        raise ValueError(f"{p}: label_size_mm.width_mm/height_mm must be positive")
    return (w, h)
