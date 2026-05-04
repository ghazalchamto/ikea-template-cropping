"""
Pure geometry helpers for layout-only validation (mm ↔ pixels, IoU, ink bboxes).
"""

from __future__ import annotations

from typing import Sequence, Tuple

import cv2
import numpy as np

from src.core.pdf_text_introspector import TextSpan


def line_count_from_spans(spans: Sequence[TextSpan]) -> int:
    """
    Count distinct text lines by vertical clustering of span centres.

    Handles multi-span single lines (e.g. font/colour runs) without splitting
    them into false extra lines.
    """
    if not spans:
        return 0
    centres = sorted((s.bbox_pdf[1] + s.bbox_pdf[3]) * 0.5 for s in spans)
    heights = [max(1.0, s.bbox_pdf[3] - s.bbox_pdf[1]) for s in spans]
    med_h = float(np.median(np.asarray(heights, dtype=np.float64)))
    gap = max(2.5, 0.35 * med_h)
    n_lines = 1
    for i in range(1, len(centres)):
        if centres[i] - centres[i - 1] > gap:
            n_lines += 1
    return n_lines


def spans_overflow_region(
    spans: Sequence[TextSpan],
    rx0: float,
    ry0: float,
    rx1: float,
    ry1: float,
    eps_pt: float,
) -> Tuple[bool, Tuple[float, float, float, float] | None]:
    """
    Return (True, bad_bbox) if any span extends outside the region rect in PDF
    space (with *eps_pt* tolerance on each edge). Returns (False, None) if OK.
    """
    e = max(0.0, float(eps_pt))
    lo0, lo1, lo2, lo3 = rx0 - e, ry0 - e, rx1 + e, ry1 + e
    for s in spans:
        bx0, by0, bx1, by1 = s.bbox_pdf
        if bx0 < lo0 - 1e-6 or bx1 > lo2 + 1e-6 or by0 < lo1 - 1e-6 or by1 > lo3 + 1e-6:
            return True, (bx0, by0, bx1, by1)
    return False, None


def union_dark_pixel_bbox(
    gray: np.ndarray,
    *,
    dark_thresh: int = 180,
    min_dark_pixels: int = 8,
) -> Tuple[float, float, float, float]:
    """
    Bounding box of *all* pixels darker than *dark_thresh* as (x, y, w, h) in
    region-image pixel coordinates.

    Unlike ``largest_ink_bbox``, this is not tied to a single connected
    component — internal bright gaps (e.g. different barcode modules) still
    yield one outer extent, which is what we want for layout-only symbology.
    """
    if gray.size == 0:
        return (0.0, 0.0, 0.0, 0.0)
    mask = gray <= int(dark_thresh)
    count = int(mask.sum())
    if count < int(min_dark_pixels):
        return (0.0, 0.0, 0.0, 0.0)
    ys, xs = np.where(mask)
    x0 = float(xs.min())
    y0 = float(ys.min())
    x1 = float(xs.max() + 1)
    y1 = float(ys.max() + 1)
    return (x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0))


def region_image_xywh_to_pdf_rect(
    xywh: Tuple[float, float, float, float],
    region_pdf_bbox: Tuple[float, float, float, float],
    region_w_px: int,
    region_h_px: int,
) -> Tuple[float, float, float, float]:
    """Map region-clip (x,y,w,h) pixels to PDF (x0, y0, x1, y1)."""
    x, y, w, h = xywh
    rx0, ry0, rx1, ry1 = region_pdf_bbox
    rw = max(float(rx1 - rx0), 1e-6)
    rh = max(float(ry1 - ry0), 1e-6)
    W = max(int(region_w_px), 1)
    H = max(int(region_h_px), 1)
    px0 = rx0 + (float(x) / float(W)) * rw
    py0 = ry0 + (float(y) / float(H)) * rh
    px1 = rx0 + (float(x + w) / float(W)) * rw
    py1 = ry0 + (float(y + h) / float(H)) * rh
    return (px0, py0, px1, py1)


def pdf_boxes_shift_size_mm(
    bbox_t: Tuple[float, float, float, float],
    bbox_c: Tuple[float, float, float, float],
    *,
    label_w_mm: float,
    label_h_mm: float,
    label_rect_w_pt: float,
    label_rect_h_pt: float,
) -> Tuple[float, float, float, float]:
    """
    Centre shift (sx_mm, sy_mm) and size deltas (dw_mm, dh_mm) for two PDF
    axis-aligned rects as (x0, y0, x1, y1).

    Each tuple is **not** (x, y, width, height); use explicit PDF corners so
    centre and width/height match ``region_image_xywh_to_pdf_rect`` output.
    """
    tcx = 0.5 * (bbox_t[0] + bbox_t[2])
    tcy = 0.5 * (bbox_t[1] + bbox_t[3])
    ccx = 0.5 * (bbox_c[0] + bbox_c[2])
    ccy = 0.5 * (bbox_c[1] + bbox_c[3])
    mw = label_w_mm / max(label_rect_w_pt, 1e-6)
    mh = label_h_mm / max(label_rect_h_pt, 1e-6)
    sx = (ccx - tcx) * mw
    sy = (ccy - tcy) * mh
    tw = max(bbox_t[2] - bbox_t[0], 1e-6)
    th = max(bbox_t[3] - bbox_t[1], 1e-6)
    cw = max(bbox_c[2] - bbox_c[0], 1e-6)
    ch = max(bbox_c[3] - bbox_c[1], 1e-6)
    dw = (cw - tw) * mw
    dh = (ch - th) * mh
    return (float(sx), float(sy), float(dw), float(dh))


def union_span_bbox(spans: Sequence[TextSpan]) -> Tuple[float, float, float, float]:
    """Axis-aligned union of span boxes in canonical pixel coordinates (x,y,w,h)."""
    if not spans:
        return (0.0, 0.0, 0.0, 0.0)
    x0 = min(s.x for s in spans)
    y0 = min(s.y for s in spans)
    x1 = max(s.x + s.width for s in spans)
    y1 = max(s.y + s.height for s in spans)
    return (x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0))


def iou_xywh(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> float:
    """IoU for boxes (x,y,w,h)."""
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = aw * ah + bw * bh - inter
    return float(inter / max(ua, 1e-9))


def largest_ink_bbox(
    gray: np.ndarray,
    *,
    dark_thresh: int = 200,
    min_area_frac: float = 0.0005,
) -> Tuple[float, float, float, float]:
    """
    Largest dark connected component as (x, y, w, h) in pixel coordinates.
    Returns (0,0,0,0) if none found.
    """
    if gray.size == 0:
        return (0.0, 0.0, 0.0, 0.0)
    mask = (gray <= dark_thresh).astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    h, w = gray.shape
    min_area = max(4, int(min_area_frac * h * w))
    best = (0, 0, 0, 0, 0)  # area, x,y,w,h
    for lbl in range(1, n):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[lbl, cv2.CC_STAT_LEFT])
        y = int(stats[lbl, cv2.CC_STAT_TOP])
        bw = int(stats[lbl, cv2.CC_STAT_WIDTH])
        bh = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        if area > best[0]:
            best = (area, x, y, bw, bh)
    if best[0] == 0:
        return (0.0, 0.0, 0.0, 0.0)
    return (float(best[1]), float(best[2]), float(best[3]), float(best[4]))


def region_tol_px(
    tol_mm: float,
    pdf_bbox: Tuple[float, float, float, float],
    region_w_px: int,
    region_h_px: int,
    label_w_mm: float,
    label_h_mm: float,
    label_rect_w_pt: float,
    label_rect_h_pt: float,
) -> Tuple[float, float]:
    """
    Approximate allowed center shift in pixels along x/y from *tol_mm*.

    Uses label physical size vs label rect in PDF points to map mm → px
    inside this region clip.
    """
    rw_pt = max(pdf_bbox[2] - pdf_bbox[0], 1e-6)
    rh_pt = max(pdf_bbox[3] - pdf_bbox[1], 1e-6)
    rw_mm = rw_pt * (label_w_mm / max(label_rect_w_pt, 1e-6))
    rh_mm = rh_pt * (label_h_mm / max(label_rect_h_pt, 1e-6))
    tol_px_x = float(tol_mm) * float(region_w_px) / max(rw_mm, 1e-6)
    tol_px_y = float(tol_mm) * float(region_h_px) / max(rh_mm, 1e-6)
    return max(1.0, tol_px_x), max(1.0, tol_px_y)


def center_delta_mm(
    rect_t: Tuple[float, float, float, float],
    rect_c: Tuple[float, float, float, float],
    pdf_bbox: Tuple[float, float, float, float],
    label_w_mm: float,
    label_h_mm: float,
    label_rect_w_pt: float,
    label_rect_h_pt: float,
) -> Tuple[float, float]:
    """Center shift (dx_mm, dy_mm) candidate relative to template."""
    tcx = rect_t[0] + 0.5 * rect_t[2]
    tcy = rect_t[1] + 0.5 * rect_t[3]
    ccx = rect_c[0] + 0.5 * rect_c[2]
    ccy = rect_c[1] + 0.5 * rect_c[3]
    dpx, dpy = ccx - tcx, ccy - tcy
    rw_pt = max(pdf_bbox[2] - pdf_bbox[0], 1e-6)
    rh_pt = max(pdf_bbox[3] - pdf_bbox[1], 1e-6)
    mm_per_px_x = (rw_pt * (label_w_mm / max(label_rect_w_pt, 1e-6))) / max(rect_t[2], 1e-6)
    mm_per_px_y = (rh_pt * (label_h_mm / max(label_rect_h_pt, 1e-6))) / max(rect_t[3], 1e-6)
    return (float(dpx * mm_per_px_x), float(dpy * mm_per_px_y))


def size_delta_mm(
    rect_t: Tuple[float, float, float, float],
    rect_c: Tuple[float, float, float, float],
    pdf_bbox: Tuple[float, float, float, float],
    label_w_mm: float,
    label_h_mm: float,
    label_rect_w_pt: float,
    label_rect_h_pt: float,
) -> Tuple[float, float]:
    """Width/height deltas (dw_mm, dh_mm) candidate minus template."""
    rw_pt = max(pdf_bbox[2] - pdf_bbox[0], 1e-6)
    rh_pt = max(pdf_bbox[3] - pdf_bbox[1], 1e-6)
    mm_per_px_w = (rw_pt * (label_w_mm / max(label_rect_w_pt, 1e-6))) / max(rect_t[2], 1e-6)
    mm_per_px_h = (rh_pt * (label_h_mm / max(label_rect_h_pt, 1e-6))) / max(rect_t[3], 1e-6)
    dw = (rect_c[2] - rect_t[2]) * mm_per_px_w
    dh = (rect_c[3] - rect_t[3]) * mm_per_px_h
    return (float(dw), float(dh))


def edge_structure_score(template_gray: np.ndarray, candidate_gray: np.ndarray) -> float:
    """0–1 similarity of Canny edge maps (dilated for tolerance)."""
    if template_gray.size == 0 or candidate_gray.size == 0:
        return 0.0
    if template_gray.shape != candidate_gray.shape:
        candidate_gray = cv2.resize(
            candidate_gray,
            (template_gray.shape[1], template_gray.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    te = cv2.Canny(template_gray, 50, 150)
    ce = cv2.Canny(candidate_gray, 50, 150)
    k = np.ones((3, 3), np.uint8)
    te = cv2.dilate(te, k)
    ce = cv2.dilate(ce, k)
    inter = np.logical_and(te > 0, ce > 0).sum()
    union = np.logical_or(te > 0, ce > 0).sum()
    if union <= 0:
        return 1.0
    return float(inter / max(union, 1))
