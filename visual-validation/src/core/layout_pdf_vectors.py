"""
Vector drawing helpers for layout validation (PyMuPDF).

Compares drawing-path bounds inside a region clip — borders, dividers, and
simple boxes — without using full-page SSIM.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Tuple

try:
    import fitz  # PyMuPDF
except ImportError as exc:  # pragma: no cover
    raise ImportError("PyMuPDF is required for layout_pdf_vectors") from exc


@dataclass(frozen=True)
class VectorClipStats:
    """Drawings from ``page.get_drawings()`` intersecting *clip* (PDF space)."""

    path_count: int
    union_pdf: Tuple[float, float, float, float]  # x0, y0, x1, y1


def vector_stats_in_clip(page: Any, clip: fitz.Rect) -> VectorClipStats:
    """
    Union bounding box of every drawing path whose ``rect`` intersects *clip*,
    plus a path count (defensive: one drawing record ≈ one path group).
    """
    try:
        drawings: List[dict] = page.get_drawings()
    except Exception:  # pragma: no cover
        return VectorClipStats(0, (0.0, 0.0, 0.0, 0.0))

    rects: List[fitz.Rect] = []
    for d in drawings:
        r = d.get("rect")
        if r is None or r.is_empty:
            continue
        if not clip.intersects(r):
            continue
        inter = r & clip
        if inter.is_empty or inter.get_area() <= 0:
            continue
        rects.append(inter)

    if not rects:
        return VectorClipStats(0, (0.0, 0.0, 0.0, 0.0))

    x0 = min(float(ir.x0) for ir in rects)
    y0 = min(float(ir.y0) for ir in rects)
    x1 = max(float(ir.x1) for ir in rects)
    y1 = max(float(ir.y1) for ir in rects)
    return VectorClipStats(len(rects), (x0, y0, x1, y1))


def vector_stats_for_region(
    pdf_path: str | Path,
    page_index: int,
    clip_pdf: Tuple[float, float, float, float],
) -> VectorClipStats:
    """Open PDF briefly and compute ``vector_stats_in_clip``."""
    clip = fitz.Rect(clip_pdf)
    doc = fitz.open(str(pdf_path))
    try:
        if page_index < 0 or page_index >= doc.page_count:
            return VectorClipStats(0, (0.0, 0.0, 0.0, 0.0))
        return vector_stats_in_clip(doc[page_index], clip)
    finally:
        doc.close()
