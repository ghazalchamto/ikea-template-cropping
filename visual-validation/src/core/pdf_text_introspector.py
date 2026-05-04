"""
src/core/pdf_text_introspector.py
=================================
Lightweight helper that returns PyMuPDF text spans inside a region's PDF
rect. Used by the typed validators (``region_validators.text_block``).

This module is **read-only** and stateless. It does not modify the PDF and
does not duplicate any geometry logic — it consumes the per-axis scale
values already computed by ``pdf_region_extractor`` and stored on
``PDFExtraction`` (``scale_x``, ``scale_y``).

Why a separate module?
The extractor renders region clips. Validators that need *text data*
(positions, font sizes, characters) can't use the rendered images — they
need the original PDF. Rather than re-opening PDFs in every validator,
this module concentrates the bookkeeping in one place.

Coordinate model
----------------
Same model as ``pdf_region_extractor``: per-axis stretch-to-fit.

    config-pixel (cx, cy) → PDF point (cx * scale_x, cy * scale_y)

Span coordinates returned by ``page.get_text("dict")`` are already in PDF
points. We convert them back to **canonical pixels** (the coordinate
system the region config is authored in) so validators can compare them
directly to ``Region.x``, ``Region.y``, etc.

    PDF point (px, py) → config-pixel (px / scale_x, py / scale_y)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

try:
    import fitz   # PyMuPDF
except ImportError as exc:   # pragma: no cover
    raise ImportError(
        "PyMuPDF (fitz) is required for pdf_text_introspector. "
        "Install with: pip install pymupdf"
    ) from exc

from src.core.pdf_region_extractor import PDFExtraction
from src.core.region_loader import Region, bbox_norm_to_pdf_rect

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Result types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TextSpan:
    """A single PyMuPDF text span, projected back to canonical pixels."""

    text:       str
    font:       str
    size_pt:    float                          # font size in PDF points
    # Bounding box in canonical pixel coordinates (the same frame the region
    # config uses). Allows direct comparison to Region.x / y / width / height.
    x:          float
    y:          float
    width:      float
    height:     float
    # Original bbox in PDF points, kept for diagnostics.
    bbox_pdf:   Tuple[float, float, float, float]

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_text_spans_in_region(
    pdf_path:    str | Path,
    page_index:  int,
    region:      Region,
    scale_x:     float,
    scale_y:     float,
) -> List[TextSpan]:
    """
    Return every PyMuPDF text span that overlaps *region*'s PDF rect.

    The clip used here matches the one the extractor uses for the region
    image, so spans returned here are the spans that are visible in the
    rendered clip.

    Notes
    -----
    * Spans are filtered server-side by PyMuPDF (clip parameter), then
      projected back to canonical pixels.
    * Trims whitespace-only spans because PyMuPDF returns plenty of those.
    """
    if scale_x <= 0 or scale_y <= 0:
        raise ValueError(
            f"get_text_spans_in_region: scale_x={scale_x} and scale_y={scale_y} "
            f"must both be positive"
        )

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Region rect in PDF points (must match pdf_region_extractor mapping).
    x0_pt = region.x  * scale_x
    y0_pt = region.y  * scale_y
    x1_pt = region.x2 * scale_x
    y1_pt = region.y2 * scale_y

    spans: List[TextSpan] = []

    doc = fitz.open(str(pdf_path))
    try:
        if page_index < 0 or page_index >= doc.page_count:
            raise ValueError(
                f"{pdf_path.name}: page_index={page_index} out of range "
                f"(page_count={doc.page_count})"
            )

        page = doc[page_index]
        # Use page.rect, not raw mediabox — matches the extractor's frame.
        Wp = float(page.rect.width)
        Hp = float(page.rect.height)

        clip = fitz.Rect(
            max(0.0, x0_pt),
            max(0.0, y0_pt),
            min(Wp,  x1_pt),
            min(Hp,  y1_pt),
        )
        if clip.width <= 0 or clip.height <= 0:
            return spans

        try:
            data = page.get_text("dict", clip=clip)
        except TypeError:
            # Older PyMuPDF: no clip kwarg — fetch page-wide and filter.
            data = page.get_text("dict")

        for block in data.get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "")
                    if not text.strip():
                        continue
                    bx0, by0, bx1, by1 = span.get("bbox", (0.0, 0.0, 0.0, 0.0))
                    # Re-clip to the region rect (defensive — PyMuPDF clip is
                    # not always strict on partial overlaps).
                    if bx1 < x0_pt or bx0 > x1_pt or by1 < y0_pt or by0 > y1_pt:
                        continue
                    spans.append(TextSpan(
                        text=text,
                        font=span.get("font", ""),
                        size_pt=float(span.get("size", 0.0)),
                        x=bx0 / scale_x,
                        y=by0 / scale_y,
                        width=(bx1 - bx0) / scale_x,
                        height=(by1 - by0) / scale_y,
                        bbox_pdf=(float(bx0), float(by0), float(bx1), float(by1)),
                    ))
    finally:
        doc.close()

    return spans


def get_text_spans_for_extraction(
    extraction:  PDFExtraction,
    region:      Region,
) -> List[TextSpan]:
    """
    Convenience wrapper — pulls scale_x/scale_y/page_index off a PDFExtraction.

    When ``region.bbox_norm`` is set, uses ``extraction.label_rect_pdf`` +
    ``bbox_norm_to_pdf_rect`` so the PDF clip matches ``pdf_region_extractor``.
    Otherwise uses the legacy full-page stretch mapping.
    """
    if region.bbox_norm is not None:
        return _get_text_spans_label_norm(extraction, region)
    return get_text_spans_in_region(
        pdf_path=extraction.pdf_path,
        page_index=extraction.page_index,
        region=region,
        scale_x=extraction.scale_x,
        scale_y=extraction.scale_y,
    )


def _get_text_spans_label_norm(
    extraction: PDFExtraction,
    region:     Region,
) -> List[TextSpan]:
    """Text spans for ``bbox_norm`` regions (same PDF clip as extraction)."""
    assert region.bbox_norm is not None

    pdf_path = Path(extraction.pdf_path)
    x0_pt, y0_pt, x1_pt, y1_pt = bbox_norm_to_pdf_rect(
        extraction.label_rect_pdf, region.bbox_norm,
    )

    spans: List[TextSpan] = []
    doc = fitz.open(str(pdf_path))
    try:
        if extraction.page_index < 0 or extraction.page_index >= doc.page_count:
            return spans
        page = doc[extraction.page_index]
        Wp = float(page.rect.width)
        Hp = float(page.rect.height)

        cx0 = max(0.0, x0_pt)
        cy0 = max(0.0, y0_pt)
        cx1 = min(Wp,  x1_pt)
        cy1 = min(Hp,  y1_pt)
        if cx1 <= cx0 or cy1 <= cy0:
            return spans

        clip = fitz.Rect(cx0, cy0, cx1, cy1)
        try:
            data = page.get_text("dict", clip=clip)
        except TypeError:
            data = page.get_text("dict")

        full_w = x1_pt - x0_pt
        full_h = y1_pt - y0_pt
        if full_w <= 0 or full_h <= 0:
            return spans

        for block in data.get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "")
                    if not text.strip():
                        continue
                    bx0, by0, bx1, by1 = span.get("bbox", (0.0, 0.0, 0.0, 0.0))
                    if bx1 < x0_pt or bx0 > x1_pt or by1 < y0_pt or by0 > y1_pt:
                        continue
                    # Map PDF → canonical pixels aligned with Region.x/y/width/height
                    sx = region.x + (bx0 - x0_pt) / full_w * region.width
                    sy = region.y + (by0 - y0_pt) / full_h * region.height
                    sw = (bx1 - bx0) / full_w * region.width
                    sh = (by1 - by0) / full_h * region.height
                    spans.append(TextSpan(
                        text=text,
                        font=span.get("font", ""),
                        size_pt=float(span.get("size", 0.0)),
                        x=sx, y=sy, width=sw, height=sh,
                        bbox_pdf=(float(bx0), float(by0), float(bx1), float(by1)),
                    ))
    finally:
        doc.close()

    return spans


def page_text_length(pdf_path: str | Path, page_index: int = 0) -> int:
    """
    Return the total number of characters PyMuPDF can extract from
    *page_index* of *pdf_path*. A return value of ``0`` is the canonical
    signal for "this PDF has its text outlined / converted to vector
    paths" — i.e. the text is visually present but cannot be queried
    through ``page.get_text``.

    Validators that depend on text-span data should use this to detect
    when they need a visual-presence fallback instead of a hard fail.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(str(pdf_path))
    try:
        if page_index < 0 or page_index >= doc.page_count:
            return 0
        page = doc[page_index]
        return len(page.get_text("text") or "")
    finally:
        doc.close()


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate helpers
# ─────────────────────────────────────────────────────────────────────────────

def total_text_length(spans: Sequence[TextSpan]) -> int:
    return sum(len(s.text) for s in spans)


def fraction_contained(spans: Sequence[TextSpan], region: Region) -> float:
    """
    Fraction of span area that lies INSIDE region (canonical pixel coords).

    A span fully inside contributes 1.0; a span fully outside, 0.0;
    partial overlap proportional to clipped area / span area.
    Returns 1.0 when there are no spans (vacuously true).
    """
    if not spans:
        return 1.0

    rx0, ry0 = float(region.x), float(region.y)
    rx1, ry1 = float(region.x2), float(region.y2)

    total_area    = 0.0
    inside_area   = 0.0
    for s in spans:
        sx0, sy0 = s.x, s.y
        sx1, sy1 = s.x + s.width, s.y + s.height
        a = max(0.0, sx1 - sx0) * max(0.0, sy1 - sy0)
        if a <= 0:
            continue
        ix0 = max(rx0, sx0); iy0 = max(ry0, sy0)
        ix1 = min(rx1, sx1); iy1 = min(ry1, sy1)
        ia  = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
        total_area  += a
        inside_area += ia

    return (inside_area / total_area) if total_area > 0 else 1.0
