"""
Internal shared utilities for the IKEA label extractor.

NOT part of the public API — import via the extractor package, not directly.

Contains:
  _OcrWord            – lightweight word record produced by every OCR engine
  BBox factory funcs  – _make_bbox, _polygon_to_bbox, _pyzbar_to_bbox, _union_bbox
  _field()            – ExtractedField factory (reduces constructor boilerplate)
  Text helpers        – _group_into_lines, _words_to_text
  Token helpers       – _strip_token, _is_product_name_token
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .models import BBox, ExtractedField

# ── Token helpers ─────────────────────────────────────────────────────────────

_RE_PRODUCT_NAME_TOKEN = re.compile(r"^[A-ZÅÄÖ]{2,}(?:[-/][A-ZÅÄÖ]+)*$")
_RE_ARTICLE_NUMBER = re.compile(r"\b(\d{3})[\s.\-](\d{3})[\s.\-](\d{2})\b")

_NAME_STOP_WORDS = {
    "MADE", "IN", "OF", "THE", "AND", "OR", "FOR", "WITH", "BY", "AT",
    "ASSEMBLED", "REPUBLIC", "CHINA", "PEOPLE", "INTER",
    "IKEA", "SWEDEN", "SE", "AB", "BV", "DATE", "ATE",
    "DESIGN", "QUALITY",
}


# ── OcrWord ───────────────────────────────────────────────────────────────────

@dataclass
class _OcrWord:
    """Lightweight word record produced by every OCR engine in ocr_engine.py."""
    text: str
    conf: float     # 0.0–1.0; PDF-native words always have conf=1.0
    bbox: BBox      # absolute pixels + normalised fractions
    zone: str = ""


# ── BBox factories ────────────────────────────────────────────────────────────

def _make_bbox(x: int, y: int, w: int, h: int,
               img_w: int, img_h: int, zone: str = "") -> BBox:
    return BBox(
        x=x, y=y, w=w, h=h,
        x_frac=x / img_w if img_w else 0.0,
        y_frac=y / img_h if img_h else 0.0,
        w_frac=w / img_w if img_w else 0.0,
        h_frac=h / img_h if img_h else 0.0,
        zone=zone,
    )


def _polygon_to_bbox(polygon, img_w: int, img_h: int, zone: str = "") -> BBox:
    """Convert EasyOCR 4-corner polygon [[x,y],...] to BBox with normalised coords."""
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    x, y = int(min(xs)), int(min(ys))
    w = max(1, int(max(xs) - x))
    h = max(1, int(max(ys) - y))
    return _make_bbox(x, y, w, h, img_w, img_h, zone)


def _pyzbar_to_bbox(loc, img_w: int, img_h: int) -> BBox:
    xs = [p.x for p in loc.polygon]
    ys = [p.y for p in loc.polygon]
    x, y = min(xs), min(ys)
    w, h = max(xs) - x, max(ys) - y
    return _make_bbox(x, y, w, h, img_w, img_h)


def _union_bbox(bboxes: list[BBox], img_w: int, img_h: int, zone: str = "") -> BBox:
    """Merge a list of BBoxes into their bounding union."""
    x0 = min(b.x for b in bboxes)
    y0 = min(b.y for b in bboxes)
    x1 = max(b.x + b.w for b in bboxes)
    y1 = max(b.y + b.h for b in bboxes)
    return _make_bbox(x0, y0, x1 - x0, y1 - y0, img_w, img_h, zone)


# ── ExtractedField factory ────────────────────────────────────────────────────

def _field(
    value,
    raw: str,
    confidence: float,
    zone: str,
    bbox: Optional[BBox] = None,
    source: str = "ocr",
    pattern_match: Optional[bool] = None,
) -> ExtractedField:
    return ExtractedField(
        value=value,
        raw_text=raw,
        confidence=confidence,
        zone=zone,
        bbox=bbox,
        source=source,
        pattern_match=pattern_match,
    )


# ── Text grouping ─────────────────────────────────────────────────────────────

def _group_into_lines(
    words: list[_OcrWord], line_gap_frac: float = 0.012
) -> list[list[_OcrWord]]:
    """
    Sort words by top-y then group into lines by proximity.

    line_gap_frac: fraction of image height that separates lines (default ~7 px
    at 600 DPI).  In practice the gap is inferred from median word height so the
    parameter is only a coarse guide.
    """
    if not words:
        return []
    words_sorted = sorted(words, key=lambda w: (w.bbox.y, w.bbox.x))
    heights = [w.bbox.h for w in words_sorted if w.bbox.h > 0]
    if heights:
        med_h = sorted(heights)[len(heights) // 2]
        gap_px = max(4, int(med_h * 0.7))
    else:
        gap_px = 8

    lines: list[list[_OcrWord]] = []
    current_line: list[_OcrWord] = []
    prev_bottom = -9999

    for w in words_sorted:
        top = w.bbox.y
        if top > prev_bottom + gap_px and current_line:
            lines.append(current_line)
            current_line = []
        current_line.append(w)
        prev_bottom = max(prev_bottom, w.bbox.y + w.bbox.h)

    if current_line:
        lines.append(current_line)

    for line in lines:
        line.sort(key=lambda w: w.bbox.x)

    return lines


def _words_to_text(words: list[_OcrWord]) -> str:
    """Join words into a single string, preserving approximate line breaks."""
    lines = _group_into_lines(words)
    return "\n".join(" ".join(w.text for w in line) for line in lines)


# ── Token classifiers ─────────────────────────────────────────────────────────

def _strip_token(t: str) -> str:
    return t.strip('.,;:|"\'`')


def _is_product_name_token(t: str) -> bool:
    """Return True only if this token could be part of an IKEA product name."""
    clean = t.strip('.,;:|"\'`')
    if not clean:
        return False
    if not _RE_PRODUCT_NAME_TOKEN.match(clean):
        return False
    if clean.upper() in _NAME_STOP_WORDS:
        return False
    if re.search(r"(?:cm|mm|kg|lbs?)\b", clean, re.IGNORECASE):
        return False
    if _RE_ARTICLE_NUMBER.search(clean):
        return False
    return True
