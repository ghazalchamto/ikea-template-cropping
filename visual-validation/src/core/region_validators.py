"""
src/core/region_validators.py
=============================
Typed region validators.

Different region types need different validation logic. SSIM/pixel/edge
comparison only makes sense when the template and candidate are expected
to look pixel-similar — that is true for static visual elements (logos,
fixed graphics) but **not** for text regions and barcode regions where
the candidate carries real product data and the template is a specimen
or stub.

Each region's ``type`` (set in ``config/regions/<product>.json``) selects
a validator from the registry below. Every validator returns a single
``RegionScore`` so the existing aggregator (``region_aggregator``)
continues to work unchanged.

Canonical types
---------------
* ``text_block``      — PyMuPDF text spans on the candidate; presence +
                        containment + font-size sanity. **Ignores text values.**
* ``barcode_block``   — Dark-pixel + edge density on the candidate clip.
                        **Does not decode** the barcode.
* ``logo_block``      — Visual comparison of clips (SSIM + pixel + edge).
* ``free_form_block`` — Visual comparison of clips (default fallback).
* ``optional_image``  — Lightweight presence check; the aggregator already
                        ignores the score because ``mandatory: false``.

Unknown types fall back to ``free_form_block`` with a one-time warning.

Determinism
-----------
Pure functions over images + PDF data. No randomness, no global state.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np

from src.core.pdf_region_extractor import ExtractedRegion, PDFExtraction
from src.core.pdf_text_introspector import (
    TextSpan, fraction_contained, get_text_spans_in_region,
    page_text_length, total_text_length,
)
from src.core.region_comparator import (
    CANNY_HIGH, CANNY_LOW, RegionScore, compare_region,
)
from src.core.region_loader import Region

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Canonical type names + method names (single source of truth)
# ─────────────────────────────────────────────────────────────────────────────

TYPE_TEXT_BLOCK:       str = "text_block"
TYPE_BARCODE_BLOCK:    str = "barcode_block"
TYPE_LOGO_BLOCK:       str = "logo_block"
TYPE_FREE_FORM_BLOCK:  str = "free_form_block"
TYPE_OPTIONAL_IMAGE:   str = "optional_image"

METHOD_TEXT_LAYOUT:           str = "text_layout"
METHOD_TEXT_LAYOUT_OUTLINED:  str = "text_layout_outlined"
METHOD_BARCODE_PRESENCE:      str = "barcode_presence"
METHOD_LOGO_VISUAL:           str = "logo_visual"
METHOD_FREE_FORM_VISUAL:      str = "free_form_visual"
METHOD_OPTIONAL_SKIP:         str = "optional_skip"


# Tunables — module-level so they're easy to audit.
TEXT_OVERFLOW_OK_FRAC:     float = 0.95   # ≥ this fraction-contained → no overflow penalty
TEXT_FONT_MIN_PT:          float = 3.0
TEXT_FONT_MAX_PT:          float = 100.0

BARCODE_DARK_THRESHOLD:    int   = 128    # grayscale ≤ this counts as "dark"
BARCODE_DARK_TARGET_FRAC:  float = 0.10   # ≥ this dark fraction → strong presence
BARCODE_EDGE_TARGET_FRAC:  float = 0.05   # ≥ this edge fraction → strong edges

# Visual presence proxy used when a candidate PDF has its text outlined
# (PyMuPDF can extract zero spans page-wide). Targets are tuned for text:
# typical labels show 1–10 % "dark" (text strokes) and 0.2–2 % edge density.
TEXT_OUTLINED_DARK_THRESHOLD: int   = 200
TEXT_OUTLINED_DARK_TARGET:    float = 0.020
TEXT_OUTLINED_EDGE_TARGET:    float = 0.005

OPTIONAL_DARK_FLOOR:       int   = 240    # below this brightness → "has content"


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def validate_extractions(
    template_extr:  PDFExtraction,
    candidate_extr: PDFExtraction,
) -> List[RegionScore]:
    """
    Run the typed validator for each region and return a flat list of
    ``RegionScore`` objects, one per region in the **template** extraction
    order. Pairs each template region with the same-named candidate region.
    """
    t_by_name = template_extr.by_name()
    c_by_name = candidate_extr.by_name()

    if set(t_by_name) != set(c_by_name):
        only_t = sorted(set(t_by_name) - set(c_by_name))
        only_c = sorted(set(c_by_name) - set(t_by_name))
        raise ValueError(
            f"validate_extractions: region sets differ. "
            f"only-in-template={only_t} only-in-candidate={only_c}"
        )

    scores: List[RegionScore] = []
    for er_t in template_extr.extracted:
        er_c = c_by_name[er_t.name]
        validator = _resolve_validator(er_t.region.type)
        score = validator(
            region=er_t.region,
            template=er_t,
            candidate=er_c,
            template_extr=template_extr,
            candidate_extr=candidate_extr,
        )
        scores.append(score)

    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Dispatch
# ─────────────────────────────────────────────────────────────────────────────

ValidatorFn = Callable[..., RegionScore]


def _resolve_validator(region_type: str) -> ValidatorFn:
    fn = _REGISTRY.get(region_type)
    if fn is not None:
        return fn
    if region_type:
        logger.warning(
            "Unknown region type %r — falling back to free_form_block.",
            region_type,
        )
    return validate_free_form


# ─────────────────────────────────────────────────────────────────────────────
# 1) text_block — PyMuPDF text layout, ignores values
# ─────────────────────────────────────────────────────────────────────────────

def validate_text_block(
    region:          Region,
    template:        ExtractedRegion,
    candidate:       ExtractedRegion,
    template_extr:   PDFExtraction,
    candidate_extr:  PDFExtraction,
    **_unused: Any,
) -> RegionScore:
    """
    Validate a text region by inspecting the **candidate** PDF's text spans.

    Two paths:
      • Standard (``text_layout``) — when PyMuPDF can extract text from the
        candidate page. Scores presence + containment + font-size sanity.
      • Outlined fallback (``text_layout_outlined``) — when the candidate
        page has *zero* extractable text page-wide. This is the canonical
        signal that the PDF was exported with text converted to vector
        paths (typical for print-ready label PDFs). We can't introspect
        spans, so we fall back to a visual-presence proxy on the rendered
        candidate clip (dark-pixel + edge density). The text **is** there
        visually; we just can't query it.

    The chosen method is recorded in ``RegionScore.method`` so reports
    show which path was taken.
    """
    spans = get_text_spans_in_region(
        pdf_path    = candidate_extr.pdf_path,
        page_index  = candidate_extr.page_index,
        region      = region,
        scale_x     = candidate_extr.scale_x,
        scale_y     = candidate_extr.scale_y,
    )

    if not spans:
        # Distinguish "outlined PDF" (no text page-wide) from "this region
        # is genuinely empty" (page has text elsewhere).
        page_chars = page_text_length(
            candidate_extr.pdf_path, candidate_extr.page_index,
        )
        if page_chars == 0:
            return _validate_text_block_outlined(region, candidate)

    return _validate_text_block_standard(region, spans)


def _validate_text_block_standard(
    region: Region,
    spans:  List[TextSpan],
) -> RegionScore:
    notes: List[str] = []
    n_spans  = len(spans)
    text_len = total_text_length(spans)
    presence = 1.0 if n_spans > 0 and text_len > 0 else 0.0

    contained_frac = fraction_contained(spans, region)
    # Don't penalise tiny anti-aliasing slop — only flag genuine overflow.
    overflow_score = 1.0 if contained_frac >= TEXT_OVERFLOW_OK_FRAC else contained_frac

    if spans:
        ok = sum(1 for s in spans if TEXT_FONT_MIN_PT <= s.size_pt <= TEXT_FONT_MAX_PT)
        font_score = ok / len(spans)
    else:
        font_score = 0.0

    if n_spans == 0:
        notes.append("no_text_in_region")
    if contained_frac < TEXT_OVERFLOW_OK_FRAC:
        notes.append(f"text_overflow:contained={contained_frac:.2f}")
    if spans and font_score < 1.0:
        out_of_range = [round(s.size_pt, 1) for s in spans
                        if not (TEXT_FONT_MIN_PT <= s.size_pt <= TEXT_FONT_MAX_PT)]
        notes.append(f"font_size_out_of_range:{out_of_range[:3]}")

    if presence == 0.0:
        combined = 0.0
    else:
        combined = 0.6 * presence + 0.3 * overflow_score + 0.1 * font_score

    return RegionScore(
        region_name = region.name,
        mandatory   = region.mandatory,
        weight      = region.weight,
        ssim        = 0.0,
        pixel       = 0.0,
        edge        = 0.0,
        combined    = float(np.clip(combined, 0.0, 1.0)),
        width       = int(region.width),
        height      = int(region.height),
        notes       = tuple(notes),
        region_type = region.type,
        method      = METHOD_TEXT_LAYOUT,
        details     = {
            "n_spans":        n_spans,
            "text_length":    text_len,
            "contained_frac": round(contained_frac, 4),
            "font_score":     round(font_score, 4),
        },
    )


def _validate_text_block_outlined(
    region:    Region,
    candidate: ExtractedRegion,
) -> RegionScore:
    """
    Visual presence proxy used when the candidate PDF has its text outlined.
    Reports ``method = text_layout_outlined`` so the trail is auditable.
    """
    img = candidate.image_bgr
    if img is None or img.size == 0:
        return _empty_score(
            region, METHOD_TEXT_LAYOUT_OUTLINED,
            ["empty_clip", "candidate_text_outlined"], combined=0.0,
        )

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = gray.shape
    total = max(1, h * w)

    dark_count = int((gray <= TEXT_OUTLINED_DARK_THRESHOLD).sum())
    dark_frac  = dark_count / total

    edge_map   = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)
    edge_count = int((edge_map > 0).sum())
    edge_frac  = edge_count / total

    presence   = 1.0 if (dark_frac > 0.003 or edge_frac > 0.001) else 0.0
    dark_score = min(1.0, dark_frac / TEXT_OUTLINED_DARK_TARGET) if TEXT_OUTLINED_DARK_TARGET > 0 else 0.0
    edge_score = min(1.0, edge_frac / TEXT_OUTLINED_EDGE_TARGET) if TEXT_OUTLINED_EDGE_TARGET > 0 else 0.0

    combined = (
        0.6 * presence
        + 0.25 * dark_score
        + 0.15 * edge_score
    ) if presence > 0 else 0.0

    notes: List[str] = ["candidate_text_outlined"]
    if presence == 0.0:
        notes.append("region_appears_empty")

    return RegionScore(
        region_name = region.name,
        mandatory   = region.mandatory,
        weight      = region.weight,
        ssim        = 0.0,
        pixel       = float(dark_score),
        edge        = float(edge_score),
        combined    = float(np.clip(combined, 0.0, 1.0)),
        width       = int(w),
        height      = int(h),
        notes       = tuple(notes),
        region_type = region.type,
        method      = METHOD_TEXT_LAYOUT_OUTLINED,
        details     = {
            "dark_frac": round(dark_frac, 6),
            "edge_frac": round(edge_frac, 6),
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2) barcode_block — presence/density on the candidate clip
# ─────────────────────────────────────────────────────────────────────────────

def validate_barcode_block(
    region:          Region,
    template:        ExtractedRegion,
    candidate:       ExtractedRegion,
    template_extr:   PDFExtraction,
    candidate_extr:  PDFExtraction,
    **_unused: Any,
) -> RegionScore:
    """
    Validate a barcode region by checking that the **candidate** clip has
    plausible barcode-like content: enough dark pixels and enough edges.

    No decoding, no template-pixel comparison — the template usually carries
    a placeholder spec ("AI(240)0000…") that would never match real data.
    """
    img = candidate.image_bgr
    if img is None or img.size == 0:
        return _empty_score(
            region, METHOD_BARCODE_PRESENCE, ["empty_clip"], combined=0.0,
        )

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = gray.shape
    total = max(1, h * w)

    dark_count = int((gray <= BARCODE_DARK_THRESHOLD).sum())
    dark_frac  = dark_count / total

    edge_map   = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)
    edge_count = int((edge_map > 0).sum())
    edge_frac  = edge_count / total

    presence = 1.0 if (dark_frac > 0.005 or edge_frac > 0.001) else 0.0

    # Saturate at the targets so a barcode well above the threshold scores 1.0.
    dark_score = min(1.0, dark_frac / BARCODE_DARK_TARGET_FRAC) if BARCODE_DARK_TARGET_FRAC > 0 else 0.0
    edge_score = min(1.0, edge_frac / BARCODE_EDGE_TARGET_FRAC) if BARCODE_EDGE_TARGET_FRAC > 0 else 0.0

    combined = (
        0.5 * presence
        + 0.3 * dark_score
        + 0.2 * edge_score
    ) if presence > 0 else 0.0

    notes: List[str] = []
    if presence == 0.0:
        notes.append("no_barcode_content")
    if dark_frac < 0.01:
        notes.append(f"low_dark_frac:{dark_frac:.4f}")
    if edge_frac < 0.005:
        notes.append(f"low_edge_frac:{edge_frac:.4f}")

    return RegionScore(
        region_name = region.name,
        mandatory   = region.mandatory,
        weight      = region.weight,
        ssim        = 0.0,
        pixel       = float(dark_score),
        edge        = float(edge_score),
        combined    = float(np.clip(combined, 0.0, 1.0)),
        width       = int(w),
        height      = int(h),
        notes       = tuple(notes),
        region_type = region.type,
        method      = METHOD_BARCODE_PRESENCE,
        details     = {
            "dark_frac": round(dark_frac, 6),
            "edge_frac": round(edge_frac, 6),
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3) logo_block — visual SSIM/pixel/edge on clips
# ─────────────────────────────────────────────────────────────────────────────

def validate_logo_block(
    region:          Region,
    template:        ExtractedRegion,
    candidate:       ExtractedRegion,
    template_extr:   PDFExtraction,
    candidate_extr:  PDFExtraction,
    **_unused: Any,
) -> RegionScore:
    """
    Logos are static graphics: visual comparison is appropriate. Reuses
    the existing comparator and re-stamps the result with the typed
    metadata for the report.
    """
    score = compare_region(
        template_img  = template.image_bgr,
        candidate_img = candidate.image_bgr,
        region_name   = region.name,
        mandatory     = region.mandatory,
        weight        = region.weight,
    )
    return _restamp(score, region, METHOD_LOGO_VISUAL)


# ─────────────────────────────────────────────────────────────────────────────
# 4) free_form_block — visual SSIM/pixel/edge on clips (default fallback)
# ─────────────────────────────────────────────────────────────────────────────

def validate_free_form(
    region:          Region,
    template:        ExtractedRegion,
    candidate:       ExtractedRegion,
    template_extr:   PDFExtraction,
    candidate_extr:  PDFExtraction,
    **_unused: Any,
) -> RegionScore:
    score = compare_region(
        template_img  = template.image_bgr,
        candidate_img = candidate.image_bgr,
        region_name   = region.name,
        mandatory     = region.mandatory,
        weight        = region.weight,
    )
    return _restamp(score, region, METHOD_FREE_FORM_VISUAL)


# ─────────────────────────────────────────────────────────────────────────────
# 5) optional_image — lightweight presence check
# ─────────────────────────────────────────────────────────────────────────────

def validate_optional_image(
    region:          Region,
    template:        ExtractedRegion,
    candidate:       ExtractedRegion,
    template_extr:   PDFExtraction,
    candidate_extr:  PDFExtraction,
    **_unused: Any,
) -> RegionScore:
    """
    Minimal presence check on the candidate clip. The aggregator ignores
    the score for non-mandatory regions, so this is mostly informational.
    """
    img = candidate.image_bgr
    if img is None or img.size == 0:
        present = 0.0
        details: Dict[str, Any] = {"present": False, "dark_frac": 0.0}
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        dark_count = int((gray <= OPTIONAL_DARK_FLOOR).sum())
        dark_frac  = dark_count / max(1, gray.size)
        present = 1.0 if dark_frac > 0.005 else 0.0
        details = {
            "present":   bool(present),
            "dark_frac": round(dark_frac, 6),
        }

    notes = ["optional"] if not region.mandatory else []
    if not region.mandatory and present == 0.0:
        notes.append("optional_empty")

    return RegionScore(
        region_name = region.name,
        mandatory   = region.mandatory,
        weight      = region.weight,
        ssim        = 0.0,
        pixel       = 0.0,
        edge        = 0.0,
        combined    = float(present),
        width       = int(region.width),
        height      = int(region.height),
        notes       = tuple(notes),
        region_type = region.type,
        method      = METHOD_OPTIONAL_SKIP,
        details     = details,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _restamp(score: RegionScore, region: Region, method: str) -> RegionScore:
    """Return *score* with region_type + method populated."""
    return RegionScore(
        region_name = score.region_name,
        mandatory   = score.mandatory,
        weight      = score.weight,
        ssim        = score.ssim,
        pixel       = score.pixel,
        edge        = score.edge,
        combined    = score.combined,
        width       = score.width,
        height      = score.height,
        notes       = score.notes,
        region_type = region.type,
        method      = method,
        details     = dict(score.details) if score.details else {},
    )


def _empty_score(
    region:   Region,
    method:   str,
    notes:    List[str],
    combined: float = 0.0,
) -> RegionScore:
    return RegionScore(
        region_name = region.name,
        mandatory   = region.mandatory,
        weight      = region.weight,
        ssim        = 0.0,
        pixel       = 0.0,
        edge        = 0.0,
        combined    = float(combined),
        width       = int(region.width),
        height      = int(region.height),
        notes       = tuple(notes),
        region_type = region.type,
        method      = method,
        details     = {},
    )


# Registry — built last so all validators are defined.
_REGISTRY: Dict[str, ValidatorFn] = {
    TYPE_TEXT_BLOCK:      validate_text_block,
    TYPE_BARCODE_BLOCK:   validate_barcode_block,
    TYPE_LOGO_BLOCK:      validate_logo_block,
    TYPE_FREE_FORM_BLOCK: validate_free_form,
    TYPE_OPTIONAL_IMAGE:  validate_optional_image,
}
