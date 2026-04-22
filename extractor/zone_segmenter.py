"""
Zone segmenter for IKEA labels — shape-aware.

Assigns OCR words and barcode bounding boxes to semantic zones.  Three label
shapes are supported, each with its own zone layout:

  landscape  – wide horizontal roll labels (1R10, 1R20, etc.)
               Zones are x-fraction strips (left → right).
  portrait   – tall card labels (1C70-CAA, 1C70-IDADM, etc.)
               Zones are y-fraction strips (top → bottom).
  circular   – circular plant/product labels
               Zones are concentric rings by y-fraction.

Zone boundaries are expressed as fractions in [0.0, 1.0].  For landscape
labels the fraction is x-based; for portrait/circular it is y-based.  The
``detect_label_shape`` function selects the right scheme based on aspect ratio.
"""

from __future__ import annotations
from .models import BBox, ZoneSummary

# ── Landscape zone table (original default) ───────────────────────────────────
# Entry format: (zone_id, f_start, f_end, rotate_deg)
# f_start / f_end are x-fractions (0 = left edge, 1 = right edge).

LANDSCAPE_ZONE_CFG: list[tuple[str, float, float, int]] = [
    ("left_identity",     0.00, 0.18, 0),   # product name, dims, weight, article
    ("center_barcode",    0.18, 0.40, 0),   # ITF-14 linear barcode + human-readable
    ("center_datamatrix", 0.40, 0.58, 0),   # DataMatrix 2-D symbol + printed AI text
    ("center_address",    0.58, 0.72, 0),   # address block, origin, copyright
    ("right_compliance",  0.72, 0.84, 0),   # CE marks, age-rating logo
    ("right_date",        0.84, 0.95, 0),   # DATE: YYWW, alpha code, numeric prefix
    ("right_copy_count",  0.95, 1.00, -90), # Large rotated copy-count digit
]

# ── Portrait zone table ───────────────────────────────────────────────────────
# Entry format: (zone_id, f_start, f_end, rotate_deg)
# f_start / f_end are y-fractions (0 = top edge, 1 = bottom edge).

PORTRAIT_ZONE_CFG: list[tuple[str, float, float, int]] = [
    ("top_identity",   0.00, 0.28, 0),  # product name, article, logo
    ("mid_barcode",    0.28, 0.55, 0),  # ITF-14 or DataMatrix barcode
    ("mid_address",    0.55, 0.75, 0),  # address, origin, copyright
    ("bot_compliance", 0.75, 0.90, 0),  # CE marks, wash/care symbols
    ("bot_date",       0.90, 1.00, 0),  # date, copy count
]

# ── Circular / square zone table ──────────────────────────────────────────────
# y-fraction strips from top to bottom.

CIRCULAR_ZONE_CFG: list[tuple[str, float, float, int]] = [
    ("circle_top",  0.00, 0.33, 0),  # product name, logo
    ("circle_mid",  0.33, 0.67, 0),  # barcode, article number
    ("circle_bot",  0.67, 1.00, 0),  # text, care info
]

# ── Default zones (backward compat alias) ─────────────────────────────────────
DEFAULT_ZONES: list[tuple[float, float, str]] = [
    (cfg[1], cfg[2], cfg[0]) for cfg in LANDSCAPE_ZONE_CFG
]
ALL_ZONE_IDS = [z[0] for z in LANDSCAPE_ZONE_CFG]


# ── Shape detection ───────────────────────────────────────────────────────────

def detect_label_shape(width: int, height: int) -> str:
    """
    Classify the label's geometric shape from its pixel dimensions.

    Returns one of: ``"landscape"``, ``"portrait"``, ``"circular"``.

    Thresholds
    ----------
    aspect ratio > 1.8            → landscape  (wide roll labels)
    0.85 ≤ aspect ratio ≤ 1.15   → circular   (approx. square / round)
    aspect ratio < 0.7            → portrait   (tall card labels)
    everything else               → landscape  (safe fallback)
    """
    if height == 0:
        return "landscape"
    ratio = width / height
    if ratio > 1.8:
        return "landscape"
    if 0.85 <= ratio <= 1.15:
        return "circular"
    if ratio < 0.7:
        return "portrait"
    return "landscape"


def get_zone_cfg_for_shape(shape: str) -> list[tuple[str, float, float, int]]:
    """Return the zone-config table for the given shape string."""
    return {
        "landscape": LANDSCAPE_ZONE_CFG,
        "portrait":  PORTRAIT_ZONE_CFG,
        "circular":  CIRCULAR_ZONE_CFG,
    }.get(shape, LANDSCAPE_ZONE_CFG)


# ── Zone assignment ───────────────────────────────────────────────────────────

def assign_zone_for_shape(
    cx_frac: float,
    cy_frac: float,
    shape: str,
    zone_cfg: list[tuple[str, float, float, int]],
) -> str:
    """
    Return zone_id for a word at (cx_frac, cy_frac).

    For landscape labels the x-fraction is used; for portrait and circular
    labels the y-fraction is used.
    """
    coord = cy_frac if shape in ("portrait", "circular") else cx_frac
    for zone_id, f0, f1, _rot in zone_cfg:
        if f0 <= coord < f1:
            return zone_id
    return zone_cfg[-1][0]


def assign_zone(
    center_x_frac: float,
    zones: list[tuple[float, float, str]] = DEFAULT_ZONES,
) -> str:
    """Return zone_id for a given normalised x-centre (backward-compatible API)."""
    for x_start, x_end, zone_id in zones:
        if x_start <= center_x_frac < x_end:
            return zone_id
    if center_x_frac < DEFAULT_ZONES[0][0]:
        return DEFAULT_ZONES[0][2]
    return DEFAULT_ZONES[-1][2]


def group_words_by_zone(
    ocr_data: dict,
    image_width: int,
    min_conf: int = 5,
    zones: list[tuple[float, float, str]] = DEFAULT_ZONES,
) -> dict[str, ZoneSummary]:
    """
    Partition pytesseract word-level results into zone buckets.

    Parameters
    ----------
    ocr_data   : dict from pytesseract.image_to_data(..., output_type=Output.DICT)
    image_width: pixel width of the source image
    min_conf   : minimum OCR confidence to include (0–100)
    zones      : zone boundary table (override for non-standard labels)
    """
    summaries: dict[str, ZoneSummary] = {
        z_id: ZoneSummary(zone_id=z_id)
        for _, _, z_id in zones
    }
    summaries["unknown"] = ZoneSummary(zone_id="unknown")

    n = len(ocr_data.get("text", []))
    conf_sum: dict[str, float] = {k: 0.0 for k in summaries}
    word_counts: dict[str, int] = {k: 0 for k in summaries}

    for i in range(n):
        text = (ocr_data["text"][i] or "").strip()
        conf = int(ocr_data["conf"][i])
        if not text or conf < min_conf:
            continue

        x = ocr_data["left"][i]
        w = ocr_data["width"][i]

        center_x_frac = (x + w / 2) / image_width
        zone_id = assign_zone(center_x_frac, zones)

        summary = summaries[zone_id]
        summary.raw_lines.append(text)
        word_counts[zone_id] += 1
        conf_sum[zone_id] += conf / 100.0

    for zone_id, summary in summaries.items():
        wc = word_counts[zone_id]
        summary.word_count = wc
        summary.avg_confidence = (conf_sum[zone_id] / wc) if wc else 0.0

    return summaries


def bbox_zone(bbox: BBox, image_width: int) -> str:
    """Return the zone for a barcode bounding box (backward-compatible API)."""
    center_x_frac = bbox.center_x / image_width
    return assign_zone(center_x_frac)
