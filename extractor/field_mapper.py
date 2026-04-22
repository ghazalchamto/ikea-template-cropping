"""
Field mapper for the IKEA label extractor.

Converts a merged OCR word list (produced by ocr_engine.run_ocr) plus decoded
barcodes into a fully-populated ExtractedLabel.

Key improvements vs the original monolith:
  • Confidence gating  – OCR words below MIN_OCR_CONFIDENCE are excluded from
    field extraction (they are still stored in label.all_word_boxes for display).
    PDF-native words (conf=1.0) are always kept regardless of the threshold.
  • Pure standalone functions – no class state; every _extr_* function takes its
    inputs explicitly, making each extractor independently testable.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ._helpers import (
    _OcrWord,
    _field,
    _group_into_lines,
    _is_product_name_token,
    _make_bbox,
    _union_bbox,
    _words_to_text,
)
from .gs1_parser import detect_and_parse, merge_ai_dicts, parse_parentheses_form
from .models import (
    BBox,
    DataMatrixContent,
    ExtractionMetadata,
    ExtractedLabel,
    ZoneSummary,
)
from .pdf_passes import RawElement

logger = logging.getLogger(__name__)

# ── Minimum OCR confidence threshold ─────────────────────────────────────────
# Words below this are considered too uncertain for field extraction.
# PDF-native words (conf=1.0) are always exempt.
# Tune via config.py → MIN_OCR_CONFIDENCE; default 0.25.
try:
    from config import MIN_OCR_CONFIDENCE as _MIN_OCR_CONF
except ImportError:
    _MIN_OCR_CONF = 0.25

MIN_OCR_CONFIDENCE: float = _MIN_OCR_CONF

# ── Regex patterns ────────────────────────────────────────────────────────────

_RE_ARTICLE_NUMBER   = re.compile(r"\b(\d{3})[\s.\-](\d{3})[\s.\-](\d{2})\b")
_RE_ARTICLE_CODE     = re.compile(r"(?<!\d)(\d{5})(?!\d)")
_RE_DATE_LABEL       = re.compile(r"[Dd][Aa][Tt][Ee]\s*[:\-]\s*(\d{4})")
_RE_DATE_PARTIAL     = re.compile(r"[Aa][Tt][Ee]\s*[:\-]?\s*(\d{4})")
_RE_YYWW_BARE        = re.compile(r"\b(\d{4})\b")
_RE_ALPHA_CODE       = re.compile(r"\b([a-z]{4})\b")
_RE_DATE_PREFIX      = re.compile(r"\b(\d{3})\b")
_RE_PATENT           = re.compile(
    r"P[Ii1l]\s*[-–—:\s]+\s*([\d][\d\s\-]{4,})",
    re.IGNORECASE,
)
_RE_AGE_RATING       = re.compile(r"(\d+\s*[Mm]\s*\+|\d+\+)", re.IGNORECASE)
_RE_COPYRIGHT        = re.compile(
    r"(?:©|\(c\)|\||[©\)])?\s*Inter\s+I?KEA[\s\S]{0,180}?B\.?V\.?",
    re.IGNORECASE,
)
_RE_WEIGHT_KG        = re.compile(r"\b(\d+(?:\.\d+)?)\s*k(?:g)?\b", re.IGNORECASE)
_RE_WEIGHT_LBS       = re.compile(r"\(?\s*(\d+(?:\.\d+)?)\s*(?:lbs?|Ibs?)\s*\)?", re.IGNORECASE)
_RE_DIM_CM           = re.compile(r"\b(\d{2,4}(?:\.\d+)?)\s*(?:cm|mm)\b", re.IGNORECASE)
_RE_DIM_METRIC       = re.compile(
    r"(\d{1,4}(?:\.\d+)?(?:\s*[xX×]\s*\d{1,4}(?:\.\d+)?)+)\s*(cm|mm)\b",
    re.IGNORECASE,
)
_RE_DIM_IMPERIAL     = re.compile(
    r"""(\d+\s*[''′]\s*(?:\d+(?:[''′""″¼½¾⅛⅜⅝⅞/]\d*)?)?"""
    r"""|\d+(?:\s*/\s*\d+)?\s*[xX×]\s*\d+(?:\s*/\s*\d+)?\s*[""″¼½¾⅛⅜⅝⅞'"])""",
    re.IGNORECASE,
)
_RE_GROSS_KG         = re.compile(r"\bGross\s+(\d+(?:\.\d+)?)\s*k(?:g)?\b", re.IGNORECASE)
_RE_GROSS_LBS        = re.compile(r"\bGross[^)]{0,30}\(\s*(\d+(?:\.\d+)?)\s*(?:lbs?|Ibs?)\s*\)", re.IGNORECASE)
_RE_NET_KG           = re.compile(r"\bNet\s+(\d+(?:\.\d+)?)\s*k(?:g)?\b", re.IGNORECASE)
_RE_NET_LBS          = re.compile(r"\bNet[^)]{0,30}\(\s*(\d+(?:\.\d+)?)\s*(?:lbs?|Ibs?)\s*\)", re.IGNORECASE)
_RE_QTY_MULT         = re.compile(r"(?<![0-9/.])\b[Xx×]\s*([2-9]|1[0-2])\b(?![0-9/.])")
_RE_ITEM_NUM         = re.compile(r"(?<!\d)(\d{4,6})(?!\d)")
_RE_PLANT_ID         = re.compile(r"\bPI[-–]\d{3,9}[-–]\d{1,3}\b", re.IGNORECASE)
_RE_HUMAN_DATE       = re.compile(
    r"(\d{2}[-./]\d{2}[-./]\d{2})\s*\(?\s*YY[-.]?MM[-.]?DD\s*\)?",
    re.IGNORECASE,
)
_RE_SUPPLIER_NAME    = re.compile(r"I?KEA\s+of\s+Sweden\s+AB", re.IGNORECASE)
_RE_SUPPLIER_ADDR    = re.compile(r"SE\s*[-–]\s*\d{5}\s*\S+", re.IGNORECASE)
_RE_CUSTOM_ID        = re.compile(
    r"([A-Za-z]{3,5})[-\s](\d{4})[-\s]([A-Za-z0-9]{3,6})",
    re.IGNORECASE,
)
_RE_PKG_NUM          = re.compile(r"(?<!\d)(\d{2,4})(?!\d)")
_RE_PKG_TYPE         = re.compile(r"(Not\s+RTS\s+\w+|\bRTS\s+\w+|\bRegular\b|\bSpecial\b)", re.IGNORECASE)

_COMPLIANCE_KEYWORDS = {
    "CE":    re.compile(r"\bCE\b"),
    "TSCA":  re.compile(r"\bTSCA\b"),
    "ULEF":  re.compile(r"\bULEF\b"),
    "CARB":  re.compile(r"\bCARB\b"),
    "RoHS":  re.compile(r"\bRoHS\b", re.IGNORECASE),
    "EPEAT": re.compile(r"\bEPEAT\b"),
}
_ADDR_ANCHOR_RE = re.compile(
    r"IKEA|CEA|Sweden|SE\s*-\s*34\d{3}|Almhult|Älmhult|Inter\s+IKEA|IKEA\.com|B\.V\.|\.com",
    re.IGNORECASE,
)
_ADDR_STOP_RE = re.compile(
    r"\b(Made|Assembled|Republic|China|Vietnam|Poland|India|USA)\b",
    re.IGNORECASE,
)


# ── Field-level helpers ───────────────────────────────────────────────────────

def _resolve_words(
    zone_words: list[_OcrWord],
    all_words: list[_OcrWord] | None,
    min_zone: int = 2,
) -> list[_OcrWord]:
    """
    Return zone_words if they meet the minimum threshold, else all_words.

    Layout-agnostic fallback: if a zone is sparsely populated (e.g. a
    square/portrait label where horizontal zones don't align with content),
    fall back to searching the entire word set.
    """
    if len(zone_words) >= min_zone:
        return zone_words
    return all_words if all_words is not None else zone_words


def _find_word_bbox(
    words: list[_OcrWord],
    target: str,
    img_w: int,
    img_h: int,
    zone: str,
) -> Optional[BBox]:
    """Find the bbox of the word(s) in the list that best match target text."""
    for w in words:
        if target in w.text or w.text in target:
            return w.bbox
    return None


# ── Sub-extractors ────────────────────────────────────────────────────────────

def _extr_barcodes(
    label: ExtractedLabel,
    barcodes: dict,
    img_w: int,
    img_h: int,
) -> None:
    itf = barcodes.get("itf14")
    dm  = barcodes.get("datamatrix")
    ean = barcodes.get("ean13")

    if itf:
        label.itf14_barcode = itf

    if ean:
        label.ean_barcode = ean
        label.ean_barcode_value = _field(
            ean.raw_data, ean.raw_data, 1.0, "center_barcode",
            bbox=ean.bbox, source="barcode", pattern_match=True,
        )

    if dm:
        label.datamatrix_barcode = dm
        ai_fields = detect_and_parse(dm.raw_data)
        label.datamatrix_content = DataMatrixContent(
            raw_data=dm.raw_data,
            encoding="GS1" if ai_fields else "raw",
            ai_fields=ai_fields,
        )
        _promote_ai_fields(label, ai_fields, 1.0, "barcode", dm.bbox)


def _promote_ai_fields(
    label: ExtractedLabel,
    ai_fields: dict,
    conf: float,
    source: str,
    bbox: Optional[BBox] = None,
) -> None:
    for ai_code, attr in (("240", "ai_240"), ("13", "ai_13"), ("11", "ai_11"), ("10", "ai_10")):
        ai = ai_fields.get(ai_code)
        if ai and ai.present:
            setattr(label, attr, _field(
                ai.formatted or ai.value, ai.value,
                conf, "center_datamatrix", bbox=bbox, source=source, pattern_match=True
            ))


def _extr_left_identity(
    label: ExtractedLabel,
    words: list[_OcrWord],
    barcodes: dict,
    img_w: int,
    img_h: int,
    all_words: list[_OcrWord] | None = None,
) -> None:
    z = "left_identity"
    words = _resolve_words(words, all_words)
    text = _words_to_text(words)
    conf = 0.82

    # ── Copy count inline "1(2)" ───────────────────────────────────────────────
    m = re.search(r"([1lIi|'`])\s*\((\d+)\)", text)
    if m:
        digit = m.group(1) if m.group(1).isdigit() else "1"
        bbox_cc = _find_word_bbox(words, m.group(0), img_w, img_h, z)
        label.copy_count_inline = _field(
            f"{digit}({m.group(2)})", m.group(0), conf, z,
            bbox=bbox_cc, pattern_match=True
        )

    # ── Product name ──────────────────────────────────────────────────────────
    zone_top_y    = min((w.bbox.y for w in words), default=0)
    zone_bottom_y = max((w.bbox.y + w.bbox.h for w in words), default=img_h)
    zone_mid_y    = zone_top_y + (zone_bottom_y - zone_top_y) * 0.55

    top_words = [w for w in words if (w.bbox.y + w.bbox.h / 2) <= zone_mid_y]
    top_lines  = _group_into_lines(top_words)

    name_words: list[_OcrWord] = []

    for line in top_lines:
        candidates: list[_OcrWord] = []
        for w in line:
            sub_tokens = w.text.split()
            valid_sub = [t for t in sub_tokens if _is_product_name_token(t)]
            if not valid_sub:
                if candidates:
                    break
                continue
            candidates.append(w)
        if candidates:
            name_words = candidates
            break

    for w in top_words:
        parts = w.text.split()
        if (len(parts) >= 1
                and all(_is_product_name_token(p) for p in parts)
                and len(parts) >= len(name_words)):
            if not name_words or w.bbox.y <= name_words[0].bbox.y:
                name_words = [w]

    if name_words:
        def _dedup_name(ws: list[_OcrWord]) -> list[_OcrWord]:
            if len(ws) <= 1:
                return ws
            tok_sets = [set(w.text.upper().split()) for w in ws]
            step1 = [
                w for i, w in enumerate(ws)
                if not any(
                    i != j and tok_sets[i] < tok_sets[j]
                    for j in range(len(ws))
                )
            ] or ws
            all_toks: set[str] = set()
            for w in step1:
                for tok in w.text.upper().split():
                    all_toks.add(tok)
            filtered: list[_OcrWord] = []
            for w in step1:
                upper = w.text.upper()
                if " " not in upper:
                    is_concat = any(
                        upper[:i] in all_toks and upper[i:] in all_toks
                        for i in range(2, len(upper) - 1)
                    )
                    if is_concat:
                        continue
                filtered.append(w)
            return filtered or step1

        name_words = _dedup_name(name_words)
        raw_name = " ".join(
            " ".join(t for t in w.text.split() if _is_product_name_token(t))
            for w in name_words
        ).strip()
        if raw_name:
            norm = raw_name.upper()
            bbox_name = _union_bbox([w.bbox for w in name_words], img_w, img_h, z)
            avg_c = sum(w.conf for w in name_words) / len(name_words)
            label.product_name = _field(norm, raw_name.upper(), avg_c, z, bbox=bbox_name)

    # ── Metric dimensions ─────────────────────────────────────────────────────
    dim_metric_words: list[_OcrWord] = []
    dim_metric_val: Optional[str] = None
    for w in words:
        m_dm = _RE_DIM_METRIC.search(w.text)
        if m_dm:
            dim_metric_val = m_dm.group(0)
            dim_metric_words.append(w)
            break
    if dim_metric_val is None:
        m_dm = _RE_DIM_METRIC.search(text)
        if m_dm:
            dim_metric_val = m_dm.group(0)
            dim_metric_words = [w for w in words if _RE_DIM_CM.search(w.text)]
    if dim_metric_val is None:
        for w in words:
            m_cm = _RE_DIM_CM.search(w.text)
            if m_cm:
                dim_metric_val = m_cm.group(0)
                dim_metric_words = [w]
                break
    if dim_metric_val:
        bbox_dm = _union_bbox([w.bbox for w in dim_metric_words], img_w, img_h, z) \
            if dim_metric_words else None
        label.product_dimensions_metric = _field(
            dim_metric_val, dim_metric_val, conf, z,
            bbox=bbox_dm, pattern_match=True,
        )

    # ── Imperial dimensions ───────────────────────────────────────────────────
    dim_imp_val: Optional[str] = None
    dim_imp_words: list[_OcrWord] = []
    for w in words:
        m_di = _RE_DIM_IMPERIAL.search(w.text)
        if m_di:
            dim_imp_val = m_di.group(0)
            dim_imp_words.append(w)
            break
    if dim_imp_val is None:
        m_di = _RE_DIM_IMPERIAL.search(text)
        if m_di:
            dim_imp_val = m_di.group(0)
            dim_imp_words = [w for w in words if _RE_DIM_IMPERIAL.search(w.text)]
    if dim_imp_val:
        bbox_di = _union_bbox([w.bbox for w in dim_imp_words], img_w, img_h, z) \
            if dim_imp_words else None
        label.product_dimensions_imperial = _field(
            dim_imp_val, dim_imp_val, conf, z,
            bbox=bbox_di, pattern_match=True,
        )

    # ── Legacy combined dimensions field ──────────────────────────────────────
    dim_words: list[_OcrWord] = []
    dim_vals: list[str] = []
    for w in words:
        m_cm = _RE_DIM_CM.search(w.text)
        if m_cm:
            dim_vals.append(m_cm.group(0))
            dim_words.append(w)
    if not dim_vals:
        for w in words:
            solo = re.findall(r"(?<!\d)(\d{2,4})(?!\d)", w.text)
            for d in solo:
                if 10 <= int(d) <= 600:
                    dim_vals.append(f"{d} cm")
                    dim_words.append(w)
                    break
    if dim_vals:
        dim_str = ", ".join(dim_vals)
        bbox_dim = _union_bbox([w.bbox for w in dim_words], img_w, img_h, z) if dim_words else None
        label.dimensions = _field(dim_str, text, conf, z, bbox=bbox_dim)

    # ── Gross weight ──────────────────────────────────────────────────────────
    gross_val: Optional[str] = None
    gross_words: list[_OcrWord] = []
    for w in words:
        mg = _RE_GROSS_KG.search(w.text) or _RE_GROSS_LBS.search(w.text)
        if mg:
            gross_val = w.text
            gross_words.append(w)
    if gross_val is None:
        mg_kg  = _RE_GROSS_KG.search(text)
        mg_lbs = _RE_GROSS_LBS.search(text)
        if mg_kg:
            gross_val = f"Gross {mg_kg.group(1)} kg"
            gross_words = [w for w in words if re.search(r"\bGross\b", w.text, re.IGNORECASE)]
        elif mg_lbs:
            gross_val = f"Gross {mg_lbs.group(1)} lbs"
            gross_words = [w for w in words if re.search(r"\bGross\b", w.text, re.IGNORECASE)]
    if gross_val is None:
        for w in words:
            m_kg = _RE_WEIGHT_KG.search(w.text)
            if m_kg and not re.search(r"\b(?:Net|net)\b", w.text):
                gross_val = f"{m_kg.group(1)} kg"
                m_lbs = _RE_WEIGHT_LBS.search(w.text)
                if m_lbs:
                    gross_val += f" ({m_lbs.group(1)} lbs)"
                gross_words = [w]
                break
    if gross_val:
        bbox_gw = _union_bbox([w.bbox for w in gross_words], img_w, img_h, z) \
            if gross_words else None
        label.gross_weight = _field(gross_val, gross_val, conf, z, bbox=bbox_gw, pattern_match=True)

    # ── Net weight ────────────────────────────────────────────────────────────
    net_val: Optional[str] = None
    net_words: list[_OcrWord] = []
    for w in words:
        mn = _RE_NET_KG.search(w.text) or _RE_NET_LBS.search(w.text)
        if mn:
            net_val = w.text
            net_words.append(w)
    if net_val is None:
        mn_kg  = _RE_NET_KG.search(text)
        mn_lbs = _RE_NET_LBS.search(text)
        if mn_kg:
            net_val = f"Net {mn_kg.group(1)} kg"
            net_words = [w for w in words if re.search(r"\bNet\b", w.text, re.IGNORECASE)]
        elif mn_lbs:
            net_val = f"Net {mn_lbs.group(1)} lbs"
            net_words = [w for w in words if re.search(r"\bNet\b", w.text, re.IGNORECASE)]
    if net_val is None and label.gross_weight is not None:
        for w in words:
            m_lbs = _RE_WEIGHT_LBS.search(w.text)
            m_kg  = _RE_WEIGHT_KG.search(w.text)
            if m_lbs and not m_kg:
                net_val = f"{m_lbs.group(1)} lbs"
                net_words = [w]
                break
    if net_val:
        bbox_nw = _union_bbox([w.bbox for w in net_words], img_w, img_h, z) \
            if net_words else None
        label.net_weight = _field(net_val, net_val, conf, z, bbox=bbox_nw, pattern_match=True)

    # ── Legacy combined weight field ──────────────────────────────────────────
    wt_words: list[_OcrWord] = []
    wt_parts: list[str] = []
    for w in words:
        wt_kg  = _RE_WEIGHT_KG.search(w.text)
        wt_lbs = _RE_WEIGHT_LBS.search(w.text)
        if wt_kg:
            wt_parts.append(f"{wt_kg.group(1)} kg")
            wt_words.append(w)
        elif wt_lbs:
            wt_parts.append(f"{wt_lbs.group(1)} lbs")
            wt_words.append(w)
    if not wt_parts:
        wt_kg  = _RE_WEIGHT_KG.search(text)
        wt_lbs = _RE_WEIGHT_LBS.search(text)
        if wt_kg:
            wt_parts.append(f"{wt_kg.group(1)} kg")
        if wt_lbs:
            wt_parts.append(f"{wt_lbs.group(1)} lbs")
    if wt_parts:
        bbox_wt = _union_bbox([w.bbox for w in wt_words], img_w, img_h, z) if wt_words else None
        label.weight = _field(" / ".join(wt_parts), text, conf, z, bbox=bbox_wt)

    # ── Quantity multiplier ───────────────────────────────────────────────────
    m_qty = _RE_QTY_MULT.search(text)
    if m_qty:
        qty_words = [w for w in words if _RE_QTY_MULT.search(w.text)]
        bbox_qty = _union_bbox([w.bbox for w in qty_words], img_w, img_h, z) \
            if qty_words else None
        label.quantity_multiplier = _field(
            f"×{m_qty.group(1)}", m_qty.group(0), conf, z,
            bbox=bbox_qty, pattern_match=True,
        )

    # ── Package type ──────────────────────────────────────────────────────────
    m_pt = _RE_PKG_TYPE.search(text)
    if m_pt:
        pt_words = [w for w in words if _RE_PKG_TYPE.search(w.text)]
        bbox_pt = _union_bbox([w.bbox for w in pt_words], img_w, img_h, z) \
            if pt_words else None
        label.package_type = _field(
            m_pt.group(0), m_pt.group(0), conf, z,
            bbox=bbox_pt, pattern_match=True,
        )

    # ── Package number ────────────────────────────────────────────────────────
    dim_digits: set[str] = set()
    for fld in (
        label.product_dimensions_metric, label.dimensions,
        label.gross_weight, label.net_weight, label.weight,
    ):
        if fld and fld.value:
            for d in re.findall(r"\d+", fld.value):
                dim_digits.add(d)

    used_vals: set[str | None] = {
        label.date_stamp.value if label.date_stamp else None,
        label.article_number.value if label.article_number else None,
    } | dim_digits

    for w in words:
        if re.search(r"\b(?:cm|mm|kg|lbs?)\b", w.text, re.IGNORECASE):
            continue
        if re.search(r"[/x×X'\".]", w.text):
            continue
        clean_w = w.text.strip()
        if not re.match(r"^\d{2,4}$", clean_w):
            continue
        if clean_w not in used_vals:
            label.package_number = _field(
                clean_w, clean_w, conf * 0.7, z,
                bbox=w.bbox, pattern_match=True,
            )
            break

    # ── IKEA logo text detection ──────────────────────────────────────────────
    logo_words = [
        w for w in words
        if re.search(r"\bIKEA\b|Design\s+and\s+Quality|of\s+Sweden", w.text, re.IGNORECASE)
    ]
    if logo_words:
        bbox_logo = _union_bbox([w.bbox for w in logo_words], img_w, img_h, z)
        logo_text = " ".join(w.text for w in logo_words)
        label.ikea_logo = _field("IKEA logo detected", logo_text, 0.90, z, bbox=bbox_logo)

    # ── Article code from ITF-14 ──────────────────────────────────────────────
    itf = barcodes.get("itf14")
    if itf and itf.raw_data:
        digits = re.sub(r"\D", "", itf.raw_data)
        if len(digits) >= 13:
            art_code = digits[8:13]
            if art_code:
                label.article_code = _field(
                    art_code, digits, 1.0, z,
                    bbox=itf.bbox, source="barcode", pattern_match=True
                )


def _extr_center_barcode(
    label: ExtractedLabel,
    words: list[_OcrWord],
    img_w: int,
    img_h: int,
    all_words: list[_OcrWord] | None = None,
) -> None:
    z = "center_barcode"
    words = _resolve_words(words, all_words)
    text = _words_to_text(words)
    conf = 0.75

    m = _RE_ARTICLE_NUMBER.search(text)
    if m:
        art = f"{m.group(1)}.{m.group(2)}.{m.group(3)}"
        bbox_art = _find_word_bbox(words, m.group(0), img_w, img_h, z)
        label.article_number = _field(art, m.group(0), conf, z, bbox=bbox_art, pattern_match=True)
        label.itf14_human_readable = _field(art, m.group(0), conf, z, bbox=bbox_art)

    if label.article_number is None and label.itf14_barcode:
        digits = re.sub(r"\D", "", label.itf14_barcode.raw_data)
        if len(digits) >= 8:
            d = digits[:8]
            art = f"{d[0:3]}.{d[3:6]}.{d[6:8]}"
            label.article_number = _field(
                art, digits, 0.95, z,
                bbox=label.itf14_barcode.bbox, source="barcode", pattern_match=True
            )
            label.itf14_human_readable = _field(
                art, digits, 0.95, z,
                bbox=label.itf14_barcode.bbox, source="barcode"
            )

    art_digits = re.sub(r"\D", "", label.article_number.value) \
        if label.article_number and label.article_number.value else ""
    art_code_val = label.article_code.value if label.article_code else None
    date_val = label.date_stamp.value if label.date_stamp else None
    for w in words:
        m_in = _RE_ITEM_NUM.search(w.text)
        if m_in:
            val = m_in.group(1)
            if val == date_val or val in art_digits or val == art_code_val:
                continue
            label.internal_item_number = _field(
                val, val, conf * 0.8, z, bbox=w.bbox, pattern_match=True
            )
            break


def _extr_datamatrix(
    label: ExtractedLabel,
    words: list[_OcrWord],
    barcodes: dict,
    img_w: int,
    img_h: int,
    all_words: list[_OcrWord] | None = None,
) -> None:
    z = "center_datamatrix"
    words = _resolve_words(words, all_words)
    text = _words_to_text(words)
    conf = 0.75

    ocr_ais = parse_parentheses_form(text)
    if ocr_ais:
        existing = label.datamatrix_content.ai_fields if label.datamatrix_content else {}
        merged = merge_ai_dicts(ocr_ais, existing)
        if label.datamatrix_content:
            label.datamatrix_content.ai_fields = merged
        _promote_ai_fields(label, merged, conf, "ocr")

    date_line = re.search(
        r"(\d{2})[\-./](\d{2})[\-./](\d{2})\s*\(YY[\-.]?MM[\-.]?DD\)",
        text, re.IGNORECASE
    )
    if date_line:
        yy, mm, dd = date_line.group(1), date_line.group(2), date_line.group(3)
        formatted = f"{yy}-{mm}-{dd}"
        date_wds = [w for w in words if re.search(r"\d{2}[-./]\d{2}[-./]\d{2}", w.text)]
        bbox_hd = _union_bbox([w.bbox for w in date_wds], img_w, img_h, z) if date_wds else None
        label.human_readable_date = _field(
            formatted, date_line.group(0), conf, z,
            bbox=bbox_hd, source="ocr", pattern_match=True,
        )
        if label.ai_13 is None:
            label.ai_13 = _field(formatted, f"{yy}{mm}{dd}", conf, z,
                                  source="ocr", pattern_match=True)


def _extr_address(
    label: ExtractedLabel,
    addr_words: list[_OcrWord],
    dm_words: list[_OcrWord],
    img_w: int,
    img_h: int,
    global_words: list[_OcrWord] | None = None,
) -> None:
    z = "center_address"
    conf = 0.85
    if len(addr_words) + len(dm_words) < 3 and global_words:
        addr_words = global_words
        dm_words = []
    all_words_local = dm_words + addr_words
    combined_text = _words_to_text(all_words_local).lower()
    addr_text = _words_to_text(addr_words)

    # ── Origin text ───────────────────────────────────────────────────────────
    verb = "Assembled in" if re.search(r"\bassembled\s+in\b", combined_text) else "Made in"

    def _has(*tokens) -> bool:
        return all(t in combined_text for t in tokens)

    has_republic = any(t in combined_text for t in ["republic", "publi", "repub"])
    origin_country: Optional[str] = None

    if _has("people", "china") or (_has("people") and has_republic):
        origin_country = "People's Republic of China"
    elif has_republic and "china" in combined_text:
        origin_country = "Republic of China"
    elif "china" in combined_text:
        origin_country = "China"
    elif "vietnam" in combined_text:
        origin_country = "Vietnam"
    elif "poland" in combined_text:
        origin_country = "Poland"
    elif "india" in combined_text and "indiana" not in combined_text:
        origin_country = "India"
    elif "united states" in combined_text or " usa" in combined_text:
        origin_country = "United States"

    if origin_country:
        origin_words = [
            w for w in all_words_local
            if re.search(
                r"\bmade\b|\bassembled\b|\bchina\b|\bvietnam\b|\bpoland\b"
                r"|\bindia\b|\bpeople\b|\brepublic\b|\bstates\b|\busa\b",
                w.text, re.IGNORECASE
            )
        ]
        bbox_ori = _union_bbox([w.bbox for w in origin_words], img_w, img_h, z) \
            if origin_words else None
        label.origin_text = _field(
            f"{verb} {origin_country}", combined_text, conf, z,
            bbox=bbox_ori, pattern_match=True
        )

    # ── Address block ─────────────────────────────────────────────────────────
    _ORIGIN_TOKENS_RE = re.compile(
        r"\b(?:Made|Assembled|Republic|China|Vietnam|Poland|India|USA)\b",
        re.IGNORECASE,
    )
    addr_candidate_words = [
        w for w in addr_words
        if _ADDR_ANCHOR_RE.search(w.text) and not _ORIGIN_TOKENS_RE.search(w.text)
    ]

    addr_lines_text = []
    collecting = False
    for line in _group_into_lines(addr_words):
        ln = " ".join(w.text for w in line).strip()
        if not ln:
            continue
        if not collecting and _ADDR_ANCHOR_RE.search(ln):
            collecting = True
        if collecting:
            if _ADDR_STOP_RE.search(ln):
                break
            ln = re.sub(r"^\d{1,3}\s+(?=I?KEA|CEA\b)", "", ln, flags=re.IGNORECASE)
            addr_lines_text.append(ln)

    if not addr_lines_text and addr_candidate_words:
        anchor_sorted = sorted(addr_candidate_words, key=lambda w: (w.bbox.y, w.bbox.x))
        addr_lines_text = [w.text.strip() for w in anchor_sorted if w.text.strip()]

    if addr_lines_text:
        addr_str = "\n".join(addr_lines_text)
        addr_anchor_words = addr_candidate_words if addr_candidate_words else [
            w for w in addr_words if _ADDR_ANCHOR_RE.search(w.text)
        ]
        bbox_addr = _union_bbox([w.bbox for w in addr_anchor_words], img_w, img_h, z) \
            if addr_anchor_words else None
        label.address_block = _field(addr_str, addr_str, conf, z, bbox=bbox_addr)

    # ── Copyright ─────────────────────────────────────────────────────────────
    copyright_found = False
    for src in (addr_text, _words_to_text(all_words_local)):
        m = _RE_COPYRIGHT.search(src)
        if m:
            copy_words = [
                w for w in addr_words
                if re.search(r"\bInter\b|\bI?KEA\b|\bSystems\b|\bB\.V\b", w.text, re.IGNORECASE)
            ]
            bbox_copy = _union_bbox([w.bbox for w in copy_words], img_w, img_h, z) \
                if copy_words else None
            label.copyright_notice = _field(m.group(0), m.group(0), conf, z, bbox=bbox_copy)
            copyright_found = True
            break

    if not copyright_found:
        has_inter_ikea = re.search(r"\bInter\s+I?KEA\b", addr_text, re.IGNORECASE)
        has_bv = re.search(r"\bB\.?V\.?\b", addr_text, re.IGNORECASE)
        if has_inter_ikea and has_bv:
            copy_words = [
                w for w in addr_words
                if re.search(r"\bInter\b|\bIKEA\b|\bSystems\b|\bB\.?V\.?\b", w.text, re.IGNORECASE)
            ]
            bbox_copy = _union_bbox([w.bbox for w in copy_words], img_w, img_h, z) \
                if copy_words else None
            year_m = re.search(r"\b(20\d{2})\b", addr_text)
            year_str = f" {year_m.group(1)}" if year_m else ""
            label.copyright_notice = _field(
                f"© Inter IKEA Systems B.V.{year_str}",
                addr_text, conf * 0.8, z, bbox=bbox_copy,
            )

    # ── Plant identifier ──────────────────────────────────────────────────────
    for src_words, zone_tag in ((dm_words, "center_datamatrix"), (addr_words, z)):
        src_text = _words_to_text(src_words)
        m_pi = _RE_PLANT_ID.search(src_text)
        if m_pi:
            pi_words = [w for w in src_words if _RE_PLANT_ID.search(w.text)]
            if not pi_words:
                pi_words = [w for w in src_words
                            if re.search(r"\bPI\b", w.text, re.IGNORECASE)]
            bbox_pi = _union_bbox([w.bbox for w in pi_words], img_w, img_h, zone_tag) \
                if pi_words else None
            label.plant_identifier = _field(
                m_pi.group(0), m_pi.group(0), conf, zone_tag,
                bbox=bbox_pi, pattern_match=True,
            )
            label.patent_info = label.plant_identifier
            break

    if label.plant_identifier is None:
        for src_words, zone_tag in ((dm_words, "center_datamatrix"), (addr_words, z)):
            src_text = _words_to_text(src_words)
            m = _RE_PATENT.search(src_text)
            if m:
                pat_words = [w for w in src_words
                             if re.search(r"P[Ii1l]|patent", w.text, re.IGNORECASE)]
                bbox_pat = _union_bbox([w.bbox for w in pat_words], img_w, img_h, zone_tag) \
                    if pat_words else None
                label.plant_identifier = _field(
                    m.group(0).strip(), m.group(0).strip(),
                    conf, zone_tag, bbox=bbox_pat, pattern_match=True,
                )
                label.patent_info = label.plant_identifier
                break

    # ── Supplier name ─────────────────────────────────────────────────────────
    for src_words, zone_tag in ((addr_words, z), (dm_words, "center_datamatrix")):
        src_text = _words_to_text(src_words)
        m_sn = _RE_SUPPLIER_NAME.search(src_text)
        if m_sn:
            sn_words = [w for w in src_words
                        if re.search(r"\bI?KEA\b", w.text, re.IGNORECASE)]
            bbox_sn = _union_bbox([w.bbox for w in sn_words], img_w, img_h, zone_tag) \
                if sn_words else None
            label.supplier_name = _field(
                m_sn.group(0), m_sn.group(0), conf, zone_tag,
                bbox=bbox_sn, pattern_match=True,
            )
            break

    # ── Supplier address ──────────────────────────────────────────────────────
    for src_words, zone_tag in ((addr_words, z), (dm_words, "center_datamatrix")):
        src_text = _words_to_text(src_words)
        m_sa = _RE_SUPPLIER_ADDR.search(src_text)
        if m_sa:
            sa_words = [w for w in src_words
                        if re.search(r"\bSE\b|\d{5}", w.text, re.IGNORECASE)]
            bbox_sa = _union_bbox([w.bbox for w in sa_words], img_w, img_h, zone_tag) \
                if sa_words else None
            label.supplier_address = _field(
                m_sa.group(0), m_sa.group(0), conf, zone_tag,
                bbox=bbox_sa, pattern_match=True,
            )
            break


def _extr_compliance(
    label: ExtractedLabel,
    words: list[_OcrWord],
    img_w: int,
    img_h: int,
    all_words: list[_OcrWord] | None = None,
) -> None:
    z = "right_compliance"
    conf = 0.65
    words = _resolve_words(words, all_words)
    text = _words_to_text(words)

    marks: list[str] = []
    mark_words: list[_OcrWord] = []
    for mark_name, pattern in _COMPLIANCE_KEYWORDS.items():
        if pattern.search(text):
            marks.append(mark_name)
            mark_words += [w for w in words if pattern.search(w.text)]

    m = _RE_AGE_RATING.search(text)
    if m:
        age_wds = [w for w in words if _RE_AGE_RATING.search(w.text)]
        bbox_age = _union_bbox([w.bbox for w in age_wds], img_w, img_h, z) if age_wds else None
        label.age_rating = _field(
            m.group(1).replace(" ", ""), m.group(1), conf, z, bbox=bbox_age
        )

    if marks or text.strip():
        bbox_comp = _union_bbox([w.bbox for w in mark_words], img_w, img_h, z) if mark_words else None
        label.compliance_marks = _field(
            ", ".join(marks) if marks else "",
            text.strip(), conf, z, bbox=bbox_comp,
        )


def _extr_date(
    label: ExtractedLabel,
    words: list[_OcrWord],
    img_w: int,
    img_h: int,
    all_words: list[_OcrWord] | None = None,
) -> None:
    z = "right_date"
    conf = 0.88
    words = _resolve_words(words, all_words)
    text = _words_to_text(words)

    m = _RE_DATE_LABEL.search(text)
    if m:
        label.date_label_present = True
        bw = _find_word_bbox(words, m.group(1), img_w, img_h, z)
        label.date_stamp = _field(m.group(1), m.group(1), conf, z, bbox=bw, pattern_match=True)
    else:
        m2 = _RE_DATE_PARTIAL.search(text)
        if m2:
            label.date_label_present = True
            bw = _find_word_bbox(words, m2.group(1), img_w, img_h, z)
            label.date_stamp = _field(m2.group(1), m2.group(1), conf * 0.9, z,
                                      bbox=bw, pattern_match=True)

    if label.date_stamp is None:
        m3 = _RE_YYWW_BARE.search(text)
        if m3:
            bw = _find_word_bbox(words, m3.group(1), img_w, img_h, z)
            label.date_stamp = _field(m3.group(1), m3.group(1), conf * 0.7, z,
                                      bbox=bw, pattern_match=False)

    all_tokens: list[_OcrWord] = []
    for w in words:
        parts = w.text.split()
        if len(parts) > 1:
            sub_w = max(1, w.bbox.w // len(parts))
            for i, part in enumerate(parts):
                sub_bbox = _make_bbox(
                    w.bbox.x + i * sub_w, w.bbox.y, sub_w, w.bbox.h, img_w, img_h, z
                )
                all_tokens.append(_OcrWord(text=part, conf=w.conf, bbox=sub_bbox, zone=z))
        else:
            all_tokens.append(w)

    for tok in all_tokens:
        t = tok.text.strip()
        if (re.match(r"^\d{3}$", t)
                and label.date_numeric_prefix is None
                and (label.date_stamp is None or t != label.date_stamp.value)):
            label.date_numeric_prefix = _field(t, t, conf, z, bbox=tok.bbox)
        m_a = re.match(r"^([a-zA-Z]{4})$", t)
        if m_a and label.date_alpha_code is None:
            code = m_a.group(1).lower()
            label.date_alpha_code = _field(code, m_a.group(1), conf, z, bbox=tok.bbox)

    m_ci = _RE_CUSTOM_ID.search(text)
    if m_ci:
        ci_val = f"{m_ci.group(1)}-{m_ci.group(2)}-{m_ci.group(3)}"
        ci_words = [w for w in words if _RE_CUSTOM_ID.search(w.text)]
        bbox_ci = _union_bbox([w.bbox for w in ci_words], img_w, img_h, z) if ci_words else None
        label.custom_identifier = _field(ci_val, m_ci.group(0), conf, z,
                                         bbox=bbox_ci, pattern_match=True)
    elif label.date_alpha_code and label.date_stamp and label.date_numeric_prefix:
        ci_val = (f"{label.date_alpha_code.value}-"
                  f"{label.date_stamp.value}-"
                  f"{label.date_numeric_prefix.value}")
        label.custom_identifier = _field(ci_val, ci_val, conf * 0.85, z, pattern_match=True)


def _extr_copy_count(
    label: ExtractedLabel,
    words: list[_OcrWord],
    img_w: int,
    img_h: int,
    all_words: list[_OcrWord] | None = None,
) -> None:
    z = "right_copy_count"
    conf = 0.80
    text = _words_to_text(words)
    m = re.search(r"\b(\d{1,2})\b", text)
    if m:
        bw = words[0].bbox if words else None
        label.copy_count = _field(m.group(1), text.strip(), conf, z, bbox=bw, pattern_match=True)


# ── Overall confidence ────────────────────────────────────────────────────────

def _overall_confidence(label: ExtractedLabel) -> float:
    """
    Weighted confidence score based on the quality of found fields.

    Core fields (product_name, article_number, itf14_barcode) are weighted 2×
    because they are present on every IKEA label.  Absence of standard fields
    does NOT penalise the score — not every label type carries all fields.
    """
    core_fields = [label.product_name, label.article_number, label.itf14_barcode]
    standard_fields = [
        label.product_dimensions_metric, label.article_code, label.internal_item_number,
        label.quantity_multiplier, label.gross_weight, label.net_weight,
        label.package_number, label.package_type, label.ean_barcode_value,
        label.datamatrix_barcode, label.human_readable_date, label.plant_identifier,
        label.supplier_name, label.supplier_address, label.address_block,
        label.origin_text, label.copyright_notice, label.compliance_marks,
        label.date_stamp, label.custom_identifier, label.copy_count,
    ]
    weighted_sum = 0.0
    weight_total = 0.0
    for f in core_fields:
        if f is not None:
            weighted_sum += getattr(f, "confidence", 1.0) * 2.0
            weight_total += 2.0
    for f in standard_fields:
        if f is not None:
            weighted_sum += getattr(f, "confidence", 1.0)
            weight_total += 1.0
    if weight_total == 0.0:
        return 0.0
    return round(weighted_sum / weight_total, 3)


# ── Public entry point ────────────────────────────────────────────────────────

def extract_fields(
    image,
    ocr_words: list[_OcrWord],
    barcodes: dict,
    zone_cfg: list,
    raw_elements: list[RawElement],
    source_path: Path,
    dpi: int,
    warnings: list[str],
    errors: list[str],
    libraries_used: dict | None = None,
    min_conf: float = MIN_OCR_CONFIDENCE,
) -> ExtractedLabel:
    """
    Map OCR words + barcodes → ExtractedLabel.

    Confidence gating
    -----------------
    Words with confidence below ``min_conf`` are excluded from field extraction
    but are still stored in ``label.all_word_boxes`` for display.  PDF-native
    words (conf=1.0) are always kept regardless of the threshold — they come
    from the vector data and are perfectly accurate.

    Parameters
    ----------
    image        : PIL Image of the label (for size metadata only)
    ocr_words    : merged word list from ocr_engine.run_ocr()
    barcodes     : dict from decode_barcodes() {"itf14", "datamatrix", "ean13"}
    zone_cfg     : active zone boundary table (list of (zone_id, f0, f1, rot))
    raw_elements : list of RawElement from run_pdf_passes()
    source_path  : Path to the original source file
    dpi          : render DPI used during load
    warnings     : mutable list — extraction warnings are appended here
    errors       : mutable list — extraction errors are appended here
    libraries_used : dict describing which libraries are active (for metadata)
    min_conf     : minimum OCR confidence; lower words go to display only
    """
    img_w, img_h = image.size

    # ── Confidence gating ──────────────────────────────────────────────────────
    # PDF words always have conf=1.0 so they pass unconditionally.
    # OCR words below the threshold are kept for display but skipped for field matching.
    high_conf_words = [
        w for w in ocr_words
        if w.conf >= min_conf or w.conf == 1.0
    ]
    low_conf_count = len(ocr_words) - len(high_conf_words)
    if low_conf_count:
        logger.debug(
            "Confidence gating: excluded %d/%d low-confidence OCR words (threshold=%.2f)",
            low_conf_count, len(ocr_words), min_conf,
        )

    label = ExtractedLabel(
        metadata=ExtractionMetadata(
            source_file=source_path.name,
            extracted_at=datetime.now(timezone.utc).isoformat(),
            image_width_px=img_w,
            image_height_px=img_h,
            render_dpi=dpi,
            libraries_used=libraries_used or {},
        )
    )

    # Group high-confidence words by zone
    by_zone: dict[str, list[_OcrWord]] = {z[0]: [] for z in zone_cfg}
    by_zone.setdefault("unknown", [])
    for w in high_conf_words:
        by_zone.setdefault(w.zone, []).append(w)

    # ZoneSummary objects (built from ALL words, including low-confidence)
    all_by_zone: dict[str, list[_OcrWord]] = {z[0]: [] for z in zone_cfg}
    all_by_zone.setdefault("unknown", [])
    for w in ocr_words:
        all_by_zone.setdefault(w.zone, []).append(w)
    for zone_id, zone_words in all_by_zone.items():
        lines = _group_into_lines(zone_words)
        raw_lines = [" ".join(w.text for w in line) for line in lines]
        avg_conf = (sum(w.conf for w in zone_words) / len(zone_words)) if zone_words else 0.0
        label.zones[zone_id] = ZoneSummary(
            zone_id=zone_id,
            raw_lines=raw_lines,
            word_count=len(zone_words),
            avg_confidence=round(avg_conf, 3),
        )

    # All word boxes — show ALL words in the UI (including low-confidence)
    label.all_word_boxes = [
        {
            "text": w.text,
            "x": w.bbox.x,
            "y": w.bbox.y,
            "w": w.bbox.w,
            "h": w.bbox.h,
            "conf": round(w.conf, 3),
            "zone": w.zone,
        }
        for w in ocr_words
    ]
    label.raw_elements = list(raw_elements)

    # ── Field extraction (using high-confidence words only) ───────────────────
    all_hc = high_conf_words  # full set alias for layout-agnostic fallback

    _extr_barcodes(label, barcodes, img_w, img_h)
    _extr_left_identity(label, by_zone.get("left_identity", []) or by_zone.get("top_identity", []),
                        barcodes, img_w, img_h, all_hc)
    _extr_center_barcode(label, by_zone.get("center_barcode", []) or by_zone.get("mid_barcode", []),
                         img_w, img_h, all_hc)
    _extr_datamatrix(label, by_zone.get("center_datamatrix", []),
                     barcodes, img_w, img_h, all_hc)
    _extr_address(
        label,
        by_zone.get("center_address", []) or by_zone.get("mid_address", []),
        by_zone.get("center_datamatrix", []),
        img_w, img_h,
        all_hc,
    )
    _extr_compliance(label,
                     by_zone.get("right_compliance", []) or by_zone.get("bot_compliance", []),
                     img_w, img_h, all_hc)
    _extr_date(label,
               by_zone.get("right_date", []) or by_zone.get("bot_date", []),
               img_w, img_h, all_hc)
    _extr_copy_count(label, by_zone.get("right_copy_count", []), img_w, img_h, all_hc)

    label.metadata.warnings = warnings
    label.metadata.errors = errors
    label.metadata.overall_confidence = _overall_confidence(label)
    return label
