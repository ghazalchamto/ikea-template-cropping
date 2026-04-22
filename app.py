"""
IKEA Label Validation Dashboard

Upload a production label (PDF / image) and get a full breakdown:
  - Annotated label image with per-field bounding boxes
  - Field-level validation against label-rules/ schemas
  - Layout & zone compliance  (coming soon)
  - Overlap detection         (coming soon)
  - Completeness & consistency (coming soon)

Run:
    streamlit run app.py
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Optional

import numpy as _np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from config import (
    DEFAULT_DPI, UI_DPI_MIN, UI_DPI_MAX, UI_DPI_DEFAULT, UI_DPI_STEP,
    ALLOWED_UPLOAD_TYPES,
)
from extractor import LabelExtractor, warmup_ocr_readers
from extractor.models import ExtractedLabel, ExtractedField
from validator import (
    auto_validate,
    check_layout,
    detect_overlaps,
    check_completeness,
)
from validator.models import ValidationReport, TypeMatch

logging.getLogger("extractor").setLevel(logging.WARNING)
logging.getLogger("validator").setLevel(logging.WARNING)


# ── JSON encoder ──────────────────────────────────────────────────────────────

class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, _np.integer):  return int(obj)
        if isinstance(obj, _np.floating): return float(obj)
        if isinstance(obj, _np.ndarray):  return obj.tolist()
        return super().default(obj)


# ── OCR warm-up (runs once per Streamlit session) ─────────────────────────────

@st.cache_resource(show_spinner="Loading OCR models…")
def _cached_warmup_ocr():
    return warmup_ocr_readers()

_cached_warmup_ocr()


# ── Section / field colour map ────────────────────────────────────────────────

_SECTION_COLORS = {
    "identity":   (59,  130, 246),
    "barcodes":   (234, 88,  12),
    "address":    (168, 85,  247),
    "compliance": (236, 72,  153),
    "date":       (20,  184, 166),
    "copy":       (245, 158, 11),
}

_FIELD_SECTION = {
    "product_name": "identity", "product_dimensions_metric": "identity",
    "product_dimensions_imperial": "identity", "article_number": "identity",
    "article_code": "identity", "internal_item_number": "identity",
    "quantity_multiplier": "identity", "gross_weight": "identity",
    "net_weight": "identity", "package_number": "identity",
    "package_type": "identity", "ikea_logo": "identity",
    "ean_barcode": "barcodes", "ean_barcode_value": "barcodes",
    "itf14_barcode": "barcodes", "itf14_human_readable": "barcodes",
    "datamatrix_barcode": "barcodes", "ai_240": "barcodes",
    "ai_13": "barcodes", "ai_11": "barcodes", "ai_10": "barcodes",
    "human_readable_date": "date", "date_stamp": "date",
    "date_alpha_code": "date", "date_numeric_prefix": "date",
    "custom_identifier": "date",
    "plant_identifier": "address", "supplier_name": "address",
    "supplier_address": "address", "address_block": "address",
    "origin_text": "address", "copyright_notice": "address",
    "compliance_marks": "compliance", "age_rating": "compliance",
    "copy_count": "copy", "copy_count_inline": "copy",
}


# ── File helpers ──────────────────────────────────────────────────────────────

def _sniff_suffix(data: bytes) -> str | None:
    if data[:4] == b"%PDF":             return ".pdf"
    if data[:3] == b"\xff\xd8\xff":    return ".jpg"
    if data[:8] == b"\x89PNG\r\n\x1a\n": return ".png"
    if data[:4] in (b"II*\x00", b"MM\x00*"): return ".tif"
    if data[:2] == b"BM":              return ".bmp"
    return None


def _materialise(data: bytes) -> str:
    if not data:
        raise ValueError("Uploaded file is empty.")
    sniff = _sniff_suffix(data)
    if sniff in {".pdf", ".jpg", ".png", ".tif", ".bmp"}:
        fd, p = tempfile.mkstemp(suffix=sniff, prefix="ikea_")
        os.close(fd)
        Path(p).write_bytes(data)
        return p
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
    except Exception as e:
        raise ValueError(f"Cannot read image: {e}") from e
    fd, p = tempfile.mkstemp(suffix=".png", prefix="ikea_")
    os.close(fd)
    im.convert("RGB").save(p, "PNG")
    return p


def _get_preview_image(data: bytes, tmp_path: str, dpi: int) -> Optional[Image.Image]:
    sniff = _sniff_suffix(data)
    if sniff == ".pdf":
        try:
            from pdf2image import convert_from_path
            pages = convert_from_path(tmp_path, dpi=dpi, first_page=1, last_page=1)
            return pages[0] if pages else None
        except Exception:
            return None
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        return None


def _file_hash(data: bytes, dpi: int) -> str:
    return hashlib.sha256(data + str(dpi).encode()).hexdigest()[:16]


# ── Session-state cached extraction ───────────────────────────────────────────

def _run_extraction(data: bytes, dpi: int):
    """
    Run the full extraction + validation pipeline.
    Results are cached in st.session_state keyed by (file_hash, dpi) so that
    toggling display options or switching tabs never re-runs OCR.
    """
    key = _file_hash(data, dpi)
    cache = st.session_state.get("extraction_cache", {})

    if key in cache:
        return cache[key]

    tmp_path = None
    try:
        tmp_path = _materialise(data)
        extractor = LabelExtractor(tmp_path, dpi=dpi)
        result    = extractor.run()
    finally:
        if tmp_path and os.path.isfile(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    validation_report, type_shortlist = auto_validate(result)
    layout_report      = check_layout(result, validation_report.label_type_id,
                                      result.metadata.image_width_px,
                                      result.metadata.image_height_px)
    overlap_report     = detect_overlaps(result)
    completeness_report = check_completeness(result, validation_report.label_type_id)

    payload = {
        "result":               result,
        "validation_report":    validation_report,
        "type_shortlist":       type_shortlist,
        "layout_report":        layout_report,
        "overlap_report":       overlap_report,
        "completeness_report":  completeness_report,
    }

    # Keep only the most recent result in cache to avoid memory build-up
    st.session_state["extraction_cache"] = {key: payload}
    return payload


# ── Annotation ────────────────────────────────────────────────────────────────

def _annotate(
    img: Image.Image,
    result: ExtractedLabel,
    show_words: bool = False,
    show_regions: bool = False,
    show_pdf_text: bool = False,
) -> Image.Image:
    canvas  = img.convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)

    try:
        font    = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)
        font_sm = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
    except Exception:
        font = font_sm = ImageFont.load_default()

    all_wb   = result.all_word_boxes
    raw_els  = result.raw_elements

    def _pill(draw_ctx, x, y, text, r, g, b):
        tw = draw_ctx.textlength(text, font=font_sm)
        tx, ty = x, max(0, y - 16)
        draw_ctx.rectangle([tx - 1, ty - 1, tx + tw + 4, ty + 14], fill=(r, g, b, 210))
        draw_ctx.text((tx + 2, ty), text, fill=(255, 255, 255, 255), font=font_sm)

    def _value_tokens(value: str) -> set[str]:
        raw_tokens = re.split(r'[\s,;:\(\)\[\]©®™\\]+', str(value))
        out = set()
        for tok in raw_tokens:
            tok = tok.strip('.,/-')
            if len(tok) >= 3:
                out.add(tok.lower())
        return out

    def _tok_matches(vt: str, wt: str) -> bool:
        vt, wt = vt.lower(), wt.lower()
        if len(vt) < 5:
            return vt == wt
        return vt == wt or vt in wt or wt in vt

    def _find_word_boxes(field_value: str, field_zone: Optional[str]) -> list[dict]:
        tokens = _value_tokens(field_value)
        if not tokens or not all_wb:
            return []

        def _scan(strict_zone: bool) -> list[dict]:
            found, seen = [], set()
            for wb in all_wb:
                if strict_zone and field_zone and wb.get("zone") and wb["zone"] != field_zone:
                    continue
                wb_clean = re.sub(r'[^\w.\-]', '', wb["text"])
                if not wb_clean:
                    continue
                if any(_tok_matches(t, wb_clean) for t in tokens):
                    pos = (wb["x"] // 12, wb["y"] // 12)
                    if pos not in seen:
                        seen.add(pos)
                        found.append(wb)
            return found

        res = _scan(strict_zone=True)
        return res if res else _scan(strict_zone=False)

    def _draw_field(fld, lbl: str, section: str):
        if not fld or not fld.value:
            return
        r, g, b = _SECTION_COLORS.get(section, (107, 114, 128))
        if getattr(fld, "source", "ocr") == "barcode" and fld.bbox:
            x, y, w, h = fld.bbox.x, fld.bbox.y, fld.bbox.w, fld.bbox.h
            draw.rectangle([x, y, x + w, y + h], fill=(r, g, b, 35), outline=(r, g, b, 210), width=2)
            _pill(draw, x, y, lbl, r, g, b)
            return
        boxes = _find_word_boxes(str(fld.value), fld.zone)
        if not boxes and fld.bbox:
            boxes = [{"x": fld.bbox.x, "y": fld.bbox.y, "w": fld.bbox.w, "h": fld.bbox.h}]
        if not boxes:
            return
        lbl_placed = False
        for wb in boxes:
            x, y, w, h = wb["x"], wb["y"], wb["w"], wb["h"]
            if w <= 0 or h <= 0:
                continue
            draw.rectangle([x, y, x + w, y + h], fill=(r, g, b, 45), outline=(r, g, b, 220), width=2)
            if not lbl_placed:
                _pill(draw, x, y, lbl, r, g, b)
                lbl_placed = True

    def _draw_barcode_bbox(bbox, lbl: str, section: str):
        if not bbox:
            return
        r, g, b = _SECTION_COLORS.get(section, (107, 114, 128))
        x, y, w, h = bbox.x, bbox.y, bbox.w, bbox.h
        draw.rectangle([x, y, x + w, y + h], fill=(r, g, b, 35), outline=(r, g, b, 210), width=2)
        _pill(draw, x, y, lbl, r, g, b)

    if show_regions:
        for el in raw_els:
            if el.type == "region":
                x, y, w, h = el.x, el.y, el.w, el.h
                if w > 0 and h > 0:
                    draw.rectangle([x, y, x + w, y + h], fill=(255, 165, 0, 18), outline=(255, 165, 0, 120), width=1)
            elif el.type == "line":
                meta = el.meta if hasattr(el, "meta") else {}
                x1, y1 = meta.get("x1", el.x), meta.get("y1", el.y)
                x2, y2 = meta.get("x2", el.x + el.w), meta.get("y2", el.y + el.h)
                draw.line([x1, y1, x2, y2], fill=(255, 200, 0, 140), width=1)

    if show_pdf_text:
        for el in raw_els:
            if el.type == "image_pdf":
                x, y, w, h = el.x, el.y, el.w, el.h
                if w > 0 and h > 0:
                    draw.rectangle([x, y, x + w, y + h], fill=(0, 200, 100, 25), outline=(0, 200, 100, 160), width=1)
                    _pill(draw, x, y, "img", 0, 160, 80)
            elif el.type == "text_pdf":
                x, y, w, h = el.x, el.y, el.w, el.h
                if w > 0 and h > 0:
                    draw.rectangle([x, y, x + w, y + h], fill=(100, 200, 255, 20), outline=(100, 200, 255, 130), width=1)

    if show_words:
        for wb in all_wb:
            x, y, w, h = wb["x"], wb["y"], wb["w"], wb["h"]
            alpha = int(40 + wb.get("conf", 0.5) * 80)
            draw.rectangle([x, y, x + w, y + h],
                           fill=(180, 180, 180, max(8, alpha // 4)),
                           outline=(180, 180, 180, alpha), width=1)

    _draw_field(result.product_name,                 "Product name",      "identity")
    _draw_field(result.product_dimensions_metric,    "Dim (metric)",      "identity")
    _draw_field(result.product_dimensions_imperial,  "Dim (imperial)",    "identity")
    _draw_field(result.article_number,               "Article No.",       "identity")
    _draw_field(result.article_code,                 "Article code",      "identity")
    _draw_field(result.internal_item_number,         "Item No.",          "identity")
    _draw_field(result.quantity_multiplier,          "Qty ×",             "identity")
    _draw_field(result.gross_weight,                 "Gross wt",          "identity")
    _draw_field(result.net_weight,                   "Net wt",            "identity")
    _draw_field(result.package_number,               "Pkg No.",           "identity")
    _draw_field(result.package_type,                 "Pkg type",          "identity")
    _draw_field(result.dimensions,                   "Dimensions",        "identity")
    _draw_field(result.weight,                       "Weight",            "identity")
    _draw_field(result.copy_count_inline,            "Copy (inline)",     "copy")
    _draw_field(result.ikea_logo,                    "IKEA logo",         "identity")
    _draw_field(result.ean_barcode_value,            "EAN-13",            "barcodes")
    _draw_field(result.itf14_human_readable,         "Human readable",    "barcodes")
    _draw_field(result.ai_240,                       "AI(240)",           "barcodes")
    _draw_field(result.ai_13,                        "AI(13) date",       "barcodes")
    _draw_field(result.ai_11,                        "AI(11) prod",       "barcodes")
    _draw_field(result.ai_10,                        "AI(10) batch",      "barcodes")
    _draw_field(result.human_readable_date,          "Prod date",         "date")
    _draw_field(result.date_stamp,                   "Date stamp",        "date")
    _draw_field(result.date_alpha_code,              "Alpha code",        "date")
    _draw_field(result.date_numeric_prefix,          "Prefix",            "date")
    _draw_field(result.custom_identifier,            "Custom ID",         "date")
    _draw_field(result.plant_identifier,             "Plant ID",          "address")
    _draw_field(result.supplier_name,                "Supplier name",     "address")
    _draw_field(result.supplier_address,             "Supplier addr",     "address")
    _draw_field(result.address_block,                "Address",           "address")
    _draw_field(result.origin_text,                  "Origin",            "address")
    _draw_field(result.copyright_notice,             "Copyright",         "address")
    _draw_field(result.compliance_marks,             "Compliance",        "compliance")
    _draw_field(result.age_rating,                   "Age rating",        "compliance")
    _draw_field(result.copy_count,                   "Copy count",        "copy")
    _draw_barcode_bbox(result.ean_barcode.bbox if result.ean_barcode else None,       "EAN-13", "barcodes")
    _draw_barcode_bbox(result.itf14_barcode.bbox if result.itf14_barcode else None,   "ITF-14", "barcodes")
    _draw_barcode_bbox(result.datamatrix_barcode.bbox if result.datamatrix_barcode else None, "DataMatrix", "barcodes")

    return Image.alpha_composite(canvas, overlay).convert("RGB")


# ── UI helpers ────────────────────────────────────────────────────────────────

def _pill_html(text: str, color: str) -> str:
    return (f'<span style="background:{color};color:#fff;font-size:10px;'
            f'padding:2px 7px;border-radius:10px;font-weight:600;'
            f'white-space:nowrap">{text}</span>')


def _conf_color(c: float) -> str:
    if c >= 0.85: return "#22c55e"
    if c >= 0.60: return "#f59e0b"
    return "#ef4444"


def _score_color(c: float) -> str:
    if c >= 0.90: return "#22c55e"
    if c >= 0.65: return "#f59e0b"
    return "#ef4444"


# ── Shared CSS ────────────────────────────────────────────────────────────────

_CSS = """
<style>
.block-container { padding-top: 1.5rem; }
[data-testid="stSidebar"] { background: #0f172a; }
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #f1f5f9 !important; }
[data-testid="stSidebar"] hr { border-color: #1e293b; }
[data-testid="stAppViewContainer"] { background: #0f172a; }
[data-testid="stHeader"] { background: #0f172a; }
html, body, [class*="css"] { color: #e2e8f0; }
[data-testid="stMetric"] {
    background: #1e293b; border-radius: 10px;
    padding: 12px 16px; border: 1px solid #334155;
}
[data-testid="stMetricLabel"] { color: #94a3b8 !important; }
[data-testid="stMetricValue"] { color: #f1f5f9 !important; }
[data-testid="stExpander"] {
    background: #1e293b; border: 1px solid #334155; border-radius: 8px;
}
[data-testid="stFileUploader"] {
    background: #1e293b; border: 2px dashed #334155; border-radius: 10px;
}
hr { border-color: #1e293b !important; }
[data-testid="stDownloadButton"] button {
    background: #3b82f6 !important; color: white !important;
    border: none !important; border-radius: 8px !important;
}
[data-testid="stTabs"] [data-baseweb="tab-list"] { background: #1e293b; border-radius: 8px; }
[data-testid="stTabs"] [data-baseweb="tab"] { color: #94a3b8 !important; }
[data-testid="stTabs"] [aria-selected="true"] { color: #f1f5f9 !important; }
</style>
"""


# ── Tab renderers ─────────────────────────────────────────────────────────────

def _render_label_view(data, result, dpi, show_words, show_regions, show_pdf_text, show_clean):
    tmp_path = None
    try:
        tmp_path = _materialise(data)
        preview_img = _get_preview_image(data, tmp_path, dpi)
    finally:
        if tmp_path and os.path.isfile(tmp_path):
            try: os.unlink(tmp_path)
            except OSError: pass

    if not preview_img:
        st.info("Image preview not available for this file type.")
        return

    annotated = _annotate(preview_img, result, show_words=show_words,
                          show_regions=show_regions, show_pdf_text=show_pdf_text)

    left_col, right_col = st.columns([5, 4], gap="large")

    with left_col:
        st.markdown(
            '<div style="background:#f8fafc;border-radius:12px;padding:20px 20px 12px 20px;'
            'border:1px solid #e2e8f0">'
            '<p style="color:#1e293b;font-size:16px;font-weight:700;margin:0 0 12px 0">Label</p>',
            unsafe_allow_html=True,
        )
        st.image(annotated, use_container_width=True)
        if show_clean:
            st.markdown('<p style="color:#64748b;font-size:12px;margin:10px 0 4px 0">Original</p>',
                        unsafe_allow_html=True)
            st.image(preview_img, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right_col:
        _render_extracted_elements(result)


def _render_extracted_elements(result: ExtractedLabel):
    st.markdown(
        '<div style="background:#0f172a;border-radius:12px;padding:20px;border:1px solid #1e293b">'
        '<p style="color:#f1f5f9;font-size:16px;font-weight:700;margin:0 0 12px 0">Extracted elements</p>',
        unsafe_allow_html=True,
    )

    def _section_header(title, section_key, found, total):
        r, g, b = _SECTION_COLORS.get(section_key, (107, 114, 128))
        hex_col = f"#{r:02x}{g:02x}{b:02x}"
        sc = "#22c55e" if found == total else ("#f59e0b" if found > 0 else "#ef4444")
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;margin:16px 0 6px 0">'
            f'<div style="width:4px;height:18px;background:{hex_col};border-radius:2px"></div>'
            f'<span style="font-weight:700;font-size:14px;color:#f1f5f9">{title}</span>'
            f'<span style="color:{sc};font-size:12px;margin-left:auto">{found}/{total} found</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    def _el(element_id, label, fld):
        found = fld is not None and getattr(fld, "value", None) is not None
        if not found:
            st.markdown(
                f'<div style="display:flex;align-items:center;padding:7px 0;'
                f'border-bottom:1px solid #1e293b;gap:8px">'
                f'<span style="color:#334155;font-size:16px">○</span>'
                f'<span style="color:#475569;font-size:13px;flex:1">{label}</span>'
                f'<span style="color:#334155;font-size:11px">not found</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            return
        src_colors = {"barcode": "#0ea5e9", "ocr": "#7c3aed", "derived": "#64748b"}
        src_color  = src_colors.get(fld.source, "#94a3b8")
        conf       = fld.confidence
        conf_color = _conf_color(conf)
        pos_str = ""
        if fld.bbox:
            b = fld.bbox
            pos_str = (f'<span style="color:#94a3b8;font-size:10px;font-family:monospace">'
                       f'[{b.x},{b.y} {b.w}×{b.h}]</span>')
        st.markdown(
            f'<div style="padding:7px 0;border-bottom:1px solid #1e293b">'
            f'<div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap">'
            f'<span style="color:{conf_color};font-size:15px">●</span>'
            f'<span style="font-size:13px;font-weight:600;color:#f1f5f9;flex:1">{label}</span>'
            f'{_pill_html(fld.source, src_color)}'
            f'<span style="color:{conf_color};font-size:11px;font-weight:600">{conf:.0%}</span>'
            f'{pos_str}'
            f'</div>'
            f'<div style="margin-top:3px;margin-left:21px;font-family:monospace;'
            f'font-size:13px;color:#93c5fd;word-break:break-all">{fld.value}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    def _bc_el(label, bc):
        if bc:
            bb = bc.bbox
            pos = f"[{bb.x},{bb.y} {bb.w}×{bb.h}]" if bb else ""
            st.markdown(
                f'<div style="padding:7px 0;border-bottom:1px solid #1e293b">'
                f'<div style="display:flex;align-items:center;gap:6px">'
                f'<span style="color:#22c55e;font-size:15px">●</span>'
                f'<span style="font-size:13px;font-weight:600;color:#f1f5f9;flex:1">{label}</span>'
                f'{_pill_html("barcode", "#0ea5e9")}'
                f'<span style="color:#94a3b8;font-size:10px;font-family:monospace">{pos}</span>'
                f'</div>'
                f'<div style="margin-top:3px;margin-left:21px;font-family:monospace;'
                f'font-size:13px;color:#93c5fd;word-break:break-all">{bc.raw_data}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            _el(label.lower().replace(" ", "_"), label, None)

    id_fields = [result.product_name, result.product_dimensions_metric,
                 result.product_dimensions_imperial, result.article_number,
                 result.article_code, result.internal_item_number,
                 result.quantity_multiplier, result.gross_weight, result.net_weight,
                 result.package_number, result.package_type, result.ikea_logo]
    _section_header("Product identity", "identity", sum(1 for f in id_fields if f), len(id_fields))
    _el("product_name",               "Product name",         result.product_name)
    _el("product_dimensions_metric",  "Dimensions (metric)",  result.product_dimensions_metric)
    _el("product_dimensions_imperial","Dimensions (imperial)", result.product_dimensions_imperial)
    _el("article_number",             "Article number",       result.article_number)
    _el("article_code",               "Article code",         result.article_code)
    _el("internal_item_number",       "Internal item no.",    result.internal_item_number)
    _el("quantity_multiplier",        "Quantity multiplier",  result.quantity_multiplier)
    _el("gross_weight",               "Gross weight",         result.gross_weight)
    _el("net_weight",                 "Net weight",           result.net_weight)
    _el("package_number",             "Package number",       result.package_number)
    _el("package_type",               "Package type",         result.package_type)
    _el("ikea_logo",                  "IKEA logo",            result.ikea_logo)

    bc_fields = [result.ean_barcode, result.itf14_barcode, result.datamatrix_barcode,
                 result.ai_240, result.ai_13]
    _section_header("Barcodes & data codes", "barcodes", sum(1 for f in bc_fields if f), len(bc_fields))
    _bc_el("EAN-13 barcode",  result.ean_barcode)
    _el("ean_barcode_value",  "EAN-13 value",         result.ean_barcode_value)
    _bc_el("ITF-14 barcode",  result.itf14_barcode)
    _el("itf14_human_readable","ITF-14 human readable", result.itf14_human_readable)
    _bc_el("DataMatrix",      result.datamatrix_barcode)
    _el("ai_240", "AI(240) — Item ID",    result.ai_240)
    _el("ai_13",  "AI(13) — Pack date",   result.ai_13)
    _el("ai_11",  "AI(11) — Prod date",   result.ai_11)
    _el("ai_10",  "AI(10) — Batch / lot", result.ai_10)

    date_fields = [result.human_readable_date, result.date_stamp,
                   result.date_alpha_code, result.date_numeric_prefix, result.custom_identifier]
    _section_header("Date & production", "date", sum(1 for f in date_fields if f), len(date_fields))
    _el("human_readable_date", "Production date (readable)", result.human_readable_date)
    _el("date_stamp",          "Date stamp (YYWW)",          result.date_stamp)
    _el("date_alpha_code",     "Alpha code",                 result.date_alpha_code)
    _el("date_numeric_prefix", "Numeric prefix",             result.date_numeric_prefix)
    _el("custom_identifier",   "Custom identifier",          result.custom_identifier)

    addr_fields = [result.plant_identifier, result.supplier_name, result.supplier_address,
                   result.address_block, result.origin_text, result.copyright_notice]
    _section_header("Supplier & address", "address", sum(1 for f in addr_fields if f), len(addr_fields))
    _el("plant_identifier", "Plant identifier",  result.plant_identifier)
    _el("supplier_name",    "Supplier name",     result.supplier_name)
    _el("supplier_address", "Supplier address",  result.supplier_address)
    _el("address_block",    "Address block",     result.address_block)
    _el("origin_text",      "Country of origin", result.origin_text)
    _el("copyright_notice", "Copyright",         result.copyright_notice)

    comp_fields = [result.compliance_marks, result.age_rating]
    _section_header("Compliance marks", "compliance", sum(1 for f in comp_fields if f), len(comp_fields))
    _el("compliance_marks", "Compliance marks", result.compliance_marks)
    _el("age_rating",       "Age rating",       result.age_rating)

    cc_fields = [result.copy_count, result.copy_count_inline]
    _section_header("Copy count", "copy", sum(1 for f in cc_fields if f), len(cc_fields))
    _el("copy_count",        "Copy count (strip)",  result.copy_count)
    _el("copy_count_inline", "Copy count (inline)", result.copy_count_inline)

    st.markdown('</div>', unsafe_allow_html=True)


def _render_rule_result(r) -> None:
    status_icon = {"pass": "✅", "fail": "❌", "skip": "⏭️", "not_found": "⚠️"}.get(r.status, "?")
    sev_color   = {"error": "#ef4444", "warning": "#f59e0b", "info": "#6366f1"}.get(r.severity, "#94a3b8")
    row_bg      = {"pass": "#052e16", "fail": "#2d0a0a", "skip": "#1c1c2e", "not_found": "#2a1a00"}.get(r.status, "#1e293b")
    val_html = ""
    if r.extracted_value:
        val_html = (f'<span style="font-family:monospace;font-size:11px;color:#93c5fd;'
                    f'word-break:break-all">{r.extracted_value}</span>')
    exp_html = ""
    if r.expected and r.status in ("fail", "not_found"):
        exp_html = (f'<span style="font-size:10px;color:#94a3b8"> → expected: '
                    f'<code style="font-size:10px">{r.expected[:60]}</code></span>')
    st.markdown(
        f'<div style="background:{row_bg};border-radius:6px;padding:6px 10px;'
        f'margin-bottom:4px;border-left:3px solid {sev_color}">'
        f'<div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap">'
        f'{status_icon} '
        f'<span style="font-size:11px;font-weight:600;color:#94a3b8;font-family:monospace">'
        f'{r.rule_id}</span>'
        f'<span style="font-size:12px;color:#e2e8f0;flex:1">{r.message}</span>'
        f'<span style="font-size:10px;color:{sev_color};font-weight:600">{r.severity}</span>'
        f'</div>'
        f'{"<div style=margin-top:2px;margin-left:20px>" + val_html + exp_html + "</div>" if val_html or exp_html else ""}'
        f'</div>',
        unsafe_allow_html=True,
    )


def _render_field_validation(report: ValidationReport, shortlist: list[TypeMatch]) -> None:
    score_pct   = f"{report.match_confidence:.0%}"
    score_color = "#22c55e" if report.match_confidence >= 0.6 else (
                  "#f59e0b" if report.match_confidence >= 0.35 else "#ef4444")
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;'
        f'margin-bottom:12px;padding:8px 12px;background:#1e293b;border-radius:8px">'
        f'<span style="font-size:13px;color:#94a3b8">Detected type:</span>'
        f'<span style="font-size:13px;font-weight:700;color:#f1f5f9">{report.label_type_name}</span>'
        f'<span style="font-family:monospace;font-size:11px;color:#64748b">({report.label_type_id})</span>'
        f'<span style="margin-left:auto;font-size:12px;font-weight:600;color:{score_color}">'
        f'{score_pct} match</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    cs = report.compliance_score
    cs_color = _score_color(cs)
    bar_w = max(4, int(cs * 100))
    n_e = len(report.errors); n_w = len(report.warnings)
    n_p = len(report.passed); n_s = sum(1 for r in report.results if r.status == "skip")

    st.markdown(
        f'<div style="margin-bottom:14px">'
        f'<div style="display:flex;justify-content:space-between;margin-bottom:4px">'
        f'<span style="font-size:12px;color:#94a3b8">Compliance score</span>'
        f'<span style="font-size:13px;font-weight:700;color:{cs_color}">{cs:.0%}</span>'
        f'</div>'
        f'<div style="background:#334155;border-radius:4px;height:8px">'
        f'<div style="background:{cs_color};width:{bar_w}%;height:8px;border-radius:4px"></div>'
        f'</div>'
        f'<div style="display:flex;gap:16px;margin-top:6px">'
        f'<span style="font-size:11px;color:#22c55e">✅ {n_p} passed</span>'
        f'<span style="font-size:11px;color:#ef4444">❌ {n_e} errors</span>'
        f'<span style="font-size:11px;color:#f59e0b">⚠️ {n_w} warnings</span>'
        f'<span style="font-size:11px;color:#64748b">⏭️ {n_s} skipped</span>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    if report.errors:
        st.markdown('<p style="font-size:12px;color:#ef4444;font-weight:600;margin:6px 0 4px 0">Errors</p>',
                    unsafe_allow_html=True)
        for r in report.errors:
            _render_rule_result(r)
    if report.warnings:
        st.markdown('<p style="font-size:12px;color:#f59e0b;font-weight:600;margin:6px 0 4px 0">Warnings</p>',
                    unsafe_allow_html=True)
        for r in report.warnings:
            _render_rule_result(r)
    if report.passed:
        with st.expander(f"✅ {n_p} passing rules", expanded=False):
            for r in report.passed:
                _render_rule_result(r)
    skipped = [r for r in report.results if r.status in ("skip", "not_found")]
    if skipped:
        with st.expander(f"⏭️ {len(skipped)} skipped / not found", expanded=False):
            for r in skipped:
                _render_rule_result(r)
    if shortlist and len(shortlist) > 1:
        with st.expander("🔍 Other label type candidates"):
            for tm in shortlist[1:]:
                bar = max(2, int(tm.score * 100))
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">'
                    f'<span style="font-size:11px;font-family:monospace;color:#94a3b8;width:180px">'
                    f'{tm.label_type_id}</span>'
                    f'<div style="flex:1;background:#334155;border-radius:3px;height:6px">'
                    f'<div style="background:#3b82f6;width:{bar}%;height:6px;border-radius:3px"></div>'
                    f'</div>'
                    f'<span style="font-size:11px;color:#94a3b8;width:36px;text-align:right">'
                    f'{tm.score:.0%}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )


def _render_coming_soon(title: str, description: str, checklist: list[str]) -> None:
    st.markdown(
        f'<div style="background:#1e293b;border-radius:12px;padding:32px;'
        f'border:1px solid #334155;text-align:center;margin-top:16px">'
        f'<div style="font-size:36px;margin-bottom:12px">🚧</div>'
        f'<p style="color:#f1f5f9;font-size:18px;font-weight:700;margin-bottom:8px">{title}</p>'
        f'<p style="color:#64748b;font-size:14px;max-width:520px;margin:0 auto 20px auto">{description}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown("**Checks that will be implemented:**")
    for item in checklist:
        st.markdown(f"- {item}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="IKEA Label Validation", page_icon="🔬", layout="wide")
    st.markdown(_CSS, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Options")
        dpi = st.slider("Render DPI (PDF)", UI_DPI_MIN, UI_DPI_MAX, UI_DPI_DEFAULT, UI_DPI_STEP)
        show_words    = st.checkbox("Show OCR word boxes",         value=True)
        show_regions  = st.checkbox("Show visual regions (Pass 4)", value=False)
        show_pdf_text = st.checkbox("Show PDF text spans (Pass 1)", value=False)
        show_clean    = st.checkbox("Show clean image too",         value=False)
        show_json     = st.checkbox("Show full JSON payload",       value=False)
        st.divider()
        st.markdown("**Legend**")
        for key, (r, g, b) in _SECTION_COLORS.items():
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:4px">'
                f'<div style="width:12px;height:12px;background:#{r:02x}{g:02x}{b:02x};'
                f'border-radius:2px"></div>'
                f'<span style="font-size:12px;color:#475569">{key}</span></div>',
                unsafe_allow_html=True,
            )

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown('<h2 style="color:#f1f5f9">🔬 IKEA Label Validation</h2>', unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#94a3b8;margin-top:-8px">'
        'Upload a production label — every element is extracted, validated against '
        'the label-rules schemas, and checked for spatial compliance.</p>',
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Upload label file", type=ALLOWED_UPLOAD_TYPES,
        label_visibility="collapsed",
        help="PDF, PNG, JPEG, TIFF or BMP",
    )

    if uploaded is None:
        st.markdown(
            '<div style="text-align:center;padding:60px 0;color:#475569">'
            '<div style="font-size:48px">📄</div>'
            '<div style="font-size:16px;margin-top:8px;color:#64748b">Drop a label file above to begin</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    data = uploaded.getvalue()

    # ── Run extraction (cached per file + DPI) ────────────────────────────────
    with st.spinner("Extracting and validating label…"):
        try:
            payload = _run_extraction(data, dpi)
        except Exception as exc:
            st.error(f"**Extraction failed:** {exc}")
            logging.exception("Extraction failed")
            return

    result              = payload["result"]
    validation_report   = payload["validation_report"]
    type_shortlist      = payload["type_shortlist"]
    layout_report       = payload["layout_report"]
    overlap_report      = payload["overlap_report"]
    completeness_report = payload["completeness_report"]
    md                  = result.metadata

    # ── Top scorecard ─────────────────────────────────────────────────────────
    all_fields = [
        result.product_name, result.product_dimensions_metric, result.product_dimensions_imperial,
        result.article_number, result.article_code, result.internal_item_number,
        result.quantity_multiplier, result.gross_weight, result.net_weight,
        result.package_number, result.package_type, result.ikea_logo,
        result.ean_barcode, result.ean_barcode_value,
        result.itf14_barcode, result.itf14_human_readable,
        result.datamatrix_barcode, result.ai_240, result.ai_13, result.ai_11, result.ai_10,
        result.human_readable_date, result.date_stamp, result.date_alpha_code,
        result.date_numeric_prefix, result.custom_identifier,
        result.plant_identifier, result.supplier_name, result.supplier_address,
        result.address_block, result.origin_text, result.copyright_notice,
        result.compliance_marks, result.age_rating, result.copy_count, result.copy_count_inline,
    ]
    n_found = sum(1 for f in all_fields if f is not None)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Fields extracted",    f"{n_found}/{len(all_fields)}")
    c2.metric("Confidence",          f"{md.overall_confidence:.0%}")
    c3.metric("Compliance score",    f"{validation_report.compliance_score:.0%}")
    c4.metric("Field rule errors",   len(validation_report.errors))
    c5.metric("Layout issues",       len(layout_report.errors) + len(layout_report.warnings),
              help="Spatial checks — implemented in next phase")
    c6.metric("Overlaps detected",   len(overlap_report.errors) + len(overlap_report.warnings),
              help="Bounding-box overlap checks — implemented in next phase")

    if md.warnings:
        with st.expander(f"⚠️ {len(md.warnings)} extraction warning(s)"):
            for w in md.warnings:
                st.warning(w)

    st.divider()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_view, tab_rules, tab_layout, tab_overlaps, tab_complete, tab_json = st.tabs([
        "🖼️  Label View",
        "✅ Field Validation",
        "📐 Layout & Zones",
        "⚡ Overlaps",
        "🔍 Completeness",
        "{}  JSON",
    ])

    with tab_view:
        _render_label_view(data, result, dpi, show_words, show_regions, show_pdf_text, show_clean)

    with tab_rules:
        _render_field_validation(validation_report, type_shortlist)

    with tab_layout:
        _render_coming_soon(
            "Layout & Zone Validator",
            "Checks that each extracted field lives in the zone the template defines for it, "
            "and that no element overflows the label boundary.",
            [
                "Zone compliance — is `product_name` in the identity zone, `itf14` in the barcode strip, etc.?",
                "Boundary check — is any bounding box outside the image dimensions?",
                "Barcode clear zone — does any text element encroach on a barcode quiet zone?",
                "Relative ordering — is the IKEA logo above the article number?",
            ],
        )

    with tab_overlaps:
        _render_coming_soon(
            "Bounding-Box Overlap Detector",
            "Runs deterministic pairwise intersection checks across all extracted element "
            "bounding boxes and flags unexpected overlaps.",
            [
                "Field ↔ field overlap — do two named fields share pixel area?",
                "Field ↔ barcode overlap — does text bleed into a barcode's quiet zone?",
                "Severity by intersection ratio — warning < 25%, error ≥ 25%.",
                "Visual highlight — overlap regions drawn in red on the label image.",
            ],
        )

    with tab_complete:
        _render_coming_soon(
            "Completeness & Consistency Checker",
            "Confirms all required fields are present and that fields which encode the "
            "same information agree with each other.",
            [
                "Missing required fields — from the detected label type's `required_fields` list.",
                "Missing expected barcodes — schema says `has_itf: true` but no ITF-14 decoded.",
                "ITF-14 digits ↔ human-readable text — do they match?",
                "DataMatrix AI(13) date ↔ human-readable date on label face.",
                "DataMatrix AI(240) item code ↔ article number.",
            ],
        )

    with tab_json:
        payload_dict = result.to_dict()
        payload_dict["validation"] = validation_report.to_dict()
        json_bytes = json.dumps(
            payload_dict, indent=2, ensure_ascii=False, cls=_NumpyEncoder
        ).encode("utf-8")
        stem = Path(uploaded.name).stem
        dl_col, _ = st.columns([1, 3])
        with dl_col:
            st.download_button(
                "⬇️ Download extraction JSON",
                data=json_bytes,
                file_name=f"{stem}_extracted.json",
                mime="application/json",
                use_container_width=True,
            )
        if show_json:
            st.json(json.dumps(payload_dict, ensure_ascii=False, cls=_NumpyEncoder))


if __name__ == "__main__":
    main()
