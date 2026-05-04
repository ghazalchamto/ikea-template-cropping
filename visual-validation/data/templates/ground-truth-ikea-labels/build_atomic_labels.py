#!/usr/bin/env python3
"""
build_atomic_labels.py — Atomic IKEA label ground truth builder.

For every label variant detected in label 2.pdf, produces:
  ground_truth/templates/{family}/v{N}/
    label.png       — 300 DPI crop of the individual label
    template.json   — rich JSON (elements, rules, validation_targets)

Detection: uses PDF vector stroke colors to locate label bounding rects.
Three label-border stroke colors in this PDF:
  GRAY_A  (0.577, 0.586, 0.596) — narrow row labels (1R10, 1R12)
  GRAY_B  (0.779, 0.785, 0.793) — wider row + column + circular labels
  DARK    (0.137, 0.122, 0.125) — some tiny + column labels
"""

import fitz  # pymupdf
import json
import re
import shutil
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
PDF_PATH = Path("label 2.pdf")
OUT_ROOT = Path("ground_truth")
OLD_RICH = Path("/Users/sham_sara/Desktop/ikea-template-cropping/ground_truth/templates")

CROP_DPI  = 300          # resolution for individual label crops
FULL_DPI  = 150          # resolution kept for full-page reference images
CROP_PAD  = 3            # pt padding around each detected label rect
SMALL_W   = 15           # minimum rect width (pt) to be a label
SMALL_H   = 5            # minimum rect height (pt) to be a label
MIN_LARGE_DIM = 40       # at least one dimension must be >= this to avoid compliance symbol cells

# Label border stroke colors — approximate equality threshold
STROKE_COLORS = [
    (0.5772945880889893, 0.5855344533920288, 0.5958037972450256),  # GRAY_A
    (0.7785152792930603, 0.7854581475257874, 0.7930571436882019),  # GRAY_B
    (0.13669031858444214, 0.12195010483264923, 0.1252918243408203), # DARK
]
COLOR_TOL = 0.025

# ── Template map (page→family) ─────────────────────────────────────────────
# pages: 1-based page numbers that are part of this family's spec
# existing_variants: list of JSON basenames from old repo (None = auto-generate)
TEMPLATE_MAP = [
    {"key": "general_rules",        "pages": [1],       "family": None,          "existing": None},
    {"key": "element_variants",     "pages": [2],       "family": None,          "existing": None},
    {"key": "1R10-PRADM",          "pages": [3, 6],    "family": "1R10-PRADM",  "existing": [
        "1R10-PRADM_v1.json","1R10-PRADM_v2.json","1R10-PRADM_v3.json",
        "1R10-PRADM_v4.json","1R10-PRADM_v5.json"]},
    {"key": "1R20-PRADM",          "pages": [4, 5],    "family": "1R20-PRADM",  "existing": [
        "1R20-PRADM_04x_v1.json","1R20-PRADM_04x_v2.json","1R20-PRADM_04x_v3.json",
        "1R20-PRADM_04x_v4.json","1R20-PRADM_04x_v5.json",
        "1R20-PRADM_075x_v1.json","1R20-PRADM_075x_v2.json","1R20-PRADM_075x_v3.json",
        "1R20-PRADM_075x_v4.json","1R20-PRADM_075x_v5.json"]},
    {"key": "1R12-IDADM-IDBDM",    "pages": [7,8,9],   "family": "1R12",        "existing": None},
    {"key": "1R35-IDADM-IDBDM",    "pages": [10,11,12],"family": "1R35",        "existing": None},
    {"key": "1R25-IDCDM-IDCSDM",   "pages": [13],      "family": "1R25",        "existing": None},
    {"key": "1C70-IDBDM",          "pages": [14],      "family": "1C70",        "existing": None},
    {"key": "1C50-IDBDM",          "pages": [15, 16],  "family": "1C50",        "existing": None},
    {"key": "1R35-IDGDM",          "pages": [17],      "family": "1R35",        "existing": None},
    {"key": "1R15-PRADM-USA",      "pages": [18, 19],  "family": "1R15",        "existing": None},
    {"key": "1CIR50-1CIR60",       "pages": [20],      "family": "1CIR",        "existing": None},
    {"key": "L3815-L8015-PRADM",   "pages": [21],      "family": "L-PRADM",     "existing": None},
    {"key": "L567-L578-1R7-PRADM", "pages": [22],      "family": "L-PRADM",     "existing": None},
    {"key": "1CIR30",              "pages": [23],      "family": "1CIR",        "existing": None},
    {"key": "ITF-barcode-specs",   "pages": [24],      "family": None,          "existing": None},
    {"key": "1C50-IDEDM",          "pages": [25, 26],  "family": "1C50",        "existing": None},
    {"key": "1C70-IDDDM",          "pages": [27, 28],  "family": "1C70",        "existing": None},
    {"key": "1C50-CAA",            "pages": [29],      "family": "1C50",        "existing": None},
    {"key": "1C70-CAA",            "pages": [30],      "family": "1C70",        "existing": None},
    {"key": "L10555-MIADM",        "pages": [31],      "family": "L-MIADM",     "existing": None},
    {"key": "L5727-PRADM",         "pages": [32],      "family": "L-PRADM",     "existing": None},
    {"key": "L4515-PRADM",         "pages": [33, 34],  "family": "L-PRADM",     "existing": None},
    {"key": "2R20-IDEDM",          "pages": [35],      "family": "2R20",        "existing": None},
    {"key": "2R35-IDEDM",          "pages": [36],      "family": "2R35",        "existing": None},
    {"key": "1R90-1R70-1R50",      "pages": [37],      "family": "1R-large",    "existing": None},
    {"key": "1R7-IDBDM",           "pages": [38],      "family": "1R7",         "existing": None},
    {"key": "1R25-IDEDM",          "pages": [39,40,41],"family": "1R25",        "existing": None},
    {"key": "1R35-ITF-addendum",   "pages": [42, 49],  "family": "1R35",        "existing": None},
    {"key": "1R50-ITF-addendum",   "pages": [43],      "family": "1R50",        "existing": None},
    {"key": "L25025-IDFDM",        "pages": [44],      "family": "L-IDFDM",     "existing": None},
    {"key": "1CIR24",              "pages": [45],      "family": "1CIR",        "existing": None},
    {"key": "1CIR40-IDBDM",        "pages": [46],      "family": "1CIR40",      "existing": None},
    {"key": "COA-IDADM-NUA",       "pages": [47],      "family": "COA",         "existing": None},
    {"key": "1CIR40-IDFDM",        "pages": [48],      "family": "1CIR40",      "existing": None},
]

# Build page→template lookup
PAGE_TO_TEMPLATE = {}
for t in TEMPLATE_MAP:
    for pg in t["pages"]:
        PAGE_TO_TEMPLATE[pg] = t


# ── Label rect detection ───────────────────────────────────────────────────────

def is_label_color(c):
    if c is None or len(c) < 3:
        return False
    return any(
        abs(c[0]-sc[0]) < COLOR_TOL and abs(c[1]-sc[1]) < COLOR_TOL and abs(c[2]-sc[2]) < COLOR_TOL
        for sc in STROKE_COLORS
    )

def find_label_rects(page):
    """Return sorted list of fitz.Rect for label borders (stroke-only large rects)."""
    rects = []
    for d in page.get_drawings():
        c = d.get("color")
        if not is_label_color(c):
            continue
        r = d["rect"]
        if (r.width >= SMALL_W and r.height >= SMALL_H
                and d.get("fill") is None
                and max(r.width, r.height) >= MIN_LARGE_DIM):
            rects.append(fitz.Rect(r))
    # Sort top-to-bottom, then left-to-right
    rects.sort(key=lambda r: (round(r.y0), round(r.x0)))
    # Remove near-duplicates (within 2pt)
    deduped = []
    for r in rects:
        if not deduped or abs(r.y0 - deduped[-1].y0) > 2 or abs(r.x0 - deduped[-1].x0) > 2:
            deduped.append(r)
    return deduped


# ── Text extraction ────────────────────────────────────────────────────────────

def extract_text_in_rect(page, rect, pad=2):
    """Extract text blocks within (padded) rect, return list of dicts."""
    clip = fitz.Rect(rect.x0 - pad, rect.y0 - pad, rect.x1 + pad, rect.y1 + pad)
    raw = page.get_text("dict", clip=clip)
    blocks = []
    for b in raw.get("blocks", []):
        if b.get("type") != 0:
            continue
        for line in b.get("lines", []):
            for span in line.get("spans", []):
                txt = span.get("text", "").strip()
                if not txt:
                    continue
                bbox = fitz.Rect(span["bbox"])
                blocks.append({
                    "text": txt,
                    "x0": bbox.x0, "y0": bbox.y0,
                    "x1": bbox.x1, "y1": bbox.y1,
                    "font_size_pt": span.get("size", 0),
                    "font_name": span.get("font", ""),
                    "flags": span.get("flags", 0),
                })
    return blocks


# ── Element classification ─────────────────────────────────────────────────────

ARTICLE_NUMBER_RE = re.compile(r"^\d{3}[.\-]\d{3}[.\-]\d{2}$")
ARTICLE_CODE_RE   = re.compile(r"^\d{5}$")
COPY_COUNT_RE     = re.compile(r"^\d{1,2}(\(\d+\))?$")
YYWW_RE           = re.compile(r"^(YYWW|\d{4})$")
DATE_LABEL_RE     = re.compile(r"^DATE:?$")
ABCD_RE           = re.compile(r"^ABCD$")
AI_RE             = re.compile(r"AI\(\d+\)")
MADE_IN_RE        = re.compile(r"^(Made in|Assembled in)", re.IGNORECASE)
IKEA_ADDR_RE      = re.compile(r"IKEA of Swee", re.IGNORECASE)
LEGAL_RE          = re.compile(r"^Legal( symbol)?$", re.IGNORECASE)
WASH_RE           = re.compile(r"^(Wash|Care|Dry|Iron|Bleach)( symbol)?$", re.IGNORECASE)
CARE_INSTR_RE     = re.compile(r"(Wash by|Do not|max \d+°|machine wash|tumble dry|dryclean|dry clean|bleach)", re.IGNORECASE)
EPA_RE            = re.compile(r"(EPA|TSCA|ULEF|CARB|Conforme)", re.IGNORECASE)
PATENTS_RE        = re.compile(r"(Patents|IKEA\.com/patents)", re.IGNORECASE)
CATALOG_NUM_RE    = re.compile(r"^[A-Z]{2}\d{4,6}$")
DIM_RE            = re.compile(r"\d+[xX×]\d+[xX×]\d+")
WEIGHT_RE         = re.compile(r"\d+\s*(kg|lbs|lb)\b", re.IGNORECASE)
DESIGN_QUAL_RE    = re.compile(r"Design and Quality|IKEA of Sweden", re.IGNORECASE)
PROD_NAME_RE      = re.compile(r"^(PRODUCTNAME|PRODUCT|[A-ZÅÄÖ][A-ZÅÄÖ0-9 /\-]{2,})$")
DM_SPEC_RE        = re.compile(r"(DM Symbol|X-dimension|Physical size|quiet zone)", re.IGNORECASE)
PI_RE             = re.compile(r"^PI-\d+$")
IKEA_URL_RE       = re.compile(r"IKEA\.com$", re.IGNORECASE)
COPY_LABEL_RE     = re.compile(r"^1\(3\)$|^[1-9]$")


def classify_span(span, label_rect):
    """Return an element-type string or None for non-elements."""
    txt = span["text"]
    fs  = span["font_size_pt"]
    lw  = label_rect.width
    lh  = label_rect.height
    rel_x = (span["x0"] - label_rect.x0) / lw if lw else 0
    rel_y = (span["y0"] - label_rect.y0) / lh if lh else 0

    if DM_SPEC_RE.search(txt) or txt.startswith("*"):
        return "barcode_spec_annotation"

    if ARTICLE_NUMBER_RE.match(txt):
        return "article_number"
    if ARTICLE_CODE_RE.match(txt):
        return "article_code"
    if AI_RE.search(txt):
        return "datamatrix_ai_text"
    if MADE_IN_RE.match(txt):
        return "origin_text"
    if IKEA_ADDR_RE.search(txt):
        return "address_block"
    if DESIGN_QUAL_RE.search(txt):
        return "ikea_logo"
    if LEGAL_RE.match(txt):
        return "legal_symbol"
    if WASH_RE.match(txt):
        return "wash_care_symbol"
    if CARE_INSTR_RE.search(txt):
        return "care_instruction_text"
    if CATALOG_NUM_RE.match(txt):
        return "catalog_number"
    if EPA_RE.search(txt):
        return "compliance_block"
    if PATENTS_RE.search(txt):
        return "patents_block"
    if ABCD_RE.match(txt):
        return "abcd_block"
    if DATE_LABEL_RE.match(txt):
        return "date_label"
    if YYWW_RE.match(txt) and rel_x > 0.55:
        return "date_stamp"
    if PI_RE.match(txt):
        return "pi_number"
    if IKEA_URL_RE.search(txt):
        return "address_block"
    if DIM_RE.search(txt):
        return "dimensions"
    if WEIGHT_RE.search(txt):
        return "weight"
    # product name: large text (>6pt), left side, ALL CAPS-ish
    if PROD_NAME_RE.match(txt) and rel_x < 0.45 and rel_y < 0.6 and fs >= 6:
        return "product_name"
    # copy count: small digit(s), right side OR "1(3)" anywhere
    if COPY_COUNT_RE.match(txt) and (rel_x > 0.55 or COPY_LABEL_RE.match(txt)):
        return "copy_count"
    return None


def summarise_elements(page, label_rect):
    """Return a dict of element_type → [spans] for all elements in label_rect."""
    spans = extract_text_in_rect(page, label_rect)
    summary = {}
    for sp in spans:
        etype = classify_span(sp, label_rect)
        if etype and etype != "barcode_spec_annotation":
            summary.setdefault(etype, []).append(sp)
    return summary, spans


# ── JSON building helpers ──────────────────────────────────────────────────────

def pt_to_mm(pt):
    return round(pt * 25.4 / 72, 1)

def rect_to_mm(rect):
    return {
        "width_mm_approx": round(rect.width * 25.4 / 72, 1),
        "height_mm_approx": round(rect.height * 25.4 / 72, 1),
    }

def zone_for_position(rel_x, family):
    """Guess zone from normalised x-position within a landscape label."""
    if rel_x < 0.30:
        return "left_identity"
    if rel_x < 0.48:
        return "center_barcode"
    if rel_x < 0.68:
        return "center_address"
    if rel_x < 0.82:
        return "right_legal_symbols"
    if rel_x < 0.93:
        return "right_date"
    return "right_copy_count"

def zone_for_column_position(rel_y, family):
    """Guess zone from normalised y-position within a portrait/column label."""
    if rel_y < 0.15:
        return "top_identity"
    if rel_y < 0.40:
        return "mid_dimensions"
    if rel_y < 0.65:
        return "center_barcode"
    if rel_y < 0.80:
        return "bot_legal"
    return "bot_address"

IS_BOLD_FLAG = 2**4  # PyMuPDF: bit 4 = bold

def make_element(etype, spans, label_rect, page_num, family, orientation="landscape"):
    """Build a comprehensive element dict for a given element type."""
    if not spans:
        return None

    rep = spans[0]
    txt = rep["text"]
    fs  = rep["font_size_pt"]
    bold = bool(rep.get("flags", 0) & IS_BOLD_FLAG)
    lw = label_rect.width
    lh = label_rect.height
    rel_x = (rep["x0"] - label_rect.x0) / lw if lw else 0
    rel_y = (rep["y0"] - label_rect.y0) / lh if lh else 0

    zone = zone_for_position(rel_x, family) if orientation == "landscape" else zone_for_column_position(rel_y, family)

    base = {
        "element_id": etype,
        "label": _label_name(etype),
        "semantic_role": _semantic_role(etype),
        "type": _element_type(etype),
        "required": _is_required(etype),
        "repeatable": etype == "legal_symbol",
        "content_type": "variable" if _is_variable(etype) else "static",
        "example_value": txt,
        "placement": {
            "zone": zone,
            "position_notes": f"Detected at relative position ({rel_x:.2f}, {rel_y:.2f}) within label",
            "alignment": "left" if rel_x < 0.6 else "center",
            "relative_to": "label_top_left",
        },
        "orientation": _orientation(etype, rel_x),
        "evidence": {
            "page": page_num,
            "evidence_type": "label_spec_text",
            "snippet": txt[:80],
        },
        "font": _font_spec(etype, fs, bold),
        "color": _color_spec(etype),
        "size_spec": rect_to_mm(fitz.Rect(rep["x0"], rep["y0"], rep["x1"], rep["y1"])),
    }

    # Element-type-specific additions
    if etype == "article_number":
        base["allowed_patterns"] = ["^\\d{3}[.\\-]\\d{3}[.\\-]\\d{2}$"]
    elif etype == "article_code":
        base["allowed_patterns"] = ["^\\d{5}$"]
    elif etype == "product_name":
        base["allowed_patterns"] = ["^[A-ZÅÄÖ0-9 /\\-]+$"]
    elif etype == "origin_text":
        base["allowed_patterns"] = ["^(Made in|Assembled in USA from Imported Materials|Assembled in) .+$"]
        base["allowed_values"] = [
            "Made in People's Republic of China",
            "Assembled in USA from Imported Materials",
            "Made in Vietnam", "Made in Poland", "Made in India",
            "Made in Bangladesh", "Made in Pakistan",
        ]
    elif etype == "date_stamp":
        base["allowed_patterns"] = ["^\\d{4}$"]
        base["content_rules"] = {
            "format": "YYWW",
            "yyww_pattern": "^(2[0-9]|[3-9][0-9])(0[1-9]|[1-4][0-9]|5[0-3])$",
            "note": "YY = 2-digit year; WW = ISO week 01-53",
        }
    elif etype == "copy_count":
        base["allowed_patterns"] = ["^\\d+$"]
    elif etype == "datamatrix_ai_text":
        base["element_id"] = "datamatrix_block"
        base["label"] = "DataMatrix barcode block"
        base["semantic_role"] = "barcode_symbol"
        base["type"] = "datamatrix"
        base["allowed_patterns"] = ["GS1 DataMatrix application identifiers"]
        base["shared_spec_ref"] = "DM_RECT"
    elif etype == "ikea_logo":
        base["content_type"] = "static"
        base["color"]["background_color"] = "#FFD100"
    elif etype == "abcd_block":
        base["allowed_patterns"] = ["^[A-Z]{4}$"]
    elif etype == "dimensions":
        base["allowed_patterns"] = ["^\\d+[x×]\\d+[x×]\\d+\\s*(cm|mm)?$"]
    elif etype == "weight":
        base["allowed_patterns"] = ["^\\d+(\\.\\d+)?\\s*(kg|lbs)$"]

    return base


def _label_name(etype):
    names = {
        "product_name": "Product name",
        "article_number": "Article number",
        "article_code": "Article code (short)",
        "ikea_logo": "IKEA logo",
        "datamatrix_ai_text": "DataMatrix barcode block",
        "datamatrix_block": "DataMatrix barcode block",
        "address_block": "Address and copyright block",
        "origin_text": "Country of origin / assembly text",
        "date_stamp": "YYWW datestamp",
        "date_label": "DATE: label prefix",
        "copy_count": "Copy count",
        "legal_symbol": "Legal / compliance symbol",
        "abcd_block": "ABCD alpha block",
        "compliance_block": "US EPA TSCA compliance text",
        "patents_block": "Patents reference",
        "dimensions": "Product dimensions",
        "weight": "Product weight",
        "pi_number": "PI number",
        "wash_care_symbol": "Wash / care care symbol",
        "care_instruction_text": "Care instruction text",
        "catalog_number": "Care label catalog number",
    }
    return names.get(etype, etype.replace("_", " ").title())

def _semantic_role(etype):
    roles = {
        "product_name": "product_identity",
        "article_number": "product_identity",
        "article_code": "product_identity",
        "ikea_logo": "brand_identity",
        "datamatrix_ai_text": "barcode_symbol",
        "datamatrix_block": "barcode_symbol",
        "address_block": "address_block",
        "origin_text": "origin_text",
        "date_stamp": "date_stamp",
        "date_label": "date_stamp",
        "copy_count": "copy_count",
        "legal_symbol": "compliance_symbol",
        "abcd_block": "date_stamp",
        "compliance_block": "compliance_text",
        "patents_block": "reference_text",
        "dimensions": "product_spec",
        "weight": "product_spec",
        "pi_number": "reference_text",
        "wash_care_symbol": "care_symbol",
        "care_instruction_text": "care_text",
        "catalog_number": "reference_text",
    }
    return roles.get(etype, "other")

def _element_type(etype):
    types = {
        "ikea_logo": "image",
        "datamatrix_ai_text": "datamatrix",
        "datamatrix_block": "datamatrix",
        "legal_symbol": "image",
        "address_block": "composite_block",
        "wash_care_symbol": "image",
        "care_instruction_text": "text",
        "catalog_number": "text",
        "compliance_block": "text",
        "patents_block": "text",
        "dimensions": "text",
        "weight": "text",
    }
    return types.get(etype, "text")

def _is_required(etype):
    optional = {"legal_symbol", "compliance_block", "patents_block", "abcd_block",
                "date_stamp", "date_label", "copy_count", "dimensions", "weight", "pi_number"}
    return etype not in optional

def _is_variable(etype):
    static = {"ikea_logo", "legal_symbol", "compliance_block", "patents_block", "date_label"}
    return etype not in static

def _orientation(etype, rel_x):
    if rel_x > 0.82 and etype in ("date_stamp", "abcd_block", "copy_count"):
        return "vertical_cw90"
    return "horizontal"

def _font_spec(etype, fs_pt, bold):
    size_mm = pt_to_mm(fs_pt)
    weight = "bold" if bold else "regular"
    spec = {
        "family": "IKEA proprietary sans-serif",
        "weight": weight,
        "style": "normal",
        "size_mm_approx": size_mm,
    }
    if etype == "product_name":
        spec["case"] = "upper"
        spec["weight"] = "bold"
    elif etype in ("article_number", "article_code"):
        spec["case"] = "mixed"
    elif etype in ("date_stamp", "copy_count"):
        spec["case"] = "numeric"
        spec["weight"] = "bold"
    elif etype == "origin_text":
        spec["case"] = "mixed"
    elif etype == "ikea_logo":
        spec["case"] = "upper"
        spec["weight"] = "bold"
        spec["note"] = "Logo asset; use LOGO_LG variant — see element_variants.json"
    return spec

def _color_spec(etype):
    if etype == "ikea_logo":
        return {"text_color": "#000000", "background_color": "#FFD100",
                "border": "solid_rectangle", "note": "IKEA corporate yellow"}
    if etype in ("article_number", "abcd_block"):
        return {"text_color": "#000000", "background_color": "variant_dependent",
                "note": "Background: transparent (SM), grey (MD/LG), black (XL) — see element_variants.json"}
    return {"text_color": "#000000", "background_color": "transparent"}


# ── Validation target generation ───────────────────────────────────────────────

def make_validation_targets(elements, template_id):
    targets = []
    chk = 1

    def _id():
        nonlocal chk
        tid = f"{template_id}-CHK-{chk:03d}"
        chk += 1
        return tid

    for el in elements:
        eid = el["element_id"]
        required = el.get("required", True)
        sev = "must" if required else "should"

        # presence check
        targets.append({
            "check_id": _id(),
            "check_type": "element_presence",
            "element_id": eid,
            "severity": sev,
            "description": f"{'Required' if required else 'Optional'} element '{eid}' {'must' if required else 'should'} be present",
        })

        # content format
        if el.get("allowed_patterns"):
            targets.append({
                "check_id": _id(),
                "check_type": "element_content_format",
                "element_id": eid,
                "severity": sev,
                "description": f"Element '{eid}' content must match allowed pattern(s)",
                "allowed_patterns": el["allowed_patterns"],
            })

        # placement zone
        zone = el.get("placement", {}).get("zone")
        if zone:
            targets.append({
                "check_id": _id(),
                "check_type": "element_placement",
                "element_id": eid,
                "severity": sev,
                "description": f"Element '{eid}' must be in zone '{zone}'",
                "expected_zone": zone,
            })

        # orientation
        orient = el.get("orientation")
        if orient and orient != "horizontal":
            targets.append({
                "check_id": _id(),
                "check_type": "element_orientation",
                "element_id": eid,
                "severity": sev,
                "description": f"Element '{eid}' must be oriented {orient}",
                "expected_orientation": orient,
            })

    # cross-element: product_name > article_number font size
    eids = [e["element_id"] for e in elements]
    if "product_name" in eids and "article_number" in eids:
        targets.append({
            "check_id": _id(),
            "check_type": "visual_hierarchy",
            "element_id": "product_name",
            "related_element_id": "article_number",
            "severity": "must",
            "description": "product_name font size must be visually larger than article_number",
        })
    if "article_number" in eids and "article_code" in eids:
        targets.append({
            "check_id": _id(),
            "check_type": "visual_hierarchy",
            "element_id": "article_number",
            "related_element_id": "article_code",
            "severity": "must",
            "description": "article_number font size must be visually larger than article_code",
        })
    if "origin_text" in eids and "address_block" in eids:
        targets.append({
            "check_id": _id(),
            "check_type": "cross_element_consistency",
            "element_id": "origin_text",
            "related_element_id": "address_block",
            "severity": "must",
            "description": "Origin country must not contradict address block region",
        })
    return targets


# ── Rules generation ───────────────────────────────────────────────────────────

def make_rules(elements, template_id, variant_num, base_has_compliance, has_copy):
    rules = []
    eids = {e["element_id"] for e in elements}
    r = 1

    def _rid():
        nonlocal r
        rid = f"{template_id}-R{r:03d}"
        r += 1
        return rid

    if "compliance_block" not in eids:
        rules.append({
            "rule_id": _rid(),
            "rule_type": "presence_rule",
            "scope": template_id,
            "condition": "always",
            "target": "compliance_block",
            "actions": [{"action_type": "remove",
                         "description": "No compliance block in this variant"}],
            "severity": "must",
            "normalized_text": "compliance_block MUST NOT be present",
            "evidence": {"page": None, "evidence_type": "diagram_annotation",
                         "snippet": f"Variant {variant_num} diagram shows no compliance zone"},
        })

    if "copy_count" in eids:
        rules.append({
            "rule_id": _rid(),
            "rule_type": "presence_rule",
            "scope": template_id,
            "condition": "always",
            "target": "copy_count",
            "actions": [{"action_type": "require",
                         "description": "Copy count strip must be present"}],
            "severity": "must",
            "normalized_text": "copy_count MUST be present",
            "evidence": {"page": None, "evidence_type": "diagram_annotation",
                         "snippet": f"Variant {variant_num} shows copy count strip"},
        })

    if "date_stamp" in eids:
        rules.append({
            "rule_id": _rid(),
            "rule_type": "format_rule",
            "scope": template_id,
            "condition": "always",
            "target": "date_stamp",
            "actions": [{"action_type": "validate",
                         "description": "Date must be YYWW (2-digit year + ISO week)"}],
            "severity": "must",
            "normalized_text": "date_stamp content MUST match ^(2[0-9]|[3-9][0-9])(0[1-9]|[1-4][0-9]|5[0-3])$",
            "evidence": {"page": None, "evidence_type": "diagram_annotation",
                         "snippet": "YYWW date format shown in right date strip"},
        })

    return rules


# ── Image cropping ─────────────────────────────────────────────────────────────

def crop_label(page, rect, out_path, dpi=CROP_DPI, pad=CROP_PAD):
    """Render a padded crop of rect at dpi and save PNG."""
    padded = fitz.Rect(
        max(0, rect.x0 - pad), max(0, rect.y0 - pad),
        min(page.rect.width,  rect.x1 + pad),
        min(page.rect.height, rect.y1 + pad),
    )
    scale = dpi / 72.0
    mat = fitz.Matrix(scale, scale)
    # Use clip param — render only the desired region
    pix = page.get_pixmap(matrix=mat, alpha=False, clip=padded)
    pix.save(str(out_path))


def render_full_page(page, out_path, dpi=FULL_DPI):
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    pix.save(str(out_path))


# ── Main build ─────────────────────────────────────────────────────────────────

def build():
    doc = fitz.open(str(PDF_PATH))
    print(f"Opened '{PDF_PATH}' — {len(doc)} pages\n")

    templates_root = OUT_ROOT / "templates"
    templates_root.mkdir(parents=True, exist_ok=True)

    # Copy shared specs + schemas from existing ground_truth
    for subdir in ["shared_specs", "schema"]:
        src = OUT_ROOT / subdir
        if src.exists():
            print(f"  ✓ keeping existing {subdir}/")

    # ── Pass 1: collect all detected label rects per template ──────────────────
    # Maps template_key → list of (page_num, fitz.Rect) sorted by (page, y0, x0)
    template_rects = {t["key"]: [] for t in TEMPLATE_MAP}

    for pg_num in range(1, len(doc) + 1):
        tmpl = PAGE_TO_TEMPLATE.get(pg_num)
        if tmpl is None:
            continue
        page = doc[pg_num - 1]
        rects = find_label_rects(page)
        if rects:
            for r in rects:
                template_rects[tmpl["key"]].append((pg_num, r))
        else:
            # Fallback: at least produce a full-page entry
            template_rects[tmpl["key"]].append((pg_num, page.rect))

    # ── Pass 2: process each template ─────────────────────────────────────────
    index_entries = []
    total_variants = 0

    for tmpl in TEMPLATE_MAP:
        key     = tmpl["key"]
        family  = tmpl["family"]
        pg_list = tmpl["pages"]
        existing = tmpl["existing"]

        folder = templates_root / key
        folder.mkdir(parents=True, exist_ok=True)

        # Render full-page reference images
        for pg_num in pg_list:
            pg_img = folder / f"page_{pg_num:02d}.png"
            if not pg_img.exists():
                render_full_page(doc[pg_num - 1], pg_img)
                print(f"  page ref: {key}/page_{pg_num:02d}.png")

        rects_for_tmpl = template_rects[key]

        if not rects_for_tmpl or family is None:
            # General spec / reference page — no variant subfolders
            index_entries.append({
                "key": key,
                "title": key.replace("-", " "),
                "family": family,
                "source_pages": pg_list,
                "folder": f"templates/{key}",
                "visual_artifacts": [f"templates/{key}/page_{p:02d}.png" for p in pg_list],
                "variants": [],
                "has_rich_json": False,
            })
            continue

        # ── Build variant list ─────────────────────────────────────────────
        variant_entries = []
        existing_jsons = existing or []

        for v_idx, (pg_num, rect) in enumerate(rects_for_tmpl):
            v_num = v_idx + 1
            v_name = f"v{v_num}"
            v_folder = folder / v_name
            v_folder.mkdir(exist_ok=True)

            label_png = v_folder / "label.png"
            tmpl_json_path = v_folder / "template.json"

            # Crop the label image
            page = doc[pg_num - 1]
            if rect == page.rect:
                # Fallback: use full page
                render_full_page(page, label_png, dpi=FULL_DPI)
            else:
                crop_label(page, rect, label_png)

            # ── Determine which JSON to use ────────────────────────────────
            rich_json_data = None

            if existing_jsons and v_idx < len(existing_jsons):
                # Try to load from old repo
                jname = existing_jsons[v_idx]
                old_path = OLD_RICH / jname
                if not old_path.exists():
                    # Try family subfolder
                    old_path = OLD_RICH / key / jname
                if old_path.exists():
                    rich_json_data = json.loads(old_path.read_text("utf-8"))
                    # Inject label_image path
                    rich_json_data["label_image"] = "label.png"
                    rich_json_data["source_page"] = pg_num
                    rich_json_data["label_rect_pts"] = {
                        "x0": round(rect.x0, 2), "y0": round(rect.y0, 2),
                        "x1": round(rect.x1, 2), "y1": round(rect.y1, 2),
                    }
                else:
                    print(f"  ⚠ old JSON not found: {old_path}")

            if rich_json_data is None:
                # Auto-generate comprehensive JSON
                rich_json_data = generate_json(
                    page, rect, key, family, pg_num, v_num, v_name
                )

            # Write template.json
            tmpl_json_path.write_text(
                json.dumps(rich_json_data, indent=2, ensure_ascii=False)
            )

            variant_entries.append({
                "variant_id": v_name,
                "template_id": rich_json_data.get("template_id", f"{key}_{v_name}"),
                "source_page": pg_num,
                "label_image": f"templates/{key}/{v_name}/label.png",
                "template_json": f"templates/{key}/{v_name}/template.json",
                "has_rich_json": rich_json_data.get("review_state", {}).get("confidence", 0) > 0.3,
            })
            total_variants += 1
            print(f"  ✓ {key}/{v_name}  (page {pg_num}, rect {rect.width:.0f}×{rect.height:.0f} pt)")

        index_entries.append({
            "key": key,
            "title": key.replace("-", " "),
            "family": family,
            "source_pages": pg_list,
            "folder": f"templates/{key}",
            "visual_artifacts": [f"templates/{key}/page_{p:02d}.png" for p in pg_list],
            "variants": variant_entries,
            "has_rich_json": any(v["has_rich_json"] for v in variant_entries),
        })

    # ── Write master index ─────────────────────────────────────────────────
    index = {
        "version": "2.0",
        "source_pdf": str(PDF_PATH),
        "total_pages": len(doc),
        "total_atomic_variants": total_variants,
        "description": (
            "Atomic IKEA label ground truth. Each template variant has its own "
            "subfolder containing a cropped label.png (300 DPI) and a comprehensive "
            "template.json with elements, rules, and validation_targets."
        ),
        "templates": index_entries,
    }
    (OUT_ROOT / "index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False)
    )

    print(f"\n✅ Done — {total_variants} atomic label variants")
    print(f"   Output: {OUT_ROOT.resolve()}")


# ── Auto JSON generation ───────────────────────────────────────────────────────

def generate_json(page, rect, key, family, page_num, variant_num, variant_name):
    """Produce a comprehensive template JSON by analysing the label's text."""
    template_id = f"{key}_{variant_name}"
    elem_map, raw_spans = summarise_elements(page, rect)

    # Determine label orientation from aspect ratio
    is_landscape = rect.width > rect.height * 1.2
    is_portrait  = rect.height > rect.width * 1.1
    orientation  = "landscape" if is_landscape else ("portrait" if is_portrait else "square")

    # Determine family category
    is_1R = family and family.startswith("1R")
    is_1C = family and family.startswith("1C")
    is_2R = family and family.startswith("2R")
    is_CIR = family and family.startswith("1CIR") or (family and "CIR" in family)
    is_L  = family and family.startswith("L-")

    # Build element list from classified spans
    elements = []
    seen_etypes = set()

    # Canonical element order for rich output
    CANONICAL_ORDER = [
        "product_name", "article_number", "article_code", "ikea_logo",
        "datamatrix_ai_text", "address_block", "origin_text",
        "compliance_block", "patents_block", "legal_symbol",
        "wash_care_symbol", "care_instruction_text", "catalog_number",
        "abcd_block", "date_stamp", "date_label", "copy_count",
        "dimensions", "weight", "pi_number",
    ]

    for etype in CANONICAL_ORDER:
        spans = elem_map.get(etype, [])
        if not spans:
            continue
        if etype in seen_etypes:
            continue
        seen_etypes.add(etype)
        el = make_element(etype, spans, rect, page_num, family, orientation)
        if el:
            elements.append(el)

    # Ensure datamatrix is represented if we saw AI text
    if "datamatrix_ai_text" in seen_etypes:
        for el in elements:
            if el["element_id"] == "datamatrix_ai_text":
                el["element_id"] = "datamatrix_block"

    # Detect variant features
    has_compliance = any(e["element_id"] == "compliance_block" for e in elements)
    has_copy       = any(e["element_id"] == "copy_count" for e in elements)
    has_date       = any(e["element_id"] == "date_stamp" for e in elements)
    has_legal      = any(e["element_id"] == "legal_symbol" for e in elements)
    has_abcd       = any(e["element_id"] == "abcd_block" for e in elements)

    # Variant description
    features = []
    if has_compliance: features.append("compliance")
    if has_date:       features.append("date")
    if has_copy:       features.append("copy-count")
    if has_legal:      features.append("legal-symbols")
    if has_abcd:       features.append("ABCD")
    feature_str = ", ".join(features) if features else "base"

    # Generate rules and validation targets
    rules = make_rules(elements, template_id, variant_name, has_compliance, has_copy)
    validation_targets = make_validation_targets(elements, template_id)

    # Estimate confidence based on how many standard elements were found
    standard_found = sum(1 for e in elements
                         if e["element_id"] in
                         ("product_name","article_number","datamatrix_block",
                          "address_block","origin_text","ikea_logo"))
    confidence = min(0.5 + standard_found * 0.08, 0.85)

    label_mm = rect_to_mm(rect)

    return {
        "$schema": "../../schema/template.schema.json",
        "template_id": template_id,
        "display_name": f"{key} Variant {variant_num} — {feature_str}",
        "family": family,
        "inherits_from": key,
        "label_image": "label.png",
        "source_page": page_num,
        "label_rect_pts": {
            "x0": round(rect.x0, 2), "y0": round(rect.y0, 2),
            "x1": round(rect.x1, 2), "y1": round(rect.y1, 2),
        },
        "label_size_mm": {
            "width_mm": label_mm["width_mm_approx"],
            "height_mm": label_mm["height_mm_approx"],
        },
        "orientation": orientation,
        "source_evidence": {
            "page": page_num,
            "position_on_page": f"variant row {variant_num}",
            "snippet": f"{key} — {feature_str}",
            "evidence_type": "label_spec_text",
        },
        "elements": elements,
        "overrides": [],
        "rules": rules,
        "validation_targets": validation_targets,
        "review_state": {
            "confidence": round(confidence, 2),
            "needs_review": confidence < 0.8,
            "review_notes": (
                f"Auto-generated from PDF text extraction (page {page_num}). "
                f"Elements detected: {[e['element_id'] for e in elements]}. "
                "Verify font sizes, zone boundaries, and rule completeness against label image."
            ),
        },
    }


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).parent)
    build()
