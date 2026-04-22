"""
Data models for the IKEA label extraction pipeline.

All extracted fields follow the ground truth element_ids from /ground_truth/
so the validation layer can directly match extracted values to rules.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional

try:
    import numpy as _np
    _NP_INTEGER = _np.integer
    _NP_FLOATING = _np.floating
    _NP_NDARRAY = _np.ndarray
except ImportError:  # numpy not available in some test environments
    _NP_INTEGER = None
    _NP_FLOATING = None
    _NP_NDARRAY = None


@dataclass
class BBox:
    x: int
    y: int
    w: int
    h: int
    # Normalized position (0.0–1.0) relative to source image dimensions.
    # Populated by the extractor after the image dimensions are known.
    x_frac: float = 0.0
    y_frac: float = 0.0
    w_frac: float = 0.0
    h_frac: float = 0.0
    zone: Optional[str] = None

    @property
    def center_x(self) -> float:
        return self.x + self.w / 2

    @property
    def center_y(self) -> float:
        return self.y + self.h / 2

    @property
    def right(self) -> int:
        return self.x + self.w

    @property
    def bottom(self) -> int:
        return self.y + self.h


@dataclass
class ExtractedField:
    """A single extracted text value with provenance."""
    value: Optional[str]
    raw_text: Optional[str]
    confidence: float           # 0.0–1.0  (OCR conf / 100, or 1.0 for barcode)
    zone: Optional[str]
    bbox: Optional[BBox]
    source: str                 # "ocr" | "barcode" | "derived"
    pattern_match: Optional[bool] = None  # True if value matched expected regex

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


@dataclass
class BarcodeResult:
    """A decoded barcode symbol (linear or 2-D)."""
    symbology: str              # "ITF-14" | "DataMatrix" | "EAN-13"
    raw_data: str               # raw bytes decoded as UTF-8 / latin-1
    decoded_text: Optional[str]
    bbox: Optional[BBox]
    confidence: float           # 1.0 = successfully decoded, 0.0 = failed
    source: str = "barcode"     # always "barcode"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AIField:
    """One GS1 Application Identifier extracted from a DataMatrix."""
    ai_code: str
    name: str
    description: str
    value: Optional[str]
    formatted: Optional[str]    # human-friendly value (e.g. date as YY-MM-DD)
    present: bool


@dataclass
class DataMatrixContent:
    """Parsed content of a GS1 DataMatrix symbol."""
    raw_data: str
    encoding: str               # "GS1" | "raw"
    ai_fields: dict[str, AIField] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d: dict = {
            "raw_data": self.raw_data,
            "encoding": self.encoding,
            "ai_fields": {k: asdict(v) for k, v in self.ai_fields.items()},
        }
        return d


@dataclass
class ZoneSummary:
    """Aggregated OCR content for one spatial zone."""
    zone_id: str
    raw_lines: list[str] = field(default_factory=list)
    word_count: int = 0
    avg_confidence: float = 0.0


@dataclass
class ExtractionMetadata:
    source_file: str
    extracted_at: str
    image_width_px: int
    image_height_px: int
    render_dpi: int
    libraries_used: dict = field(default_factory=dict)
    overall_confidence: float = 0.0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class ExtractedLabel:
    """
    Complete extraction result for one uploaded label.

    Field names mirror the ground truth element_ids so the validation
    layer can do a direct lookup without any key translation.
    """
    metadata: ExtractionMetadata

    # ── product identification ──────────────────────────────────────
    product_name: Optional[ExtractedField] = None
    product_dimensions_metric: Optional[ExtractedField] = None    # "20x25x10 cm"
    product_dimensions_imperial: Optional[ExtractedField] = None  # "7 ¼x9 ¾x4\""
    dimensions: Optional[ExtractedField] = None                   # legacy combined field

    # ── article / item data ─────────────────────────────────────────
    article_number: Optional[ExtractedField] = None      # "803.804.64"
    article_code: Optional[ExtractedField] = None        # derived from barcode digits
    internal_item_number: Optional[ExtractedField] = None  # "50029", "22714"
    quantity_multiplier: Optional[ExtractedField] = None   # "X 4"

    # ── branding ─────────────────────────────────────────────────────
    ikea_logo: Optional[ExtractedField] = None

    # ── weight ───────────────────────────────────────────────────────
    gross_weight: Optional[ExtractedField] = None   # "Gross 12 kg"
    net_weight: Optional[ExtractedField] = None     # "Net 10 kg"
    weight: Optional[ExtractedField] = None         # legacy combined field

    # ── packaging ────────────────────────────────────────────────────
    package_number: Optional[ExtractedField] = None  # "43", "2425"
    package_type: Optional[ExtractedField] = None    # "Not RTS Regular"
    copy_count_inline: Optional[ExtractedField] = None  # "1(2)"

    # ── linear barcode ────────────────────────────────────────────────
    ean_barcode: Optional[BarcodeResult] = None
    ean_barcode_value: Optional[ExtractedField] = None
    itf14_barcode: Optional[BarcodeResult] = None
    itf14_human_readable: Optional[ExtractedField] = None

    # ── datamatrix / 2-D barcode ─────────────────────────────────────
    datamatrix_barcode: Optional[BarcodeResult] = None
    datamatrix_content: Optional[DataMatrixContent] = None
    ai_240: Optional[ExtractedField] = None  # AI(240) product code
    ai_13: Optional[ExtractedField] = None   # AI(13) production date YYMMDD
    ai_11: Optional[ExtractedField] = None   # AI(11) pack date YYMMDD
    ai_10: Optional[ExtractedField] = None   # AI(10) batch / lot

    # ── date information ─────────────────────────────────────────────
    human_readable_date: Optional[ExtractedField] = None  # "24-06-19 (YY-MM-DD)"
    date_stamp: Optional[ExtractedField] = None           # YYWW value
    date_label_present: bool = False
    date_alpha_code: Optional[ExtractedField] = None      # e.g. "QWED"
    date_numeric_prefix: Optional[ExtractedField] = None  # e.g. "8989"
    custom_identifier: Optional[ExtractedField] = None    # full: "QWED-2501-8989"

    # ── supplier / factory ───────────────────────────────────────────
    plant_identifier: Optional[ExtractedField] = None   # "PI-200948-2"
    supplier_name: Optional[ExtractedField] = None      # "IKEA of Sweden AB"
    supplier_address: Optional[ExtractedField] = None   # "SE - 34381 Älmhult"
    patent_info: Optional[ExtractedField] = None        # legacy alias for plant_identifier

    # ── address / legal ──────────────────────────────────────────────
    address_block: Optional[ExtractedField] = None
    origin_text: Optional[ExtractedField] = None
    copyright_notice: Optional[ExtractedField] = None

    # ── compliance ───────────────────────────────────────────────────
    compliance_marks: Optional[ExtractedField] = None
    age_rating: Optional[ExtractedField] = None

    # ── copy count ───────────────────────────────────────────────────
    copy_count: Optional[ExtractedField] = None

    # ── raw zone text (debugging / fallback) ─────────────────────────
    zones: dict[str, ZoneSummary] = field(default_factory=dict)

    # ── every OCR word detected across all zones ──────────────────────
    # Each entry: {text, x, y, w, h, conf, zone}
    # Exposed so the UI can draw a box around every detected token.
    all_word_boxes: list = field(default_factory=list)

    # ── raw multi-pass elements (pdf_passes output) ────────────────────
    # Complete primitive inventory from all dissection passes:
    #   Pass 1: text_pdf  — PyMuPDF span positions (exact, confidence=1.0)
    #   Pass 2: image_pdf — embedded raster objects (logos, icons)
    #   Pass 4: region    — OpenCV contour-detected layout zones
    #   Pass 4: line      — OpenCV Hough-detected dividers / borders
    # text_ocr entries added for words found only by OCR (not in PDF text).
    raw_elements: list = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialise to a plain dict ready for json.dumps()."""
        def _serialise(obj):
            if obj is None:
                return None
            if _NP_INTEGER is not None and isinstance(obj, _NP_INTEGER):
                return int(obj)
            if _NP_FLOATING is not None and isinstance(obj, _NP_FLOATING):
                return float(obj)
            if _NP_NDARRAY is not None and isinstance(obj, _NP_NDARRAY):
                return obj.tolist()
            if hasattr(obj, "to_dict"):
                return obj.to_dict()
            if hasattr(obj, "__dataclass_fields__"):
                return asdict(obj)
            if isinstance(obj, dict):
                return {k: _serialise(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_serialise(i) for i in obj]
            return obj

        return {
            "extraction_metadata": asdict(self.metadata),
            "elements": {
                # product identification
                "product_name":               _serialise(self.product_name),
                "product_dimensions_metric":  _serialise(self.product_dimensions_metric),
                "product_dimensions_imperial":_serialise(self.product_dimensions_imperial),
                "dimensions":                 _serialise(self.dimensions),
                # article / item data
                "article_number":      _serialise(self.article_number),
                "article_code":        _serialise(self.article_code),
                "internal_item_number":_serialise(self.internal_item_number),
                "quantity_multiplier": _serialise(self.quantity_multiplier),
                # branding
                "ikea_logo":           _serialise(self.ikea_logo),
                # weight
                "gross_weight":        _serialise(self.gross_weight),
                "net_weight":          _serialise(self.net_weight),
                "weight":              _serialise(self.weight),
                # packaging
                "package_number":      _serialise(self.package_number),
                "package_type":        _serialise(self.package_type),
                "copy_count_inline":   _serialise(self.copy_count_inline),
                # barcodes
                "ean_barcode":         _serialise(self.ean_barcode),
                "ean_barcode_value":   _serialise(self.ean_barcode_value),
                "itf14_barcode":       _serialise(self.itf14_barcode),
                "itf14_human_readable":_serialise(self.itf14_human_readable),
                # datamatrix
                "datamatrix_block": {
                    "barcode": _serialise(self.datamatrix_barcode),
                    "content": _serialise(self.datamatrix_content),
                    "ai_fields": {
                        "240": _serialise(self.ai_240),
                        "13":  _serialise(self.ai_13),
                        "11":  _serialise(self.ai_11),
                        "10":  _serialise(self.ai_10),
                    },
                },
                # date
                "human_readable_date": _serialise(self.human_readable_date),
                "date_stamp":          _serialise(self.date_stamp),
                "date_label_present":  self.date_label_present,
                "date_alpha_code":     _serialise(self.date_alpha_code),
                "date_numeric_prefix": _serialise(self.date_numeric_prefix),
                "custom_identifier":   _serialise(self.custom_identifier),
                # supplier / factory
                "plant_identifier":    _serialise(self.plant_identifier),
                "supplier_name":       _serialise(self.supplier_name),
                "supplier_address":    _serialise(self.supplier_address),
                "patent_info":         _serialise(self.patent_info),
                # address / legal
                "address_block":       _serialise(self.address_block),
                "origin_text":         _serialise(self.origin_text),
                "copyright_notice":    _serialise(self.copyright_notice),
                # compliance
                "compliance_marks":    _serialise(self.compliance_marks),
                "age_rating":          _serialise(self.age_rating),
                # copy count
                "copy_count":          _serialise(self.copy_count),
            },
            "zones": {
                k: asdict(v) for k, v in self.zones.items()
            },
            "raw_words": _serialise(self.all_word_boxes),
            "raw_elements": [_serialise(e) for e in self.raw_elements],
        }
