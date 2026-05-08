"""
IKEA Label Extractor package.

Responsible for: loading label files (PDF/image), running the multi-pass
extraction pipeline (PDF native text, barcode decoding, OCR ensemble), and
returning a fully-populated ExtractedLabel with bounding boxes, confidence
scores, zone assignments, and decoded barcode data.

Entry point:

    from extractor import LabelExtractor

    result = LabelExtractor("label.pdf").run()
    print(result.to_json(indent=2))
"""

from .label_extractor import LabelExtractor
from .label_dissector import LabelDissector
from .ocr_engine import warmup_ocr_readers
from .models import ExtractedLabel, ExtractedField, BarcodeResult, DataMatrixContent
from .zone_segmenter import detect_label_shape, get_zone_cfg_for_shape
from .dissection_store import (
    save_dissection,
    save_extraction,
    load_dissection,
    load_extraction,
    get_output_dir,
)

__all__ = [
    "LabelExtractor",
    "LabelDissector",
    "warmup_ocr_readers",
    "ExtractedLabel",
    "ExtractedField",
    "BarcodeResult",
    "DataMatrixContent",
    "detect_label_shape",
    "get_zone_cfg_for_shape",
    "save_dissection",
    "save_extraction",
    "load_dissection",
    "load_extraction",
    "get_output_dir",
]
