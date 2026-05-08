"""
Central configuration for the IKEA label validation system.

All tunable constants live here so they are easy to find and adjust without
hunting through source files.  Import from config.py instead of hardcoding
values in individual modules.

Usage:
    from config import DEFAULT_DPI, LABEL_RULES_DIR
"""

from pathlib import Path

# ── Repo root ──────────────────────────────────────────────────────────────────

ROOT_DIR = Path(__file__).resolve().parent

# ── Data directories ──────────────────────────────────────────────────────────

LABEL_RULES_DIR  = ROOT_DIR / "label-rules"
GROUND_TRUTH_DIR = ROOT_DIR / "ground_truth"

# ── Extraction settings ───────────────────────────────────────────────────────

# DPI used to rasterise PDFs.  Higher = more accurate OCR and barcode decoding
# but slower and more memory.  600 is the recommended minimum for production labels.
DEFAULT_DPI = 600

# Pixel width bounds for the processed image.  The extractor will upscale or
# downscale to stay within these limits before passing to OCR / barcode decoders.
MIN_IMAGE_WIDTH = 2400
MAX_IMAGE_WIDTH = 4500

# Minimum OCR confidence for a word to be included in zone summaries.
# Words below this threshold are still stored in all_word_boxes but are not
# used when matching field patterns.
MIN_OCR_CONFIDENCE = 0.25

# ── Validation settings ───────────────────────────────────────────────────────

# Overlap ratio thresholds for the overlap detector.
OVERLAP_IGNORE_THRESHOLD  = 0.05   # < 5%  → ignored (OCR jitter)
OVERLAP_WARNING_THRESHOLD = 0.25   # 5–25% → warning
# ≥ 25% → error

# Minimum type-detection score for the detected label type to be considered
# reliable.  Below this, validation results should be treated with caution.
MIN_TYPE_MATCH_CONFIDENCE = 0.35

# ── UI settings ───────────────────────────────────────────────────────────────

# DPI slider range in the Streamlit sidebar
UI_DPI_MIN     = 150
UI_DPI_MAX     = 600
UI_DPI_DEFAULT = 600
UI_DPI_STEP    = 50

# Allowed file types for the file uploader
ALLOWED_UPLOAD_TYPES = ["pdf", "png", "jpg", "jpeg", "tiff", "tif", "bmp"]

# Human corrections & JSONL fine-tune export
FINETUNE_DATA_DIR = ROOT_DIR / "finetune_data"

# Intermediate pipeline results (dissection.json, extraction.json per label)
OUTPUT_DIR = ROOT_DIR / "output"

# Upload cache: stores raw bytes of uploaded labels keyed by ``file_hash`` so
# the React UI can survive a page refresh and the validation microservice can
# re-run validation without making the browser re-upload the file. Both
# servers point at the same path; visual-validation reads it via the
# ``EXTRACTOR_UPLOAD_CACHE_DIR`` environment variable (defaults to this).
UPLOAD_CACHE_DIR = OUTPUT_DIR / "uploads"
