"""
LabelExtractor — orchestrator for the IKEA label extraction pipeline.

Pipeline
--------
1. load()              PDF / image → high-res PIL Image
2. decode_barcodes()   ITF-14 (pyzbar) + DataMatrix (pylibdmtx) + EAN-13
3. calibrate_zones()   Anchor zone boundaries to detected barcode positions
4. run_pdf_passes()    PDF-native text (Pass 1), embedded images (Pass 2),
                       OpenCV visual regions (Pass 4)
5. run_ocr()           PaddleOCR → EasyOCR → Tesseract ensemble
                       (delegated to ocr_engine.run_ocr)
6. extract_fields()    Pattern-match word data → every ExtractedLabel field
                       (delegated to field_mapper.extract_fields)
7. detect_compliance() Pixel-level analysis for graphical CE / compliance symbols

The heavy lifting lives in the sub-modules:
  extractor/_helpers.py     – shared data types and utility functions
  extractor/ocr_engine.py   – OCR engine management, run_ocr(), _merge_word_lists()
  extractor/field_mapper.py – regex patterns, sub-extractors, extract_fields()
  extractor/zone_segmenter.py – shape detection, zone-config tables
  extractor/pdf_passes.py   – PyMuPDF + OpenCV document dissection
  extractor/gs1_parser.py   – GS1 AI barcode parsing
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional, Callable

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from ._helpers import _make_bbox
from .field_mapper import extract_fields
from .gs1_parser import detect_and_parse
from .models import (
    BBox,
    BarcodeResult,
    ExtractedLabel,
)
from .ocr_engine import run_ocr, warmup_ocr_readers
from .pdf_passes import (
    RawElement,
    run_pass1_text,
    run_pass2_images,
    run_pass4_visual_regions,
    merge_raw_elements,
    HAS_FITZ,
)
from .zone_segmenter import (
    LANDSCAPE_ZONE_CFG,
    detect_label_shape,
    get_zone_cfg_for_shape,
)

logger = logging.getLogger(__name__)

# ── macOS Homebrew dylib discovery ────────────────────────────────────────────
import ctypes.util as _ctypes_util
import os as _os

_BREW_LIB_DIRS = ["/opt/homebrew/lib", "/usr/local/lib"]
_KNOWN_ALIASES: dict[str, list[str]] = {
    "zbar": ["libzbar.0.dylib", "libzbar.dylib"],
    "dmtx": ["libdmtx.0.dylib", "libdmtx.dylib"],
}
_orig_find_library = _ctypes_util.find_library


def _brew_find_library(name: str) -> str | None:
    for alias in _KNOWN_ALIASES.get(name, []):
        for lib_dir in _BREW_LIB_DIRS:
            candidate = _os.path.join(lib_dir, alias)
            if _os.path.exists(candidate):
                return candidate
    return _orig_find_library(name)


_ctypes_util.find_library = _brew_find_library

# ── Optional dependencies ─────────────────────────────────────────────────────

try:
    from pdf2image import convert_from_path as _pdf2img
    HAS_PDF2IMAGE = True
except ImportError:
    HAS_PDF2IMAGE = False

try:
    from pyzbar.pyzbar import decode as _pyzbar_decode, ZBarSymbol
    HAS_PYZBAR = True
except ImportError:
    HAS_PYZBAR = False

try:
    from pylibdmtx.pylibdmtx import decode as _dmtx_decode
    HAS_DMTX = True
except ImportError:
    HAS_DMTX = False

try:
    import cv2 as _cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import easyocr as _easyocr_chk
    HAS_EASYOCR = True
except ImportError:
    HAS_EASYOCR = False

try:
    _os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    from paddleocr import PaddleOCR as _paddle_chk
    HAS_PADDLE = True
except Exception:
    HAS_PADDLE = False

try:
    import pytesseract as _tess_chk
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_DPI     = 600
MAX_IMAGE_WIDTH = 4500
MIN_IMAGE_WIDTH = 2400


# ── Image preprocessing ───────────────────────────────────────────────────────

def _barcode_variants(img: Image.Image) -> list[Image.Image]:
    """
    Return a sequence of increasingly aggressive preprocessed images to try
    when decoding barcodes.  PIL-only variants are always produced; cv2-based
    variants are added when opencv-python is available.
    """
    gray_pil = img.convert("L")
    variants: list[Image.Image] = [
        img,
        gray_pil,
        ImageEnhance.Contrast(img).enhance(2.0),
        img.filter(ImageFilter.SHARPEN),
        ImageEnhance.Contrast(gray_pil).enhance(2.0),
    ]
    if HAS_CV2:
        arr = np.array(gray_pil)
        at = _cv2.adaptiveThreshold(
            arr, 255, _cv2.ADAPTIVE_THRESH_GAUSSIAN_C, _cv2.THRESH_BINARY, 11, 2
        )
        variants.append(Image.fromarray(at))
        _, ot = _cv2.threshold(arr, 0, 255, _cv2.THRESH_BINARY + _cv2.THRESH_OTSU)
        variants.append(Image.fromarray(ot))
        variants.append(Image.fromarray(_cv2.equalizeHist(arr)))
        kernel = _cv2.getStructuringElement(_cv2.MORPH_RECT, (3, 3))
        variants.append(Image.fromarray(
            _cv2.morphologyEx(arr, _cv2.MORPH_CLOSE, kernel)
        ))
    return variants


# ── LabelExtractor ────────────────────────────────────────────────────────────

class LabelExtractor:
    """
    Full extraction pipeline for one uploaded IKEA label file.

    Usage
    -----
    result = LabelExtractor("label.pdf").run()
    print(result.to_json(indent=2))
    """

    def __init__(self, source_path: str | Path, dpi: int = DEFAULT_DPI):
        self.source_path = Path(source_path)
        self.dpi = dpi
        self._image: Optional[Image.Image] = None
        self._label_shape: str = "landscape"
        self._zone_cfg: list = list(LANDSCAPE_ZONE_CFG)
        self._warnings: list[str] = []
        self._errors: list[str] = []
        self._pdf_path: Optional[str] = None
        self._raw_elements: list[RawElement] = []

    # ── Step 1: Load ──────────────────────────────────────────────────────────

    def load(self) -> Image.Image:
        suffix = self.source_path.suffix.lower()
        if suffix == ".pdf":
            if not HAS_PDF2IMAGE:
                raise RuntimeError("pdf2image required for PDF files.")
            pages = _pdf2img(str(self.source_path), dpi=self.dpi, fmt="png",
                             first_page=1, last_page=1)
            if not pages:
                raise ValueError(f"pdf2image returned no pages for {self.source_path}")
            self._image = pages[0]
            self._pdf_path = str(self.source_path)
        elif suffix in {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}:
            self._image = Image.open(self.source_path).convert("RGB")
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

        w, h = self._image.size

        if suffix != ".pdf" and w < MIN_IMAGE_WIDTH:
            scale = MIN_IMAGE_WIDTH / w
            self._image = self._image.resize(
                (MIN_IMAGE_WIDTH, int(h * scale)), Image.LANCZOS
            )
            w, h = self._image.size
            self._warnings.append(
                f"Image upscaled ×{scale:.2f} to {w}×{h} px "
                f"(original was below {MIN_IMAGE_WIDTH} px wide)."
            )
            logger.info("Upscaled image to %dx%d px (scale ×%.2f)", w, h, scale)

        if w > MAX_IMAGE_WIDTH:
            scale = MAX_IMAGE_WIDTH / w
            self._image = self._image.resize(
                (MAX_IMAGE_WIDTH, int(h * scale)), Image.LANCZOS
            )
            self._warnings.append(f"Image scaled from {w}px to {MAX_IMAGE_WIDTH}px wide.")

        # Detect shape and choose appropriate zone config
        self._label_shape = detect_label_shape(self._image.size[0], self._image.size[1])
        self._zone_cfg = list(get_zone_cfg_for_shape(self._label_shape))
        if self._label_shape != "landscape":
            self._warnings.append(
                f"Non-landscape label detected (shape={self._label_shape!r}); "
                f"using {self._label_shape} zone layout."
            )

        logger.info(
            "Loaded %s → %dx%d px (shape=%s)",
            self.source_path.name, *self._image.size, self._label_shape
        )
        return self._image

    # ── Step 2: Decode barcodes ───────────────────────────────────────────────

    def decode_barcodes(self) -> dict[str, BarcodeResult | None]:
        """
        Decode ITF-14, EAN-13, and DataMatrix from the label image.

        Three-pass strategy:
          Pass A – full image with all preprocessing variants (up to 9 filters)
          Pass B – full image rescaled ×1.5 and ×2.0
          Pass C – zone crops targeting the expected barcode/datamatrix region
        """
        results: dict[str, BarcodeResult | None] = {
            "itf14": None, "datamatrix": None, "ean13": None
        }
        img_w, img_h = self._image.size
        _ITF = {"I25", "ITF", "ITF-14", "CODE-ITF"}
        _EAN = {"EAN13", "EAN-13"}

        def _try_pyzbar(img: Image.Image, x_offset: int = 0, y_offset: int = 0) -> None:
            if not HAS_PYZBAR:
                return
            try:
                for sym in _pyzbar_decode(img, symbols=[ZBarSymbol.I25, ZBarSymbol.EAN13]):
                    data = sym.data.decode("utf-8", errors="replace")
                    xs = [p.x + x_offset for p in sym.polygon]
                    ys = [p.y + y_offset for p in sym.polygon]
                    x, y = min(xs), min(ys)
                    w, h = max(xs) - x, max(ys) - y
                    bb = _make_bbox(x, y, w, h, img_w, img_h)
                    br = BarcodeResult(sym.type, data, data, bb, 1.0)
                    if sym.type in _ITF and results["itf14"] is None:
                        results["itf14"] = br
                    elif sym.type in _EAN and results["ean13"] is None:
                        results["ean13"] = br
            except Exception as e:
                self._warnings.append(f"pyzbar: {e}")

        def _try_dmtx(img: Image.Image, x_offset: int = 0, y_offset: int = 0) -> None:
            if not HAS_DMTX or results["datamatrix"] is not None:
                return
            try:
                decoded = _dmtx_decode(img, timeout=5000)
                if decoded:
                    sym = decoded[0]
                    raw = sym.data.decode("utf-8", errors="replace")
                    r = sym.rect
                    results["datamatrix"] = BarcodeResult(
                        "DataMatrix", raw, raw,
                        _make_bbox(r.left + x_offset, r.top + y_offset,
                                   r.width, r.height, img_w, img_h),
                        1.0,
                    )
            except Exception as e:
                self._warnings.append(f"pylibdmtx: {e}")

        def _all_found() -> bool:
            return all(v is not None for v in results.values())

        def _run_variants(img: Image.Image, x_offset: int = 0, y_offset: int = 0) -> None:
            for variant in _barcode_variants(img):
                if results["itf14"] is None or results["ean13"] is None:
                    _try_pyzbar(variant, x_offset, y_offset)
                if results["datamatrix"] is None:
                    _try_dmtx(variant, x_offset, y_offset)
                if _all_found():
                    return

        # Pass A: full image
        _run_variants(self._image)
        if _all_found():
            return results

        logger.debug("decode_barcodes: Pass A incomplete, trying multi-scale (Pass B)")

        # Pass B: rescaled full image (1.5× and 2.0×)
        for scale in (1.5, 2.0):
            if _all_found():
                break
            sw, sh = int(img_w * scale), int(img_h * scale)
            scaled = self._image.resize((sw, sh), Image.LANCZOS)
            for variant in _barcode_variants(scaled):
                if results["itf14"] is None or results["ean13"] is None:
                    if HAS_PYZBAR:
                        try:
                            for sym in _pyzbar_decode(
                                variant, symbols=[ZBarSymbol.I25, ZBarSymbol.EAN13]
                            ):
                                data = sym.data.decode("utf-8", errors="replace")
                                xs = [p.x / scale for p in sym.polygon]
                                ys = [p.y / scale for p in sym.polygon]
                                x, y = int(min(xs)), int(min(ys))
                                w, h = int(max(xs) - x), int(max(ys) - y)
                                bb = _make_bbox(x, y, w, h, img_w, img_h)
                                br = BarcodeResult(sym.type, data, data, bb, 1.0)
                                if sym.type in _ITF and results["itf14"] is None:
                                    results["itf14"] = br
                                elif sym.type in _EAN and results["ean13"] is None:
                                    results["ean13"] = br
                        except Exception as e:
                            self._warnings.append(f"pyzbar scale={scale}: {e}")
                if results["datamatrix"] is None and HAS_DMTX:
                    try:
                        decoded = _dmtx_decode(variant, timeout=5000)
                        if decoded:
                            sym = decoded[0]
                            raw = sym.data.decode("utf-8", errors="replace")
                            r = sym.rect
                            results["datamatrix"] = BarcodeResult(
                                "DataMatrix", raw, raw,
                                _make_bbox(
                                    int(r.left / scale), int(r.top / scale),
                                    int(r.width / scale), int(r.height / scale),
                                    img_w, img_h,
                                ),
                                1.0,
                            )
                    except Exception as e:
                        self._warnings.append(f"pylibdmtx scale={scale}: {e}")
                if _all_found():
                    break

        if _all_found():
            return results

        logger.debug("decode_barcodes: Pass B incomplete, trying zone crops (Pass C)")

        # Pass C: zone-crop retries
        # For portrait/circular labels, barcode is in the middle y-fraction strip.
        if self._label_shape in ("portrait", "circular"):
            # Mid zone — y 28%–55%
            y0 = int(0.28 * img_h)
            y1 = int(0.55 * img_h)
            crop_bc = self._image.crop((0, y0, img_w, y1))
            _run_variants(crop_bc, y_offset=y0)
        else:
            if results["itf14"] is None or results["ean13"] is None:
                x0 = int(0.18 * img_w)
                x1 = int(0.40 * img_w)
                crop_itf = self._image.crop((x0, 0, x1, img_h))
                _run_variants(crop_itf, x_offset=x0)
            if results["datamatrix"] is None:
                x0 = int(0.40 * img_w)
                x1 = int(0.58 * img_w)
                crop_dm = self._image.crop((x0, 0, x1, img_h))
                _run_variants(crop_dm, x_offset=x0)

        return results

    # ── Step 3: Calibrate zones using barcode bboxes ──────────────────────────

    def calibrate_zones(self, barcodes: dict) -> None:
        """Refine landscape zone boundaries by anchoring to detected barcode positions."""
        if self._label_shape != "landscape":
            return  # portrait/circular zones are geometry-fixed, not barcode-calibrated
        img_w = self._image.size[0]
        itf = barcodes.get("itf14")
        dm  = barcodes.get("datamatrix")

        if itf and itf.bbox and dm and dm.bbox:
            itf_left = itf.bbox.x / img_w
            dm_left  = dm.bbox.x  / img_w
            dm_text_right = min((dm.bbox.x + dm.bbox.w) / img_w + 0.12, 0.58)

            _default_left_end = 0.18
            left_end = max(
                itf_left * 0.85,
                min(_default_left_end, itf_left - 0.01),
            )
            left_end = min(left_end, itf_left - 0.005)

            self._zone_cfg = [
                ("left_identity",     0.00,            left_end,       0),
                ("center_barcode",    itf_left * 0.85, dm_left - 0.01, 0),
                ("center_datamatrix", dm_left - 0.01,  dm_text_right,  0),
                ("center_address",    dm_text_right,   0.72,           0),
                ("right_compliance",  0.72,            0.84,           0),
                ("right_date",        0.84,            0.95,           0),
                ("right_copy_count",  0.95,            1.00,          -90),
            ]
            logger.info(
                "Zones calibrated: itf_left=%.3f dm_left=%.3f dm_text_right=%.3f",
                itf_left, dm_left, dm_text_right,
            )
        else:
            logger.info("Using default zone boundaries (no barcode pair to calibrate from).")

    # ── Step 3b: Multi-pass PDF dissection ────────────────────────────────────

    def run_pdf_passes(self) -> list[RawElement]:
        """
        Run all applicable dissection passes and return a merged element list.

        Pass 1  — PyMuPDF text spans (PDFs only).
        Pass 2  — PyMuPDF embedded images (PDFs only).
        Pass 4  — OpenCV visual regions (all input types).
        """
        img_w, img_h = self._image.size
        pass1: list[RawElement] = []
        pass2: list[RawElement] = []

        if self._pdf_path and HAS_FITZ:
            pass1 = run_pass1_text(self._pdf_path, dpi=self.dpi)
            pass2 = run_pass2_images(self._pdf_path, dpi=self.dpi)

            for el in pass1:
                cx_frac = (el.x + el.w / 2) / img_w
                cy_frac = (el.y + el.h / 2) / img_h
                el.zone = self._zone_for_position(cx_frac, cy_frac)

            for el in pass1 + pass2:
                el.x = max(0, min(el.x, img_w - 1))
                el.y = max(0, min(el.y, img_h - 1))
                el.w = max(1, min(el.w, img_w - el.x))
                el.h = max(1, min(el.h, img_h - el.y))

            logger.info("PDF passes: %d text spans, %d embedded images", len(pass1), len(pass2))
        else:
            if self._pdf_path and not HAS_FITZ:
                self._warnings.append(
                    "PyMuPDF not installed — exact PDF text positions unavailable. "
                    "Run: pip install pymupdf"
                )

        img_arr = np.array(self._image.convert("RGB"))
        pass4 = run_pass4_visual_regions(img_arr)
        for el in pass4:
            cx_frac = (el.x + el.w / 2) / img_w
            cy_frac = (el.y + el.h / 2) / img_h
            el.zone = self._zone_for_position(cx_frac, cy_frac)

        self._raw_elements = merge_raw_elements(pass1, pass2, pass4)
        logger.info(
            "run_pdf_passes: %d total raw elements (%d text, %d images, %d regions)",
            len(self._raw_elements), len(pass1), len(pass2), len(pass4),
        )
        return self._raw_elements

    # ── Step 6: Compliance mark pixel analysis ────────────────────────────────

    def detect_compliance_marks(self, barcodes: dict) -> list[str]:
        """Pixel-level detection of CE and other compliance symbols."""
        if self._label_shape == "landscape":
            x0_frac, x1_frac = 0.72, 0.84
            for entry in self._zone_cfg:
                if entry[0] == "right_compliance":
                    x0_frac, x1_frac = entry[1], entry[2]
                    break
            img_w, img_h = self._image.size
            x0 = int(x0_frac * img_w)
            x1 = min(img_w, int(x1_frac * img_w))
            crop = self._image.crop((x0, 0, x1, img_h)).convert("L")
        else:
            # For portrait/circular, compliance zone is in the bot_compliance strip
            img_w, img_h = self._image.size
            y0_frac, y1_frac = 0.75, 0.90
            for entry in self._zone_cfg:
                if "compliance" in entry[0]:
                    y0_frac, y1_frac = entry[1], entry[2]
                    break
            y0 = int(y0_frac * img_h)
            y1 = min(img_h, int(y1_frac * img_h))
            crop = self._image.crop((0, y0, img_w, y1)).convert("L")

        arr = np.array(crop)
        detected: list[str] = []
        dark = (arr < 128).astype(np.uint8)
        dark_ratio = dark.mean()

        if dark_ratio > 0.05:
            detected.append("compliance_symbols_detected")
            left_half  = dark[:, : dark.shape[1] // 2].mean()
            right_half = dark[:, dark.shape[1] // 2:].mean()
            if left_half > 0.04 and right_half > 0.04:
                detected.append("CE")

        return detected

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _zone_for_position(self, x_frac: float, y_frac: float = 0.5) -> str:
        """Return zone_id for a word at (x_frac, y_frac).

        For landscape labels, x_frac drives the assignment.
        For portrait and circular labels, y_frac drives the assignment.
        """
        coord = y_frac if self._label_shape in ("portrait", "circular") else x_frac
        for zone_id, f0, f1, _rot in self._zone_cfg:
            if f0 <= coord < f1:
                return zone_id
        if self._zone_cfg:
            return self._zone_cfg[-1][0]
        return "unknown"

    def _build_zone_fn(self) -> "Callable[[float, float], str]":
        """Return a closure over the current zone config for use by run_ocr."""
        zone_cfg = self._zone_cfg
        shape = self._label_shape

        def _zone_fn(x_frac: float, y_frac: float = 0.5) -> str:
            coord = y_frac if shape in ("portrait", "circular") else x_frac
            for zone_id, f0, f1, _rot in zone_cfg:
                if f0 <= coord < f1:
                    return zone_id
            return zone_cfg[-1][0] if zone_cfg else "unknown"

        return _zone_fn

    def _libraries_used(self) -> dict:
        return {
            "pdf": "pdf2image" if HAS_PDF2IMAGE else "unavailable",
            "ocr": (
                "paddleocr+easyocr" if (HAS_PADDLE and HAS_EASYOCR) else
                "paddleocr"         if HAS_PADDLE   else
                "easyocr"           if HAS_EASYOCR  else
                "pytesseract"       if HAS_TESSERACT else "unavailable"
            ),
            "barcode_linear": "pyzbar"    if HAS_PYZBAR else "unavailable",
            "barcode_2d":     "pylibdmtx" if HAS_DMTX   else "unavailable",
        }

    # ── Public entry point ────────────────────────────────────────────────────

    def run(self) -> ExtractedLabel:
        """
        Execute the full multi-pass dissection pipeline and return an ExtractedLabel.

        Step 1   load()                 PDF/image → high-res PIL Image; shape detected
        Step 2   decode_barcodes()      ITF-14 / EAN-13 / DataMatrix (3-pass)
        Step 3   calibrate_zones()      Anchor landscape zones to barcode positions
        Step 3b  run_pdf_passes()       PDF Pass 1 + Pass 2 + visual Pass 4
        Step 4   run_ocr()              PaddleOCR → EasyOCR → Tesseract ensemble
        Step 5   extract_fields()       Confidence-gated pattern matching
        Step 6   detect_compliance()    Pixel analysis for CE symbols
        """
        self.load()
        barcodes = self.decode_barcodes()
        self.calibrate_zones(barcodes)
        self.run_pdf_passes()

        zone_fn = self._build_zone_fn()
        ocr_words = run_ocr(
            image=self._image,
            zone_cfg=self._zone_cfg,
            zone_fn=zone_fn,
            raw_elements=self._raw_elements,
            warnings=self._warnings,
            errors=self._errors,
            is_portrait=self._label_shape in ("portrait", "circular"),
        )

        label = extract_fields(
            image=self._image,
            ocr_words=ocr_words,
            barcodes=barcodes,
            zone_cfg=self._zone_cfg,
            raw_elements=self._raw_elements,
            source_path=self.source_path,
            dpi=self.dpi,
            warnings=self._warnings,
            errors=self._errors,
            libraries_used=self._libraries_used(),
        )

        # Augment compliance with pixel-level symbol detection
        detected_marks = self.detect_compliance_marks(barcodes)
        if detected_marks and label.compliance_marks is not None:
            existing = label.compliance_marks.value or ""
            combined = ", ".join(filter(None, [existing] + [
                m for m in detected_marks if m not in existing
            ]))
            label.compliance_marks.value = combined
        elif detected_marks:
            from ._helpers import _field as _f
            label.compliance_marks = _f(
                ", ".join(detected_marks), "", 0.60, "right_compliance"
            )

        return label

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.run().to_dict(), indent=indent, ensure_ascii=False)
