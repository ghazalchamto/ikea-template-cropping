"""
OCR engine management for the IKEA label extractor.

Responsibilities
----------------
  • Manage lazy-loaded singletons for PaddleOCR, EasyOCR, and Tesseract.
  • Expose ``warmup_ocr_readers()`` for Streamlit startup.
  • Expose ``run_ocr()`` — a four-pass OCR pipeline that returns a merged,
    deduplicated ``_OcrWord`` list covering all label zones.
  • ``_merge_word_lists()`` — confidence-weighted deduplication: when two
    engines return a word at the same position, keep the higher-confidence one.

Pass order (highest → lowest priority)
  0  PaddleOCR PP-OCRv5      — best-in-class for structured documents
  A  EasyOCR full-image       — fast, good coverage
  B  EasyOCR per-zone crops   — rescues small-text zones missed in pass A
  C  Tesseract ensemble merge  — catches characters EasyOCR misses
  PDF PDF-native text injection — confidence=1.0, highest priority merge
"""

from __future__ import annotations

import logging
import re
from typing import Callable, Optional

import numpy as np
from PIL import Image, ImageEnhance

from ._helpers import _OcrWord, _make_bbox, _polygon_to_bbox
from .pdf_passes import RawElement

logger = logging.getLogger(__name__)

# ── Optional dependency flags ─────────────────────────────────────────────────

try:
    import easyocr as _easyocr
    HAS_EASYOCR = True
except ImportError:
    HAS_EASYOCR = False

try:
    import os as _os_paddle
    _os_paddle.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    from paddleocr import PaddleOCR as _PaddleOCR
    HAS_PADDLE = True
except Exception:
    HAS_PADDLE = False

try:
    import pytesseract
    from pytesseract import Output as TessOutput
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False

try:
    import cv2 as _cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# ── Reader singletons ─────────────────────────────────────────────────────────

_easyocr_reader = None
_paddle_reader = None


def _get_easyocr_reader():
    global _easyocr_reader
    if _easyocr_reader is None and HAS_EASYOCR:
        logger.info("Initialising EasyOCR reader (English)…")
        _easyocr_reader = _easyocr.Reader(["en"], verbose=False)
    return _easyocr_reader


def _get_paddle_reader():
    global _paddle_reader
    if _paddle_reader is None and HAS_PADDLE:
        logger.info("Initialising PaddleOCR reader…")
        try:
            _paddle_reader = _PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
        except Exception as e:
            logger.warning("PaddleOCR init failed: %s", e)
    return _paddle_reader


def warmup_ocr_readers() -> dict:
    """
    Eagerly initialise all OCR readers and return a status dict.

    Intended to be called once at application startup (e.g. wrapped with
    ``@st.cache_resource`` in the Streamlit layer) so the heavy model weights
    are loaded into memory before the first upload arrives.
    """
    status = {"paddle": False, "easyocr": False}
    paddle = _get_paddle_reader()
    status["paddle"] = paddle is not None
    easyocr = _get_easyocr_reader()
    status["easyocr"] = easyocr is not None
    return status


# ── Image preprocessing ───────────────────────────────────────────────────────

def _ocr_image_variants(img: Image.Image) -> list[np.ndarray]:
    """
    Return up to 3 numpy arrays (RGB) for EasyOCR to try on a zone crop.

    Fewer variants than barcode preprocessing because EasyOCR has its own
    internal preprocessing — we only help when raw image quality is poor.
    """
    rgb = np.array(img.convert("RGB"))
    variants: list[np.ndarray] = [rgb]
    enhanced = np.array(ImageEnhance.Contrast(img).enhance(2.0).convert("RGB"))
    variants.append(enhanced)
    if HAS_CV2:
        gray = _cv2.cvtColor(rgb, _cv2.COLOR_RGB2GRAY)
        at = _cv2.adaptiveThreshold(
            gray, 255, _cv2.ADAPTIVE_THRESH_GAUSSIAN_C, _cv2.THRESH_BINARY, 11, 2
        )
        variants.append(_cv2.cvtColor(at, _cv2.COLOR_GRAY2RGB))
    return variants


# ── Confidence-weighted word deduplication ─────────────────────────────────────

def _merge_word_lists(
    primary: list[_OcrWord],
    secondary: list[_OcrWord],
    px_threshold: int = 15,
) -> list[_OcrWord]:
    """
    Merge two OCR word lists, keeping only one word per spatial position.

    Deduplication rule (confidence-weighted):
      • If the secondary word is at the same position as a primary word AND has
        *higher* confidence, the secondary word replaces the primary one.
      • This means PaddleOCR wins unless EasyOCR/Tesseract is more certain for
        a specific token — the best result from any engine survives.
      • Words from either list with no overlap partner are always kept.

    Parameters
    ----------
    primary      : preferred word list (higher-priority engine output)
    secondary    : words to merge in (only kept if not a duplicate)
    px_threshold : Manhattan-distance threshold in pixels for "same position"
    """
    result = list(primary)
    for sw in secondary:
        scx = sw.bbox.x + sw.bbox.w / 2
        scy = sw.bbox.y + sw.bbox.h / 2
        dup_idx: Optional[int] = None
        for i, pw in enumerate(result):
            if (abs((pw.bbox.x + pw.bbox.w / 2) - scx) < px_threshold
                    and abs((pw.bbox.y + pw.bbox.h / 2) - scy) < px_threshold):
                dup_idx = i
                break
        if dup_idx is None:
            result.append(sw)
        elif sw.conf > result[dup_idx].conf:
            # Replace the incumbent only when the challenger is more confident
            result[dup_idx] = sw
    return result


# ── Per-engine OCR passes ─────────────────────────────────────────────────────

def _run_paddle_ocr(
    img_arr: np.ndarray,
    img_w: int,
    img_h: int,
    zone_fn: Callable[[float, float], str],
    warnings: list[str],
) -> list[_OcrWord]:
    """Run PaddleOCR PP-OCRv5 and return an _OcrWord list."""
    paddle = _get_paddle_reader()
    if paddle is None:
        return []
    try:
        results = paddle.predict(img_arr)
        words: list[_OcrWord] = []
        for page in results:
            texts  = page.get("rec_texts",  [])
            scores = page.get("rec_scores", [])
            polys  = page.get("dt_polys",   [])
            for text, conf, poly in zip(texts, scores, polys):
                text = text.strip()
                if not text:
                    continue
                pts = [[int(p[0]), int(p[1])] for p in poly]
                bbox = _polygon_to_bbox(pts, img_w, img_h)
                cy_frac = (bbox.y + bbox.h / 2) / img_h if img_h else 0.5
                zone = zone_fn(bbox.center_x / img_w, cy_frac)
                bbox.zone = zone
                words.append(_OcrWord(text=text, conf=float(conf), bbox=bbox, zone=zone))
        logger.info("PaddleOCR Pass 0: %d words", len(words))
        return words
    except Exception as e:
        warnings.append(f"PaddleOCR: {e}")
        logger.exception("PaddleOCR failed")
        return []


def _run_easyocr_per_zone(
    image: Image.Image,
    img_w: int,
    img_h: int,
    zone_cfg: list,
    zone_fn: Callable[[float, float], str],
    existing_words: list[_OcrWord],
    warnings: list[str],
    is_portrait: bool = False,
) -> list[_OcrWord]:
    """
    Run EasyOCR on individual zone crops for zones with few words from Pass A.

    Zone crops give EasyOCR full focus on small-text areas — the date strip and
    compliance zone in particular tend to be under-served in a full-image pass.
    Multiple preprocessing variants are tried; the best yield is kept.
    """
    MIN_ZONE_WORDS = 3
    reader = _get_easyocr_reader()
    if reader is None:
        return []

    zone_counts: dict[str, int] = {}
    for w in existing_words:
        zone_counts[w.zone] = zone_counts.get(w.zone, 0) + 1

    all_new: list[_OcrWord] = []

    for zone_id, f0, f1, _rotate in zone_cfg:
        if zone_id.endswith("copy_count"):
            continue  # handled separately with rotation
        if zone_counts.get(zone_id, 0) >= MIN_ZONE_WORDS:
            continue

        if is_portrait:
            y0 = max(0, int(f0 * img_h))
            y1 = min(img_h, int(f1 * img_h))
            if y1 - y0 < 20:
                continue
            crop = image.crop((0, y0, img_w, y1))
            offset_x, offset_y = 0, y0
        else:
            x0 = max(0, int(f0 * img_w))
            x1 = min(img_w, int(f1 * img_w))
            if x1 - x0 < 20:
                continue
            crop = image.crop((x0, 0, x1, img_h))
            offset_x, offset_y = x0, 0

        best_results: list = []
        for arr in _ocr_image_variants(crop):
            try:
                results = reader.readtext(arr, detail=1, paragraph=False)
                if len(results) > len(best_results):
                    best_results = results
                if len(best_results) >= MIN_ZONE_WORDS * 2:
                    break
            except Exception as e:
                warnings.append(f"EasyOCR per-zone {zone_id}: {e}")

        zone_new: list[_OcrWord] = []
        for (polygon, text, conf) in best_results:
            text = text.strip()
            if not text:
                continue
            xs = [p[0] + offset_x for p in polygon]
            ys = [p[1] + offset_y for p in polygon]
            bx, by = int(min(xs)), int(min(ys))
            bw = max(1, int(max(xs) - bx))
            bh = max(1, int(max(ys) - by))
            bbox = _make_bbox(bx, by, bw, bh, img_w, img_h, zone_id)
            zone_new.append(_OcrWord(text=text, conf=float(conf), bbox=bbox, zone=zone_id))

        if zone_new:
            logger.debug("Per-zone EasyOCR %s: %d words", zone_id, len(zone_new))
        all_new.extend(zone_new)

    return all_new


def _ocr_copy_count_zone(
    image: Image.Image,
    img_w: int,
    img_h: int,
    zone_cfg: list,
    warnings: list[str],
) -> list[_OcrWord]:
    """Crop, rotate, and OCR the copy-count strip separately."""
    x0_frac, x1_frac = 0.95, 1.00
    for entry in zone_cfg:
        if entry[0].endswith("copy_count"):
            x0_frac, x1_frac = entry[1], entry[2]
            break

    x0 = int(x0_frac * img_w)
    x1 = min(img_w, int(x1_frac * img_w))
    if x1 <= x0:
        return []

    crop = image.crop((x0, 0, x1, img_h))
    rotated = crop.rotate(90, expand=True)

    reader = _get_easyocr_reader()
    words: list[_OcrWord] = []

    if reader is not None:
        for angle in (0, 180, 90, 270):
            img_try = rotated if angle == 0 else rotated.rotate(angle, expand=True)
            try:
                results = reader.readtext(np.array(img_try), detail=1, paragraph=False)
                for (polygon, text, conf) in results:
                    text = text.strip()
                    if text:
                        norm = re.sub(r"^[lI|]+$", "1", text)
                        orig_bbox = _make_bbox(x0, 0, x1 - x0, img_h, img_w, img_h, "right_copy_count")
                        words.append(_OcrWord(text=norm, conf=float(conf),
                                              bbox=orig_bbox, zone="right_copy_count"))
                if words:
                    break
            except Exception as e:
                warnings.append(f"EasyOCR copy-count angle={angle}: {e}")

    if not words and HAS_TESSERACT:
        for angle, psm in [(0, 8), (0, 10), (180, 8)]:
            img_try = rotated if angle == 0 else rotated.rotate(180, expand=True)
            g = img_try.convert("L")
            try:
                t = pytesseract.image_to_string(g, config=f"--oem 3 --psm {psm}").strip()
                if re.search(r"\d", t):
                    bb = _make_bbox(x0, 0, x1 - x0, img_h, img_w, img_h, "right_copy_count")
                    words.append(_OcrWord(text=t, conf=0.70, bbox=bb, zone="right_copy_count"))
                    break
            except Exception:
                pass

    logger.debug("Copy-count zone OCR: %r", [w.text for w in words])
    return words


def _run_tesseract_zones(
    image: Image.Image,
    img_w: int,
    img_h: int,
    zone_cfg: list,
    warnings: list[str],
    is_portrait: bool = False,
) -> list[_OcrWord]:
    """Tesseract per-zone pass with zone-specific page-segmentation modes."""
    if not HAS_TESSERACT:
        return []

    _ZONE_PSM = {
        "left_identity":    6,
        "top_identity":     6,
        "center_barcode":  11,
        "mid_barcode":     11,
        "center_datamatrix": 6,
        "center_address":   6,
        "mid_address":      6,
        "right_compliance": 11,
        "bot_compliance":  11,
        "right_date":       6,
        "bot_date":         6,
        "right_copy_count": 8,
        "circle_outer":     6,
        "circle_mid":       6,
        "circle_inner":     6,
    }
    words: list[_OcrWord] = []
    for zone_id, f0, f1, rotate_deg in zone_cfg:
        if is_portrait:
            y0 = max(0, int(f0 * img_h))
            y1 = min(img_h, int(f1 * img_h))
            crop = image.crop((0, y0, img_w, y1)).convert("L")
            x_offset, y_offset = 0, y0
        else:
            x0 = max(0, int(f0 * img_w))
            x1 = min(img_w, int(f1 * img_w))
            if x1 <= x0:
                continue
            crop = image.crop((x0, 0, x1, img_h)).convert("L")
            x_offset, y_offset = x0, 0

        if rotate_deg:
            crop = crop.rotate(rotate_deg, expand=True)
        psm = _ZONE_PSM.get(zone_id, 6)
        try:
            data = pytesseract.image_to_data(
                crop, config=f"--oem 3 --psm {psm}",
                output_type=TessOutput.DICT
            )
            for i in range(len(data["text"])):
                t = (data["text"][i] or "").strip()
                c = int(data["conf"][i])
                if not t or c < 10:
                    continue
                bx = data["left"][i] + x_offset
                by = data["top"][i] + y_offset
                bw = data["width"][i]
                bh = data["height"][i]
                bb = _make_bbox(bx, by, bw, bh, img_w, img_h, zone_id)
                words.append(_OcrWord(text=t, conf=c / 100.0, bbox=bb, zone=zone_id))
        except Exception as e:
            warnings.append(f"Tesseract zone {zone_id}: {e}")

    return words


# ── Main OCR entry point ──────────────────────────────────────────────────────

def run_ocr(
    image: Image.Image,
    zone_cfg: list,
    zone_fn: Callable[[float, float], str],
    raw_elements: list[RawElement],
    warnings: list[str],
    errors: list[str],
    is_portrait: bool = False,
) -> list[_OcrWord]:
    """
    Four-pass OCR strategy that maximises word coverage across all label zones.

    Pass 0  — PaddleOCR PP-OCRv5 on the full image.  Highest accuracy; used as
              primary when available.
    Pass A  — EasyOCR full-image scan.  Merged into PaddleOCR results or used
              as primary when Paddle is unavailable.
    Pass B  — EasyOCR per-zone crops for zones sparse after Pass A/0.
    Pass C  — Tesseract ensemble merge on top of the combined results.
    PDF     — PDF-native text (confidence=1.0) injected and OCR duplicates
              removed via deduplicate_text_elements().

    The copy-count strip is handled separately in all cases (requires 90° rotation).
    Returns a merged, confidence-weighted deduplicated word list.
    """
    from .pdf_passes import deduplicate_text_elements

    img_w, img_h = image.size
    words: list[_OcrWord] = []

    # ── Pass 0: PaddleOCR ─────────────────────────────────────────────────────
    paddle_words = _run_paddle_ocr(
        np.array(image), img_w, img_h, zone_fn, warnings
    )
    if paddle_words:
        words = paddle_words
        logger.info("Using PaddleOCR as primary OCR engine (%d words)", len(words))

    # ── Pass A: EasyOCR full-image ─────────────────────────────────────────────
    reader = _get_easyocr_reader()
    if reader is not None:
        # Exclude the copy-count strip; it needs rotation and is handled separately
        cc_x0_frac = next(
            (e[1] for e in zone_cfg if e[0].endswith("copy_count")), 0.95
        )
        cc_x0_px = int(cc_x0_frac * img_w)
        main_img = image.crop((0, 0, cc_x0_px, img_h))
        main_arr = np.array(main_img)

        try:
            results = reader.readtext(main_arr, detail=1, paragraph=False)

            if len(results) < 5:
                logger.debug(
                    "EasyOCR returned only %d results — retrying with enhanced image",
                    len(results),
                )
                for arr in _ocr_image_variants(main_img)[1:]:
                    results2 = reader.readtext(arr, detail=1, paragraph=False)
                    if len(results2) > len(results):
                        results = results2
                        warnings.append(
                            "EasyOCR raw scan returned few words; "
                            "used enhanced image instead."
                        )
                    if len(results) >= 5:
                        break

            easy_words: list[_OcrWord] = []
            for (polygon, text, conf) in results:
                text = text.strip()
                if not text:
                    continue
                bbox = _polygon_to_bbox(polygon, img_w, img_h)
                cy_frac = (bbox.y + bbox.h / 2) / img_h if img_h else 0.5
                zone = zone_fn(bbox.center_x / img_w, cy_frac)
                bbox.zone = zone
                easy_words.append(_OcrWord(text=text, conf=float(conf), bbox=bbox, zone=zone))

            if paddle_words:
                before = len(words)
                words = _merge_word_lists(words, easy_words)
                logger.info(
                    "EasyOCR Pass A: %d new/replaced words (total %d)",
                    len(words) - before + sum(1 for _ in easy_words), len(words),
                )
            else:
                words = easy_words
                logger.info("EasyOCR Pass A: %d words from full-image scan", len(words))

        except Exception as e:
            warnings.append(f"EasyOCR main body: {e}")
            logger.exception("EasyOCR failed on main body")

        # ── Pass B: per-zone EasyOCR ──────────────────────────────────────────
        zone_words = _run_easyocr_per_zone(
            image, img_w, img_h, zone_cfg, zone_fn, words, warnings, is_portrait
        )
        before = len(words)
        words = _merge_word_lists(words, zone_words)
        logger.info(
            "EasyOCR Pass B: %d new/replaced words from per-zone crops (total %d)",
            len(words) - before + len(zone_words), len(words),
        )

        # Copy-count strip (rotation required)
        words += _ocr_copy_count_zone(image, img_w, img_h, zone_cfg, warnings)

    elif HAS_TESSERACT:
        warnings.append("EasyOCR unavailable — falling back to Tesseract.")
        words = _run_tesseract_zones(image, img_w, img_h, zone_cfg, warnings, is_portrait)
    else:
        errors.append("No OCR engine available (install easyocr or pytesseract).")

    # ── Pass C: Tesseract ensemble merge ──────────────────────────────────────
    if HAS_TESSERACT and reader is not None:
        try:
            tess_words = _run_tesseract_zones(
                image, img_w, img_h, zone_cfg, warnings, is_portrait
            )
            before = len(words)
            words = _merge_word_lists(words, tess_words, px_threshold=20)
            logger.info(
                "Tesseract Pass C: %d new/replaced words merged (total %d)",
                len(words) - before + len(tess_words), len(words),
            )
        except Exception as e:
            warnings.append(f"Tesseract ensemble pass: {e}")

    # ── PDF Pass 1 merge: inject exact-position PyMuPDF text ──────────────────
    # PDF text elements have conf=1.0 and come from the vector data — they beat
    # any OCR result.  Inject them and let deduplication remove near-duplicate
    # OCR words, keeping the PDF version (which has pixel-perfect bboxes).
    pdf_text_elements = [e for e in raw_elements if e.type == "text_pdf"]
    if pdf_text_elements:
        pdf_words: list[_OcrWord] = [
            _OcrWord(
                text=el.content,
                conf=1.0,
                bbox=_make_bbox(el.x, el.y, el.w, el.h, img_w, img_h, el.zone),
                zone=el.zone,
            )
            for el in pdf_text_elements
            if el.content
        ]
        pdf_as_raw = [
            RawElement(
                type="text_pdf", content=w.text,
                x=w.bbox.x, y=w.bbox.y, w=w.bbox.w, h=w.bbox.h,
                confidence=1.0, zone=w.zone,
            )
            for w in pdf_words
        ]
        ocr_as_raw = [
            RawElement(
                type="text_ocr", content=w.text,
                x=w.bbox.x, y=w.bbox.y, w=w.bbox.w, h=w.bbox.h,
                confidence=w.conf, zone=w.zone,
            )
            for w in words
        ]
        merged_raw = deduplicate_text_elements(pdf_as_raw, ocr_as_raw, px_threshold=20)
        ocr_only = [e for e in merged_raw if e.type == "text_ocr"]

        ocr_only_words = [
            _OcrWord(
                text=e.content,
                conf=e.confidence,
                bbox=_make_bbox(e.x, e.y, e.w, e.h, img_w, img_h, e.zone),
                zone=e.zone,
            )
            for e in ocr_only
            if e.content
        ]
        words = pdf_words + ocr_only_words
        logger.info(
            "Pass 1 PDF merge: %d exact PDF words + %d OCR-only words = %d total",
            len(pdf_words), len(ocr_only_words), len(words),
        )

    return words
