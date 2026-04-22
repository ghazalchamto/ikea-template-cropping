"""
Multi-pass label dissection  —  open-source pipeline.

Inspired by the label-dissection-guide.md approach.

Pass 1  — PyMuPDF span-level text: exact positions, font, size, bold/italic.
          For programmatically-generated PDFs this gives pixel-perfect bboxes
          and completely bypasses the need for OCR on text elements.

Pass 2  — PyMuPDF embedded images: raster logos, icons, barcode images stored
          inside the PDF as image objects (xref).

Pass 4  — OpenCV visual regions: contour-based layout zone detection + Hough
          line detection for dividers and borders on the rasterised image.
          (Pass 3 = barcodes is handled by the main extractor's pyzbar/dmtx.)

All passes produce RawElement objects.  These are merged and stored on the
ExtractedLabel so downstream validators and the UI have the full picture.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Optional dependencies ─────────────────────────────────────────────────────

try:
    import fitz as _fitz          # PyMuPDF
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False

try:
    import cv2 as _cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# ── RawElement ────────────────────────────────────────────────────────────────

@dataclass
class RawElement:
    """
    A single extracted primitive from any dissection pass.

    All pixel coordinates are in the *rendered image* space so they can be
    drawn directly on top of the preview without further scaling.

    Downstream use
    ──────────────
    - ``type`` drives the annotation colour and icon in the UI.
    - ``content`` holds the decoded text for text elements.
    - ``confidence`` is used to rank competing extractions.
    - ``meta`` carries pass-specific extras (font color, image bytes, line
      orientation, region colour, …).
    """

    type: str           # "text_pdf" | "text_ocr" | "image_pdf" | "region" | "line"
    content: Optional[str]
    x: int
    y: int
    w: int
    h: int
    confidence: float = 1.0
    font: Optional[str] = None
    font_size: float = 0.0
    bold: bool = False
    italic: bool = False
    zone: str = ""
    meta: dict = field(default_factory=dict)

    # ── derived properties ────────────────────────────────────────────────────

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2

    @property
    def x1(self) -> int:
        return self.x + self.w

    @property
    def y1(self) -> int:
        return self.y + self.h

    def to_dict(self) -> dict:
        d = {
            "type":       self.type,
            "content":    self.content,
            "x":          self.x,
            "y":          self.y,
            "w":          self.w,
            "h":          self.h,
            "confidence": round(self.confidence, 3),
            "font":       self.font,
            "font_size":  self.font_size,
            "bold":       self.bold,
            "italic":     self.italic,
            "zone":       self.zone,
        }
        # Exclude raw image bytes from the JSON output — too large
        meta_clean = {k: v for k, v in self.meta.items() if k != "image_bytes"}
        if meta_clean:
            d["meta"] = meta_clean
        return d


# ── Pass 1: PDF text spans ────────────────────────────────────────────────────

def run_pass1_text(pdf_path: str, dpi: int = 600) -> list[RawElement]:
    """
    Extract all text spans from the first page of a PDF with exact positions.

    Uses PyMuPDF's ``rawdict`` text extraction, which gives:
    - Sub-word bounding boxes (span level)
    - Font name and size in points
    - Bold / italic flags

    PDF point coordinates are scaled to pixel space: px = pt × (dpi / 72).
    These coordinates match the rendered image produced by pdf2image at the
    same DPI, so they can be used directly to draw annotation boxes.
    """
    if not HAS_FITZ:
        logger.warning(
            "PyMuPDF not installed — Pass 1 (exact PDF text) skipped. "
            "Run: pip install pymupdf"
        )
        return []

    scale = dpi / 72.0
    elements: list[RawElement] = []

    try:
        doc = _fitz.open(pdf_path)
        page = doc[0]
        blocks = page.get_text(
            "rawdict",
            flags=_fitz.TEXT_PRESERVE_WHITESPACE,
        )["blocks"]

        for block in blocks:
            if block.get("type") != 0:   # 0 = text block
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if not text:
                        continue
                    x0, y0, x1, y1 = span.get("bbox", (0, 0, 0, 0))
                    flags = span.get("flags", 0)
                    elements.append(RawElement(
                        type="text_pdf",
                        content=text,
                        x=int(x0 * scale),
                        y=int(y0 * scale),
                        w=max(1, int((x1 - x0) * scale)),
                        h=max(1, int((y1 - y0) * scale)),
                        confidence=1.0,
                        font=span.get("font", ""),
                        font_size=round(span.get("size", 0.0), 2),
                        bold=bool(flags & (1 << 4)),
                        italic=bool(flags & (1 << 1)),
                        meta={"color": span.get("color", 0)},
                    ))
        doc.close()

    except Exception as exc:
        logger.error("Pass 1 (PyMuPDF text): %s", exc)

    logger.info("Pass 1: %d text spans extracted from PDF", len(elements))
    return elements


# ── Pass 2: Embedded images ───────────────────────────────────────────────────

def run_pass2_images(pdf_path: str, dpi: int = 600) -> list[RawElement]:
    """
    Extract raster images embedded in the PDF (logos, icons, etc.).

    Returns RawElements of type ``image_pdf``.
    The raw image bytes are stored in ``meta["image_bytes"]`` and
    ``meta["ext"]`` (e.g. "png", "jpeg").

    Note: vector graphics (drawn shapes, filled areas) are NOT image objects
    in the PDF — they are captured as layout regions in Pass 4.
    """
    if not HAS_FITZ:
        return []

    scale = dpi / 72.0
    elements: list[RawElement] = []

    try:
        doc = _fitz.open(pdf_path)
        page = doc[0]

        for img_info in page.get_images(full=True):
            xref = img_info[0]
            try:
                rects = page.get_image_rects(xref)
                if not rects:
                    continue
                bbox = rects[0]
                base_img = doc.extract_image(xref)
                elements.append(RawElement(
                    type="image_pdf",
                    content=None,
                    x=int(bbox.x0 * scale),
                    y=int(bbox.y0 * scale),
                    w=max(1, int((bbox.x1 - bbox.x0) * scale)),
                    h=max(1, int((bbox.y1 - bbox.y0) * scale)),
                    confidence=1.0,
                    meta={
                        "xref":       xref,
                        "ext":        base_img.get("ext", "png"),
                        "image_bytes": base_img.get("image", b""),
                        "width_px":   base_img.get("width", 0),
                        "height_px":  base_img.get("height", 0),
                        "colorspace": base_img.get("colorspace", 0),
                    },
                ))
            except Exception as exc:
                logger.debug("Pass 2: xref %d failed: %s", xref, exc)

        doc.close()

    except Exception as exc:
        logger.error("Pass 2 (PyMuPDF images): %s", exc)

    logger.info("Pass 2: %d embedded images extracted from PDF", len(elements))
    return elements


# ── Pass 4: Visual region detection ──────────────────────────────────────────

def run_pass4_visual_regions(img_array: np.ndarray) -> list[RawElement]:
    """
    Pass 4: OpenCV contour + Hough line detection on the rasterised image.

    ``img_array`` must be an RGB uint8 numpy array of shape (H, W, 3).

    Returns:
    - ``region`` elements  — rectangular layout zones / color blocks.
    - ``line`` elements    — detected horizontal / vertical dividers.

    These are used to:
    1. Verify zone boundaries (do the detected borders agree with the zone
       fractions calibrated from barcode positions?).
    2. Detect compliance symbol areas as regions (not just OCR text).
    3. Identify the IKEA logo band as a distinct colored block.
    4. Store layout structure for downstream visual validators.
    """
    if not HAS_CV2:
        logger.debug("opencv-python not installed — Pass 4 skipped.")
        return []

    elements: list[RawElement] = []
    h_img, w_img = img_array.shape[:2]
    page_area = h_img * w_img

    try:
        img_bgr = _cv2.cvtColor(img_array, _cv2.COLOR_RGB2BGR)
        gray    = _cv2.cvtColor(img_bgr,   _cv2.COLOR_BGR2GRAY)
        blurred = _cv2.GaussianBlur(gray, (5, 5), 0)
        edges   = _cv2.Canny(blurred, 50, 150)

        # ── Contour-based rectangular regions ─────────────────────────────────
        contours, _ = _cv2.findContours(
            edges, _cv2.RETR_EXTERNAL, _cv2.CHAIN_APPROX_SIMPLE,
        )
        seen: set[tuple] = set()

        for cnt in contours:
            x, y, w, h = _cv2.boundingRect(cnt)
            area = w * h

            # Skip noise (< ~50px × ~40px) and near-full-page containers
            if area < 2000 or area > page_area * 0.85:
                continue

            # Deduplicate rectangles that are nearly identical (within 10px grid)
            key = (x // 10, y // 10, w // 10, h // 10)
            if key in seen:
                continue
            seen.add(key)

            roi       = img_bgr[y:y+h, x:x+w]
            avg_color = roi.mean(axis=(0, 1)).tolist()   # BGR mean

            # Classify region type by dominant colour
            b, g, r = avg_color
            if r > 180 and g < 80 and b < 80:
                region_label = "red_zone"
            elif b > 150 and r < 80 and g < 120:
                region_label = "blue_zone"
            elif r > 200 and g > 200 and b < 80:
                region_label = "yellow_zone"
            elif r > 230 and g > 230 and b > 230:
                region_label = "white_zone"
            elif r < 60 and g < 60 and b < 60:
                region_label = "black_zone"
            else:
                region_label = "neutral_zone"

            elements.append(RawElement(
                type="region",
                content=region_label,
                x=x, y=y, w=w, h=h,
                confidence=0.75,
                meta={"avg_color_bgr": [round(c, 1) for c in avg_color]},
            ))

        # ── Hough line detection (horizontal/vertical dividers) ───────────────
        lines_raw = _cv2.HoughLinesP(
            edges, 1, np.pi / 180,
            threshold=100, minLineLength=100, maxLineGap=20,
        )
        if lines_raw is not None:
            for ln in lines_raw:
                x1, y1, x2, y2 = ln[0]
                dx, dy = abs(x2 - x1), abs(y2 - y1)
                # Only accept lines that are clearly horizontal or vertical
                if dx < 20 and dy < 20:
                    continue
                orientation = "horizontal" if dx > dy else "vertical"
                bx, by = min(x1, x2), min(y1, y2)
                elements.append(RawElement(
                    type="line",
                    content=orientation,
                    x=bx, y=by,
                    w=max(1, dx), h=max(1, dy),
                    confidence=0.70,
                    meta={
                        "x1": x1, "y1": y1,
                        "x2": x2, "y2": y2,
                        "orientation": orientation,
                    },
                ))

    except Exception as exc:
        logger.error("Pass 4 (OpenCV regions): %s", exc)

    n_regions = sum(1 for e in elements if e.type == "region")
    n_lines   = sum(1 for e in elements if e.type == "line")
    logger.info("Pass 4: %d regions, %d lines detected", n_regions, n_lines)
    return elements


# ── Merge ─────────────────────────────────────────────────────────────────────

def merge_raw_elements(*element_lists: list[RawElement]) -> list[RawElement]:
    """
    Merge multiple pass outputs into one list sorted in reading order
    (top-to-bottom, left-to-right, grouped into ~10-pixel row buckets).
    """
    merged: list[RawElement] = []
    for lst in element_lists:
        merged.extend(lst)

    merged.sort(key=lambda e: ((e.y // 10) * 10, e.x))
    return merged


def deduplicate_text_elements(
    pdf_elements: list[RawElement],
    ocr_elements: list[RawElement],
    px_threshold: int = 20,
) -> list[RawElement]:
    """
    Merge OCR words into a list of PDF text elements, dropping OCR words that
    are within ``px_threshold`` pixels (centre-to-centre) of a PDF element.

    PDF text takes precedence — it has confidence=1.0 and exact positions.
    OCR words are kept only where the PDF extraction found nothing.
    """
    result = list(pdf_elements)
    for ow in ocr_elements:
        ocx, ocy = ow.cx, ow.cy
        is_dup = any(
            abs(pe.cx - ocx) < px_threshold and abs(pe.cy - ocy) < px_threshold
            for pe in result
        )
        if not is_dup:
            result.append(ow)
    return result
