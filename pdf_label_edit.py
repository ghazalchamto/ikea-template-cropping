"""
Redact+insert text on the first page of a PDF, using pixel bboxes that match
pdf2image / LabelExtractor at a given DPI.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    import fitz
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False


def _px_rect_to_pt(x: int, y: int, w: int, h: int, dpi: int) -> "fitz.Rect":
    s = dpi / 72.0
    return fitz.Rect(x / s, y / s, (x + w) / s, (y + h) / s)


def apply_text_edits_to_pdf_bytes(
    pdf_bytes: bytes,
    edits: list[dict[str, Any]],
    *,
    dpi: int,
    default_fontsize: float = 7.0,
) -> bytes:
    """
    Each edit: ``x, y, w, h`` in **full-resolution** label pixels, ``text`` str,
    optional ``font_size`` (pt).
    """
    if not HAS_FITZ:
        raise RuntimeError("PyMuPDF (pymupdf) is required for PDF text edits.")
    if not edits:
        return pdf_bytes

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        page = doc[0]

        for ed in edits:
            x, y, w, h = int(ed["x"]), int(ed["y"]), int(ed["w"]), int(ed["h"])
            if w <= 0 or h <= 0:
                continue
            r = _px_rect_to_pt(x, y, w, h, dpi)
            try:
                page.add_redact_annot(r, text="")
            except Exception as e:
                logger.warning("add_redact_annot failed: %s", e)

        page.apply_redactions(images=0)

        for ed in edits:
            text = (ed.get("text") or "").strip()
            if not text:
                continue
            x, y, w, h = int(ed["x"]), int(ed["y"]), int(ed["w"]), int(ed["h"])
            if w <= 0 or h <= 0:
                continue
            r = _px_rect_to_pt(x, y, w, h, dpi)
            fs = float(ed.get("font_size") or default_fontsize)
            try:
                page.insert_textbox(
                    r,
                    text,
                    fontname="helv",
                    fontsize=fs,
                    color=(0, 0, 0),
                    align=0,
                )
            except Exception as e:
                logger.warning("insert_textbox failed: %s", e)
                try:
                    page.insert_text(
                        (r.x0, r.y0 + fs * 0.8),
                        text,
                        fontname="helv",
                        fontsize=fs,
                        color=(0, 0, 0),
                    )
                except Exception as e2:
                    logger.warning("insert_text fallback failed: %s", e2)
        return doc.tobytes()
    finally:
        doc.close()
