"""
Shared upload handling: same temp-file materialisation and label image loading
as `LabelExtractor`, so UIs and human correction use the same pixel grid as OCR.
"""

from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path
from typing import Optional

from PIL import Image

from extractor import LabelExtractor


def sniff_suffix(data: bytes) -> Optional[str]:
    if data[:4] == b"%PDF":
        return ".pdf"
    if data[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if data[:4] in (b"II*\x00", b"MM\x00*"):
        return ".tif"
    if data[:2] == b"BM":
        return ".bmp"
    return None


def materialise(data: bytes) -> str:
    if not data:
        raise ValueError("Uploaded file is empty.")
    sniff = sniff_suffix(data)
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


def load_image_matching_extractor(data: bytes, dpi: int) -> Image.Image:
    """Same preprocessing as the extraction pipeline (up/downscale for images)."""
    p = materialise(data)
    try:
        return LabelExtractor(p, dpi=dpi).load()
    finally:
        if os.path.isfile(p):
            try:
                os.unlink(p)
            except OSError:
                pass
