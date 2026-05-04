"""
src/utils/ingestion.py
======================
Load PDFs and raster images into BGR numpy arrays.

PDF rendering uses PyMuPDF (fitz) which handles MuPDF-generated PDFs
perfectly — exactly what your ground_truth labels are (Written by MuPDF 1.27.2).
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def load_image(
    path: str | Path,
    dpi: int = 150,
    page: int = 0,
) -> np.ndarray:
    """
    Load any supported label file and return a BGR uint8 numpy array.

    Supported: .pdf  .png  .jpg  .jpeg  .tiff  .tif  .bmp  .webp

    Parameters
    ----------
    path : file to load
    dpi  : resolution for PDF rasterisation (150 = fast, 300 = precise)
    page : zero-based page index for PDFs (IKEA labels are single-page)

    Returns
    -------
    np.ndarray  shape (H, W, 3), dtype uint8, BGR colour order
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path, dpi=dpi, page=page)
    elif suffix in {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}:
        return _load_raster(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix!r}  (path={path})")


def _load_pdf(path: Path, dpi: int, page: int) -> np.ndarray:
    try:
        import fitz
    except ImportError as exc:
        raise ImportError(
            "PyMuPDF is required.  Install with:  pip install PyMuPDF"
        ) from exc

    doc = fitz.open(str(path))
    n_pages = len(doc)
    if page >= n_pages:
        raise ValueError(
            f"PDF has {n_pages} page(s); requested page {page} is out of range."
        )

    # MuPDF baseline is 72 DPI
    scale = dpi / 72.0
    mat = fitz.Matrix(scale, scale)
    pix = doc[page].get_pixmap(matrix=mat, alpha=False)
    doc.close()

    img_rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, 3
    )
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    logger.debug("PDF loaded: %s  page=%d  size=%dx%d  dpi=%d",
                 path.name, page, pix.width, pix.height, dpi)
    return bgr


def _load_raster(path: Path) -> np.ndarray:
    pil = Image.open(path)

    # Flatten transparency onto white background
    if pil.mode in ("RGBA", "LA", "P"):
        bg = Image.new("RGB", pil.size, (255, 255, 255))
        pil = pil.convert("RGBA")
        bg.paste(pil, mask=pil.split()[3])
        pil = bg
    elif pil.mode != "RGB":
        pil = pil.convert("RGB")

    arr = np.array(pil, dtype=np.uint8)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    logger.debug("Raster loaded: %s  size=%dx%d", path.name, pil.width, pil.height)
    return bgr
