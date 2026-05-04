"""
src/core/pdf_region_extractor.py
================================
Extract per-region clips from a PDF using PyMuPDF.

Each region is rendered on its own clip rectangle (never the full page).

Coordinate modes
----------------
1) **Legacy** — integer ``x, y, width, height`` in canonical pixels
   (``RegionConfig.page_size``). Mapped to the full **page.rect** with
   per-axis stretch: ``scale_x = Wp/Wc``, ``scale_y = Hp/Hc``.

2) **Label-normalised** — ``Region.bbox_norm = [nx, ny, nw, nh]`` with each
   value in ``[0, 1]``, relative to **label_rect** (not the full page unless
   they coincide). ``label_rect`` defaults to ``page.rect``; optional
   ``RegionConfig.label_rect_pts`` supplies a PDF-space rectangle
   ``[x0,y0,x1,y1]`` intersected with ``page.rect`` per document.

Output clips are always ``region.height × region.width`` BGR uint8.

Rotation / CropBox
------------------
``page.rect`` is post-rotation and post-CropBox — used as the page frame.
``label_rect`` is clamped to ``page.rect``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    import fitz   # PyMuPDF
except ImportError as exc:   # pragma: no cover
    raise ImportError(
        "PyMuPDF (fitz) is required for pdf_region_extractor. "
        "Install with: pip install pymupdf"
    ) from exc

from src.core.region_loader import Region, RegionConfig, bbox_norm_to_pdf_rect

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Result types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExtractedRegion:
    region:        Region
    image_bgr:     np.ndarray                # shape (height, width, 3), uint8
    pdf_bbox:      Tuple[float, float, float, float]   # (x0, y0, x1, y1) in PDF points
    fully_on_page: bool

    @property
    def name(self) -> str:
        return self.region.name


@dataclass
class PDFExtraction:
    pdf_path:        Path
    page_index:      int
    page_size_pdf:   Tuple[float, float]       # (Wp, Hp) in PDF points (post-rotation)
    page_rotation:   int                       # /Rotate value
    config_size:     Tuple[int, int]           # (Wc, Hc) from the config
    scale_x:         float                     # legacy: Wp/Wc; still stored for introspector
    scale_y:         float                     # legacy: Hp/Hc
    label_rect_pdf:  Tuple[float, float, float, float]   # (x0,y0,x1,y1) label frame in PDF pts
    extracted:       List[ExtractedRegion] = field(default_factory=list)

    def by_name(self) -> dict[str, ExtractedRegion]:
        return {e.name: e for e in self.extracted}


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def extract_regions(
    pdf_path:    str | Path,
    config:      RegionConfig,
    page_index:  int   = 0,
    render_dpi:  int   = 150,
    save_dir:    Optional[str | Path] = None,
) -> PDFExtraction:
    """
    Open *pdf_path*, clip every region in *config*, and return a PDFExtraction.

    ``render_dpi`` is ignored; kept for API compatibility.
    """
    del render_dpi

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    save_dir_path: Optional[Path] = Path(save_dir) if save_dir else None
    if save_dir_path is not None:
        save_dir_path.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(pdf_path))
    try:
        if page_index < 0 or page_index >= doc.page_count:
            raise ValueError(
                f"{pdf_path.name}: page_index={page_index} out of range "
                f"(page_count={doc.page_count})"
            )
        page      = doc[page_index]
        page_rect = page.rect
        Wp = float(page_rect.width)
        Hp = float(page_rect.height)
        Wc, Hc = config.page_size

        if Wp <= 0 or Hp <= 0:
            raise ValueError(
                f"{pdf_path.name}: page rect has non-positive dimension "
                f"({Wp} × {Hp}) — cannot extract regions."
            )

        label_rect = _resolve_label_rect(page_rect, config)
        Ltuple = (float(label_rect.x0), float(label_rect.y0),
                  float(label_rect.x1), float(label_rect.y1))

        scale_x = Wp / Wc
        scale_y = Hp / Hc

        logger.info(
            "Extract %s p%d: page_rect=%.2fx%.2f pt; rotation=%d°; "
            "label_rect=(%.2f,%.2f,%.2f,%.2f) pt; config %dx%d px; "
            "legacy scale_x=%.5f scale_y=%.5f pt/px",
            pdf_path.name, page_index, Wp, Hp, page.rotation,
            Ltuple[0], Ltuple[1], Ltuple[2], Ltuple[3],
            Wc, Hc, scale_x, scale_y,
        )

        extracted: List[ExtractedRegion] = []
        for region in config.regions:
            if region.bbox_norm is not None:
                er = _extract_one_label_norm(
                    page=page, region=region,
                    label_rect=label_rect, Wp=Wp, Hp=Hp,
                )
                nx, ny, nw, nh = region.bbox_norm
                rx0, ry0, rx1, ry1 = bbox_norm_to_pdf_rect(Ltuple, region.bbox_norm)
                logger.info(
                    "  region %-26s  bbox_norm=[%.4f,%.4f,%.4f,%.4f]  "
                    "resolved_pdf=(%.2f,%.2f,%.2f,%.2f)  out=%dx%d  on_page=%s",
                    region.name, nx, ny, nw, nh, rx0, ry0, rx1, ry1,
                    region.width, region.height, er.fully_on_page,
                )
            else:
                zoom_x = 1.0 / scale_x
                zoom_y = 1.0 / scale_y
                matrix = fitz.Matrix(zoom_x, zoom_y)
                er = _extract_one_legacy(
                    page=page, region=region,
                    scale_x=scale_x, scale_y=scale_y,
                    Wp=Wp, Hp=Hp, matrix=matrix,
                )
                logger.debug(
                    "  region %-30s  %dx%d  fully_on_page=%s  pdf_bbox=%s",
                    region.name, region.width, region.height,
                    er.fully_on_page, er.pdf_bbox,
                )

            extracted.append(er)
            if save_dir_path is not None:
                out = save_dir_path / f"{region.name}.png"
                cv2.imwrite(str(out), er.image_bgr)

        return PDFExtraction(
            pdf_path=pdf_path,
            page_index=page_index,
            page_size_pdf=(Wp, Hp),
            page_rotation=int(page.rotation),
            config_size=config.page_size,
            scale_x=scale_x,
            scale_y=scale_y,
            label_rect_pdf=Ltuple,
            extracted=extracted,
        )
    finally:
        doc.close()


# ─────────────────────────────────────────────────────────────────────────────
# Label rect
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_label_rect(page_rect: fitz.Rect, config: RegionConfig) -> fitz.Rect:
    """Intersect optional config label_rect_pts with the visible page rect."""
    if config.label_rect_pts is None:
        return page_rect
    lr = fitz.Rect(config.label_rect_pts)
    inter = lr & page_rect
    if inter.is_empty or inter.width <= 0 or inter.height <= 0:
        raise ValueError(
            f"label_rect_pts {config.label_rect_pts!r} does not overlap page.rect "
            f"({page_rect!r}) — check config."
        )
    return inter


# ─────────────────────────────────────────────────────────────────────────────
# Legacy extraction (canonical pixels → full page stretch)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_one_legacy(
    page,
    region:    Region,
    scale_x:   float,
    scale_y:   float,
    Wp:        float,
    Hp:        float,
    matrix,
) -> ExtractedRegion:
    x0_pdf = region.x  * scale_x
    y0_pdf = region.y  * scale_y
    x1_pdf = region.x2 * scale_x
    y1_pdf = region.y2 * scale_y

    cx0 = max(0.0, x0_pdf)
    cy0 = max(0.0, y0_pdf)
    cx1 = min(Wp,  x1_pdf)
    cy1 = min(Hp,  y1_pdf)
    fully_on_page = (
        cx0 == x0_pdf and cy0 == y0_pdf and cx1 == x1_pdf and cy1 == y1_pdf
        and cx1 > cx0 and cy1 > cy0
    )

    canvas = np.full((region.height, region.width, 3), 255, dtype=np.uint8)

    if cx1 <= cx0 or cy1 <= cy0:
        logger.warning(
            "Region %r is entirely outside PDF page rect; returning blank clip.",
            region.name,
        )
        return ExtractedRegion(
            region=region, image_bgr=canvas,
            pdf_bbox=(x0_pdf, y0_pdf, x1_pdf, y1_pdf),
            fully_on_page=False,
        )

    clip_rect = fitz.Rect(cx0, cy0, cx1, cy1)
    pix = page.get_pixmap(matrix=matrix, clip=clip_rect, alpha=False)

    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    px_per_pt_x = 1.0 / scale_x
    px_per_pt_y = 1.0 / scale_y
    dst_x = int(round((cx0 - x0_pdf) * px_per_pt_x))
    dst_y = int(round((cy0 - y0_pdf) * px_per_pt_y))

    src_h, src_w = img.shape[:2]
    avail_w = region.width  - dst_x
    avail_h = region.height - dst_y
    if avail_w <= 0 or avail_h <= 0:
        return ExtractedRegion(
            region=region, image_bgr=canvas,
            pdf_bbox=(x0_pdf, y0_pdf, x1_pdf, y1_pdf),
            fully_on_page=fully_on_page,
        )
    use_w = min(src_w, avail_w)
    use_h = min(src_h, avail_h)
    canvas[dst_y:dst_y + use_h, dst_x:dst_x + use_w] = img[:use_h, :use_w]

    return ExtractedRegion(
        region=region, image_bgr=canvas,
        pdf_bbox=(x0_pdf, y0_pdf, x1_pdf, y1_pdf),
        fully_on_page=fully_on_page,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Label-normalised extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_one_label_norm(
    page,
    region:      Region,
    label_rect:  fitz.Rect,
    Wp:          float,
    Hp:          float,
) -> ExtractedRegion:
    """Render a region defined by ``bbox_norm`` relative to *label_rect*."""
    assert region.bbox_norm is not None

    Lt = (float(label_rect.x0), float(label_rect.y0),
          float(label_rect.x1), float(label_rect.y1))
    x0_pdf, y0_pdf, x1_pdf, y1_pdf = bbox_norm_to_pdf_rect(Lt, region.bbox_norm)

    cx0 = max(0.0, x0_pdf)
    cy0 = max(0.0, y0_pdf)
    cx1 = min(Wp,  x1_pdf)
    cy1 = min(Hp,  y1_pdf)
    fully_on_page = (
        cx0 == x0_pdf and cy0 == y0_pdf and cx1 == x1_pdf and cy1 == y1_pdf
        and cx1 > cx0 and cy1 > cy0
    )

    canvas = np.full((region.height, region.width, 3), 255, dtype=np.uint8)

    if cx1 <= cx0 or cy1 <= cy0:
        logger.warning(
            "Region %r (bbox_norm) is entirely outside page; returning blank clip.",
            region.name,
        )
        return ExtractedRegion(
            region=region, image_bgr=canvas,
            pdf_bbox=(x0_pdf, y0_pdf, x1_pdf, y1_pdf),
            fully_on_page=False,
        )

    clip_w = cx1 - cx0
    clip_h = cy1 - cy0
    zoom_x = region.width  / clip_w
    zoom_y = region.height / clip_h
    matrix = fitz.Matrix(zoom_x, zoom_y)

    clip_rect = fitz.Rect(cx0, cy0, cx1, cy1)
    pix = page.get_pixmap(matrix=matrix, clip=clip_rect, alpha=False)

    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    full_w = x1_pdf - x0_pdf
    full_h = y1_pdf - y0_pdf
    if full_w <= 0 or full_h <= 0:
        return ExtractedRegion(
            region=region, image_bgr=canvas,
            pdf_bbox=(x0_pdf, y0_pdf, x1_pdf, y1_pdf),
            fully_on_page=False,
        )

    dst_x = int(round((cx0 - x0_pdf) * region.width  / full_w))
    dst_y = int(round((cy0 - y0_pdf) * region.height / full_h))

    src_h, src_w = img.shape[:2]
    avail_w = region.width  - dst_x
    avail_h = region.height - dst_y
    if avail_w <= 0 or avail_h <= 0:
        return ExtractedRegion(
            region=region, image_bgr=canvas,
            pdf_bbox=(x0_pdf, y0_pdf, x1_pdf, y1_pdf),
            fully_on_page=fully_on_page,
        )
    use_w = min(src_w, avail_w)
    use_h = min(src_h, avail_h)
    canvas[dst_y:dst_y + use_h, dst_x:dst_x + use_w] = img[:use_h, :use_w]

    return ExtractedRegion(
        region=region,
        image_bgr=canvas,
        pdf_bbox=(x0_pdf, y0_pdf, x1_pdf, y1_pdf),
        fully_on_page=fully_on_page,
    )
