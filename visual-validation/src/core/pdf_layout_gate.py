"""
Pre-flight checks for layout-only PDF validation (page alignment, geometry).

Deterministic; no ML. Used by ``pipeline_runner`` before region extraction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

try:
    import fitz  # PyMuPDF
except ImportError as exc:  # pragma: no cover
    raise ImportError("PyMuPDF is required for pdf_layout_gate") from exc

from src.core.pdf_region_extractor import PDFExtraction


def validate_pdf_pair_page_index(
    template_pdf: Path,
    candidate_pdf: Path,
    *,
    page_index: int,
) -> None:
    """
    Ensure both PDFs have the same page count and *page_index* is valid.

    Raises
    ------
    ValueError
        With a message suitable for wrapping as ``RegionPipelineError``.
    """
    tp = Path(template_pdf)
    cp = Path(candidate_pdf)
    doc_t = fitz.open(str(tp))
    doc_c = fitz.open(str(cp))
    try:
        nt, nc = doc_t.page_count, doc_c.page_count
        if nt != nc:
            raise ValueError(
                f"Page count mismatch: template {tp.name!r} has {nt} page(s), "
                f"candidate {cp.name!r} has {nc} page(s). Layout validation "
                f"requires the same number of pages (compare page_index={page_index}).",
            )
        if page_index < 0 or page_index >= nt:
            raise ValueError(
                f"page_index={page_index} is out of range for both PDFs "
                f"(page_count={nt}).",
            )
    finally:
        doc_t.close()
        doc_c.close()


def assert_matching_page_dimensions_pt(
    template_extr: PDFExtraction,
    candidate_extr: PDFExtraction,
    *,
    max_pt_diff: float = 0.5,
) -> None:
    """
    Template and candidate must use the same page box in PDF points so that
    ``bbox_norm`` maps to the same physical frame on both sides.
    """
    wt, ht = template_extr.page_size_pdf
    wc, hc = candidate_extr.page_size_pdf
    if abs(wt - wc) > max_pt_diff or abs(ht - hc) > max_pt_diff:
        raise ValueError(
            f"PDF page size mismatch: template page is {wt:.2f}×{ht:.2f} pt, "
            f"candidate page is {wc:.2f}×{hc:.2f} pt (tolerance {max_pt_diff} pt). "
            f"Layout mode refuses to compare different page geometries.",
        )


def assert_label_aspect_consistent_with_pdf(
    template_extr: PDFExtraction,
    label_width_mm: float,
    label_height_mm: float,
    *,
    rel_tol: float = 0.06,
) -> None:
    """
    Ground-truth ``label_size_mm`` aspect ratio should match the template PDF
    page aspect (when label_rect is the full page). Catches wrong template.json
    vs PDF pairing.
    """
    if label_width_mm <= 0 or label_height_mm <= 0:
        raise ValueError("label_width_mm and label_height_mm must be positive")
    wt, ht = template_extr.page_size_pdf
    if wt <= 0 or ht <= 0:
        raise ValueError("template PDF page has non-positive dimensions")
    r_pdf = wt / ht
    r_mm = label_width_mm / label_height_mm
    if abs(r_pdf - r_mm) / max(r_mm, 1e-9) > rel_tol:
        raise ValueError(
            f"label_size_mm aspect ({label_width_mm:.3f} / {label_height_mm:.3f} "
            f"≈ {r_mm:.4f}) does not match template PDF page aspect "
            f"({wt:.2f} / {ht:.2f} ≈ {r_pdf:.4f}); relative tolerance {rel_tol:.2f}. "
            f"Fix template.json label_size_mm or the template PDF page size.",
        )
