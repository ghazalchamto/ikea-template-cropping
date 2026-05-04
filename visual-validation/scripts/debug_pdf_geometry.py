#!/usr/bin/env python3
"""
scripts/debug_pdf_geometry.py
=============================
Print PDF page geometry for the candidate and template PDFs side by side,
and warn about anything that would cause region extraction to misalign:

* MediaBox / CropBox / TrimBox / BleedBox values
* /Rotate
* page.rect (post-rotation)
* aspect ratio vs. canonical region config
* implied scale_x / scale_y
* candidate vs template mismatch

This script does NOT modify any PDFs and does NOT write images. It is purely
diagnostic. Use it before/after a change to ``pdf_region_extractor.py``
to verify the coordinate model is correct.

Usage
-----
    PYTHONPATH=. python scripts/debug_pdf_geometry.py "data/labels/label.pdf" \
        --product L10555-MIADM \
        --ground-truth-dir data/templates/ground-truth-ikea-labels/ground_truth
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF

THIS = Path(__file__).resolve()
REPO = THIS.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.core.region_loader import (
    RegionConfig, RegionConfigError, default_regions_dir, load_region_config,
)

# Reuse template lookup from extract_regions.py.
_spec = importlib.util.spec_from_file_location(
    "extract_regions", THIS.parent / "extract_regions.py")
_extract_regions_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_extract_regions_mod)
find_template_pdf = _extract_regions_mod.find_template_pdf


# ─────────────────────────────────────────────────────────────────────────────
# Geometry inspection
# ─────────────────────────────────────────────────────────────────────────────

def _box_to_tuple(box) -> Optional[Tuple[float, float, float, float]]:
    """Convert a fitz.Rect / sequence to (x0, y0, x1, y1) — or None if missing."""
    if box is None:
        return None
    try:
        return (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
    except Exception:
        return None


def inspect_pdf(path: Path, page_index: int = 0) -> Dict[str, Any]:
    """Open *path* and gather every geometry value we care about for *page_index*."""
    doc = fitz.open(str(path))
    try:
        if page_index < 0 or page_index >= doc.page_count:
            raise ValueError(
                f"{path.name}: page_index={page_index} out of range "
                f"(page_count={doc.page_count})"
            )

        page      = doc[page_index]
        rect      = page.rect                # post-rotation visible rect
        mediabox  = getattr(page, "mediabox",  None)
        cropbox   = getattr(page, "cropbox",   None)
        trimbox   = getattr(page, "trimbox",   None)
        bleedbox  = getattr(page, "bleedbox",  None)
        rotation  = page.rotation

        info: Dict[str, Any] = {
            "path":        path,
            "page_count":  doc.page_count,
            "page_index":  page_index,
            "page_rect":   (rect.width, rect.height),       # post-rotation (W, H)
            "mediabox":    _box_to_tuple(mediabox),
            "cropbox":     _box_to_tuple(cropbox),
            "trimbox":     _box_to_tuple(trimbox),
            "bleedbox":    _box_to_tuple(bleedbox),
            "rotation":    int(rotation),
        }
        info["aspect_ratio"] = (rect.width / rect.height) if rect.height else float("nan")
        info["orientation"]  = (
            "landscape" if rect.width > rect.height
            else "portrait" if rect.width < rect.height
            else "square"
        )
        return info
    finally:
        doc.close()


# ─────────────────────────────────────────────────────────────────────────────
# Pretty printing
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_box(box: Optional[Tuple[float, float, float, float]]) -> str:
    if box is None:
        return "—"
    x0, y0, x1, y1 = box
    return f"[{x0:.2f}, {y0:.2f}, {x1:.2f}, {y1:.2f}]"


def print_pdf_block(label: str, info: Dict[str, Any]) -> None:
    Wp, Hp = info["page_rect"]
    print(f"  {label}")
    print(f"    file:          {info['path']}")
    print(f"    page count:    {info['page_count']}   (using page {info['page_index']})")
    print(f"    page rect:     {Wp:.2f} × {Hp:.2f} pt   "
          f"({Wp / 72:.2f}\" × {Hp / 72:.2f}\")")
    print(f"    aspect ratio:  {info['aspect_ratio']:.4f}  ({info['orientation']})")
    print(f"    rotation:      {info['rotation']}°")
    print(f"    MediaBox:      {_fmt_box(info['mediabox'])}")
    print(f"    CropBox:       {_fmt_box(info['cropbox'])}")
    print(f"    TrimBox:       {_fmt_box(info['trimbox'])}")
    print(f"    BleedBox:      {_fmt_box(info['bleedbox'])}")
    print()


def diff_warnings(
    template:  Dict[str, Any],
    candidate: Dict[str, Any],
    config:    RegionConfig,
) -> List[str]:
    warnings: List[str] = []

    Wc, Hc = config.page_size
    cfg_aspect = Wc / Hc

    # 1) Per-PDF: aspect ratio vs config
    for label, info in (("template", template), ("candidate", candidate)):
        Wp, Hp = info["page_rect"]
        rel = abs(info["aspect_ratio"] - cfg_aspect) / cfg_aspect
        if rel > 0.05:
            warnings.append(
                f"{label} aspect ratio {info['aspect_ratio']:.4f} differs from "
                f"config {cfg_aspect:.4f} by {rel * 100:.1f}% — "
                f"the canonical layout assumed by the config does not match the PDF "
                f"orientation. Possible cause: rotation, swapped width/height, or "
                f"a stub PDF that crops the live area."
            )

    # 2) Rotation
    for label, info in (("template", template), ("candidate", candidate)):
        if info["rotation"] != 0:
            warnings.append(
                f"{label} has /Rotate = {info['rotation']}° — extraction must "
                f"work in post-rotation coordinates (page.rect already swaps W/H)."
            )

    # 3) Cropbox != Mediabox
    for label, info in (("template", template), ("candidate", candidate)):
        mb = info["mediabox"]; cb = info["cropbox"]
        if mb is not None and cb is not None and mb != cb:
            warnings.append(
                f"{label} CropBox ≠ MediaBox — visible area is a sub-rect of the "
                f"PDF page. Use page.rect (the visible/cropped rect), not raw MediaBox."
            )

    # 4) Template vs candidate
    Wt, Ht = template["page_rect"]
    Wc2, Hc2 = candidate["page_rect"]
    rel_w = abs(Wt - Wc2) / max(Wt, Wc2) if max(Wt, Wc2) else 0.0
    rel_h = abs(Ht - Hc2) / max(Ht, Hc2) if max(Ht, Hc2) else 0.0
    if rel_w > 0.01 or rel_h > 0.01:
        warnings.append(
            f"template/candidate page rects differ "
            f"({Wt:.2f}×{Ht:.2f} vs {Wc2:.2f}×{Hc2:.2f}) — uniform scaling "
            f"means each PDF computes its own scale; that's expected. But if "
            f"the difference is large, check whether one PDF includes bleed/trim margins."
        )
    if template["rotation"] != candidate["rotation"]:
        warnings.append(
            f"template rotation {template['rotation']}° differs from candidate "
            f"rotation {candidate['rotation']}°. Region extractor must normalize "
            f"to the post-rotation coordinate system on both sides."
        )

    return warnings


def print_implied_mapping(
    template:  Dict[str, Any],
    candidate: Dict[str, Any],
    config:    RegionConfig,
) -> None:
    """Show the scale_x / scale_y the extractor would use under different strategies."""
    Wc, Hc = config.page_size

    print("  Implied config → PDF point mapping")
    print(f"    canonical config size:   {Wc} × {Hc} px   "
          f"(aspect {Wc / Hc:.4f})")
    print()
    print(f"    {'PDF':<11} {'page (pt)':<22} {'fit-min scale':<18} "
          f"{'fit-stretch sx,sy':<22}")
    print("    " + "─" * 76)

    for label, info in (("template", template), ("candidate", candidate)):
        Wp, Hp = info["page_rect"]
        s_min = min(Wp / Wc, Hp / Hc)
        sx = Wp / Wc
        sy = Hp / Hc
        print(f"    {label:<11} {Wp:>8.2f} × {Hp:>7.2f}   "
              f"{s_min:>14.5f}    "
              f"sx={sx:.4f}  sy={sy:.4f}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inspect candidate + template PDF page geometry and "
                    "report any mismatches that would cause region extraction "
                    "to misalign.",
    )
    p.add_argument("candidate_pdf", help="Path to the candidate label PDF.")
    p.add_argument("--product", "-p", required=True,
                   help="Product code, e.g. L10555-MIADM.")
    p.add_argument("--ground-truth-dir",
                   default="data/templates/ground-truth-ikea-labels/ground_truth",
                   help="Directory containing approved template PDFs (read-only).")
    p.add_argument("--regions-dir", default=None,
                   help="Directory containing region configs. "
                        "Default: <repo>/config/regions")
    p.add_argument("--page", type=int, default=0,
                   help="Page index to inspect (default: 0).")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-7s  %(name)s — %(message)s",
        stream=sys.stdout,
    )

    candidate_pdf    = Path(args.candidate_pdf).resolve()
    ground_truth_dir = Path(args.ground_truth_dir).resolve()
    regions_dir      = Path(args.regions_dir).resolve() if args.regions_dir else default_regions_dir()

    if not candidate_pdf.exists():
        print(f"ERROR: candidate PDF not found: {candidate_pdf}", file=sys.stderr)
        return 2

    try:
        config = load_region_config(args.product, regions_dir)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr); return 2
    except RegionConfigError as exc:
        print(f"ERROR: invalid region config — {exc}", file=sys.stderr); return 2

    try:
        template_pdf = find_template_pdf(ground_truth_dir, args.product)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr); return 2

    print()
    print(f"  Product code:   {args.product}")
    print(f"  Region config:  {config.source_path}")
    print(f"  Page size cfg:  {config.page_width} × {config.page_height} px")
    print()

    template_info  = inspect_pdf(template_pdf,  args.page)
    candidate_info = inspect_pdf(candidate_pdf, args.page)

    print_pdf_block("TEMPLATE",  template_info)
    print_pdf_block("CANDIDATE", candidate_info)

    print_implied_mapping(template_info, candidate_info, config)

    warnings = diff_warnings(template_info, candidate_info, config)
    if warnings:
        print("  WARNINGS")
        print("  " + "─" * 76)
        for i, w in enumerate(warnings, 1):
            print(f"  [{i}] {w}")
            print()
    else:
        print("  No geometry warnings detected.")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
