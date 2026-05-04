#!/usr/bin/env python3
"""
scripts/extract_regions.py
==========================
Debug tool: load a region config, extract every region from both the
candidate label PDF and the matching base template PDF, and save the
clips to outputs/regions/<product_code>/ for visual inspection.

Usage
-----
    python scripts/extract_regions.py "data/labels/label.pdf" \
        --product L10555-MIADM

    python scripts/extract_regions.py candidate.pdf \
        --product L10555-MIADM \
        --ground-truth-dir data/templates/ground-truth-ikea-labels/ground_truth \
        --regions-dir      config/regions \
        --out-dir          outputs/regions \
        --dpi              150

Outputs
-------
    outputs/regions/<product_code>/
        candidate/<region_name>.png
        template/<region_name>.png
        side_by_side/<region_name>.png      (template | candidate)
        manifest.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

# Make `src` importable when the script is run from the repo root.
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.region_loader import (
    RegionConfig, RegionConfigError, default_regions_dir, load_region_config,
)
from src.core.pdf_region_extractor import (
    ExtractedRegion, PDFExtraction, extract_regions,
)


# ─────────────────────────────────────────────────────────────────────────────
# Template lookup (read-only against ground_truth)
# ─────────────────────────────────────────────────────────────────────────────

def find_template_pdf(
    ground_truth_dir: Path,
    product_code:     str,
) -> Path:
    """
    Locate the approved template PDF for *product_code* under ground_truth/.

    Strategy (in order):
      1. <gt>/<product_code>/*.pdf  — first PDF found
      2. <gt>/<product_code>.pdf
      3. recursive: any *.pdf whose path contains product_code

    Never modifies anything inside ground_truth.
    """
    if not ground_truth_dir.exists():
        raise FileNotFoundError(f"ground_truth_dir does not exist: {ground_truth_dir}")

    # 1. directory named after product code
    prod_dir = ground_truth_dir / product_code
    if prod_dir.is_dir():
        pdfs = sorted(prod_dir.glob("*.pdf"))
        if pdfs:
            return pdfs[0]

    # 2. flat file
    flat = ground_truth_dir / f"{product_code}.pdf"
    if flat.is_file():
        return flat

    # 3. recursive search
    for p in sorted(ground_truth_dir.rglob("*.pdf")):
        if product_code in p.parts or product_code in p.stem:
            return p

    raise FileNotFoundError(
        f"No template PDF found for product_code={product_code!r} under {ground_truth_dir}. "
        f"Looked for: {prod_dir}/*.pdf, {flat}, and recursive matches."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_side_by_side(
    template_clip:  np.ndarray,
    candidate_clip: np.ndarray,
    out_path:       Path,
    region_name:    str,
) -> None:
    """Concatenate template | candidate horizontally with a header strip."""
    # Both clips are guaranteed equal-sized by the extractor.
    h, w = template_clip.shape[:2]
    header_h = 28
    header = np.full((header_h, w * 2, 3), 30, dtype=np.uint8)
    cv2.putText(
        header, f"TEMPLATE  |  CANDIDATE   —   {region_name}",
        (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
    )
    body = np.hstack([template_clip, candidate_clip])
    out  = np.vstack([header, body])
    cv2.imwrite(str(out_path), out)


def write_manifest(
    out_dir:           Path,
    product_code:      str,
    config:            RegionConfig,
    candidate_pdf:     Path,
    template_pdf:      Path,
    candidate_extract: PDFExtraction,
    template_extract:  PDFExtraction,
    dpi:               int,
) -> Path:
    def _region_entry(c_er: ExtractedRegion, t_er: ExtractedRegion) -> dict:
        r = c_er.region
        return {
            "name":          r.name,
            "type":          r.type,
            "mandatory":     r.mandatory,
            "weight":        r.weight,
            "bbox_norm":     list(r.bbox_norm) if r.bbox_norm is not None else None,
            "config_bbox":   {"x": r.x, "y": r.y, "width": r.width, "height": r.height},
            "candidate": {
                "image_path":    f"candidate/{r.name}.png",
                "pdf_bbox":      list(c_er.pdf_bbox),
                "fully_on_page": c_er.fully_on_page,
                "image_size":    [c_er.image_bgr.shape[1], c_er.image_bgr.shape[0]],
            },
            "template": {
                "image_path":    f"template/{r.name}.png",
                "pdf_bbox":      list(t_er.pdf_bbox),
                "fully_on_page": t_er.fully_on_page,
                "image_size":    [t_er.image_bgr.shape[1], t_er.image_bgr.shape[0]],
            },
            "side_by_side_path": f"side_by_side/{r.name}.png",
        }

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "product_code":     product_code,
        "config_path":      str(config.source_path),
        "config_page_size": {"width": config.page_width, "height": config.page_height},
        "label_rect_pts":   list(config.label_rect_pts) if config.label_rect_pts else None,
        "candidate_pdf":    str(candidate_pdf),
        "template_pdf":     str(template_pdf),
        "render_dpi":       dpi,
        "candidate_extraction": {
            "page_index":    candidate_extract.page_index,
            "page_size_pdf": list(candidate_extract.page_size_pdf),
            "page_rotation": candidate_extract.page_rotation,
            "scale_x":       candidate_extract.scale_x,
            "scale_y":       candidate_extract.scale_y,
            "label_rect_pdf": list(candidate_extract.label_rect_pdf),
        },
        "template_extraction": {
            "page_index":    template_extract.page_index,
            "page_size_pdf": list(template_extract.page_size_pdf),
            "page_rotation": template_extract.page_rotation,
            "scale_x":       template_extract.scale_x,
            "scale_y":       template_extract.scale_y,
            "label_rect_pdf": list(template_extract.label_rect_pdf),
        },
        "regions": [
            _region_entry(c, t)
            for c, t in zip(candidate_extract.extracted, template_extract.extracted)
        ],
    }
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Pretty printing
# ─────────────────────────────────────────────────────────────────────────────

def print_loaded_regions(config: RegionConfig) -> None:
    print()
    print(f"  Loaded region config for {config.product_code}")
    print(f"  Source: {config.source_path}")
    print(f"  Page size (config): {config.page_width} × {config.page_height} px")
    if config.label_rect_pts:
        lr = config.label_rect_pts
        print(f"  label_rect_pts (PDF): ({lr[0]}, {lr[1]}) → ({lr[2]}, {lr[3]})")
    print(f"  Regions: {len(config.regions)} "
          f"({len(config.mandatory())} mandatory, {len(config.optional())} optional)")
    print()
    print(f"  {'name':<28} {'type':<14} {'mand':<6} "
          f"{'bbox_norm / xywh':<40}")
    print("  " + "─" * 94)
    for r in config.regions:
        if r.bbox_norm is not None:
            bn = r.bbox_norm
            desc = f"norm[{bn[0]:.2f},{bn[1]:.2f},{bn[2]:.2f},{bn[3]:.2f}] → {r.width}x{r.height}px"
        else:
            desc = f"({r.x},{r.y},{r.width},{r.height})"
        print(f"  {r.name:<28} {r.type:<14} {str(r.mandatory):<6} {desc:<40}")
    print()


def print_extraction_summary(label: str, extraction: PDFExtraction) -> None:
    Wp, Hp = extraction.page_size_pdf
    Wc, Hc = extraction.config_size
    lr = extraction.label_rect_pdf
    print(f"  {label}: {extraction.pdf_path.name}")
    print(f"    page rect (PDF pts):  {Wp:.2f} × {Hp:.2f}   (rotation {extraction.page_rotation}°)")
    print(f"    label_rect (PDF pts): ({lr[0]:.2f}, {lr[1]:.2f}) → ({lr[2]:.2f}, {lr[3]:.2f})")
    print(f"    canonical (config):   {Wc} × {Hc}")
    print(f"    scale_x (pt per px):  {extraction.scale_x:.5f}   (legacy full-page mapping)")
    print(f"    scale_y (pt per px):  {extraction.scale_y:.5f}")
    off_page = [e.region.name for e in extraction.extracted if not e.fully_on_page]
    if off_page:
        print(f"    NOTE: {len(off_page)} region(s) not fully on page: {off_page}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract regions from a candidate label PDF and the "
                    "matching base template PDF, then save them side-by-side "
                    "for visual debugging.",
    )
    p.add_argument("candidate_pdf", help="Path to the candidate label PDF.")
    p.add_argument("--product", "-p", required=True,
                   help="Product code, e.g. L10555-MIADM.")
    p.add_argument("--ground-truth-dir", default="ground_truth",
                   help="Directory containing approved template PDFs (read-only). "
                        "Default: ./ground_truth")
    p.add_argument("--regions-dir", default=None,
                   help="Directory containing region configs. "
                        "Default: <repo>/config/regions")
    p.add_argument("--out-dir", default="outputs/regions",
                   help="Where to write debug outputs. "
                        "Default: ./outputs/regions")
    p.add_argument("--dpi", type=int, default=150,
                   help="Render DPI for each region clip. Default: 150")
    p.add_argument("--page", type=int, default=0,
                   help="Page index of the PDFs to use (default: 0).")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Verbose logging.")
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
    out_root         = Path(args.out_dir).resolve()

    if not candidate_pdf.exists():
        print(f"ERROR: candidate PDF not found: {candidate_pdf}", file=sys.stderr)
        return 2

    # 1. Load config
    try:
        config = load_region_config(args.product, regions_dir)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except RegionConfigError as exc:
        print(f"ERROR: invalid region config — {exc}", file=sys.stderr)
        return 2

    print_loaded_regions(config)

    # 2. Resolve template
    try:
        template_pdf = find_template_pdf(ground_truth_dir, args.product)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"  Template PDF (read-only): {template_pdf}")
    print(f"  Candidate PDF:            {candidate_pdf}")
    print()

    # 3. Prepare output dirs
    out_dir         = out_root / args.product
    cand_dir        = out_dir / "candidate"
    tmpl_dir        = out_dir / "template"
    sbs_dir         = out_dir / "side_by_side"
    for d in (cand_dir, tmpl_dir, sbs_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 4. Extract regions from both PDFs
    print("  Extracting regions from candidate and template…")
    print()
    candidate_extract = extract_regions(
        candidate_pdf, config,
        page_index=args.page, render_dpi=args.dpi, save_dir=cand_dir,
    )
    template_extract = extract_regions(
        template_pdf, config,
        page_index=args.page, render_dpi=args.dpi, save_dir=tmpl_dir,
    )

    print_extraction_summary("Candidate", candidate_extract)
    print_extraction_summary("Template",  template_extract)

    # 5. Side-by-side composites
    by_name_t = template_extract.by_name()
    by_name_c = candidate_extract.by_name()
    for name in (e.name for e in candidate_extract.extracted):
        save_side_by_side(
            by_name_t[name].image_bgr,
            by_name_c[name].image_bgr,
            sbs_dir / f"{name}.png",
            name,
        )

    # 6. Manifest
    manifest_path = write_manifest(
        out_dir, args.product, config,
        candidate_pdf, template_pdf,
        candidate_extract, template_extract, args.dpi,
    )

    print("  ✓ Region extraction complete.")
    print()
    print(f"  Outputs at: {out_dir}")
    print(f"    • candidate/<region>.png      ({len(candidate_extract.extracted)} files)")
    print(f"    • template/<region>.png       ({len(template_extract.extracted)} files)")
    print(f"    • side_by_side/<region>.png   ({len(candidate_extract.extracted)} files)")
    print(f"    • manifest.json               → {manifest_path.name}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
