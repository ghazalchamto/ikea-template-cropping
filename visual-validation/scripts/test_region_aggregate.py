#!/usr/bin/env python3
"""
scripts/test_region_aggregate.py
================================
End-to-end debug runner for the region-based validation pipeline:

    load region config  →  extract regions from template + candidate PDFs
                       →  compare regions
                       →  aggregate scores
                       →  print final verdict

This is the first script that produces a final VALID / INVALID result.
It does **not** write reports — that's a later step.

Usage
-----
    PYTHONPATH=. python scripts/test_region_aggregate.py "data/labels/label.pdf" \
        --product L10555-MIADM \
        --ground-truth-dir data/templates/ground-truth-ikea-labels/ground_truth

Optional
--------
    --regions-dir   <path>   override config/regions location
    --dpi           <int>    render DPI for region clips (default 150)
    --page          <int>    page index (default 0)
    --no-save-clips          skip writing per-region PNGs to disk
    --json-out      <path>   write the ValidationSummary as JSON
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

# Make `src` importable when run from the repo root.
THIS = Path(__file__).resolve()
REPO = THIS.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config.settings import DEFAULT_CONFIG
from src.core.pdf_region_extractor import (
    ExtractedRegion, PDFExtraction, extract_regions,
)
from src.core.region_aggregator import (
    STATUS_OPTIONAL, STATUS_PASS, ValidationSummary, aggregate_region_scores,
)
from src.core.region_validators import validate_extractions
from src.core.region_loader import (
    RegionConfigError, default_regions_dir, load_region_config,
)

# Reuse template lookup from extract_regions.py (same loader used by the
# scoring debug script — keeps behaviour identical).
_spec = importlib.util.spec_from_file_location(
    "extract_regions", THIS.parent / "extract_regions.py")
_extract_regions_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_extract_regions_mod)
find_template_pdf = _extract_regions_mod.find_template_pdf


# ─────────────────────────────────────────────────────────────────────────────
# Pretty printing
# ─────────────────────────────────────────────────────────────────────────────

def _status_marker(status: str) -> str:
    return {
        STATUS_PASS:     "PASS    ",
        STATUS_OPTIONAL: "OPTIONAL",
    }.get(status, "FAIL    ")


def _save_debug_region(
    debug_dir:      Path,
    region_name:    str,
    template_clip:  np.ndarray,
    candidate_clip: np.ndarray,
) -> None:
    """Write template / candidate / side-by-side PNGs for one region."""
    debug_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(debug_dir / f"{region_name}_template.png"),  template_clip)
    cv2.imwrite(str(debug_dir / f"{region_name}_candidate.png"), candidate_clip)

    # Side-by-side: pad the shorter clip to the taller height (clips of the
    # same region are normally equal-sized, but this keeps it robust).
    h_t, w_t = template_clip.shape[:2]
    h_c, w_c = candidate_clip.shape[:2]
    pad = max(h_t, h_c)
    t_pad = template_clip
    c_pad = candidate_clip
    if h_t < pad:
        t_pad = np.vstack([
            template_clip,
            np.full((pad - h_t, w_t, 3), 255, dtype=np.uint8),
        ])
    if h_c < pad:
        c_pad = np.vstack([
            candidate_clip,
            np.full((pad - h_c, w_c, 3), 255, dtype=np.uint8),
        ])

    header_h = 28
    width    = t_pad.shape[1] + c_pad.shape[1]
    header   = np.full((header_h, width, 3), 30, dtype=np.uint8)
    cv2.putText(
        header, f"TEMPLATE  |  CANDIDATE   {region_name}",
        (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
    )
    body = np.hstack([t_pad, c_pad])
    out  = np.vstack([header, body])
    cv2.imwrite(str(debug_dir / f"{region_name}_side_by_side.png"), out)


def save_debug_regions(
    template_extr:  PDFExtraction,
    candidate_extr: PDFExtraction,
    debug_dir:      Path,
) -> None:
    """For every region, save template / candidate / side-by-side PNGs."""
    by_t = template_extr.by_name()
    by_c = candidate_extr.by_name()
    for er_t in template_extr.extracted:
        er_c = by_c.get(er_t.name)
        if er_c is None:
            continue
        _save_debug_region(debug_dir, er_t.name, er_t.image_bgr, er_c.image_bgr)


def print_summary(summary: ValidationSummary) -> None:
    print()
    print("  Region Summary")
    print("  " + "─" * 110)

    name_w = max((len(r.name)        for r in summary.region_results), default=10)
    type_w = max((len(r.region_type) for r in summary.region_results), default=10)
    meth_w = max((len(r.method)      for r in summary.region_results), default=10)
    name_w = max(name_w, 22)
    type_w = max(type_w, 14)
    meth_w = max(meth_w, 18)

    header = (f"  {'region':<{name_w}}  {'type':<{type_w}}  "
              f"{'method':<{meth_w}}  {'score':<7}  status")
    print(header)
    print("  " + "─" * (len(header) - 2))

    for r in summary.region_results:
        marker = _status_marker(r.status)
        if r.mandatory:
            score_s = f"{r.combined_score:.4f}"
            tail    = f"thr={r.threshold:.2f}"
        else:
            score_s = "n/a"
            tail    = f"score={r.combined_score:.2f}  ignored"
        print(f"  {r.name:<{name_w}}  {r.region_type:<{type_w}}  "
              f"{r.method:<{meth_w}}  {score_s:<7}  {marker}  {tail}")

    print()
    print("  Final")
    print("  " + "─" * 110)
    print(f"  Verdict:           {summary.verdict}")
    print(f"  Final score:       {summary.final_score:.4f}      "
          f"(threshold {summary.global_threshold:.2f})")
    print(f"  Mandatory passed:  {summary.mandatory_passed}")
    print(f"  Mandatory failed:  {summary.mandatory_failed}")
    print(f"  Optional ignored:  {summary.optional_count}")
    print(f"  Total regions:     {summary.total_regions}")
    if summary.failed_regions:
        print(f"  Failed names:      {', '.join(summary.failed_regions)}")
    if summary.ignored_regions:
        print(f"  Ignored names:     {', '.join(summary.ignored_regions)}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the full region-based pipeline: extract → compare → "
                    "aggregate → print final VALID/INVALID verdict.",
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
    p.add_argument("--out-dir", default="outputs/regions",
                   help="Where to write extraction debug clips. "
                        "Default: ./outputs/regions")
    p.add_argument("--dpi", type=int, default=150,
                   help="Render DPI for region clips. Default: 150")
    p.add_argument("--page", type=int, default=0,
                   help="Page index to use (default: 0).")
    p.add_argument("--json-out", default=None,
                   help="Optional path for a JSON dump of the ValidationSummary.")
    p.add_argument("--no-save-clips", action="store_true",
                   help="Skip writing region PNGs to disk (faster, in-memory only).")
    p.add_argument("--debug-regions-dir",
                   default="outputs/debug_regions",
                   help="Where to write per-region debug PNGs "
                        "(<region>_template.png, <region>_candidate.png, "
                        "<region>_side_by_side.png). Default: outputs/debug_regions/<product>")
    p.add_argument("--no-debug-regions", action="store_true",
                   help="Skip writing per-region debug PNGs.")
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
    out_root         = Path(args.out_dir).resolve()

    if not candidate_pdf.exists():
        print(f"ERROR: candidate PDF not found: {candidate_pdf}", file=sys.stderr)
        return 2

    # 1. Region config
    try:
        config = load_region_config(args.product, regions_dir)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr); return 2
    except RegionConfigError as exc:
        print(f"ERROR: invalid region config — {exc}", file=sys.stderr); return 2

    # 2. Template PDF
    try:
        template_pdf = find_template_pdf(ground_truth_dir, args.product)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr); return 2

    print(f"  Product code:   {args.product}")
    print(f"  Region config:  {config.source_path}")
    print(f"  Template PDF:   {template_pdf}")
    print(f"  Candidate PDF:  {candidate_pdf}")
    print(f"  Regions:        {len(config.regions)} "
          f"({len(config.mandatory())} mandatory, {len(config.optional())} optional)")
    if config.label_rect_pts:
        lr = config.label_rect_pts
        print(f"  label_rect_pts (config): ({lr[0]}, {lr[1]}) → ({lr[2]}, {lr[3]}) PDF pt")

    # 3. Extraction
    out_dir = out_root / args.product
    save_t  = None if args.no_save_clips else (out_dir / "template")
    save_c  = None if args.no_save_clips else (out_dir / "candidate")

    template_extr = extract_regions(
        template_pdf, config, page_index=args.page,
        render_dpi=args.dpi, save_dir=save_t,
    )
    candidate_extr = extract_regions(
        candidate_pdf, config, page_index=args.page,
        render_dpi=args.dpi, save_dir=save_c,
    )

    lt = template_extr.label_rect_pdf
    lc = candidate_extr.label_rect_pdf
    print(f"  Label rect (template PDF): ({lt[0]:.2f},{lt[1]:.2f})→({lt[2]:.2f},{lt[3]:.2f}) pt")
    print(f"  Label rect (candidate PDF): ({lc[0]:.2f},{lc[1]:.2f})→({lc[2]:.2f},{lc[3]:.2f}) pt")

    # 4. Typed validation (per-region; dispatches by region.type).
    region_scores = validate_extractions(template_extr, candidate_extr)

    # 4b. Debug PNGs for every region (template / candidate / side_by_side)
    if not args.no_debug_regions:
        debug_dir = Path(args.debug_regions_dir).resolve() / args.product
        save_debug_regions(template_extr, candidate_extr, debug_dir)
        print(f"  Debug clips:    {debug_dir}")

    # 5. Aggregation → final verdict
    summary = aggregate_region_scores(
        region_scores=region_scores,
        region_schema=config,
        cfg=DEFAULT_CONFIG,
    )

    # 6. Print
    print_summary(summary)

    if args.json_out:
        payload = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "product_code":     args.product,
            "candidate_pdf":    str(candidate_pdf),
            "template_pdf":     str(template_pdf),
            "config_path":      str(config.source_path),
            "label_rect_pts_config": list(config.label_rect_pts)
                if config.label_rect_pts else None,
            "label_rect_pdf_template":  list(template_extr.label_rect_pdf),
            "label_rect_pdf_candidate": list(candidate_extr.label_rect_pdf),
            "summary":          summary.to_dict(),
            "region_scores":    [s.to_dict() for s in region_scores],
        }
        out_path = Path(args.json_out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"  Summary JSON written → {out_path}")
        print()

    # Exit code mirrors the verdict so CI can branch on it.
    return 0 if summary.verdict == "VALID" else 2


if __name__ == "__main__":
    raise SystemExit(main())
