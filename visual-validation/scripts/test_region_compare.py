#!/usr/bin/env python3
"""
scripts/test_region_compare.py
==============================
Debug tool: extract regions from a candidate label and the matching base
template, then compute per-region similarity scores and print them.

This script does NOT aggregate scores or render a final verdict.
That is a separate step. Output is metric-only.

Usage
-----
    python scripts/test_region_compare.py "data/labels/label.pdf" \
        --product L10555-MIADM

    python scripts/test_region_compare.py candidate.pdf \
        --product L10555-MIADM \
        --ground-truth-dir data/templates/ground-truth-ikea-labels/ground_truth \
        --regions-dir      config/regions \
        --out-dir          outputs/regions \
        --dpi              150 \
        --json-out         scores.json

Output
------
* A console table with one row per region:
      region | mandatory | weight | ssim | pixel | edge | combined | size | notes
* Optional --json-out path: structured JSON with every score for diffing.
* Region clip PNGs and a manifest under outputs/regions/<product_code>/
  (extraction debug output, same as the previous step).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

# Make `src` importable when run from the repo root.
THIS = Path(__file__).resolve()
REPO = THIS.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.core.region_loader import (
    RegionConfig, RegionConfigError, default_regions_dir, load_region_config,
)
from src.core.pdf_region_extractor import extract_regions
from src.core.region_comparator import (
    RegionScore, compare_extractions,
    W_SSIM, W_PIXEL, W_EDGE, PIXEL_DIFF_TOLERANCE, CANNY_LOW, CANNY_HIGH,
)

# Reuse the same template lookup from extract_regions.py to keep behavior
# identical. (Importing the script as a module rather than copying the function.)
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "extract_regions", THIS.parent / "extract_regions.py")
_extract_regions_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_extract_regions_mod)
find_template_pdf = _extract_regions_mod.find_template_pdf


# ─────────────────────────────────────────────────────────────────────────────
# Pretty printing
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_score(v: float) -> str:
    return f"{v:6.4f}"


def print_score_table(scores: List[RegionScore]) -> None:
    cols = ("region", "mand", "weight", "ssim", "pixel", "edge", "combined",
            "size", "notes")
    widths = (32, 5, 6, 6, 6, 6, 8, 11, 26)
    header = "  " + "  ".join(f"{c:<{w}}" for c, w in zip(cols, widths))
    print()
    print(header)
    print("  " + "─" * (sum(widths) + 2 * (len(widths) - 1)))
    for s in scores:
        size = f"{s.width}x{s.height}"
        notes = ",".join(s.notes) if s.notes else ""
        if len(notes) > widths[-1]:
            notes = notes[: widths[-1] - 1] + "…"
        row = (
            s.region_name[:widths[0]].ljust(widths[0]),
            ("yes" if s.mandatory else "no").ljust(widths[1]),
            f"{s.weight:.2f}".ljust(widths[2]),
            _fmt_score(s.ssim),
            _fmt_score(s.pixel),
            _fmt_score(s.edge),
            _fmt_score(s.combined).ljust(widths[6]),
            size.ljust(widths[7]),
            notes.ljust(widths[8]),
        )
        print("  " + "  ".join(row))
    print()


def print_metric_settings() -> None:
    print()
    print("  Metric configuration (deterministic, offline)")
    print(f"    weights:               ssim={W_SSIM}  pixel={W_PIXEL}  edge={W_EDGE}")
    print(f"    pixel diff tolerance:  {PIXEL_DIFF_TOLERANCE} (0–255 grayscale)")
    print(f"    canny thresholds:      low={CANNY_LOW}  high={CANNY_HIGH}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute per-region similarity scores between a candidate "
                    "label PDF and the matching base template PDF. "
                    "Region-level only — no aggregation, no verdict.",
    )
    p.add_argument("candidate_pdf", help="Path to the candidate label PDF.")
    p.add_argument("--product", "-p", required=True,
                   help="Product code, e.g. L10555-MIADM.")
    p.add_argument("--ground-truth-dir", default="ground_truth",
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
                   help="Optional path for a JSON dump of all per-region scores.")
    p.add_argument("--no-save-clips", action="store_true",
                   help="Skip writing region PNGs to disk (faster, in-memory only).")
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

    # 1. Load region config (multi-template: keyed by product_code)
    try:
        config = load_region_config(args.product, regions_dir)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr); return 2
    except RegionConfigError as exc:
        print(f"ERROR: invalid region config — {exc}", file=sys.stderr); return 2

    # 2. Resolve the template PDF (read-only)
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

    # 3. Extract clips from both PDFs
    out_dir  = out_root / args.product
    save_t   = None if args.no_save_clips else (out_dir / "template")
    save_c   = None if args.no_save_clips else (out_dir / "candidate")

    template_extr = extract_regions(
        template_pdf, config, page_index=args.page,
        render_dpi=args.dpi, save_dir=save_t,
    )
    candidate_extr = extract_regions(
        candidate_pdf, config, page_index=args.page,
        render_dpi=args.dpi, save_dir=save_c,
    )

    # 4. Compare per region (in-memory; no disk round-trip required)
    scores = compare_extractions(template_extr, candidate_extr)

    # 5. Output
    print_metric_settings()
    print_score_table(scores)

    if args.json_out:
        payload = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "product_code":     args.product,
            "candidate_pdf":    str(candidate_pdf),
            "template_pdf":     str(template_pdf),
            "config_path":      str(config.source_path),
            "metric_settings": {
                "weights":               {"ssim": W_SSIM,
                                          "pixel": W_PIXEL,
                                          "edge":  W_EDGE},
                "pixel_diff_tolerance":  PIXEL_DIFF_TOLERANCE,
                "canny_low":             CANNY_LOW,
                "canny_high":            CANNY_HIGH,
            },
            "regions": [s.to_dict() for s in scores],
        }
        out_path = Path(args.json_out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"  Scores JSON written → {out_path}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
