#!/usr/bin/env python3
"""
scripts/validate.py
===================
CLI tool for label validation.

ALWAYS run from inside the visual-validation/ directory:

  # See all available product codes
  python scripts/validate.py --list-templates

  # Validate one label against its product template
  python scripts/validate.py data/labels/my_label.pdf --product 1C50-CAA

  # Specific template version
  python scripts/validate.py data/labels/my_label.pdf --product 1C50-CAA --version v1

  # High-DPI (more precise, slower)
  python scripts/validate.py data/labels/my_label.pdf --product 1C50-CAA --dpi 300

  # Batch: all PDFs in a folder against the same product template
  python scripts/validate.py data/labels/ --product 1C50-CAA --batch

  # Custom output folder
  python scripts/validate.py data/labels/my_label.pdf --product 1C50-CAA --output outputs/run_01/

  # Machine-readable: print only VALID or INVALID
  python scripts/validate.py data/labels/my_label.pdf --product 1C50-CAA --quiet

Exit codes
  0  VALID
  1  Error
  2  INVALID
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.settings import ValidationConfig
from src.validator import LabelValidator

GROUND_TRUTH = "data/templates/ground-truth-ikea-labels/ground_truth"
SUPPORTED    = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif"}


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="validate.py",
        description="IKEA label visual validation — direct product-code comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("input", nargs="?",
                   help="Label file (PDF/PNG/JPG) or directory (with --batch)")
    p.add_argument("--product",  "-p", default=None,
                   help="Product code to compare against, e.g. 1C50-CAA  (REQUIRED for validation)")
    p.add_argument("--version",  "-v", default=None,
                   help="Template version, e.g. v1.  Default: latest available.")
    p.add_argument("--output",   "-o", default="outputs/",
                   help="Output directory.  Default: outputs/")
    p.add_argument("--batch",    "-b", action="store_true",
                   help="Treat INPUT as a directory and process all supported files.")
    p.add_argument("--ground-truth", default=GROUND_TRUTH,
                   help=f"Path to ground_truth/.  Default: {GROUND_TRUTH}")
    p.add_argument("--dpi",  type=int,   default=None,
                   help="PDF render DPI.  Default: from settings (300).")
    p.add_argument("--threshold", type=float, default=None,
                   help="Override valid_score_threshold.")
    p.add_argument("--list-templates", "-l", action="store_true",
                   help="List all available product templates and exit.")
    p.add_argument("--no-comparison", action="store_true",
                   help="Skip saving the side-by-side comparison image.")
    p.add_argument("--quiet", "-q", action="store_true",
                   help="Print only VALID or INVALID.")
    p.add_argument(
        "--comparison-mode",
        choices=("full", "region_split"),
        default=None,
        help="full = single global score drives verdict; region_split = verdict from "
             "mandatory quadrants (full-image scores still in report). "
             "Default: from ValidationConfig (region_split).",
    )
    return p


def main() -> int:
    args = _parser().parse_args()

    # ── Build config ──────────────────────────────────────────────────────────
    cfg = ValidationConfig()
    if args.dpi       is not None: cfg.render_dpi            = args.dpi
    if args.threshold is not None: cfg.valid_score_threshold = args.threshold
    if args.comparison_mode is not None:
        cfg.comparison_mode = args.comparison_mode

    # ── Check ground_truth path ───────────────────────────────────────────────
    gt = Path(args.ground_truth)
    if not gt.exists():
        print(f"\nERROR: ground_truth directory not found:\n  {gt.resolve()}\n"
              f"Run from inside visual-validation/, or pass --ground-truth <path>\n",
              file=sys.stderr)
        return 1

    v = LabelValidator(ground_truth_dir=gt, cfg=cfg)

    # ── List mode ─────────────────────────────────────────────────────────────
    if args.list_templates:
        v.list_templates()
        return 0

    # ── Need an input ─────────────────────────────────────────────────────────
    if not args.input:
        _parser().print_help()
        return 1

    inp = Path(args.input)
    if not inp.exists():
        print(f"ERROR: Input not found: {inp}", file=sys.stderr)
        return 1

    if not args.product:
        print("\nERROR: --product is required.\n"
              "Example: python scripts/validate.py label.pdf --product 1C50-CAA\n"
              "Run with --list-templates to see available product codes.\n",
              file=sys.stderr)
        return 1

    out_dir = Path(args.output)

    # ── Batch ─────────────────────────────────────────────────────────────────
    if args.batch or inp.is_dir():
        files = sorted(f for f in inp.iterdir() if f.suffix.lower() in SUPPORTED)
        if not files:
            print(f"No supported files in {inp}", file=sys.stderr)
            return 1

        print(f"\nBatch: {len(files)} files  |  product: {args.product}\n")
        results = v.validate_batch(
            label_paths=files,
            product_code=args.product,
            template_version=args.version,
            output_dir=out_dir,
        )
        print(f"\n{'File':<38} {'Tmpl':<12} {'Score':>7}  {'SSIM':>6}  "
              f"{'Pixel':>6}  {'Tile':>6}  Verdict")
        print("─" * 92)
        for r in results:
            fn = Path(r.report_json).stem.replace("_report","") if r.report_json else "?"
            print(f"{fn:<38} {r.template_version:<12} "
                  f"{r.score:>7.4f}  "
                  f"{r.comparison.ssim_score:>6.4f}  "
                  f"{r.comparison.pixel_similarity:>6.4f}  "
                  f"{r.comparison.tile_pass_rate:>6.4f}  "
                  f"{r.verdict}")
        print("─" * 92)
        n_invalid = sum(1 for r in results if r.verdict == "INVALID")
        print(f"VALID: {len(results)-n_invalid}   INVALID: {n_invalid}\n")
        return 0 if n_invalid == 0 else 2

    # ── Single file ───────────────────────────────────────────────────────────
    r = v.validate(
        label_path=inp,
        product_code=args.product,
        template_version=args.version,
        output_dir=out_dir,
        save_comparison=not args.no_comparison,
    )

    if args.quiet:
        print(r.verdict)
        return 0 if r.verdict == "VALID" else 2

    print(f"\n{'='*62}")
    print(f"  Verdict          : {r.verdict}")
    print(f"  Product / Version: {r.product_code} / {r.template_version}")
    if getattr(r.comparison, "comparison_mode", "full") == "region_split":
        print(f"  Comparison mode  : region_split (verdict = mandatory quadrants)")
        print(
            f"  Full-image ref.  : final={r.comparison.full_image_final_score:.4f}  "
            f"valid={r.comparison.full_image_is_valid}  (not used for verdict)"
        )
        if r.comparison.failed_mandatory_regions:
            print(f"  Failed mandatory quadrants: {', '.join(r.comparison.failed_mandatory_regions)}")
    print(f"  Final score      : {r.score:.4f}")
    print(f"  ├ SSIM           : {r.comparison.ssim_score:.4f}")
    print(f"  ├ Pixel sim      : {r.comparison.pixel_similarity:.4f}")
    print(f"  ├ Edge sim       : {r.comparison.edge_similarity:.4f}")
    print(f"  └ Tile pass rate : {r.comparison.tile_pass_rate:.4f}")
    print(f"  Alignment OK     : {r.comparison.alignment_ok}")
    print(f"  Mismatch regions : {len(r.comparison.mismatch_regions)}")
    print(f"  Failed tiles     : "
          f"{sum(1 for t in r.comparison.tile_results if not t.passed)}"
          f" / {len(r.comparison.tile_results)}")

    if r.comparison.mismatch_regions:
        print(f"\n  Top mismatch regions:")
        for i, reg in enumerate(r.comparison.mismatch_regions[:8]):
            print(f"    #{i+1}  [{reg.severity.upper():8}]  "
                  f"src={reg.source:<5}  "
                  f"x={reg.x:4d} y={reg.y:4d}  "
                  f"{reg.w}x{reg.h}px  "
                  f"diff={reg.pixel_diff_ratio*100:.0f}%")

    if r.comparison.named_region_scores:
        print(f"\n  Named region scores:")
        for name, score in r.comparison.named_region_scores.items():
            status = "PASS" if score >= 0.85 else "FAIL"
            print(f"    {name:<20}  {score:.4f}  {status}")

    if r.report_html:
        print(f"\n  HTML report → {r.report_html}")
    if r.grid_map_img:
        print(f"  Grid map    → {r.grid_map_img}")
    print(f"{'='*62}\n")

    return 0 if r.verdict == "VALID" else 2


if __name__ == "__main__":
    sys.exit(main())
