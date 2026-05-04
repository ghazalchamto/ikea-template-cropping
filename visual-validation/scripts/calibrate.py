#!/usr/bin/env python3
"""
scripts/calibrate.py
====================
Run this on known-GOOD labels to find the right threshold values.

Usage
-----
  python scripts/calibrate.py data/labels/known_good/ --product 1C50-CAA

Output
------
  Per-file score breakdown + recommended valid_score_threshold and ssim_threshold.
  Saves outputs/calibration/calibration_report.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import numpy as np
from src.config.settings import ValidationConfig
from src.validator import LabelValidator

GROUND_TRUTH = "data/templates/ground-truth-ikea-labels/ground_truth"
SUPPORTED    = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif"}


def main():
    p = argparse.ArgumentParser(description="Threshold calibration on known-good labels")
    p.add_argument("labels_dir", help="Directory of known-good label files")
    p.add_argument("--product",      "-p", required=True, help="Product code, e.g. 1C50-CAA")
    p.add_argument("--version",      "-v", default=None)
    p.add_argument("--ground-truth",       default=GROUND_TRUTH)
    p.add_argument("--dpi",          type=int, default=300)
    p.add_argument("--output",       "-o",   default="outputs/calibration")
    args = p.parse_args()

    d = Path(args.labels_dir)
    if not d.is_dir():
        print(f"ERROR: Not a directory: {d}"); sys.exit(1)

    files = sorted(f for f in d.iterdir() if f.suffix.lower() in SUPPORTED)
    if not files:
        print(f"No supported files in {d}"); sys.exit(1)

    # During calibration set thresholds to 0 so nothing is rejected
    cfg = ValidationConfig(
        render_dpi=args.dpi,
        valid_score_threshold=0.0,
        ssim_threshold=0.0,
    )
    v = LabelValidator(ground_truth_dir=args.ground_truth, cfg=cfg)

    print(f"\nCalibrating on {len(files)} known-good labels → product: {args.product}\n")
    header = f"{'File':<36} {'Score':>7}  {'SSIM':>7}  {'Pixel':>7}  {'Edge':>7}  {'Tile':>7}"
    print(header)
    print("─" * len(header))

    records = []
    for f in files:
        try:
            r = v.validate(f, product_code=args.product,
                           template_version=args.version, output_dir=None)
            c = r.comparison
            print(f"{f.name:<36} {r.score:>7.4f}  {c.ssim_score:>7.4f}  "
                  f"{c.pixel_similarity:>7.4f}  {c.edge_similarity:>7.4f}  "
                  f"{c.tile_pass_rate:>7.4f}")
            records.append({
                "file":             str(f),
                "final_score":      r.score,
                "ssim_score":       c.ssim_score,
                "pixel_similarity": c.pixel_similarity,
                "edge_similarity":  c.edge_similarity,
                "tile_pass_rate":   c.tile_pass_rate,
            })
        except Exception as e:
            print(f"{f.name:<36}  ERROR: {e}")

    if not records:
        print("No files processed."); sys.exit(1)

    finals = [r["final_score"] for r in records]
    ssims  = [r["ssim_score"]  for r in records]

    p5f  = float(np.percentile(finals, 5))
    p5s  = float(np.percentile(ssims,  5))
    rec_score = round(p5f - 0.015, 3)
    rec_ssim  = round(p5s - 0.015, 3)

    print("─" * len(header))
    print(f"\n  Files processed : {len(records)}")
    print(f"  Score  — min:{min(finals):.4f}  mean:{np.mean(finals):.4f}  p5:{p5f:.4f}")
    print(f"  SSIM   — min:{min(ssims):.4f}   mean:{np.mean(ssims):.4f}   p5:{p5s:.4f}")
    print()
    print(f"  ✓  Recommended  valid_score_threshold : {rec_score}")
    print(f"  ✓  Recommended  ssim_threshold        : {rec_ssim}")
    print()
    print(f"  → Edit src/config/settings.py and set these values.")
    print(f"    Or pass  --threshold {rec_score}  to validate.py\n")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rpt = out / "calibration_report.json"
    rpt.write_text(json.dumps({
        "product_code": args.product,
        "n_files": len(records),
        "recommended_valid_score_threshold": rec_score,
        "recommended_ssim_threshold": rec_ssim,
        "stats": {
            "min_score": min(finals),   "mean_score": float(np.mean(finals)),
            "p5_score":  p5f,           "min_ssim":   min(ssims),
            "p5_ssim":   p5s,
        },
        "records": records,
    }, indent=2))
    print(f"  Calibration report → {rpt}\n")


if __name__ == "__main__":
    main()
