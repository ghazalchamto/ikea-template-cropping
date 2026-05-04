#!/usr/bin/env python3
"""
Generate JSON + HTML + candidate-page overview image for region validation.

Resolves the approved template from ground_truth/templates/<PRODUCT>/<VERSION>/.

**Region schema (strict by default)**  
Requires ``<regions-dir>/<PRODUCT>.json`` (default: ``config/regions/``). If the
file is missing, the script **fails** unless you pass ``--allow-auto-schema``,
which generates an approximate schema from ``template.json`` (not for
production) and saves it under ``outputs/generated_regions/``.

Usage
-----
    PYTHONPATH=. python scripts/generate_region_report.py "data/labels/label (2).pdf" \\
        --product 1R10-PRADM \\
        --version v6 \\
        --ground-truth-dir data/templates/ground-truth-ikea-labels/ground_truth \\
        --out-dir outputs/reports
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

# Repo root (visual-validation/) on sys.path
THIS = Path(__file__).resolve()
REPO = THIS.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.core.region_loader import default_regions_dir
from src.reporting.pipeline_runner import RegionPipelineError, run_region_report_pipeline


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Resolve template from ground_truth, load region schema (strict), "
                    "validate candidate, write JSON/HTML/overview.",
    )
    p.add_argument("candidate_pdf", help="Path to the candidate label PDF.")
    p.add_argument("--product", "-p", required=True, help="Product code, e.g. 1R10-PRADM.")
    p.add_argument(
        "--version",
        default=None,
        help="Template version folder (e.g. v6). Default: highest vN under the product.",
    )
    p.add_argument(
        "--ground-truth-dir",
        default="data/templates/ground-truth-ikea-labels/ground_truth",
        help="Path to the ground_truth root (must contain templates/).",
    )
    p.add_argument(
        "--regions-dir",
        default=None,
        help="Where to look for hand-written config/regions/<PRODUCT>.json first. "
             "Default: <repo>/config/regions",
    )
    p.add_argument(
        "--allow-auto-schema",
        action="store_true",
        help="If the manual region JSON is missing, generate one from template.json "
             "(approximate — not production-safe). Default: require manual schema.",
    )
    p.add_argument(
        "--generated-regions-dir",
        default="outputs/generated_regions",
        help="With --allow-auto-schema only: directory for generated JSON "
             "(relative to repo root unless absolute).",
    )
    p.add_argument("--out-dir", default="outputs/reports",
                   help="Output directory for report bundle.")
    p.add_argument("--page", type=int, default=0, help="PDF page index (default 0).")
    p.add_argument("--dpi", type=int, default=150,
                   help="Passed to extract_regions for API compatibility (ignored internally).")
    p.add_argument(
        "--overview-dpi", type=float, default=150.0,
        help="Raster DPI for the full-page candidate overview image.",
    )
    p.add_argument(
        "--layout-only",
        action="store_true",
        help="Use layout-first validators (geometry vs template; text/barcode payload ignored).",
    )
    p.add_argument(
        "--layout-tolerance-mm",
        type=float,
        default=1.5,
        help="Default layout tolerance in mm when --layout-only is set.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)-7s  %(name)s — %(message)s",
        stream=sys.stdout,
    )

    candidate_pdf = Path(args.candidate_pdf).resolve()
    ground_truth_dir = Path(args.ground_truth_dir)
    if not ground_truth_dir.is_absolute():
        ground_truth_dir = REPO / ground_truth_dir
    ground_truth_dir = ground_truth_dir.resolve()

    regions_lookup = (
        Path(args.regions_dir).resolve() if args.regions_dir else default_regions_dir()
    )
    gen_dir = Path(args.generated_regions_dir)
    if not gen_dir.is_absolute():
        gen_dir = (REPO / gen_dir).resolve()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (REPO / out_dir).resolve()
    out_dir = out_dir.resolve()

    if not candidate_pdf.exists():
        print(f"ERROR: candidate PDF not found: {candidate_pdf}", file=sys.stderr)
        return 2

    try:
        result = run_region_report_pipeline(
            candidate_pdf=candidate_pdf,
            product_code=args.product,
            template_version=args.version,
            ground_truth_dir=ground_truth_dir,
            regions_lookup=regions_lookup,
            generated_regions_dir=gen_dir,
            out_reports_dir=out_dir,
            allow_auto_schema=args.allow_auto_schema,
            page_index=args.page,
            render_dpi=args.dpi,
            overview_dpi=args.overview_dpi,
            layout_only=args.layout_only,
            layout_tolerance_mm=args.layout_tolerance_mm,
        )
    except RegionPipelineError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.version is None:
        print(f"  Selected template version (latest): {result.template_version}")
    else:
        print(f"  Selected template version:        {result.template_version}")

    print(f"  Validation mode:                {'layout' if args.layout_only else 'legacy'}")
    print(f"  Product:                          {result.product_code}")
    print(f"  Template PDF:                     {result.template_pdf}")
    print(f"  Template JSON:                    {result.template_json}")
    if result.template_png:
        print(f"  Template PNG:                     {result.template_png}")
    print(f"  Region names:                     {', '.join(result.region_names)}")

    print(f"  Final verdict:                    {result.summary.verdict}")
    print(f"  Report JSON:                      {result.report_json}")
    print(f"  Report HTML:                      {result.report_html}")
    print(f"  Overview image (candidate):       {result.overview_png}")
    print(f"  Overview image (template):        {result.template_overview_png}")

    return 0 if result.summary.verdict == "VALID" else 2


if __name__ == "__main__":
    raise SystemExit(main())
