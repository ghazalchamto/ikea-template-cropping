#!/usr/bin/env python3
"""
One-shot region validation + report for a candidate label PDF (demo / thesis).

Strict manual schema by default; pass ``--allow-auto-schema`` for the same
non-production escape hatch as ``generate_region_report.py``.

Usage
-----
    PYTHONPATH=. python scripts/validate_label.py "data/labels/label.pdf" \\
        --product 1R10-PRADM \\
        --version v6

    PYTHONPATH=. python scripts/validate_label.py "data/labels/label.pdf" -p 1R10-PRADM --open-report
"""

from __future__ import annotations

import argparse
import logging
import platform
import subprocess
import sys
import webbrowser
from pathlib import Path
from typing import List, Optional

THIS = Path(__file__).resolve()
REPO = THIS.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.core.region_loader import default_regions_dir
from src.reporting.pipeline_runner import RegionPipelineError, run_region_report_pipeline


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate a label PDF against a product template (region pipeline + report).",
    )
    p.add_argument("candidate_pdf", help="Path to the candidate label PDF.")
    p.add_argument("--product", "-p", required=True, help="Product code, e.g. 1R10-PRADM.")
    p.add_argument(
        "--version",
        default=None,
        help="Template version (e.g. v6). Default: latest vN under the product.",
    )
    p.add_argument(
        "--ground-truth-dir",
        default="data/templates/ground-truth-ikea-labels/ground_truth",
        help="ground_truth root (contains templates/).",
    )
    p.add_argument(
        "--regions-dir",
        default=None,
        help="Region configs directory (default: <repo>/config/regions).",
    )
    p.add_argument(
        "--allow-auto-schema",
        action="store_true",
        help="Allow auto-generated region schema if manual JSON is missing (not production-safe).",
    )
    p.add_argument(
        "--generated-regions-dir",
        default="outputs/generated_regions",
        help="With --allow-auto-schema: where to write generated schema.",
    )
    p.add_argument(
        "--out-dir",
        default="outputs/reports",
        help="Where to write HTML, JSON, and overview PNG.",
    )
    p.add_argument("--page", type=int, default=0, help="PDF page index (default 0).")
    p.add_argument("--dpi", type=int, default=150, help="Render DPI (API compat; often ignored).")
    p.add_argument("--overview-dpi", type=float, default=150.0, help="Overview PNG DPI.")
    p.add_argument(
        "--open-report",
        action="store_true",
        help="Open the HTML report in a browser after validation (macOS/Linux).",
    )
    p.add_argument(
        "--layout-only",
        action="store_true",
        help="Layout-first validation: compare geometry vs template (text/barcode "
             "payload ignored). Requires template.json label_size_mm. "
             "Default: legacy pixel/heuristic validators.",
    )
    p.add_argument(
        "--layout-tolerance-mm",
        type=float,
        default=1.5,
        help="Default per-region layout tolerance in mm when --layout-only is set "
             "(override with tolerance_mm on each region in JSON).",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def _open_html(path: Path) -> None:
    uri = path.as_uri()
    if platform.system() == "Darwin":
        subprocess.run(["open", str(path)], check=False)
    elif platform.system() == "Linux":
        subprocess.run(["xdg-open", str(path)], check=False)
    else:
        webbrowser.open(uri)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)-7s  %(name)s — %(message)s",
        stream=sys.stdout,
    )

    candidate_pdf = Path(args.candidate_pdf).resolve()
    if not candidate_pdf.exists():
        print(f"ERROR: candidate PDF not found: {candidate_pdf}", file=sys.stderr)
        return 2

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

    print(f"  Validation mode:    {'layout' if args.layout_only else 'legacy'}")
    print(f"  Product:            {result.product_code}")
    print(f"  Template version:   {result.template_version}")
    print(f"  Schema source:      {result.schema_source} ({result.config_path})")
    print(f"  Verdict:            {result.summary.verdict}")
    print(f"  Final score:        {result.summary.final_score:.4f}  "
          f"(global threshold {result.summary.global_threshold:.4f})")
    print(f"  Report HTML:        {result.report_html}")
    print(f"  Overview (label):   {result.overview_png}")
    print(f"  Overview (template): {result.template_overview_png}")
    print()

    if args.open_report:
        _open_html(result.report_html)

    return 0 if result.summary.verdict == "VALID" else 2


if __name__ == "__main__":
    raise SystemExit(main())
