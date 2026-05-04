#!/usr/bin/env python3
"""
Batch-generate region JSON files from ground-truth ``template.json`` (latest vN).

For each ``ground_truth/templates/<PRODUCT>/``, writes ``<out-dir>/<PRODUCT>.json``
only if that file does not already exist. Never modifies files under ground_truth.

Example
-------
    PYTHONPATH=. python scripts/generate_all_region_schemas.py \\
        --ground-truth-dir data/templates/ground-truth-ikea-labels/ground_truth \\
        --out-dir config/regions

Example output::

    1C50-CAA          v3       CREATED
    1R10-PRADM        v15      SKIPPED
    ...

    Summary:
      total products:  42
      created:         18
      skipped:         23
      failed:          1

With ``--validate``, runs a quick region pipeline (strict: on-disk schema only)
for each product that has a sample PDF under ``--labels-dir``; validation
failures are reported but do not change the script exit code (unless schema
generation failed).

With ``--fallback-coarse``, if the precision generator raises, a coarse
``bbox_norm`` schema is written instead and the action is ``CREATED_FALLBACK``.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import List, Optional, Tuple

THIS = Path(__file__).resolve()
REPO = THIS.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.config.settings import DEFAULT_CONFIG
from src.core.pdf_region_extractor import extract_regions
from src.core.region_aggregator import aggregate_region_scores
from src.core.region_loader import load_region_config_file
from src.core.region_schema_fallback import build_coarse_fallback_schema_dict
from src.core.region_schema_from_template import (
    build_region_schema_dict,
    load_template_json,
    write_region_schema,
)
from src.core.region_validators import validate_extractions
from src.core.template_store import GroundTruthTemplateCatalog


def list_template_products(catalog: GroundTruthTemplateCatalog) -> List[str]:
    """Product folder names that have at least one ``vN`` version (excludes shared data dirs)."""
    out: List[str] = []
    for name in catalog.list_products():
        try:
            catalog.list_versions(name)
        except FileNotFoundError:
            continue
        out.append(name)
    return out


def find_sample_label(product: str, labels_dir: Path) -> Optional[Path]:
    """First matching ``*.pdf`` under *labels_dir*, or None."""
    if not labels_dir.is_dir():
        return None
    exact = labels_dir / f"{product}.pdf"
    if exact.is_file():
        return exact.resolve()
    for p in sorted(labels_dir.glob("*.pdf")):
        if product in p.stem or product in p.name:
            return p.resolve()
    return None


def run_quick_validation(
    product: str,
    region_path: Path,
    template_pdf: Path,
    candidate_pdf: Path,
) -> Tuple[str, str]:
    """
    Run extract → validate → aggregate. Returns (status, detail) where
    status is VALID / INVALID / ERROR.
    """
    try:
        cfg = load_region_config_file(region_path, expected_product_code=product)
        template_extr = extract_regions(template_pdf, cfg, page_index=0, render_dpi=150)
        candidate_extr = extract_regions(candidate_pdf, cfg, page_index=0, render_dpi=150)
        scores = validate_extractions(template_extr, candidate_extr)
        summary = aggregate_region_scores(
            region_scores=scores,
            region_schema=cfg,
            cfg=DEFAULT_CONFIG,
        )
        return summary.verdict, f"score={summary.final_score:.4f}"
    except Exception as exc:
        return "ERROR", f"{type(exc).__name__}: {exc}"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create missing config/regions/<PRODUCT>.json from latest template.json.",
    )
    p.add_argument(
        "--ground-truth-dir",
        default="data/templates/ground-truth-ikea-labels/ground_truth",
        help="ground_truth root (contains templates/).",
    )
    p.add_argument(
        "--out-dir",
        default="config/regions",
        help="Where to write <PRODUCT>.json (relative to repo unless absolute).",
    )
    p.add_argument(
        "--validate",
        action="store_true",
        help="After batch, run quick region validation per product when a sample PDF exists.",
    )
    p.add_argument(
        "--labels-dir",
        default="data/labels",
        help="Directory to search for sample candidate PDFs (with --validate).",
    )
    p.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print tracebacks for schema generation failures.",
    )
    p.add_argument(
        "--fallback-coarse",
        action="store_true",
        help="On generator failure, write a coarse bbox_norm schema from PDF aspect "
             "(prints CREATED_FALLBACK).",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    gt = Path(args.ground_truth_dir)
    if not gt.is_absolute():
        gt = REPO / gt
    gt = gt.resolve()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO / out_dir
    out_dir = out_dir.resolve()
    labels_dir = Path(args.labels_dir)
    if not labels_dir.is_absolute():
        labels_dir = REPO / labels_dir
    labels_dir = labels_dir.resolve()

    try:
        catalog = GroundTruthTemplateCatalog(gt)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    products = list_template_products(catalog)
    rows: List[Tuple[str, str, str, str]] = []  # product, version, action, note
    n_created = n_created_fallback = n_skipped = n_failed = 0

    for product in products:
        out_path = out_dir / f"{product}.json"
        if out_path.is_file():
            try:
                res_skip = catalog.resolve(product, version=None)
                ver_skip = res_skip.version
            except Exception:
                ver_skip = "—"
            rows.append((product, ver_skip, "SKIPPED", "file already exists"))
            n_skipped += 1
            continue
        resolved = catalog.resolve(product, version=None)
        raw: dict = {}
        try:
            raw = load_template_json(resolved.template_json)
        except Exception:
            raw = {}
        try:
            schema = build_region_schema_dict(
                raw, product, template_version=resolved.version,
            )
            write_region_schema(out_path, schema)
            load_region_config_file(out_path, expected_product_code=product)
        except Exception as exc:
            if out_path.is_file():
                try:
                    out_path.unlink()
                except OSError:
                    pass
            if args.fallback_coarse:
                try:
                    schema_fb = build_coarse_fallback_schema_dict(
                        product,
                        template=raw,
                        label_pdf_path=resolved.label_pdf,
                        label_png_path=resolved.label_png,
                        template_version=resolved.version,
                    )
                    write_region_schema(out_path, schema_fb)
                    load_region_config_file(out_path, expected_product_code=product)
                    rows.append((
                        product,
                        resolved.version,
                        "CREATED_FALLBACK",
                        f"{type(exc).__name__}: {exc}",
                    ))
                    n_created_fallback += 1
                except Exception as exc2:
                    rows.append((
                        product,
                        resolved.version,
                        "FAILED",
                        f"{type(exc2).__name__}: {exc2} (after {type(exc).__name__})",
                    ))
                    n_failed += 1
                    if args.verbose:
                        traceback.print_exc()
            else:
                rows.append((product, resolved.version, "FAILED", f"{type(exc).__name__}: {exc}"))
                n_failed += 1
                if args.verbose:
                    traceback.print_exc()
            continue

        rows.append((product, resolved.version, "CREATED", str(out_path)))
        n_created += 1

    # Column widths
    w_p = max((len(r[0]) for r in rows), default=8)
    w_v = max((len(r[1]) for r in rows), default=8)
    w_a = max((len(r[2]) for r in rows), default=len("CREATED_FALLBACK"))

    print()
    for product, version, action, note in rows:
        print(f"  {product:<{w_p}}  {version:<{w_v}}  {action:<{w_a}}  {note}")
    print()
    print("  Summary:")
    print(f"    total products:       {len(products)}")
    print(f"    created:              {n_created}")
    print(f"    created (fallback):   {n_created_fallback}")
    print(f"    skipped:              {n_skipped}")
    print(f"    failed:               {n_failed}")
    print()

    if args.validate:
        print("  Validation (sample labels, strict on-disk schema):")
        for product, version, action, _ in rows:
            if action == "FAILED":
                print(f"    {product:<{w_p}}  —           N/A (schema not created)")
                continue
            region_path = out_dir / f"{product}.json"
            sample = find_sample_label(product, labels_dir)
            if sample is None:
                print(f"    {product:<{w_p}}  {version:<{w_v}}  N/A (no sample PDF)")
                continue
            try:
                resolved = catalog.resolve(product, version=None)
            except FileNotFoundError:
                print(f"    {product:<{w_p}}  {version:<{w_v}}  ERROR (template resolve)")
                continue
            verdict, detail = run_quick_validation(
                product, region_path, resolved.label_pdf, sample,
            )
            print(f"    {product:<{w_p}}  {version:<{w_v}}  {verdict}  ({detail})  sample={sample.name}")
        print()

    return 1 if n_failed > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
