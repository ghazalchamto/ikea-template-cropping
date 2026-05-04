"""
Shared region validation + HTML/JSON/overview report pipeline.

Used by ``scripts/generate_region_report.py`` and ``scripts/validate_label.py``.
Does not change scoring or validator implementations — orchestration only.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from src.config.settings import DEFAULT_CONFIG
from src.core.pdf_region_extractor import extract_regions
from src.core.region_aggregator import ValidationSummary, aggregate_region_scores
from src.core.region_loader import RegionConfigError, load_region_config_file
from src.core.region_schema_from_template import (
    build_region_schema_dict,
    load_template_json,
    write_region_schema,
)
from src.core.layout_region_validators import (
    load_label_size_mm_from_template_json,
    validate_layout_extractions,
)
from src.core.pdf_layout_gate import (
    assert_label_aspect_consistent_with_pdf,
    assert_matching_page_dimensions_pt,
    validate_pdf_pair_page_index,
)
from src.core.region_validators import validate_extractions
from src.core.template_store import GroundTruthTemplateCatalog
from src.reporting.region_report import (
    build_report_payload,
    render_candidate_overview,
    render_template_overview,
    safe_label_tag,
    write_html_report,
    write_json_report,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RegionReportResult:
    product_code:       str
    template_version:   str
    schema_source:      str          # "manual" | "auto-generated"
    config_path:        Path
    template_pdf:       Path
    template_json:      Path
    template_png:       Optional[Path]
    region_names:       Tuple[str, ...]
    summary:            ValidationSummary
    report_html:        Path
    report_json:        Path
    overview_png:       Path
    template_overview_png: Path


class RegionPipelineError(Exception):
    """User-facing pipeline failure (missing schema, IO, extraction)."""


def run_region_report_pipeline(
    *,
    candidate_pdf:      Path,
    product_code:       str,
    template_version:   Optional[str],
    ground_truth_dir:   Path,
    regions_lookup:     Path,
    generated_regions_dir: Path,
    out_reports_dir:    Path,
    allow_auto_schema:  bool,
    page_index:         int = 0,
    render_dpi:         int = 150,
    overview_dpi:       float = 150.0,
    layout_only:        bool = False,
    layout_tolerance_mm: float = 1.5,
) -> RegionReportResult:
    """
    Load schema (strict unless *allow_auto_schema*), extract, validate, aggregate,
    write JSON + HTML + overview PNG.

    If *layout_only* is True, use ``validate_layout_extractions`` (geometry / edges,
    content-invariant for text and barcode regions) instead of legacy pixel heuristics.

    Raises
    ------
    RegionPipelineError
        Missing strict schema, invalid config, template resolve failure, extraction error.
    """
    candidate_pdf = candidate_pdf.resolve()
    hand_path = regions_lookup / f"{product_code}.json"
    have_manual = hand_path.is_file()

    if not have_manual and not allow_auto_schema:
        expected = hand_path.resolve()
        raise RegionPipelineError(
            f"Region schema not found for {product_code}.\n"
            f"        Expected file: {expected}\n"
            f"        Create that file or pass allow_auto_schema=True."
        )

    try:
        store = GroundTruthTemplateCatalog(ground_truth_dir)
        resolved = store.resolve(product_code, version=template_version)
    except FileNotFoundError as exc:
        raise RegionPipelineError(str(exc)) from exc

    if have_manual:
        try:
            config = load_region_config_file(hand_path, expected_product_code=product_code)
        except RegionConfigError as exc:
            raise RegionPipelineError(f"invalid region config — {exc}") from exc
        schema_source = "manual"
    else:
        try:
            template_obj = load_template_json(resolved.template_json)
            schema = build_region_schema_dict(
                template_obj,
                product_code,
                template_version=resolved.version,
            )
        except (ValueError, OSError, json.JSONDecodeError) as exc:
            raise RegionPipelineError(f"could not build region schema — {exc}") from exc

        gen_path = generated_regions_dir / f"{product_code}_{resolved.version}.json"
        write_region_schema(gen_path, schema)
        try:
            config = load_region_config_file(gen_path, expected_product_code=product_code)
        except (FileNotFoundError, RegionConfigError) as exc:
            raise RegionPipelineError(f"could not load generated region config — {exc}") from exc
        schema_source = "auto-generated"

    tag = safe_label_tag(candidate_pdf)
    base = f"report_{product_code}_{tag}"
    overview_name = f"{base}_overview.png"
    template_overview_name = f"{base}_template_overview.png"
    json_name = f"{base}.json"
    html_name = f"{base}.html"

    out_reports_dir.mkdir(parents=True, exist_ok=True)
    overview_path = out_reports_dir / overview_name
    template_overview_path = out_reports_dir / template_overview_name
    json_path = out_reports_dir / json_name
    html_path = out_reports_dir / html_name

    template_pdf = resolved.label_pdf
    try:
        validate_pdf_pair_page_index(
            template_pdf, candidate_pdf, page_index=page_index,
        )
    except ValueError as exc:
        raise RegionPipelineError(str(exc)) from exc

    try:
        template_extr = extract_regions(
            template_pdf, config, page_index=page_index, render_dpi=render_dpi,
        )
        candidate_extr = extract_regions(
            candidate_pdf, config, page_index=page_index, render_dpi=render_dpi,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise RegionPipelineError(f"extraction failed — {exc}") from exc

    if layout_only:
        try:
            lw_mm, lh_mm = load_label_size_mm_from_template_json(resolved.template_json)
        except (ValueError, OSError, json.JSONDecodeError) as exc:
            raise RegionPipelineError(
                f"layout mode requires readable template.json with label_size_mm — {exc}"
            ) from exc
        try:
            assert_matching_page_dimensions_pt(template_extr, candidate_extr)
            if config.label_rect_pts is None:
                assert_label_aspect_consistent_with_pdf(
                    template_extr, lw_mm, lh_mm, rel_tol=0.12,
                )
        except ValueError as exc:
            raise RegionPipelineError(str(exc)) from exc
        region_scores = validate_layout_extractions(
            template_extr,
            candidate_extr,
            label_width_mm=lw_mm,
            label_height_mm=lh_mm,
            default_tolerance_mm=layout_tolerance_mm,
        )
    else:
        region_scores = validate_extractions(template_extr, candidate_extr)
    summary = aggregate_region_scores(
        region_scores=region_scores,
        region_schema=config,
        cfg=DEFAULT_CONFIG,
    )

    render_candidate_overview(
        candidate_pdf,
        candidate_extr,
        summary,
        overview_path,
        page_index=page_index,
        overview_dpi=overview_dpi,
        region_scores=region_scores,
    )
    render_template_overview(
        template_pdf,
        template_extr,
        summary,
        template_overview_path,
        page_index=page_index,
        overview_dpi=overview_dpi,
        region_scores=region_scores,
    )

    pipeline = {
        "selected_product":       product_code,
        "selected_version":       resolved.version,
        "template_json":          str(resolved.template_json),
        "template_version":       resolved.version,
        "region_schema_mode":     schema_source,
        "hand_region_config":     str(hand_path.resolve()) if schema_source == "manual" else None,
        "generated_regions_dir":  str(generated_regions_dir)
            if schema_source == "auto-generated" else None,
        "label_png":              str(resolved.label_png) if resolved.label_png else None,
        "validation_mode":        "layout" if layout_only else "legacy",
        "layout_tolerance_mm":    float(layout_tolerance_mm) if layout_only else None,
        "page_index":             int(page_index),
    }

    payload = build_report_payload(
        product_code=product_code,
        candidate_pdf=candidate_pdf,
        template_pdf=template_pdf,
        region_config=config,
        summary=summary,
        region_scores=region_scores,
        candidate_extr=candidate_extr,
        overview_filename=overview_name,
        pipeline=pipeline,
        template_overview_filename=template_overview_name,
        template_extr=template_extr,
    )
    write_json_report(json_path, payload)
    write_html_report(html_path, payload)

    return RegionReportResult(
        product_code=product_code,
        template_version=resolved.version,
        schema_source=schema_source,
        config_path=config.source_path,
        template_pdf=template_pdf,
        template_json=resolved.template_json,
        template_png=resolved.label_png,
        region_names=tuple(r.name for r in config.regions),
        summary=summary,
        report_html=html_path,
        report_json=json_path,
        overview_png=overview_path,
        template_overview_png=template_overview_path,
    )
