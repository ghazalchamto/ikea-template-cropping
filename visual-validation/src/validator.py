"""
src/validator.py
================
Top-level public API — direct product-code lookup, no auto-matching.

Usage
-----
    from src.validator import LabelValidator

    v = LabelValidator(
        ground_truth_dir="data/templates/ground-truth-ikea-labels/ground_truth"
    )

    # See all available product codes
    v.list_templates()

    # Validate against a specific product template (exact lookup)
    result = v.validate(
        label_path   = "data/labels/my_label.pdf",
        product_code = "1C50-CAA",        # required
        output_dir   = "outputs/",
    )
    print(result.verdict)     # "VALID" or "INVALID"
    print(result.score)       # 0.0 – 1.0
    print(result.report_html) # open this in your browser

    # Batch
    results = v.validate_batch(
        label_paths  = ["a.pdf", "b.pdf"],
        product_code = "1C50-CAA",
        output_dir   = "outputs/",
    )
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np

from src.config.settings import ValidationConfig, DEFAULT_CONFIG
from src.core.alignment import preprocess, align
from src.core.annotator import (annotate, make_heatmap, make_grid_map,
                                 make_side_by_side, save_outputs)
from src.core.comparator import compare, ComparisonResult
from src.core.optional_zones import load_optional_zones, build_ignore_mask
from src.core.template_store import TemplateStore
from src.utils.ingestion import load_image
from src.utils.reporter import save_json, save_html

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s  %(name)s — %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    verdict:          str                  # "VALID" | "INVALID"
    score:            float                # 0.0 – 1.0
    product_code:     str
    template_version: str
    comparison:       ComparisonResult
    report_html:      Optional[str]
    report_json:      Optional[str]
    annotated_img:    Optional[str]
    heatmap_img:      Optional[str]
    grid_map_img:     Optional[str]

    def __str__(self) -> str:
        n_fail = sum(1 for t in self.comparison.tile_results if not t.passed)
        return (
            f"[{self.verdict}]  {self.product_code}/{self.template_version}  "
            f"score={self.score:.4f}  "
            f"regions={len(self.comparison.mismatch_regions)}  "
            f"failed_tiles={n_fail}"
        )


class LabelValidator:
    """
    Reference-based visual label validator.

    Always requires an explicit product_code — no auto-matching.
    This ensures you compare against exactly the right template every time.

    Parameters
    ----------
    ground_truth_dir : path to the ground_truth/ directory
                       e.g. "data/templates/ground-truth-ikea-labels/ground_truth"
    cfg              : ValidationConfig — all thresholds and tuning params
    """

    def __init__(
        self,
        ground_truth_dir: Union[str, Path],
        cfg: ValidationConfig = DEFAULT_CONFIG,
        optional_zones_dir: Optional[Union[str, Path]] = None,
    ):
        self._cfg   = cfg
        self._store = TemplateStore(ground_truth_dir=ground_truth_dir, cfg=cfg)
        self._optional_zones_dir = optional_zones_dir   # None = use default
        logger.info(
            "LabelValidator ready — %d template versions across %d products",
            len(self._store),
            len(self._store.product_codes()),
        )

    # ── Single validation ─────────────────────────────────────────────────────

    def validate(
        self,
        label_path:       Union[str, Path],
        product_code:     str,
        template_version: Optional[str]       = None,
        output_dir:       Optional[Union[str, Path]] = None,
        save_comparison:  bool                = True,
    ) -> ValidationResult:
        """
        Validate *label_path* against the approved template for *product_code*.

        Parameters
        ----------
        label_path       : PDF or image file to validate.
        product_code     : Exact product code, e.g. "1C50-CAA".
                           Use list_templates() to see available codes.
        template_version : e.g. "v1".  If None, the latest version is used.
        output_dir       : Where to write annotated images + reports.
                           Pass None to skip file output (scores still returned).
        save_comparison  : If True, save the side-by-side comparison strip.
        """
        label_path = Path(label_path)
        stem       = label_path.stem

        logger.info("=" * 60)
        logger.info("Validating: %s", label_path.name)
        logger.info("Template  : %s", product_code)

        # ── 1. Load template (exact lookup, no scanning) ──────────────────────
        template, version_used = self._store.get(product_code, template_version)
        logger.info("Template  : %s / %s", product_code, version_used)

        # ── 2. Load + preprocess test label ───────────────────────────────────
        raw       = load_image(label_path, dpi=self._cfg.render_dpi)
        processed = preprocess(raw, self._cfg)

        # ── 3. Align test → template ──────────────────────────────────────────
        aligned, align_ok = align(template, processed, self._cfg)

        # ── 4. Load optional zones + build ignore mask ────────────────────────
        zones = load_optional_zones(product_code, self._optional_zones_dir)
        H, W  = template.shape[:2]
        ignore_mask = build_ignore_mask(zones, H, W)
        if zones:
            logger.info(
                "Optional zones for %s: %d zone(s) will be excluded from scoring — %s",
                product_code, len(zones), [z.name for z in zones],
            )

        # ── 5. Deep comparison ────────────────────────────────────────────────
        result = compare(
            template, aligned, self._cfg,
            alignment_ok=align_ok,
            ignore_mask=ignore_mask,
            ignored_zones=zones,
        )

        verdict = "VALID" if result.is_valid else "INVALID"
        logger.info(
            "Result    : %s  score=%.4f  SSIM=%.4f  pixel=%.4f  "
            "edge=%.4f  tile_pass=%.4f  regions=%d  failed_tiles=%d",
            verdict,
            result.final_score,
            result.ssim_score,
            result.pixel_similarity,
            result.edge_similarity,
            result.tile_pass_rate,
            len(result.mismatch_regions),
            sum(1 for t in result.tile_results if not t.passed),
        )
        logger.info("=" * 60)

        # ── 6. Produce output files ───────────────────────────────────────────
        html_path = json_path = ann_path = hm_path = grid_path = None

        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            ann_img   = annotate(aligned, result, self._cfg)
            heat_img  = make_heatmap(aligned, result, self._cfg)
            grid_img  = make_grid_map(result, self._cfg)
            sbs_img   = make_side_by_side(template, aligned, result) if save_comparison else None

            paths = save_outputs(
                ann_img, heat_img, output_dir, stem=stem,
                grid_map=grid_img, side_by_side=sbs_img,
            )
            ann_path  = paths.get("annotated")
            hm_path   = paths.get("heatmap")
            grid_path = paths.get("grid_map")

            json_path = str(output_dir / f"{stem}_report.json")
            save_json(result, json_path, extra={
                "label_file":       str(label_path),
                "product_code":     product_code,
                "template_version": version_used,
            })

            html_path = str(output_dir / f"{stem}_report.html")
            save_html(
                result, ann_img, heat_img, html_path,
                label_name=stem,
                product_code=product_code,
                template_version=version_used,
                grid_map_img=grid_img,
                side_by_side_img=sbs_img,
            )

        return ValidationResult(
            verdict=verdict,
            score=result.final_score,
            product_code=product_code,
            template_version=version_used,
            comparison=result,
            report_html=html_path,
            report_json=json_path,
            annotated_img=ann_path,
            heatmap_img=hm_path,
            grid_map_img=grid_path,
        )

    # ── Batch ─────────────────────────────────────────────────────────────────

    def validate_batch(
        self,
        label_paths:      List[Union[str, Path]],
        product_code:     str,
        template_version: Optional[str]       = None,
        output_dir:       Optional[Union[str, Path]] = None,
        fail_fast:        bool                = False,
    ) -> List[ValidationResult]:
        """
        Validate a list of labels against the same product template.

        fail_fast : stop processing after the first INVALID result.
        """
        results: List[ValidationResult] = []
        for p in label_paths:
            try:
                r = self.validate(
                    p,
                    product_code=product_code,
                    template_version=template_version,
                    output_dir=output_dir,
                )
                results.append(r)
                if fail_fast and r.verdict == "INVALID":
                    logger.warning("fail_fast=True — stopping at first INVALID.")
                    break
            except Exception as exc:
                logger.error("Error validating %s: %s", p, exc, exc_info=True)
                raise

        n_valid   = sum(1 for r in results if r.verdict == "VALID")
        n_invalid = len(results) - n_valid
        logger.info(
            "Batch complete: %d total | %d VALID | %d INVALID",
            len(results), n_valid, n_invalid,
        )
        return results

    # ── Utility ───────────────────────────────────────────────────────────────

    def list_templates(self) -> None:
        """Print all available product codes and their template paths."""
        catalog = self._store.catalog()
        if not catalog:
            print("\nNo templates found — check your ground_truth_dir path.\n")
            return
        print(f"\n{'Product code':<30} {'Version':<8} {'Source file'}")
        print("─" * 90)
        for e in catalog:
            print(f"{e['product_code']:<30} {e['version']:<8} {e['path']}")
        print(f"\n  Total: {len(catalog)} template versions\n")
