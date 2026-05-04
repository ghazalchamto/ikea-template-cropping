"""
src/core/region_aggregator.py
=============================
Convert per-region similarity scores into a final VALID / INVALID verdict.

This module is the **decision layer** sitting on top of
``region_comparator.py``. It is metric-free: it never opens a PDF, never
loads pixels, and never reruns SSIM/pixel/edge math. It only consumes
``RegionScore`` objects produced by the comparator and the matching
``RegionConfig`` returned by ``region_loader``.

Rules (in plain English)
------------------------
* Mandatory regions count toward the final score and verdict.
* Optional regions are reported but **never** affect the weighted score or verdict.
  Optional regions absent on the candidate (``candidate_absent`` in score details)
  still count as pass with combined 1.0 and are listed in ``skipped_optional_regions``.
* A mandatory region passes iff ``combined >= threshold``.
* Per-region threshold lookup order:
      1. ``region.extras["threshold"]`` (JSON-level override)
      2. ``ValidationConfig.region_default_threshold``
* Final score:
      ``Σ(weight * combined) / Σ(weight)``
  over **mandatory** regions only.
* Verdict is ``VALID`` iff:
      - every mandatory region passed, AND
      - final_score >= ``ValidationConfig.valid_score_threshold``.
  Otherwise ``INVALID``.

Edge cases
----------
* If the configuration has no mandatory regions, the loader has already
  rejected it (it requires at least one). We still defend here: empty
  mandatory set → ``INVALID`` with ``final_score=0.0``.
* If every mandatory region has weight 0, ``final_score=0.0`` and verdict
  is ``INVALID``. This is intentionally strict — silent zero-weight
  configs would otherwise pass with no real signal.
* If the comparator returns a score for a region not present in the
  config, the extra score is ignored (with a warning). If the config
  lists a mandatory region not present in the scores, the region is
  recorded as a hard FAIL with score 0.0 (defensive — should never happen
  when called via the standard pipeline since ``compare_extractions``
  enforces matched names).

Determinism
-----------
Pure function over the input lists. No randomness, no I/O.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.config.settings import DEFAULT_CONFIG, ValidationConfig
from src.core.region_comparator import RegionScore
from src.core.region_loader import Region, RegionConfig

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Result types
# ─────────────────────────────────────────────────────────────────────────────

# Status strings — kept as constants so callers can match without typos.
STATUS_PASS:     str = "PASS"
STATUS_FAIL:     str = "FAIL"
STATUS_OPTIONAL: str = "OPTIONAL"

VERDICT_VALID:   str = "VALID"
VERDICT_INVALID: str = "INVALID"


@dataclass(frozen=True)
class RegionVerdict:
    """Per-region decision after aggregation."""

    name:           str
    mandatory:      bool
    weight:         float
    combined_score: float
    threshold:      float
    passed:         bool         # True for optional regions (always)
    status:         str          # "PASS" | "FAIL" | "OPTIONAL"
    # Typed-validator metadata (optional; defaults preserve back-compat).
    region_type:    str = ""
    method:         str = ""

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "name":           self.name,
            "mandatory":      self.mandatory,
            "weight":         self.weight,
            "combined_score": round(float(self.combined_score), 4),
            "threshold":      round(float(self.threshold), 4),
            "passed":         self.passed,
            "status":         self.status,
        }
        if self.region_type:
            d["region_type"] = self.region_type
        if self.method:
            d["method"] = self.method
        return d


@dataclass(frozen=True)
class ValidationSummary:
    """Final verdict produced by ``aggregate_region_scores``."""

    verdict:          str           # "VALID" | "INVALID"
    final_score:      float         # weighted mean over mandatory regions
    global_threshold: float         # value final_score had to clear

    mandatory_passed: int
    mandatory_failed: int
    optional_count:   int
    total_regions:    int

    failed_regions:   List[str]     # mandatory failures, in input order
    ignored_regions:  List[str]     # optional regions, in input order
    warning_regions:  Tuple[str, ...]  # mandatory passed but combined < 1.0 (soft pass)
    skipped_optional_regions: Tuple[str, ...]  # optional with no candidate clip (layout)
    region_results:   List[RegionVerdict] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict":          self.verdict,
            "final_score":      round(float(self.final_score), 4),
            "global_threshold": round(float(self.global_threshold), 4),
            "mandatory_passed": self.mandatory_passed,
            "mandatory_failed": self.mandatory_failed,
            "optional_count":   self.optional_count,
            "total_regions":    self.total_regions,
            "failed_regions":   list(self.failed_regions),
            "ignored_regions":  list(self.ignored_regions),
            "warning_regions":  list(self.warning_regions),
            "skipped_optional_regions": list(self.skipped_optional_regions),
            "region_results":   [r.to_dict() for r in self.region_results],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_region_scores(
    region_scores:  List[RegionScore],
    region_schema:  RegionConfig,
    cfg:            ValidationConfig = DEFAULT_CONFIG,
) -> ValidationSummary:
    """
    Aggregate per-region scores into a final verdict.

    Parameters
    ----------
    region_scores : list of RegionScore from ``region_comparator``.
    region_schema : the ``RegionConfig`` those scores were computed against.
    cfg           : ``ValidationConfig`` providing default thresholds.

    Returns
    -------
    ValidationSummary
        Verdict, weighted final score, per-region results, counts, and the
        names of failed mandatory + ignored optional regions.
    """
    if region_schema is None:
        raise ValueError("aggregate_region_scores: region_schema must not be None")
    if region_scores is None:
        raise ValueError("aggregate_region_scores: region_scores must not be None")

    score_by_name: Dict[str, RegionScore] = {s.region_name: s for s in region_scores}

    # Warn on extras (scored regions not in the config). They are harmless
    # but indicate a drift between the config and the comparator inputs.
    extras = sorted(set(score_by_name) - {r.name for r in region_schema.regions})
    if extras:
        logger.warning(
            "aggregate_region_scores: %d score(s) not present in config — ignored: %s",
            len(extras), extras,
        )

    region_results: List[RegionVerdict] = []
    failed_regions:  List[str] = []
    ignored_regions: List[str] = []
    warning_list: List[str] = []

    sum_weight       = 0.0
    sum_weighted     = 0.0
    mandatory_passed = 0
    mandatory_failed = 0

    for region in region_schema.regions:
        threshold = _resolve_threshold(region, cfg)
        score_obj = score_by_name.get(region.name)

        if score_obj is None:
            if region.mandatory:
                logger.error(
                    "aggregate_region_scores: no score for mandatory region %r — FAIL.",
                    region.name,
                )
                combined = 0.0
                method = ""
            else:
                combined = 1.0
                method = "optional_no_score"
        else:
            combined = float(score_obj.combined)
            method = score_obj.method

        if region.mandatory:
            passed = combined >= threshold
            status = STATUS_PASS if passed else STATUS_FAIL

            if passed:
                mandatory_passed += 1
                if combined < 1.0 - 1e-12:
                    warning_list.append(region.name)
            else:
                mandatory_failed += 1
                failed_regions.append(region.name)

            sum_weight   += float(region.weight)
            sum_weighted += float(region.weight) * combined
        else:
            # Optional regions never fail and never affect the score.
            passed = True
            status = STATUS_OPTIONAL
            ignored_regions.append(region.name)

        region_results.append(RegionVerdict(
            name=region.name,
            mandatory=region.mandatory,
            weight=float(region.weight),
            combined_score=combined,
            threshold=float(threshold),
            passed=passed,
            status=status,
            region_type=region.type,
            method=method,
        ))

    # ── Final score ─────────────────────────────────────────────────────────
    if sum_weight > 0:
        final_score = sum_weighted / sum_weight
    else:
        # No mandatory weight → can't make a positive decision.
        final_score = 0.0
        logger.warning(
            "aggregate_region_scores: total mandatory weight is 0 — "
            "final_score forced to 0.0 (verdict will be INVALID)."
        )

    # ── Verdict ─────────────────────────────────────────────────────────────
    global_threshold = float(cfg.valid_score_threshold)
    all_passed       = (mandatory_failed == 0) and (mandatory_passed > 0)
    score_ok         = final_score >= global_threshold
    verdict          = VERDICT_VALID if (all_passed and score_ok) else VERDICT_INVALID

    skipped_optional: List[str] = []
    for r in region_schema.regions:
        if r.mandatory:
            continue
        sc = score_by_name.get(r.name)
        if sc is not None and bool(sc.details.get("candidate_absent")):
            skipped_optional.append(r.name)

    return ValidationSummary(
        verdict=verdict,
        final_score=float(final_score),
        global_threshold=global_threshold,
        mandatory_passed=mandatory_passed,
        mandatory_failed=mandatory_failed,
        optional_count=len(ignored_regions),
        total_regions=len(region_schema.regions),
        failed_regions=failed_regions,
        ignored_regions=ignored_regions,
        warning_regions=tuple(warning_list),
        skipped_optional_regions=tuple(skipped_optional),
        region_results=region_results,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Internal
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_threshold(region: Region, cfg: ValidationConfig) -> float:
    """
    Per-region threshold lookup chain:
        1. region.extras["threshold"] if present and a valid number
        2. cfg.region_default_threshold
    """
    raw = region.extras.get("threshold") if region.extras else None
    if raw is None:
        return float(cfg.region_default_threshold)

    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning(
            "Region %r has non-numeric 'threshold'=%r in extras — falling back to default.",
            region.name, raw,
        )
        return float(cfg.region_default_threshold)

    if not (0.0 <= value <= 1.0):
        logger.warning(
            "Region %r has out-of-range 'threshold'=%s in extras — clamping to default.",
            region.name, value,
        )
        return float(cfg.region_default_threshold)

    return value
