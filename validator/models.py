"""
Validation result data classes.

All validator modules return instances of these types so that app.py
has a single, consistent shape to render regardless of which check produced
the result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ── Field-level rule result (from field_rules.py) ─────────────────────────────

@dataclass
class RuleResult:
    rule_id: str
    field: str
    check: str
    severity: str           # "error" | "warning" | "info"
    status: str             # "pass" | "fail" | "skip" | "not_found"
    extracted_value: Optional[str]
    expected: Optional[str]
    message: str

    def to_dict(self) -> dict:
        return {
            "rule_id":          self.rule_id,
            "field":            self.field,
            "check":            self.check,
            "severity":         self.severity,
            "status":           self.status,
            "extracted_value":  self.extracted_value,
            "expected":         self.expected,
            "message":          self.message,
        }


@dataclass
class TypeMatch:
    label_type_id: str
    label_type_name: str
    score: float            # 0–1 presence score


@dataclass
class ValidationReport:
    label_type_id: str
    label_type_name: str
    match_confidence: float
    detection_score: float
    results: list[RuleResult] = field(default_factory=list)

    @property
    def errors(self) -> list[RuleResult]:
        return [r for r in self.results if r.severity == "error" and r.status == "fail"]

    @property
    def warnings(self) -> list[RuleResult]:
        return [r for r in self.results if r.severity == "warning" and r.status == "fail"]

    @property
    def passed(self) -> list[RuleResult]:
        return [r for r in self.results if r.status == "pass"]

    @property
    def compliance_score(self) -> float:
        """Weighted score: errors count double. Returns 1.0 when all rules pass."""
        if not self.results:
            return 0.0
        runnable = [r for r in self.results if r.status != "skip"]
        if not runnable:
            return 1.0
        weights = {"error": 2.0, "warning": 1.0, "info": 0.5}
        total_w = sum(weights.get(r.severity, 1.0) for r in runnable)
        pass_w  = sum(
            weights.get(r.severity, 1.0) for r in runnable if r.status == "pass"
        )
        return round(pass_w / total_w, 3) if total_w else 1.0

    def to_dict(self) -> dict:
        return {
            "label_type_id":    self.label_type_id,
            "label_type_name":  self.label_type_name,
            "match_confidence": round(self.match_confidence, 3),
            "detection_score":  round(self.detection_score, 3),
            "compliance_score": self.compliance_score,
            "n_pass":    len(self.passed),
            "n_error":   len(self.errors),
            "n_warning": len(self.warnings),
            "n_skip":    sum(1 for r in self.results if r.status == "skip"),
            "results":   [r.to_dict() for r in self.results],
        }


# ── Layout issue (from layout_checker.py) ────────────────────────────────────

@dataclass
class LayoutIssue:
    """A field found outside its expected zone or violating a position rule."""
    field: str
    severity: str               # "error" | "warning"
    expected_zone: Optional[str]
    actual_zone: Optional[str]
    message: str
    bbox: Optional[tuple]       # (x, y, w, h) in pixels — for UI highlight

    def to_dict(self) -> dict:
        return {
            "field":          self.field,
            "severity":       self.severity,
            "expected_zone":  self.expected_zone,
            "actual_zone":    self.actual_zone,
            "message":        self.message,
            "bbox":           list(self.bbox) if self.bbox else None,
        }


@dataclass
class LayoutReport:
    issues: list[LayoutIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[LayoutIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[LayoutIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    def to_dict(self) -> dict:
        return {
            "n_error":   len(self.errors),
            "n_warning": len(self.warnings),
            "issues":    [i.to_dict() for i in self.issues],
        }


# ── Overlap result (from overlap_detector.py) ────────────────────────────────

@dataclass
class OverlapResult:
    """Two extracted elements whose bounding boxes intersect unexpectedly."""
    field_a: str
    field_b: str
    overlap_ratio: float        # fraction of the smaller box that is covered
    severity: str               # "warning" (<25%) | "error" (≥25%)
    bbox_intersection: tuple    # (x, y, w, h) of the overlapping region
    message: str

    def to_dict(self) -> dict:
        return {
            "field_a":           self.field_a,
            "field_b":           self.field_b,
            "overlap_ratio":     round(self.overlap_ratio, 3),
            "severity":          self.severity,
            "bbox_intersection": list(self.bbox_intersection),
            "message":           self.message,
        }


@dataclass
class OverlapReport:
    overlaps: list[OverlapResult] = field(default_factory=list)

    @property
    def errors(self) -> list[OverlapResult]:
        return [o for o in self.overlaps if o.severity == "error"]

    @property
    def warnings(self) -> list[OverlapResult]:
        return [o for o in self.overlaps if o.severity == "warning"]

    def to_dict(self) -> dict:
        return {
            "n_error":   len(self.errors),
            "n_warning": len(self.warnings),
            "overlaps":  [o.to_dict() for o in self.overlaps],
        }


# ── Completeness result (from completeness.py) ───────────────────────────────

@dataclass
class MissingElement:
    element: str                # field name or zone name
    element_type: str           # "field" | "zone" | "barcode" | "consistency"
    severity: str               # "error" | "warning"
    message: str

    def to_dict(self) -> dict:
        return {
            "element":      self.element,
            "element_type": self.element_type,
            "severity":     self.severity,
            "message":      self.message,
        }


@dataclass
class ConsistencyIssue:
    """A cross-field mismatch (e.g. DataMatrix date ≠ human-readable date)."""
    field_a: str
    field_b: str
    value_a: Optional[str]
    value_b: Optional[str]
    severity: str
    message: str

    def to_dict(self) -> dict:
        return {
            "field_a":   self.field_a,
            "field_b":   self.field_b,
            "value_a":   self.value_a,
            "value_b":   self.value_b,
            "severity":  self.severity,
            "message":   self.message,
        }


@dataclass
class CompletenessReport:
    missing: list[MissingElement] = field(default_factory=list)
    consistency_issues: list[ConsistencyIssue] = field(default_factory=list)

    @property
    def errors(self) -> list:
        return [m for m in self.missing if m.severity == "error"] + \
               [c for c in self.consistency_issues if c.severity == "error"]

    @property
    def warnings(self) -> list:
        return [m for m in self.missing if m.severity == "warning"] + \
               [c for c in self.consistency_issues if c.severity == "warning"]

    def to_dict(self) -> dict:
        return {
            "n_missing":             len(self.missing),
            "n_consistency_issues":  len(self.consistency_issues),
            "missing":               [m.to_dict() for m in self.missing],
            "consistency_issues":    [c.to_dict() for c in self.consistency_issues],
        }
