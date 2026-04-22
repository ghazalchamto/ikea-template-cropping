"""
IKEA Label Validator package.

Responsible for all post-extraction checks:

  field_rules       – Value-level checks: regex, exact match, presence,
                      AI field checks.  Driven by label-rules/*.json schemas.

  layout_checker    – Spatial checks: is each field in the correct zone?
                      Are any elements outside the label boundary?

  overlap_detector  – Geometry checks: do any two bounding boxes intersect
                      unexpectedly?  Are barcode quiet zones clear?

  completeness      – Completeness + cross-field consistency: are all required
                      fields present?  Do DataMatrix dates match printed dates?

Quick usage:

    from validator import auto_validate, check_layout, detect_overlaps, check_completeness

    report,  shortlist  = auto_validate(extracted_label)
    layout              = check_layout(extracted_label, report.label_type_id)
    overlaps            = detect_overlaps(extracted_label)
    completeness        = check_completeness(extracted_label, report.label_type_id)
"""

from .field_rules       import auto_validate, validate, detect_label_type
from .layout_checker    import check_layout
from .overlap_detector  import detect_overlaps
from .completeness      import check_completeness
from .models            import (
    ValidationReport,
    RuleResult,
    TypeMatch,
    LayoutReport,
    LayoutIssue,
    OverlapReport,
    OverlapResult,
    CompletenessReport,
    MissingElement,
    ConsistencyIssue,
)

__all__ = [
    # field rules
    "auto_validate",
    "validate",
    "detect_label_type",
    # spatial
    "check_layout",
    "detect_overlaps",
    # completeness
    "check_completeness",
    # models
    "ValidationReport",
    "RuleResult",
    "TypeMatch",
    "LayoutReport",
    "LayoutIssue",
    "OverlapReport",
    "OverlapResult",
    "CompletenessReport",
    "MissingElement",
    "ConsistencyIssue",
]
