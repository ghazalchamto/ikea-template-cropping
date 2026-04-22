"""
Rule-based label field validator.

Two-phase pipeline
──────────────────
1. detect_label_type(label)
   Scores every rule schema in label-rules/ against the extracted fields and
   returns the best-matching label type plus a ranked shortlist.

2. validate(label, schema)
   Runs every validation_rule in the selected schema and returns a
   ValidationReport with per-rule pass / fail / skip results and an overall
   compliance score.

Supported rule checks
─────────────────────
  regex_match       – field value must match ``pattern``
  exact_match       – field value must equal ``value`` (accent-tolerant)
  present           – field must be extracted (non-None, non-empty)
  ai_field_present  – DataMatrix must contain AI(``ai``)
  range_mm          – skipped (physical mm cannot be measured from pixels alone)
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Optional

from extractor.models import ExtractedLabel, ExtractedField
from .models import RuleResult, ValidationReport, TypeMatch

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

# label-rules/ lives at repo root — one level up from this file's package
_RULES_DIR = Path(__file__).resolve().parent.parent / "label-rules"


# ── Schema loader ─────────────────────────────────────────────────────────────

def _load_schemas() -> dict[str, dict]:
    """Return {label_type_id: schema_dict} for all JSON files in label-rules/."""
    schemas: dict[str, dict] = {}
    if not _RULES_DIR.exists():
        logger.warning("label-rules/ directory not found at %s", _RULES_DIR)
        return schemas
    for path in sorted(_RULES_DIR.glob("*.json")):
        if path.stem == "index":
            continue
        try:
            with open(path, encoding="utf-8") as f:
                obj = json.load(f)
            lid = obj.get("label_type_id", path.stem)
            schemas[lid] = obj
        except Exception as exc:
            logger.warning("Could not load rule schema %s: %s", path, exc)
    return schemas


_SCHEMAS: dict[str, dict] = _load_schemas()


# ── Field resolver ────────────────────────────────────────────────────────────

_FIELD_MAP: dict[str, str] = {
    "product_name":           "product_name",
    "article_number":         "article_number",
    "article_code":           "article_code",
    "internal_item_number":   "internal_item_number",
    "origin_text":            "origin_text",
    "supplier_name":          "supplier_name",
    "supplier_address":       "supplier_address",
    "copyright_notice":       "copyright_notice",
    "plant_identifier":       "plant_identifier",
    "human_readable_date":    "human_readable_date",
    "custom_identifier":      "custom_identifier",
    "date_alpha_code":        "date_alpha_code",
    "date_stamp_yyww":        "date_stamp",
    "date_stamp":             "date_stamp",
    "date_numeric_prefix":    "date_numeric_prefix",
    "dimensions_metric":      "product_dimensions_metric",
    "dimensions_imperial":    "product_dimensions_imperial",
    "product_dimensions_metric":   "product_dimensions_metric",
    "product_dimensions_imperial": "product_dimensions_imperial",
    "gross_weight":           "gross_weight",
    "net_weight":             "net_weight",
    "weight_compact":         "gross_weight",
    "weight":                 "weight",
    "quantity_multiplier":    "quantity_multiplier",
    "ikea_logo":              "ikea_logo",
    "package_number":         "package_number",
    "package_type":           "package_type",
    "compliance_marks":       "compliance_marks",
    "legal_symbols":          "compliance_marks",
    "age_rating":             "age_rating",
    "copy_count_inline":      "copy_count_inline",
    "copy_count":             "copy_count",
    "address_block":          "address_block",
    "ai_240":                 "ai_240",
    "ai_11":                  "ai_11",
    "ai_13":                  "ai_13",
    "ai_10":                  "ai_10",
}


def _get_field_value(label: ExtractedLabel, field_name: str) -> Optional[str]:
    attr = _FIELD_MAP.get(field_name, field_name)
    obj = getattr(label, attr, None)
    if obj is None:
        return None
    if isinstance(obj, ExtractedField):
        return obj.value
    if hasattr(obj, "raw_data"):
        return obj.raw_data
    return str(obj) if obj else None


def _is_field_present(label: ExtractedLabel, field_name: str) -> bool:
    v = _get_field_value(label, field_name)
    return bool(v and v.strip())


# ── Accent-tolerant string comparison ─────────────────────────────────────────

def _normalise(s: str) -> str:
    return unicodedata.normalize("NFKD", s).casefold()


# ── Label type detection ──────────────────────────────────────────────────────

def detect_label_type(
    label: ExtractedLabel,
    schemas: Optional[dict[str, dict]] = None,
    top_k: int = 5,
) -> tuple[Optional[TypeMatch], list[TypeMatch]]:
    """
    Score every known label type against the extracted fields.

    Scoring heuristic
    -----------------
    - +1 point for each required_field that is present
    - +0.4 for each optional_field that is present
    - ±1 / ±0.5 barcode bonus / penalty
    - Score normalised to 0–1

    Returns (best_match | None, ranked_shortlist[0:top_k])
    """
    if schemas is None:
        schemas = _SCHEMAS

    if not schemas:
        return None, []

    ranked: list[TypeMatch] = []

    for lid, schema in schemas.items():
        req_fields  = schema.get("required_fields", [])
        opt_fields  = schema.get("optional_fields", [])

        has_itf = label.itf14_barcode is not None
        has_dm  = label.datamatrix_barcode is not None
        has_ean = label.ean_barcode is not None

        schema_has_itf = schema.get("itf_barcode") is not None
        schema_has_ean = schema.get("ean13_barcode") is not None
        schema_has_dm  = bool(schema.get("datamatrix_variants"))

        score = 0.0
        max_score = max(len(req_fields) * 1.0 + len(opt_fields) * 0.4 + 3, 1)

        for rf in req_fields:
            if _is_field_present(label, rf):
                score += 1.0
        for of_ in opt_fields:
            if _is_field_present(label, of_):
                score += 0.4

        if schema_has_itf and has_itf:
            score += 1.0
        elif schema_has_itf and not has_itf:
            score -= 0.5
        if schema_has_dm and has_dm:
            score += 1.0
        elif schema_has_dm and not has_dm:
            score -= 0.5
        if schema_has_ean and has_ean:
            score += 1.0
        elif schema_has_ean and not has_ean:
            score -= 0.5

        normalised = max(0.0, score / max_score)
        ranked.append(TypeMatch(
            label_type_id=lid,
            label_type_name=schema.get("label_type_name", lid),
            score=round(normalised, 3),
        ))

    ranked.sort(key=lambda m: m.score, reverse=True)
    best = ranked[0] if ranked else None
    return best, ranked[:top_k]


# ── Rule evaluation ───────────────────────────────────────────────────────────

def _eval_rule(label: ExtractedLabel, rule: dict) -> RuleResult:
    rid   = rule.get("rule_id", "?")
    fld   = rule.get("field", "")
    check = rule.get("check", "")
    sev   = rule.get("severity", "warning")

    def _result(status, extracted, expected, msg):
        return RuleResult(
            rule_id=rid, field=fld, check=check, severity=sev,
            status=status, extracted_value=extracted,
            expected=expected, message=msg,
        )

    if check == "present":
        v = _get_field_value(label, fld)
        if v and v.strip():
            return _result("pass", v, None, f"Field '{fld}' is present.")
        return _result("fail", None, None, f"Field '{fld}' not found in extraction.")

    if check == "ai_field_present":
        ai_code = str(rule.get("ai", ""))
        ai_attr_map = {"240": "ai_240", "11": "ai_11", "13": "ai_13", "10": "ai_10"}
        attr = ai_attr_map.get(ai_code)
        if attr is None:
            return _result("skip", None, None, f"Unknown AI code {ai_code!r} — skipped.")
        ai_fld = getattr(label, attr, None)
        if ai_fld and ai_fld.value:
            return _result("pass", ai_fld.value, None,
                           f"AI({ai_code}) present: {ai_fld.value!r}")
        if label.datamatrix_content:
            ai_fields = label.datamatrix_content.ai_fields or {}
            if ai_code in ai_fields:
                raw_ai = ai_fields[ai_code]
                val = raw_ai.get("value") if isinstance(raw_ai, dict) else getattr(raw_ai, "value", None)
                if val:
                    return _result("pass", str(val), None,
                                   f"AI({ai_code}) found in DataMatrix: {val!r}")
        if label.datamatrix_barcode is None:
            return _result("not_found", None, f"AI({ai_code})",
                           "DataMatrix not decoded — cannot verify AI field.")
        return _result("fail", None, f"AI({ai_code})",
                       f"DataMatrix decoded but AI({ai_code}) is missing.")

    if check == "regex_match":
        pattern = rule.get("pattern", "")
        v = _get_field_value(label, fld)
        if v is None:
            return _result("not_found", None, pattern,
                           f"Field '{fld}' not extracted — cannot check pattern.")
        try:
            m = re.fullmatch(pattern, v.strip())
        except re.error as e:
            return _result("skip", v, pattern, f"Invalid regex in rule {rid}: {e}")
        if m:
            return _result("pass", v, pattern, f"Value {v!r} matches pattern.")
        return _result("fail", v, pattern, f"Value {v!r} does not match /{pattern}/.")

    if check == "exact_match":
        expected = str(rule.get("value", ""))
        v = _get_field_value(label, fld)
        if v is None:
            return _result("not_found", None, expected,
                           f"Field '{fld}' not extracted — cannot verify exact value.")
        if _normalise(v.strip()) == _normalise(expected):
            return _result("pass", v, expected, f"Value matches {expected!r}.")
        return _result("fail", v, expected, f"Expected {expected!r}, got {v!r}.")

    if check == "range_mm":
        return _result("skip", None,
                       f"{rule.get('min')}–{rule.get('max')} mm",
                       "Physical mm measurement not available from raster image — skipped.")

    return _result("skip", None, None, f"Unknown check type {check!r} — skipped.")


# ── Public API ────────────────────────────────────────────────────────────────

def validate(
    label: ExtractedLabel,
    schema: Optional[dict] = None,
    label_type_id: Optional[str] = None,
    schemas: Optional[dict[str, dict]] = None,
) -> ValidationReport:
    """
    Validate ``label`` against a rule schema.

    At least one of ``schema`` or ``label_type_id`` must be provided.
    If neither is given, ``detect_label_type`` is called and the best match used.
    """
    if schemas is None:
        schemas = _SCHEMAS

    if schema is None:
        if label_type_id:
            schema = schemas.get(label_type_id)
            if schema is None:
                logger.warning("Schema '%s' not found in rules directory.", label_type_id)
                schema = {}
        else:
            best, _ = detect_label_type(label, schemas)
            if best:
                schema = schemas.get(best.label_type_id, {})
                label_type_id = best.label_type_id
            else:
                schema = {}

    lid   = schema.get("label_type_id",   label_type_id or "unknown")
    lname = schema.get("label_type_name", lid)

    best, _ = detect_label_type(label, {lid: schema} if schema else {})
    match_conf = best.score if best else 0.0

    rules = schema.get("validation_rules", [])
    results: list[RuleResult] = []
    for rule in rules:
        results.append(_eval_rule(label, rule))

    return ValidationReport(
        label_type_id=lid,
        label_type_name=lname,
        match_confidence=match_conf,
        detection_score=match_conf,
        results=results,
    )


def auto_validate(
    label: ExtractedLabel,
    schemas: Optional[dict[str, dict]] = None,
) -> tuple[ValidationReport, list[TypeMatch]]:
    """
    Convenience wrapper: detect label type and validate in one call.

    Returns (ValidationReport for best match, ranked shortlist[0:5])
    """
    if schemas is None:
        schemas = _SCHEMAS
    best, shortlist = detect_label_type(label, schemas)
    if best is None:
        empty = ValidationReport(
            label_type_id="unknown",
            label_type_name="Unknown",
            match_confidence=0.0,
            detection_score=0.0,
        )
        return empty, []
    schema = schemas.get(best.label_type_id, {})
    report = validate(label, schema=schema, schemas=schemas)
    return report, shortlist
