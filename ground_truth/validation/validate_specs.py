#!/usr/bin/env python3
"""
validate_specs.py
Validates the IKEA ground_truth spec repository for:
  1. JSON validity (all .json files parse)
  2. Schema conformance (templates, elements, rules, shared_specs)
  3. Duplicate IDs (template_ids, element_ids, rule_ids, spec_ids)
  4. Cross-reference integrity (shared_spec_ref, inherits_from, family references)
  5. Missing evidence (every rule and element must have a page number)
  6. Inventory/template consistency (every template_id in templates_index exists as a file)
  7. Family variant coverage (every variant_id in a family has a corresponding template file)
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
TEMPLATES_DIR = ROOT / "templates"
FAMILIES_DIR = ROOT / "families"
SHARED_SPECS_DIR = ROOT / "shared_specs"
REVIEW_DIR = ROOT / "review"
SCHEMA_DIR = ROOT / "schema"

errors: list[str] = []
warnings: list[str] = []


def err(msg: str) -> None:
    errors.append(msg)


def warn(msg: str) -> None:
    warnings.append(msg)


# ---------------------------------------------------------------------------
# 1. Load and parse all JSON files
# ---------------------------------------------------------------------------

def load_json(path: Path) -> Any | None:
    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        err(f"[JSON_PARSE] {path.name}: {e}")
        return None
    except FileNotFoundError:
        err(f"[FILE_MISSING] {path}")
        return None


def load_all_json_in(directory: Path) -> dict[str, Any]:
    results = {}
    if not directory.exists():
        err(f"[DIR_MISSING] Directory not found: {directory}")
        return results
    for p in sorted(directory.glob("*.json")):
        data = load_json(p)
        if data is not None:
            results[p.stem] = data
    return results


# ---------------------------------------------------------------------------
# 2. Collect known IDs from each domain
# ---------------------------------------------------------------------------

def collect_spec_ids(barcode_specs: dict) -> set[str]:
    ids = set()
    for spec in barcode_specs.get("specs", []):
        sid = spec.get("spec_id")
        if sid:
            if sid in ids:
                err(f"[DUPLICATE_SPEC_ID] spec_id '{sid}' appears more than once in barcode_specs.json")
            ids.add(sid)
    return ids


def collect_template_ids(templates: dict[str, Any]) -> dict[str, Any]:
    """Returns {template_id: template_data}"""
    by_id: dict[str, Any] = {}
    for stem, data in templates.items():
        tid = data.get("template_id")
        if not tid:
            err(f"[MISSING_FIELD] templates/{stem}.json: missing 'template_id'")
            continue
        if tid in by_id:
            err(f"[DUPLICATE_TEMPLATE_ID] template_id '{tid}' in {stem}.json duplicates an existing id")
        by_id[tid] = data
    return by_id


def collect_family_ids(families: dict[str, Any]) -> dict[str, Any]:
    by_id: dict[str, Any] = {}
    for stem, data in families.items():
        fid = data.get("family_id")
        if not fid:
            err(f"[MISSING_FIELD] families/{stem}.json: missing 'family_id'")
            continue
        by_id[fid] = data
    return by_id


# ---------------------------------------------------------------------------
# 3. Validate required top-level fields
# ---------------------------------------------------------------------------

REQUIRED_TEMPLATE_FIELDS = ["template_id", "display_name", "family", "source_evidence", "review_state"]
REQUIRED_ELEMENT_FIELDS = ["element_id", "label", "semantic_role", "type", "required", "content_type", "placement", "evidence"]
REQUIRED_RULE_FIELDS = ["rule_id", "rule_type", "scope", "condition", "target", "actions", "severity", "normalized_text", "evidence"]
REQUIRED_SPEC_FIELDS = ["spec_id", "spec_type", "standard", "evidence"]

VALID_RULE_TYPES = {
    "presence_rule", "conditional_layout_transform", "alignment_rule",
    "placement_rule", "resize_rule", "orientation_rule",
    "application_method_rule", "symbol_logic_rule"
}

VALID_SEVERITY = {"must", "should", "may"}


def validate_evidence(evidence: Any, context: str) -> None:
    if not isinstance(evidence, dict):
        err(f"[MISSING_EVIDENCE] {context}: 'evidence' must be an object")
        return
    if "page" not in evidence:
        err(f"[MISSING_EVIDENCE] {context}: evidence missing 'page'")
    elif not isinstance(evidence["page"], int):
        err(f"[BAD_EVIDENCE] {context}: evidence.page must be an integer, got {type(evidence['page']).__name__}")
    if "evidence_type" not in evidence:
        warn(f"[WEAK_EVIDENCE] {context}: evidence missing 'evidence_type'")


def validate_element(elem: dict, context: str, known_spec_ids: set[str]) -> str | None:
    eid = elem.get("element_id")
    if not eid:
        err(f"[MISSING_FIELD] {context}: element missing 'element_id'")
        return None
    ctx = f"{context}/element:{eid}"

    for field in REQUIRED_ELEMENT_FIELDS:
        if field not in elem:
            # Allow $ref_family_element as a stub
            if "$ref_family_element" in elem:
                return None
            err(f"[MISSING_FIELD] {ctx}: missing required field '{field}'")

    evidence = elem.get("evidence")
    if evidence:
        validate_evidence(evidence, ctx)
    else:
        if "$ref_family_element" not in elem:
            err(f"[MISSING_EVIDENCE] {ctx}: element missing 'evidence'")

    spec_ref = elem.get("shared_spec_ref")
    if spec_ref and spec_ref not in known_spec_ids:
        err(f"[BROKEN_REF] {ctx}: shared_spec_ref '{spec_ref}' not found in barcode_specs.json")

    return eid


def validate_rule(rule: dict, context: str) -> str | None:
    rid = rule.get("rule_id")
    if not rid:
        err(f"[MISSING_FIELD] {context}: rule missing 'rule_id'")
        return None
    ctx = f"{context}/rule:{rid}"

    for field in REQUIRED_RULE_FIELDS:
        if field not in rule:
            err(f"[MISSING_FIELD] {ctx}: missing required field '{field}'")

    rt = rule.get("rule_type")
    if rt and rt not in VALID_RULE_TYPES:
        err(f"[INVALID_VALUE] {ctx}: rule_type '{rt}' not in allowed set")

    sev = rule.get("severity")
    if sev and sev not in VALID_SEVERITY:
        err(f"[INVALID_VALUE] {ctx}: severity '{sev}' not in {{must, should, may}}")

    actions = rule.get("actions")
    if actions is not None and not isinstance(actions, list):
        err(f"[INVALID_VALUE] {ctx}: 'actions' must be a list")
    elif actions:
        for i, action in enumerate(actions):
            if "action_type" not in action:
                err(f"[MISSING_FIELD] {ctx}/action[{i}]: missing 'action_type'")
            if "description" not in action:
                warn(f"[WEAK_SPEC] {ctx}/action[{i}]: missing 'description'")

    evidence = rule.get("evidence")
    if evidence:
        validate_evidence(evidence, ctx)
    else:
        err(f"[MISSING_EVIDENCE] {ctx}: rule missing 'evidence'")

    return rid


def validate_template(data: dict, stem: str, known_spec_ids: set[str],
                       known_family_ids: set[str], known_template_ids: set[str]) -> None:
    ctx = f"templates/{stem}.json"

    for field in REQUIRED_TEMPLATE_FIELDS:
        if field not in data:
            err(f"[MISSING_FIELD] {ctx}: missing required field '{field}'")

    # Family reference
    fam = data.get("family")
    if fam and fam not in known_family_ids:
        err(f"[BROKEN_REF] {ctx}: family '{fam}' not found in families/")

    # inherits_from reference
    parent = data.get("inherits_from")
    if parent and parent not in known_template_ids and parent not in known_family_ids:
        warn(f"[BROKEN_REF] {ctx}: inherits_from '{parent}' not found in templates or families (may be forward reference)")

    # Source evidence
    se = data.get("source_evidence")
    if se:
        validate_evidence(se, f"{ctx}/source_evidence")

    # Elements
    element_ids: list[str] = []
    for elem in data.get("elements", []):
        if "$ref_family_element" in elem:
            continue
        eid = validate_element(elem, ctx, known_spec_ids)
        if eid:
            if eid in element_ids:
                err(f"[DUPLICATE_ELEMENT_ID] {ctx}: element_id '{eid}' appears more than once")
            element_ids.append(eid)

    # Rules
    rule_ids: list[str] = []
    for rule in data.get("rules", []):
        rid = validate_rule(rule, ctx)
        if rid:
            if rid in rule_ids:
                err(f"[DUPLICATE_RULE_ID] {ctx}: rule_id '{rid}' appears more than once")
            rule_ids.append(rid)

    # Review state
    rs = data.get("review_state", {})
    conf = rs.get("confidence")
    if conf is not None and not (0 <= conf <= 1):
        err(f"[INVALID_VALUE] {ctx}: review_state.confidence must be between 0 and 1")


# ---------------------------------------------------------------------------
# 4. Cross-check inventory vs templates
# ---------------------------------------------------------------------------

def validate_inventory(index_data: dict, known_template_ids: set[str]) -> None:
    ctx = "review/templates_index.json"
    templates_in_index = {t["template_id"] for t in index_data.get("templates", []) if "template_id" in t}

    for tid in templates_in_index:
        if tid not in known_template_ids:
            err(f"[INVENTORY_MISMATCH] {ctx}: '{tid}' listed in index but no matching templates/{tid}.json found")

    for tid in known_template_ids:
        if tid not in templates_in_index:
            warn(f"[INVENTORY_MISMATCH] templates/{tid}.json exists but '{tid}' not listed in templates_index.json")

    declared_total = index_data.get("total_templates")
    if declared_total is not None and declared_total != len(templates_in_index):
        warn(f"[INVENTORY_COUNT] {ctx}: total_templates={declared_total} but index contains {len(templates_in_index)} entries")


# ---------------------------------------------------------------------------
# 5. Cross-check family variant_ids vs actual template files
# ---------------------------------------------------------------------------

def validate_family_variants(families: dict[str, Any], known_template_ids: set[str]) -> None:
    for stem, data in families.items():
        fid = data.get("family_id", stem)
        for vid in data.get("variant_ids", []):
            if vid not in known_template_ids:
                err(f"[BROKEN_REF] families/{stem}.json: variant_id '{vid}' listed but no matching template file found")


# ---------------------------------------------------------------------------
# 6. Global rule/element ID uniqueness across all templates
# ---------------------------------------------------------------------------

def validate_global_uniqueness(templates_by_id: dict[str, Any]) -> None:
    all_rule_ids: dict[str, str] = {}
    for tid, data in templates_by_id.items():
        for rule in data.get("rules", []):
            rid = rule.get("rule_id")
            if rid:
                if rid in all_rule_ids:
                    warn(f"[GLOBAL_DUPLICATE_RULE_ID] rule_id '{rid}' appears in both '{all_rule_ids[rid]}' and '{tid}'")
                else:
                    all_rule_ids[rid] = tid


# ---------------------------------------------------------------------------
# 7. Validate shared_specs structure
# ---------------------------------------------------------------------------

def validate_shared_specs(barcode_specs: dict, known_spec_ids: set[str]) -> None:
    ctx = "shared_specs/barcode_specs.json"
    for spec in barcode_specs.get("specs", []):
        sid = spec.get("spec_id", "?")
        sctx = f"{ctx}/spec:{sid}"

        for field in REQUIRED_SPEC_FIELDS:
            if field not in spec:
                err(f"[MISSING_FIELD] {sctx}: missing required field '{field}'")

        evidence = spec.get("evidence")
        if evidence:
            validate_evidence(evidence, sctx)
        else:
            err(f"[MISSING_EVIDENCE] {sctx}: spec missing 'evidence'")

        spec_type = spec.get("spec_type", "")
        if spec_type in ("datamatrix_rectangular", "datamatrix_square"):
            if not spec.get("x_dim_variants"):
                warn(f"[WEAK_SPEC] {sctx}: DataMatrix spec has no x_dim_variants")
        elif spec_type in ("ean13", "ean13_80pct"):
            if not spec.get("height_variants"):
                warn(f"[WEAK_SPEC] {sctx}: EAN spec has no height_variants")
        elif spec_type == "itf":
            if not spec.get("size_variants"):
                warn(f"[WEAK_SPEC] {sctx}: ITF spec has no size_variants")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 60)
    print("IKEA Ground Truth Spec Validator")
    print("=" * 60)

    # Load all data
    barcode_data = load_json(SHARED_SPECS_DIR / "barcode_specs.json") or {}
    templates_raw = load_all_json_in(TEMPLATES_DIR)
    families_raw = load_all_json_in(FAMILIES_DIR)
    index_data = load_json(REVIEW_DIR / "templates_index.json") or {}

    known_spec_ids = collect_spec_ids(barcode_data)
    templates_by_id = collect_template_ids(templates_raw)
    families_by_id = collect_family_ids(families_raw)

    print(f"\nLoaded: {len(templates_raw)} template files, "
          f"{len(families_raw)} family files, "
          f"{len(known_spec_ids)} shared spec IDs")

    # Run validations
    print("\n[1] Validating shared specs...")
    validate_shared_specs(barcode_data, known_spec_ids)

    print("[2] Validating templates...")
    for stem, data in templates_raw.items():
        validate_template(
            data, stem,
            known_spec_ids,
            set(families_by_id.keys()),
            set(templates_by_id.keys())
        )

    print("[3] Validating inventory consistency...")
    validate_inventory(index_data, set(templates_by_id.keys()))

    print("[4] Validating family variant coverage...")
    validate_family_variants(families_raw, set(templates_by_id.keys()))

    print("[5] Checking global ID uniqueness...")
    validate_global_uniqueness(templates_by_id)

    # Report
    print("\n" + "=" * 60)
    if warnings:
        print(f"WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"  ⚠  {w}")

    if errors:
        print(f"\nERRORS ({len(errors)}):")
        for e in errors:
            print(f"  ✗  {e}")
        print(f"\n{'=' * 60}")
        print(f"RESULT: FAILED — {len(errors)} error(s), {len(warnings)} warning(s)")
        return 1
    else:
        print(f"\n{'=' * 60}")
        print(f"RESULT: PASSED — 0 errors, {len(warnings)} warning(s)")
        return 0


if __name__ == "__main__":
    sys.exit(main())
