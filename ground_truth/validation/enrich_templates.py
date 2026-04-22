#!/usr/bin/env python3
"""
enrich_templates.py
Flattens $ref_family_element stubs in all templates and adds visual/spatial
enrichment (font, color, size_spec, placement anchors, validation_targets).
Run once; idempotent — skips elements already fully defined.
"""
import json
from pathlib import Path
from copy import deepcopy

ROOT = Path(__file__).parent.parent
TEMPLATES_DIR = ROOT / "templates"
FAMILIES_DIR = ROOT / "families"

# ---------------------------------------------------------------------------
# Visual enrichment data keyed by element_id
# Derived from PDF pages 2-5 and IKEA brand guidelines
# ---------------------------------------------------------------------------

ELEMENT_VISUAL_SPECS = {
    "product_name": {
        "font": {
            "family": "IKEA proprietary sans-serif",
            "weight": "bold",
            "style": "normal",
            "case": "upper",
            "size_mm_approx": 3.0,
            "note": "1R10 uses ~3mm (PNAME_SM); 1R20 uses ~4mm (PNAME_MD) — see element_variants.json"
        },
        "color": { "text_color": "#000000", "background_color": "transparent" },
        "size_spec": {
            "width_mm": "variable — expands to fill left_identity zone width",
            "height_mm_approx": 4
        },
        "placement": {
            "zone": "left_identity",
            "alignment": "left",
            "relative_to": "label_top_left",
            "position_notes": "First (topmost) element in left_identity zone"
        }
    },
    "article_number": {
        "font": {
            "family": "IKEA proprietary sans-serif",
            "weight": "regular",
            "style": "normal",
            "case": "mixed",
            "size_mm_approx": 2.5,
            "note": "1R10 uses ARTNO_SM (no background); 1R20_04x uses ARTNO_MD (grey bg); 1R20_075x uses ARTNO_LG — see element_variants.json"
        },
        "color": {
            "text_color": "#000000",
            "background_color": "variant_dependent",
            "note": "Background depends on ARTNO variant: transparent (XS/SM), grey (MD/LG), black (XL)"
        },
        "size_spec": { "width_mm_approx": 18, "height_mm_approx": 3 },
        "placement": {
            "zone": "left_identity",
            "alignment": "left",
            "relative_to": "product_name",
            "margin_below_product_name_mm": 1,
            "position_notes": "Directly below product_name"
        }
    },
    "article_code": {
        "font": {
            "family": "IKEA proprietary sans-serif",
            "weight": "regular",
            "style": "normal",
            "case": "numeric",
            "size_mm_approx": 1.8
        },
        "color": { "text_color": "#000000", "background_color": "transparent" },
        "size_spec": { "width_mm_approx": 10, "height_mm_approx": 2 },
        "placement": {
            "zone": "left_identity",
            "alignment": "left",
            "relative_to": "article_number",
            "margin_below_article_number_mm": 0.5,
            "position_notes": "Directly below article_number, smaller font"
        }
    },
    "ikea_logo": {
        "font": {
            "family": "IKEA proprietary sans-serif",
            "weight": "bold",
            "case": "upper",
            "note": "Logo asset — not a text render; use LOGO_LG variant for all current templates"
        },
        "color": {
            "text_color": "#000000",
            "background_color": "#FFD100",
            "border": "solid_rectangle",
            "note": "IKEA corporate yellow; may render black-on-white in monochrome"
        },
        "size_spec": {
            "width_mm_approx": 20,
            "height_mm_approx": 15,
            "variant_ref": "LOGO_LG",
            "note": "All current templates use LOGO_LG variant — see element_variants.json"
        },
        "placement": {
            "zone": "left_identity",
            "alignment": "left",
            "relative_to": "article_code",
            "margin_below_article_code_mm": 1,
            "position_notes": "Below article_code; in 1R20 may appear in a dedicated left strip"
        }
    },
    "datamatrix_block": {
        "font": {
            "ai_text": {
                "family": "monospace",
                "weight": "regular",
                "size_mm_approx": 1.5,
                "note": "Human-readable AI lines printed below or beside the matrix symbol"
            }
        },
        "color": { "text_color": "#000000", "background_color": "#FFFFFF", "note": "High contrast required for scanner readability" },
        "size_spec": {
            "note": "Determined by x-dim variant — see shared_specs/barcode_specs.json DM_RECT or DM_SQ",
            "x_dim_0_4_typical_mm": "6.4×19.2 to 7.2×20",
            "x_dim_0_5_typical_mm": "8×24 to 9×25",
            "x_dim_0_75_typical_mm": "12×36 to 13.5×37.5"
        },
        "placement": {
            "zone": "center_barcode",
            "alignment": "center",
            "relative_to": "ikea_logo",
            "position_notes": "Immediately right of left_identity zone; human-readable AI lines printed below the matrix symbol"
        }
    },
    "address_block": {
        "font": {
            "family": "IKEA proprietary sans-serif",
            "weight": "regular",
            "style": "normal",
            "case": "mixed",
            "size_mm_approx": 1.5,
            "line_count": 4,
            "note": "Multi-line block; 1R20 variants include additional AI fields and PI number making it 6-8 lines"
        },
        "color": { "text_color": "#000000", "background_color": "transparent" },
        "size_spec": {
            "width_mm_approx": 30,
            "height_mm_approx": 10,
            "note": "1R20 address block is taller due to additional lines"
        },
        "placement": {
            "zone": "center_address",
            "alignment": "left",
            "relative_to": "datamatrix_block",
            "position_notes": "Immediately right of DataMatrix block"
        },
        "content_structure": {
            "lines": [
                { "line": 1, "content": "Company name", "example": "IKEA of Sweeden AB" },
                { "line": 2, "content": "Address", "example": "SE – 34381 Älmhult" },
                { "line": 3, "content": "Copyright", "example": "© Inter IKEA Systems" },
                { "line": 4, "content": "Year and URL", "example": "B.V. 2022  IKEA.com" }
            ]
        }
    },
    "origin_text": {
        "font": {
            "family": "IKEA proprietary sans-serif",
            "weight": "regular",
            "style": "normal",
            "case": "mixed",
            "size_mm_approx": 1.5
        },
        "color": { "text_color": "#000000", "background_color": "transparent" },
        "size_spec": { "width_mm_approx": 30, "height_mm_approx": 2 },
        "placement": {
            "zone": "center_address",
            "alignment": "left",
            "relative_to": "address_block",
            "margin_below_address_mm": 0.5,
            "position_notes": "Below address_block"
        },
        "allowed_values": [
            "Made in People's Republic of China",
            "Assembled in USA from Imported Materials",
            "Made in Vietnam",
            "Made in Poland",
            "Made in India"
        ],
        "allowed_patterns": ["^(Made in|Assembled in USA from Imported Materials|Assembled in) .+$"]
    },
    "compliance_block": {
        "font": {
            "family": "IKEA proprietary sans-serif",
            "weight": "regular",
            "style": "normal",
            "case": "mixed",
            "size_mm_approx": 1.2,
            "note": "Small text; multi-column or multi-line block"
        },
        "color": { "text_color": "#000000", "background_color": "transparent" },
        "size_spec": { "width_mm_approx": 20, "height_mm_approx": 15 },
        "placement": {
            "zone": "right_compliance",
            "alignment": "center",
            "relative_to": "origin_text",
            "position_notes": "Right of address/origin zone; rotated 90° CW in 1R20 variants"
        },
        "static_content": {
            "en": "US EPA TSCA title VI compliant",
            "fr_partial": "Conforme au US EPA TSCA titre VI TFEF",
            "abbreviation": "ULEF"
        }
    },
    "patents_block": {
        "font": {
            "family": "IKEA proprietary sans-serif",
            "weight": "regular",
            "style": "normal",
            "case": "mixed",
            "size_mm_approx": 1.0
        },
        "color": { "text_color": "#000000", "background_color": "transparent" },
        "size_spec": { "width_mm_approx": 8, "height_mm_approx": 6 },
        "placement": {
            "zone": "right_patents",
            "alignment": "center",
            "relative_to": "compliance_block",
            "position_notes": "Right of or adjacent to compliance block; orientation matches compliance block"
        }
    },
    "abcd_alpha_block": {
        "font": {
            "family": "IKEA proprietary sans-serif",
            "weight": "bold",
            "style": "normal",
            "case": "upper",
            "size_mm_approx": 4.0,
            "note": "Large alpha identifier; exact semantic meaning TBD — see open_questions.json OQ-001"
        },
        "color": { "text_color": "#000000", "background_color": "transparent" },
        "size_spec": { "width_mm_approx": 12, "height_mm_approx": 5 },
        "placement": {
            "zone": "right_date",
            "alignment": "center",
            "relative_to": "patents_block",
            "position_notes": "Above DATE: YYWW in the date zone"
        },
        "allowed_patterns": ["^[A-Z]{1,4}$"]
    },
    "date_stamp": {
        "font": {
            "family": "IKEA proprietary sans-serif",
            "weight": "bold",
            "style": "normal",
            "case": "upper",
            "size_mm_approx": 3.0,
            "note": "DATE: label in regular weight; YYWW value in bold"
        },
        "color": { "text_color": "#000000", "background_color": "transparent" },
        "size_spec": {
            "width_mm_collapsed": 7,
            "width_mm_collapsed_1R20": 5,
            "width_mm_expanded": 22.5,
            "height_mm_approx": 5,
            "note": "Expands from collapsed to 22.5mm when DATE is present per family rules"
        },
        "placement": {
            "zone": "right_date",
            "alignment": "left",
            "relative_to": "abcd_alpha_block",
            "position_notes": "Below ABCD block; YYWW must appear immediately after DATE: label with no gap"
        },
        "content_rules": {
            "format": "DATE: YYWW",
            "yyww_pattern": "^(2[0-9]|[3-9][0-9])(0[1-9]|[1-4][0-9]|5[0-3])$",
            "note": "YY = 2-digit year (20-99); WW = ISO week number 01-53"
        }
    },
    "numeric_suffix_1234": {
        "font": {
            "family": "IKEA proprietary sans-serif",
            "weight": "regular",
            "style": "normal",
            "case": "numeric",
            "size_mm_approx": 2.0
        },
        "color": { "text_color": "#000000", "background_color": "transparent" },
        "size_spec": { "width_mm_approx": 8, "height_mm_approx": 3 },
        "placement": {
            "zone": "right_date",
            "alignment": "left",
            "relative_to": "date_stamp",
            "position_notes": "Below YYWW value in date zone"
        },
        "allowed_patterns": ["^\\d{4}$"],
        "note": "Exact semantic meaning unclear — see open_questions.json OQ-005"
    },
    "copy_count": {
        "font": {
            "family": "IKEA proprietary sans-serif",
            "weight": "bold",
            "style": "normal",
            "case": "numeric",
            "size_mm_approx": 6.0,
            "note": "Single large digit rotated 90° CW"
        },
        "color": { "text_color": "#000000", "background_color": "transparent" },
        "size_spec": { "width_mm_approx": 5, "height_mm_approx": 8 },
        "placement": {
            "zone": "right_copy_count",
            "alignment": "center",
            "relative_to": "right_edge_of_label",
            "position_notes": "Far right vertical strip; rotated 90° CW so digit reads from bottom to top"
        },
        "allowed_patterns": ["^\\d+$"]
    },
    "legal_symbols_column": {
        "color": { "text_color": "#000000", "background_color": "transparent" },
        "size_spec": {
            "width_mm_approx": 8,
            "height_mm_approx": 25,
            "note": "Approximate — cell count and individual symbol dimensions not fully legible at available PDF resolution"
        },
        "placement": {
            "zone": "right_legal_symbols",
            "alignment": "center",
            "relative_to": "center_address",
            "position_notes": "Column of rotated symbol cells; each cell rotated 90° CW"
        }
    },
    "abcd_yyww_date_strip": {
        "font": {
            "family": "IKEA proprietary sans-serif",
            "weight": "bold",
            "style": "normal",
            "case": "mixed",
            "size_mm_approx": 2.5,
            "note": "Combined ABCD-YYWW-1234 rendered as single vertical strip"
        },
        "color": { "text_color": "#000000", "background_color": "transparent" },
        "size_spec": { "width_mm_approx": 6, "height_mm_approx": 20 },
        "placement": {
            "zone": "far_right_strip",
            "alignment": "center",
            "relative_to": "right_edge_of_label",
            "position_notes": "Far right vertical strip rotated 90° CW; reads ABCD-YYWW-1234 from bottom to top"
        },
        "content_rules": {
            "format": "ABCD-YYWW-1234",
            "pattern": "^[A-Z]{1,4}-\\d{4}-\\d{1,4}$",
            "separator": "-"
        }
    },
    "colour_dot": {
        "color": {
            "text_color": "N/A",
            "background_color": "product_colour_variable",
            "note": "Solid filled circle in product colour — exact colour is product-specific"
        },
        "size_spec": { "width_mm_approx": 5, "height_mm_approx": 5, "shape": "circle" },
        "placement": {
            "zone": "left_identity",
            "alignment": "left",
            "relative_to": "article_code",
            "position_notes": "To the left of or below article code, before IKEA logo"
        }
    }
}


# ---------------------------------------------------------------------------
# Validation targets template — keyed by element_id + check_type
# ---------------------------------------------------------------------------

def build_validation_targets(elements: list, template_id: str) -> list:
    """Build a flat validation checklist for a template."""
    targets = []
    check_num = 1

    def add(check_type, element_id, severity, description, extra=None):
        nonlocal check_num
        t = {
            "check_id": f"{template_id}-CHK-{check_num:03d}",
            "check_type": check_type,
            "element_id": element_id,
            "severity": severity,
            "description": description
        }
        if extra:
            t.update(extra)
        targets.append(t)
        check_num += 1

    present_ids = set()
    for elem in elements:
        if "$ref_family_element" in elem:
            continue
        eid = elem.get("element_id")
        if not eid:
            continue
        present_ids.add(eid)

        # Presence check
        req = elem.get("required", True)
        add("element_presence", eid,
            "must" if req else "may",
            f"{'Required' if req else 'Optional'} element '{eid}' must {'be present' if req else 'be absent or correctly placed when present'}")

        # Orientation check
        orientation = elem.get("orientation", "horizontal")
        if orientation != "horizontal":
            add("element_orientation", eid, "must",
                f"Element '{eid}' must be oriented {orientation}",
                {"expected_orientation": orientation})

        # Content format check
        patterns = elem.get("allowed_patterns", [])
        if patterns:
            add("element_content_format", eid, "must",
                f"Element '{eid}' content must match allowed pattern(s)",
                {"allowed_patterns": patterns})

        # Barcode spec check
        spec_ref = elem.get("shared_spec_ref")
        if spec_ref:
            add("barcode_spec", eid, "must",
                f"Element '{eid}' must conform to barcode spec '{spec_ref}' in shared_specs/barcode_specs.json",
                {"spec_ref": spec_ref})

        # Placement zone check
        zone = elem.get("placement", {}).get("zone")
        if zone:
            add("element_placement", eid, "must",
                f"Element '{eid}' must be located in zone '{zone}'",
                {"expected_zone": zone})

    # Cross-element consistency checks
    if "origin_text" in present_ids and "address_block" in present_ids:
        check_num_saved = check_num
        targets.append({
            "check_id": f"{template_id}-CHK-{check_num:03d}",
            "check_type": "cross_element_consistency",
            "element_id": "origin_text",
            "related_element_id": "address_block",
            "severity": "must",
            "description": "Origin text country must not contradict address block region (e.g. 'Assembled in USA' must not co-occur with a Chinese address)"
        })
        check_num += 1

    # Zone order check
    zones_present = list(dict.fromkeys([
        elem.get("placement", {}).get("zone")
        for elem in elements
        if "$ref_family_element" not in elem and elem.get("placement", {}).get("zone")
    ]))
    if len(zones_present) > 1:
        targets.append({
            "check_id": f"{template_id}-CHK-{check_num:03d}",
            "check_type": "zone_order",
            "element_id": "layout",
            "severity": "must",
            "description": "Zones must appear in left-to-right order as defined in shared_specs/zone_definitions.json",
            "expected_zone_sequence": zones_present
        })
        check_num += 1

    # Visual hierarchy checks
    if "product_name" in present_ids and "article_number" in present_ids:
        targets.append({
            "check_id": f"{template_id}-CHK-{check_num:03d}",
            "check_type": "visual_hierarchy",
            "element_id": "product_name",
            "related_element_id": "article_number",
            "severity": "must",
            "description": "product_name font size must be visually larger than article_number font size"
        })
        check_num += 1

    if "article_number" in present_ids and "article_code" in present_ids:
        targets.append({
            "check_id": f"{template_id}-CHK-{check_num:03d}",
            "check_type": "visual_hierarchy",
            "element_id": "article_number",
            "related_element_id": "article_code",
            "severity": "must",
            "description": "article_number font size must be visually larger than article_code font size"
        })
        check_num += 1

    return targets


# ---------------------------------------------------------------------------
# Flatten a single template
# ---------------------------------------------------------------------------

def flatten_template(template: dict, family: dict) -> dict:
    family_elements = {e["element_id"]: e for e in family.get("base_elements", [])}
    new_elements = []

    for elem in template.get("elements", []):
        if "$ref_family_element" in elem:
            ref_id = elem["$ref_family_element"]
            base = deepcopy(family_elements.get(ref_id, {}))
            if not base:
                base = {"element_id": ref_id, "_unresolved": True}
            # Apply template-level overrides for this element
            for override in template.get("overrides", []):
                if override.get("element_id") == ref_id:
                    for k, v in override.get("override_fields", {}).items():
                        base[k] = v
            new_elements.append(base)
        else:
            new_elements.append(deepcopy(elem))

    # Apply visual enrichment to each element
    for elem in new_elements:
        eid = elem.get("element_id")
        if eid and eid in ELEMENT_VISUAL_SPECS:
            enrichment = ELEMENT_VISUAL_SPECS[eid]
            for k, v in enrichment.items():
                if k not in elem:
                    elem[k] = deepcopy(v)
                elif k in ("placement",) and isinstance(v, dict):
                    # Merge placement fields without overwriting existing keys
                    for pk, pv in v.items():
                        if pk not in elem[k]:
                            elem[k][pk] = pv

    result = deepcopy(template)
    result["elements"] = new_elements
    result["validation_targets"] = build_validation_targets(new_elements, template["template_id"])
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    families = {}
    for p in FAMILIES_DIR.glob("*.json"):
        data = json.loads(p.read_text())
        families[data["family_id"]] = data

    count = 0
    for p in sorted(TEMPLATES_DIR.glob("*.json")):
        template = json.loads(p.read_text())
        family_id = template.get("family")
        family = families.get(family_id, {})

        enriched = flatten_template(template, family)
        p.write_text(json.dumps(enriched, indent=2, ensure_ascii=False))
        count += 1
        print(f"Enriched: {p.name} ({len(enriched['elements'])} elements, {len(enriched['validation_targets'])} checks)")

    print(f"\nDone — {count} templates enriched.")


if __name__ == "__main__":
    main()
