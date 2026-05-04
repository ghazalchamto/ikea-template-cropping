"""Coarse fallback region schema (PDF aspect only)."""

from pathlib import Path

import pytest

from src.core.region_schema_fallback import build_coarse_fallback_schema_dict
from src.core.region_schema_from_template import load_template_json
from src.core.region_loader import load_region_config_file


def test_fallback_1r20_roundtrip(tmp_path: Path):
    root = Path(__file__).resolve().parent.parent
    pdf = root / "data/templates/ground-truth-ikea-labels/ground_truth/templates/1R20-PRADM/v1/label.pdf"
    tjson = root / "data/templates/ground-truth-ikea-labels/ground_truth/templates/1R20-PRADM/v1/template.json"
    if not pdf.is_file():
        pytest.skip("1R20-PRADM v1 label.pdf not found")
    raw = load_template_json(tjson)
    schema = build_coarse_fallback_schema_dict(
        "1R20-PRADM",
        template=raw,
        label_pdf_path=pdf,
        label_png_path=None,
        template_version="v1",
    )
    assert schema["product_code"] == "1R20-PRADM"
    assert len(schema["regions"]) >= 4
    for r in schema["regions"]:
        x, y, w, h = r["bbox_norm"]
        assert 0 <= x <= 1 and 0 <= y <= 1 and w > 0 and h > 0
        assert x + w <= 1.001 and y + h <= 1.001

    p = tmp_path / "1R20-PRADM.json"
    p.write_text(__import__("json").dumps(schema, indent=2), encoding="utf-8")
    cfg = load_region_config_file(p, expected_product_code="1R20-PRADM")
    assert len(cfg.regions) == len(schema["regions"])
