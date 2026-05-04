"""Region schema generation from template.json (bbox_norm zones)."""

import json
from pathlib import Path

import pytest

from src.core.region_schema_from_template import build_region_schema_dict, load_template_json


@pytest.fixture
def template_1r10_v6_path() -> Path:
    root = Path(__file__).resolve().parent.parent
    p = (
        root
        / "data/templates/ground-truth-ikea-labels/ground_truth/templates/1R10-PRADM/v6/template.json"
    )
    if not p.is_file():
        pytest.skip("1R10-PRADM v6 template.json not found")
    return p


def test_build_region_schema_1r10_v6_snapshot(template_1r10_v6_path: Path):
    raw = load_template_json(template_1r10_v6_path)
    schema = build_region_schema_dict(raw, "1R10-PRADM", template_version="v6")

    assert schema["product_code"] == "1R10-PRADM"
    assert schema["page_size"]["width"] == 1000
    assert "label_rect_pts" in schema
    assert schema["label_rect_pts"] == [0.25, 0.25, 409.39, 28.1]

    names = [r["name"] for r in schema["regions"]]
    assert "left_identity" in names
    assert "center_barcode" in names
    assert "right_legal_symbols" in names
    assert "right_date_logo" in names
    assert "right_optional_strip" in names

    by_name = {r["name"]: r for r in schema["regions"]}
    assert by_name["center_barcode"]["type"] == "text_block"
    assert by_name["right_legal_symbols"]["type"] == "barcode_block"
    assert by_name["right_date_logo"]["type"] == "logo_block"

    # bbox_norm in [0,1]
    for r in schema["regions"]:
        x, y, w, h = r["bbox_norm"]
        assert 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1
        assert x + w <= 1.0001
        assert y + h <= 1.0001


def test_example_generated_json_roundtrip_region_loader(template_1r10_v6_path: Path):
    from src.core.region_loader import load_region_config_file

    raw = load_template_json(template_1r10_v6_path)
    schema = build_region_schema_dict(raw, "1R10-PRADM", template_version="v6")
    # Minimal round-trip without writing to disk: validate JSON serialisable
    text = json.dumps(schema)
    assert "bbox_norm" in text

    # load_region_config_file requires a file — use tmp path
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write(text)
        tmp = Path(f.name)
    try:
        cfg = load_region_config_file(tmp, expected_product_code="1R10-PRADM")
        assert len(cfg.regions) == len(schema["regions"])
    finally:
        tmp.unlink(missing_ok=True)
