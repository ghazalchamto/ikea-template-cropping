"""Tests for optional ``subdivide_y_bands`` in region JSON configs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.core.region_loader import (
    RegionConfigError,
    expand_bbox_norm_regions_with_y_bands,
    load_region_config_file,
)


def test_expand_bbox_norm_regions_with_y_bands():
    regions = [
        {
            "name": "col_a",
            "type": "text_block",
            "bbox_norm": [0.0, 0.0, 0.5, 1.0],
            "mandatory": True,
            "weight": 1.0,
        },
    ]
    bands = [("header", 0.0, 0.3), ("footer", 0.8, 1.0)]
    path = Path("<synthetic>")
    out = expand_bbox_norm_regions_with_y_bands(regions, bands, path)
    names = [r["name"] for r in out]
    assert names == ["col_a__header", "col_a__footer"]
    assert out[0]["bbox_norm"] == [0.0, 0.0, 0.5, 0.3]
    assert out[1]["bbox_norm"] == [0.0, 0.8, 0.5, 0.2]


def test_expand_preserves_legacy_region_without_bbox_norm():
    regions = [
        {"name": "legacy", "type": "text_block", "x": 0, "y": 0, "width": 10, "height": 10},
    ]
    bands = [("header", 0.0, 0.5)]
    path = Path("<synthetic>")
    out = expand_bbox_norm_regions_with_y_bands(regions, bands, path)
    assert len(out) == 1
    assert out[0]["name"] == "legacy"


def test_expand_raises_when_band_produces_no_slice(tmp_path: Path):
    regions = [
        {
            "name": "only_bottom",
            "type": "text_block",
            "bbox_norm": [0.0, 0.9, 1.0, 0.1],
            "mandatory": True,
        },
    ]
    bands = [("header", 0.0, 0.05)]
    p = tmp_path / "x.json"
    with pytest.raises(RegionConfigError, match="no slice"):
        expand_bbox_norm_regions_with_y_bands(regions, bands, p)


def test_expand_optional_footer_band_marks_slices_optional():
    regions = [
        {
            "name": "col_a",
            "type": "text_block",
            "bbox_norm": [0.0, 0.0, 0.5, 1.0],
            "mandatory": True,
            "weight": 1.0,
        },
    ]
    bands = [("header", 0.0, 0.5), ("footer", 0.5, 1.0)]
    path = Path("<synthetic>")
    out = expand_bbox_norm_regions_with_y_bands(
        regions, bands, path, optional_band_ids={"footer"}
    )
    by_name = {r["name"]: r for r in out}
    assert by_name["col_a__header"]["mandatory"] is True
    assert by_name["col_a__footer"]["mandatory"] is False
    assert by_name["col_a__footer"]["weight"] == 0.0


def test_load_region_config_with_subdivide_roundtrip(tmp_path: Path):
    cfg_path = tmp_path / "1R10-PRADM.json"
    cfg_path.write_text(
        json.dumps(
            {
                "product_code": "1R10-PRADM",
                "page_size": {"width": 100, "height": 100},
                "subdivide_y_bands": [
                    {"id": "top", "y0": 0, "y1": 0.5},
                    {"id": "bottom", "y0": 0.5, "y1": 1.0},
                ],
                "regions": [
                    {
                        "name": "zone",
                        "type": "text_block",
                        "bbox_norm": [0.1, 0.0, 0.8, 1.0],
                        "mandatory": True,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    cfg = load_region_config_file(cfg_path, expected_product_code="1R10-PRADM")
    assert len(cfg.regions) == 2
    assert {r.name for r in cfg.regions} == {"zone__top", "zone__bottom"}
