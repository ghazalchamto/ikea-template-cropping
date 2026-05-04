"""
Regression and hardening tests for layout validation: coordinates, PDF gates,
aggregation, region schema, and optional-region pairing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

try:
    import fitz
except ImportError:  # pragma: no cover
    fitz = None  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataclasses import replace

from src.config.settings import DEFAULT_CONFIG, ValidationConfig
from src.core.layout_geometry import pdf_boxes_shift_size_mm, region_image_xywh_to_pdf_rect
from src.core.layout_region_validators import (
    _score_from_layout_metrics,
    validate_layout_extractions,
)
from src.core.pdf_layout_gate import (
    assert_label_aspect_consistent_with_pdf,
    assert_matching_page_dimensions_pt,
    validate_pdf_pair_page_index,
)
from src.core.pdf_region_extractor import PDFExtraction, extract_regions
from src.core.region_aggregator import aggregate_region_scores
from src.core.region_comparator import RegionScore
from src.core.region_loader import RegionConfigError, load_region_config_file

pytestmark = pytest.mark.skipif(fitz is None, reason="PyMuPDF required")


def _write_pdf_rect(path: Path, dx: float = 0.0) -> None:
    doc = fitz.open()
    page = doc.new_page(width=220, height=120)
    page.draw_rect(fitz.Rect(60 + dx, 40, 140 + dx, 80), color=(0, 0, 0), fill=(0, 0, 0))
    doc.save(str(path))
    doc.close()


def _region_config_barcode(tmp: Path) -> Path:
    cfg = {
        "product_code": "LAYOUTTST",
        "page_size": {"width": 220, "height": 120},
        "regions": [
            {
                "name": "mark",
                "type": "barcode_block",
                "bbox_norm": [0.0, 0.0, 0.88, 1.0],
                "mandatory": True,
                "weight": 1.0,
                "tolerance_mm": 2.0,
            },
            {
                "name": "opt_strip",
                "type": "optional_image",
                "bbox_norm": [0.9, 0.0, 0.1, 1.0],
                "mandatory": False,
                "weight": 0.0,
            },
        ],
    }
    p = tmp / "LAYOUTTST.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    return p


def test_pdf_boxes_shift_size_mm_non_square_label_rect() -> None:
    """Centre shift: template (0,0)-(100,50) vs candidate (10,0)-(110,50) in pt."""
    sx, sy, dw, dh = pdf_boxes_shift_size_mm(
        (0.0, 0.0, 100.0, 50.0),
        (10.0, 0.0, 110.0, 50.0),
        label_w_mm=400.0,
        label_h_mm=100.0,
        label_rect_w_pt=800.0,
        label_rect_h_pt=200.0,
    )
    # Δcentre_x = 10 pt → 10 * (400 mm / 800 pt) = 5 mm
    assert abs(sx - 5.0) < 1e-5
    assert abs(sy) < 1e-8
    assert abs(dw) < 1e-8
    assert abs(dh) < 1e-8


def test_pdf_boxes_shift_size_mm_vertical_and_size() -> None:
    """Rects are PDF (x0,y0,x1,y1); vertical centre moves by 10 pt on 220×120 pt page."""
    sx, sy, dw, dh = pdf_boxes_shift_size_mm(
        (0.0, 0.0, 40.0, 40.0),
        (0.0, 20.0, 50.0, 40.0),
        label_w_mm=220.0,
        label_h_mm=120.0,
        label_rect_w_pt=220.0,
        label_rect_h_pt=120.0,
    )
    assert abs(sx - 5.0) < 1e-5
    assert abs(sy - 10.0) < 1e-5
    assert abs(dw - 10.0) < 1e-5
    assert abs(dh - (-20.0)) < 1e-5


def test_region_image_xywh_to_pdf_rect() -> None:
    r = region_image_xywh_to_pdf_rect(
        (0.0, 0.0, 50.0, 50.0),
        (0.0, 0.0, 100.0, 100.0),
        100,
        100,
    )
    assert r == (0.0, 0.0, 50.0, 50.0)


def test_score_from_layout_metrics_large_shift_under_tight_tolerance() -> None:
    combined, notes, _ = _score_from_layout_metrics(
        iou=0.95,
        shift_x_mm=4.0,
        shift_y_mm=0.0,
        dw_mm=0.0,
        dh_mm=0.0,
        tol_mm=1.0,
        missing=False,
    )
    assert combined < 0.99
    assert notes


def test_validate_pdf_pair_page_count_mismatch(tmp_path: Path) -> None:
    t = tmp_path / "one.pdf"
    c = tmp_path / "two.pdf"
    d1 = fitz.open()
    d1.new_page(width=100, height=100)
    d1.save(str(t))
    d1.close()
    d2 = fitz.open()
    d2.new_page(width=100, height=100)
    d2.new_page(width=100, height=100)
    d2.save(str(c))
    d2.close()
    with pytest.raises(ValueError, match="Page count mismatch"):
        validate_pdf_pair_page_index(t, c, page_index=0)


def test_validate_pdf_pair_page_index_oob(tmp_path: Path) -> None:
    p = tmp_path / "p.pdf"
    d = fitz.open()
    d.new_page(width=50, height=50)
    d.save(str(p))
    d.close()
    with pytest.raises(ValueError, match="out of range"):
        validate_pdf_pair_page_index(p, p, page_index=1)


def test_assert_matching_page_dimensions_pt_raises() -> None:
    a = PDFExtraction(
        pdf_path=Path("a.pdf"),
        page_index=0,
        page_size_pdf=(220.0, 120.0),
        page_rotation=0,
        config_size=(220, 120),
        scale_x=1.0,
        scale_y=1.0,
        label_rect_pdf=(0.0, 0.0, 220.0, 120.0),
        extracted=[],
    )
    b = replace(a, page_size_pdf=(300.0, 120.0))
    with pytest.raises(ValueError, match="page size mismatch"):
        assert_matching_page_dimensions_pt(a, b, max_pt_diff=0.5)


def test_assert_label_aspect_consistent_with_pdf() -> None:
    extr = PDFExtraction(
        pdf_path=Path("t.pdf"),
        page_index=0,
        page_size_pdf=(220.0, 120.0),
        page_rotation=0,
        config_size=(220, 120),
        scale_x=1.0,
        scale_y=1.0,
        label_rect_pdf=(0.0, 0.0, 220.0, 120.0),
        extracted=[],
    )
    assert_label_aspect_consistent_with_pdf(extr, 220.0, 120.0, rel_tol=0.01)
    with pytest.raises(ValueError, match="aspect"):
        assert_label_aspect_consistent_with_pdf(extr, 200.0, 120.0, rel_tol=0.01)


def test_load_region_config_negative_tolerance_mm(tmp_path: Path) -> None:
    raw = {
        "product_code": "BAD",
        "page_size": {"width": 220, "height": 120},
        "regions": [
            {
                "name": "a",
                "type": "text_block",
                "bbox_norm": [0.0, 0.0, 0.5, 0.5],
                "mandatory": True,
                "weight": 1.0,
                "tolerance_mm": -0.5,
            },
        ],
    }
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(RegionConfigError, match="tolerance_mm"):
        load_region_config_file(p, expected_product_code="BAD")


def test_load_region_config_threshold_out_of_range(tmp_path: Path) -> None:
    raw = {
        "product_code": "BAD",
        "page_size": {"width": 220, "height": 120},
        "regions": [
            {
                "name": "a",
                "type": "text_block",
                "bbox_norm": [0.0, 0.0, 0.5, 0.5],
                "mandatory": True,
                "weight": 1.0,
                "threshold": 1.5,
            },
        ],
    }
    p = tmp_path / "bad2.json"
    p.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(RegionConfigError, match="threshold"):
        load_region_config_file(p, expected_product_code="BAD")


def test_load_region_config_line_count_tolerance_bool_rejected(tmp_path: Path) -> None:
    raw = {
        "product_code": "BAD",
        "page_size": {"width": 220, "height": 120},
        "regions": [
            {
                "name": "a",
                "type": "text_block",
                "bbox_norm": [0.0, 0.0, 0.5, 0.5],
                "mandatory": True,
                "weight": 1.0,
                "line_count_tolerance": True,
            },
        ],
    }
    p = tmp_path / "bad3.json"
    p.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(RegionConfigError, match="line_count_tolerance"):
        load_region_config_file(p, expected_product_code="BAD")


def test_load_region_config_wrong_product_code(tmp_path: Path) -> None:
    raw = {
        "product_code": "ABC",
        "page_size": {"width": 220, "height": 120},
        "regions": [
            {
                "name": "a",
                "type": "text_block",
                "bbox_norm": [0.0, 0.0, 0.5, 0.5],
                "mandatory": True,
                "weight": 1.0,
            },
        ],
    }
    p = tmp_path / "pc.json"
    p.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(RegionConfigError):
        load_region_config_file(p, expected_product_code="XYZ")


def test_aggregate_weighted_scoring() -> None:
    cfg_t = {
        "product_code": "WGTST",
        "page_size": {"width": 220, "height": 120},
        "regions": [
            {
                "name": "r1",
                "type": "text_block",
                "bbox_norm": [0.0, 0.0, 0.5, 0.5],
                "mandatory": True,
                "weight": 0.6,
            },
            {
                "name": "r2",
                "type": "text_block",
                "bbox_norm": [0.5, 0.0, 0.5, 0.5],
                "mandatory": True,
                "weight": 0.4,
            },
        ],
    }
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write(json.dumps(cfg_t))
        path = Path(f.name)
    try:
        config = load_region_config_file(path, expected_product_code="WGTST")
    finally:
        path.unlink(missing_ok=True)

    def _rs(name: str, combined: float, w: float) -> RegionScore:
        return RegionScore(
            region_name=name,
            mandatory=True,
            weight=w,
            ssim=combined,
            pixel=combined,
            edge=combined,
            combined=combined,
            width=10,
            height=10,
            region_type="text_block",
            method="test",
        )

    vcfg = ValidationConfig(valid_score_threshold=0.9, region_default_threshold=0.5)
    s_ok = aggregate_region_scores(
        [_rs("r1", 1.0, 0.6), _rs("r2", 0.92, 0.4)],
        config,
        cfg=vcfg,
    )
    assert abs(s_ok.final_score - (0.6 * 1.0 + 0.4 * 0.92)) < 1e-6
    assert s_ok.verdict == "VALID"

    s_bad = aggregate_region_scores(
        [_rs("r1", 1.0, 0.6), _rs("r2", 0.7, 0.4)],
        config,
        cfg=vcfg,
    )
    assert s_bad.verdict == "INVALID"


def test_validate_layout_optional_absent_on_candidate(tmp_path: Path) -> None:
    cfg_path = _region_config_barcode(tmp_path)
    pdf_path = tmp_path / "label.pdf"
    _write_pdf_rect(pdf_path, dx=0.0)
    config = load_region_config_file(cfg_path, expected_product_code="LAYOUTTST")
    full_t = extract_regions(pdf_path, config, page_index=0)
    full_c = extract_regions(pdf_path, config, page_index=0)
    sub_c = replace(full_c, extracted=[e for e in full_c.extracted if e.name == "mark"])
    scores = validate_layout_extractions(
        full_t,
        sub_c,
        label_width_mm=220.0,
        label_height_mm=120.0,
        default_tolerance_mm=1.5,
    )
    by = {s.region_name: s for s in scores}
    assert by["opt_strip"].details.get("candidate_absent") is True
    summary = aggregate_region_scores(scores, config, cfg=DEFAULT_CONFIG)
    assert "opt_strip" in summary.skipped_optional_regions


def test_validate_layout_extra_candidate_region_raises(tmp_path: Path) -> None:
    cfg_path = _region_config_barcode(tmp_path)
    pdf_path = tmp_path / "label.pdf"
    _write_pdf_rect(pdf_path, dx=0.0)
    config = load_region_config_file(cfg_path, expected_product_code="LAYOUTTST")
    full_t = extract_regions(pdf_path, config, page_index=0)
    full_c = extract_regions(pdf_path, config, page_index=0)
    ghost = replace(full_c.extracted[0], region=replace(full_c.extracted[0].region, name="ghost"))
    bad_c = replace(full_c, extracted=list(full_c.extracted) + [ghost])
    with pytest.raises(ValueError, match="unexpected region name"):
        validate_layout_extractions(
            full_t,
            bad_c,
            label_width_mm=220.0,
            label_height_mm=120.0,
            default_tolerance_mm=1.5,
        )


def test_optional_no_score_still_passes_aggregate(tmp_path: Path) -> None:
    cfg_t = {
        "product_code": "OPTNS",
        "page_size": {"width": 220, "height": 120},
        "regions": [
            {
                "name": "m",
                "type": "text_block",
                "bbox_norm": [0.0, 0.0, 0.5, 0.5],
                "mandatory": True,
                "weight": 1.0,
            },
            {
                "name": "o",
                "type": "optional_image",
                "bbox_norm": [0.5, 0.0, 0.5, 0.5],
                "mandatory": False,
                "weight": 0.0,
            },
        ],
    }
    p = tmp_path / "optns.json"
    p.write_text(json.dumps(cfg_t), encoding="utf-8")
    config = load_region_config_file(p, expected_product_code="OPTNS")
    scores = [
        RegionScore(
            region_name="m",
            mandatory=True,
            weight=1.0,
            ssim=1.0,
            pixel=1.0,
            edge=1.0,
            combined=1.0,
            width=10,
            height=10,
            region_type="text_block",
            method="t",
        ),
    ]
    summary = aggregate_region_scores(scores, config, cfg=DEFAULT_CONFIG)
    assert summary.verdict == "VALID"
    assert summary.final_score == 1.0
