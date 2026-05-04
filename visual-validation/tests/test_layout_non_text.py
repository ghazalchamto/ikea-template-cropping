"""Non-text layout validators: barcode union-ink, logo, vectors, image/placeholder."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

try:
    import fitz
except ImportError:  # pragma: no cover
    fitz = None  # type: ignore

from src.config.settings import DEFAULT_CONFIG
from src.core.layout_region_validators import validate_layout_extractions
from src.core.pdf_region_extractor import extract_regions
from src.core.region_aggregator import aggregate_region_scores
from src.core.region_loader import load_region_config_file

pytestmark = pytest.mark.skipif(fitz is None, reason="PyMuPDF required")


def _template_json(path: Path, w_mm: float = 220.0, h_mm: float = 120.0) -> None:
    path.write_text(
        json.dumps(
            {
                "template_id": "NONTE_v1",
                "label_size_mm": {"width_mm": w_mm, "height_mm": h_mm},
                "elements": [],
            }
        ),
        encoding="utf-8",
    )


def _gt(tmp: Path) -> tuple[Path, Path]:
    gt = tmp / "ground_truth"
    vdir = gt / "templates" / "NONTE" / "v1"
    vdir.mkdir(parents=True)
    _template_json(vdir / "template.json")
    return gt, vdir


def _write_barcode_solid(path: Path) -> None:
    doc = fitz.open()
    p = doc.new_page(width=220, height=120)
    p.draw_rect(fitz.Rect(55, 45, 125, 75), color=(0, 0, 0), fill=(0, 0, 0))
    doc.save(str(path))
    doc.close()


def _write_barcode_striped(path: Path) -> None:
    doc = fitz.open()
    p = doc.new_page(width=220, height=120)
    outer = fitz.Rect(55, 45, 125, 75)
    p.draw_rect(outer, color=(0, 0, 0), fill=(0, 0, 0))
    for y in range(48, 73, 3):
        p.draw_rect(fitz.Rect(58, float(y), 122, float(y) + 1.2), color=(1, 1, 1), fill=(1, 1, 1))
    doc.save(str(path))
    doc.close()


def _cfg_barcode_only(tmp: Path) -> Path:
    cfg = {
        "product_code": "NONTE",
        "page_size": {"width": 220, "height": 120},
        "regions": [
            {
                "name": "mark",
                "type": "barcode_block",
                "bbox_norm": [0.0, 0.0, 1.0, 1.0],
                "mandatory": True,
                "weight": 1.0,
                "tolerance_mm": 2.0,
            },
        ],
    }
    p = tmp / "NONTE.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    return p


def test_barcode_same_position_different_internal_pattern_passes(tmp_path: Path) -> None:
    _gt(tmp_path)
    vdir = tmp_path / "ground_truth" / "templates" / "NONTE" / "v1"
    tpl = vdir / "label.pdf"
    cand = tmp_path / "cand.pdf"
    _write_barcode_solid(tpl)
    _write_barcode_striped(cand)
    cfg_path = _cfg_barcode_only(tmp_path)
    cfg = load_region_config_file(cfg_path, expected_product_code="NONTE")
    t_ext = extract_regions(tpl, cfg, page_index=0)
    c_ext = extract_regions(cand, cfg, page_index=0)
    scores = validate_layout_extractions(
        t_ext, c_ext, label_width_mm=220.0, label_height_mm=120.0, default_tolerance_mm=2.0,
    )
    summary = aggregate_region_scores(scores, cfg, DEFAULT_CONFIG)
    assert summary.verdict == "VALID"


def test_barcode_missing_candidate_fails(tmp_path: Path) -> None:
    _gt(tmp_path)
    vdir = tmp_path / "ground_truth" / "templates" / "NONTE" / "v1"
    tpl = vdir / "label.pdf"
    cand = tmp_path / "cand.pdf"
    _write_barcode_solid(tpl)
    doc = fitz.open()
    doc.new_page(width=220, height=120)
    doc.save(str(cand))
    doc.close()
    cfg_path = _cfg_barcode_only(tmp_path)
    cfg = load_region_config_file(cfg_path, expected_product_code="NONTE")
    t_ext = extract_regions(tpl, cfg, page_index=0)
    c_ext = extract_regions(cand, cfg, page_index=0)
    scores = validate_layout_extractions(
        t_ext, c_ext, label_width_mm=220.0, label_height_mm=120.0, default_tolerance_mm=2.0,
    )
    summary = aggregate_region_scores(scores, cfg, DEFAULT_CONFIG)
    assert summary.verdict == "INVALID"
    sc = scores[0]
    assert "candidate_barcode_missing" in sc.notes


def _cfg_logo(tmp: Path) -> Path:
    cfg = {
        "product_code": "NONTE",
        "page_size": {"width": 220, "height": 120},
        "regions": [
            {
                "name": "logo",
                "type": "logo_block",
                "bbox_norm": [0.55, 0.1, 0.4, 0.45],
                "mandatory": True,
                "weight": 1.0,
                "tolerance_mm": 2.0,
                "logo_edge_min": 0.2,
            },
        ],
    }
    p = tmp / "NONTE.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    return p


def _write_logo_pdf(path: Path, x0: float = 130, y0: float = 25) -> None:
    doc = fitz.open()
    p = doc.new_page(width=220, height=120)
    p.draw_rect(fitz.Rect(x0, y0, x0 + 45, y0 + 28), color=(0.1, 0.2, 0.7), fill=(0.1, 0.2, 0.7))
    doc.save(str(path))
    doc.close()


def test_logo_identical_passes(tmp_path: Path) -> None:
    _gt(tmp_path)
    vdir = tmp_path / "ground_truth" / "templates" / "NONTE" / "v1"
    tpl = vdir / "label.pdf"
    cand = tmp_path / "cand.pdf"
    _write_logo_pdf(tpl, 130, 25)
    shutil.copy(tpl, cand)
    cfg_path = _cfg_logo(tmp_path)
    cfg = load_region_config_file(cfg_path, expected_product_code="NONTE")
    t_ext = extract_regions(tpl, cfg, page_index=0)
    c_ext = extract_regions(cand, cfg, page_index=0)
    scores = validate_layout_extractions(
        t_ext, c_ext, label_width_mm=220.0, label_height_mm=120.0, default_tolerance_mm=2.0,
    )
    summary = aggregate_region_scores(scores, cfg, DEFAULT_CONFIG)
    assert summary.verdict == "VALID"


def test_logo_shifted_fails(tmp_path: Path) -> None:
    _gt(tmp_path)
    vdir = tmp_path / "ground_truth" / "templates" / "NONTE" / "v1"
    tpl = vdir / "label.pdf"
    cand = tmp_path / "cand.pdf"
    _write_logo_pdf(tpl, 130, 25)
    _write_logo_pdf(cand, 148, 25)
    cfg_path = _cfg_logo(tmp_path)
    cfg = load_region_config_file(cfg_path, expected_product_code="NONTE")
    t_ext = extract_regions(tpl, cfg, page_index=0)
    c_ext = extract_regions(cand, cfg, page_index=0)
    scores = validate_layout_extractions(
        t_ext, c_ext, label_width_mm=220.0, label_height_mm=120.0, default_tolerance_mm=0.4,
    )
    summary = aggregate_region_scores(scores, cfg, DEFAULT_CONFIG)
    assert summary.verdict == "INVALID"


def test_logo_resized_fails(tmp_path: Path) -> None:
    _gt(tmp_path)
    vdir = tmp_path / "ground_truth" / "templates" / "NONTE" / "v1"
    tpl = vdir / "label.pdf"
    cand = tmp_path / "cand.pdf"
    _write_logo_pdf(tpl, 130, 25)
    doc = fitz.open()
    p = doc.new_page(width=220, height=120)
    p.draw_rect(fitz.Rect(130, 25, 130 + 70, 25 + 28), color=(0.1, 0.2, 0.7), fill=(0.1, 0.2, 0.7))
    doc.save(str(cand))
    doc.close()
    cfg_path = _cfg_logo(tmp_path)
    cfg = load_region_config_file(cfg_path, expected_product_code="NONTE")
    t_ext = extract_regions(tpl, cfg, page_index=0)
    c_ext = extract_regions(cand, cfg, page_index=0)
    scores = validate_layout_extractions(
        t_ext, c_ext, label_width_mm=220.0, label_height_mm=120.0, default_tolerance_mm=0.5,
    )
    summary = aggregate_region_scores(scores, cfg, DEFAULT_CONFIG)
    assert summary.verdict == "INVALID"


def test_logo_missing_fails(tmp_path: Path) -> None:
    _gt(tmp_path)
    vdir = tmp_path / "ground_truth" / "templates" / "NONTE" / "v1"
    tpl = vdir / "label.pdf"
    cand = tmp_path / "cand.pdf"
    _write_logo_pdf(tpl, 130, 25)
    doc = fitz.open()
    doc.new_page(width=220, height=120)
    doc.save(str(cand))
    doc.close()
    cfg_path = _cfg_logo(tmp_path)
    cfg = load_region_config_file(cfg_path, expected_product_code="NONTE")
    t_ext = extract_regions(tpl, cfg, page_index=0)
    c_ext = extract_regions(cand, cfg, page_index=0)
    scores = validate_layout_extractions(
        t_ext, c_ext, label_width_mm=220.0, label_height_mm=120.0, default_tolerance_mm=2.0,
    )
    summary = aggregate_region_scores(scores, cfg, DEFAULT_CONFIG)
    assert summary.verdict == "INVALID"
    assert any("candidate_logo_missing" in n for n in scores[0].notes)


def _cfg_divider(tmp: Path) -> Path:
    cfg = {
        "product_code": "NONTE",
        "page_size": {"width": 220, "height": 120},
        "regions": [
            {
                "name": "rule",
                "type": "divider_line",
                "bbox_norm": [0.0, 0.0, 1.0, 0.55],
                "mandatory": True,
                "weight": 1.0,
                "tolerance_mm": 1.5,
            },
        ],
    }
    p = tmp / "NONTE.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    return p


def _write_with_horizontal_line(path: Path, y: float = 32.0) -> None:
    doc = fitz.open()
    p = doc.new_page(width=220, height=120)
    p.draw_line(fitz.Point(12, y), fitz.Point(205, y), color=(0, 0, 0), width=1.2)
    doc.save(str(path))
    doc.close()


def test_divider_line_missing_on_candidate_fails(tmp_path: Path) -> None:
    _gt(tmp_path)
    vdir = tmp_path / "ground_truth" / "templates" / "NONTE" / "v1"
    tpl = vdir / "label.pdf"
    cand = tmp_path / "cand.pdf"
    _write_with_horizontal_line(tpl, 32)
    doc = fitz.open()
    doc.new_page(width=220, height=120)
    doc.save(str(cand))
    doc.close()
    cfg_path = _cfg_divider(tmp_path)
    cfg = load_region_config_file(cfg_path, expected_product_code="NONTE")
    t_ext = extract_regions(tpl, cfg, page_index=0)
    c_ext = extract_regions(cand, cfg, page_index=0)
    scores = validate_layout_extractions(
        t_ext, c_ext, label_width_mm=220.0, label_height_mm=120.0, default_tolerance_mm=1.5,
    )
    summary = aggregate_region_scores(scores, cfg, DEFAULT_CONFIG)
    assert summary.verdict == "INVALID"


def test_divider_line_shifted_fails(tmp_path: Path) -> None:
    _gt(tmp_path)
    vdir = tmp_path / "ground_truth" / "templates" / "NONTE" / "v1"
    tpl = vdir / "label.pdf"
    cand = tmp_path / "cand.pdf"
    _write_with_horizontal_line(tpl, 30)
    _write_with_horizontal_line(cand, 48)
    cfg_path = _cfg_divider(tmp_path)
    cfg = load_region_config_file(cfg_path, expected_product_code="NONTE")
    t_ext = extract_regions(tpl, cfg, page_index=0)
    c_ext = extract_regions(cand, cfg, page_index=0)
    scores = validate_layout_extractions(
        t_ext, c_ext, label_width_mm=220.0, label_height_mm=120.0, default_tolerance_mm=0.6,
    )
    summary = aggregate_region_scores(scores, cfg, DEFAULT_CONFIG)
    assert summary.verdict == "INVALID"


def _cfg_border(tmp: Path) -> Path:
    cfg = {
        "product_code": "NONTE",
        "page_size": {"width": 220, "height": 120},
        "regions": [
            {
                "name": "frame",
                "type": "border_box",
                "bbox_norm": [0.1, 0.1, 0.8, 0.8],
                "mandatory": True,
                "weight": 1.0,
                "tolerance_mm": 2.0,
            },
        ],
    }
    p = tmp / "NONTE.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    return p


def _write_stroked_rect(path: Path, inset: float = 18.0) -> None:
    doc = fitz.open()
    p = doc.new_page(width=220, height=120)
    r = fitz.Rect(inset, inset, 220 - inset, 120 - inset)
    shape = p.new_shape()
    shape.draw_rect(r)
    shape.finish(color=(0, 0, 0), width=2.0)
    shape.commit()
    doc.save(str(path))
    doc.close()


def test_border_box_missing_on_candidate_fails(tmp_path: Path) -> None:
    _gt(tmp_path)
    vdir = tmp_path / "ground_truth" / "templates" / "NONTE" / "v1"
    tpl = vdir / "label.pdf"
    cand = tmp_path / "cand.pdf"
    _write_stroked_rect(tpl, 20)
    doc = fitz.open()
    doc.new_page(width=220, height=120)
    doc.save(str(cand))
    doc.close()
    cfg_path = _cfg_border(tmp_path)
    cfg = load_region_config_file(cfg_path, expected_product_code="NONTE")
    t_ext = extract_regions(tpl, cfg, page_index=0)
    c_ext = extract_regions(cand, cfg, page_index=0)
    scores = validate_layout_extractions(
        t_ext, c_ext, label_width_mm=220.0, label_height_mm=120.0, default_tolerance_mm=2.0,
    )
    summary = aggregate_region_scores(scores, cfg, DEFAULT_CONFIG)
    assert summary.verdict == "INVALID"


def _cfg_image_mandatory_plus_optional_ph(tmp: Path) -> Path:
    cfg = {
        "product_code": "NONTE",
        "page_size": {"width": 220, "height": 120},
        "regions": [
            {
                "name": "pic",
                "type": "image_block",
                "bbox_norm": [0.05, 0.15, 0.35, 0.45],
                "mandatory": True,
                "weight": 1.0,
                "tolerance_mm": 2.0,
                "image_dark_thresh": 200,
                "image_min_dark_pixels": 40,
            },
            {
                "name": "ph",
                "type": "placeholder_block",
                "bbox_norm": [0.72, 0.0, 0.28, 1.0],
                "mandatory": False,
                "weight": 0.0,
            },
        ],
    }
    p = tmp / "NONTE.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    return p


def _write_gray_patch(path: Path) -> None:
    doc = fitz.open()
    p = doc.new_page(width=220, height=120)
    p.draw_rect(fitz.Rect(18, 28, 85, 72), color=(0.35, 0.35, 0.35), fill=(0.35, 0.35, 0.35))
    doc.save(str(path))
    doc.close()


def test_mandatory_image_missing_fails(tmp_path: Path) -> None:
    _gt(tmp_path)
    vdir = tmp_path / "ground_truth" / "templates" / "NONTE" / "v1"
    tpl = vdir / "label.pdf"
    cand = tmp_path / "cand.pdf"
    _write_gray_patch(tpl)
    doc = fitz.open()
    doc.new_page(width=220, height=120)
    doc.save(str(cand))
    doc.close()
    cfg_path = _cfg_image_mandatory_plus_optional_ph(tmp_path)
    cfg = load_region_config_file(cfg_path, expected_product_code="NONTE")
    t_ext = extract_regions(tpl, cfg, page_index=0)
    c_ext = extract_regions(cand, cfg, page_index=0)
    scores = validate_layout_extractions(
        t_ext, c_ext, label_width_mm=220.0, label_height_mm=120.0, default_tolerance_mm=2.0,
    )
    summary = aggregate_region_scores(scores, cfg, DEFAULT_CONFIG)
    assert summary.verdict == "INVALID"
    pic = next(s for s in scores if s.region_name == "pic")
    assert any("candidate_image_missing" in n for n in pic.notes)


def test_optional_placeholder_missing_passes(tmp_path: Path) -> None:
    _gt(tmp_path)
    vdir = tmp_path / "ground_truth" / "templates" / "NONTE" / "v1"
    tpl = vdir / "label.pdf"
    cand = tmp_path / "cand.pdf"
    _write_gray_patch(tpl)
    shutil.copy(tpl, cand)
    cfg_path = _cfg_image_mandatory_plus_optional_ph(tmp_path)
    cfg = load_region_config_file(cfg_path, expected_product_code="NONTE")
    t_ext = extract_regions(tpl, cfg, page_index=0)
    c_ext = extract_regions(cand, cfg, page_index=0)
    scores = validate_layout_extractions(
        t_ext, c_ext, label_width_mm=220.0, label_height_mm=120.0, default_tolerance_mm=2.0,
    )
    summary = aggregate_region_scores(scores, cfg, DEFAULT_CONFIG)
    assert summary.verdict == "VALID"
