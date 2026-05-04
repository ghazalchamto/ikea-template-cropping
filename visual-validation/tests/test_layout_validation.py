"""Tests for layout-first region validation (geometry vs template)."""

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
from src.core.layout_geometry import (
    iou_xywh,
    line_count_from_spans,
    spans_overflow_region,
    union_span_bbox,
)
from src.core.layout_region_validators import (
    load_label_size_mm_from_template_json,
    validate_layout_extractions,
)
from src.core.pdf_region_extractor import extract_regions
from src.core.pdf_text_introspector import TextSpan
from src.core.region_aggregator import aggregate_region_scores
from src.core.region_loader import load_region_config_file

pytestmark = pytest.mark.skipif(fitz is None, reason="PyMuPDF required")


def _write_template_json(path: Path, w_mm: float = 220.0, h_mm: float = 120.0) -> None:
    path.write_text(
        json.dumps(
            {
                "template_id": "LAYOUTTST_v1",
                "label_size_mm": {"width_mm": w_mm, "height_mm": h_mm},
                "elements": [],
            }
        ),
        encoding="utf-8",
    )


def _write_pdf_rect(path: Path, dx: float = 0.0) -> None:
    doc = fitz.open()
    page = doc.new_page(width=220, height=120)
    page.draw_rect(fitz.Rect(60 + dx, 40, 140 + dx, 80), color=(0, 0, 0), fill=(0, 0, 0))
    doc.save(str(path))
    doc.close()


def _write_pdf_text_at(
    path: Path, text: str, x: float, y: float, *, fontsize: float = 11.0,
) -> None:
    doc = fitz.open()
    page = doc.new_page(width=220, height=120)
    page.insert_text(fitz.Point(x, y), text, fontsize=fontsize)
    doc.save(str(path))
    doc.close()


def _write_pdf_text(path: Path, text: str) -> None:
    _write_pdf_text_at(path, text, 65, 55)


def _write_pdf_barcode_like_rect(path: Path, y0: float, y1: float) -> None:
    """Black rectangle in the same horizontal band as _write_pdf_rect (for size-change tests)."""
    doc = fitz.open()
    page = doc.new_page(width=220, height=120)
    page.draw_rect(fitz.Rect(60, y0, 140, y1), color=(0, 0, 0), fill=(0, 0, 0))
    doc.save(str(path))
    doc.close()


def _write_pdf_text_mono(path: Path, text: str) -> None:
    """Courier so same-length strings share the same span bbox at one origin."""
    doc = fitz.open()
    page = doc.new_page(width=220, height=120)
    page.insert_text(fitz.Point(65, 55), text, fontname="cour", fontsize=11)
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


def _ground_truth_tree(tmp: Path) -> tuple[Path, Path]:
    gt = tmp / "ground_truth"
    vdir = gt / "templates" / "LAYOUTTST" / "v1"
    vdir.mkdir(parents=True)
    _write_template_json(vdir / "template.json")
    return gt, vdir


def test_load_region_config_wrong_product_code_raises(tmp_path: Path) -> None:
    p = tmp_path / "X.json"
    p.write_text(json.dumps({"product_code": "AAA", "page_size": {"width": 1, "height": 1}, "regions": []}))
    with pytest.raises(ValueError, match="expected"):
        load_region_config_file(p, expected_product_code="BBB")


def test_load_label_size_mm_from_template_json(tmp_path: Path) -> None:
    _ground_truth_tree(tmp_path)
    vdir = tmp_path / "ground_truth" / "templates" / "LAYOUTTST" / "v1"
    w, h = load_label_size_mm_from_template_json(vdir / "template.json")
    assert w == 220.0 and h == 120.0


def test_layout_barcode_identical_passes(tmp_path: Path) -> None:
    _ground_truth_tree(tmp_path)
    vdir = tmp_path / "ground_truth" / "templates" / "LAYOUTTST" / "v1"
    tpl = vdir / "label.pdf"
    cand = tmp_path / "cand.pdf"
    _write_pdf_rect(tpl, 0)
    _write_pdf_rect(cand, 0)
    cfg_path = _region_config_barcode(tmp_path)
    cfg = load_region_config_file(cfg_path, expected_product_code="LAYOUTTST")
    t_ext = extract_regions(tpl, cfg, page_index=0)
    c_ext = extract_regions(cand, cfg, page_index=0)
    scores = validate_layout_extractions(
        t_ext, c_ext, label_width_mm=220.0, label_height_mm=120.0, default_tolerance_mm=2.0,
    )
    summary = aggregate_region_scores(scores, cfg, DEFAULT_CONFIG)
    assert summary.verdict == "VALID"
    assert summary.mandatory_failed == 0


def test_layout_barcode_shift_fails(tmp_path: Path) -> None:
    _ground_truth_tree(tmp_path)
    vdir = tmp_path / "ground_truth" / "templates" / "LAYOUTTST" / "v1"
    tpl = vdir / "label.pdf"
    cand = tmp_path / "cand.pdf"
    _write_pdf_rect(tpl, 0)
    _write_pdf_rect(cand, 18.0)
    cfg_path = _region_config_barcode(tmp_path)
    cfg = load_region_config_file(cfg_path, expected_product_code="LAYOUTTST")
    t_ext = extract_regions(tpl, cfg, page_index=0)
    c_ext = extract_regions(cand, cfg, page_index=0)
    scores = validate_layout_extractions(
        t_ext, c_ext, label_width_mm=220.0, label_height_mm=120.0, default_tolerance_mm=0.3,
    )
    summary = aggregate_region_scores(scores, cfg, DEFAULT_CONFIG)
    assert summary.verdict == "INVALID"


def test_layout_barcode_taller_fails(tmp_path: Path) -> None:
    _ground_truth_tree(tmp_path)
    vdir = tmp_path / "ground_truth" / "templates" / "LAYOUTTST" / "v1"
    tpl = vdir / "label.pdf"
    cand = tmp_path / "cand.pdf"
    _write_pdf_rect(tpl, 0)
    _write_pdf_barcode_like_rect(cand, 25.0, 95.0)
    cfg_path = _region_config_barcode(tmp_path)
    cfg = load_region_config_file(cfg_path, expected_product_code="LAYOUTTST")
    t_ext = extract_regions(tpl, cfg, page_index=0)
    c_ext = extract_regions(cand, cfg, page_index=0)
    scores = validate_layout_extractions(
        t_ext, c_ext, label_width_mm=220.0, label_height_mm=120.0, default_tolerance_mm=0.5,
    )
    summary = aggregate_region_scores(scores, cfg, DEFAULT_CONFIG)
    assert summary.verdict == "INVALID"


def _write_pdf_rect_with_right_strip(path: Path, include_strip: bool) -> None:
    doc = fitz.open()
    page = doc.new_page(width=220, height=120)
    page.draw_rect(fitz.Rect(60, 40, 140, 80), color=(0, 0, 0), fill=(0, 0, 0))
    if include_strip:
        page.draw_rect(fitz.Rect(200, 20, 215, 100), color=(0.2, 0.2, 0.2), fill=(0.2, 0.2, 0.2))
    doc.save(str(path))
    doc.close()


def test_layout_optional_strip_missing_on_candidate_passes(tmp_path: Path) -> None:
    """Template has ink in optional strip; candidate does not → optional skipped, VALID."""
    _ground_truth_tree(tmp_path)
    vdir = tmp_path / "ground_truth" / "templates" / "LAYOUTTST" / "v1"
    tpl = vdir / "label.pdf"
    cand = tmp_path / "cand.pdf"
    _write_pdf_rect_with_right_strip(tpl, include_strip=True)
    _write_pdf_rect_with_right_strip(cand, include_strip=False)
    cfg_path = _region_config_barcode(tmp_path)
    cfg = load_region_config_file(cfg_path, expected_product_code="LAYOUTTST")
    t_ext = extract_regions(tpl, cfg, page_index=0)
    c_ext = extract_regions(cand, cfg, page_index=0)
    scores = validate_layout_extractions(
        t_ext, c_ext, label_width_mm=220.0, label_height_mm=120.0, default_tolerance_mm=2.0,
    )
    summary = aggregate_region_scores(scores, cfg, DEFAULT_CONFIG)
    assert summary.verdict == "VALID"


def _write_pdf_blank(path: Path) -> None:
    doc = fitz.open()
    doc.new_page(width=220, height=120)
    doc.save(str(path))
    doc.close()


def test_layout_mandatory_text_missing_fails(tmp_path: Path) -> None:
    _ground_truth_tree(tmp_path)
    vdir = tmp_path / "ground_truth" / "templates" / "LAYOUTTST" / "v1"
    tpl = vdir / "label.pdf"
    cand = tmp_path / "cand.pdf"
    _write_pdf_text(tpl, "X")
    _write_pdf_blank(cand)
    cfg_path = tmp_path / "LAYOUTTST.json"
    cfg_path.write_text(
        json.dumps(
            {
                "product_code": "LAYOUTTST",
                "page_size": {"width": 220, "height": 120},
                "regions": [
                    {
                        "name": "txt",
                        "type": "text_block",
                        "bbox_norm": [0.0, 0.0, 1.0, 1.0],
                        "mandatory": True,
                        "weight": 1.0,
                        "tolerance_mm": 2.0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    cfg = load_region_config_file(cfg_path, expected_product_code="LAYOUTTST")
    t_ext = extract_regions(tpl, cfg, page_index=0)
    c_ext = extract_regions(cand, cfg, page_index=0)
    scores = validate_layout_extractions(
        t_ext, c_ext, label_width_mm=220.0, label_height_mm=120.0, default_tolerance_mm=2.0,
    )
    summary = aggregate_region_scores(scores, cfg, DEFAULT_CONFIG)
    assert summary.verdict == "INVALID"


def test_layout_text_identical_passes(tmp_path: Path) -> None:
    _ground_truth_tree(tmp_path)
    vdir = tmp_path / "ground_truth" / "templates" / "LAYOUTTST" / "v1"
    tpl = vdir / "label.pdf"
    cand = tmp_path / "cand.pdf"
    _write_pdf_text(tpl, "HELLO")
    shutil.copy(tpl, cand)
    cfg_path = tmp_path / "LAYOUTTST.json"
    cfg_path.write_text(
        json.dumps(
            {
                "product_code": "LAYOUTTST",
                "page_size": {"width": 220, "height": 120},
                "regions": [
                    {
                        "name": "txt",
                        "type": "text_block",
                        "bbox_norm": [0.0, 0.0, 1.0, 1.0],
                        "mandatory": True,
                        "weight": 1.0,
                        "tolerance_mm": 3.0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    cfg = load_region_config_file(cfg_path, expected_product_code="LAYOUTTST")
    t_ext = extract_regions(tpl, cfg, page_index=0)
    c_ext = extract_regions(cand, cfg, page_index=0)
    scores = validate_layout_extractions(
        t_ext, c_ext, label_width_mm=220.0, label_height_mm=120.0, default_tolerance_mm=3.0,
    )
    summary = aggregate_region_scores(scores, cfg, DEFAULT_CONFIG)
    assert summary.verdict == "VALID"


def test_layout_text_different_content_monospace_same_bbox_passes(tmp_path: Path) -> None:
    """Different glyphs, same layout footprint (Courier, same length)."""
    _ground_truth_tree(tmp_path)
    vdir = tmp_path / "ground_truth" / "templates" / "LAYOUTTST" / "v1"
    tpl = vdir / "label.pdf"
    cand = tmp_path / "cand.pdf"
    _write_pdf_text_mono(tpl, "AAAAA")
    _write_pdf_text_mono(cand, "BBBBB")
    cfg_path = tmp_path / "LAYOUTTST.json"
    cfg_path.write_text(
        json.dumps(
            {
                "product_code": "LAYOUTTST",
                "page_size": {"width": 220, "height": 120},
                "regions": [
                    {
                        "name": "txt",
                        "type": "text_block",
                        "bbox_norm": [0.0, 0.0, 1.0, 1.0],
                        "mandatory": True,
                        "weight": 1.0,
                        "tolerance_mm": 2.0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    cfg = load_region_config_file(cfg_path, expected_product_code="LAYOUTTST")
    t_ext = extract_regions(tpl, cfg, page_index=0)
    c_ext = extract_regions(cand, cfg, page_index=0)
    scores = validate_layout_extractions(
        t_ext, c_ext, label_width_mm=220.0, label_height_mm=120.0, default_tolerance_mm=2.0,
    )
    summary = aggregate_region_scores(scores, cfg, DEFAULT_CONFIG)
    assert summary.verdict == "VALID"


def test_layout_text_proportional_different_content_same_anchor_passes(tmp_path: Path) -> None:
    """Default (proportional) font: different strings, same origin — width may differ."""
    _ground_truth_tree(tmp_path)
    vdir = tmp_path / "ground_truth" / "templates" / "LAYOUTTST" / "v1"
    tpl = vdir / "label.pdf"
    cand = tmp_path / "cand.pdf"
    _write_pdf_text_at(tpl, "HELLO", 65, 55, fontsize=11)
    _write_pdf_text_at(cand, "WORLDWIDE", 65, 55, fontsize=11)
    cfg_path = tmp_path / "LAYOUTTST.json"
    cfg_path.write_text(
        json.dumps(
            {
                "product_code": "LAYOUTTST",
                "page_size": {"width": 220, "height": 120},
                "regions": [
                    {
                        "name": "txt",
                        "type": "text_block",
                        "bbox_norm": [0.0, 0.0, 1.0, 1.0],
                        "mandatory": True,
                        "weight": 1.0,
                        "tolerance_mm": 2.0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    cfg = load_region_config_file(cfg_path, expected_product_code="LAYOUTTST")
    t_ext = extract_regions(tpl, cfg, page_index=0)
    c_ext = extract_regions(cand, cfg, page_index=0)
    scores = validate_layout_extractions(
        t_ext, c_ext, label_width_mm=220.0, label_height_mm=120.0, default_tolerance_mm=2.0,
    )
    summary = aggregate_region_scores(scores, cfg, DEFAULT_CONFIG)
    assert summary.verdict == "VALID"


def test_layout_text_longer_still_inside_region_passes(tmp_path: Path) -> None:
    _ground_truth_tree(tmp_path)
    vdir = tmp_path / "ground_truth" / "templates" / "LAYOUTTST" / "v1"
    tpl = vdir / "label.pdf"
    cand = tmp_path / "cand.pdf"
    _write_pdf_text_at(tpl, "OK", 12, 44, fontsize=10)
    _write_pdf_text_at(cand, "M" * 42, 12, 44, fontsize=10)
    cfg_path = tmp_path / "LAYOUTTST.json"
    cfg_path.write_text(
        json.dumps(
            {
                "product_code": "LAYOUTTST",
                "page_size": {"width": 220, "height": 120},
                "regions": [
                    {
                        "name": "txt",
                        "type": "text_block",
                        "bbox_norm": [0.0, 0.0, 1.0, 0.92],
                        "mandatory": True,
                        "weight": 1.0,
                        "tolerance_mm": 2.0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    cfg = load_region_config_file(cfg_path, expected_product_code="LAYOUTTST")
    t_ext = extract_regions(tpl, cfg, page_index=0)
    c_ext = extract_regions(cand, cfg, page_index=0)
    scores = validate_layout_extractions(
        t_ext, c_ext, label_width_mm=220.0, label_height_mm=120.0, default_tolerance_mm=2.0,
    )
    summary = aggregate_region_scores(scores, cfg, DEFAULT_CONFIG)
    assert summary.verdict == "VALID"


def test_layout_text_overflow_narrow_region_fails(tmp_path: Path) -> None:
    _ground_truth_tree(tmp_path)
    vdir = tmp_path / "ground_truth" / "templates" / "LAYOUTTST" / "v1"
    tpl = vdir / "label.pdf"
    cand = tmp_path / "cand.pdf"
    _write_pdf_text_at(tpl, "OK", 5, 60, fontsize=11)
    _write_pdf_text_at(cand, "M" * 36, 5, 60, fontsize=11)
    cfg_path = tmp_path / "LAYOUTTST.json"
    cfg_path.write_text(
        json.dumps(
            {
                "product_code": "LAYOUTTST",
                "page_size": {"width": 220, "height": 120},
                "regions": [
                    {
                        "name": "txt",
                        "type": "text_block",
                        "bbox_norm": [0.0, 0.0, 0.36, 1.0],
                        "mandatory": True,
                        "weight": 1.0,
                        "tolerance_mm": 2.0,
                        "overflow_epsilon_pt": 0.25,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    cfg = load_region_config_file(cfg_path, expected_product_code="LAYOUTTST")
    t_ext = extract_regions(tpl, cfg, page_index=0)
    c_ext = extract_regions(cand, cfg, page_index=0)
    scores = validate_layout_extractions(
        t_ext, c_ext, label_width_mm=220.0, label_height_mm=120.0, default_tolerance_mm=2.0,
    )
    summary = aggregate_region_scores(scores, cfg, DEFAULT_CONFIG)
    assert summary.verdict == "INVALID"
    txt = next(s for s in scores if s.region_name == "txt")
    assert any("overflow" in n for n in txt.notes)


def test_layout_text_vertical_shift_beyond_tolerance_fails(tmp_path: Path) -> None:
    _ground_truth_tree(tmp_path)
    vdir = tmp_path / "ground_truth" / "templates" / "LAYOUTTST" / "v1"
    tpl = vdir / "label.pdf"
    cand = tmp_path / "cand.pdf"
    _write_pdf_text_at(tpl, "HELLO", 65, 55, fontsize=11)
    _write_pdf_text_at(cand, "HELLO", 65, 68, fontsize=11)
    cfg_path = tmp_path / "LAYOUTTST.json"
    cfg_path.write_text(
        json.dumps(
            {
                "product_code": "LAYOUTTST",
                "page_size": {"width": 220, "height": 120},
                "regions": [
                    {
                        "name": "txt",
                        "type": "text_block",
                        "bbox_norm": [0.0, 0.0, 1.0, 1.0],
                        "mandatory": True,
                        "weight": 1.0,
                        "tolerance_mm": 0.5,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    cfg = load_region_config_file(cfg_path, expected_product_code="LAYOUTTST")
    t_ext = extract_regions(tpl, cfg, page_index=0)
    c_ext = extract_regions(cand, cfg, page_index=0)
    scores = validate_layout_extractions(
        t_ext, c_ext, label_width_mm=220.0, label_height_mm=120.0, default_tolerance_mm=0.5,
    )
    summary = aggregate_region_scores(scores, cfg, DEFAULT_CONFIG)
    assert summary.verdict == "INVALID"
    txt = next(s for s in scores if s.region_name == "txt")
    assert any("vertical_shift" in n for n in txt.notes)


def test_layout_text_shifted_fails(tmp_path: Path) -> None:
    """Same string at different x — left anchor shift vs template exceeds tolerance."""
    _ground_truth_tree(tmp_path)
    vdir = tmp_path / "ground_truth" / "templates" / "LAYOUTTST" / "v1"
    tpl = vdir / "label.pdf"
    cand = tmp_path / "cand.pdf"
    doc = fitz.open()
    page = doc.new_page(width=220, height=120)
    page.insert_text(fitz.Point(65, 55), "HELLO", fontsize=11)
    doc.save(str(tpl))
    doc.close()
    doc = fitz.open()
    page = doc.new_page(width=220, height=120)
    page.insert_text(fitz.Point(95, 55), "HELLO", fontsize=11)
    doc.save(str(cand))
    doc.close()
    cfg_path = tmp_path / "LAYOUTTST.json"
    cfg_path.write_text(
        json.dumps(
            {
                "product_code": "LAYOUTTST",
                "page_size": {"width": 220, "height": 120},
                "regions": [
                    {
                        "name": "txt",
                        "type": "text_block",
                        "bbox_norm": [0.0, 0.0, 1.0, 1.0],
                        "mandatory": True,
                        "weight": 1.0,
                        "tolerance_mm": 0.5,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    cfg = load_region_config_file(cfg_path, expected_product_code="LAYOUTTST")
    t_ext = extract_regions(tpl, cfg, page_index=0)
    c_ext = extract_regions(cand, cfg, page_index=0)
    scores = validate_layout_extractions(
        t_ext, c_ext, label_width_mm=220.0, label_height_mm=120.0, default_tolerance_mm=0.5,
    )
    summary = aggregate_region_scores(scores, cfg, DEFAULT_CONFIG)
    assert summary.verdict == "INVALID"


def test_line_count_from_spans_two_spans_one_visual_line() -> None:
    s1 = TextSpan("aa", "helv", 11.0, 0, 0, 10, 12, (0.0, 50.0, 20.0, 62.0))
    s2 = TextSpan("bb", "helv", 11.0, 20, 0, 10, 12, (22.0, 50.0, 40.0, 62.0))
    assert line_count_from_spans([s1, s2]) == 1


def test_spans_overflow_region_detects_outside() -> None:
    s = TextSpan("x", "helv", 11.0, 0, 0, 5, 10, (0.0, 0.0, 120.0, 10.0))
    bad, bb = spans_overflow_region([s], 0, 0, 100, 100, eps_pt=0.5)
    assert bad and bb is not None


def test_iou_xywh_perfect() -> None:
    assert iou_xywh((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)


def test_union_span_bbox_empty() -> None:
    assert union_span_bbox([]) == (0.0, 0.0, 0.0, 0.0)


def test_pipeline_layout_writes_reports(tmp_path: Path) -> None:
    from src.reporting.pipeline_runner import run_region_report_pipeline

    _ground_truth_tree(tmp_path)
    vdir = tmp_path / "ground_truth" / "templates" / "LAYOUTTST" / "v1"
    tpl = vdir / "label.pdf"
    _write_pdf_rect(tpl, 0)
    cand = tmp_path / "cand.pdf"
    _write_pdf_rect(cand, 0)
    regions_dir = tmp_path / "regions"
    regions_dir.mkdir()
    shutil.copy(_region_config_barcode(tmp_path), regions_dir / "LAYOUTTST.json")
    out_dir = tmp_path / "out"
    gt = tmp_path / "ground_truth"
    result = run_region_report_pipeline(
        candidate_pdf=cand,
        product_code="LAYOUTTST",
        template_version="v1",
        ground_truth_dir=gt,
        regions_lookup=regions_dir,
        generated_regions_dir=tmp_path / "gen",
        out_reports_dir=out_dir,
        allow_auto_schema=False,
        layout_only=True,
        layout_tolerance_mm=2.0,
    )
    assert result.report_html.is_file()
    assert result.report_json.is_file()
    payload = json.loads(result.report_json.read_text(encoding="utf-8"))
    assert payload["pipeline"]["validation_mode"] == "layout"
    assert payload["verdict"] == "VALID"
    mark_row = next(r for r in payload["regions"] if r["name"] == "mark")
    assert mark_row.get("pdf_bbox_region_template") is not None
    assert mark_row.get("pdf_bbox_candidate") is not None
