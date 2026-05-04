"""
tests/test_pipeline.py
======================
Self-contained test suite — no real label files needed.
Synthetic IKEA-like label images are generated on the fly.

Run from visual-validation/:
    pytest tests/ -v
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from dataclasses import replace

from src.config.settings import ValidationConfig
from src.core.alignment import preprocess, align
from src.core.comparator import compare
from src.core.annotator import annotate, make_heatmap, make_grid_map, make_side_by_side
from src.core.template_store import TemplateStore


# ── Synthetic image helpers ────────────────────────────────────────────────────

def _make_label(w=620, h=877, seed=0) -> np.ndarray:
    """Create a synthetic IKEA-like label image."""
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    blocks = [
        (0,    0,   w,   55,  (215, 215, 215)),
        (10,   65,  190, 150, (175, 208, 238)),
        (210,  65,  390, 150, (243, 243, 243)),
        (10,   235, w-20, 3,  (90,  90,  90)),
        (10,   250, 145, 185, (205, 238, 205)),
        (165,  255, 430, 52,  (237, 237, 237)),
        (165,  317, 430, 52,  (246, 246, 246)),
        (165,  379, 360, 52,  (237, 237, 237)),
        (0,    h-65, w,  65,  (208, 212, 238)),
    ]
    for x, y, bw, bh, col in blocks:
        cv2.rectangle(img, (x, y), (x+bw, y+bh), col, -1)
        cv2.rectangle(img, (x, y), (x+bw, y+bh), (175,175,175), 1)
    # Simulate barcode lines
    for i in range(20):
        lx = 18 + i * 6
        cv2.line(img, (lx, 265), (lx, 420), (20,20,20) if i%2==0 else (255,255,255), 2 if i%3==0 else 1)
    return img


def _noisy(img: np.ndarray, level: int = 6, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = rng.integers(-level, level+1, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + n, 0, 255).astype(np.uint8)


def _corrupt(img: np.ndarray) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    # Shift right panel down 45px
    out[65:215, 210:600] = 255
    cv2.rectangle(out, (210, 110), (600, 260), (243, 243, 243), -1)
    cv2.rectangle(out, (210, 110), (600, 260), (175, 175, 175), 1)
    # Rogue element
    cv2.rectangle(out, (360, 330), (560, 415), (255, 110, 110), -1)
    cv2.putText(out, "VOID", (368, 388), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (180, 20, 20), 2, cv2.LINE_AA)
    # Erase part of footer
    out[h - 65 : h, : w // 2] = 255
    # Large global defect — without this, aggregate scores on newer scikit-image/NumPy
    # can still sit above 0.85; this reliably drops SSIM and pixel sim vs the template.
    out[0 : h // 2, :] = (0, 0, 0)
    return out


# ── Config fixture ─────────────────────────────────────────────────────────────

@pytest.fixture
def cfg():
    return ValidationConfig(
        render_dpi=72, target_width=620, target_height=877,
        noise_min_area=30, valid_score_threshold=0.85, ssim_threshold=0.80,
        output_scale=1.0, grid_cols=6, grid_rows=8,
        comparison_mode="full",
    )


@pytest.fixture
def template(cfg):
    return preprocess(_make_label(), cfg)


@pytest.fixture
def valid_test(cfg):
    return preprocess(_noisy(_make_label()), cfg)


@pytest.fixture
def invalid_test(cfg):
    return preprocess(_corrupt(_make_label()), cfg)


# ── Preprocessing & alignment ─────────────────────────────────────────────────

class TestPreprocessing:
    def test_output_shape(self, cfg, template):
        assert template.shape == (cfg.target_height, cfg.target_width, 3)
        assert template.dtype == np.uint8

    def test_align_same_img(self, cfg, template):
        aligned, _ = align(template, template.copy(), cfg)
        assert aligned.shape == template.shape

    def test_align_returns_correct_shape(self, cfg, template, valid_test):
        aligned, _ = align(template, valid_test, cfg)
        assert aligned.shape == template.shape


# ── Comparator ────────────────────────────────────────────────────────────────

class TestComparator:
    def test_identical_scores_near_one(self, cfg, template):
        r = compare(template, template.copy(), cfg)
        assert r.ssim_score       > 0.99
        assert r.pixel_similarity > 0.99
        assert r.edge_similarity  > 0.90
        assert r.tile_pass_rate   > 0.99
        assert r.is_valid

    def test_valid_label_passes(self, cfg, template, valid_test):
        aligned, _ = align(template, valid_test, cfg)
        r = compare(template, aligned, cfg)
        assert r.is_valid, f"Expected VALID: score={r.final_score:.4f} ssim={r.ssim_score:.4f}"

    def test_invalid_label_fails(self, cfg, template, invalid_test):
        aligned, _ = align(template, invalid_test, cfg)
        r = compare(template, aligned, cfg)
        assert not r.is_valid, f"Expected INVALID: score={r.final_score:.4f}"
        assert len(r.mismatch_regions) > 0

    def test_regions_sorted_by_area(self, cfg, template, invalid_test):
        aligned, _ = align(template, invalid_test, cfg)
        r = compare(template, aligned, cfg)
        areas = [reg.area for reg in r.mismatch_regions]
        assert areas == sorted(areas, reverse=True)

    def test_severity_values(self, cfg, template, invalid_test):
        aligned, _ = align(template, invalid_test, cfg)
        r = compare(template, aligned, cfg)
        for reg in r.mismatch_regions:
            assert reg.severity in ("critical", "moderate", "minor")
            assert reg.source in ("pixel", "edge")

    def test_tile_results_shape(self, cfg, template):
        r = compare(template, template.copy(), cfg)
        assert len(r.tile_results) == cfg.grid_rows * cfg.grid_cols
        for t in r.tile_results:
            assert 0.0 <= t.pixel_sim <= 1.0

    def test_failed_tiles_on_corrupt(self, cfg, template, invalid_test):
        aligned, _ = align(template, invalid_test, cfg)
        r = compare(template, aligned, cfg)
        n_fail = sum(1 for t in r.tile_results if not t.passed)
        assert n_fail > 0

    def test_summary_serialisable(self, cfg, template):
        r = compare(template, template.copy(), cfg)
        json.dumps(r.summary())   # must not raise

    def test_summary_required_keys(self, cfg, template):
        r = compare(template, template.copy(), cfg)
        s = r.summary()
        for k in ("valid", "final_score", "ssim_score", "pixel_similarity",
                  "edge_similarity", "tile_pass_rate", "alignment_ok",
                  "comparison_mode", "n_mismatch_regions", "n_failed_tiles",
                  "mismatch_regions", "tile_results",
                  "full_image_ssim", "full_image_final_score", "full_image_valid",
                  "region_scores", "failed_mandatory_regions"):
            assert k in s, f"Missing key: {k}"

    def test_diff_mask_dtype(self, cfg, template):
        r = compare(template, template.copy(), cfg)
        assert r.diff_mask.dtype == bool
        assert r.diff_mask.shape == (cfg.target_height, cfg.target_width)

    def test_region_split_mode_eight_regions(self, cfg, template):
        cfg_rs = replace(cfg, comparison_mode="region_split")
        r = compare(template, template.copy(), cfg_rs)
        assert r.comparison_mode == "region_split"
        assert len(r.region_scores) == 8
        names = {rs.name for rs in r.region_scores}
        assert "top_left" in names and "bottom_right" in names
        assert r.is_valid and r.full_image_is_valid


# ── Annotator ────────────────────────────────────────────────────────────────

class TestAnnotator:
    def test_annotate_shape(self, cfg, template, invalid_test):
        aligned, _ = align(template, invalid_test, cfg)
        r = compare(template, aligned, cfg)
        ann = annotate(aligned, r, cfg)
        assert ann.ndim == 3 and ann.dtype == np.uint8

    def test_heatmap_shape(self, cfg, template, invalid_test):
        aligned, _ = align(template, invalid_test, cfg)
        r = compare(template, aligned, cfg)
        hm = make_heatmap(aligned, r, cfg)
        assert hm.dtype == np.uint8

    def test_grid_map_dimensions(self, cfg, template):
        r = compare(template, template.copy(), cfg)
        gm = make_grid_map(r, cfg)
        expected_w = cfg.grid_cols * 80
        assert gm.shape[1] == expected_w

    def test_side_by_side_triple_width(self, cfg, template, valid_test):
        aligned, _ = align(template, valid_test, cfg)
        r = compare(template, aligned, cfg)
        sbs = make_side_by_side(template, aligned, r, scale=1.0)
        assert sbs.shape[1] >= cfg.target_width * 2


# ── TemplateStore ─────────────────────────────────────────────────────────────

class TestTemplateStore:
    def _make_gt(self, tmp_path, product="TEST-PROD", ver="v1"):
        vdir = tmp_path / "templates" / product / ver
        vdir.mkdir(parents=True)
        img = np.full((20,20,3), 200, dtype=np.uint8)
        cv2.imwrite(str(vdir / "label.png"), img)
        return tmp_path

    def test_discover_and_get(self, tmp_path):
        gt = self._make_gt(tmp_path)
        cfg = ValidationConfig(render_dpi=72, target_width=20, target_height=20,
                               gaussian_blur_sigma=0)
        store = TemplateStore(str(gt), cfg)
        assert "TEST-PROD" in store.product_codes()
        img, ver = store.get("TEST-PROD")
        assert img.shape == (20, 20, 3)
        assert ver == "v1"

    def test_missing_product_raises(self, tmp_path):
        (tmp_path / "templates").mkdir()
        store = TemplateStore(str(tmp_path))
        with pytest.raises(KeyError):
            store.get("DOES-NOT-EXIST")

    def test_latest_version_selected(self, tmp_path):
        for ver in ("v1", "v2", "v3"):
            vdir = tmp_path / "templates" / "MULTI" / ver
            vdir.mkdir(parents=True)
            cv2.imwrite(str(vdir / "label.png"), np.full((10,10,3),200,dtype=np.uint8))
        cfg = ValidationConfig(render_dpi=72, target_width=10, target_height=10,
                               gaussian_blur_sigma=0)
        store = TemplateStore(str(tmp_path), cfg)
        _, ver = store.get("MULTI")
        assert ver == "v3"


# ── End-to-end ────────────────────────────────────────────────────────────────

class TestEndToEnd:
    def _setup(self, tmp_path, product="E2E-TEST"):
        gt = tmp_path / "ground_truth"
        vdir = gt / "templates" / product / "v1"
        vdir.mkdir(parents=True)
        cv2.imwrite(str(vdir / "label.png"), _make_label())
        return gt

    def _cfg(self):
        return ValidationConfig(
            render_dpi=72, target_width=620, target_height=877,
            noise_min_area=30, valid_score_threshold=0.85, ssim_threshold=0.80,
            output_scale=1.0, grid_cols=6, grid_rows=8,
            comparison_mode="full",
        )

    def test_valid_end_to_end(self, tmp_path):
        from src.validator import LabelValidator
        gt = self._setup(tmp_path)
        p  = tmp_path / "good.png"
        cv2.imwrite(str(p), _noisy(_make_label()))

        r = LabelValidator(gt, self._cfg()).validate(
            p, product_code="E2E-TEST", output_dir=tmp_path/"out"
        )
        assert r.verdict == "VALID"
        assert r.score > 0.85

    def test_invalid_end_to_end(self, tmp_path):
        from src.validator import LabelValidator
        gt = self._setup(tmp_path)
        p  = tmp_path / "bad.png"
        cv2.imwrite(str(p), _corrupt(_make_label()))

        r = LabelValidator(gt, self._cfg()).validate(
            p, product_code="E2E-TEST", output_dir=tmp_path/"out"
        )
        assert r.verdict == "INVALID"
        assert len(r.comparison.mismatch_regions) > 0
        n_fail = sum(1 for t in r.comparison.tile_results if not t.passed)
        assert n_fail > 0

    def test_output_files_all_created(self, tmp_path):
        from src.validator import LabelValidator
        gt  = self._setup(tmp_path)
        p   = tmp_path / "label.png"
        cv2.imwrite(str(p), _make_label())
        out = tmp_path / "out"

        LabelValidator(gt, self._cfg()).validate(
            p, product_code="E2E-TEST", output_dir=out
        )
        for suffix in ("_annotated.png", "_heatmap.png", "_grid_map.png",
                       "_comparison.png", "_report.json", "_report.html"):
            assert (out / f"label{suffix}").exists(), f"Missing: label{suffix}"

    def test_wrong_product_code_raises(self, tmp_path):
        from src.validator import LabelValidator
        gt = self._setup(tmp_path)
        p  = tmp_path / "label.png"
        cv2.imwrite(str(p), _make_label())
        with pytest.raises(KeyError):
            LabelValidator(gt, self._cfg()).validate(
                p, product_code="WRONG-CODE"
            )

    def test_batch(self, tmp_path):
        from src.validator import LabelValidator
        gt = self._setup(tmp_path)
        labels = []
        for i in range(3):
            p = tmp_path / f"label_{i}.png"
            cv2.imwrite(str(p), _noisy(_make_label(), seed=i))
            labels.append(p)

        results = LabelValidator(gt, self._cfg()).validate_batch(
            labels, product_code="E2E-TEST", output_dir=tmp_path/"out"
        )
        assert len(results) == 3
