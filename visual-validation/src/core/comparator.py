"""
src/core/comparator.py
======================
Deep, multi-signal visual comparison engine.

Since we always compare against ONE known-correct template (no auto-matching),
we can run a thorough analysis across four independent signal families:

Signal 1 — SSIM (perceptual structural similarity)
    Full-image SSIM on greyscale.  Catches layout shifts, missing elements,
    spacing changes.  The skimage implementation uses a sliding window that
    also returns a pixel-wise similarity MAP — we use that map for spatial
    localisation of structural problems.

Signal 2 — Pixel-level colour difference
    • Absolute diff on each BGR channel independently
    • Combined diff = max across channels (catches hue changes)
    • Morphological closing joins adjacent diff pixels into coherent regions
    • Connected-component analysis extracts bounding boxes + severity

Signal 3 — Edge / structural diff (Canny)
    • Canny edge maps extracted from both images
    • Edge diff = XOR of the two edge maps
    • Catches layout structural differences (missing border lines, shifted
      dividers, changed element outlines) that pure pixel diff misses on
      near-white areas where the change is geometrically thin

Signal 4 — Grid tile analysis
    • The label is divided into (grid_rows × grid_cols) tiles
    • Each tile independently scored on pixel similarity
    • Tiles failing above tile_fail_threshold are marked red on the grid map
    • Tile pass rate feeds into the final weighted score
    • This gives exact spatial localisation ("top-left quarter has a problem")

Named-region scoring (optional)
    If cfg.named_regions is populated, per-zone scores are computed for
    zones like "barcode", "logo", "header".  These appear in the report.

Decision
    comparison_mode "full":  final_score = weighted four-signal score on the full image;
        VALID iff that score passes thresholds (legacy).
    comparison_mode "region_split":  same full-image metrics kept as reference; verdict
        uses mandatory quadrants only (see _run_region_split). Optional-heavy regions
        are skipped. Half-regions are scored for reporting but do not drive verdict.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from skimage.metrics import structural_similarity as _ssim

from src.config.settings import ValidationConfig, DEFAULT_CONFIG
from src.core.optional_zones import OptionalZone

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MismatchRegion:
    x: int
    y: int
    w: int
    h: int
    pixel_diff_ratio: float
    severity: str        # "critical" | "moderate" | "minor"
    source: str          # "pixel" | "edge" | "tile"

    @property
    def area(self) -> int:
        return self.w * self.h

    def as_dict(self) -> dict:
        return {
            "x":              self.x,
            "y":              self.y,
            "width":          self.w,
            "height":         self.h,
            "area_px":        self.area,
            "pixel_diff_pct": round(self.pixel_diff_ratio * 100, 2),
            "severity":       self.severity,
            "source":         self.source,
        }


@dataclass
class TileResult:
    row:         int
    col:         int
    x:           int
    y:           int
    w:           int
    h:           int
    pixel_sim:   float   # 0–1, 1 = identical
    passed:      bool

    def as_dict(self) -> dict:
        return {
            "row": self.row, "col": self.col,
            "x": self.x, "y": self.y,
            "w": self.w, "h": self.h,
            "pixel_sim": round(float(self.pixel_sim), 4),
            "passed": bool(self.passed),
        }


@dataclass
class RegionScore:
    """Per-region metrics for region_split comparison mode."""

    name: str
    role: str  # "quadrant" | "half"
    y: int
    x: int
    width: int
    height: int
    optional_fraction: float
    ignored_due_to_optional: bool
    ssim_score:       Optional[float] = None
    pixel_similarity: Optional[float] = None
    edge_similarity:  Optional[float] = None
    tile_pass_rate:   Optional[float] = None
    final_score:      Optional[float] = None
    passed:           bool = True
    counts_for_verdict: bool = False

    def as_dict(self) -> dict:
        return {
            "name":                  self.name,
            "role":                  self.role,
            "y":                     self.y,
            "x":                     self.x,
            "width":                 self.width,
            "height":                self.height,
            "optional_fraction":     round(float(self.optional_fraction), 4),
            "ignored_due_to_optional": self.ignored_due_to_optional,
            "ssim_score":            None if self.ssim_score is None else round(float(self.ssim_score), 4),
            "pixel_similarity":      None if self.pixel_similarity is None else round(float(self.pixel_similarity), 4),
            "edge_similarity":       None if self.edge_similarity is None else round(float(self.edge_similarity), 4),
            "tile_pass_rate":        None if self.tile_pass_rate is None else round(float(self.tile_pass_rate), 4),
            "final_score":           None if self.final_score is None else round(float(self.final_score), 4),
            "passed":                bool(self.passed),
            "counts_for_verdict":    bool(self.counts_for_verdict),
        }


@dataclass
class ComparisonResult:
    # ── Scores ────────────────────────────────────────────────────────────────
    ssim_score:       float
    pixel_similarity: float
    edge_similarity:  float
    tile_pass_rate:   float
    final_score:      float
    is_valid:         bool
    alignment_ok:     bool

    # ── Named region scores ───────────────────────────────────────────────────
    named_region_scores: Dict[str, float]

    # ── Spatial data ──────────────────────────────────────────────────────────
    diff_mask:        np.ndarray   # bool H×W — True where pixels differ
    ssim_map:         np.ndarray   # float32 H×W — per-pixel SSIM (0=bad,1=good)
    diff_heatmap:     np.ndarray   # uint8 H×W — combined intensity heatmap
    edge_diff_mask:   np.ndarray   # bool H×W — edge-level structural diff
    mismatch_regions: List[MismatchRegion]
    tile_results:     List[TileResult]

    # ── Optional zones ────────────────────────────────────────────────────────
    ignored_zones:       List["OptionalZone"] = field(default_factory=list)
    optional_zone_mask:  np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=bool))

    # Region split (when comparison_mode == "region_split", headline scores = aggregate)
    comparison_mode:          str   = "full"
    region_scores:            List[RegionScore] = field(default_factory=list)
    failed_mandatory_regions: List[str] = field(default_factory=list)
    full_image_ssim:          float = 0.0
    full_image_pixel_similarity: float = 0.0
    full_image_edge_similarity:  float = 0.0
    full_image_tile_pass_rate:   float = 0.0
    full_image_final_score:      float = 0.0
    full_image_is_valid:         bool  = False

    def summary(self) -> dict:
        out = {
            "valid":               bool(self.is_valid),
            "final_score":         round(float(self.final_score), 4),
            "ssim_score":          round(float(self.ssim_score), 4),
            "pixel_similarity":    round(float(self.pixel_similarity), 4),
            "edge_similarity":     round(float(self.edge_similarity), 4),
            "tile_pass_rate":      round(float(self.tile_pass_rate), 4),
            "alignment_ok":        bool(self.alignment_ok),
            "comparison_mode":     self.comparison_mode,
            "named_region_scores": {k: round(float(v), 4)
                                    for k, v in self.named_region_scores.items()},
            "n_mismatch_regions":  len(self.mismatch_regions),
            "n_failed_tiles":      sum(1 for t in self.tile_results if not t.passed),
            "mismatch_regions":    [r.as_dict() for r in self.mismatch_regions],
            "tile_results":        [t.as_dict() for t in self.tile_results],
            "ignored_optional_zones": [z.as_dict() for z in self.ignored_zones],
            "full_image_ssim":          round(float(self.full_image_ssim), 4),
            "full_image_pixel_similarity": round(float(self.full_image_pixel_similarity), 4),
            "full_image_edge_similarity": round(float(self.full_image_edge_similarity), 4),
            "full_image_tile_pass_rate": round(float(self.full_image_tile_pass_rate), 4),
            "full_image_final_score": round(float(self.full_image_final_score), 4),
            "full_image_valid":        bool(self.full_image_is_valid),
            "region_scores":           [r.as_dict() for r in self.region_scores],
            "failed_mandatory_regions": list(self.failed_mandatory_regions),
        }
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Region-split helpers
# ─────────────────────────────────────────────────────────────────────────────

QUADRANT_NAMES = frozenset({"top_left", "top_right", "bottom_left", "bottom_right"})


def _safe_ssim_win(h: int, w: int, cfg_win: int) -> int:
    m = int(min(h, w))
    if m < 3:
        return 3
    wn = min(int(cfg_win), m)
    if wn % 2 == 0:
        wn -= 1
    return max(3, wn)


def _iter_region_boxes(H: int, W: int) -> List[Tuple[str, str, int, int, int, int]]:
    """Return (name, role, y, x, height, width) in pixel coordinates."""
    hh = H // 2
    hw = W // 2
    return [
        ("top_half", "half", 0, 0, hh, W),
        ("bottom_half", "half", hh, 0, H - hh, W),
        ("left_half", "half", 0, 0, H, hw),
        ("right_half", "half", 0, hw, H, W - hw),
        ("top_left", "quadrant", 0, 0, hh, hw),
        ("top_right", "quadrant", 0, hw, hh, W - hw),
        ("bottom_left", "quadrant", hh, 0, H - hh, hw),
        ("bottom_right", "quadrant", hh, hw, H - hh, W - hw),
    ]


def _morph_kernel_for_size(cfg_ks: int, h: int, w: int) -> int:
    cap = max(3, min(h, w) - 2)
    ks  = min(cfg_ks, cap)
    if ks % 2 == 0:
        ks -= 1
    return max(3, ks)


def _region_metrics(
    template:     np.ndarray,
    aligned_test: np.ndarray,
    ignore_mask:  np.ndarray,
    cfg:          ValidationConfig,
    *,
    grid_rows:    int,
    grid_cols:    int,
    noise_min_area: int,
) -> Tuple[float, float, float, float, float]:
    """
    Same four signals as global compare(), for a rectangular crop.
    Returns (ssim, pixel_sim, edge_sim, tile_pass_rate, weighted_final).
    """
    H, W = template.shape[:2]
    assert aligned_test.shape[:2] == (H, W) and ignore_mask.shape == (H, W)

    mandatory_mask = ~ignore_mask
    mandatory_px   = float(mandatory_mask.sum())

    gray_tmpl = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    gray_test = cv2.cvtColor(aligned_test, cv2.COLOR_BGR2GRAY)

    gray_tmpl_ssim = gray_tmpl.copy()
    gray_test_ssim = gray_test.copy()
    if ignore_mask.any():
        gray_tmpl_ssim[ignore_mask] = 128
        gray_test_ssim[ignore_mask] = 128

    win = _safe_ssim_win(H, W, cfg.ssim_win_size)
    ssim_score, _ = _ssim(
        gray_tmpl_ssim, gray_test_ssim,
        full=True,
        win_size=win,
        data_range=255,
    )
    ssim_score = float(ssim_score)

    diff_b = cv2.absdiff(template[:, :, 0], aligned_test[:, :, 0]).astype(np.float32)
    diff_g = cv2.absdiff(template[:, :, 1], aligned_test[:, :, 1]).astype(np.float32)
    diff_r = cv2.absdiff(template[:, :, 2], aligned_test[:, :, 2]).astype(np.float32)
    diff_gray = cv2.absdiff(gray_tmpl, gray_test).astype(np.float32)
    combined_diff = np.maximum.reduce([diff_b, diff_g, diff_r, diff_gray])

    raw_mask = combined_diff > cfg.diff_pixel_threshold
    raw_mask[ignore_mask] = False

    ks = _morph_kernel_for_size(cfg.morph_kernel_size, H, W)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    closed     = cv2.morphologyEx(raw_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    clean_mask = _remove_small_blobs(closed.astype(bool), noise_min_area)
    clean_mask[ignore_mask] = False

    if mandatory_px > 0:
        pixel_similarity = 1.0 - float(clean_mask[mandatory_mask].sum()) / mandatory_px
    else:
        pixel_similarity = 1.0

    edges_tmpl = cv2.Canny(gray_tmpl, cfg.canny_low, cfg.canny_high)
    edges_test = cv2.Canny(gray_test, cfg.canny_low, cfg.canny_high)
    edge_kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges_tmpl_d = cv2.dilate(edges_tmpl, edge_kernel, iterations=1)
    edges_test_d = cv2.dilate(edges_test, edge_kernel, iterations=1)

    edge_diff_mask = ((edges_tmpl_d > 0) ^ (edges_test_d > 0))
    edge_diff_mask[ignore_mask] = False
    edge_diff_mask = _remove_small_blobs(edge_diff_mask, min_area=max(5, noise_min_area // 4))
    edge_diff_mask[ignore_mask] = False

    tmpl_edges_mandatory = (edges_tmpl_d > 0) & mandatory_mask
    test_edges_mandatory = (edges_test_d > 0) & mandatory_mask
    total_edges = max(int(tmpl_edges_mandatory.sum() + test_edges_mandatory.sum()), 1)
    edge_similarity = 1.0 - float(edge_diff_mask.sum()) / float(total_edges)
    edge_similarity = float(np.clip(edge_similarity, 0, 1))

    tile_results = _analyse_tiles(
        template, aligned_test, combined_diff, cfg,
        ignore_mask=ignore_mask,
        grid_rows=grid_rows, grid_cols=grid_cols,
    )
    tile_pass_rate = (
        sum(1 for t in tile_results if t.passed) / max(len(tile_results), 1)
    )

    final = (
        cfg.weight_ssim  * ssim_score
        + cfg.weight_pixel * pixel_similarity
        + cfg.weight_edge  * edge_similarity
        + cfg.weight_tile  * tile_pass_rate
    )
    return ssim_score, pixel_similarity, edge_similarity, tile_pass_rate, float(final)


def _run_region_split(
    template:     np.ndarray,
    aligned_test: np.ndarray,
    ignore_mask:  np.ndarray,
    cfg:          ValidationConfig,
    full_ssim:    float,
    full_px:      float,
    full_edge:    float,
    full_tile:    float,
    full_final:   float,
    full_valid:   bool,
) -> Tuple[
    float, float, float, float, float, bool,
    List[RegionScore], List[str],
]:
    """
    Per-region metrics; verdict = all mandatory *quadrants* pass
    (optional-heavy quadrants are skipped).
    Headline score = unweighted mean of mandatory quadrant final_score values.
    """
    H, W  = template.shape[:2]
    thr   = cfg.region_optional_ignore_fraction
    out: List[RegionScore] = []

    def _log_region(rs: RegionScore) -> None:
        if rs.ignored_due_to_optional:
            logger.info(
                "Region %s: ignored due to optional zones (optional_fraction=%.3f)",
                rs.name, rs.optional_fraction,
            )
        else:
            assert rs.final_score is not None
            st = "PASS" if rs.passed else "FAIL"
            logger.info(
                "Region %s: score=%.4f %s  (SSIM=%.4f pixel=%.4f edge=%.4f tile=%.4f)",
                rs.name, rs.final_score, st,
                rs.ssim_score or 0, rs.pixel_similarity or 0,
                rs.edge_similarity or 0, rs.tile_pass_rate or 0,
            )

    for name, role, y, x, rh, rw in _iter_region_boxes(H, W):
        ign = ignore_mask[y:y + rh, x:x + rw]
        region_area = float(rh * rw)
        opt_area    = float(ign.sum())
        opt_frac    = opt_area / max(region_area, 1.0)

        is_quad = name in QUADRANT_NAMES
        if opt_frac > thr:
            rs = RegionScore(
                name=name, role=role, y=y, x=x, width=rw, height=rh,
                optional_fraction=opt_frac,
                ignored_due_to_optional=True,
                passed=True,
                counts_for_verdict=False,
            )
            out.append(rs)
            _log_region(rs)
            continue

        t_crop  = template[y:y + rh, x:x + rw]
        a_crop  = aligned_test[y:y + rh, x:x + rw]
        ign_crop = ignore_mask[y:y + rh, x:x + rw]
        n_min    = max(5, int(cfg.noise_min_area * (region_area / max(float(H * W), 1.0))))
        gr, gc   = max(2, cfg.grid_rows // 2), max(2, cfg.grid_cols // 2)

        ssim_v, px_v, ed_v, tl_v, fn_v = _region_metrics(
            t_crop, a_crop, ign_crop, cfg,
            grid_rows=gr, grid_cols=gc, noise_min_area=n_min,
        )
        reg_pass = (
            fn_v       >= cfg.valid_score_threshold
            and ssim_v >= cfg.ssim_threshold
        )
        rs = RegionScore(
            name=name, role=role, y=y, x=x, width=rw, height=rh,
            optional_fraction=opt_frac,
            ignored_due_to_optional=False,
            ssim_score=ssim_v, pixel_similarity=px_v,
            edge_similarity=ed_v, tile_pass_rate=tl_v,
            final_score=fn_v, passed=reg_pass,
            counts_for_verdict=is_quad,
        )
        out.append(rs)
        _log_region(rs)

    mandatory = [r for r in out if r.counts_for_verdict and not r.ignored_due_to_optional]
    failed = [r.name for r in mandatory if not r.passed]

    if not mandatory:
        logger.info(
            "Region split: no scored mandatory quadrants — falling back to full-image verdict"
        )
        return (
            full_ssim, full_px, full_edge, full_tile, full_final, full_valid,
            out, failed,
        )

    n = float(len(mandatory))
    agg_ssim  = sum(r.ssim_score       or 0.0 for r in mandatory) / n
    agg_px    = sum(r.pixel_similarity or 0.0 for r in mandatory) / n
    agg_edge  = sum(r.edge_similarity  or 0.0 for r in mandatory) / n
    agg_tile  = sum(r.tile_pass_rate   or 0.0 for r in mandatory) / n
    agg_final = sum(r.final_score      or 0.0 for r in mandatory) / n
    all_pass  = len(failed) == 0

    return (
        agg_ssim, agg_px, agg_edge, agg_tile, agg_final, all_pass,
        out, failed,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def compare(
    template:     np.ndarray,
    aligned_test: np.ndarray,
    cfg:          ValidationConfig = DEFAULT_CONFIG,
    alignment_ok: bool = True,
    ignore_mask:  np.ndarray | None = None,
    ignored_zones: list | None = None,
) -> ComparisonResult:
    """
    Run the full deep comparison of *aligned_test* against *template*.
    Both must be BGR uint8, same shape (call preprocess + align first).

    Parameters
    ----------
    ignore_mask   : bool H×W array where True = optional zone.
                    Pixels inside these zones are excluded from every signal.
                    Build with optional_zones.build_ignore_mask().
    ignored_zones : list of OptionalZone objects (stored in result for reporting).
    """
    assert template.shape == aligned_test.shape, (
        f"Shape mismatch {template.shape} vs {aligned_test.shape}. "
        "Call preprocess() + align() first."
    )

    H, W = template.shape[:2]

    # Normalise ignore_mask — must be bool H×W, default all-False
    if ignore_mask is None or ignore_mask.size == 1:
        ignore_mask = np.zeros((H, W), dtype=bool)
    mandatory_mask = ~ignore_mask          # True where we MUST match
    mandatory_px   = float(mandatory_mask.sum())

    gray_tmpl = cv2.cvtColor(template,     cv2.COLOR_BGR2GRAY)
    gray_test = cv2.cvtColor(aligned_test, cv2.COLOR_BGR2GRAY)

    # ── Signal 1: SSIM ────────────────────────────────────────────────────────
    # Blank out optional zones in both greys before SSIM so they don't
    # contribute to the structural similarity measurement.
    gray_tmpl_ssim = gray_tmpl.copy()
    gray_test_ssim = gray_test.copy()
    if ignore_mask.any():
        # Fill optional zones with the same value in both images → zero diff
        gray_tmpl_ssim[ignore_mask] = 128
        gray_test_ssim[ignore_mask] = 128

    ssim_score, ssim_map_raw = _ssim(
        gray_tmpl_ssim, gray_test_ssim,
        full=True,
        win_size=cfg.ssim_win_size,
        data_range=255,
    )
    # ssim_map_raw is in [-1, 1].  We want 0 = bad, 1 = good for display.
    ssim_map = np.clip(ssim_map_raw, 0, 1).astype(np.float32)

    # ── Signal 2: Pixel-level colour diff ─────────────────────────────────────
    # Per-channel absolute diff  (catches hue changes SSIM misses)
    diff_b = cv2.absdiff(template[:,:,0], aligned_test[:,:,0]).astype(np.float32)
    diff_g = cv2.absdiff(template[:,:,1], aligned_test[:,:,1]).astype(np.float32)
    diff_r = cv2.absdiff(template[:,:,2], aligned_test[:,:,2]).astype(np.float32)
    # Also greyscale diff
    diff_gray = cv2.absdiff(gray_tmpl, gray_test).astype(np.float32)

    # Combined diff = max of all four channels
    combined_diff = np.maximum.reduce([diff_b, diff_g, diff_r, diff_gray])

    raw_mask = combined_diff > cfg.diff_pixel_threshold
    # Zero out optional zones BEFORE morphology so they don't bleed into nearby
    # mandatory regions via the closing kernel.
    raw_mask[ignore_mask] = False

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (cfg.morph_kernel_size, cfg.morph_kernel_size),
    )
    closed     = cv2.morphologyEx(raw_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    clean_mask = _remove_small_blobs(closed.astype(bool), cfg.noise_min_area)
    # Re-apply after morphology to ensure no bleed-through
    clean_mask[ignore_mask] = False

    # Score only over mandatory pixels
    if mandatory_px > 0:
        pixel_similarity = 1.0 - float(clean_mask[mandatory_mask].sum()) / mandatory_px
    else:
        pixel_similarity = 1.0

    # ── Signal 3: Edge / structural diff (Canny) ──────────────────────────────
    edges_tmpl = cv2.Canny(gray_tmpl, cfg.canny_low, cfg.canny_high)
    edges_test = cv2.Canny(gray_test, cfg.canny_low, cfg.canny_high)

    # Dilate edges slightly before XOR — absorbs sub-pixel rendering variation
    edge_kernel    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges_tmpl_d   = cv2.dilate(edges_tmpl, edge_kernel, iterations=1)
    edges_test_d   = cv2.dilate(edges_test, edge_kernel, iterations=1)

    edge_diff_mask = ((edges_tmpl_d > 0) ^ (edges_test_d > 0))
    # Suppress optional zones from edge diff — before AND after blob removal
    # so that blobs that span the zone boundary don't survive.
    edge_diff_mask[ignore_mask] = False
    edge_diff_mask = _remove_small_blobs(edge_diff_mask, min_area=50)
    # Re-apply a final time: blob removal can reconnect fragments across the
    # zone boundary, leaving thin survivors that produce spurious edge regions.
    edge_diff_mask[ignore_mask] = False

    # Score only over mandatory edges
    tmpl_edges_mandatory = (edges_tmpl_d > 0) & mandatory_mask
    test_edges_mandatory = (edges_test_d > 0) & mandatory_mask
    total_edges = max(int(tmpl_edges_mandatory.sum() + test_edges_mandatory.sum()), 1)
    edge_similarity = 1.0 - float(edge_diff_mask.sum()) / float(total_edges)
    edge_similarity = float(np.clip(edge_similarity, 0, 1))

    # ── Signal 4: Grid tile analysis ──────────────────────────────────────────
    tile_results = _analyse_tiles(
        template, aligned_test, combined_diff, cfg, ignore_mask=ignore_mask,
    )
    tile_pass_rate = (
        sum(1 for t in tile_results if t.passed) / max(len(tile_results), 1)
    )

    # ── Named region scoring ──────────────────────────────────────────────────
    named_region_scores: Dict[str, float] = {}
    for name, (xf, yf, wf, hf) in cfg.named_regions.items():
        rx, ry = int(xf * W), int(yf * H)
        rw, rh = int(wf * W), int(hf * H)
        zone   = clean_mask[ry:ry + rh, rx:rx + rw]
        named_region_scores[name] = 1.0 - float(zone.sum()) / max(float(zone.size), 1)

    # ── Weighted final score ──────────────────────────────────────────────────
    final_score = (
        cfg.weight_ssim  * float(ssim_score)
        + cfg.weight_pixel * pixel_similarity
        + cfg.weight_edge  * edge_similarity
        + cfg.weight_tile  * tile_pass_rate
    )

    is_valid = (
        final_score       >= cfg.valid_score_threshold
        and float(ssim_score) >= cfg.ssim_threshold
    )

    full_image_ssim          = float(ssim_score)
    full_image_pixel_similarity = float(pixel_similarity)
    full_image_edge_similarity  = float(edge_similarity)
    full_image_tile_pass_rate   = float(tile_pass_rate)
    full_image_final_score      = float(final_score)
    full_image_is_valid         = bool(is_valid)

    region_scores:           List[RegionScore] = []
    failed_mandatory_regions:  List[str] = []
    comp_mode                 = cfg.comparison_mode

    if comp_mode == "region_split":
        (
            ssim_score,
            pixel_similarity,
            edge_similarity,
            tile_pass_rate,
            final_score,
            is_valid,
            region_scores,
            failed_mandatory_regions,
        ) = _run_region_split(
            template,
            aligned_test,
            ignore_mask,
            cfg,
            full_image_ssim,
            full_image_pixel_similarity,
            full_image_edge_similarity,
            full_image_tile_pass_rate,
            full_image_final_score,
            full_image_is_valid,
        )
        ssim_score       = float(ssim_score)
        pixel_similarity = float(pixel_similarity)
        edge_similarity  = float(edge_similarity)
        tile_pass_rate   = float(tile_pass_rate)
        final_score      = float(final_score)
        is_valid         = bool(is_valid)

    # ── Mismatch regions from pixel diff (full image) ───────────────────────
    px_regions = _extract_bboxes(
        clean_mask,
        combined_diff.astype(np.uint8),
        "pixel",
        ignore_mask=ignore_mask,
    )
    edge_regions = _extract_bboxes(
        edge_diff_mask,
        np.zeros_like(gray_tmpl),
        "edge",
        ignore_mask=ignore_mask,
    )

    # Merge and deduplicate (suppress edge regions that overlap pixel regions)
    all_regions = _merge_regions(px_regions, edge_regions)

    # ── Combined heatmap ──────────────────────────────────────────────────────
    # Blend: pixel diff (60%) + SSIM diff (40%)
    ssim_diff_u8 = ((1.0 - ssim_map) * 255).astype(np.uint8)
    diff_u8      = np.clip(combined_diff, 0, 255).astype(np.uint8)
    heatmap      = cv2.addWeighted(diff_u8, 0.6, ssim_diff_u8, 0.4, 0)

    logger.info(
        "Compare | full-image ref | SSIM=%.4f  pixel=%.4f  edge=%.4f  "
        "tile_pass=%.4f  final=%.4f  valid=%s",
        full_image_ssim, full_image_pixel_similarity,
        full_image_edge_similarity, full_image_tile_pass_rate,
        full_image_final_score, full_image_is_valid,
    )
    if comp_mode == "region_split":
        logger.info(
            "Compare | region_split verdict | SSIM=%.4f  pixel=%.4f  edge=%.4f  "
            "tile_pass=%.4f  final=%.4f  valid=%s  failed_mandatory=%s  "
            "mismatch_regions=%d  failed_tiles=%d",
            ssim_score, pixel_similarity, edge_similarity, tile_pass_rate,
            final_score, is_valid, failed_mandatory_regions,
            len(all_regions),
            sum(1 for t in tile_results if not t.passed),
        )
    else:
        logger.info(
            "Compare | mismatch_regions=%d  failed_tiles=%d",
            len(all_regions),
            sum(1 for t in tile_results if not t.passed),
        )

    return ComparisonResult(
        ssim_score=float(ssim_score),
        pixel_similarity=float(pixel_similarity),
        edge_similarity=float(edge_similarity),
        tile_pass_rate=float(tile_pass_rate),
        final_score=float(final_score),
        is_valid=is_valid,
        alignment_ok=alignment_ok,
        named_region_scores=named_region_scores,
        diff_mask=clean_mask,
        ssim_map=ssim_map,
        diff_heatmap=heatmap,
        edge_diff_mask=edge_diff_mask,
        mismatch_regions=all_regions,
        tile_results=tile_results,
        ignored_zones=ignored_zones or [],
        optional_zone_mask=ignore_mask,
        comparison_mode=comp_mode,
        region_scores=region_scores,
        failed_mandatory_regions=failed_mandatory_regions,
        full_image_ssim=full_image_ssim,
        full_image_pixel_similarity=full_image_pixel_similarity,
        full_image_edge_similarity=full_image_edge_similarity,
        full_image_tile_pass_rate=full_image_tile_pass_rate,
        full_image_final_score=full_image_final_score,
        full_image_is_valid=full_image_is_valid,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _remove_small_blobs(mask: np.ndarray, min_area: int) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    clean = np.zeros_like(mask, dtype=bool)
    for lbl in range(1, n):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == lbl] = True
    return clean


def _classify(area: int, diff_ratio: float) -> str:
    if diff_ratio > 0.55 or area > 8000:
        return "critical"
    if diff_ratio > 0.25 or area > 2500:
        return "moderate"
    return "minor"


def _extract_bboxes(
    mask:        np.ndarray,
    diff_map:    np.ndarray,
    source:      str,
    ignore_mask: np.ndarray | None = None,
) -> List[MismatchRegion]:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    regions = []
    for lbl in range(1, n):
        x    = int(stats[lbl, cv2.CC_STAT_LEFT])
        y    = int(stats[lbl, cv2.CC_STAT_TOP])
        w    = int(stats[lbl, cv2.CC_STAT_WIDTH])
        h    = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        area = int(stats[lbl, cv2.CC_STAT_AREA])

        # ── Bbox-level optional-zone filter (final safety net) ────────────────
        # Skip regions whose bbox lies mostly inside optional zones (ignores
        # thin morphological survivors at zone boundaries).
        if ignore_mask is not None and ignore_mask.any():
            box_area    = w * h
            bbox_ignore = ignore_mask[y:y + h, x:x + w]
            ignored_area = int(bbox_ignore.sum())
            if box_area > 0 and (ignored_area / float(box_area)) > 0.80:
                continue

        zone_mask  = mask[y:y + h, x:x + w]
        diff_ratio = float(zone_mask.sum()) / max(area, 1)
        severity   = _classify(area, diff_ratio)

        regions.append(MismatchRegion(
            x=x, y=y, w=w, h=h,
            pixel_diff_ratio=diff_ratio,
            severity=severity,
            source=source,
        ))

    regions.sort(key=lambda r: r.area, reverse=True)
    return regions


def _iou(a: MismatchRegion, b: MismatchRegion) -> float:
    """Intersection-over-union for two bounding boxes."""
    ax2, ay2 = a.x + a.w, a.y + a.h
    bx2, by2 = b.x + b.w, b.y + b.h
    ix = max(0, min(ax2, bx2) - max(a.x, b.x))
    iy = max(0, min(ay2, by2) - max(a.y, b.y))
    inter = ix * iy
    if inter == 0:
        return 0.0
    union = a.area + b.area - inter
    return inter / max(union, 1)


def _merge_regions(
    primary:   List[MismatchRegion],
    secondary: List[MismatchRegion],
    iou_threshold: float = 0.3,
) -> List[MismatchRegion]:
    """
    Combine pixel and edge region lists.
    Edge regions that overlap heavily with pixel regions are suppressed
    (they describe the same real mismatch).
    Non-overlapping edge regions are kept — they flag structural changes
    not caught by pixel diff.
    """
    merged = list(primary)
    for sec in secondary:
        overlaps = any(_iou(sec, p) >= iou_threshold for p in primary)
        if not overlaps:
            merged.append(sec)
    merged.sort(key=lambda r: r.area, reverse=True)
    return merged


def _analyse_tiles(
    template:      np.ndarray,
    aligned_test:  np.ndarray,
    combined_diff: np.ndarray,
    cfg:           ValidationConfig,
    ignore_mask:   np.ndarray | None = None,
    grid_rows:     int | None = None,
    grid_cols:     int | None = None,
) -> List[TileResult]:
    """
    Divide the label into a grid and score each tile independently.
    Uses the already-computed combined_diff map for speed.
    Pixels inside ignore_mask are excluded from each tile's diff ratio.
    """
    H, W = template.shape[:2]
    gr = cfg.grid_rows if grid_rows is None else grid_rows
    gc = cfg.grid_cols if grid_cols is None else grid_cols
    gr = max(1, gr)
    gc = max(1, gc)
    results = []

    tile_h = H // gr
    tile_w = W // gc

    thresh = cfg.diff_pixel_threshold

    for row in range(gr):
        for col in range(gc):
            y  = row * tile_h
            x  = col * tile_w
            th = tile_h if row < gr - 1 else H - y
            tw = tile_w if col < gc - 1 else W - x

            tile_diff = combined_diff[y:y + th, x:x + tw]

            if ignore_mask is not None and ignore_mask.any():
                tile_ignore = ignore_mask[y:y + th, x:x + tw]
                tile_mandatory = ~tile_ignore
                mandatory_px = float(tile_mandatory.sum())
                if mandatory_px == 0:
                    # Tile is entirely optional — always passes
                    results.append(TileResult(
                        row=row, col=col,
                        x=x, y=y, w=tw, h=th,
                        pixel_sim=1.0,
                        passed=True,
                    ))
                    continue
                diff_px    = float(((tile_diff > thresh) & tile_mandatory).sum())
                diff_ratio = diff_px / mandatory_px
            else:
                diff_px    = float((tile_diff > thresh).sum())
                total_px   = float(tile_diff.size)
                diff_ratio = diff_px / max(total_px, 1)

            pixel_sim = 1.0 - diff_ratio
            passed    = diff_ratio <= cfg.tile_fail_threshold

            results.append(TileResult(
                row=row, col=col,
                x=x, y=y, w=tw, h=th,
                pixel_sim=pixel_sim,
                passed=passed,
            ))

    return results