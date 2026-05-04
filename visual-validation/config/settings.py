"""
src/config/settings.py
======================
All tunable parameters for the visual validation system.

After your first real run on known-good labels, use scripts/calibrate.py
to find the right valid_score_threshold for your specific label stock.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class ValidationConfig:

    # ── Rendering ─────────────────────────────────────────────────────────────
    # 150 DPI = fast (good for layout diff)
    # 300 DPI = precise (recommended for production QA)
    render_dpi: int = 300

    # Canonical resolution every image is normalised to.
    # At 300 DPI:  A4  = 2480x3508   |   A5 = 1748x2480
    # At 150 DPI:  A4  = 1240x1754   |   A5 =  874x1240
    # Set these to match your actual label size after your first inspect run.
    target_width:  int = 1748
    target_height: int = 2480

    # ── Alignment ─────────────────────────────────────────────────────────────
    orb_n_features:              int   = 5000
    homography_min_matches:      int   = 10
    homography_ransac_threshold: float = 3.0
    clahe_clip_limit:            float = 3.0
    clahe_tile_size:             int   = 8

    # ── Pre-processing ────────────────────────────────────────────────────────
    gaussian_blur_sigma: float = 0.8

    # ── Pixel comparison ─────────────────────────────────────────────────────
    diff_pixel_threshold: int   = 18
    morph_kernel_size:    int   = 9
    noise_min_area:       int   = 200

    # ── SSIM ─────────────────────────────────────────────────────────────────
    ssim_win_size: int = 11

    # ── Edge / structural comparison ─────────────────────────────────────────
    canny_low:  int = 30
    canny_high: int = 100

    # ── Grid tile analysis ────────────────────────────────────────────────────
    grid_cols: int   = 8
    grid_rows: int   = 12
    tile_fail_threshold: float = 0.04

    # ── Decision weights ──────────────────────────────────────────────────────
    weight_ssim:  float = 0.35
    weight_pixel: float = 0.35
    weight_edge:  float = 0.15
    weight_tile:  float = 0.15

    valid_score_threshold: float = 0.88
    ssim_threshold:        float = 0.85

    # ── Comparison mode ──────────────────────────────────────────────────────
    # "full" = single global score drives verdict (legacy).
    # "region_split" = full-image metrics kept as reference; verdict from
    # mandatory quadrants only (see region_optional_ignore_fraction).
    comparison_mode: str = "region_split"

    # If optional (ignored) pixels cover more than this fraction of a region's
    # area, that region is skipped for scoring and does not fail the label.
    region_optional_ignore_fraction: float = 0.80

    # ── Annotation ───────────────────────────────────────────────────────────
    bbox_thickness: int   = 3
    heatmap_alpha:  float = 0.55
    output_scale:   float = 0.5

    # ── Named region zones (optional) ────────────────────────────────────────
    # {name: (x_frac, y_frac, w_frac, h_frac)}
    named_regions: Dict[str, Tuple[float, float, float, float]] = field(
        default_factory=dict
    )


DEFAULT_CONFIG = ValidationConfig()
