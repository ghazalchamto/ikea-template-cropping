"""
src/core/annotator.py
=====================
Generates all visual output assets.

annotate()        Test image with colour-coded bounding boxes + verdict banner.
make_heatmap()    False-colour JET overlay showing diff intensity.
make_grid_map()   Colour-coded tile pass/fail grid (unique to this system).
make_side_by_side() [Template | Test | Heatmap] strip.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from src.core.comparator import ComparisonResult, MismatchRegion
from src.config.settings import ValidationConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)

# BGR palette
_RED      = (30,  30,  220)
_ORANGE   = (30, 130, 230)
_YELLOW   = (30, 210, 230)
_GREEN    = (30, 170,  30)
_GREEN_BG = (20, 120,  20)
_RED_BG   = (25,  25, 185)
_WHITE    = (255, 255, 255)
_BLACK    = (  0,   0,   0)
_GREY     = (120, 120, 120)
_CYAN     = (200, 180,  30)   # optional-zone outline (BGR teal/cyan)

_SEVERITY_COLOR = {
    "critical": _RED,
    "moderate": _ORANGE,
    "minor":    _YELLOW,
}
_SOURCE_DASH = {
    "pixel": False,   # solid
    "edge":  True,    # dashed (structural)
    "tile":  False,
}


def annotate(
    test_image: np.ndarray,
    result:     ComparisonResult,
    cfg:        ValidationConfig = DEFAULT_CONFIG,
) -> np.ndarray:
    """
    Returns the test image with mismatch overlays and a verdict banner.
    Image is scaled by cfg.output_scale before returning.
    """
    img = test_image.copy()
    H, W = img.shape[:2]

    # ── Optional zone overlays (drawn first, underneath mismatch boxes) ───────
    for zone in result.ignored_zones:
        # Semi-transparent fill
        overlay = img.copy()
        cv2.rectangle(overlay,
                      (zone.x, zone.y),
                      (zone.x + zone.width, zone.y + zone.height),
                      _CYAN, -1)
        cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)
        # Dashed cyan border
        _draw_dashed_rect(img, zone.x, zone.y,
                          zone.x + zone.width, zone.y + zone.height,
                          _CYAN, thickness=2, dash=10)
        _put_text(img, f"OPT: {zone.name}", zone.x + 4, zone.y - 6, _CYAN, scale=0.42)

    # ── Mismatch bounding boxes ───────────────────────────────────────────────
    for i, reg in enumerate(result.mismatch_regions):
        color  = _SEVERITY_COLOR.get(reg.severity, _RED)
        dashed = _SOURCE_DASH.get(reg.source, False)

        if dashed:
            _draw_dashed_rect(img, reg.x, reg.y, reg.x + reg.w, reg.y + reg.h,
                              color, cfg.bbox_thickness)
        else:
            cv2.rectangle(img, (reg.x, reg.y),
                          (reg.x + reg.w, reg.y + reg.h),
                          color, cfg.bbox_thickness)

        tag = (f"#{i+1} {reg.severity[0].upper()}"
               f"{'*' if reg.source == 'edge' else ''}"
               f" {reg.pixel_diff_ratio*100:.0f}%")
        _put_text(img, tag, reg.x + 4, reg.y - 6, color)

    # ── Alignment warning ─────────────────────────────────────────────────────
    if not result.alignment_ok:
        cv2.rectangle(img, (0, 0), (W - 1, 32), (0, 130, 220), -1)
        cv2.putText(img,
                    "WARNING: alignment fallback — geometric differences may affect accuracy",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _WHITE, 1, cv2.LINE_AA)

    # ── Legend ────────────────────────────────────────────────────────────────
    _draw_legend(img)

    # ── Verdict banner ────────────────────────────────────────────────────────
    banner_h = max(62, H // 16)
    banner   = np.zeros((banner_h, W, 3), dtype=np.uint8)
    bg_color = _GREEN_BG if result.is_valid else _RED_BG
    banner[:] = bg_color

    verdict = "VALID" if result.is_valid else "INVALID"
    cv2.putText(banner, verdict,
                (14, banner_h // 2 + 9),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, _WHITE, 2, cv2.LINE_AA)

    scores = (
        f"Score:{result.final_score:.3f}  "
        f"SSIM:{result.ssim_score:.3f}  "
        f"Pixel:{result.pixel_similarity:.3f}  "
        f"Edge:{result.edge_similarity:.3f}  "
        f"Tiles:{result.tile_pass_rate:.3f}  "
        f"Regions:{len(result.mismatch_regions)}"
    )
    cv2.putText(banner, scores,
                (155, banner_h // 2 + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, _WHITE, 1, cv2.LINE_AA)

    out = np.vstack([img, banner])

    if cfg.output_scale != 1.0:
        nw = int(out.shape[1] * cfg.output_scale)
        nh = int(out.shape[0] * cfg.output_scale)
        out = cv2.resize(out, (nw, nh), interpolation=cv2.INTER_AREA)

    return out


def make_heatmap(
    test_image: np.ndarray,
    result:     ComparisonResult,
    cfg:        ValidationConfig = DEFAULT_CONFIG,
) -> np.ndarray:
    """JET false-colour heatmap blended onto the test image."""
    jet     = cv2.applyColorMap(result.diff_heatmap, cv2.COLORMAP_JET)
    blended = cv2.addWeighted(
        test_image, 1.0 - cfg.heatmap_alpha,
        jet,        cfg.heatmap_alpha, 0,
    )
    if cfg.output_scale != 1.0:
        nw = int(blended.shape[1] * cfg.output_scale)
        nh = int(blended.shape[0] * cfg.output_scale)
        blended = cv2.resize(blended, (nw, nh), interpolation=cv2.INTER_AREA)
    return blended


def make_grid_map(
    result: ComparisonResult,
    cfg:    ValidationConfig = DEFAULT_CONFIG,
) -> np.ndarray:
    """
    Render a colour-coded tile grid showing pass (green) / fail (red) per tile.
    Each cell also shows its pixel_sim value as a number.
    Returns a BGR uint8 image.
    """
    cell_w, cell_h = 80, 60
    rows, cols     = cfg.grid_rows, cfg.grid_cols
    W = cols * cell_w
    H = rows * cell_h

    canvas = np.full((H, W, 3), 245, dtype=np.uint8)

    for tile in result.tile_results:
        x0 = tile.col * cell_w
        y0 = tile.row * cell_h

        if tile.passed:
            fill = (180, 230, 180)   # light green
            border = (60, 160, 60)
        else:
            fill   = (180, 180, 240)  # light red
            border = (30, 30, 200)

        cv2.rectangle(canvas, (x0, y0), (x0 + cell_w - 2, y0 + cell_h - 2), fill,  -1)
        cv2.rectangle(canvas, (x0, y0), (x0 + cell_w - 2, y0 + cell_h - 2), border, 1)

        label = f"{tile.pixel_sim*100:.1f}%"
        fs = 0.38
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        cv2.putText(canvas, label,
                    (x0 + (cell_w - tw) // 2, y0 + cell_h // 2 + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, _BLACK, 1, cv2.LINE_AA)

    # Title row
    title_h = 28
    title   = np.zeros((title_h, W, 3), dtype=np.uint8)
    title[:] = (50, 50, 50)
    n_fail   = sum(1 for t in result.tile_results if not t.passed)
    cv2.putText(title,
                f"Tile grid  ({rows}x{cols})  —  {n_fail} failed / {rows*cols} total",
                (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48, _WHITE, 1, cv2.LINE_AA)

    return np.vstack([title, canvas])


def make_side_by_side(
    template:     np.ndarray,
    aligned_test: np.ndarray,
    result:       ComparisonResult,
    scale:        float = 0.25,
) -> np.ndarray:
    """[Template | Test label | Diff heatmap] horizontal strip."""
    jet = cv2.applyColorMap(result.diff_heatmap, cv2.COLORMAP_JET)

    def _col(img: np.ndarray, label: str) -> np.ndarray:
        strip = np.zeros((30, img.shape[1], 3), dtype=np.uint8)
        strip[:] = (40, 40, 40)
        cv2.putText(strip, label, (6, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, _WHITE, 1, cv2.LINE_AA)
        return np.vstack([img, strip])

    combined = np.hstack([
        _col(template.copy(),     "TEMPLATE"),
        _col(aligned_test.copy(), "TEST LABEL"),
        _col(jet,                 "DIFF HEATMAP"),
    ])

    if scale != 1.0:
        nw = int(combined.shape[1] * scale)
        nh = int(combined.shape[0] * scale)
        combined = cv2.resize(combined, (nw, nh), interpolation=cv2.INTER_AREA)

    return combined


def save_outputs(
    annotated:    np.ndarray,
    heatmap:      np.ndarray,
    output_dir:   str | Path,
    stem:         str,
    grid_map:     Optional[np.ndarray] = None,
    side_by_side: Optional[np.ndarray] = None,
) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {}

    def _save(img, suffix):
        p = str(out / f"{stem}{suffix}")
        cv2.imwrite(p, img)
        return p

    paths["annotated"]   = _save(annotated,   "_annotated.png")
    paths["heatmap"]     = _save(heatmap,      "_heatmap.png")
    if grid_map     is not None:
        paths["grid_map"]    = _save(grid_map,     "_grid_map.png")
    if side_by_side is not None:
        paths["side_by_side"] = _save(side_by_side, "_comparison.png")

    return paths


# ─────────────────────────────────────────────────────────────────────────────
# Internal drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _put_text(
    img:   np.ndarray,
    text:  str,
    x:     int,
    y:     int,
    color: Tuple[int, int, int],
    scale: float = 0.46,
) -> None:
    y = max(y, 14)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, (x + 1, y + 1), font, scale, _BLACK, 2, cv2.LINE_AA)
    cv2.putText(img, text, (x,     y    ), font, scale, color,  1, cv2.LINE_AA)


def _draw_dashed_rect(
    img:   np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    color: Tuple[int, int, int],
    thickness: int = 2,
    dash: int = 12,
) -> None:
    """Draw a dashed rectangle (used for edge-sourced regions)."""
    pts = [
        ((x1, y1), (x2, y1)),
        ((x2, y1), (x2, y2)),
        ((x2, y2), (x1, y2)),
        ((x1, y2), (x1, y1)),
    ]
    for (ax, ay), (bx, by) in pts:
        length = max(abs(bx - ax), abs(by - ay))
        steps  = length // (dash * 2) if length > 0 else 1
        for s in range(steps):
            t0  = (s * 2 * dash) / max(length, 1)
            t1  = min((s * 2 * dash + dash) / max(length, 1), 1.0)
            px0 = int(ax + t0 * (bx - ax))
            py0 = int(ay + t0 * (by - ay))
            px1 = int(ax + t1 * (bx - ax))
            py1 = int(ay + t1 * (by - ay))
            cv2.line(img, (px0, py0), (px1, py1), color, thickness)


def _draw_legend(img: np.ndarray) -> None:
    H, W = img.shape[:2]
    items = [
        ("Critical  (solid)",   _RED),
        ("Moderate  (solid)",   _ORANGE),
        ("Minor     (solid)",   _YELLOW),
        ("Structural (dashed)", _RED),
        ("Optional zone",       _CYAN),
    ]
    x0     = W - 170
    y0     = 42
    lh     = 20
    bg_pad = 6

    # Semi-transparent background patch
    total_h = len(items) * lh + bg_pad * 2
    overlay = img.copy()
    cv2.rectangle(overlay, (x0 - bg_pad, y0 - bg_pad),
                  (W - 4, y0 + total_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    for i, (label, color) in enumerate(items):
        y = y0 + i * lh
        cv2.rectangle(img, (x0, y), (x0 + 12, y + 12), color, -1)
        cv2.putText(img, label, (x0 + 17, y + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, _WHITE, 1, cv2.LINE_AA)
