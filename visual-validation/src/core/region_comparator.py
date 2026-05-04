"""
src/core/region_comparator.py
=============================
Region-level visual similarity scoring.

Compares pre-extracted region image pairs (template vs candidate) and
returns per-region scores: SSIM, pixel similarity, edge similarity, and a
weighted combined score.

This module deliberately operates on REGION CLIPS, never on full pages.
A region clip is a small image produced by `pdf_region_extractor.extract_regions`,
sized exactly to the region's configured (width, height) in pixels.

Scoring metrics
---------------
1. SSIM (structural similarity, grayscale)
   Captures structural changes: shifted/missing/extra glyphs, shape changes,
   repositioned elements. Computed on grayscale clips with a small odd
   window clamped to the clip size (some regions are very narrow).

2. Pixel similarity
   Fraction of pixels whose grayscale intensity differs by ≤ tolerance.
   Robust to anti-aliasing jitter at glyph edges, which would otherwise
   produce false negatives on visually-identical content.

3. Edge similarity
   Canny edge maps of both clips, compared via XOR over union:
       edge_sim = 1 − (popcount(A XOR B) / popcount(A OR B))
   Returns 1.0 if both clips have no detected edges.

Combined score
--------------
   combined = w_ssim * ssim + w_pixel * pixel + w_edge * edge

Default weights are exposed as module-level constants. They are intentionally
weighted toward SSIM and pixel because Canny is the most volatile signal on
dense text content. A future step may override weights per region.

Optional regions
----------------
Optional regions are scored exactly the same way as mandatory ones — this
module is metric-only and does not participate in pass/fail. The aggregator
(future step) is responsible for excluding optional regions from the verdict.

Determinism
-----------
All operations are deterministic. No randomness, no thread-dependent ordering.
Repeated calls on identical inputs produce bit-identical outputs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim_skimage

from src.core.pdf_region_extractor import ExtractedRegion, PDFExtraction
from src.core.region_loader import Region, RegionConfig

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Tunable parameters (module-level so they're easy to adjust + audit)
# ─────────────────────────────────────────────────────────────────────────────

# Combined-score weights. Must sum to ~1.0 for the combined value to be
# interpretable as a similarity in [0,1]. The weighting deliberately reduces
# the influence of edge similarity, which is the noisiest of the three
# metrics on text-heavy content.
W_SSIM:  float = 0.50
W_PIXEL: float = 0.30
W_EDGE:  float = 0.20

# Pixel similarity: per-pixel grayscale intensity difference at or below
# this threshold (0–255 scale) counts the pixel as "the same". Tolerates
# sub-pixel rendering differences and minor anti-aliasing variation.
PIXEL_DIFF_TOLERANCE: int = 24

# Canny thresholds for edge similarity. Conservative low/high; tuned to
# pick up text and shape outlines without being saturated by JPEG noise.
CANNY_LOW:  int = 50
CANNY_HIGH: int = 150

# SSIM window size cap. SSIM requires win_size to be odd and ≤ min(H, W).
# We cap at 7 — typical for 8-bit greyscale on small image patches.
SSIM_WIN_MAX: int = 7

# Below this size in either dimension we skip SSIM entirely and fall back
# to pixel-only because SSIM windows can't fit. The combined score still
# computes; we just down-weight to (pixel + edge) and renormalize.
SSIM_MIN_SIDE: int = 3


# ─────────────────────────────────────────────────────────────────────────────
# Result types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RegionScore:
    region_name: str
    mandatory:   bool
    weight:      float
    ssim:        float          # 0.0 – 1.0
    pixel:       float          # 0.0 – 1.0
    edge:        float          # 0.0 – 1.0
    combined:    float          # 0.0 – 1.0
    # Diagnostic info — populated when known, may be empty otherwise.
    width:       int = 0
    height:      int = 0
    notes:       Tuple[str, ...] = field(default_factory=tuple)

    # Typed-validator metadata (optional; defaults preserve back-compat for
    # callers that produce RegionScore via the old comparator path).
    region_type: str            = ""    # canonical type from the region config
    method:      str            = ""    # validator method name, e.g. "text_layout"
    details:     Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "region_name": self.region_name,
            "mandatory":   self.mandatory,
            "weight":      self.weight,
            "scores": {
                "ssim":     round(self.ssim,     4),
                "pixel":    round(self.pixel,    4),
                "edge":     round(self.edge,     4),
                "combined": round(self.combined, 4),
            },
            "size":  [self.width, self.height],
            "notes": list(self.notes),
        }
        if self.region_type:
            d["region_type"] = self.region_type
        if self.method:
            d["method"] = self.method
        if self.details:
            d["details"] = self.details
        return d


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def compare_region(
    template_img:  np.ndarray,
    candidate_img: np.ndarray,
    region_name:   str   = "",
    mandatory:     bool  = True,
    weight:        float = 1.0,
) -> RegionScore:
    """
    Compare a single template/candidate region image pair.

    Both images must be BGR uint8 and ideally the same shape. If shapes
    differ, the candidate is resized to match the template (with a note
    recorded). Pure metric function — no I/O.
    """
    if template_img is None or candidate_img is None:
        raise ValueError(f"compare_region({region_name!r}): null image")
    if template_img.dtype != np.uint8 or candidate_img.dtype != np.uint8:
        raise ValueError(f"compare_region({region_name!r}): images must be uint8")

    notes: List[str] = []

    if template_img.shape != candidate_img.shape:
        notes.append(
            f"shape_mismatch:template={template_img.shape},"
            f"candidate={candidate_img.shape}"
        )
        candidate_img = cv2.resize(
            candidate_img,
            (template_img.shape[1], template_img.shape[0]),
            interpolation=cv2.INTER_AREA,
        )

    # Convert to grayscale once for all three metrics.
    t_gray = (
        cv2.cvtColor(template_img,  cv2.COLOR_BGR2GRAY)
        if template_img.ndim == 3 else template_img
    )
    c_gray = (
        cv2.cvtColor(candidate_img, cv2.COLOR_BGR2GRAY)
        if candidate_img.ndim == 3 else candidate_img
    )
    h, w = t_gray.shape

    ssim_score = _ssim_score(t_gray, c_gray, notes)
    pixel_sim  = _pixel_similarity(t_gray, c_gray)
    edge_sim   = _edge_similarity(t_gray, c_gray)

    combined = _combine_scores(
        ssim=ssim_score, pixel=pixel_sim, edge=edge_sim,
        ssim_valid=("ssim_skipped_too_small" not in notes),
    )

    return RegionScore(
        region_name=region_name,
        mandatory=mandatory,
        weight=weight,
        ssim=float(ssim_score),
        pixel=float(pixel_sim),
        edge=float(edge_sim),
        combined=float(combined),
        width=int(w), height=int(h),
        notes=tuple(notes),
    )


def compare_all_regions(
    region_list:    List[Region],
    template_dir:   str | Path,
    candidate_dir:  str | Path,
) -> List[RegionScore]:
    """
    Compare every region by loading clip PNGs from two directories.

    Expects clip filenames of the form `<region_name>.png` in both directories.
    Useful for offline re-scoring of previously extracted clips and for
    debug scripts that operate on saved outputs.
    """
    template_dir  = Path(template_dir)
    candidate_dir = Path(candidate_dir)
    if not template_dir.is_dir():
        raise FileNotFoundError(f"template_dir not a directory: {template_dir}")
    if not candidate_dir.is_dir():
        raise FileNotFoundError(f"candidate_dir not a directory: {candidate_dir}")

    scores: List[RegionScore] = []
    for region in region_list:
        t_path = template_dir  / f"{region.name}.png"
        c_path = candidate_dir / f"{region.name}.png"
        if not t_path.exists():
            raise FileNotFoundError(f"template clip missing: {t_path}")
        if not c_path.exists():
            raise FileNotFoundError(f"candidate clip missing: {c_path}")

        t_img = cv2.imread(str(t_path), cv2.IMREAD_COLOR)
        c_img = cv2.imread(str(c_path), cv2.IMREAD_COLOR)
        if t_img is None:
            raise IOError(f"cv2 could not decode {t_path}")
        if c_img is None:
            raise IOError(f"cv2 could not decode {c_path}")

        scores.append(compare_region(
            template_img=t_img,
            candidate_img=c_img,
            region_name=region.name,
            mandatory=region.mandatory,
            weight=region.weight,
        ))
    return scores


def compare_extractions(
    template:  PDFExtraction,
    candidate: PDFExtraction,
) -> List[RegionScore]:
    """
    Compare two PDFExtractions in memory (no disk round-trip).

    The two extractions must come from the same RegionConfig — i.e. they
    must have the same region names in the same order. This is the
    preferred entry point when you've just extracted both PDFs and want
    scores immediately.
    """
    t_by_name = template.by_name()
    c_by_name = candidate.by_name()

    if set(t_by_name) != set(c_by_name):
        only_t = sorted(set(t_by_name) - set(c_by_name))
        only_c = sorted(set(c_by_name) - set(t_by_name))
        raise ValueError(
            f"compare_extractions: region sets differ. "
            f"only-in-template={only_t} only-in-candidate={only_c}"
        )

    scores: List[RegionScore] = []
    for er_t in template.extracted:                # preserve template order
        er_c = c_by_name[er_t.region.name]
        scores.append(compare_region(
            template_img=er_t.image_bgr,
            candidate_img=er_c.image_bgr,
            region_name=er_t.region.name,
            mandatory=er_t.region.mandatory,
            weight=er_t.region.weight,
        ))
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Metric primitives
# ─────────────────────────────────────────────────────────────────────────────

def _ssim_score(
    t_gray: np.ndarray,
    c_gray: np.ndarray,
    notes:  List[str],
) -> float:
    """SSIM with adaptive window. Returns 1.0 for identical, 0.0 worst-case."""
    h, w = t_gray.shape
    if h < SSIM_MIN_SIDE or w < SSIM_MIN_SIDE:
        notes.append("ssim_skipped_too_small")
        return 1.0 if np.array_equal(t_gray, c_gray) else 0.0

    win = min(SSIM_WIN_MAX, h, w)
    if win % 2 == 0:
        win -= 1
    if win < 3:
        notes.append("ssim_skipped_too_small")
        return 1.0 if np.array_equal(t_gray, c_gray) else 0.0

    score = ssim_skimage(t_gray, c_gray, win_size=win, data_range=255)
    # SSIM can return small negatives on highly dissimilar content; clamp.
    return float(np.clip(score, 0.0, 1.0))


def _pixel_similarity(t_gray: np.ndarray, c_gray: np.ndarray) -> float:
    """Fraction of pixels within PIXEL_DIFF_TOLERANCE of each other."""
    diff = cv2.absdiff(t_gray, c_gray)
    if diff.size == 0:
        return 1.0
    different = int((diff > PIXEL_DIFF_TOLERANCE).sum())
    return 1.0 - (different / diff.size)


def _edge_similarity(t_gray: np.ndarray, c_gray: np.ndarray) -> float:
    """
    Edge similarity via Canny + XOR-over-union.

    Returns 1.0 when both edge maps are empty (no detected edges in either
    image — typically a blank region) since they are equally edge-less.
    """
    e_t = cv2.Canny(t_gray, CANNY_LOW, CANNY_HIGH) > 0
    e_c = cv2.Canny(c_gray, CANNY_LOW, CANNY_HIGH) > 0
    union = int((e_t | e_c).sum())
    if union == 0:
        return 1.0
    diff = int((e_t ^ e_c).sum())
    return 1.0 - (diff / union)


def _combine_scores(
    ssim:       float,
    pixel:      float,
    edge:       float,
    ssim_valid: bool,
) -> float:
    """
    Weighted combination. If SSIM was skipped (region too small), fall back
    to pixel + edge with their weights renormalized so the result stays in
    [0,1] and isn't artificially dragged down by a missing third of the score.
    """
    if ssim_valid:
        c = W_SSIM * ssim + W_PIXEL * pixel + W_EDGE * edge
    else:
        denom = W_PIXEL + W_EDGE
        c = (W_PIXEL * pixel + W_EDGE * edge) / denom if denom > 0 else 0.0
    return float(np.clip(c, 0.0, 1.0))
