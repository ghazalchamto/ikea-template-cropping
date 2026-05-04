"""
src/core/alignment.py
=====================
Pre-processing and geometric alignment.

Pipeline
--------
1.  Resize both images to the configured canonical resolution.
2.  Mild Gaussian blur to absorb PDF anti-aliasing and minor scan noise.
3.  CLAHE contrast enhancement to help ORB on low-contrast label areas.
4.  ORB keypoint detection + BF-Hamming matching + Lowe ratio test.
5.  RANSAC homography estimation + perspective warp.

Fallback: if ORB cannot find enough matches (happens on very plain /
mostly-white label areas), the function returns the resized test image
unchanged and sets success=False.  The comparison pipeline still runs
but results will reflect any residual geometric offset.
"""

from __future__ import annotations

import logging
from typing import Tuple

import cv2
import numpy as np

from src.config.settings import ValidationConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


def preprocess(
    image: np.ndarray,
    cfg: ValidationConfig = DEFAULT_CONFIG,
) -> np.ndarray:
    """
    Resize to canonical (target_width x target_height) and apply light blur.
    Returns BGR uint8.
    """
    resized = cv2.resize(
        image,
        (cfg.target_width, cfg.target_height),
        interpolation=cv2.INTER_AREA,
    )
    if cfg.gaussian_blur_sigma > 0:
        ks = _odd(int(cfg.gaussian_blur_sigma * 6) + 1)
        resized = cv2.GaussianBlur(resized, (ks, ks), cfg.gaussian_blur_sigma)
    return resized


def align(
    template: np.ndarray,
    test:     np.ndarray,
    cfg:      ValidationConfig = DEFAULT_CONFIG,
) -> Tuple[np.ndarray, bool]:
    """
    Warp *test* to align with *template* using ORB + RANSAC homography.
    Both images must be preprocessed (same size, BGR uint8).

    Returns (aligned_test, success_flag).
    """
    h, w = template.shape[:2]

    gray_t = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    gray_s = cv2.cvtColor(test,     cv2.COLOR_BGR2GRAY)

    # CLAHE — improves keypoint detection on flat / low-contrast label areas
    clahe = cv2.createCLAHE(
        clipLimit=cfg.clahe_clip_limit,
        tileGridSize=(cfg.clahe_tile_size, cfg.clahe_tile_size),
    )
    gray_t = clahe.apply(gray_t)
    gray_s = clahe.apply(gray_s)

    orb = cv2.ORB_create(
        nfeatures=cfg.orb_n_features,
        scaleFactor=1.2,
        nlevels=8,
        fastThreshold=10,
    )
    kp1, des1 = orb.detectAndCompute(gray_t, None)
    kp2, des2 = orb.detectAndCompute(gray_s, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        logger.warning(
            "ORB: too few keypoints (tmpl=%d  test=%d) — alignment skipped.",
            len(kp1) if kp1 else 0,
            len(kp2) if kp2 else 0,
        )
        return test.copy(), False

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw     = matcher.knnMatch(des1, des2, k=2)
    good    = [m for m, n in raw if m.distance < 0.75 * n.distance]

    logger.debug("ORB: %d raw  ->  %d good matches", len(raw), len(good))

    if len(good) < cfg.homography_min_matches:
        logger.warning(
            "ORB: only %d good matches (min=%d) — alignment skipped.",
            len(good), cfg.homography_min_matches,
        )
        return test.copy(), False

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(
        dst_pts, src_pts,
        cv2.RANSAC,
        cfg.homography_ransac_threshold,
    )

    if H is None:
        logger.warning("Homography failed — alignment skipped.")
        return test.copy(), False

    n_inliers = int(mask.sum()) if mask is not None else 0
    logger.info("Alignment: %d inliers / %d good matches", n_inliers, len(good))

    aligned = cv2.warpPerspective(
        test, H, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return aligned, True


def _odd(n: int) -> int:
    return n if n % 2 == 1 else n + 1
