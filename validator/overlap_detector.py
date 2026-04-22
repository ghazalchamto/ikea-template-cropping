"""
Bounding-box overlap detector.

Checks every pair of extracted element bounding boxes for unexpected
intersection.  Uses purely deterministic geometry — no ML.

Overlap severity thresholds
────────────────────────────
  < 5%   → ignored (normal OCR jitter / adjacent text)
  5–25%  → warning  (elements are close; may indicate layout issue)
  ≥ 25%  → error    (elements are clearly on top of each other)

Special rule: any overlap between a text field and a barcode element is
always at least a warning, because text inside a barcode quiet zone will
cause scan failures.

Algorithm
─────────
For each pair (A, B) of extracted fields that both have a bbox:

    ix1 = max(A.x, B.x)
    iy1 = max(A.y, B.y)
    ix2 = min(A.x + A.w, B.x + B.w)
    iy2 = min(A.y + A.h, B.y + B.h)

    if ix2 > ix1 and iy2 > iy1:
        intersection_area = (ix2 - ix1) * (iy2 - iy1)
        smaller_area      = min(A.w * A.h, B.w * B.h)
        overlap_ratio     = intersection_area / smaller_area

Returns an OverlapReport with a list of OverlapResult instances.

Implementation notes (TODO)
────────────────────────────
- Collect all (field_name, bbox) pairs from ExtractedLabel attributes.
- Include barcode bboxes (itf14_barcode.bbox, datamatrix_barcode.bbox, etc.).
- Run pairwise intersection; skip pairs where either bbox is None.
- Apply thresholds to assign severity.
- Return OverlapReport.
"""

from __future__ import annotations

import logging
from typing import Optional

from extractor.models import ExtractedLabel
from .models import OverlapResult, OverlapReport

logger = logging.getLogger(__name__)


def detect_overlaps(label: ExtractedLabel) -> OverlapReport:
    """
    Detect all unexpected bounding-box overlaps in ``label``.

    Returns an OverlapReport with one OverlapResult per overlapping pair.
    """
    # TODO: collect all (name, bbox) pairs from ExtractedLabel
    # TODO: run pairwise intersection-over-union
    # TODO: apply severity thresholds
    # TODO: flag barcode quiet-zone violations as special case
    #
    # Stub: returns an empty report until implemented.
    logger.debug("overlap_detector: detect_overlaps called (stub — not yet implemented)")
    return OverlapReport(overlaps=[])
