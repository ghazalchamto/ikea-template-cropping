"""
Spatial / layout validator.

Checks that every extracted field lives in the zone where the label template
says it should be.  Uses the zone assignments from the extractor (each
ExtractedField carries a `zone` string and a `bbox`) and the expected zone
mapping from label-rules/<type>.json.

Checks performed
────────────────
1. Zone compliance  – Is field X in the expected zone for this label type?
2. Boundary check   – Is any bbox entirely outside the image boundary?
3. Barcode clear zone – Do text elements encroach on the barcode quiet zone?
4. Relative order   – Is the IKEA logo above the product name? etc.

Returns a LayoutReport with a list of LayoutIssue instances.

Implementation notes (TODO)
────────────────────────────
- Load expected zone mapping per label type from label-rules/<type>.json
  (the "required_fields" entries carry zone hints in some schemas).
- Cross-reference with ground_truth/shared_specs/zone_definitions.json for
  canonical zone boundaries expressed as normalised y-fractions.
- Compare each ExtractedField.zone against the expected zone.
- Use ExtractedField.bbox + image dimensions for boundary / clear-zone checks.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from extractor.models import ExtractedLabel, ExtractedField
from .models import LayoutIssue, LayoutReport

logger = logging.getLogger(__name__)

_RULES_DIR = Path(__file__).resolve().parent.parent / "label-rules"


def check_layout(
    label: ExtractedLabel,
    label_type_id: Optional[str] = None,
    image_width_px: Optional[int] = None,
    image_height_px: Optional[int] = None,
) -> LayoutReport:
    """
    Run all spatial checks on ``label`` and return a LayoutReport.

    Parameters
    ----------
    label           : fully-extracted label
    label_type_id   : detected label type (used to load expected zone map)
    image_width_px  : image width in pixels (for boundary checks)
    image_height_px : image height in pixels (for boundary checks)
    """
    # TODO: implement zone compliance checks
    # TODO: implement boundary checks
    # TODO: implement barcode clear-zone checks
    # TODO: implement relative ordering checks
    #
    # Stub: returns an empty report until implemented.
    logger.debug("layout_checker: check_layout called (stub — not yet implemented)")
    return LayoutReport(issues=[])
