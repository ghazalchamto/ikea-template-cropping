"""
Completeness and cross-field consistency checker.

Answers the question: "what should be here but isn't, and do the fields
that ARE present agree with each other?"

Two sub-checks
──────────────

1. Missing elements
   - Required fields listed in label-rules/<type>.json that were not extracted
   - Expected barcode types (has_itf / has_datamatrix in index.json) not decoded
   - Zones that should contain content but have zero extracted words

2. Cross-field consistency (deterministic cross-checks)
   - ITF-14 barcode digits  ↔  human-readable ITF text
     The human-readable text printed below the bars must match the raw barcode
     data (minus check digit formatting differences).
   - DataMatrix AI(13) date  ↔  human_readable_date OCR'd from label face
     Both encode the same production/packaging date — a mismatch means the
     barcode content was updated without updating the printed text, or vice versa.
   - DataMatrix AI(240) item code  ↔  article_number
     AI(240) holds the 14-digit item product code; the article_number field
     holds the human-readable "XXX.XXX.XX" form.  They should resolve to the
     same internal item number.
   - EAN barcode value  ↔  ean_barcode_value (OCR'd check digits)

Returns a CompletenessReport with lists of MissingElement and ConsistencyIssue.

Implementation notes (TODO)
────────────────────────────
- Load the detected label type schema from label-rules/<type>.json.
- Iterate required_fields; any not present → MissingElement(severity="error").
- Iterate optional_fields; any not present → MissingElement(severity="warning").
- Check index.json has_itf / has_datamatrix flags vs decoded barcodes.
- For cross-field checks, normalise both sides before comparing
  (strip spaces, dashes; handle YYMMDD ↔ YY-MM-DD etc.).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from extractor.models import ExtractedLabel
from .models import MissingElement, ConsistencyIssue, CompletenessReport

logger = logging.getLogger(__name__)

_RULES_DIR = Path(__file__).resolve().parent.parent / "label-rules"


def check_completeness(
    label: ExtractedLabel,
    label_type_id: Optional[str] = None,
) -> CompletenessReport:
    """
    Run completeness and cross-field consistency checks on ``label``.

    Parameters
    ----------
    label           : fully-extracted label
    label_type_id   : detected label type id (used to load expected field list)
    """
    # TODO: load schema for label_type_id from label-rules/
    # TODO: check required_fields presence → MissingElement list
    # TODO: check expected barcode types → MissingElement list
    # TODO: run cross-field consistency checks → ConsistencyIssue list
    #
    # Stub: returns an empty report until implemented.
    logger.debug("completeness: check_completeness called (stub — not yet implemented)")
    return CompletenessReport(missing=[], consistency_issues=[])
