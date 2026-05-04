"""
src/core/optional_zones.py
==========================
Load and apply optional-zone masks per product code.

Optional zones define rectangular areas of a label where differences
should NOT count against the score — e.g. fields that may legitimately
be empty on some valid labels.

Zone JSON format  (config/optional_zones/<PRODUCT_CODE>.json)
-------------------------------------------------------------
{
    "product_code": "1C50-CAA",
    "optional_zones": [
        {
            "name": "optional_field_name",
            "x": 100,
            "y": 200,
            "width": 300,
            "height": 80,
            "reason": "This field may be empty on some variants"
        }
    ]
}

Coordinates are in pixels relative to the canonical (preprocessed) image
size defined by ValidationConfig.target_width / target_height.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_ZONES_DIR = Path(__file__).resolve().parent.parent.parent / "config" / "optional_zones"


@dataclass
class OptionalZone:
    name:   str
    x:      int
    y:      int
    width:  int
    height: int
    reason: str = ""

    def as_dict(self) -> dict:
        return {
            "name":   self.name,
            "x":      self.x,
            "y":      self.y,
            "width":  self.width,
            "height": self.height,
            "reason": self.reason,
        }


def load_optional_zones(
    product_code: str,
    zones_dir: str | Path | None = None,
) -> List[OptionalZone]:
    """
    Load optional zones for *product_code* from the zones directory.
    Returns an empty list if no config file exists (all regions mandatory).
    """
    d = Path(zones_dir) if zones_dir else _DEFAULT_ZONES_DIR
    json_path = d / f"{product_code}.json"

    if not json_path.exists():
        logger.debug("No optional-zones config for %s (looked at %s)", product_code, json_path)
        return []

    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to parse optional-zones JSON %s: %s", json_path, exc)
        return []

    zones = []
    for entry in data.get("optional_zones", []):
        try:
            zones.append(OptionalZone(
                name=entry["name"],
                x=int(entry["x"]),
                y=int(entry["y"]),
                width=int(entry["width"]),
                height=int(entry["height"]),
                reason=entry.get("reason", ""),
            ))
        except (KeyError, ValueError) as exc:
            logger.warning("Skipping malformed optional-zone entry %s: %s", entry, exc)

    logger.info(
        "Optional zones loaded for %s: %d zone(s) — %s",
        product_code,
        len(zones),
        [z.name for z in zones],
    )
    return zones


def build_ignore_mask(
    zones: List[OptionalZone],
    height: int,
    width: int,
) -> np.ndarray:
    """
    Build a boolean mask (H×W) where True = optional zone (ignore differences).
    If zones is empty the mask is all False (nothing ignored).
    """
    mask = np.zeros((height, width), dtype=bool)
    for z in zones:
        x1 = max(0, z.x)
        y1 = max(0, z.y)
        x2 = min(width,  z.x + z.width)
        y2 = min(height, z.y + z.height)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = True
    return mask
