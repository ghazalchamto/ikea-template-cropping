"""
Intermediate result persistence for the label dissection pipeline.

Each uploaded label gets its own sub-directory under ``output/``:

    output/
        <label_stem>/
            dissection.json   — raw multi-pass PDF dissection (preprocessing checkpoint)
            extraction.json   — field extraction result (ExtractedLabel.to_dict())

``dissection.json`` is written after run_pdf_passes() completes — before field
extraction and before any validation.  It is the canonical preprocessing output
that downstream validators consume.  Saving it separately means:

- Re-running validation does not require re-running OCR or barcode decoding.
- Intermediate state is inspectable for debugging.
- Multiple validators can consume the same dissection in parallel.

``extraction.json`` is the complete ExtractedLabel.to_dict() written at the
end of LabelExtractor.run().  It is the field-level view of the same label.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _output_root() -> Path:
    from config import OUTPUT_DIR  # late import avoids circular deps at module level
    return OUTPUT_DIR


def get_output_dir(label_path: str | Path) -> Path:
    """Return (and create) the output directory for a given label file."""
    d = _output_root() / Path(label_path).stem
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_dissection(
    label_path: str | Path,
    dissection: dict,
) -> Path:
    """
    Persist the full structured dissection to ``output/<stem>/dissection.json``.

    Called at the end of LabelExtractor.run() after LabelDissector.run().
    The dissection dict contains both the data_layer and layout_layer alongside
    raw_element_counts and meta.

    Args:
        label_path:  Source PDF / image path (determines the output sub-dir).
        dissection:  Full dict from LabelDissector.run().

    Returns:
        Path to the written JSON file.
    """
    out_dir = get_output_dir(label_path)
    out_path = out_dir / "dissection.json"

    payload = {
        "dissected_at": datetime.now(timezone.utc).isoformat(),
        "source_file":  Path(label_path).name,
        **dissection,
    }

    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )

    counts = dissection.get("raw_element_counts", {})
    n_text  = counts.get("text_pdf", 0) + counts.get("text_ocr", 0)
    n_paths = counts.get("path_rect", 0) + counts.get("path_line", 0) + counts.get("path_other", 0)
    n_img   = counts.get("image_pdf", 0)
    logger.info(
        "Dissection saved → %s  (text:%d  paths:%d  images:%d)",
        out_path, n_text, n_paths, n_img,
    )
    return out_path


def _json_default(obj):
    """Fallback serialiser for types json.dumps can't handle natively."""
    if hasattr(obj, "tolist"):   # numpy arrays
        return obj.tolist()
    if hasattr(obj, "item"):     # numpy scalars
        return obj.item()
    return str(obj)


def save_extraction(label_path: str | Path, label_dict: dict) -> Path:
    """
    Persist the full ``ExtractedLabel.to_dict()`` to ``output/<stem>/extraction.json``.

    Called at the end of ``LabelExtractor.run()``.

    Returns:
        Path to the written JSON file.
    """
    out_dir = get_output_dir(label_path)
    out_path = out_dir / "extraction.json"

    out_path.write_text(
        json.dumps(label_dict, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Extraction saved → %s", out_path)
    return out_path


def load_dissection(label_path: str | Path) -> Optional[dict]:
    """Load a cached dissection JSON if it exists, else return None."""
    out_path = _output_root() / Path(label_path).stem / "dissection.json"
    if out_path.exists():
        return json.loads(out_path.read_text(encoding="utf-8"))
    return None


def load_extraction(label_path: str | Path) -> Optional[dict]:
    """Load a cached extraction JSON if it exists, else return None."""
    out_path = _output_root() / Path(label_path).stem / "extraction.json"
    if out_path.exists():
        return json.loads(out_path.read_text(encoding="utf-8"))
    return None
