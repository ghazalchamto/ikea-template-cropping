"""
QA contract tests for the extractor upload cache.

These tests verify that the cache written during extraction can be read back
for UI hydration and cleaned up deterministically.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

from PIL import Image

from server.main import _load_from_cache, _save_to_cache, cached, cached_delete


def _sample_pdf_bytes() -> bytes:
    p = Path("visual-validation/data/labels/label (1).pdf")
    return p.read_bytes()


def test_cache_roundtrip_and_hydrate_shape():
    data = _sample_pdf_bytes()
    file_hash = hashlib.sha256(data + b"600").hexdigest()[:16]

    buf = io.BytesIO()
    Image.new("RGB", (16, 12), "red").save(buf, format="PNG")

    meta = {
        "file_name": "label (1).pdf",
        "file_hash": file_hash,
        "dpi": 600,
        "is_pdf": True,
        "label": {
            "elements": {},
            "extraction_metadata": {"overall_confidence": 1.0},
        },
        "image_width": 16,
        "image_height": 12,
    }

    _save_to_cache(file_hash, data, ".pdf", meta, buf.getvalue())

    # Internal loader contract.
    loaded = _load_from_cache(file_hash)
    assert loaded is not None
    assert loaded["file_hash"] == file_hash
    assert loaded["file_name"] == "label (1).pdf"
    assert loaded["is_pdf"] is True
    assert isinstance(loaded.get("preview_png_base64"), str)
    assert len(loaded["preview_png_base64"]) > 0

    # Public endpoint contract (same shape used by frontend restore flow).
    hydrated = cached(file_hash)
    assert hydrated["file_hash"] == file_hash
    assert hydrated["image_width"] == 16
    assert hydrated["image_height"] == 12
    assert "preview_png_base64" in hydrated

    # Cleanup path used by "Clear cached upload".
    resp = cached_delete(file_hash)
    assert resp["file_hash"] == file_hash
    assert resp["removed"] >= 1
