"""
QA contract tests for hash-based validation.

This verifies the key behavior requested by QA users: validation can execute
from server-side cache via ``file_hash`` (no browser re-upload required).
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import io
import sys
from pathlib import Path

from PIL import Image

from server.main import _save_to_cache, cached_delete


def _sample_pdf_bytes() -> bytes:
    p = Path("visual-validation/data/labels/label (1).pdf")
    return p.read_bytes()


def _write_cached_sample(file_hash: str, data: bytes) -> None:
    buf = io.BytesIO()
    Image.new("RGB", (10, 10), "green").save(buf, format="PNG")
    meta = {
        "file_name": "label (1).pdf",
        "file_hash": file_hash,
        "dpi": 600,
        "is_pdf": True,
        "label": {},
        "image_width": 10,
        "image_height": 10,
    }
    _save_to_cache(file_hash, data, ".pdf", meta, buf.getvalue())


def test_validate_with_file_hash_runs_without_upload():
    data = _sample_pdf_bytes()
    file_hash = hashlib.sha256(data + b"600").hexdigest()[:16]
    _write_cached_sample(file_hash, data)

    # Import validation service directly from visual-validation package tree.
    vv_root = Path("visual-validation").resolve()
    vv_main = vv_root / "server" / "main.py"
    spec = importlib.util.spec_from_file_location("vv_server_main", vv_main)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load validation server module from {vv_main}")
    module = importlib.util.module_from_spec(spec)
    if str(vv_root) not in sys.path:
        sys.path.insert(0, str(vv_root))
    spec.loader.exec_module(module)
    validate = module.validate

    result = asyncio.run(
        validate(
            file=None,
            file_hash=file_hash,
            product_code="L10555-MIADM",
            version=None,
            page=0,
            layout_only=False,  # More robust in local test fixtures.
            layout_tolerance_mm=1.5,
            allow_auto_schema=False,
        )
    )

    assert result["product_code"] == "L10555-MIADM"
    assert result["verdict"] in {"VALID", "INVALID"}
    assert isinstance(result["final_score"], float)
    assert len(result["region_results"]) > 0
    assert isinstance(result.get("candidate_overview_png_base64"), str)
    assert isinstance(result.get("template_overview_png_base64"), str)

    cached_delete(file_hash)

