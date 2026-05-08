"""
FastAPI server for the label extraction API + human corrections.

Run from repo root:
  pip install -r requirements.txt
  uvicorn server.main:app --reload --port 8000
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Repo root (parent of server/)
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import DEFAULT_DPI, FINETUNE_DATA_DIR, UPLOAD_CACHE_DIR
from extractor import LabelExtractor, warmup_ocr_readers
from finetune_store import (
    all_field_choices,
    apply_corrections,
    build_training_records,
    new_export_path,
    save_jsonl,
)
from label_serde import label_from_dict
from label_upload import load_image_matching_extractor, materialise, sniff_suffix
from pdf_label_edit import apply_text_edits_to_pdf_bytes, HAS_FITZ
from validator import auto_validate, check_completeness, check_layout, detect_overlaps
from validator.models import TypeMatch

# Upload cache helpers — see config.UPLOAD_CACHE_DIR for the rationale.
_HASH_RE = __import__("re").compile(r"^[A-Za-z0-9]{8,64}$")


def _cache_paths(file_hash: str) -> tuple[Path, Path, Path]:
    if not _HASH_RE.match(file_hash or ""):
        raise HTTPException(400, "Invalid file_hash")
    base = UPLOAD_CACHE_DIR / file_hash
    return (
        base.with_suffix(".bin"),
        base.with_suffix(".json"),
        base.with_suffix(".preview.png"),
    )


def _save_to_cache(
    file_hash: str,
    raw_bytes: bytes,
    suffix: str,
    meta: dict,
    preview_png: Optional[bytes],
) -> None:
    UPLOAD_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    bin_path, json_path, png_path = _cache_paths(file_hash)
    bin_path.write_bytes(raw_bytes)
    bin_path.with_suffix(bin_path.suffix + ".meta").write_text(
        json.dumps({"suffix": suffix}), encoding="utf-8"
    )
    json_path.write_text(
        json.dumps(meta, ensure_ascii=False), encoding="utf-8"
    )
    if preview_png is not None:
        png_path.write_bytes(preview_png)


def _load_from_cache(file_hash: str) -> Optional[dict]:
    _bin_path, json_path, png_path = _cache_paths(file_hash)
    if not json_path.is_file():
        return None
    try:
        meta = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if png_path.is_file():
        meta["preview_png_base64"] = base64.b64encode(
            png_path.read_bytes()
        ).decode("ascii")
    return meta


def cached_bytes_path(file_hash: str) -> Optional[Path]:
    """Return the on-disk path for cached upload bytes, or None if missing."""
    bin_path, _, _ = _cache_paths(file_hash)
    return bin_path if bin_path.is_file() else None

# Warm OCR once at import (same pattern as app.py)
try:
    warmup_ocr_readers()
except Exception:
    pass

app = FastAPI(title="IKEA Label API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _typematch_to_dict(tm: TypeMatch) -> dict:
    return {
        "label_type_id": tm.label_type_id,
        "label_type_name": tm.label_type_name,
        "score": float(tm.score),
    }


def _image_to_b64(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/fields")
def get_fields() -> list[dict]:
    return [
        {"id": fid, "label": label} for fid, label in all_field_choices()
    ]


class MergeValidateBody(BaseModel):
    label: dict = Field(..., description="Result of ExtractedLabel.to_dict() from /api/extract")
    corrections: dict[str, dict[str, Any]] = Field(default_factory=dict)


class ExportJsonlBody(BaseModel):
    file_hash: str
    dpi: int
    label: dict
    corrections: dict[str, dict[str, Any]] = Field(default_factory=dict)


@app.post("/api/extract")
async def extract(
    file: UploadFile = File(...),
    dpi: int = Form(DEFAULT_DPI),
) -> dict:
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file")
    path: Optional[str] = None
    try:
        path = materialise(data)
        ext = LabelExtractor(path, dpi=dpi)
        result = ext.run()
        file_hash = hashlib.sha256(data + str(dpi).encode()).hexdigest()[:16]
        prev: Any = None
        try:
            prev = load_image_matching_extractor(data, dpi)
        except Exception:
            pass
        preview_bytes: Optional[bytes] = None
        b64: Optional[str] = None
        if prev is not None:
            buf = io.BytesIO()
            prev.save(buf, format="PNG", optimize=True)
            preview_bytes = buf.getvalue()
            b64 = base64.b64encode(preview_bytes).decode("ascii")
        suffix = sniff_suffix(data) or ""
        meta = {
            "file_name": file.filename,
            "file_hash": file_hash,
            "dpi": dpi,
            "is_pdf": suffix == ".pdf",
            "label": result.to_dict(),
            "image_width": int(result.metadata.image_width_px),
            "image_height": int(result.metadata.image_height_px),
        }
        try:
            _save_to_cache(file_hash, data, suffix, meta, preview_bytes)
        except OSError as cache_err:
            # Cache failures must not block extraction.
            logging.getLogger(__name__).warning(
                "upload cache write failed for %s: %s", file_hash, cache_err
            )
        return {**meta, "preview_png_base64": b64}
    except Exception as e:
        raise HTTPException(500, str(e)) from e
    finally:
        if path and os.path.isfile(path):
            try:
                os.unlink(path)
            except OSError:
                pass


@app.get("/api/cached/{file_hash}")
def cached(file_hash: str) -> dict:
    """Hydrate the extractor UI after a page refresh without re-uploading.

    Returns the same payload shape as ``/api/extract`` so the frontend can
    drop it straight into state.
    """
    payload = _load_from_cache(file_hash)
    if payload is None:
        raise HTTPException(404, f"No cached upload for {file_hash}")
    return payload


@app.delete("/api/cached/{file_hash}")
def cached_delete(file_hash: str) -> dict:
    bin_path, json_path, png_path = _cache_paths(file_hash)
    removed = 0
    for p in (bin_path, json_path, png_path,
              bin_path.with_suffix(bin_path.suffix + ".meta")):
        if p.is_file():
            try:
                p.unlink()
                removed += 1
            except OSError:
                pass
    return {"removed": removed, "file_hash": file_hash}


@app.post("/api/merge-validate")
def merge_validate(body: MergeValidateBody) -> dict:
    try:
        el = label_from_dict(body.label)
    except Exception as e:
        raise HTTPException(400, f"Invalid label: {e}") from e
    merged = apply_corrections(el, body.corrections)
    vrep, shortlist = auto_validate(merged)
    layout = check_layout(
        merged,
        vrep.label_type_id,
        merged.metadata.image_width_px,
        merged.metadata.image_height_px,
    )
    overlaps = detect_overlaps(merged)
    complete = check_completeness(merged, vrep.label_type_id)
    return {
        "merged": merged.to_dict(),
        "validation": vrep.to_dict(),
        "type_shortlist": [_typematch_to_dict(t) for t in shortlist],
        "layout": layout.to_dict(),
        "overlap": overlaps.to_dict() if hasattr(overlaps, "to_dict") else {},
        "completeness": complete.to_dict() if hasattr(complete, "to_dict") else {},
    }


@app.post("/api/export-jsonl")
def export_jsonl(body: ExportJsonlBody) -> JSONResponse:
    try:
        model = label_from_dict(body.label)
    except Exception as e:
        raise HTTPException(400, f"Invalid label: {e}") from e
    recs = build_training_records(
        body.file_hash, body.dpi, model, body.corrections
    )
    text = "\n".join(json.dumps(r, ensure_ascii=False) for r in recs) + "\n"
    if not recs:
        return JSONResponse({"lines": 0, "ndjson": ""})
    p = new_export_path(FINETUNE_DATA_DIR)
    try:
        save_jsonl(p, recs)
    except Exception as e:
        raise HTTPException(500, str(e)) from e
    return JSONResponse(
        {
            "lines": len(recs),
            "path": str(p),
            "ndjson": text,
        }
    )


@app.post("/api/pdf-apply")
async def pdf_apply(
    file: UploadFile = File(...),
    dpi: int = Form(DEFAULT_DPI),
    edits: str = Form("[]"),
) -> JSONResponse:
    if not HAS_FITZ:
        raise HTTPException(500, "PyMuPDF not installed")
    data = await file.read()
    if data[:4] != b"%PDF":
        raise HTTPException(400, "Expected PDF file")
    try:
        edit_list: list[dict] = json.loads(edits) if isinstance(edits, str) else edits
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"Invalid edits JSON: {e}") from e
    if not edit_list:
        raise HTTPException(400, "edits is empty")
    try:
        out = apply_text_edits_to_pdf_bytes(data, edit_list, dpi=dpi)
    except Exception as e:
        raise HTTPException(500, str(e)) from e
    b64 = base64.b64encode(out).decode("ascii")
    return JSONResponse(
        {
            "pdf_base64": b64,
            "bytes_length": len(out),
        }
    )
