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

from config import DEFAULT_DPI, FINETUNE_DATA_DIR
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
        b64: Optional[str] = None
        if prev is not None:
            b64 = _image_to_b64(prev)
        return {
            "file_name": file.filename,
            "file_hash": file_hash,
            "dpi": dpi,
            "is_pdf": sniff_suffix(data) == ".pdf",
            "label": result.to_dict(),
            "preview_png_base64": b64,
            "image_width": int(result.metadata.image_width_px),
            "image_height": int(result.metadata.image_height_px),
        }
    except Exception as e:
        raise HTTPException(500, str(e)) from e
    finally:
        if path and os.path.isfile(path):
            try:
                os.unlink(path)
            except OSError:
                pass


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
