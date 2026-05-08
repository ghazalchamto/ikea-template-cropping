"""
FastAPI server for the visual-validation region pipeline.

Wraps the existing CLI (``scripts/validate_label.py`` →
``src.reporting.pipeline_runner.run_region_report_pipeline``) so the React UI
can drive layout validation without invoking ``python scripts/...`` shell
commands.

Run from the ``visual-validation/`` directory so the ``src`` package and
default ``config/regions`` / ``data/templates/...`` paths resolve correctly:

    cd visual-validation
    PYTHONPATH=. uvicorn server.main:app --reload --port 8001

Endpoints
---------
GET  /api/health            Liveness probe used by the launcher script.
GET  /api/products          List available product codes + versions from the
                            ground-truth catalog (drives the dropdown).
POST /api/validate          Multipart upload + run the region pipeline. Returns
                            verdict, per-region results, base64 overview PNGs
                            for the template and candidate, and the full HTML
                            report so the frontend can show side-by-side images
                            and link to the deep-dive.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# Repo root for this microservice = ``visual-validation/`` (parent of server/).
_VV_ROOT = Path(__file__).resolve().parent.parent
if str(_VV_ROOT) not in sys.path:
    sys.path.insert(0, str(_VV_ROOT))

from src.core.region_loader import default_regions_dir  # noqa: E402
from src.core.template_store import GroundTruthTemplateCatalog  # noqa: E402
from src.reporting.pipeline_runner import (  # noqa: E402
    RegionPipelineError,
    run_region_report_pipeline,
)

logger = logging.getLogger(__name__)

# Default ground-truth root used by the CLI (`--ground-truth-dir` default).
_DEFAULT_GROUND_TRUTH_DIR = (
    _VV_ROOT
    / "data"
    / "templates"
    / "ground-truth-ikea-labels"
    / "ground_truth"
)
_DEFAULT_GENERATED_REGIONS_DIR = (
    _VV_ROOT / "outputs" / "generated_regions"
)

# Shared upload cache. The extractor (server.main) writes uploads here keyed
# by ``file_hash``; we read them back so the browser can re-run validation
# without re-uploading bytes. Default points at ``<repo_root>/output/uploads``
# which matches ``config.UPLOAD_CACHE_DIR`` in the extractor.
_DEFAULT_UPLOAD_CACHE_DIR = _VV_ROOT.parent / "output" / "uploads"

# Hash format mirrors server/main.py — sha256 truncated hex.
_HASH_RE = __import__("re").compile(r"^[A-Za-z0-9]{8,64}$")

app = FastAPI(title="IKEA Visual Validation API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _ground_truth_dir() -> Path:
    return Path(
        os.environ.get("VISUAL_VALIDATION_GROUND_TRUTH_DIR")
        or _DEFAULT_GROUND_TRUTH_DIR
    ).resolve()


def _regions_dir() -> Path:
    env = os.environ.get("VISUAL_VALIDATION_REGIONS_DIR")
    return Path(env).resolve() if env else default_regions_dir()


def _upload_cache_dir() -> Path:
    env = os.environ.get("EXTRACTOR_UPLOAD_CACHE_DIR")
    return Path(env).resolve() if env else _DEFAULT_UPLOAD_CACHE_DIR.resolve()


def _cached_pdf_path(file_hash: str) -> Path:
    if not _HASH_RE.match(file_hash or ""):
        raise HTTPException(400, "Invalid file_hash")
    p = _upload_cache_dir() / f"{file_hash}.bin"
    if not p.is_file():
        raise HTTPException(
            404,
            f"No cached upload for {file_hash}. "
            "Re-upload from the Extractor tab.",
        )
    with p.open("rb") as fh:
        head = fh.read(4)
    if head != b"%PDF":
        raise HTTPException(
            400,
            "Cached upload is not a PDF; layout validation needs a PDF.",
        )
    return p


def _read_b64(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    return base64.b64encode(path.read_bytes()).decode("ascii")


@app.get("/api/health")
def health() -> dict:
    gt = _ground_truth_dir()
    cache = _upload_cache_dir()
    return {
        "status": "ok",
        "ground_truth_dir": str(gt),
        "ground_truth_exists": gt.is_dir(),
        "regions_dir": str(_regions_dir()),
        "upload_cache_dir": str(cache),
        "upload_cache_exists": cache.is_dir(),
    }


@app.get("/api/products")
def list_products() -> List[Dict[str, Any]]:
    """List products + available template versions for the dropdown.

    Each item also flags whether a hand-authored region schema exists, so the
    frontend can warn the user before they pick a product that requires
    ``--allow-auto-schema``.
    """
    gt = _ground_truth_dir()
    try:
        cat = GroundTruthTemplateCatalog(gt)
    except FileNotFoundError as exc:
        raise HTTPException(500, f"ground-truth catalog: {exc}") from exc

    regions_dir = _regions_dir()
    out: List[Dict[str, Any]] = []
    for product in cat.list_products():
        try:
            versions = cat.list_versions(product)
        except FileNotFoundError:
            versions = []
        schema_path = regions_dir / f"{product}.json"
        out.append(
            {
                "product_code": product,
                "versions": versions,
                "has_manual_schema": schema_path.is_file(),
            }
        )
    return out


def _summarise_pipeline(
    *,
    out_dir: Path,
    report_html_path: Path,
    report_json_path: Path,
    overview_png_path: Path,
    template_overview_png_path: Path,
    product_code: str,
    template_version: str,
    schema_source: str,
    layout_only: bool,
    layout_tolerance_mm: float,
    page_index: int,
) -> Dict[str, Any]:
    """Read the pipeline's on-disk artefacts back into a frontend payload."""
    try:
        report = json.loads(report_json_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise HTTPException(500, f"failed to read report JSON: {exc}") from exc

    overall = report.get("overall_result") or {
        "verdict": report.get("verdict"),
        "final_score": report.get("final_score"),
        "global_threshold": report.get("global_threshold"),
    }

    region_results: List[Dict[str, Any]] = []
    for r in report.get("regions") or []:
        det = r.get("details") or {}
        region_results.append(
            {
                "name": r.get("name"),
                "type": r.get("type"),
                "method": r.get("method"),
                "status": r.get("status"),
                "mandatory": r.get("mandatory"),
                "weight": r.get("weight"),
                "score": r.get("score"),
                "threshold": r.get("threshold"),
                "notes": r.get("notes") or [],
                "shift_x_mm": r.get("shift_x_mm") or det.get("shift_x_mm"),
                "shift_y_mm": r.get("shift_y_mm") or det.get("shift_y_mm"),
                "width_delta_mm": (
                    r.get("width_delta_mm") or det.get("width_delta_mm")
                ),
                "height_delta_mm": (
                    r.get("height_delta_mm") or det.get("height_delta_mm")
                ),
                "severity": det.get("severity"),
                "failure_reasons": det.get("failure_reasons") or [],
            }
        )

    try:
        report_html = report_html_path.read_text(encoding="utf-8")
    except OSError:
        report_html = ""

    return {
        "verdict": overall.get("verdict"),
        "final_score": overall.get("final_score"),
        "global_threshold": overall.get("global_threshold"),
        "schema_source": schema_source,
        "product_code": product_code,
        "template_version": template_version,
        "page_index": page_index,
        "validation_mode": "layout" if layout_only else "legacy",
        "layout_tolerance_mm": layout_tolerance_mm if layout_only else None,
        "metrics": report.get("metrics") or {},
        "region_results": region_results,
        "failed_regions": report.get("failed_regions") or [],
        "warning_regions": report.get("warning_regions") or [],
        "skipped_optional_regions": report.get("skipped_optional_regions") or [],
        "candidate_overview_png_base64": _read_b64(overview_png_path),
        "template_overview_png_base64": _read_b64(template_overview_png_path),
        "report_html": report_html,
    }


@app.post("/api/validate")
async def validate(
    file: Optional[UploadFile] = File(None),
    file_hash: Optional[str] = Form(None),
    product_code: str = Form(...),
    version: Optional[str] = Form(None),
    page: int = Form(0),
    layout_only: bool = Form(True),
    layout_tolerance_mm: float = Form(1.5),
    allow_auto_schema: bool = Form(False),
) -> Dict[str, Any]:
    """Run the region pipeline on either an uploaded ``file`` or a previously
    cached upload identified by ``file_hash``. Exactly one of the two must be
    given; ``file_hash`` lets the browser skip re-uploading bytes after the
    extractor has already cached the file."""
    if (file is None or file.filename is None) and not file_hash:
        raise HTTPException(
            400, "Provide either 'file' (multipart) or 'file_hash' (form field)"
        )

    work_dir = Path(tempfile.mkdtemp(prefix="visual_validation_"))
    candidate_path = work_dir / "candidate.pdf"
    out_dir = work_dir / "report"
    out_dir.mkdir(parents=True, exist_ok=True)

    if file_hash:
        cached = _cached_pdf_path(file_hash)
        candidate_path.write_bytes(cached.read_bytes())
    else:
        assert file is not None  # narrowed by the check above
        data = await file.read()
        if not data:
            shutil.rmtree(work_dir, ignore_errors=True)
            raise HTTPException(400, "Empty file")
        suffix = Path(file.filename or "").suffix.lower() or ".pdf"
        if suffix != ".pdf":
            shutil.rmtree(work_dir, ignore_errors=True)
            raise HTTPException(
                400, "Validation pipeline expects a PDF candidate file"
            )
        candidate_path.write_bytes(data)

    try:
        try:
            result = run_region_report_pipeline(
                candidate_pdf=candidate_path,
                product_code=product_code,
                template_version=version,
                ground_truth_dir=_ground_truth_dir(),
                regions_lookup=_regions_dir(),
                generated_regions_dir=_DEFAULT_GENERATED_REGIONS_DIR,
                out_reports_dir=out_dir,
                allow_auto_schema=allow_auto_schema,
                page_index=page,
                layout_only=layout_only,
                layout_tolerance_mm=layout_tolerance_mm,
            )
        except RegionPipelineError as exc:
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:  # pragma: no cover — defensive
            logger.exception("Validation pipeline failed")
            raise HTTPException(500, f"validation failed: {exc}") from exc

        return _summarise_pipeline(
            out_dir=out_dir,
            report_html_path=result.report_html,
            report_json_path=result.report_json,
            overview_png_path=result.overview_png,
            template_overview_png_path=result.template_overview_png,
            product_code=result.product_code,
            template_version=result.template_version,
            schema_source=result.schema_source,
            layout_only=layout_only,
            layout_tolerance_mm=layout_tolerance_mm,
            page_index=page,
        )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
