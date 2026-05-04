"""
Build human- and machine-readable reports from a completed validation run.

Does not perform extraction or scoring — callers pass in ``ValidationSummary``,
``RegionScore`` list, and the candidate ``PDFExtraction`` (for ``pdf_bbox``).
"""

from __future__ import annotations

import html
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    import fitz  # PyMuPDF
except ImportError as exc:  # pragma: no cover
    raise ImportError("PyMuPDF is required for region_report") from exc

from src.core.pdf_region_extractor import PDFExtraction
from src.core.region_aggregator import (
    STATUS_FAIL,
    STATUS_OPTIONAL,
    STATUS_PASS,
    ValidationSummary,
)
from src.core.region_comparator import RegionScore
from src.core.region_loader import RegionConfig


def safe_label_tag(candidate_pdf: Path) -> str:
    """Filesystem-safe fragment from candidate filename (no path)."""
    s = re.sub(r"[^a-zA-Z0-9]+", "_", candidate_pdf.stem).strip("_")
    return s[:120] if s else "candidate"


def infer_template_version(template_pdf: Path) -> str:
    """Best-effort version label, e.g. *v7* from .../1R10-PRADM/v7/label.pdf."""
    for part in template_pdf.parts:
        if len(part) >= 2 and part[0] == "v" and part[1:].isdigit():
            return part
    return template_pdf.parent.name


def _status_color_bgr(status: str) -> Tuple[int, int, int]:
    if status == STATUS_PASS:
        return (0, 180, 0)
    if status == STATUS_FAIL:
        return (0, 0, 220)
    if status == STATUS_OPTIONAL:
        return (255, 180, 0)  # cyan/teal accent in BGR
    return (128, 128, 128)


def _draw_pdf_rect_bgr(
    img: np.ndarray,
    scale: float,
    bbox_pdf: Sequence[float],
    color_bgr: Tuple[int, int, int],
    thickness: int,
) -> None:
    """Draw axis-aligned PDF rect (x0,y0,x1,y1) onto a rasterised page image."""
    x0, y0, x1, y1 = (float(bbox_pdf[0]), float(bbox_pdf[1]), float(bbox_pdf[2]), float(bbox_pdf[3]))
    px0 = int(math.floor(x0 * scale))
    py0 = int(math.floor(y0 * scale))
    px1 = int(math.ceil(x1 * scale))
    py1 = int(math.ceil(y1 * scale))
    h, w = img.shape[:2]
    px0 = max(0, min(w - 1, px0))
    px1 = max(0, min(w - 1, px1))
    py0 = max(0, min(h - 1, py0))
    py1 = max(0, min(h - 1, py1))
    if px1 <= px0 or py1 <= py0:
        return
    cv2.rectangle(img, (px0, py0), (px1, py1), color_bgr, thickness, lineType=cv2.LINE_AA)


def render_candidate_overview(
    candidate_pdf: Path,
    candidate_extr: PDFExtraction,
    summary: ValidationSummary,
    out_path: Path,
    page_index: int = 0,
    overview_dpi: float = 150.0,
    region_scores: Optional[List[RegionScore]] = None,
) -> None:
    """
    Rasterise the full candidate *page* at *overview_dpi* and draw region
    outlines from ``ExtractedRegion.pdf_bbox`` with colours from verdict status.
    """
    scale = overview_dpi / 72.0
    doc = fitz.open(str(candidate_pdf))
    try:
        if page_index < 0 or page_index >= doc.page_count:
            raise ValueError(
                f"page_index={page_index} out of range (count={doc.page_count})"
            )
        page = doc[page_index]
        mat = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=mat, alpha=False)
    finally:
        doc.close()

    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n
    )
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    status_by_name = {r.name: r.status for r in summary.region_results}
    h, w = img.shape[:2]
    thickness = max(2, int(round(0.002 * min(w, h))))

    for er in candidate_extr.extracted:
        st = status_by_name.get(er.name, STATUS_FAIL)
        color = _status_color_bgr(st)
        x0, y0, x1, y1 = er.pdf_bbox
        px0 = int(math.floor(x0 * scale))
        py0 = int(math.floor(y0 * scale))
        px1 = int(math.ceil(x1 * scale))
        py1 = int(math.ceil(y1 * scale))
        cv2.rectangle(img, (px0, py0), (px1, py1), color, thickness, lineType=cv2.LINE_AA)

        label = er.name
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = 0.45 if min(w, h) < 800 else 0.55
        (tw, th), baseline = cv2.getTextSize(label, font, fs, 1)
        tx = min(max(0, px0 + 2), max(0, w - tw - 4))
        ty = max(th + 4, py0 - 4)
        cv2.rectangle(
            img, (tx - 2, ty - th - 2), (tx + tw + 2, ty + baseline + 2),
            (40, 40, 40), -1, lineType=cv2.LINE_AA,
        )
        cv2.putText(
            img, label, (tx, ty), font, fs, (255, 255, 255), 1, cv2.LINE_AA,
        )

    if region_scores:
        score_by = {s.region_name: s for s in region_scores}
        for er in candidate_extr.extracted:
            st = status_by_name.get(er.name, STATUS_FAIL)
            if st != STATUS_FAIL:
                continue
            sc = score_by.get(er.name)
            if sc is None or not sc.details:
                continue
            det = sc.details
            exp = det.get("expected_object_bbox_pdf")
            cand = det.get("candidate_object_bbox_pdf")
            if isinstance(exp, (list, tuple)) and len(exp) == 4:
                _draw_pdf_rect_bgr(img, scale, exp, (0, 140, 255), max(2, thickness))
            if isinstance(cand, (list, tuple)) and len(cand) == 4:
                _draw_pdf_rect_bgr(img, scale, cand, (255, 0, 255), max(2, thickness))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


def render_template_overview(
    template_pdf: Path,
    template_extr: PDFExtraction,
    summary: ValidationSummary,
    out_path: Path,
    page_index: int = 0,
    overview_dpi: float = 150.0,
    region_scores: Optional[List[RegionScore]] = None,
) -> None:
    """Same raster + outlines as the candidate overview, for the approved template PDF."""
    render_candidate_overview(
        template_pdf,
        template_extr,
        summary,
        out_path,
        page_index=page_index,
        overview_dpi=overview_dpi,
        region_scores=region_scores,
    )


def build_region_rows(
    summary: ValidationSummary,
    region_scores: List[RegionScore],
    candidate_extr: PDFExtraction,
    template_extr: Optional[PDFExtraction] = None,
) -> List[Dict[str, Any]]:
    """Merge verdict rows with score notes/details and candidate pdf_bbox."""
    score_by = {s.region_name: s for s in region_scores}
    extr_by = candidate_extr.by_name()
    tex_by = template_extr.by_name() if template_extr is not None else {}

    layout_keys = (
        "shift_x_mm", "shift_y_mm", "width_delta_mm", "height_delta_mm",
        "expected_object_bbox_pdf", "candidate_object_bbox_pdf",
        "severity", "failure_reasons",
        "vector_union_template_pdf", "vector_union_candidate_pdf",
        "edge_structure_score",
    )

    rows: List[Dict[str, Any]] = []
    for v in summary.region_results:
        s = score_by.get(v.name)
        er = extr_by.get(v.name)
        ter = tex_by.get(v.name)
        notes: List[str] = list(s.notes) if s else []
        details: Dict[str, Any] = dict(s.details) if s and s.details else {}
        row: Dict[str, Any] = {
            "name":       v.name,
            "type":       v.region_type,
            "method":     v.method,
            "score":      None if v.status == STATUS_OPTIONAL else round(v.combined_score, 4),
            "threshold":  round(v.threshold, 4),
            "status":     v.status,
            "mandatory":  v.mandatory,
            "weight":     round(v.weight, 4),
            "passed":     v.passed,
            "notes":      notes,
            "details":    details,
            "pdf_bbox_candidate": list(er.pdf_bbox) if er else None,
            "pdf_bbox_region_template": list(ter.pdf_bbox) if ter else None,
        }
        for k in layout_keys:
            if k in details:
                row[k] = details[k]
        rows.append(row)
    return rows


def build_report_payload(
    *,
    product_code: str,
    candidate_pdf: Path,
    template_pdf: Path,
    region_config: RegionConfig,
    summary: ValidationSummary,
    region_scores: List[RegionScore],
    candidate_extr: PDFExtraction,
    overview_filename: str,
    pipeline: Optional[Dict[str, Any]] = None,
    template_overview_filename: Optional[str] = None,
    template_extr: Optional[PDFExtraction] = None,
) -> Dict[str, Any]:
    regions = build_region_rows(
        summary, region_scores, candidate_extr, template_extr=template_extr,
    )
    row_by = {r["name"]: r for r in regions}

    def _num_from_row(r: Dict[str, Any], key: str) -> Any:
        if r.get(key) is not None:
            return r.get(key)
        det = r.get("details") or {}
        return det.get(key)

    failed_regions: List[Dict[str, Any]] = []
    for r in regions:
        if r["status"] != STATUS_FAIL:
            continue
        det = r.get("details") or {}
        failed_regions.append(
            {
                "name": r["name"],
                "type": r["type"],
                "method": r["method"],
                "combined_score": r["score"],
                "threshold": r["threshold"],
                "failure_reasons": det.get("failure_reasons") or list(r.get("notes") or []),
                "notes": list(r.get("notes") or []),
                "pdf_bbox_candidate": r.get("pdf_bbox_candidate"),
                "pdf_bbox_region_template": r.get("pdf_bbox_region_template"),
                "shift_x_mm": _num_from_row(r, "shift_x_mm"),
                "shift_y_mm": _num_from_row(r, "shift_y_mm"),
                "width_delta_mm": _num_from_row(r, "width_delta_mm"),
                "height_delta_mm": _num_from_row(r, "height_delta_mm"),
                "expected_object_bbox_pdf": det.get("expected_object_bbox_pdf"),
                "candidate_object_bbox_pdf": det.get("candidate_object_bbox_pdf"),
                "severity": det.get("severity"),
            }
        )

    warning_regions: List[Dict[str, Any]] = []
    for name in summary.warning_regions:
        r = row_by.get(name)
        if r is None:
            continue
        det = r.get("details") or {}
        warning_regions.append(
            {
                "name": name,
                "type": r["type"],
                "method": r["method"],
                "combined_score": r["score"],
                "threshold": r["threshold"],
                "severity": det.get("severity"),
                "notes": list(r.get("notes") or []),
                "shift_x_mm": _num_from_row(r, "shift_x_mm"),
                "shift_y_mm": _num_from_row(r, "shift_y_mm"),
                "width_delta_mm": _num_from_row(r, "width_delta_mm"),
                "height_delta_mm": _num_from_row(r, "height_delta_mm"),
            }
        )

    skipped_optional = list(summary.skipped_optional_regions)

    overall_result = {
        "verdict": summary.verdict,
        "final_score": round(float(summary.final_score), 4),
        "global_threshold": round(float(summary.global_threshold), 4),
    }
    metrics = {
        "total_regions": summary.total_regions,
        "mandatory_passed": summary.mandatory_passed,
        "mandatory_failed": summary.mandatory_failed,
        "optional_count": summary.optional_count,
        "warning_region_count": len(summary.warning_regions),
        "skipped_optional_count": len(skipped_optional),
    }

    out: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "product_code":     product_code,
        "candidate_pdf":    str(candidate_pdf.resolve()),
        "template_pdf":     str(template_pdf.resolve()),
        "template_version": infer_template_version(template_pdf),
        "region_config":    str(region_config.source_path.resolve()),
        "overall_result":   overall_result,
        "metrics":          metrics,
        "failed_regions":   failed_regions,
        "warning_regions":  warning_regions,
        "skipped_optional_regions": skipped_optional,
        "verdict":          summary.verdict,
        "final_score":      overall_result["final_score"],
        "global_threshold": overall_result["global_threshold"],
        "total_regions":    summary.total_regions,
        "mandatory_passed": summary.mandatory_passed,
        "mandatory_failed": summary.mandatory_failed,
        "optional_ignored": summary.optional_count,
        "failed_region_names":  list(summary.failed_regions),
        "ignored_region_names": list(summary.ignored_regions),
        "warning_region_names": list(summary.warning_regions),
        "overview_image": overview_filename,
        "regions": regions,
    }
    if template_overview_filename:
        out["template_overview_image"] = template_overview_filename
    if pipeline:
        out["pipeline"] = dict(pipeline)
        if pipeline.get("template_version"):
            out["template_version"] = str(pipeline["template_version"])
        if pipeline.get("template_json"):
            out["template_json"] = str(pipeline["template_json"])
    return out


def write_json_report(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _status_css_class(status: str) -> str:
    if status == STATUS_PASS:
        return "pill pill-pass"
    if status == STATUS_FAIL:
        return "pill pill-fail"
    if status == STATUS_OPTIONAL:
        return "pill pill-opt"
    return "pill"


def write_html_report(path: Path, payload: Dict[str, Any]) -> None:
    """Standalone HTML: calm palette, verdict hero, compact table, collapsible details."""
    regions = payload["regions"]
    esc = html.escape

    def _mm_cell(r: Dict[str, Any], key: str) -> str:
        v = r.get(key)
        if v is None:
            v = (r.get("details") or {}).get(key)
        if v is None:
            return "—"
        try:
            return f"{float(v):+.2f}"
        except (TypeError, ValueError):
            return esc(str(v))

    rows_html: List[str] = []
    for r in regions:
        score_s = "—" if r["score"] is None else f"{r['score']:.4f}"
        notes_s = esc("; ".join(r["notes"])) if r["notes"] else "—"
        det = r.get("details") or {}
        det_json = json.dumps(det, sort_keys=True, indent=2) if det else ""
        pill = _status_css_class(str(r["status"]))
        if det_json:
            details_block = (
                f"<details class='row-details'><summary>JSON</summary>"
                f"<pre>{esc(det_json)}</pre></details>"
            )
        else:
            details_block = "<span class='muted'>—</span>"
        dx = _mm_cell(r, "shift_x_mm")
        dy = _mm_cell(r, "shift_y_mm")
        dw = _mm_cell(r, "width_delta_mm")
        dh = _mm_cell(r, "height_delta_mm")
        rows_html.append(
            "<tr>"
            f"<td class='col-name'><span class='mono'>{esc(r['name'])}</span></td>"
            f"<td><span class='{pill}'>{esc(r['status'])}</span></td>"
            f"<td class='num'>{score_s}</td>"
            f"<td class='num muted'>{r['threshold']:.4f}</td>"
            f"<td class='small'>{esc(r['type'])}</td>"
            f"<td class='small'>{esc(r['method'])}</td>"
            f"<td class='center'>{'✓' if r['mandatory'] else '—'}</td>"
            f"<td class='num small'>{r['weight']:.2f}</td>"
            f"<td class='num small mono'>{dx}</td>"
            f"<td class='num small mono'>{dy}</td>"
            f"<td class='num small mono'>{dw}</td>"
            f"<td class='num small mono'>{dh}</td>"
            f"<td class='notes'>{notes_s}</td>"
            f"<td>{details_block}</td>"
            "</tr>"
        )

    ignored = payload.get("ignored_region_names") or []
    failed = payload.get("failed_region_names") or []
    skipped_opt = payload.get("skipped_optional_regions") or []
    warn_items = payload.get("warning_regions") or []
    verdict = payload["verdict"]
    is_valid = verdict == "VALID"
    overview = esc(payload.get("overview_image", ""))

    failed_detail = payload.get("failed_regions") or []
    layout_diag_html = ""
    if failed_detail:
        cards: List[str] = []
        for fr in failed_detail:
            fr_reasons = fr.get("failure_reasons") or []
            fr_s = esc("; ".join(str(x) for x in fr_reasons)) if fr_reasons else "—"
            nm = esc(str(fr.get("name", "")))
            diag = {
                "shift_x_mm": fr.get("shift_x_mm"),
                "shift_y_mm": fr.get("shift_y_mm"),
                "width_delta_mm": fr.get("width_delta_mm"),
                "height_delta_mm": fr.get("height_delta_mm"),
                "expected_object_bbox_pdf": fr.get("expected_object_bbox_pdf"),
                "candidate_object_bbox_pdf": fr.get("candidate_object_bbox_pdf"),
                "pdf_bbox_region_template": fr.get("pdf_bbox_region_template"),
                "pdf_bbox_candidate": fr.get("pdf_bbox_candidate"),
            }
            diag_json = esc(json.dumps(diag, indent=2, sort_keys=True))
            cards.append(
                "<article class='layout-fail-card'>"
                f"<h3 class='mono'>{nm}</h3>"
                f"<p class='small'>{esc(str(fr.get('type', '')))} · {esc(str(fr.get('method', '')))}</p>"
                f"<p><strong>Reasons</strong> — {fr_s}</p>"
                f"<pre class='layout-fail-pre'>{diag_json}</pre>"
                "</article>",
            )
        layout_diag_html = (
            "<section class='card'>"
            "<h2>Failed regions (layout / structure)</h2>"
            "<p class='hint'>Overview below: orange = expected object bbox, magenta = candidate (PDF points).</p>"
            f"<div class='layout-fail-grid'>{''.join(cards)}</div>"
            "</section>"
        )
    else:
        layout_fail_rows = [
            r for r in regions
            if r.get("severity") == "critical" and r.get("failure_reasons")
        ]
        if layout_fail_rows:
            cards = []
            for r in layout_fail_rows:
                fr = esc("; ".join(str(x) for x in (r.get("failure_reasons") or [])))
                diag = {
                    "shift_x_mm":           r.get("shift_x_mm"),
                    "shift_y_mm":           r.get("shift_y_mm"),
                    "width_delta_mm":       r.get("width_delta_mm"),
                    "height_delta_mm":      r.get("height_delta_mm"),
                    "expected_object_bbox_pdf": r.get("expected_object_bbox_pdf"),
                    "candidate_object_bbox_pdf": r.get("candidate_object_bbox_pdf"),
                    "pdf_bbox_region_template": r.get("pdf_bbox_region_template"),
                    "pdf_bbox_candidate": r.get("pdf_bbox_candidate"),
                }
                diag_json = esc(json.dumps(diag, indent=2, sort_keys=True))
                cards.append(
                    "<article class='layout-fail-card'>"
                    f"<h3 class='mono'>{esc(r['name'])}</h3>"
                    f"<p class='small'>{esc(str(r['type']))} · {esc(str(r['method']))}</p>"
                    f"<p><strong>Failure reasons</strong> — {fr}</p>"
                    f"<pre class='layout-fail-pre'>{diag_json}</pre>"
                    "</article>",
                )
            layout_diag_html = (
                "<section class='card'>"
                "<h2>Layout failures (object / region)</h2>"
                "<p class='hint'>Overview image below: orange = expected object bbox, magenta = candidate object bbox (PDF points).</p>"
                f"<div class='layout-fail-grid'>{''.join(cards)}</div>"
                "</section>"
            )
    pl = payload.get("pipeline") or {}
    pipeline_html = ""
    if pl:
        pipeline_html = (
            "<details class='meta-details'><summary>Pipeline &amp; paths</summary>"
            "<dl class='kv'>"
            f"<dt>Region schema</dt><dd>{esc(str(pl.get('region_schema_mode', '—')))}</dd>"
            f"<dt>Template JSON</dt><dd><code>{esc(str(pl.get('template_json', '')))}</code></dd>"
        )
        if pl.get("label_png"):
            pipeline_html += (
                f"<dt>Template PNG</dt><dd><code>{esc(str(pl['label_png']))}</code></dd>"
            )
        pipeline_html += "</dl></details>"

    gen_at = esc(str(payload.get("generated_at_utc", "")))

    template_raw = payload.get("template_overview_image") or ""
    template_esc = esc(template_raw)
    if template_raw:
        template_figure_inner = (
            f'<img src="{template_esc}" alt="Template page with region outlines"/>'
        )
    else:
        template_figure_inner = '<p class="muted">No template overview generated.</p>'

    body = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Region validation — {esc(payload['product_code'])}</title>
  <style>
    :root {{
      --bg: #eef2f6;
      --surface: #ffffff;
      --text: #0f172a;
      --muted: #64748b;
      --border: #e2e8f0;
      --accent: #3b82f6;
      --pass-bg: #ecfdf5;
      --pass-fg: #047857;
      --fail-bg: #fef2f2;
      --fail-fg: #b91c1c;
      --opt-bg: #fffbeb;
      --opt-fg: #b45309;
      --shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
      --radius: 12px;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto,
                   "Helvetica Neue", Arial, sans-serif;
      margin: 0;
      min-height: 100vh;
      background: linear-gradient(165deg, #e8eef5 0%, #f1f5f9 45%, #eef2f6 100%);
      color: var(--text);
      line-height: 1.5;
      font-size: 15px;
    }}
    .wrap {{
      max-width: 1080px;
      margin: 0 auto;
      padding: 2rem 1.25rem 3rem;
    }}
    header {{
      margin-bottom: 1.5rem;
    }}
    header h1 {{
      font-size: 1.125rem;
      font-weight: 600;
      letter-spacing: -0.02em;
      margin: 0 0 0.25rem 0;
      color: var(--muted);
    }}
    header .product {{
      font-size: 1.5rem;
      font-weight: 700;
      letter-spacing: -0.03em;
      color: var(--text);
    }}
    header .when {{
      font-size: 0.8rem;
      color: var(--muted);
      margin-top: 0.35rem;
    }}
    .hero {{
      background: var(--surface);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      padding: 1.5rem 1.5rem 1.35rem;
      margin-bottom: 1.5rem;
      border: 1px solid var(--border);
    }}
    .hero-top {{
      display: flex;
      flex-wrap: wrap;
      align-items: flex-start;
      justify-content: space-between;
      gap: 1rem;
    }}
    .verdict-pill {{
      display: inline-flex;
      align-items: center;
      padding: 0.4rem 1rem;
      border-radius: 999px;
      font-weight: 700;
      font-size: 0.9rem;
      letter-spacing: 0.04em;
    }}
    .verdict-pill.ok {{
      background: var(--pass-bg);
      color: var(--pass-fg);
      border: 1px solid #a7f3d0;
    }}
    .verdict-pill.bad {{
      background: var(--fail-bg);
      color: var(--fail-fg);
      border: 1px solid #fecaca;
    }}
    .scores {{
      display: flex;
      gap: 1.5rem;
      flex-wrap: wrap;
    }}
    .score-block {{
      min-width: 7rem;
    }}
    .score-block .label {{
      font-size: 0.72rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      font-weight: 600;
    }}
    .score-block .value {{
      font-size: 1.65rem;
      font-weight: 700;
      letter-spacing: -0.02em;
      color: var(--text);
    }}
    .score-block .sub {{
      font-size: 0.8rem;
      color: var(--muted);
    }}
    .stats {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.75rem 1.25rem;
      margin-top: 1.25rem;
      padding-top: 1.1rem;
      border-top: 1px solid var(--border);
      font-size: 0.875rem;
      color: var(--muted);
    }}
    .stats strong {{ color: var(--text); font-weight: 600; }}
    .card {{
      background: var(--surface);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      border: 1px solid var(--border);
      padding: 1.25rem 1.35rem;
      margin-bottom: 1.25rem;
    }}
    .card h2 {{
      font-size: 0.95rem;
      font-weight: 600;
      margin: 0 0 0.75rem 0;
      color: var(--text);
    }}
    .path-line {{
      font-size: 0.8rem;
      color: var(--muted);
      margin: 0.35rem 0 0;
      word-break: break-all;
    }}
    .path-line code {{
      background: #f8fafc;
      padding: 0.12rem 0.35rem;
      border-radius: 4px;
      font-size: 0.78rem;
      color: #334155;
    }}
    .meta-details summary {{
      cursor: pointer;
      font-size: 0.85rem;
      color: var(--accent);
      font-weight: 600;
      margin-top: 0.5rem;
    }}
    .kv {{
      display: grid;
      grid-template-columns: 8rem 1fr;
      gap: 0.35rem 0.75rem;
      font-size: 0.8rem;
      margin: 0.75rem 0 0;
    }}
    .kv dt {{ color: var(--muted); margin: 0; }}
    .kv dd {{ margin: 0; word-break: break-all; }}
    table {{
      width: 100%;
      border-collapse: separate;
      border-spacing: 0;
      font-size: 0.82rem;
    }}
    thead th {{
      text-align: left;
      font-size: 0.68rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--muted);
      font-weight: 600;
      padding: 0.65rem 0.75rem;
      background: #f8fafc;
      border-bottom: 1px solid var(--border);
    }}
    thead th:first-child {{ border-radius: 8px 0 0 0; }}
    thead th:last-child {{ border-radius: 0 8px 0 0; }}
    tbody td {{
      padding: 0.65rem 0.75rem;
      border-bottom: 1px solid var(--border);
      vertical-align: top;
    }}
    tbody tr:last-child td {{ border-bottom: none; }}
    tbody tr:hover td {{ background: #fafbfc; }}
    .col-name {{ font-weight: 600; }}
    .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.8rem; }}
    .num {{ font-variant-numeric: tabular-nums; }}
    .center {{ text-align: center; }}
    .small {{ font-size: 0.78rem; color: var(--muted); }}
    .muted {{ color: var(--muted); }}
    .pill {{
      display: inline-block;
      padding: 0.2rem 0.55rem;
      border-radius: 6px;
      font-size: 0.72rem;
      font-weight: 700;
      letter-spacing: 0.03em;
    }}
    .pill-pass {{ background: var(--pass-bg); color: var(--pass-fg); }}
    .pill-fail {{ background: var(--fail-bg); color: var(--fail-fg); }}
    .pill-opt {{ background: var(--opt-bg); color: var(--opt-fg); }}
    td.notes {{
      max-width: 14rem;
      word-break: break-word;
      font-size: 0.78rem;
      color: #475569;
    }}
    .row-details summary {{
      cursor: pointer;
      font-size: 0.72rem;
      color: var(--accent);
    }}
    .row-details pre {{
      margin: 0.4rem 0 0;
      padding: 0.5rem;
      background: #f8fafc;
      border-radius: 6px;
      font-size: 0.68rem;
      overflow-x: auto;
      max-height: 10rem;
    }}
    .chips {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.35rem;
    }}
    .chip {{
      background: #f1f5f9;
      color: #475569;
      padding: 0.25rem 0.5rem;
      border-radius: 6px;
      font-size: 0.78rem;
    }}
    .chip.empty {{ color: var(--muted); font-style: italic; }}
    .overview img {{
      max-width: 100%;
      height: auto;
      border-radius: 8px;
      border: 1px solid var(--border);
      box-shadow: var(--shadow);
    }}
    .overview .hint {{
      font-size: 0.8rem;
      color: var(--muted);
      margin: 0.5rem 0 0.75rem;
    }}
    .compare-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1.25rem;
      align-items: start;
    }}
    .compare-grid figure {{
      margin: 0;
    }}
    .compare-grid figcaption {{
      font-size: 0.8rem;
      font-weight: 600;
      color: var(--text);
      margin-bottom: 0.5rem;
    }}
    .compare-grid .hint {{
      grid-column: 1 / -1;
      margin: 0 0 0.5rem 0;
    }}
    .layout-fail-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(18rem, 1fr));
      gap: 1rem;
    }}
    .layout-fail-card {{
      border: 1px solid #fecaca;
      border-radius: 8px;
      padding: 0.85rem 1rem;
      background: #fffafa;
    }}
    .layout-fail-card h3 {{ margin: 0 0 0.35rem 0; font-size: 0.95rem; }}
    .layout-fail-pre {{
      margin: 0.5rem 0 0;
      padding: 0.5rem;
      background: #fff;
      border-radius: 6px;
      font-size: 0.68rem;
      overflow-x: auto;
      border: 1px solid var(--border);
    }}
    @media (max-width: 720px) {{
      .wrap {{ padding: 1rem 0.75rem 2rem; }}
      table {{ font-size: 0.75rem; }}
      thead th, tbody td {{ padding: 0.45rem 0.4rem; }}
      .compare-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <h1>Region validation</h1>
      <div class="product">{esc(payload['product_code'])}</div>
      <div class="when">{gen_at}</div>
    </header>

    <section class="hero">
      <div class="hero-top">
        <span class="verdict-pill {'ok' if is_valid else 'bad'}">{esc(verdict)}</span>
        <div class="scores">
          <div class="score-block">
            <div class="label">Final score</div>
            <div class="value">{payload['final_score']:.4f}</div>
            <div class="sub">vs threshold {payload['global_threshold']:.4f}</div>
          </div>
        </div>
      </div>
      <div class="stats">
        <span><strong>{payload['total_regions']}</strong> regions</span>
        <span><strong>{payload['mandatory_passed']}</strong> mandatory pass</span>
        <span><strong>{payload['mandatory_failed']}</strong> mandatory fail</span>
        <span><strong>{len(skipped_opt)}</strong> optional absent on candidate</span>
      </div>
    </section>

    <section class="card">
      <h2>Files</h2>
      <p class="path-line"><strong>Candidate</strong><br/><code>{esc(payload['candidate_pdf'])}</code></p>
      <p class="path-line"><strong>Template</strong> · {esc(str(payload['template_version']))}<br/>
        <code>{esc(payload['template_pdf'])}</code></p>
      <p class="path-line"><strong>Region config</strong><br/><code>{esc(payload['region_config'])}</code></p>
      {pipeline_html}
    </section>

    <section class="card">
      <h2>Per region</h2>
      <div style="overflow-x: auto;">
        <table>
          <thead>
            <tr>
              <th>Region</th>
              <th>Status</th>
              <th>Score</th>
              <th>Thr.</th>
              <th>Type</th>
              <th>Method</th>
              <th>Mand.</th>
              <th>Wt.</th>
              <th class="num">Δx mm</th>
              <th class="num">Δy mm</th>
              <th class="num">Δw mm</th>
              <th class="num">Δh mm</th>
              <th>Notes</th>
              <th>Details</th>
            </tr>
          </thead>
          <tbody>
            {''.join(rows_html)}
          </tbody>
        </table>
      </div>
    </section>

    <section class="card">
      <h2>Optional (ignored for weighted score)</h2>
      <div class="chips">
        {''.join(f"<span class='chip'>{esc(n)}</span>" for n in ignored) if ignored else "<span class='chip empty'>None</span>"}
      </div>
    </section>

    <section class="card">
      <h2>Optional absent on candidate</h2>
      <p class="hint">These regions were not extracted on the candidate PDF; layout score is not reduced.</p>
      <div class="chips">
        {''.join(f"<span class='chip'>{esc(str(n))}</span>" for n in skipped_opt) if skipped_opt else "<span class='chip empty'>None</span>"}
      </div>
    </section>

    <section class="card">
      <h2>Warnings (mandatory passed, combined &lt; 1)</h2>
      <div class="chips">
        {''.join(
            f"<span class='chip'>{esc(str(w.get('name', '')))} "
            f"({float(w.get('combined_score') or 0):.3f})</span>"
            for w in warn_items
        ) if warn_items else "<span class='chip empty'>None</span>"}
      </div>
    </section>

    <section class="card">
      <h2>Failed mandatory</h2>
      <div class="chips">
        {''.join(f"<span class='chip'>{esc(n)}</span>" for n in failed) if failed else "<span class='chip empty'>None</span>"}
      </div>
    </section>

    {layout_diag_html}

    <section class="card overview">
      <h2>Template vs label under test</h2>
      <p class="hint">Same region outlines on both pages (green = pass, red = fail, amber = optional).</p>
      <div class="compare-grid">
        <figure>
          <figcaption>Approved template (reference PDF)</figcaption>
          {template_figure_inner}
        </figure>
        <figure>
          <figcaption>Your label (candidate PDF)</figcaption>
          <img src="{overview}" alt="Candidate page with region outlines"/>
        </figure>
      </div>
    </section>
  </div>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
