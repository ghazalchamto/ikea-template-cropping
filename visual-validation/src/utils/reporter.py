"""
src/utils/reporter.py
=====================
JSON and self-contained HTML report generation.

HTML report is a single file with base64-embedded images — open in any browser.
"""

from __future__ import annotations

import base64
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.core.comparator import ComparisonResult

logger = logging.getLogger(__name__)


class _Enc(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):  return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, (np.bool_,)):    return bool(o)
        return super().default(o)


def save_json(
    result: ComparisonResult,
    path:   str | Path,
    extra:  dict | None = None,
) -> None:
    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    data = {"generated_at": generated_at,
            **(extra or {}), **result.summary()}
    Path(path).write_text(json.dumps(data, indent=2, cls=_Enc))
    logger.info("JSON → %s", path)


def save_html(
    result:           ComparisonResult,
    annotated_img:    np.ndarray,
    heatmap_img:      np.ndarray,
    path:             str | Path,
    label_name:       str = "label",
    product_code:     str = "",
    template_version: str = "",
    grid_map_img:     Optional[np.ndarray] = None,
    side_by_side_img: Optional[np.ndarray] = None,
) -> None:

    ann_b64  = _b64(annotated_img)
    heat_b64 = _b64(heatmap_img)
    grid_b64 = _b64(grid_map_img)    if grid_map_img     is not None else None
    sbs_b64  = _b64(side_by_side_img) if side_by_side_img is not None else None

    gen_footer_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    verdict_bg = "#1a7a1a" if result.is_valid else "#b81818"
    verdict    = "VALID ✓"  if result.is_valid else "INVALID ✗"

    sev = {"critical": 0, "moderate": 0, "minor": 0}
    for r in result.mismatch_regions:
        sev[r.severity] = sev.get(r.severity, 0) + 1

    region_rows = "".join(
        f"<tr>"
        f"<td>#{i+1}</td>"
        f"<td class='sev-{r.severity}'>{r.severity.upper()}</td>"
        f"<td class='src-{r.source}'>{r.source}</td>"
        f"<td>{r.x}, {r.y}</td>"
        f"<td>{r.w} &times; {r.h}</td>"
        f"<td>{r.pixel_diff_ratio*100:.1f}%</td>"
        f"</tr>"
        for i, r in enumerate(result.mismatch_regions)
    )

    failed_tiles = [t for t in result.tile_results if not t.passed]
    tile_rows = "".join(
        f"<tr>"
        f"<td>R{t.row} C{t.col}</td>"
        f"<td>({t.x},{t.y})</td>"
        f"<td>{t.w}&times;{t.h}</td>"
        f"<td style='color:#c01'>{t.pixel_sim*100:.1f}%</td>"
        f"</tr>"
        for t in failed_tiles[:20]   # cap at 20 rows
    )

    grid_section = (
        f"<h2>Tile grid map</h2>"
        f"<p style='color:#555;font-size:.85rem'>Green = PASS, Red = FAIL.  "
        f"{sum(1 for t in result.tile_results if t.passed)} / {len(result.tile_results)} tiles passed.</p>"
        f"<figure><img src='data:image/png;base64,{grid_b64}' "
        f"style='max-width:100%;border:1px solid #ccc;border-radius:6px' alt='grid'/></figure>"
    ) if grid_b64 else ""

    sbs_section = (
        f"<figure style='margin-top:16px'>"
        f"<img src='data:image/png;base64,{sbs_b64}' style='width:100%' alt='comparison'/>"
        f"<figcaption>Template | Test label | Diff heatmap</figcaption></figure>"
    ) if sbs_b64 else ""

    named_region_rows = "".join(
        f"<tr><td>{name}</td><td>{score*100:.1f}%</td>"
        f"<td class='sev-{'critical' if score < 0.85 else 'minor'}'>"
        f"{'FAIL' if score < 0.85 else 'PASS'}</td></tr>"
        for name, score in result.named_region_scores.items()
    ) if result.named_region_scores else ""

    named_section = (
        f"<h2 style='margin:20px 0 10px'>Named region scores</h2>"
        f"<table><thead><tr><th>Region</th><th>Similarity</th><th>Status</th></tr></thead>"
        f"<tbody>{named_region_rows}</tbody></table>"
    ) if named_region_rows else ""

    optional_zone_rows = "".join(
        f"<tr>"
        f"<td>{z.name}</td>"
        f"<td>({z.x}, {z.y})</td>"
        f"<td>{z.width} &times; {z.height}</td>"
        f"<td style='color:#777;font-style:italic'>{z.reason or '—'}</td>"
        f"</tr>"
        for z in getattr(result, "ignored_zones", [])
    )
    optional_zones_section = (
        f"<h2 style='margin:20px 0 10px'>Optional zones ignored ({len(result.ignored_zones)})</h2>"
        f"<p style='color:#555;font-size:.85rem;margin-bottom:8px'>"
        f"Differences inside these zones were excluded from all signals and scores. "
        f"They are shown as <span style='color:#1ab8b8;font-weight:600'>teal dashed outlines</span> "
        f"in the annotated image.</p>"
        f"<table><thead><tr><th>Zone name</th><th>Origin (x,y)</th><th>Size</th><th>Reason</th></tr></thead>"
        f"<tbody>{optional_zone_rows}</tbody></table>"
    ) if optional_zone_rows else ""

    rs_list = getattr(result, "region_scores", None) or []
    comp_mode = getattr(result, "comparison_mode", "full")
    region_split_section = ""
    if comp_mode == "region_split" and rs_list:
        rs_rows = "".join(
            f"<tr>"
            f"<td><strong>{rs.name}</strong></td>"
            f"<td>{rs.role}</td>"
            f"<td>({rs.x}, {rs.y})</td>"
            f"<td>{rs.width}&times;{rs.height}</td>"
            f"<td>{rs.optional_fraction*100:.1f}%</td>"
            f"<td>{'—' if rs.ignored_due_to_optional else f'{rs.final_score*100:.1f}%' if rs.final_score is not None else '—'}</td>"
            f"<td class='sev-{'critical' if not rs.passed and not rs.ignored_due_to_optional else 'minor'}'>"
            f"{'ignored (optional)' if rs.ignored_due_to_optional else ('FAIL' if not rs.passed else 'PASS')}</td>"
            f"<td>{'yes' if rs.counts_for_verdict else '—'}</td>"
            f"</tr>"
            for rs in rs_list
        )
        fail_names = getattr(result, "failed_mandatory_regions", []) or []
        region_split_section = (
            f"<h2 style='margin:20px 0 10px'>Region split scores</h2>"
            f"<p style='color:#555;font-size:.85rem;margin-bottom:8px'>"
            f"Headline verdict uses <strong>mandatory quadrants</strong> only "
            f"(rows marked “counts for verdict”). "
            f"Failed mandatory quadrants: <strong>{', '.join(fail_names) or 'none'}</strong>."
            f"</p>"
            f"<h3 style='font-size:1rem;margin:12px 0 6px'>Full-image reference (not used for verdict)</h3>"
            f"<p style='color:#666;font-size:.88rem'>SSIM {result.full_image_ssim:.3f} &middot; "
            f"final {result.full_image_final_score:.3f} &middot; valid={result.full_image_is_valid}</p>"
            f"<table><thead><tr>"
            f"<th>Region</th><th>Role</th><th>Origin</th><th>Size</th>"
            f"<th>Optional %</th><th>Score</th><th>Status</th><th>Quadrant verdict</th>"
            f"</tr></thead><tbody>{rs_rows}</tbody></table>"
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Visual Validation — {label_name}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:system-ui,sans-serif;background:#f4f4f4;color:#222;padding:24px;font-size:14px}}
h1{{font-size:1.4rem;margin-bottom:6px}}
h2{{font-size:1.1rem;margin:18px 0 10px;border-bottom:1px solid #ddd;padding-bottom:4px}}
.meta{{color:#777;font-size:.85rem;margin-bottom:18px}}
.verdict{{display:inline-block;padding:10px 28px;border-radius:6px;background:{verdict_bg};
  color:#fff;font-size:1.5rem;font-weight:700;margin-bottom:18px}}
.scores{{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:18px}}
.card{{background:#fff;border:1px solid #ddd;border-radius:8px;padding:9px 14px;min-width:130px}}
.card .lbl{{font-size:.72rem;color:#888;text-transform:uppercase;letter-spacing:.04em}}
.card .val{{font-size:1.4rem;font-weight:700;margin-top:2px}}
.images{{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:18px}}
.images figure{{flex:1;min-width:260px}}
.images img{{width:100%;border:1px solid #ccc;border-radius:6px}}
.images figcaption{{font-size:.78rem;color:#666;margin-top:4px;text-align:center}}
table{{border-collapse:collapse;width:100%;background:#fff;border-radius:8px;
  overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,.07);margin-bottom:16px}}
th,td{{padding:8px 12px;text-align:left;border-bottom:1px solid #eee;font-size:.88rem}}
th{{background:#f0f0f0;font-weight:600}}
.sev-critical{{color:#c0160a;font-weight:600}}
.sev-moderate{{color:#c06000;font-weight:600}}
.sev-minor{{color:#888}}
.src-edge{{font-style:italic;color:#555}}
.src-pixel{{}}
footer{{color:#aaa;font-size:.72rem;margin-top:24px}}
</style>
</head>
<body>
<h1>IKEA Label Visual Validation Report</h1>
<p class="meta">
  Label: <strong>{label_name}</strong>
  {"&nbsp;&middot;&nbsp;Product: <strong>" + product_code + "</strong>" if product_code else ""}
  {"&nbsp;&middot;&nbsp;Template version: <strong>" + template_version + "</strong>" if template_version else ""}
  &nbsp;&middot;&nbsp;DPI: <strong>see config</strong>
  &nbsp;&middot;&nbsp;Alignment: <strong>{"OK" if result.alignment_ok else "FALLBACK"}</strong>
  &nbsp;&middot;&nbsp;Mode: <strong>{getattr(result, "comparison_mode", "full")}</strong>
</p>

<div class="verdict">{verdict}</div>

<div class="scores">
  <div class="card"><div class="lbl">Final score</div>
    <div class="val">{result.final_score:.3f}</div></div>
  <div class="card"><div class="lbl">SSIM</div>
    <div class="val">{result.ssim_score:.3f}</div></div>
  <div class="card"><div class="lbl">Pixel sim</div>
    <div class="val">{result.pixel_similarity:.3f}</div></div>
  <div class="card"><div class="lbl">Edge sim</div>
    <div class="val">{result.edge_similarity:.3f}</div></div>
  <div class="card"><div class="lbl">Tile pass rate</div>
    <div class="val">{result.tile_pass_rate:.3f}</div></div>
  <div class="card"><div class="lbl">Regions</div>
    <div class="val">{len(result.mismatch_regions)}</div></div>
  <div class="card"><div class="lbl">Critical / Mod / Minor</div>
    <div class="val" style="font-size:.95rem">
      <span style="color:#c01">{sev['critical']}</span> /
      <span style="color:#c60">{sev['moderate']}</span> /
      <span style="color:#888">{sev['minor']}</span>
    </div></div>
  <div class="card"><div class="lbl">Failed tiles</div>
    <div class="val">{sum(1 for t in result.tile_results if not t.passed)} / {len(result.tile_results)}</div></div>
</div>

<h2>Visual outputs</h2>
<div class="images">
  <figure>
    <img src="data:image/png;base64,{ann_b64}" alt="Annotated"/>
    <figcaption>Annotated — solid boxes=pixel diff, dashed=structural edge diff</figcaption>
  </figure>
  <figure>
    <img src="data:image/png;base64,{heat_b64}" alt="Heatmap"/>
    <figcaption>Difference heatmap — hot=large diff, cool=small diff</figcaption>
  </figure>
</div>
{sbs_section}

{grid_section}

<h2>Mismatch regions ({len(result.mismatch_regions)})</h2>
{"<p style='color:#888;font-style:italic'>No significant mismatches detected.</p>"
  if not result.mismatch_regions else
  "<table><thead><tr><th>#</th><th>Severity</th><th>Source</th>"
  "<th>Origin (x,y)</th><th>Size</th><th>Diff%</th></tr></thead>"
  f"<tbody>{region_rows}</tbody></table>"
  "<p style='font-size:.78rem;color:#888'>Source: pixel=colour diff  "
  "edge=structural/outline diff</p>"}

{"<h2>Failed tiles (top 20)</h2>"
  "<table><thead><tr><th>Tile</th><th>Origin</th><th>Size</th><th>Similarity</th></tr></thead>"
  f"<tbody>{tile_rows}</tbody></table>"
  if failed_tiles else ""}

{named_section}

{optional_zones_section}

{region_split_section}

<footer>Generated {gen_footer_ts} UTC</footer>
</body>
</html>"""

    Path(path).write_text(html, encoding="utf-8")
    logger.info("HTML → %s", path)


def _b64(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""
