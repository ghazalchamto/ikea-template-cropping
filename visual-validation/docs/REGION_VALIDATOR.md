# Region-based PDF label validator (technical reference)

This document describes the **PDF-to-PDF**, **region-schema** validation path used by `scripts/validate_label.py` and `scripts/generate_region_report.py`. For the **raster alignment** path (`scripts/validate.py`, `LabelValidator`), see the root [README](../README.md).

The architecture figure lives in the root README: [PDF-to-PDF region pipeline](../README.md#pdf-to-pdf-region-pipeline-architecture).

---

## What the validator does

1. **Resolves** an approved template for a **product code** (and optional **version** `vN`) under `ground_truth/templates/<PRODUCT>/<vN>/` — expects `label.pdf` and `template.json`.
2. **Loads** a **region schema** (`config/regions/<PRODUCT>.json` by default), defining named regions with `bbox_norm` (or legacy pixel boxes), types, weights, and optional tolerances.
3. **Opens** the template PDF and the **candidate** label PDF, optionally after **page-count / page-index** and **page-size** pre-checks (layout mode).
4. **Extracts** each region as a raster clip (and PDF-space bbox) via PyMuPDF — coordinates are relative to **label rect** (full page or `label_rect_pts` in the schema).
5. **Runs per-region validators** (legacy: text layout / barcode presence / visual SSIM; **layout-only**: geometry, ink extent, vectors/edges where applicable — **not** full-page SSIM as the primary gate).
6. **Aggregates** weighted scores over mandatory regions, applies global threshold, and writes **JSON + HTML + overview PNGs**.

---

## What it does not do

- **No ML inference** for layout detection, OCR content verification, or barcode decoding as the primary decision.
- **No automatic product detection** — you must pass the correct `--product` (template family).
- **No legal/compliance certification** — engineering QA aid only.
- **Layout-only mode** does not prove GS1 data content matches; it checks **placement and structure** (within tolerances), not payload semantics.
- **Legacy mode** does not guarantee readable text; it uses heuristics on clips and typed rules, not human proofreading.

---

## How PDF-to-PDF selected-template validation works

| Step | Component | Notes |
|------|-----------|--------|
| 1 | `GroundTruthTemplateCatalog` | Lists products/versions; resolves `label.pdf`, `template.json`, optional `label.png`. |
| 2 | `load_region_config_file` | Strict JSON; fail-fast on bad numerics / thresholds. |
| 3 | `validate_pdf_pair_page_index` | Same page count; `page_index` in range (no silent wrong page). |
| 4 | `extract_regions` | One clip per region; `PDFExtraction.page_size_pdf` records the page box. |
| 5 | Layout gates (if `--layout-only`) | Matching page dimensions (pt); optional **aspect** check vs `label_size_mm` when `label_rect_pts` is **absent**. |
| 6 | `validate_extractions` or `validate_layout_extractions` | Per-region scores + details (mm shifts, bboxes, reasons). |
| 7 | `aggregate_region_scores` | Weighted mean, mandatory pass/fail, optional regions excluded from score. |
| 8 | `build_report_payload` / `write_html_report` | Structured JSON (`overall_result`, `failed_regions`, …) and HTML tables. |

Coordinate chain (thesis-relevant): **normalized bbox** (`bbox_norm` in \([0,1]^4\) relative to label rect) → **PDF points** (`bbox_norm_to_pdf_rect`) → **clip pixels** (fixed output size per region) → **mm** for tolerances via `label_size_mm` and label rect extent in pt (`layout_geometry.pdf_boxes_shift_size_mm`, etc.).

---

## Region schemas

- **Path:** `config/regions/<PRODUCT>.json` (override with `--regions-dir`).
- **Required top-level:** `product_code`, `page_size` (`width` / `height` canonical pixels for legacy mapping), `regions` array with at least one **mandatory** region.
- **Per region:** `name`, `type`, geometry (`bbox_norm` **or** legacy `x,y,width,height`), `mandatory` (bool), `weight` (non-negative; optional regions typically `weight: 0`).
- **Optional keys:** `label_rect_pts` (PDF crop for norm coords), `tolerance_mm`, `threshold`, type-specific extras (`barcode_dark_thresh`, …).
- **Auto-schema:** `--allow-auto-schema` generates JSON from `template.json` into `outputs/generated_regions/` — approximate, not recommended for production.

---

## Supported region types

**Legacy validators** (`region_validators.py`) — selected by `type` in JSON:

| `type` | Behaviour (summary) |
|--------|---------------------|
| `text_block` | Span geometry, containment, font sanity; **ignores string content**. |
| `barcode_block` | Dark-pixel / edge density; **does not decode** symbology. |
| `logo_block`, `free_form_block` | SSIM / pixel / edge on clips. |
| `optional_image` | Skipped for weighted score when `mandatory: false`. |

**Layout-only mode** (`--layout-only`, `layout_region_validators.py`) maps additional aliases to geometry/vector/edge handlers, including: `header`, `footer`, `datamatrix_block`, `icon_block`, `image_block`, `placeholder_block`, `table_block`, `divider_line`, `border_box`. Unknown types fall back to edge/structure heuristics with a warning.

---

## Variable content ignored (typical cases)

- **Text literal content** in template vs production label — validators focus on **lines, bbox, overflow, font size bands**, not whether the SKU string matches.
- **Barcode / DataMatrix payload** — outer **ink extent** and layout tolerances (layout mode), not decoded value equality.
- **Optional regions** — excluded from weighted `final_score`; candidate may omit optional extractions in layout mode (`skipped_optional_regions` in report).

---

## Layout / visual deviations detected (layout-only)

Examples of what **can** surface in `failure_reasons`, notes, and report metrics:

- Text block: overflow outside region, line-count drift, font-size mismatch, anchor shift in mm vs tolerance.
- Barcode block: centre/size shift in mm vs tolerance, missing ink on template or candidate.
- Logo / image blocks: ink bbox + edge-structure similarity vs thresholds.
- Vector-like regions: drawing-derived stats vs raster fallbacks when vectors are thin.
- Global: page count mismatch, page index OOB, PDF page size mismatch between template and candidate; label vs page aspect mismatch when label rect is full page.

---

## Thesis stance: not ML-first

See the dedicated subsection in the [root README](../README.md#thesis-stance-why-not-ml-first).

---

## Limitations (current, explicit)

1. **`label_rect_pts` set:** `assert_label_aspect_consistent_with_pdf` is **skipped** in the pipeline because physical aspect must be interpreted against the label crop, not the full media box — a full-page aspect check would be misleading.
2. **`severity` in layout details** (`critical` / `warning` / `none`) does **not** independently override aggregation; verdict is still **mandatory pass + combined ≥ threshold** (warnings list regions with combined &lt; 1.0 but still passed their per-region threshold).
3. **Schema defaults:** e.g. `mandatory` defaults to true, `weight` inferred if omitted — strict “every key present” is not enforced so existing JSON keeps working.
4. **Gates:** page-count, page-index, and page-size checks are covered in unit tests (`tests/test_layout_hardening.py`); end-to-end CLI tests for every `RegionPipelineError` path are not exhaustive.
5. **Vector paths** depend on PyMuPDF `get_drawings()`; flattened artwork may fall back to raster/edge analysis **within the region clip** (still not full-page SSIM as the main decision).

---

## CLI examples

From the `visual-validation/` directory:

```bash
# List raster-validation templates (validate.py)
python scripts/validate.py --list-templates

# List products under a ground_truth root (Python)
PYTHONPATH=. python -c "
from pathlib import Path
from src.core.template_store import GroundTruthTemplateCatalog
root = Path('data/templates/ground-truth-ikea-labels/ground_truth')
cat = GroundTruthTemplateCatalog(root)
for p in cat.list_products()[:15]:
    print(p, cat.list_versions(p)[-3:])
print('…', len(cat.list_products()), 'products')
"

# Validate one PDF (strict region schema)
PYTHONPATH=. python scripts/validate_label.py "data/labels/label.pdf" \
  --product 1R10-PRADM \
  --version v6

# Same with explicit page (multi-page PDFs) and custom report directory
PYTHONPATH=. python scripts/validate_label.py "data/labels/label.pdf" \
  -p 1R10-PRADM --version v6 \
  --page 0 \
  --out-dir outputs/reports

# Layout-first (geometry / tolerances mm; text & barcode payload not compared)
PYTHONPATH=. python scripts/validate_label.py "data/labels/label.pdf" \
  -p 1R10-PRADM --version v6 \
  --layout-only --layout-tolerance-mm 1.5 \
  --out-dir outputs/reports
```

**Default report paths** (`--out-dir`, default `outputs/reports/`):

| Artifact | Pattern |
|----------|---------|
| HTML | `report_<PRODUCT>_<candidate_stem>.html` |
| JSON | `report_<PRODUCT>_<candidate_stem>.json` |
| Candidate overview | `report_<PRODUCT>_<candidate_stem>_overview.png` |
| Template overview | `report_<PRODUCT>_<candidate_stem>_template_overview.png` |

With `--allow-auto-schema`, an extra file may appear under `--generated-regions-dir` (default `outputs/generated_regions/<PRODUCT>_<VERSION>.json`).
