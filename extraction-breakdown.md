# Label extraction breakdown

This document describes **how** the first-stage label extractor works (`img-upload-layer/`) and **what** it outputs. That JSON is intended for a later step that compares against rules in `ground_truth/`.

---

## Scope

| In scope | Out of scope |
|----------|----------------|
| One uploaded file → structured JSON | Pass/fail validation against templates |
| PDF (first page) or raster image | Multi-page PDF beyond page 1 |
| Heuristics + OCR + barcode decode | Sub-pixel layout truth for every glyph |

**Entry points**

- CLI: `img-upload-layer/extract_label.py`
- Web UI: `img-upload-layer/streamlit_app.py`
- API in code: `from extractor import LabelExtractor` then `LabelExtractor(path).run()`

---

## How extraction works (pipeline)

The implementation lives in `img-upload-layer/extractor/label_extractor.py`. `LabelExtractor.run()` executes these steps in order:

### 1. Load

- **PDF**: Rendered to a bitmap via `pdf2image` (default **600 DPI**; wide images may be downscaled to max width **4500 px**).
- **Images**: PNG, JPEG, TIFF, BMP opened with Pillow and converted to RGB.

### 2. Preprocess

- Grayscale, increased **contrast** and **sharpness** on crops used for OCR (and optionally for barcode retries).

### 3. Decode barcodes

- **Linear (ITF-14, EAN-13)**: `pyzbar` on colour and grayscale images.
- **DataMatrix**: `pylibdmtx` on colour and grayscale (requires system **libdmtx**; if missing, 2D decode is skipped and OCR fallbacks apply).

Decoded symbols include **pixel bounding boxes** (`BBox`: `x`, `y`, `w`, `h`) in image coordinates.

### 4. Calibrate horizontal zones

- **Default**: Fixed fractions of image width (see table below).
- **If both ITF-14 and DataMatrix decode with bboxes**: Left/right boundaries of `left_identity`, `center_barcode`, `center_datamatrix`, and `center_address` are shifted so OCR crops line up with the real symbols and nearby print. A margin keeps DataMatrix **human-readable AI text** inside the datamatrix zone instead of the address zone.

### 5. Per-zone OCR (Tesseract)

The label width is split into **vertical strips** (same height as the image). Each strip is cropped, preprocessed, and passed to **Tesseract** with a zone-specific **PSM** (page segmentation mode) and sometimes **rotation** (notably the far-right copy-count digit).

| Zone | Default horizontal span (fraction of width) | PSM | Notes |
|------|---------------------------------------------|-----|--------|
| `left_identity` | 0.00 – 0.18 | 6 | Product name, dimensions, weight, inline copy count |
| `center_barcode` | 0.18 – 0.40 | 11 | Sparse text under/near ITF |
| `center_datamatrix` | 0.40 – 0.58 | 6 | AI lines, patent text near symbol |
| `center_address` | 0.58 – 0.72 | 6 | Address, origin continuation, legal lines |
| `right_compliance` | 0.72 – 0.84 | 11 | Keywords, age rating; extra PSM **6** then **3** if age pattern not found |
| `right_date` | 0.84 – 0.95 | 6 | DATE line, YYWW, alpha code |
| `right_copy_count` | 0.95 – 1.00 | 8, rotated | Narrow strip; retries alternate rotations if no digit |

Zone OCR does **not** currently attach per-field bounding boxes; most text fields record **`zone`** only. Barcodes keep **decoder bboxes**.

### 6. Field extraction (`extract_fields`)

Zone strings (and barcode payloads) are passed through **regex and small heuristics**, for example:

- Product name: runs of uppercase letters (with Scandinavian letters allowed).
- Article number under barcode: `NNN.NNN.NN` from OCR, or derived from ITF-14 digits.
- Article **code** (5 digits): from ITF-14 positions 9–13 of the 14-digit payload.
- GS1 **DataMatrix**: parsed via `gs1_parser` (binary GS1 and parenthesis-style OCR fallback); selected AIs promoted to top-level fields.
- **Origin**: token-based detection on **combined** datamatrix + address text so splits across zones (e.g. “Republic” / “China”) still resolve.
- **Address block**: lines anchored by IKEA/Sweden/postal patterns; optional strip of short leading digit noise before `IKEA`/`CEA`.
- **Patent**: tolerant of `PI` vs `P1` OCR; zone attributed to datamatrix when matched there.
- **Copyright**: tolerant of `|` instead of `©` and multiline text through `B.V.`.
- **Compliance**: keyword scan (CE, TSCA, ULEF, CARB, RoHS, EPEAT) plus raw zone text.
- **Date**: `DATE:` / `ATE:` patterns, YYWW fallback, 3-digit prefix line, 4-letter alpha line.

Models are defined in `img-upload-layer/extractor/models.py`.

### 7. Compliance image hint

- In the **compliance** crop, a simple **dark-pixel heuristic** may append symbolic hints (e.g. possible CE) to `compliance_marks`. This is coarse and not a certified CE detector.

---

## What is extracted (output JSON)

Serialization is `ExtractedLabel.to_dict()` (same structure the CLI and Streamlit print).

### Top-level keys

1. **`extraction_metadata`** — Source filename, UTC timestamp, image dimensions, render DPI, which libraries were available, overall confidence estimate, `warnings`, `errors`.
2. **`elements`** — Normalized fields for downstream validation.
3. **`zones`** — Per-zone OCR: `raw_lines`, `word_count`, `avg_confidence` (zone-level confidence is not fully populated for crop OCR).

### `elements` fields

| JSON key | Meaning | Typical sources |
|----------|---------|-----------------|
| `product_name` | Commercial name (uppercase heuristic) | `left_identity` OCR |
| `dimensions` | cm/mm and optional imperial fragment | `left_identity` OCR |
| `weight` | kg / lbs | `left_identity` OCR |
| `article_number` | `NNN.NNN.NN` | OCR under ITF and/or ITF-14 decode |
| `article_code` | 5-digit code | ITF-14 payload |
| `copy_count_inline` | e.g. `1(2)` next to name | `left_identity` OCR |
| `itf14_barcode` | Symbology, raw string, **bbox** | `pyzbar` |
| `itf14_human_readable` | Printed article format | OCR / barcode-derived |
| `datamatrix_block.barcode` | Raw DataMatrix, **bbox** | `pylibdmtx` |
| `datamatrix_block.content` | Raw payload + GS1 `ai_fields` dict | Decoder + `gs1_parser` |
| `datamatrix_block.ai_fields` | `240`, `13`, `11`, `10` | Same, mirrored as convenience |
| `address_block` | Multi-line postal / legal block | `center_address` OCR |
| `origin_text` | e.g. Made in … | Combined datamatrix + address OCR |
| `patent_info` | Patent / PI line | Datamatrix or address OCR |
| `copyright_notice` | Inter IKEA … B.V. style | Address / combined OCR |
| `compliance_marks` | Keywords + optional image hints | Compliance zone + heuristic |
| `age_rating` | e.g. `12M+` | Compliance OCR (multi-PSM) |
| `date_stamp` | YYWW-style value | `right_date` OCR |
| `date_label_present` | Boolean | Whether a DATE-style line was seen |
| `date_alpha_code` | Short alpha line | `right_date` OCR |
| `date_numeric_prefix` | e.g. 3-digit prefix | `right_date` OCR |
| `copy_count` | Digit in right margin | `right_copy_count` OCR |

### Shape of a text field (`ExtractedField`)

When present, most entries are objects with:

- `value` — Normalized or best string.
- `raw_text` — Supporting snippet or source text.
- `confidence` — Float 0–1 (heuristic; barcode-backed fields often higher).
- `zone` — Which vertical strip produced the text (or logical zone for promoted AIs).
- `bbox` — Usually **`null`** for OCR-derived fields today.
- `source` — e.g. `ocr`, `barcode`.
- `pattern_match` — Whether a strict pattern matched (when set).

### Barcode object (`BarcodeResult`)

Includes `symbology`, `raw_data`, `decoded_text`, `bbox`, `confidence`, `source`.

---

## Dependencies (runtime)

**Python** (see `img-upload-layer/requirements.txt`): Pillow, pdf2image, pytesseract, pyzbar, pylibdmtx, numpy; optional UI: streamlit.

**System**: Poppler (PDF), Tesseract (OCR), zbar (linear codes), libdmtx (DataMatrix). Missing native libraries show up in `extraction_metadata.libraries_used` and may reduce accuracy (e.g. no zone calibration without DataMatrix bbox).

---

## Relation to `ground_truth/`

Field names and roles are chosen to align with **template element semantics** in `ground_truth/` so a validator can map **extracted `elements.*`** to **rules and elements** in the JSON specs. This extractor does **not** load those JSON files; it only produces the structured capture of the uploaded label.
