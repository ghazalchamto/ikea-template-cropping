"""
Build the IKEA label ground truth from label 2.pdf.

Extracts every page as a PNG, organises them by template family,
copies existing rich JSON templates from the old ground_truth repo,
and creates stub JSONs + a master index for every other template.
"""

import fitz  # pymupdf
import json
import shutil
from pathlib import Path

PDF_PATH = Path(__file__).parent / "label 2.pdf"
OUT_ROOT = Path(__file__).parent / "ground_truth"
OLD_GT   = Path("/Users/sham_sara/Desktop/ikea-template-cropping/ground_truth")

DPI = 150   # good quality without huge files (~A4 landscape ≈ 1754×1240 px)

# ── Page → template mapping ────────────────────────────────────────────────
# key  = folder name (safe for filesystem)
# pages = 1-based page numbers that belong to this template family
# title = human-readable name shown in index
# existing_json = list of template JSON basenames already extracted in old repo

TEMPLATE_MAP = [
    {
        "key": "general_rules",
        "title": "General Rules — DM barcode specs",
        "pages": [1],
        "family": None,
        "existing_json": [],
    },
    {
        "key": "element_variants",
        "title": "Element Variants — logos, product names, article numbers",
        "pages": [2],
        "family": None,
        "existing_json": [],
    },
    {
        "key": "1R10-PRADM",
        "title": "1R10-PRADM — Narrow 1-row label (0.5 x-dim DataMatrix)",
        "pages": [3, 6],
        "family": "1R10-PRADM",
        "existing_json": [
            "1R10-PRADM_v1.json",
            "1R10-PRADM_v2.json",
            "1R10-PRADM_v3.json",
            "1R10-PRADM_v4.json",
            "1R10-PRADM_v5.json",
        ],
    },
    {
        "key": "1R20-PRADM",
        "title": "1R20-PRADM — Wide 1-row label (0.4x and 0.75x DataMatrix variants)",
        "pages": [4, 5],
        "family": "1R20-PRADM",
        "existing_json": [
            "1R20-PRADM_04x_v1.json",
            "1R20-PRADM_04x_v2.json",
            "1R20-PRADM_04x_v3.json",
            "1R20-PRADM_04x_v4.json",
            "1R20-PRADM_04x_v5.json",
            "1R20-PRADM_075x_v1.json",
            "1R20-PRADM_075x_v2.json",
            "1R20-PRADM_075x_v3.json",
            "1R20-PRADM_075x_v4.json",
            "1R20-PRADM_075x_v5.json",
        ],
    },
    {
        "key": "1R12-IDADM-IDBDM",
        "title": "1R12-IDADM/IDBDM — 1-row 12× label with DataMatrix",
        "pages": [7, 8, 9],
        "family": "1R12",
        "existing_json": [],
    },
    {
        "key": "1R35-IDADM-IDBDM",
        "title": "1R35-IDADM/IDBDM — 1-row 35× label (0.5 and 0.75 x-dim)",
        "pages": [10, 11, 12],
        "family": "1R35",
        "existing_json": [],
    },
    {
        "key": "1R25-IDCDM-IDCSDM",
        "title": "1R25-IDCDM / 1R25-IDCSDM — 1-row 25× with copy/ITF",
        "pages": [13],
        "family": "1R25",
        "existing_json": [],
    },
    {
        "key": "1C70-IDBDM",
        "title": "1C70-IDBDM — 1-column 70mm tall label with DataMatrix",
        "pages": [14],
        "family": "1C70",
        "existing_json": [],
    },
    {
        "key": "1C50-IDBDM",
        "title": "1C50-IDBDM — 1-column 50×35mm label with DataMatrix",
        "pages": [15, 16],
        "family": "1C50",
        "existing_json": [],
    },
    {
        "key": "1R35-IDGDM",
        "title": "1R35-IDGDM — 1-row 35× label, PO version with ITF",
        "pages": [17],
        "family": "1R35",
        "existing_json": [],
    },
    {
        "key": "1R15-PRADM-USA",
        "title": "1R15-PRADM — US 'Assembled in USA from Imported Materials' variant",
        "pages": [18, 19],
        "family": "1R15",
        "existing_json": [],
    },
    {
        "key": "1CIR50-1CIR60",
        "title": "1CIR50 / 1CIR60 — Circular/round labels (IKEA address only sample)",
        "pages": [20],
        "family": "1CIR",
        "existing_json": [],
    },
    {
        "key": "L3815-L8015-PRADM",
        "title": "L3815-PRADM / L8015-PRADM — Small rectangular press-on labels",
        "pages": [21],
        "family": "L-PRADM",
        "existing_json": [],
    },
    {
        "key": "L567-L578-1R7-PRADM",
        "title": "L567-PRADM / L578-PRADM / 1R7 — Tiny and ultra-narrow labels",
        "pages": [22],
        "family": "L-PRADM",
        "existing_json": [],
    },
    {
        "key": "1CIR30",
        "title": "1CIR30 — 30mm circular label (Patents layout)",
        "pages": [23],
        "family": "1CIR",
        "existing_json": [],
    },
    {
        "key": "ITF-barcode-specs",
        "title": "ITF Barcode Specifications — shared reference for ITF symbols",
        "pages": [24],
        "family": None,
        "existing_json": [],
    },
    {
        "key": "1C50-IDEDM",
        "title": "1C50-IDEDM — 1-column 50mm label with dimensions/weight",
        "pages": [25, 26],
        "family": "1C50",
        "existing_json": [],
    },
    {
        "key": "1C70-IDDDM",
        "title": "1C70-IDDDM — 1-column 70mm label with product picture zone",
        "pages": [27, 28],
        "family": "1C70",
        "existing_json": [],
    },
    {
        "key": "1C50-CAA",
        "title": "1C50-CAA — 1-column 50mm Care & Content label",
        "pages": [29],
        "family": "1C50",
        "existing_json": [],
    },
    {
        "key": "1C70-CAA",
        "title": "1C70-CAA — 1-column 70×107mm Care & Content label",
        "pages": [30],
        "family": "1C70",
        "existing_json": [],
    },
    {
        "key": "L10555-MIADM",
        "title": "L10555-MIADM — RTS/SS/FS module pallet label",
        "pages": [31],
        "family": "L-MIADM",
        "existing_json": [],
    },
    {
        "key": "L5727-PRADM",
        "title": "L5727-PRADM — Hinge hole kitchen label (19.2×6.4mm)",
        "pages": [32],
        "family": "L-PRADM",
        "existing_json": [],
    },
    {
        "key": "L4515-PRADM",
        "title": "L4515-PRADM — 45×15mm press-on label",
        "pages": [33, 34],
        "family": "L-PRADM",
        "existing_json": [],
    },
    {
        "key": "2R20-IDEDM",
        "title": "2R20-IDEDM — 2-row 20mm label with legal symbols",
        "pages": [35],
        "family": "2R20",
        "existing_json": [],
    },
    {
        "key": "2R35-IDEDM",
        "title": "2R35-IDEDM — 2-row 35mm label with legal symbols",
        "pages": [36],
        "family": "2R35",
        "existing_json": [],
    },
    {
        "key": "1R90-1R70-1R50",
        "title": "1R90 / 1R70 / 1R50 — Large square/rectangular labels overview",
        "pages": [37],
        "family": "1R-large",
        "existing_json": [],
    },
    {
        "key": "1R7-IDBDM",
        "title": "1R7-IDBDM — Ultra-narrow 7mm label with legal symbols",
        "pages": [38],
        "family": "1R7",
        "existing_json": [],
    },
    {
        "key": "1R25-IDEDM",
        "title": "1R25-IDEDM — 1-row 25mm label with dimensions/weight/ITF",
        "pages": [39, 40, 41],
        "family": "1R25",
        "existing_json": [],
    },
    {
        "key": "1R35-ITF-addendum",
        "title": "1R35 ITF barcode addendum pages",
        "pages": [42, 49],
        "family": "1R35",
        "existing_json": [],
    },
    {
        "key": "1R50-ITF-addendum",
        "title": "1R50 ITF barcode addendum page",
        "pages": [43],
        "family": "1R50",
        "existing_json": [],
    },
    {
        "key": "L25025-IDFDM",
        "title": "L25025-IDFDM — 250×25mm UK import label",
        "pages": [44],
        "family": "L-IDFDM",
        "existing_json": [],
    },
    {
        "key": "1CIR24",
        "title": "1CIR24 — 24mm circular label (0.3mm x-dim DM)",
        "pages": [45],
        "family": "1CIR",
        "existing_json": [],
    },
    {
        "key": "1CIR40-IDBDM",
        "title": "1CIR40-IDBDM — 40mm circular label with DataMatrix",
        "pages": [46],
        "family": "1CIR40",
        "existing_json": [],
    },
    {
        "key": "COA-IDADM-NUA",
        "title": "COA-IDADM-NUA — Country of Assembly + NUA compliance label",
        "pages": [47],
        "family": "COA",
        "existing_json": [],
    },
    {
        "key": "1CIR40-IDFDM",
        "title": "1CIR40-IDFDM — 40mm circular label for plants/pots",
        "pages": [48],
        "family": "1CIR40",
        "existing_json": [],
    },
]

# Verify every page 1-49 is covered exactly once
all_pages = []
for t in TEMPLATE_MAP:
    all_pages.extend(t["pages"])
assert sorted(all_pages) == list(range(1, 50)), f"Page coverage gap: {sorted(all_pages)}"


def render_page(doc, page_no_1based: int, dest: Path):
    """Render a PDF page (1-based) to a PNG at DPI."""
    page = doc[page_no_1based - 1]
    mat = fitz.Matrix(DPI / 72, DPI / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    pix.save(str(dest))


def make_stub_json(entry: dict, img_paths: list[str]) -> dict:
    return {
        "$schema": "../../shared_specs/schema/template.schema.json",
        "template_id": entry["key"],
        "display_name": entry["title"],
        "family": entry["family"],
        "source_pages": entry["pages"],
        "visual_artifacts": img_paths,
        "review_state": {
            "confidence": 0.0,
            "needs_review": True,
            "review_notes": (
                "Stub — visual artifacts extracted from PDF. "
                "Elements, rules, and validation_targets not yet populated."
            ),
        },
        "elements": [],
        "rules": [],
        "validation_targets": [],
    }


def build():
    doc = fitz.open(str(PDF_PATH))
    print(f"PDF opened: {len(doc)} pages")

    OUT_ROOT.mkdir(exist_ok=True)

    index_entries = []

    for entry in TEMPLATE_MAP:
        folder = OUT_ROOT / "templates" / entry["key"]
        folder.mkdir(parents=True, exist_ok=True)

        img_paths = []
        for pg in entry["pages"]:
            img_name = f"page_{pg:02d}.png"
            dest = folder / img_name
            render_page(doc, pg, dest)
            img_paths.append(img_name)
            print(f"  ✓ page {pg:02d} → {entry['key']}/{img_name}")

        # Copy or create JSON templates
        json_files_written = []
        if entry["existing_json"]:
            for jname in entry["existing_json"]:
                src = OLD_GT / "templates" / jname
                if src.exists():
                    shutil.copy2(src, folder / jname)
                    json_files_written.append(jname)
                else:
                    print(f"  ⚠ missing old template: {jname}")
        else:
            # Write a stub
            stub_name = f"{entry['key']}_stub.json"
            stub = make_stub_json(entry, img_paths)
            (folder / stub_name).write_text(json.dumps(stub, indent=2, ensure_ascii=False))
            json_files_written.append(stub_name)
            print(f"  → stub: {stub_name}")

        index_entries.append({
            "key": entry["key"],
            "title": entry["title"],
            "family": entry["family"],
            "source_pages": entry["pages"],
            "folder": f"templates/{entry['key']}",
            "visual_artifacts": [f"templates/{entry['key']}/{p}" for p in img_paths],
            "json_files": [f"templates/{entry['key']}/{j}" for j in json_files_written],
            "has_rich_json": bool(entry["existing_json"]),
        })

    # Copy shared specs
    shared_dest = OUT_ROOT / "shared_specs"
    shared_dest.mkdir(exist_ok=True)
    for fname in ["element_variants.json", "barcode_specs.json", "zone_definitions.json"]:
        src = OLD_GT / "shared_specs" / fname
        if src.exists():
            shutil.copy2(src, shared_dest / fname)
            print(f"  ✓ shared_spec: {fname}")

    # Copy schemas
    schema_dest = OUT_ROOT / "schema"
    schema_dest.mkdir(exist_ok=True)
    for src in (OLD_GT / "schema").glob("*.json"):
        shutil.copy2(src, schema_dest / src.name)
        print(f"  ✓ schema: {src.name}")

    # Write master index
    index = {
        "version": "1.0",
        "source_pdf": "label 2.pdf",
        "total_pages": 49,
        "general_pages": [1, 2],
        "template_pages_start": 3,
        "description": (
            "Ground truth visual + rule index for IKEA label validation. "
            "Each template folder contains PNG renderings of the spec pages "
            "and JSON rule files. Templates with has_rich_json=true have "
            "fully extracted element specs and validation rules. "
            "Templates with has_rich_json=false have stub JSONs and need enrichment."
        ),
        "templates": index_entries,
    }
    (OUT_ROOT / "index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False))
    print(f"\n✅ Ground truth built → {OUT_ROOT}")
    print(f"   {len(index_entries)} template entries, "
          f"{sum(1 for e in index_entries if e['has_rich_json'])} with rich JSON, "
          f"{sum(1 for e in index_entries if not e['has_rich_json'])} stubs")


if __name__ == "__main__":
    build()
