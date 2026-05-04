#!/usr/bin/env python3
"""
export_label_pdfs.py — Export every atomic label variant as a single-page PDF.

Reads ground_truth/templates/{family}/v{N}/template.json for source_page +
label_rect_pts, then clips that region from the source PDF and saves it as a
vector-preserving single-page PDF at the same path structure under
ground_truth/label_pdfs/{family}/v{N}/label.pdf

Usage:
    python export_label_pdfs.py
"""

import fitz  # pymupdf
import json
from pathlib import Path

PDF_PATH  = Path("label 2.pdf")
GT_ROOT   = Path("ground_truth")
IN_ROOT   = GT_ROOT / "templates"
OUT_ROOT  = GT_ROOT / "label_pdfs"
CROP_PAD  = 3   # same 3pt padding used when making the PNG crops


def export_pdf_crop(src_doc: fitz.Document, page_no: int, rect_pts: dict, dest: Path):
    """
    Clip `rect_pts` from page `page_no` (1-based) of src_doc and write it
    as a single-page PDF to `dest`, preserving all vector content.
    """
    page = src_doc[page_no - 1]
    pw, ph = page.rect.width, page.rect.height

    x0 = max(0, rect_pts["x0"] - CROP_PAD)
    y0 = max(0, rect_pts["y0"] - CROP_PAD)
    x1 = min(pw, rect_pts["x1"] + CROP_PAD)
    y1 = min(ph, rect_pts["y1"] + CROP_PAD)
    clip = fitz.Rect(x0, y0, x1, y1)

    # New single-page PDF sized exactly to the clip region
    out_doc = fitz.open()
    new_page = out_doc.new_page(width=clip.width, height=clip.height)

    # show_pdf_page renders the source region onto the new page (vector-preserving)
    new_page.show_pdf_page(
        new_page.rect,  # destination rect = full new page
        src_doc,
        page_no - 1,    # 0-based source page index
        clip=clip,      # source region
    )

    dest.parent.mkdir(parents=True, exist_ok=True)
    out_doc.save(str(dest), garbage=4, deflate=True)
    out_doc.close()


def main():
    src_doc = fitz.open(str(PDF_PATH))
    print(f"Source PDF: {len(src_doc)} pages")

    template_jsons = sorted(IN_ROOT.glob("*/v*/template.json"))
    print(f"Found {len(template_jsons)} template variants to export\n")

    ok = skip = err = 0
    for tjson in template_jsons:
        data = json.loads(tjson.read_text())

        src_page = data.get("source_page")
        rect_pts = data.get("label_rect_pts")

        if not src_page or not rect_pts:
            print(f"  SKIP (no source_page/rect): {tjson.relative_to(GT_ROOT)}")
            skip += 1
            continue

        # Mirror the folder structure under label_pdfs/
        rel = tjson.parent.relative_to(IN_ROOT)   # e.g. 1R10-PRADM/v1
        dest = OUT_ROOT / rel / "label.pdf"

        try:
            export_pdf_crop(src_doc, src_page, rect_pts, dest)
            print(f"  ✓ {rel}/label.pdf  (page {src_page})")
            ok += 1
        except Exception as e:
            print(f"  ✗ {rel}: {e}")
            err += 1

    src_doc.close()
    print(f"\n{'='*60}")
    print(f"Done — {ok} exported, {skip} skipped, {err} errors")
    print(f"Output: {OUT_ROOT}/")


if __name__ == "__main__":
    main()
