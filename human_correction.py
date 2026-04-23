"""
Streamlit tab: set bounding boxes (sliders in % of image), edit text,
re-validate, export JSONL, optional PDF text replacement (redact + insert).

We avoid streamlit-drawable-canvas: it relies on removed private Streamlit APIs
(see streamlit.elements.image).
"""

from __future__ import annotations

import json
from typing import Any, Optional, Tuple

import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from config import FINETUNE_DATA_DIR
from finetune_store import (
    TEXT_FIELD_ATTRS,
    all_field_choices,
    apply_corrections,
    build_training_records,
    new_export_path,
    save_jsonl,
    snapshot_model_for_field,
)
from label_upload import load_image_matching_extractor, sniff_suffix
from pdf_label_edit import apply_text_edits_to_pdf_bytes, HAS_FITZ as PDF_EDIT_AVAILABLE


def _field_ids_text_only() -> set[str]:
    return {fid for fid, _a, _ in TEXT_FIELD_ATTRS}


def _default_pcts(
    w0: int, h0: int, cbb: Optional[dict], mbb: Optional[dict]
) -> Tuple[float, float, float, float]:
    if cbb and cbb.get("w", 0) and cbb.get("h", 0):
        b = cbb
    elif mbb and mbb.get("w", 0) and mbb.get("h", 0):
        b = mbb
    else:
        return 2.0, 2.0, 18.0, 6.0
    return (
        100.0 * b["x"] / w0,
        100.0 * b["y"] / h0,
        100.0 * b["w"] / w0,
        100.0 * b["h"] / h0,
    )


def _pct_to_bbox(
    w0: int, h0: int, left_pct: float, top_pct: float, w_pct: float, h_pct: float
) -> dict[str, int]:
    x = int(max(0, min(w0 - 1, w0 * left_pct / 100.0)))
    y = int(max(0, min(h0 - 1, h0 * top_pct / 100.0)))
    w = int(max(1, w0 * w_pct / 100.0))
    h = int(max(1, h0 * h_pct / 100.0))
    w = min(w, w0 - x)
    h = min(h, h0 - y)
    return {"x": x, "y": y, "w": w, "h": h}


def _draw_corrections_overlay(
    base: Image.Image,
    corrections: dict[str, dict[str, Any]],
) -> Image.Image:
    im = base.convert("RGBA")
    ovl = Image.new("RGBA", im.size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(ovl)
    try:
        fnt = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except OSError:
        try:
            fnt = ImageFont.truetype("arial.ttf", 12)
        except OSError:
            fnt = ImageFont.load_default()
    for _fid, c in corrections.items():
        bb = c.get("bbox")
        if not bb:
            continue
        col = (34, 197, 94, 55)
        out = (34, 197, 94, 230)
        x, y, w, h = int(bb["x"]), int(bb["y"]), int(bb["w"]), int(bb["h"])
        dr.rectangle([x, y, x + w, y + h], fill=col, outline=out, width=2)
        label = c.get("value") or _fid
        if len(str(label)) > 40:
            label = str(label)[:37] + "…"
        dr.text(
            (x + 2, max(0, y - 14)),
            f"{_fid}: {label}",
            fill=(34, 197, 94, 255),
            font=fnt,
        )
    return Image.alpha_composite(im, ovl).convert("RGB")


def _draw_draft_on_rgb(
    rgb: Image.Image,
    bbox: dict[str, int],
    label: str,
) -> Image.Image:
    im = rgb.convert("RGBA")
    ovl = Image.new("RGBA", im.size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(ovl)
    x, y, w, h = int(bbox["x"]), int(bbox["y"]), int(bbox["w"]), int(bbox["h"])
    dr.rectangle(
        [x, y, x + w, y + h], fill=(251, 191, 36, 50), outline=(245, 158, 11, 255), width=3
    )
    try:
        fnt = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except OSError:
        fnt = ImageFont.load_default()
    dr.text(
        (x + 2, max(0, y - 14)), f"draft: {label[:36]}", fill=(180, 83, 9, 255), font=fnt
    )
    return Image.alpha_composite(im, ovl).convert("RGB")


def render_human_correction_tab(
    *,
    file_bytes: bytes,
    file_hash: str,
    dpi: int,
    base_result: Any,
    upload_stem: str,
) -> None:
    st.markdown(
        "1) Pick a field and type the **corrected** text. 2) Use **sliders** (percent of full "
        "image) to set the bounding box, or uncheck *Include bounding box* for text-only. "
        "3) **Record correction**. 4) **Re-validate** to refresh scores. 5) **Export JSONL** for "
        "training. For **PDF** files, use **Build edited PDF** to redact+replace in stored regions."
    )

    if st.session_state.get("hc_file_key") != file_hash:
        st.session_state["hc_file_key"] = file_hash
        st.session_state["hc_corrections"] = {}
        st.session_state.pop("hc_edited_pdf_bytes", None)

    corrections: dict[str, dict[str, Any]] = st.session_state.get("hc_corrections", {})
    is_pdf = sniff_suffix(file_bytes) == ".pdf"
    text_fids = _field_ids_text_only()

    with st.spinner("Loading image (same scale as extraction)…"):
        try:
            work_img = load_image_matching_extractor(file_bytes, dpi)
        except Exception as e:
            st.error(f"Could not load label image: {e}")
            return

    w0, h0 = work_img.size
    if w0 != base_result.metadata.image_width_px or h0 != base_result.metadata.image_height_px:
        st.warning(
            "Image size from loader does not match extraction metadata; bboxes may be misaligned."
        )

    choices = all_field_choices()
    field_labels = {fid: lab for fid, lab in choices}
    field_ids = [fid for fid, _ in choices]

    c1, c2 = st.columns(2)
    with c1:
        sel = st.selectbox(
            "Field to correct",
            field_ids,
            format_func=lambda x: field_labels.get(x, x),
            key="hc_field_sel",
        )
    with c2:
        init_val, mbb, _is_bc = snapshot_model_for_field(base_result, sel)
        text_default = (corrections.get(sel) or {}).get("value", init_val) or ""
        val = st.text_input(
            "Corrected value",
            value=text_default,
            key=f"hc_v_{file_hash}_{sel}",
        )

    cbb = (corrections.get(sel) or {}).get("bbox")
    lp, tp, wp, hp = _default_pcts(w0, h0, cbb, mbb)

    use_bbox = st.checkbox(
        "Include bounding box for this field",
        value=True,
        key=f"hc_usebb_{file_hash}_{sel}",
    )

    st.caption("Bounding box: position and size as **percent** of the full label image (below).")
    cL, cT, cW, cH = st.columns(4)
    with cL:
        left_pct = st.slider("Left %", 0.0, 100.0, float(lp), 0.1, key=f"hcL_{file_hash}_{sel}")
    with cT:
        top_pct = st.slider("Top %", 0.0, 100.0, float(tp), 0.1, key=f"hcT_{file_hash}_{sel}")
    with cW:
        w_pct = st.slider("Width %", 0.1, 100.0, float(max(0.1, wp)), 0.1, key=f"hcW_{file_hash}_{sel}")
    with cH:
        h_pct = st.slider("Height %", 0.1, 100.0, float(max(0.1, hp)), 0.1, key=f"hcH_{file_hash}_{sel}")

    draft_bb: Optional[dict[str, int]] = None
    if use_bbox:
        draft_bb = _pct_to_bbox(w0, h0, left_pct, top_pct, w_pct, h_pct)

    # Preview: saved corrections in green, draft in orange
    if corrections and draft_bb and use_bbox:
        base_ovl = _draw_corrections_overlay(
            work_img, {k: v for k, v in corrections.items() if k != sel}
        )
        preview = _draw_draft_on_rgb(base_ovl, draft_bb, str(val) or sel)
    elif not corrections and draft_bb and use_bbox:
        preview = _draw_draft_on_rgb(work_img, draft_bb, str(val) or sel)
    else:
        preview = _draw_corrections_overlay(work_img, corrections)

    st.image(preview, use_container_width=True, caption="Green = stored corrections; orange = current draft (this field)")

    b1, b2, b3 = st.columns(3)
    with b1:
        record = st.button("Record correction", type="primary", key="hc_rec")
    with b2:
        if st.button("Clear this field", key="hc_clrf"):
            if sel in st.session_state.get("hc_corrections", {}):
                del st.session_state["hc_corrections"][sel]
            st.rerun()
    with b3:
        if st.button("Clear all corrections", key="hc_clra"):
            st.session_state["hc_corrections"] = {}
            st.rerun()

    if record:
        st.session_state.setdefault("hc_corrections", {})[sel] = {
            "value": val,
            "bbox": draft_bb if use_bbox else None,
        }
        if not use_bbox:
            st.info("Saved text only (no bbox).")
        st.success(f"Saved **{sel}**.")
        st.rerun()

    if corrections:
        st.markdown("**Active corrections (see preview above):**")
        for fid, c in corrections.items():
            st.write(f"- `{fid}`: `{c.get('value')}`  bbox={c.get('bbox')}")

    st.divider()
    st.subheader("Re-validate with corrections")
    if corrections:
        from validator import auto_validate  # local import: breaks cycle if any

        merged = apply_corrections(base_result, corrections)
        vrep, _sl = auto_validate(merged)
        st.metric("Compliance (merged)", f"{vrep.compliance_score:.0%}", help="From human-merged label")
        st.metric("Field rule errors (merged)", len(vrep.errors))
    else:
        st.caption("No corrections stored yet.")

    st.divider()
    st.subheader("Export for retraining (JSONL)")
    recs = build_training_records(file_hash, dpi, base_result, corrections) if corrections else []
    st.caption(f"{len(recs)} row(s) for this file.")
    if recs:
        jsonl_text = "\n".join(json.dumps(r, ensure_ascii=False) for r in recs) + "\n"
        st.download_button(
            "⬇️ Download JSONL (this document)",
            data=jsonl_text.encode("utf-8"),
            file_name=f"{upload_stem}_finetune_{file_hash}.jsonl",
            mime="application/x-ndjson",
        )
        if st.button("Append JSONL to finetune_data/ on disk"):
            p = new_export_path(FINETUNE_DATA_DIR)
            try:
                save_jsonl(p, recs)
                st.success(f"Appended to `{p}`.")
            except Exception as e:
                st.error(str(e))
    st.divider()
    st.subheader("Update PDF (redact + insert)")
    if not is_pdf:
        st.caption("Upload a PDF label to enable PDF rewriting.")
    elif not PDF_EDIT_AVAILABLE:
        st.caption("PyMuPDF not available for PDF edit.")
    else:
        st.caption("Only **text** fields with a **bbox** and non-empty value are written.")
        if st.button("Build edited PDF from corrections", key="hc_build_pdf"):
            edits = []
            for fid, c in corrections.items():
                if fid not in text_fids:
                    continue
                bb = c.get("bbox")
                txt = (c.get("value") or "").strip()
                if not bb or not txt:
                    continue
                edits.append(
                    {
                        "x": int(bb["x"]),
                        "y": int(bb["y"]),
                        "w": int(bb["w"]),
                        "h": int(bb["h"]),
                        "text": txt,
                    }
                )
            if not edits:
                st.error("Add at least one text field with a rectangle and value.")
            else:
                try:
                    st.session_state["hc_edited_pdf_bytes"] = apply_text_edits_to_pdf_bytes(
                        file_bytes, edits, dpi=dpi
                    )
                except Exception as e:
                    st.error(f"PDF edit failed: {e}")
        pdf_bytes = st.session_state.get("hc_edited_pdf_bytes")
        if pdf_bytes:
            st.download_button(
                "⬇️ Download edited PDF",
                data=pdf_bytes,
                file_name=f"{upload_stem}_edited.pdf",
                mime="application/pdf",
                key="hc_dl_pdf",
            )
