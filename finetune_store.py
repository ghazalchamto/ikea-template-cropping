"""
Human-in-the-loop corrections: apply to ExtractedLabel, export JSONL for retraining.
"""

from __future__ import annotations

import json
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from extractor._helpers import _make_bbox
from extractor.models import BBox, BarcodeResult, ExtractedField, ExtractedLabel

TEXT_FIELD_ATTRS: list[tuple[str, str, str]] = [
    ("product_name", "product_name", "Product name"),
    ("product_dimensions_metric", "product_dimensions_metric", "Dimensions (metric)"),
    ("product_dimensions_imperial", "product_dimensions_imperial", "Dimensions (imperial)"),
    ("article_number", "article_number", "Article number"),
    ("article_code", "article_code", "Article code"),
    ("internal_item_number", "internal_item_number", "Internal item no."),
    ("quantity_multiplier", "quantity_multiplier", "Quantity multiplier"),
    ("gross_weight", "gross_weight", "Gross weight"),
    ("net_weight", "net_weight", "Net weight"),
    ("package_number", "package_number", "Package number"),
    ("package_type", "package_type", "Package type"),
    ("copy_count_inline", "copy_count_inline", "Copy (inline)"),
    ("ikea_logo", "ikea_logo", "IKEA logo"),
    ("ean_barcode_value", "ean_barcode_value", "EAN-13 value (text)"),
    ("itf14_human_readable", "itf14_human_readable", "ITF-14 human readable"),
    ("ai_240", "ai_240", "AI(240)"),
    ("ai_13", "ai_13", "AI(13)"),
    ("ai_11", "ai_11", "AI(11)"),
    ("ai_10", "ai_10", "AI(10)"),
    ("human_readable_date", "human_readable_date", "Human-readable date"),
    ("date_stamp", "date_stamp", "Date stamp"),
    ("date_alpha_code", "date_alpha_code", "Alpha code"),
    ("date_numeric_prefix", "date_numeric_prefix", "Numeric prefix"),
    ("custom_identifier", "custom_identifier", "Custom identifier"),
    ("plant_identifier", "plant_identifier", "Plant identifier"),
    ("supplier_name", "supplier_name", "Supplier name"),
    ("supplier_address", "supplier_address", "Supplier address"),
    ("address_block", "address_block", "Address block"),
    ("origin_text", "origin_text", "Origin"),
    ("copyright_notice", "copyright_notice", "Copyright"),
    ("compliance_marks", "compliance_marks", "Compliance marks"),
    ("age_rating", "age_rating", "Age rating"),
    ("copy_count", "copy_count", "Copy count"),
]

BARCODE_FIELD_ATTRS: list[tuple[str, str, str]] = [
    ("ean_barcode", "ean_barcode", "EAN-13 (decoded)"),
    ("itf14_barcode", "itf14_barcode", "ITF-14 (decoded)"),
    ("datamatrix_barcode", "datamatrix_barcode", "DataMatrix (decoded)"),
]


@dataclass
class FieldCorrection:
    field_id: str
    value: str
    bbox: Optional[dict[str, int]] = None
    previous_value: Optional[str] = None
    previous_bbox: Optional[dict[str, int]] = None
    is_barcode: bool = False
    record_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


def _bbox_to_dict(b: Optional[BBox]) -> Optional[dict[str, int]]:
    if b is None:
        return None
    return {"x": b.x, "y": b.y, "w": b.w, "h": b.h}


def _get_text_snapshot(label: ExtractedLabel, attr: str) -> tuple[Optional[str], Optional[dict]]:
    fld: Optional[ExtractedField] = getattr(label, attr, None)
    if fld is None:
        return None, None
    val = str(fld.value) if fld.value is not None else None
    return val, _bbox_to_dict(fld.bbox) if fld.bbox else None


def _get_barcode_snapshot(label: ExtractedLabel, attr: str) -> tuple[Optional[str], Optional[dict]]:
    br: Optional[BarcodeResult] = getattr(label, attr, None)
    if br is None:
        return None, None
    return br.raw_data, _bbox_to_dict(br.bbox) if br.bbox else None


def snapshot_model_for_field(label: ExtractedLabel, field_id: str) -> tuple[Optional[str], Optional[dict], bool]:
    for fid, a, _ in TEXT_FIELD_ATTRS:
        if fid == field_id:
            v, bb = _get_text_snapshot(label, a)
            return v, bb, False
    for fid, a, _ in BARCODE_FIELD_ATTRS:
        if fid == field_id:
            v, bb = _get_barcode_snapshot(label, a)
            return v, bb, True
    return None, None, False


def _dict_to_bbox(d: Optional[dict[str, int]], img_w: int, img_h: int, zone: str = "") -> Optional[BBox]:
    if not d:
        return None
    return _make_bbox(
        int(d["x"]),
        int(d["y"]),
        int(d["w"]),
        int(d["h"]),
        img_w,
        img_h,
        zone=zone,
    )


def apply_corrections(
    label: ExtractedLabel,
    corrections: dict[str, dict[str, Any]],
) -> ExtractedLabel:
    out = deepcopy(label)
    iw, ih = out.metadata.image_width_px, out.metadata.image_height_px

    for fid, attr, _ in TEXT_FIELD_ATTRS:
        if fid not in corrections:
            continue
        c = corrections[fid]
        val = c.get("value")
        if val is None:
            continue
        sval = str(val)
        bb_dict = c.get("bbox")
        previous: Optional[ExtractedField] = getattr(out, attr, None)
        bbox: Optional[BBox] = _dict_to_bbox(bb_dict, iw, ih) if bb_dict else (
            previous.bbox if previous else None
        )
        if previous is None:
            setattr(
                out,
                attr,
                ExtractedField(
                    value=sval,
                    raw_text=sval,
                    confidence=1.0,
                    zone=None,
                    bbox=bbox,
                    source="human",
                ),
            )
        else:
            previous.value = sval
            previous.raw_text = sval
            previous.confidence = 1.0
            previous.source = "human"
            if bbox is not None:
                previous.bbox = bbox

    for fid, attr, _ in BARCODE_FIELD_ATTRS:
        if fid not in corrections:
            continue
        c = corrections[fid]
        val = c.get("value")
        if val is None:
            continue
        sval = str(val)
        br: Optional[BarcodeResult] = getattr(out, attr, None)
        if br is None:
            bb2 = _dict_to_bbox(c.get("bbox"), iw, ih) if c.get("bbox") else None
            setattr(
                out,
                attr,
                BarcodeResult(
                    symbology="Unknown",
                    raw_data=sval,
                    decoded_text=sval,
                    bbox=bb2,
                    confidence=1.0,
                ),
            )
        else:
            br.raw_data = sval
            br.decoded_text = sval
            br.confidence = 1.0
            if c.get("bbox"):
                br.bbox = _dict_to_bbox(c["bbox"], iw, ih)

    return out


def build_training_records(
    file_hash: str,
    dpi: int,
    model_label: ExtractedLabel,
    corrections: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for fid, c in corrections.items():
        if not c.get("value") and c.get("value") != "":
            continue
        model_val, model_bb, is_bc = snapshot_model_for_field(model_label, fid)
        records.append(
            {
                "file_hash": file_hash,
                "dpi": dpi,
                "image_width": model_label.metadata.image_width_px,
                "image_height": model_label.metadata.image_height_px,
                "field_id": fid,
                "corrected_value": c.get("value"),
                "corrected_bbox": c.get("bbox"),
                "model_value": model_val,
                "model_bbox": model_bb,
                "is_barcode": is_bc,
            }
        )
    return records


def save_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def new_export_path(base_dir: Path) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return base_dir / f"corrections_{ts}.jsonl"


def all_field_choices() -> list[tuple[str, str]]:
    o: list[tuple[str, str]] = []
    o.extend((fid, f"{lbl}  [{fid}]") for fid, _a, lbl in TEXT_FIELD_ATTRS)
    o.extend((fid, f"{lbl}  [{fid}]") for fid, _a, lbl in BARCODE_FIELD_ATTRS)
    return o
