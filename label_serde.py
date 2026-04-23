"""
Deserialize the JSON shape produced by ExtractedLabel.to_dict() back into
ExtractedLabel (for API round-trips). Used by the FastAPI server.
"""

from __future__ import annotations

from typing import Any, Optional

from extractor.models import (
    BBox,
    BarcodeResult,
    DataMatrixContent,
    ExtractedField,
    ExtractedLabel,
    ExtractionMetadata,
    ZoneSummary,
)
from extractor.pdf_passes import RawElement


def _bbox(d: Optional[dict]) -> Optional[BBox]:
    if not d:
        return None
    return BBox(
        x=int(d["x"]),
        y=int(d["y"]),
        w=int(d["w"]),
        h=int(d["h"]),
        x_frac=float(d.get("x_frac") or 0.0),
        y_frac=float(d.get("y_frac") or 0.0),
        w_frac=float(d.get("w_frac") or 0.0),
        h_frac=float(d.get("h_frac") or 0.0),
        zone=d.get("zone"),
    )


def _ef(d: Optional[dict]) -> Optional[ExtractedField]:
    if not d:
        return None
    return ExtractedField(
        value=d.get("value"),
        raw_text=d.get("raw_text"),
        confidence=float(d.get("confidence") or 0.0),
        zone=d.get("zone"),
        bbox=_bbox(d.get("bbox")),
        source=str(d.get("source") or "ocr"),
        pattern_match=d.get("pattern_match"),
    )


def _br(d: Optional[dict]) -> Optional[BarcodeResult]:
    if not d:
        return None
    return BarcodeResult(
        symbology=str(d.get("symbology") or ""),
        raw_data=str(d.get("raw_data") or ""),
        decoded_text=d.get("decoded_text"),
        bbox=_bbox(d.get("bbox")),
        confidence=float(d.get("confidence") or 0.0),
        source=str(d.get("source") or "barcode"),
    )


def _dmc(d: Optional[dict]) -> Optional[DataMatrixContent]:
    if not d:
        return None
    from extractor.models import AIField

    aif: dict = {}
    raw_af = d.get("ai_fields") or {}
    if isinstance(raw_af, dict):
        for k, v in raw_af.items():
            if not isinstance(v, dict):
                continue
            aif[k] = AIField(
                ai_code=str(v.get("ai_code") or k),
                name=str(v.get("name") or ""),
                description=str(v.get("description") or ""),
                value=v.get("value"),
                formatted=v.get("formatted"),
                present=bool(v.get("present", True)),
            )
    return DataMatrixContent(
        raw_data=str(d.get("raw_data") or ""),
        encoding=str(d.get("encoding") or "raw"),
        ai_fields=aif,
    )


def _raw_element(d: dict) -> RawElement:
    m = d.get("meta")
    if not isinstance(m, dict):
        m = {}
    return RawElement(
        type=str(d.get("type") or "region"),
        content=d.get("content"),
        x=int(d.get("x") or 0),
        y=int(d.get("y") or 0),
        w=int(d.get("w") or 0),
        h=int(d.get("h") or 0),
        confidence=float(d.get("confidence") or 1.0),
        font=d.get("font"),
        font_size=float(d.get("font_size") or 0.0),
        bold=bool(d.get("bold")),
        italic=bool(d.get("italic")),
        zone=str(d.get("zone") or ""),
        meta=m,
    )


def _zone_summary(d: dict) -> ZoneSummary:
    return ZoneSummary(
        zone_id=str(d.get("zone_id") or ""),
        raw_lines=list(d.get("raw_lines") or []),
        word_count=int(d.get("word_count") or 0),
        avg_confidence=float(d.get("avg_confidence") or 0.0),
    )


def label_from_dict(d: dict) -> ExtractedLabel:
    """Reconstruct ``ExtractedLabel`` from :meth:`ExtractedLabel.to_dict` output."""
    em = d.get("extraction_metadata")
    if not em:
        raise ValueError("Missing extraction_metadata")
    md = ExtractionMetadata(
        source_file=str(em.get("source_file") or ""),
        extracted_at=str(em.get("extracted_at") or ""),
        image_width_px=int(em.get("image_width_px") or 0),
        image_height_px=int(em.get("image_height_px") or 0),
        render_dpi=int(em.get("render_dpi") or 600),
        libraries_used=dict(em.get("libraries_used") or {}),
        overall_confidence=float(em.get("overall_confidence") or 0.0),
        warnings=list(em.get("warnings") or []),
        errors=list(em.get("errors") or []),
    )
    e = d.get("elements") or {}
    dmb = e.get("datamatrix_block") or {}
    if not isinstance(dmb, dict):
        dmb = {}
    dmx_barcode = dmb.get("barcode")
    dmx_content = dmb.get("content")
    if isinstance(dmx_content, dict) and dmx_content:
        dc = _dmc(dmx_content)
    else:
        dc = None
    # Parallel ai_fields on block store ExtractedField (same as top-level ai_240, …)
    ai_block = dmb.get("ai_fields")
    if not isinstance(ai_block, dict):
        ai_block = {}

    raw_w = d.get("raw_words")
    if not isinstance(raw_w, list):
        raw_w = []
    raw_el = d.get("raw_elements")
    if isinstance(raw_el, list):
        raw_els: list = [
            _raw_element(x) if isinstance(x, dict) else x for x in raw_el
        ]
    else:
        raw_els = []
    zd = d.get("zones")
    if isinstance(zd, dict):
        zones = {k: _zone_summary(v) if isinstance(v, dict) else v for k, v in zd.items()}
    else:
        zones = {}

    return ExtractedLabel(
        metadata=md,
        product_name=_ef(e.get("product_name")),
        product_dimensions_metric=_ef(e.get("product_dimensions_metric")),
        product_dimensions_imperial=_ef(e.get("product_dimensions_imperial")),
        dimensions=_ef(e.get("dimensions")),
        article_number=_ef(e.get("article_number")),
        article_code=_ef(e.get("article_code")),
        internal_item_number=_ef(e.get("internal_item_number")),
        quantity_multiplier=_ef(e.get("quantity_multiplier")),
        gross_weight=_ef(e.get("gross_weight")),
        net_weight=_ef(e.get("net_weight")),
        weight=_ef(e.get("weight")),
        package_number=_ef(e.get("package_number")),
        package_type=_ef(e.get("package_type")),
        copy_count_inline=_ef(e.get("copy_count_inline")),
        ikea_logo=_ef(e.get("ikea_logo")),
        ean_barcode=_br(e.get("ean_barcode")),
        ean_barcode_value=_ef(e.get("ean_barcode_value")),
        itf14_barcode=_br(e.get("itf14_barcode")),
        itf14_human_readable=_ef(e.get("itf14_human_readable")),
        datamatrix_barcode=_br(dmx_barcode) if dmx_barcode else None,
        datamatrix_content=dc,
        ai_240=_ef(ai_block.get("240") or e.get("ai_240")),
        ai_13=_ef(ai_block.get("13") or e.get("ai_13")),
        ai_11=_ef(ai_block.get("11") or e.get("ai_11")),
        ai_10=_ef(ai_block.get("10") or e.get("ai_10")),
        human_readable_date=_ef(e.get("human_readable_date")),
        date_stamp=_ef(e.get("date_stamp")),
        date_label_present=bool(
            e.get("date_label_present", d.get("date_label_present", False))
        ),
        date_alpha_code=_ef(e.get("date_alpha_code")),
        date_numeric_prefix=_ef(e.get("date_numeric_prefix")),
        custom_identifier=_ef(e.get("custom_identifier")),
        plant_identifier=_ef(e.get("plant_identifier")),
        supplier_name=_ef(e.get("supplier_name")),
        supplier_address=_ef(e.get("supplier_address")),
        patent_info=_ef(e.get("patent_info")),
        address_block=_ef(e.get("address_block")),
        origin_text=_ef(e.get("origin_text")),
        copyright_notice=_ef(e.get("copyright_notice")),
        compliance_marks=_ef(e.get("compliance_marks")),
        age_rating=_ef(e.get("age_rating")),
        copy_count=_ef(e.get("copy_count")),
        zones=zones,
        all_word_boxes=raw_w,
        raw_elements=raw_els,
    )
