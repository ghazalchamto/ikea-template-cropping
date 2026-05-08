"""
Two-layer label dissector.

Converts raw extraction outputs into a structured dict with two independent
layers — one for data validation, one for layout validation.

Data layer
~~~~~~~~~~
  text_elements  — every text span on the label: content, font, color, bbox in mm
  barcodes       — decoded barcodes with GS1 AI fields
  named_fields   — every ExtractedLabel field (value, confidence, zone, source)

Layout layer
~~~~~~~~~~~~
  zones          — each spatial zone with element inventory + bounds in mm
  color_regions  — every filled rect from Pass 3 with elements contained inside it
  typography     — unique (font, size_mm, bold, italic, color) combinations used
  dividers       — detected horizontal/vertical separator lines
  images         — embedded raster objects (logos, icons) with mm dimensions

Usage
~~~~~
    dissector = LabelDissector(dpi=600, zone_cfg=zone_cfg, page_meta=page_meta)
    result = dissector.run(raw_elements, extracted_label, barcodes)
    # result is a plain dict — pass to save_dissection() or use directly
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any

logger = logging.getLogger(__name__)

# IKEA label colours that have semantic meaning — used to label color_regions
_KNOWN_COLORS: dict[str, str] = {
    "#FFD100": "ikea_yellow",
    "#FFCC00": "ikea_yellow",
    "#000000": "black",
    "#FFFFFF": "white",
    "#1A1A1A": "near_black",
    "#F5F5F5": "near_white",
}

# Tolerance (px) for containment check — element centre must be within the
# padded rect to count as "inside" the colour region
_CONTAINMENT_PAD_PX = 5


class LabelDissector:
    """
    Organises raw extraction outputs into the two-layer dissection structure.

    All pixel values are accompanied by mm equivalents so downstream validators
    can compare directly against spec values (which are expressed in mm).
    """

    def __init__(self, dpi: int, zone_cfg: list, page_meta: dict):
        self.dpi = dpi
        self.zone_cfg = zone_cfg        # list of (zone_id, frac_start, frac_end, rotation)
        self.page_meta = page_meta
        self.px_to_mm = 25.4 / dpi

    # ── Public entry point ────────────────────────────────────────────────────

    def run(
        self,
        raw_elements: list,
        extracted_label: Any,           # ExtractedLabel
        barcodes: dict,
    ) -> dict:
        """
        Build and return the full dissection dict.

        Parameters
        ----------
        raw_elements:     All RawElement objects from run_pdf_passes().
        extracted_label:  Fully populated ExtractedLabel from extract_fields().
        barcodes:         Dict from decode_barcodes(): {itf14, datamatrix, ean13}.
        """
        img_w = self.page_meta.get("image_width_px", 1)
        img_h = self.page_meta.get("image_height_px", 1)

        text_pdf  = [e for e in raw_elements if e.type == "text_pdf"]
        text_ocr  = [e for e in raw_elements if e.type == "text_ocr"]
        path_rects = [e for e in raw_elements if e.type == "path_rect"]
        path_lines = [e for e in raw_elements if e.type == "path_line"]
        images    = [e for e in raw_elements if e.type == "image_pdf"]

        # OCR-only words supplement: all_word_boxes covers words that OCR found
        # but that have no corresponding text_pdf element.
        ocr_word_boxes = getattr(extracted_label, "all_word_boxes", []) or []

        result = {
            "meta": {
                **self.page_meta,
                "dpi": self.dpi,
                "px_to_mm": round(self.px_to_mm, 6),
            },
            "data_layer": self._build_data_layer(
                text_pdf, text_ocr, ocr_word_boxes, barcodes, extracted_label
            ),
            "layout_layer": self._build_layout_layer(
                text_pdf, text_ocr, ocr_word_boxes,
                path_rects, path_lines, images,
                img_w, img_h,
            ),
            "raw_element_counts": self._count_by_type(raw_elements),
        }

        _log_summary(result)
        return result

    # ── Data layer ────────────────────────────────────────────────────────────

    def _build_data_layer(
        self,
        text_pdf, text_ocr, ocr_word_boxes,
        barcodes, extracted_label,
    ) -> dict:
        return {
            "text_elements":  self._collect_text_elements(text_pdf, text_ocr, ocr_word_boxes),
            "barcodes":       self._collect_barcodes(barcodes, extracted_label),
            "named_fields":   self._collect_named_fields(extracted_label),
        }

    def _collect_text_elements(self, text_pdf, text_ocr, ocr_word_boxes) -> list:
        """
        Build a unified text inventory from two sources:
          1. text_pdf  — Pass 1 PDF spans: exact positions, font name, size, colour.
          2. ocr_word_boxes — OCR-derived words (covers labels with no embedded text,
             or additional words that the PDF missed).

        PDF elements take precedence; OCR words that overlap a PDF element are skipped
        to avoid duplicates.
        """
        result = []

        # PDF text spans — richest source
        for e in text_pdf:
            result.append(self._text_element_from_raw(e, "pdf"))

        # OCR words — only those not already covered by a PDF span
        pdf_centres = [(e.x + e.w / 2, e.y + e.h / 2) for e in text_pdf]

        for w in ocr_word_boxes:
            cx = w["x"] + w["w"] / 2
            cy = w["y"] + w["h"] / 2
            if any(
                abs(cx - px) < 20 and abs(cy - py) < 20
                for px, py in pdf_centres
            ):
                continue  # already covered by PDF span
            result.append({
                "content":    w.get("text", ""),
                "source":     "ocr",
                "zone":       w.get("zone", ""),
                "font":       {"name": "", "size_pt": 0.0, "size_mm": 0.0, "bold": False, "italic": False},
                "color":      {"hex": None, "rgb": None},
                "bbox_px":    self._bpx(w["x"], w["y"], w["w"], w["h"]),
                "bbox_mm":    self._bmm(w["x"], w["y"], w["w"], w["h"]),
                "confidence": round(w.get("conf", 0.0), 3),
            })

        return result

    def _text_element_from_raw(self, e, source: str) -> dict:
        size_pt = e.font_size or 0.0
        size_mm = e.meta.get("font_size_mm", round(size_pt * 25.4 / 72, 3)) if e.meta else 0.0
        color_hex = (e.meta or {}).get("color_hex")
        color_rgb = (e.meta or {}).get("color_rgb")
        return {
            "content":    e.content or "",
            "source":     source,
            "zone":       e.zone or "",
            "font": {
                "name":     e.font or "",
                "size_pt":  round(size_pt, 2),
                "size_mm":  round(size_mm, 3),
                "bold":     e.bold,
                "italic":   e.italic,
            },
            "color": {
                "hex": color_hex,
                "rgb": color_rgb,
            },
            "bbox_px":    self._bpx(e.x, e.y, e.w, e.h),
            "bbox_mm":    self._bmm(e.x, e.y, e.w, e.h),
            "confidence": round(e.confidence, 3),
        }

    def _collect_barcodes(self, barcodes: dict, extracted_label) -> list:
        """
        Collect all decoded barcodes.  For DataMatrix, include the parsed GS1 AI fields.
        """
        result = []

        for name, bc in barcodes.items():
            if bc is None:
                continue
            entry: dict = {
                "symbology":  bc.symbology,
                "value":      bc.raw_data,
                "confidence": bc.confidence,
            }
            if bc.bbox:
                entry["bbox_px"] = self._bpx(bc.bbox.x, bc.bbox.y, bc.bbox.w, bc.bbox.h)
                entry["bbox_mm"] = self._bmm(bc.bbox.x, bc.bbox.y, bc.bbox.w, bc.bbox.h)
                entry["width_mm"]  = round(bc.bbox.w * self.px_to_mm, 2)
                entry["height_mm"] = round(bc.bbox.h * self.px_to_mm, 2)

            # DataMatrix: attach parsed GS1 AI fields
            if bc.symbology == "DataMatrix" and extracted_label.datamatrix_content:
                dm = extracted_label.datamatrix_content
                entry["encoding"] = dm.encoding
                entry["gs1_fields"] = {
                    ai_code: {
                        "value":     af.value,
                        "formatted": af.formatted,
                        "name":      af.name,
                    }
                    for ai_code, af in dm.ai_fields.items()
                    if af.present
                }

            result.append(entry)

        return result

    def _collect_named_fields(self, extracted_label) -> dict:
        """
        Compact summary of every ExtractedLabel field: value, confidence, zone, source.
        Skips None fields and the raw/zone/word-box bulk data.
        """
        skip_keys = {"datamatrix_block", "zones", "raw_words", "raw_elements"}
        d = extracted_label.to_dict()
        fields: dict = {}

        for field_id, val in d.get("elements", {}).items():
            if field_id in skip_keys or val is None:
                continue
            if isinstance(val, dict) and "value" in val:
                fields[field_id] = {
                    "value":      val.get("value"),
                    "confidence": val.get("confidence"),
                    "zone":       val.get("zone"),
                    "source":     val.get("source"),
                    "bbox_px":    val.get("bbox"),
                }
            elif isinstance(val, dict):
                # BarcodeResult or nested block
                fields[field_id] = {"value": val.get("raw_data") or val.get("decoded_text")}
            else:
                fields[field_id] = {"value": val}

        return fields

    # ── Layout layer ──────────────────────────────────────────────────────────

    def _build_layout_layer(
        self,
        text_pdf, text_ocr, ocr_word_boxes,
        path_rects, path_lines, images,
        img_w, img_h,
    ) -> dict:
        all_text = text_pdf + text_ocr
        return {
            "zones":         self._build_zones(text_pdf, text_ocr, ocr_word_boxes,
                                                images, path_rects, img_w, img_h),
            "color_regions": self._build_color_regions(path_rects, all_text, images),
            "typography":    self._build_typography(text_pdf),
            "dividers":      self._build_dividers(path_lines, path_rects),
            "images":        self._build_images(images),
        }

    def _build_zones(
        self,
        text_pdf, text_ocr, ocr_word_boxes,
        images, path_rects,
        img_w, img_h,
    ) -> dict:
        """
        For each configured zone:
        - Compute exact pixel / mm bounds from zone fractions × image size.
        - Count elements in each category.
        - List the text content for quick inspection.
        - Report the background color if a path_rect fills the zone.
        """
        zones: dict = {}
        all_text = text_pdf + text_ocr

        # Map zone_id → list of OCR word texts for quick lookup
        ocr_by_zone: dict[str, list[str]] = {}
        for w in ocr_word_boxes:
            z = w.get("zone", "")
            ocr_by_zone.setdefault(z, []).append(w.get("text", ""))

        is_portrait = self.page_meta.get("image_width_px", 1) < self.page_meta.get("image_height_px", 1)

        for zone_id, f0, f1, rotation in self.zone_cfg:
            if is_portrait:
                zx, zy = 0, int(f0 * img_h)
                zw, zh = img_w, int((f1 - f0) * img_h)
            else:
                zx, zy = int(f0 * img_w), 0
                zw, zh = int((f1 - f0) * img_w), img_h

            zone_text   = [e for e in all_text  if e.zone == zone_id]
            zone_images = [e for e in images     if e.zone == zone_id]
            zone_paths  = [e for e in path_rects if e.zone == zone_id]

            # Background colour: the largest path_rect whose centre sits in the zone
            bg_color = None
            if zone_paths:
                largest = max(zone_paths, key=lambda e: e.w * e.h)
                bg_color = largest.meta.get("fill_hex") if largest.meta else None

            zones[zone_id] = {
                "zone_id":          zone_id,
                "rotation":         rotation,
                "bounds_px":        self._bpx(zx, zy, zw, zh),
                "bounds_mm":        self._bmm(zx, zy, zw, zh),
                "background_color": bg_color,
                "text_count":       len(zone_text),
                "image_count":      len(zone_images),
                "path_count":       len(zone_paths),
                # Quick-read content for debugging / validation
                "text_content":     [e.content for e in zone_text if e.content],
                "ocr_words":        ocr_by_zone.get(zone_id, []),
            }

        return zones

    def _build_color_regions(self, path_rects, all_text, images) -> list:
        """
        Each filled rectangle from Pass 3 becomes a colour region entry.

        Containment analysis: for each region, find text and image elements
        whose centre point falls inside the rectangle.  This answers questions
        like "what text is on the IKEA yellow band?" or "is any text inside the
        black barcode area?".
        """
        regions = []

        for pr in path_rects:
            fill_hex = (pr.meta or {}).get("fill_hex")
            if not fill_hex:
                continue

            rx, ry, rw, rh = pr.x, pr.y, pr.w, pr.h
            pad = _CONTAINMENT_PAD_PX

            text_inside  = []
            image_inside = []

            for e in all_text:
                cx = e.x + e.w / 2
                cy = e.y + e.h / 2
                if (rx - pad <= cx <= rx + rw + pad and
                        ry - pad <= cy <= ry + rh + pad):
                    text_inside.append(e.content or "")

            for e in images:
                cx = e.x + e.w / 2
                cy = e.y + e.h / 2
                if (rx - pad <= cx <= rx + rw + pad and
                        ry - pad <= cy <= ry + rh + pad):
                    image_inside.append("[image]")

            w_mm = (pr.meta or {}).get("width_mm",  round(rw * self.px_to_mm, 2))
            h_mm = (pr.meta or {}).get("height_mm", round(rh * self.px_to_mm, 2))

            regions.append({
                "fill_hex":        fill_hex,
                "fill_rgb":        (pr.meta or {}).get("fill_color"),
                "color_label":     _KNOWN_COLORS.get(fill_hex, "unknown"),
                "zone":            pr.zone or "",
                "bbox_px":         self._bpx(rx, ry, rw, rh),
                "bbox_mm":         self._bmm(rx, ry, rw, rh),
                "width_mm":        w_mm,
                "height_mm":       h_mm,
                "area_mm2":        round(w_mm * h_mm, 2),
                "text_inside":     text_inside,
                "image_inside":    image_inside,
            })

        # Largest area first — the dominant color zones come first
        regions.sort(key=lambda r: r["area_mm2"], reverse=True)
        return regions

    def _build_typography(self, text_pdf) -> list:
        """
        Inventory of every unique (font, size_mm, bold, italic, color) combination.

        Used by layout validators to check:
        - Approved font families are used
        - Font sizes match the spec (in mm)
        - Text colours are correct per zone
        """
        buckets: dict[tuple, list] = {}

        for e in text_pdf:
            size_pt = e.font_size or 0.0
            size_mm = (e.meta or {}).get("font_size_mm", round(size_pt * 25.4 / 72, 3))
            # Bucket to 0.1 mm to tolerate floating-point PDF size encoding noise
            size_bucket = round(size_mm * 10) / 10
            color_hex = (e.meta or {}).get("color_hex") or "#000000"
            key = (e.font or "", size_bucket, e.bold, e.italic, color_hex)
            buckets.setdefault(key, []).append({
                "content": e.content or "",
                "zone":    e.zone or "",
                "x_mm":    round(e.x * self.px_to_mm, 2),
                "y_mm":    round(e.y * self.px_to_mm, 2),
            })

        result = []
        for (font, size_mm, bold, italic, color_hex), occurrences in sorted(
            buckets.items(), key=lambda x: -len(x[1])
        ):
            result.append({
                "font_name":  font,
                "size_mm":    size_mm,
                "bold":       bold,
                "italic":     italic,
                "color_hex":  color_hex,
                "count":      len(occurrences),
                "occurrences": occurrences,
            })

        return result

    def _build_dividers(self, path_lines, path_rects) -> list:
        """
        Detect zone-separator lines from:
        1. path_line elements (explicit thin strokes)
        2. path_rect elements that are very thin in one dimension (< 0.5 mm)
        """
        dividers = []

        for e in path_lines:
            is_h = e.w > e.h
            orientation = "horizontal" if is_h else "vertical"
            dividers.append({
                "orientation":   orientation,
                "position_px":   e.y if is_h else e.x,
                "position_mm":   round((e.y if is_h else e.x) * self.px_to_mm, 2),
                "length_mm":     round((e.w if is_h else e.h) * self.px_to_mm, 2),
                "color_hex":     (e.meta or {}).get("stroke_hex") or (e.meta or {}).get("fill_hex"),
                "bbox_px":       self._bpx(e.x, e.y, e.w, e.h),
            })

        for e in path_rects:
            w_mm = (e.meta or {}).get("width_mm",  round(e.w * self.px_to_mm, 2))
            h_mm = (e.meta or {}).get("height_mm", round(e.h * self.px_to_mm, 2))
            if h_mm > 5.0 and w_mm < 0.5:          # thin vertical rule
                dividers.append({
                    "orientation":  "vertical",
                    "position_px":  e.x,
                    "position_mm":  round(e.x * self.px_to_mm, 2),
                    "length_mm":    h_mm,
                    "color_hex":    (e.meta or {}).get("fill_hex"),
                    "bbox_px":      self._bpx(e.x, e.y, e.w, e.h),
                })
            elif w_mm > 5.0 and h_mm < 0.5:        # thin horizontal rule
                dividers.append({
                    "orientation":  "horizontal",
                    "position_px":  e.y,
                    "position_mm":  round(e.y * self.px_to_mm, 2),
                    "length_mm":    w_mm,
                    "color_hex":    (e.meta or {}).get("fill_hex"),
                    "bbox_px":      self._bpx(e.x, e.y, e.w, e.h),
                })

        # Sort by position for easy reading
        dividers.sort(key=lambda d: d["position_mm"])
        return dividers

    def _build_images(self, images) -> list:
        """
        Inventory of all embedded raster objects.  The aspect ratio helps
        identify the IKEA logo (≈ 2.97:1 width:height) without needing
        colour analysis.
        """
        result = []
        for e in images:
            w_mm = round(e.w * self.px_to_mm, 2)
            h_mm = round(e.h * self.px_to_mm, 2)
            result.append({
                "zone":         e.zone or "",
                "bbox_px":      self._bpx(e.x, e.y, e.w, e.h),
                "bbox_mm":      self._bmm(e.x, e.y, e.w, e.h),
                "width_mm":     w_mm,
                "height_mm":    h_mm,
                "aspect_ratio": round(w_mm / h_mm, 3) if h_mm > 0 else None,
                "meta": {
                    k: v for k, v in (e.meta or {}).items()
                    if k not in ("image_bytes",)
                },
            })
        return result

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _bpx(self, x, y, w, h) -> dict:
        return {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}

    def _bmm(self, x, y, w, h) -> dict:
        m = self.px_to_mm
        return {
            "x": round(x * m, 2),
            "y": round(y * m, 2),
            "w": round(w * m, 2),
            "h": round(h * m, 2),
        }

    @staticmethod
    def _count_by_type(elements) -> dict:
        counts: dict[str, int] = {}
        for e in elements:
            t = getattr(e, "type", "unknown")
            counts[t] = counts.get(t, 0) + 1
        return counts


# ── Logging helper ────────────────────────────────────────────────────────────

def _log_summary(d: dict) -> None:
    dl = d.get("data_layer", {})
    ll = d.get("layout_layer", {})
    logger.info(
        "Dissection complete — "
        "text:%d  barcodes:%d  named_fields:%d  "
        "zones:%d  color_regions:%d  typography:%d  dividers:%d  images:%d",
        len(dl.get("text_elements", [])),
        len(dl.get("barcodes", [])),
        len(dl.get("named_fields", {})),
        len(ll.get("zones", {})),
        len(ll.get("color_regions", [])),
        len(ll.get("typography", [])),
        len(ll.get("dividers", [])),
        len(ll.get("images", [])),
    )
