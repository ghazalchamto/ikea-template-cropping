"""
Unit tests for human correction merge and JSONL export.

Run:  python -m pytest tests/test_finetune_store.py -q
   or: python tests/test_finetune_store.py
"""

import json
import unittest

from extractor.models import ExtractionMetadata, ExtractedField, ExtractedLabel
from finetune_store import apply_corrections, build_training_records, snapshot_model_for_field


def _label_empty(w: int = 200, h: int = 200) -> ExtractedLabel:
    md = ExtractionMetadata(
        source_file="t.png",
        extracted_at="",
        image_width_px=w,
        image_height_px=h,
        render_dpi=300,
    )
    return ExtractedLabel(metadata=md)


class TestFinetuneStore(unittest.TestCase):
    def test_apply_creates_text_field_with_bbox(self):
        base = _label_empty()
        cor = {
            "product_name": {
                "value": "BILLY",
                "bbox": {"x": 10, "y": 20, "w": 30, "h": 12},
            }
        }
        out = apply_corrections(base, cor)
        self.assertIsNotNone(out.product_name)
        self.assertEqual(out.product_name.value, "BILLY")
        self.assertEqual(out.product_name.source, "human")
        self.assertIsNotNone(out.product_name.bbox)
        self.assertEqual(out.product_name.bbox.x, 10)

    def test_apply_updates_existing_field(self):
        base = _label_empty()
        base.article_number = ExtractedField(
            value="111.222.33",
            raw_text="111.222.33",
            confidence=0.5,
            zone="identity",
            bbox=None,
            source="ocr",
        )
        out = apply_corrections(
            base,
            {
                "article_number": {
                    "value": "803.804.64",
                    "bbox": {"x": 1, "y": 1, "w": 5, "h": 5},
                }
            },
        )
        self.assertIsNotNone(out.article_number)
        self.assertEqual(out.article_number.value, "803.804.64")
        self.assertEqual(out.article_number.confidence, 1.0)
        self.assertIsNotNone(out.article_number.bbox)

    def test_jsonl_row_contains_model_and_corrected(self):
        base = _label_empty()
        base.product_name = ExtractedField(
            value="WRONG",
            raw_text="WRONG",
            confidence=0.3,
            zone=None,
            bbox=None,
            source="ocr",
        )
        cor = {
            "product_name": {
                "value": "RIGHT",
                "bbox": {"x": 0, "y": 0, "w": 10, "h": 10},
            }
        }
        rows = build_training_records("abc123", 300, base, cor)
        self.assertEqual(len(rows), 1)
        r = rows[0]
        self.assertEqual(r["field_id"], "product_name")
        self.assertEqual(r["corrected_value"], "RIGHT")
        self.assertEqual(r["model_value"], "WRONG")
        self.assertEqual(r["file_hash"], "abc123")
        json.dumps(r)

    def test_snapshot_model_for_field(self):
        base = _label_empty()
        v, bb, is_bc = snapshot_model_for_field(base, "foo_unknown")
        self.assertIsNone(v)
        self.assertIsNone(bb)
        self.assertFalse(is_bc)


if __name__ == "__main__":
    unittest.main()
