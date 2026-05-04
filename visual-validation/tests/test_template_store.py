"""Tests for TemplateStore (read-only discovery under ground_truth/templates)."""

from pathlib import Path

import pytest

from src.core.template_store import GroundTruthTemplateCatalog, _version_sort_key


def test_version_sort_key():
    assert _version_sort_key("v2") < _version_sort_key("v10")


def test_list_products_and_resolve_v6():
    root = Path(__file__).resolve().parent.parent
    gt = root / "data/templates/ground-truth-ikea-labels/ground_truth"
    if not (gt / "templates").is_dir():
        pytest.skip("ground_truth templates not present")

    store = GroundTruthTemplateCatalog(gt)
    products = store.list_products()
    assert "1R10-PRADM" in products

    versions = store.list_versions("1R10-PRADM")
    assert "v6" in versions

    r = store.resolve("1R10-PRADM", version="v6")
    assert r.version == "v6"
    assert r.label_pdf.name == "label.pdf"
    assert r.template_json.name == "template.json"
    assert r.label_pdf.is_file()
    assert r.template_json.is_file()


def test_resolve_latest_when_version_omitted():
    root = Path(__file__).resolve().parent.parent
    gt = root / "data/templates/ground-truth-ikea-labels/ground_truth"
    if not (gt / "templates").is_dir():
        pytest.skip("ground_truth templates not present")

    store = GroundTruthTemplateCatalog(gt)
    r = store.resolve("1R10-PRADM", version=None)
    versions = store.list_versions("1R10-PRADM")
    assert r.version == max(versions, key=_version_sort_key)
