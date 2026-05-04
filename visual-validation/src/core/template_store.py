"""
Template discovery and loading under ``ground_truth/templates/``.

* ``TemplateStore`` — legacy API used by ``LabelValidator``: loads a raster
  template image (``label.png`` or ``label.pdf``), resizes to
  ``ValidationConfig`` target size, exposes ``get`` / ``catalog`` / ``__len__``.

* ``GroundTruthTemplateCatalog`` — path-only resolution for the region pipeline:
  ``label.pdf``, ``template.json``, optional ``label.png``, with version pick
  ``vN`` (numeric max when omitted). Never writes under ground_truth.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from src.config.settings import DEFAULT_CONFIG, ValidationConfig
from src.utils.ingestion import load_image

_VERSION_DIR_RE = re.compile(r"^v\d+$", re.IGNORECASE)


def _version_sort_key(v: str) -> int:
    m = re.match(r"^v(\d+)$", v, re.IGNORECASE)
    if not m:
        return -1
    return int(m.group(1))


# ─────────────────────────────────────────────────────────────────────────────
# Region pipeline — PDF + template.json paths
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ResolvedLabelTemplate:
    """Resolved paths for one template variant (region / report tooling)."""

    product_code: str
    version: str
    version_dir: Path
    label_pdf: Path
    template_json: Path
    label_png: Optional[Path]


class GroundTruthTemplateCatalog:
    """
    List products / versions and resolve ``label.pdf`` + ``template.json``.

    *ground_truth_dir* is the ``ground_truth`` root (contains ``templates/``).
    """

    def __init__(self, ground_truth_dir: Union[str, Path]) -> None:
        self.ground_truth_dir = Path(ground_truth_dir).resolve()
        self.templates_root = self.ground_truth_dir / "templates"
        if not self.templates_root.is_dir():
            raise FileNotFoundError(
                f"Expected templates directory at {self.templates_root}. "
                f"Check --ground-truth-dir points at the ground_truth root."
            )

    def list_products(self) -> List[str]:
        names: List[str] = []
        for p in sorted(self.templates_root.iterdir()):
            if p.is_dir() and not p.name.startswith("."):
                names.append(p.name)
        return names

    def list_versions(self, product_code: str) -> List[str]:
        prod = self.templates_root / product_code
        if not prod.is_dir():
            raise FileNotFoundError(
                f"No template product folder {product_code!r} under {self.templates_root}"
            )
        vers = [
            p.name for p in prod.iterdir()
            if p.is_dir() and _VERSION_DIR_RE.match(p.name)
        ]
        if not vers:
            raise FileNotFoundError(
                f"No version folders (v1, v2, …) under {prod}"
            )
        return sorted(vers, key=_version_sort_key)

    def resolve(
        self,
        product_code: str,
        version: Optional[str] = None,
    ) -> ResolvedLabelTemplate:
        versions = self.list_versions(product_code)
        if version is None:
            chosen = max(versions, key=_version_sort_key)
        else:
            vnorm = version.strip()
            if vnorm not in versions:
                raise FileNotFoundError(
                    f"Version {vnorm!r} not found for {product_code!r}. "
                    f"Available: {', '.join(versions)}"
                )
            chosen = vnorm

        vdir = self.templates_root / product_code / chosen
        pdf = vdir / "label.pdf"
        if not pdf.is_file():
            raise FileNotFoundError(f"Missing label.pdf under {vdir}")
        tjson = vdir / "template.json"
        if not tjson.is_file():
            raise FileNotFoundError(f"Missing template.json under {vdir}")
        png = vdir / "label.png"
        png_path = png if png.is_file() else None

        return ResolvedLabelTemplate(
            product_code=product_code,
            version=chosen,
            version_dir=vdir,
            label_pdf=pdf,
            template_json=tjson,
            label_png=png_path,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Legacy — raster template for LabelValidator
# ─────────────────────────────────────────────────────────────────────────────


class TemplateStore:
    """
    Discover ``label.png`` / ``label.pdf`` per product/version and return a
    BGR image resized to ``cfg.target_width`` × ``cfg.target_height``.
    """

    def __init__(
        self,
        ground_truth_dir: Optional[Union[str, Path]] = None,
        cfg: Optional[ValidationConfig] = None,
        **kwargs: Any,
    ) -> None:
        if ground_truth_dir is None:
            ground_truth_dir = kwargs.pop("ground_truth_dir", None)
        if cfg is None:
            cfg = kwargs.pop("cfg", None)
        if kwargs:
            raise TypeError(f"TemplateStore: unexpected keyword arguments {set(kwargs)!r}")
        if ground_truth_dir is None:
            raise TypeError("TemplateStore: ground_truth_dir is required")

        self._cfg = cfg if cfg is not None else DEFAULT_CONFIG
        self.ground_truth_dir = Path(ground_truth_dir).resolve()
        self.templates_root = self.ground_truth_dir / "templates"
        if not self.templates_root.is_dir():
            raise FileNotFoundError(
                f"Expected templates directory at {self.templates_root}."
            )

    def product_codes(self) -> List[str]:
        codes: List[str] = []
        for p in sorted(self.templates_root.iterdir()):
            if not p.is_dir() or p.name.startswith("."):
                continue
            if any(
                (p / vn).is_dir() and _VERSION_DIR_RE.match(vn)
                for vn in (x.name for x in p.iterdir() if x.is_dir())
            ):
                codes.append(p.name)
        return codes

    def catalog(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for product in sorted(self.templates_root.iterdir()):
            if not product.is_dir() or product.name.startswith("."):
                continue
            vers = [
                p.name for p in product.iterdir()
                if p.is_dir() and _VERSION_DIR_RE.match(p.name)
            ]
            for ver in sorted(vers, key=_version_sort_key):
                vdir = product / ver
                png = vdir / "label.png"
                pdf = vdir / "label.pdf"
                if png.is_file():
                    path = png
                elif pdf.is_file():
                    path = pdf
                else:
                    continue
                out.append({
                    "product_code": product.name,
                    "version":      ver,
                    "path":         str(path.resolve()),
                })
        return out

    def __len__(self) -> int:
        return len(self.catalog())

    def get(
        self,
        product_code: str,
        template_version: Optional[str] = None,
    ) -> Tuple[np.ndarray, str]:
        prod = self.templates_root / product_code
        if not prod.is_dir():
            raise KeyError(f"No templates for product_code={product_code!r}")

        versions = [
            p.name for p in prod.iterdir()
            if p.is_dir() and _VERSION_DIR_RE.match(p.name)
        ]
        if not versions:
            raise KeyError(f"No version folders under {prod}")

        if template_version is None:
            ver = max(versions, key=_version_sort_key)
        else:
            tv = template_version.strip()
            if tv not in versions:
                raise KeyError(
                    f"Version {tv!r} not found for {product_code!r}; "
                    f"have {sorted(versions, key=_version_sort_key)}"
                )
            ver = tv

        vdir = prod / ver
        png = vdir / "label.png"
        pdf = vdir / "label.pdf"
        if png.is_file():
            src = png
        elif pdf.is_file():
            src = pdf
        else:
            raise FileNotFoundError(
                f"No label.png or label.pdf under {vdir}"
            )

        img = load_image(src, dpi=self._cfg.render_dpi)
        tw, th = int(self._cfg.target_width), int(self._cfg.target_height)
        if tw > 0 and th > 0 and (img.shape[1] != tw or img.shape[0] != th):
            img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
        return img, ver
