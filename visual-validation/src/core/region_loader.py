"""
src/core/region_loader.py
=========================
Load and validate region definitions for a product code.

Config location:  config/regions/<product_code>.json

JSON shape (legacy — absolute canonical pixels)
-----------------------------------------------
{
    "product_code": "L10555-MIADM",
    "page_size":    { "width": 1748, "height": 2480 },
    "label_rect_pts": [0, 0, 412, 31],
    "regions": [
        {
            "name": "header", "type": "text_block",
            "x": 0, "y": 0, "width": 1748, "height": 520,
            "mandatory": true, "weight": 1.0
        }
    ]
}

JSON shape (label-normalised — resolution-independent)
------------------------------------------------------
{
    "product_code": "1R10-PRADM",
    "page_size": { "width": 1326, "height": 100 },
    "regions": [
        {
            "name": "left_identity",
            "type": "free_form_block",
            "bbox_norm": [0.0, 0.0, 0.30, 1.0],
            "mandatory": true
        }
    ]
}

Optional **horizontal bands** (same ``bbox_norm`` frame): add ``subdivide_y_bands``
to slice each ``bbox_norm`` region by row (e.g. header / body / footer). Each
slice becomes ``{{name}}__{{band_id}}`` with the parent's ``type`` / mandatory /
weight preserved. List band ids in ``subdivide_y_band_optional`` to force those
horizontal slices to be optional (``mandatory: false``, ``weight: 0``), e.g. for
sparse footer strips that should not fail the verdict.

```json
"subdivide_y_bands": [
  { "id": "header", "y0": 0, "y1": 0.32 },
  { "id": "body", "y0": 0.32, "y1": 0.72 },
  { "id": "footer", "y0": 0.72, "y1": 1.0 }
],
"regions": [ ... ]
```

``bbox_norm`` is ``[x, y, width, height]`` as fractions of **label_rect**
(top-left and size, each in ``[0, 1]``). The label rectangle defaults to
the full visible ``page.rect`` unless ``label_rect_pts`` is set: either a list
``[x0, y0, x1, y1]`` or an object ``{"x0","y0","x1","y1"}`` in PDF points
(intersected with ``page.rect`` at extraction time).

For ``bbox_norm`` regions, ``width`` and ``height`` in the JSON are optional;
if omitted, output clip dimensions default to ``round(bbox_norm[2]*page_w)``
by ``round(bbox_norm[3]*page_h)``. Legacy regions still require integer
``x``, ``y``, ``width``, ``height``.

Coordinates (legacy) are in pixels relative to ``page_size``. The PDF
extractor maps legacy pixels → PDF points; ``bbox_norm`` regions map
relative to ``label_rect`` inside the PDF.

Defaults
--------
* ``mandatory`` is **true** if missing.
* ``weight``    is 1.0 for mandatory regions, 0.0 for optional regions, if missing.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Region:
    name:      str
    x:         int
    y:         int
    width:     int
    height:    int
    type:      str
    mandatory: bool
    weight:    float
    # If set, region is defined in label-relative normalised coordinates;
    # ``x, y, width, height`` are still the output raster size / canonical
    # anchor used by text introspection (derived from bbox_norm × page_size).
    bbox_norm: Optional[Tuple[float, float, float, float]] = None
    extras:    Dict[str, Any] = field(default_factory=dict)

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "name":      self.name,
            "x":         self.x,
            "y":         self.y,
            "width":     self.width,
            "height":    self.height,
            "type":      self.type,
            "mandatory": self.mandatory,
            "weight":    self.weight,
        }
        if self.bbox_norm is not None:
            d["bbox_norm"] = list(self.bbox_norm)
        if self.extras:
            d["extras"] = self.extras
        return d


@dataclass(frozen=True)
class RegionConfig:
    product_code:   str
    page_width:     int
    page_height:    int
    regions:        List[Region]
    source_path:    Path
    # Optional PDF-space label bounds [x0, y0, x1, y1] in points; intersected
    # with each document's page.rect at extraction time. If None, label_rect
    # defaults to the full visible page rect.
    label_rect_pts: Optional[Tuple[float, float, float, float]] = None

    @property
    def page_size(self) -> Tuple[int, int]:
        return (self.page_width, self.page_height)

    def mandatory(self) -> List[Region]:
        return [r for r in self.regions if r.mandatory]

    def optional(self) -> List[Region]:
        return [r for r in self.regions if not r.mandatory]


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helper (pure; used by extractor + text introspector)
# ─────────────────────────────────────────────────────────────────────────────

def bbox_norm_to_pdf_rect(
    label_rect_pdf: Tuple[float, float, float, float],
    bbox_norm:      Tuple[float, float, float, float],
) -> Tuple[float, float, float, float]:
    """
    Map a normalised bbox (fractions of label_rect) to PDF points.

    ``label_rect_pdf`` is ``(x0, y0, x1, y1)`` in PDF user space.
    ``bbox_norm`` is ``(nx, ny, nw, nh)`` each in ``[0, 1]``.
    Returns ``(x0, y0, x1, y1)`` PDF rect (not clamped to page).
    """
    Lx0, Ly0, Lx1, Ly1 = label_rect_pdf
    nx, ny, nw, nh = bbox_norm
    lw = Lx1 - Lx0
    lh = Ly1 - Ly0
    x0 = Lx0 + nx * lw
    y0 = Ly0 + ny * lh
    x1 = Lx0 + (nx + nw) * lw
    y1 = Ly0 + (ny + nh) * lh
    return (x0, y0, x1, y1)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

class RegionConfigError(ValueError):
    """Raised when a region config file is missing required fields or malformed."""


def default_regions_dir() -> Path:
    """Repo-relative default: <repo_root>/config/regions/."""
    return Path(__file__).resolve().parent.parent.parent / "config" / "regions"


def load_region_config(
    product_code: str,
    regions_dir:  Optional[str | Path] = None,
) -> RegionConfig:
    """
    Load and validate the region config for *product_code*.

    Raises
    ------
    FileNotFoundError      : config file does not exist
    RegionConfigError      : config exists but is malformed / fails validation
    """
    base = Path(regions_dir) if regions_dir else default_regions_dir()
    path = base / f"{product_code}.json"

    if not path.exists():
        raise FileNotFoundError(
            f"No region config found for product_code={product_code!r} at {path}. "
            f"Create the file or pass a different regions_dir."
        )

    return load_region_config_file(path, expected_product_code=product_code)


def load_region_config_file(
    path: str | Path,
    *,
    expected_product_code: Optional[str] = None,
) -> RegionConfig:
    """
    Load a region JSON from an explicit path (hand-written or generated).

    If *expected_product_code* is set, it must match the file's ``product_code``.
    """
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Region config not found: {path}")

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RegionConfigError(f"{path}: invalid JSON — {exc}") from exc

    if not isinstance(raw, dict):
        raise RegionConfigError(f"{path}: top-level value must be an object")

    if "product_code" not in raw:
        raise RegionConfigError(f"{path}: missing required field 'product_code'")
    file_pc = str(raw["product_code"]).strip()
    if expected_product_code is not None and file_pc != expected_product_code:
        raise RegionConfigError(
            f"{path}: declares product_code={file_pc!r} "
            f"but expected {expected_product_code!r}"
        )

    return _region_config_from_parsed(raw, path, product_code=file_pc)


def _coerce_label_rect_pts(raw_lr: Any, path: Path) -> Optional[Tuple[float, float, float, float]]:
    """Parse label_rect_pts as [x0,y0,x1,y1] list or {{x0,y0,x1,y1}} object."""
    if raw_lr is None:
        return None
    if isinstance(raw_lr, dict):
        try:
            a = float(raw_lr["x0"])
            b = float(raw_lr["y0"])
            c = float(raw_lr["x1"])
            d = float(raw_lr["y1"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RegionConfigError(
                f"{path}: label_rect_pts object needs numeric x0,y0,x1,y1 — {exc}"
            ) from exc
    elif isinstance(raw_lr, (list, tuple)) and len(raw_lr) == 4:
        try:
            a, b, c, d = float(raw_lr[0]), float(raw_lr[1]), float(raw_lr[2]), float(raw_lr[3])
        except (TypeError, ValueError) as exc:
            raise RegionConfigError(f"{path}: label_rect_pts values must be numbers — {exc}") from exc
    else:
        raise RegionConfigError(
            f"{path}: label_rect_pts must be a list of 4 numbers or an object with x0,y0,x1,y1"
        )
    if not (a < c and b < d):
        raise RegionConfigError(
            f"{path}: label_rect_pts must satisfy x0<x1 and y0<y1 (got {raw_lr!r})"
        )
    return (a, b, c, d)


def _parse_subdivide_y_bands(raw_bands: Any, path: Path) -> List[Tuple[str, float, float]]:
    """
    Parse ``subdivide_y_bands`` entries: ``{{"id": "header", "y0": 0, "y1": 0.3}}, ...``.

    Each band is a horizontal strip in label-normalised space (same frame as
    ``bbox_norm``): y runs top→bottom in [0, 1].
    """
    if not isinstance(raw_bands, list) or not raw_bands:
        return []
    bands: List[Tuple[str, float, float]] = []
    for i, b in enumerate(raw_bands):
        if not isinstance(b, dict):
            raise RegionConfigError(f"{path}: subdivide_y_bands[{i}] must be an object")
        bid = str(b.get("id", b.get("name", ""))).strip()
        if not bid:
            raise RegionConfigError(f"{path}: subdivide_y_bands[{i}] needs non-empty 'id'")
        try:
            y0 = float(b["y0"])
            y1 = float(b["y1"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RegionConfigError(
                f"{path}: subdivide_y_bands[{i}] needs numeric y0 and y1 — {exc}"
            ) from exc
        if y0 < -1e-9 or y1 > 1.0 + 1e-9 or not (y0 < y1):
            raise RegionConfigError(
                f"{path}: subdivide_y_bands[{i}] must satisfy 0 ≤ y0 < y1 ≤ 1 "
                f"(got y0={y0}, y1={y1})"
            )
        bands.append((bid, max(0.0, y0), min(1.0, y1)))
    return bands


def expand_bbox_norm_regions_with_y_bands(
    regions: List[Any],
    bands: List[Tuple[str, float, float]],
    path: Path,
    *,
    optional_band_ids: Optional[set[str]] = None,
) -> List[dict]:
    """
    Intersect each ``bbox_norm`` region with every horizontal band; emit
    ``{{parent_name}}__{{band_id}}`` regions. Legacy regions (no ``bbox_norm``)
    are copied unchanged.

    If *optional_band_ids* is set, any slice whose ``band_id`` is in the set
    becomes ``mandatory: false`` and ``weight: 0`` (optional for scoring), so
    sparse strips (e.g. footers) do not fail the run.
    """
    opt_ids = optional_band_ids or set()
    if not bands:
        return [dict(r) for r in regions if isinstance(r, dict)]
    out: List[dict] = []
    eps = 0.001
    for i, reg in enumerate(regions):
        if not isinstance(reg, dict):
            raise RegionConfigError(f"{path}: region #{i} must be an object")
        reg = dict(reg)
        if "bbox_norm" not in reg:
            out.append(reg)
            continue
        bn = reg["bbox_norm"]
        if not isinstance(bn, (list, tuple)) or len(bn) != 4:
            raise RegionConfigError(f"{path}: region #{i}: bbox_norm must be a list of 4 numbers")
        try:
            nx = float(bn[0])
            ny = float(bn[1])
            nw = float(bn[2])
            nh = float(bn[3])
        except (TypeError, ValueError) as exc:
            raise RegionConfigError(f"{path}: region #{i}: bbox_norm values must be numbers — {exc}") from exc
        parent = str(reg.get("name", "")).strip()
        if not parent:
            raise RegionConfigError(f"{path}: region #{i}: missing name")
        any_child = False
        for bid, y0, y1 in bands:
            iy0 = max(ny, y0)
            iy1 = min(ny + nh, y1)
            ih = iy1 - iy0
            if ih <= eps:
                continue
            ix0 = max(0.0, nx)
            ix1 = min(1.0, nx + nw)
            iw = ix1 - ix0
            if iw <= eps:
                continue
            child = {k: v for k, v in reg.items() if k != "name"}
            child["name"] = f"{parent}__{bid}"
            child["bbox_norm"] = [
                round(ix0, 4),
                round(iy0, 4),
                round(iw, 4),
                round(ih, 4),
            ]
            if bid in opt_ids:
                child["mandatory"] = False
                child["weight"] = 0.0
            out.append(child)
            any_child = True
        if not any_child:
            raise RegionConfigError(
                f"{path}: subdivide_y_bands produced no slice for region {parent!r} "
                f"(bbox_norm={[nx, ny, nw, nh]})"
            )
    return out


def _expand_regions_raw_for_loading(raw: dict, path: Path) -> List[Any]:
    """Apply optional ``subdivide_y_bands`` before parsing region objects."""
    if "regions" not in raw or not isinstance(raw["regions"], list):
        raise RegionConfigError(f"{path}: missing or malformed 'regions' array")
    if len(raw["regions"]) == 0:
        raise RegionConfigError(f"{path}: 'regions' array is empty")
    bands_raw = raw.get("subdivide_y_bands")
    if bands_raw is None:
        return list(raw["regions"])
    bands = _parse_subdivide_y_bands(bands_raw, path)
    if not bands:
        return list(raw["regions"])
    raw_opt = raw.get("subdivide_y_band_optional")
    opt_ids: set[str] = set()
    if raw_opt is not None:
        if not isinstance(raw_opt, list):
            raise RegionConfigError(f"{path}: subdivide_y_band_optional must be a list of band ids")
        for j, item in enumerate(raw_opt):
            s = str(item).strip()
            if not s:
                raise RegionConfigError(f"{path}: subdivide_y_band_optional[{j}] must be non-empty")
            opt_ids.add(s)
    return expand_bbox_norm_regions_with_y_bands(
        raw["regions"], bands, path, optional_band_ids=opt_ids
    )


def _region_config_from_parsed(raw: dict, path: Path, product_code: str) -> RegionConfig:
    """Shared body: *raw* is a dict, *path* is used for error messages + RegionConfig.source_path."""
    if "page_size" not in raw or not isinstance(raw["page_size"], dict):
        raise RegionConfigError(f"{path}: missing or malformed 'page_size' object")
    try:
        page_w = int(raw["page_size"]["width"])
        page_h = int(raw["page_size"]["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RegionConfigError(
            f"{path}: page_size must contain integer 'width' and 'height' — {exc}"
        ) from exc
    if page_w <= 0 or page_h <= 0:
        raise RegionConfigError(f"{path}: page_size dimensions must be positive")

    label_rect_pts: Optional[Tuple[float, float, float, float]] = None
    if "label_rect_pts" in raw and raw["label_rect_pts"] is not None:
        label_rect_pts = _coerce_label_rect_pts(raw["label_rect_pts"], path)

    regions_raw = _expand_regions_raw_for_loading(raw, path)

    regions: List[Region] = []
    seen_names: set[str] = set()

    for i, entry in enumerate(regions_raw):
        try:
            region = _parse_region(entry, page_w, page_h)
        except RegionConfigError as exc:
            raise RegionConfigError(f"{path}: region #{i}: {exc}") from None

        if region.name in seen_names:
            raise RegionConfigError(f"{path}: duplicate region name {region.name!r}")
        seen_names.add(region.name)
        regions.append(region)
        _validate_region_json_extras(path, i, region)

    if not any(r.mandatory for r in regions):
        raise RegionConfigError(
            f"{path}: at least one region must be mandatory (mandatory=true)"
        )

    config = RegionConfig(
        product_code   = product_code,
        page_width    = page_w,
        page_height   = page_h,
        regions       = regions,
        source_path   = path,
        label_rect_pts = label_rect_pts,
    )
    logger.info(
        "Loaded region config for %s — %d regions (%d mandatory, %d optional) from %s",
        product_code, len(regions),
        len(config.mandatory()), len(config.optional()), path,
    )
    return config


# ─────────────────────────────────────────────────────────────────────────────
# Internal
# ─────────────────────────────────────────────────────────────────────────────

_REQUIRED_LEGACY = ("name", "x", "y", "width", "height", "type")
_REQUIRED_NORM   = ("name", "type")


def _validate_region_json_extras(path: Path, index: int, region: Region) -> None:
    """
    Fail fast on non-numeric or out-of-range optional numeric fields stored in
    ``extras`` (JSON keys not consumed into the Region dataclass).
    """
    e = region.extras
    name = region.name

    def _float_key(key: str, *, positive: bool = False) -> None:
        if key not in e:
            return
        raw = e[key]
        try:
            v = float(raw)
        except (TypeError, ValueError) as exc:
            raise RegionConfigError(
                f"{path}: region #{index} ({name!r}): {key!r} must be a number — {exc}",
            ) from exc
        if positive and v <= 0:
            raise RegionConfigError(
                f"{path}: region #{index} ({name!r}): {key!r} must be positive (got {v})",
            )

    _float_key("tolerance_mm", positive=True)
    _float_key("font_size_tol_pt")
    _float_key("overflow_epsilon_pt")
    _float_key("logo_edge_min")
    _float_key("image_edge_min")
    _float_key("vector_raster_edge_min")
    _float_key("barcode_dark_thresh")
    _float_key("logo_dark_thresh")
    _float_key("image_dark_thresh")

    if "line_count_tolerance" in e:
        raw = e["line_count_tolerance"]
        if isinstance(raw, bool):
            raise RegionConfigError(
                f"{path}: region #{index} ({name!r}): line_count_tolerance must be an integer, not boolean",
            )
        try:
            iv = int(raw)
        except (TypeError, ValueError) as exc:
            raise RegionConfigError(
                f"{path}: region #{index} ({name!r}): line_count_tolerance must be an integer — {exc}",
            ) from exc
        if iv < 0:
            raise RegionConfigError(
                f"{path}: region #{index} ({name!r}): line_count_tolerance must be ≥ 0 (got {iv})",
            )

    if "threshold" in e:
        raw = e["threshold"]
        try:
            tv = float(raw)
        except (TypeError, ValueError) as exc:
            raise RegionConfigError(
                f"{path}: region #{index} ({name!r}): threshold must be a number — {exc}",
            ) from exc
        if not (0.0 <= tv <= 1.0):
            raise RegionConfigError(
                f"{path}: region #{index} ({name!r}): threshold must be in [0, 1] (got {tv})",
            )


def _parse_bbox_norm(raw: Any) -> Tuple[float, float, float, float]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        raise RegionConfigError("bbox_norm must be a list of 4 numbers [x,y,width,height]")
    try:
        nx, ny, nw, nh = float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3])
    except (TypeError, ValueError) as exc:
        raise RegionConfigError(f"bbox_norm entries must be numbers — {exc}") from exc
    for i, v in enumerate((nx, ny, nw, nh)):
        if v < 0.0 - 1e-9 or v > 1.0 + 1e-9:
            raise RegionConfigError(
                f"bbox_norm[{i}]={v} out of range — all values must be in [0, 1]"
            )
    if nx + nw > 1.0 + 1e-9:
        raise RegionConfigError(f"bbox_norm: x + width must be ≤ 1 (got {nx}+{nw})")
    if ny + nh > 1.0 + 1e-9:
        raise RegionConfigError(f"bbox_norm: y + height must be ≤ 1 (got {ny}+{nh})")
    return (nx, ny, nw, nh)


def _parse_region(entry: Any, page_w: int, page_h: int) -> Region:
    if not isinstance(entry, dict):
        raise RegionConfigError("region entry must be an object")

    if "bbox_norm" in entry and entry["bbox_norm"] is not None:
        return _parse_region_bbox_norm(entry, page_w, page_h)

    for f in _REQUIRED_LEGACY:
        if f not in entry:
            raise RegionConfigError(f"missing required field {f!r}")

    name = str(entry["name"]).strip()
    if not name:
        raise RegionConfigError("name is empty")

    try:
        x = int(entry["x"])
        y = int(entry["y"])
        w = int(entry["width"])
        h = int(entry["height"])
    except (TypeError, ValueError) as exc:
        raise RegionConfigError(f"x/y/width/height must be integers — {exc}") from exc

    if w <= 0 or h <= 0:
        raise RegionConfigError(f"width and height must be positive (got {w}x{h})")
    if x < 0 or y < 0:
        raise RegionConfigError(f"x and y must be non-negative (got {x},{y})")
    if x + w > page_w or y + h > page_h:
        raise RegionConfigError(
            f"region {name!r} bbox ({x},{y},{w},{h}) exceeds page_size {page_w}x{page_h}"
        )

    rtype = str(entry["type"]).strip()
    if not rtype:
        raise RegionConfigError("type is empty")

    mandatory, weight = _parse_mandatory_weight(name, entry)

    extras = {
        k: v for k, v in entry.items()
        if k not in {*_REQUIRED_LEGACY, "mandatory", "weight", "bbox_norm"}
    }

    return Region(
        name=name, x=x, y=y, width=w, height=h, type=rtype,
        mandatory=mandatory, weight=weight, bbox_norm=None, extras=extras,
    )


def _parse_region_bbox_norm(entry: Dict[str, Any], page_w: int, page_h: int) -> Region:
    for f in _REQUIRED_NORM:
        if f not in entry:
            raise RegionConfigError(f"missing required field {f!r} (bbox_norm mode)")

    name = str(entry["name"]).strip()
    if not name:
        raise RegionConfigError("name is empty")

    bbox_norm = _parse_bbox_norm(entry["bbox_norm"])

    rtype = str(entry["type"]).strip()
    if not rtype:
        raise RegionConfigError("type is empty")

    mandatory, weight = _parse_mandatory_weight(name, entry)

    nx, ny, nw, nh = bbox_norm
    out_w = entry.get("width")
    out_h = entry.get("height")
    if out_w is None and out_h is None:
        w = max(1, int(round(nw * page_w)))
        h = max(1, int(round(nh * page_h)))
    elif out_w is not None and out_h is not None:
        try:
            w = int(out_w)
            h = int(out_h)
        except (TypeError, ValueError) as exc:
            raise RegionConfigError(f"width/height must be integers — {exc}") from exc
        if w <= 0 or h <= 0:
            raise RegionConfigError(f"width and height must be positive (got {w}x{h})")
    else:
        raise RegionConfigError(
            "bbox_norm regions: specify both width and height, or omit both "
            "(they will be derived from bbox_norm × page_size)"
        )

    x = max(0, min(page_w - 1, int(round(nx * page_w))))
    y = max(0, min(page_h - 1, int(round(ny * page_h))))

    extras = {
        k: v for k, v in entry.items()
        if k not in {
            "name", "type", "bbox_norm", "mandatory", "weight",
            "x", "y", "width", "height",
        }
    }

    return Region(
        name=name,
        x=x,
        y=y,
        width=w,
        height=h,
        type=rtype,
        mandatory=mandatory,
        weight=weight,
        bbox_norm=bbox_norm,
        extras=extras,
    )


def _parse_mandatory_weight(name: str, entry: Dict[str, Any]) -> Tuple[bool, float]:
    if "mandatory" in entry:
        mandatory_val = entry["mandatory"]
        if not isinstance(mandatory_val, bool):
            raise RegionConfigError(
                f"mandatory must be a boolean, got {type(mandatory_val).__name__}"
            )
        mandatory = mandatory_val
    else:
        mandatory = True

    if "weight" in entry:
        try:
            weight = float(entry["weight"])
        except (TypeError, ValueError) as exc:
            raise RegionConfigError(f"weight must be a number — {exc}") from exc
        if weight < 0:
            raise RegionConfigError(f"weight must be non-negative (got {weight})")
        if not mandatory and weight > 0:
            logger.warning(
                "Region %r is optional but has weight=%s; weight will be ignored "
                "by the aggregator.", name, weight,
            )
    else:
        weight = 1.0 if mandatory else 0.0

    return mandatory, weight
