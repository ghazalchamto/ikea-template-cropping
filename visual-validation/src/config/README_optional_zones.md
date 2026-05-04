# config/optional_zones/

This directory holds per-product optional-zone configuration files.

## Purpose

Some label fields are legitimately optional — they may be empty on valid labels
for certain SKU variants. Without optional-zone config the validator would mark
those labels INVALID because the empty field looks like a difference vs the base
template. These JSON files tell the system to ignore specific rectangular regions
when computing all comparison signals (SSIM, pixel diff, edge diff, tile grid).

## File naming

One file per product code, named exactly after the code:

    1C50-CAA.json
    2B30-XAA.json

## JSON format

```json
{
    "product_code": "1C50-CAA",
    "optional_zones": [
        {
            "name": "sustainability_icon",
            "x": 1200,
            "y": 2100,
            "width": 250,
            "height": 180,
            "reason": "Icon printed only on recycled-packaging variant"
        },
        {
            "name": "country_of_origin",
            "x": 80,
            "y": 2300,
            "width": 400,
            "height": 60,
            "reason": "Field filled only for non-EU shipments"
        }
    ]
}
```

## Coordinate system

All coordinates (x, y, width, height) are in **pixels** relative to the
canonical preprocessed image size defined in `src/config/settings.py`:

    target_width  = 1748  (default)
    target_height = 2480  (default)

Every label image is resized to this resolution before comparison, so the
coordinates here always refer to the same physical position regardless of
whether the input label is a 150 DPI or 300 DPI PDF.

## How it works

1. `src/core/optional_zones.py` loads the JSON at validation time.
2. A boolean ignore-mask (H×W) is built from all zones.
3. The mask is passed into `compare()` where it is applied before every signal:
   - SSIM: optional zones filled with neutral grey in both images → zero diff
   - Pixel diff: optional zones zeroed out before morphological closing
   - Edge diff: optional zones zeroed out from the XOR edge mask
   - Tile grid: each tile only counts mandatory pixels toward its diff ratio
4. Optional zones are drawn as **teal dashed outlines** in the annotated image.
5. The HTML report lists all ignored zones in a dedicated section.

## What NOT to do

- Do not add zones just to hide real defects.
- Do not lower global thresholds instead — add a zone for the specific region.
- Do not edit anything inside `ground_truth/`.
