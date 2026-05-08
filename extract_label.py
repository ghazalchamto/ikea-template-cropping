#!/usr/bin/env python3
"""
CLI entry point for the IKEA label extraction component.

Usage
-----
    python extract_label.py <label_file> [--dpi DPI] [--out OUTPUT.json]

Examples
--------
    # Extract from a PDF, print JSON to stdout
    python extract_label.py path/to/label.pdf

    # Write extraction result to a file at 300 DPI
    python extract_label.py path/to/label.pdf --dpi 300 --out result.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from extractor import LabelExtractor
from config import DEFAULT_DPI

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(name)s  %(message)s",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract all elements from an IKEA production label (PDF or image)."
    )
    parser.add_argument(
        "label_file",
        help="Path to the uploaded label file (.pdf, .png, .jpg, .tiff)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help=f"Render DPI for PDF → image conversion (default: {DEFAULT_DPI})",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Write JSON output to this file instead of stdout",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation level (default: 2)",
    )
    parser.add_argument(
        "--no-intermediates",
        action="store_true",
        default=False,
        help="Skip saving dissection.json and extraction.json to output/<stem>/",
    )
    args = parser.parse_args()

    label_path = Path(args.label_file)
    if not label_path.exists():
        print(f"ERROR: file not found: {label_path}", file=sys.stderr)
        sys.exit(1)

    extractor = LabelExtractor(
        label_path,
        dpi=args.dpi,
        save_intermediates=not args.no_intermediates,
    )

    try:
        result = extractor.run()
    except Exception as exc:
        print(f"ERROR during extraction: {exc}", file=sys.stderr)
        logging.exception("Extraction failed")
        sys.exit(2)

    output = json.dumps(result.to_dict(), indent=args.indent, ensure_ascii=False)

    if args.out:
        out_path = Path(args.out)
        out_path.write_text(output, encoding="utf-8")
        print(f"Extraction saved to {out_path}")
    else:
        print(output)

    md = result.metadata
    print(
        f"\n--- Extraction summary ---\n"
        f"  File              : {md.source_file}\n"
        f"  Image size        : {md.image_width_px} × {md.image_height_px} px\n"
        f"  Overall confidence: {md.overall_confidence:.0%}\n"
        f"  Warnings          : {len(md.warnings)}\n"
        f"  Errors            : {len(md.errors)}",
        file=sys.stderr,
    )
    for w in md.warnings:
        print(f"  ⚠  {w}", file=sys.stderr)
    for e in md.errors:
        print(f"  ✗  {e}", file=sys.stderr)

    if not args.no_intermediates:
        from extractor.dissection_store import get_output_dir
        out_dir = get_output_dir(label_path)
        print(f"\n--- Intermediate results ---", file=sys.stderr)
        for fname in ("dissection.json", "extraction.json"):
            p = out_dir / fname
            if p.exists():
                size_kb = p.stat().st_size / 1024
                print(f"  {fname:<20}: {p}  ({size_kb:.0f} KB)", file=sys.stderr)


if __name__ == "__main__":
    main()
