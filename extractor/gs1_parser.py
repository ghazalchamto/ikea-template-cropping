"""
GS1 Application Identifier (AI) parser.

Handles two input formats:
  1. Human-readable parentheses form: "(240)0299999710973(13)260402"
  2. GS1 compact form with GS (ASCII 0x1D) separators (raw DataMatrix decode)

Output is a dict keyed by AI code string → AIField.
"""

from __future__ import annotations
import re
from .models import AIField

# ── AI specification table ────────────────────────────────────────────────────
# (name, fixed_length_or_None, description)
_AI_SPECS: dict[str, tuple[str, int | None, str]] = {
    "00":  ("sscc",          18,   "Serial Shipping Container Code"),
    "01":  ("gtin",          14,   "Global Trade Item Number"),
    "10":  ("batch_lot",     None, "Batch / Lot Number"),
    "11":  ("prod_date",     6,    "Production Date (YYMMDD)"),
    "13":  ("pack_date",     6,    "Packaging Date (YYMMDD)"),
    "17":  ("exp_date",      6,    "Expiration Date (YYMMDD)"),
    "21":  ("serial_no",     None, "Serial Number"),
    "30":  ("var_count",     None, "Variable Count"),
    "37":  ("qty_pieces",    None, "Number of Units Contained"),
    "240": ("additional_id", None, "Additional Item Identification"),
    "241": ("customer_pn",   None, "Customer Part Number"),
    "310": ("net_wt_kg",     6,    "Net Weight, kg"),
    "320": ("net_wt_lb",     6,    "Net Weight, lbs"),
    "91":  ("internal_1",   None, "Company Internal #1"),
    "92":  ("internal_2",   None, "Company Internal #2"),
    "93":  ("internal_3",   None, "Company Internal #3"),
    "99":  ("internal_9",   None, "Company Internal #9"),
}

# Regex for human-readable parenthesised form: (AI)VALUE(AI)VALUE...
_PAREN_RE = re.compile(r'\((\d{2,4})\)([^(]+)')


def _format_date(value: str) -> str | None:
    """Convert YYMMDD to YY-MM-DD, return None if not parseable."""
    if len(value) == 6 and value.isdigit():
        return f"{value[:2]}-{value[2:4]}-{value[4:6]}"
    return None


def _clean_value(ai_code: str, raw_value: str) -> str:
    """
    Sanitise OCR-sourced AI values.

    Numeric-only AIs (fixed-length date codes, AI 240, AI 01 etc.) should
    contain only digits.  Strip everything from the first non-digit character
    when the spec calls for a numeric value.
    """
    raw = raw_value.strip()
    spec = _AI_SPECS.get(ai_code)
    if spec is None:
        return raw

    name, fixed_len, _ = spec
    # Fields that are pure numeric by GS1 definition
    pure_numeric = {
        "00", "01", "11", "13", "17", "30", "37", "240",
        "310", "320",
    }
    if ai_code in pure_numeric:
        # Keep only leading digits
        digits = re.match(r"^(\d+)", raw)
        raw = digits.group(1) if digits else raw

    # Trim to fixed length if applicable
    if fixed_len:
        raw = raw[:fixed_len]

    return raw


def _build_ai_field(ai_code: str, raw_value: str) -> AIField:
    spec = _AI_SPECS.get(ai_code)
    clean = _clean_value(ai_code, raw_value)
    if spec:
        name, fixed_len, description = spec
        value = clean
        formatted = _format_date(value) if name.endswith("_date") else None
    else:
        name = f"ai_{ai_code}"
        description = f"Unknown AI {ai_code}"
        value = clean
        formatted = None

    return AIField(
        ai_code=ai_code,
        name=name,
        description=description,
        value=value,
        formatted=formatted,
        present=bool(value),
    )


def parse_parentheses_form(text: str) -> dict[str, AIField]:
    """
    Parse the human-readable "(AI)VALUE" string printed on IKEA labels.

    Example input:  "AI(240)0299999710973 AI(13)260402"
                or  "(240)0299999710973(13)260402"
    """
    # Normalise: strip leading "AI" prefix if printed that way
    normalised = text.replace("AI(", "(").replace("AI (", "(")
    result: dict[str, AIField] = {}
    for match in _PAREN_RE.finditer(normalised):
        ai_code = match.group(1)
        raw_value = match.group(2)
        result[ai_code] = _build_ai_field(ai_code, raw_value)
    return result


def parse_gs1_compact(raw: bytes) -> dict[str, AIField]:
    """
    Parse the compact GS1 binary format used inside DataMatrix symbols.

    FNC1/AIM prefix bytes (like ]d2) are stripped automatically.
    Variable-length AIs are delimited by GS (0x1D).
    """
    GS = "\x1d"
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1", errors="replace")

    # Strip standard GS1 DataMatrix AIM prefix
    for prefix in ("]d2", "]d1", "]d0", "]Q3"):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break

    result: dict[str, AIField] = {}
    parts = text.split(GS)

    for part in parts:
        if not part:
            continue
        # Try longest AI code first (4 → 3 → 2 digits)
        matched = False
        for ai_len in (4, 3, 2):
            candidate = part[:ai_len]
            if candidate in _AI_SPECS:
                spec = _AI_SPECS[candidate]
                fixed_len = spec[1]
                raw_value = part[ai_len: ai_len + fixed_len] if fixed_len else part[ai_len:]
                result[candidate] = _build_ai_field(candidate, raw_value)
                matched = True
                break
        if not matched:
            # Try to salvage by checking 2-digit prefix anyway
            if len(part) >= 2 and part[:2].isdigit():
                ai_code = part[:2]
                result[ai_code] = _build_ai_field(ai_code, part[2:])

    return result


def parse_ikea_url(url: str) -> dict[str, AIField]:
    """
    Parse IKEA's proprietary DataMatrix URL format.

    Example: "https://www.goto.ikea.com/240/0299999710973?13=260402"

    The path encodes AI(240) and query parameters encode other AIs.
    This format is used by IKEA instead of standard GS1 DataMatrix encoding.
    """
    result: dict[str, AIField] = {}
    if "goto.ikea.com" not in url and "ikea.com" not in url:
        return result

    # Extract AI(240) from path: /240/<value>
    path_match = re.search(r"/(\d{2,4})/([^/?#]+)", url)
    if path_match:
        ai_code = path_match.group(1)
        raw_value = path_match.group(2)
        result[ai_code] = _build_ai_field(ai_code, raw_value)

    # Extract other AIs from query string: ?13=260402&10=ABC
    from urllib.parse import urlparse, parse_qs
    parsed = urlparse(url)
    query = parse_qs(parsed.query, keep_blank_values=False)
    for ai_code, values in query.items():
        if values and ai_code.isdigit():
            result[ai_code] = _build_ai_field(ai_code, values[0])

    return result


def detect_and_parse(raw: str) -> dict[str, AIField]:
    """
    Auto-detect the encoding format of a DataMatrix raw value and parse it.

    Handles:
    - IKEA URL format  (goto.ikea.com/...)
    - GS1 parentheses  ((240)0299999710973)
    - GS1 compact      (binary with GS separators)
    """
    if "ikea.com" in raw:
        return parse_ikea_url(raw)
    if "(" in raw:
        return parse_parentheses_form(raw)
    return parse_gs1_compact(raw.encode("latin-1", errors="replace"))


def merge_ai_dicts(
    *dicts: dict[str, AIField],
) -> dict[str, AIField]:
    """Merge multiple AI dicts; later dicts win on conflict."""
    merged: dict[str, AIField] = {}
    for d in dicts:
        merged.update(d)
    return merged
