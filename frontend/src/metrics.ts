/**
 * Count filled slots — same field list as app.py all_fields.
 */
const ELEMENT_KEYS: string[] = [
  "product_name",
  "product_dimensions_metric",
  "product_dimensions_imperial",
  "article_number",
  "article_code",
  "internal_item_number",
  "quantity_multiplier",
  "gross_weight",
  "net_weight",
  "package_number",
  "package_type",
  "ikea_logo",
  "ean_barcode",
  "ean_barcode_value",
  "itf14_barcode",
  "itf14_human_readable",
  "datamatrix_barcode",
  "ai_240",
  "ai_13",
  "ai_11",
  "ai_10",
  "human_readable_date",
  "date_stamp",
  "date_alpha_code",
  "date_numeric_prefix",
  "custom_identifier",
  "plant_identifier",
  "supplier_name",
  "supplier_address",
  "address_block",
  "origin_text",
  "copyright_notice",
  "compliance_marks",
  "age_rating",
  "copy_count",
  "copy_count_inline",
];

const AI_TO_BLOCK: Record<string, string> = {
  ai_240: "240",
  ai_13: "13",
  ai_11: "11",
  ai_10: "10",
};

function fieldPresent(
  e: Record<string, unknown>,
  k: string
): boolean {
  if (k === "ean_barcode" || k === "itf14_barcode") {
    const b = e[k] as { raw_data?: string } | null | undefined;
    return b != null && (b.raw_data != null && String(b.raw_data) !== "");
  }
  if (k === "datamatrix_barcode") {
    return (e.datamatrix_block as { barcode?: { raw_data?: string } } | null)?.barcode != null;
  }
  if (k in AI_TO_BLOCK) {
    const g = AI_TO_BLOCK[k];
    const a = (e.datamatrix_block as { ai_fields?: Record<string, { value?: unknown }> } | null)
      ?.ai_fields?.[g];
    if (!a) return false;
    return a.value != null && String(a.value) !== "";
  }
  const fe = e[k] as { value?: unknown; raw_data?: string } | null | undefined;
  if (fe == null) return false;
  if (typeof fe === "object" && "raw_data" in fe) {
    return fe.raw_data != null && String(fe.raw_data) !== "";
  }
  if (typeof fe === "object" && "value" in fe) {
    return fe.value != null && String(fe.value) !== "";
  }
  return true;
}

export function countExtractedFields(label: unknown): { found: number; total: number } {
  const l = label as { elements?: Record<string, unknown> };
  const e = l?.elements;
  const total = ELEMENT_KEYS.length;
  if (!e) return { found: 0, total };
  let found = 0;
  for (const k of ELEMENT_KEYS) {
    if (fieldPresent(e, k)) found++;
  }
  return { found, total };
}

export function getRawWords(
  label: unknown
): { x: number; y: number; w: number; h: number }[] {
  const l = label as { raw_words?: { x: number; y: number; w: number; h: number }[] };
  if (!l?.raw_words || !Array.isArray(l.raw_words)) return [];
  return l.raw_words
    .filter((r) => r && r.w > 0 && r.h > 0)
    .map((r) => ({ x: r.x, y: r.y, w: r.w, h: r.h }));
}

export function getRegionBoxes(label: unknown): { x: number; y: number; w: number; h: number }[] {
  const l = label as { raw_elements?: { type?: string; x: number; y: number; w: number; h: number }[] };
  if (!l?.raw_elements || !Array.isArray(l.raw_elements)) return [];
  return l.raw_elements
    .filter((r) => r?.type === "region" && r.w > 0 && r.h > 0)
    .map((r) => ({ x: r.x, y: r.y, w: r.w, h: r.h }));
}
