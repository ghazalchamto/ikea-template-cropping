/**
 * Map extracted field ids to the same category palette as the sidebar legend
 * (identity, barcodes, address, compliance, date, copy).
 */
export type FieldCategory =
  | "identity"
  | "barcodes"
  | "address"
  | "compliance"
  | "date"
  | "copy";

const RGB: Record<FieldCategory, { r: number; g: number; b: number }> = {
  identity: { r: 59, g: 130, b: 246 },
  barcodes: { r: 234, g: 88, b: 12 },
  address: { r: 168, g: 85, b: 247 },
  compliance: { r: 236, g: 72, b: 153 },
  date: { r: 20, g: 184, b: 166 },
  copy: { r: 245, g: 158, b: 11 },
};

const ID_TO_CATEGORY: Record<string, FieldCategory> = {
  product_name: "identity",
  product_dimensions_metric: "identity",
  product_dimensions_imperial: "identity",
  article_number: "identity",
  article_code: "identity",
  internal_item_number: "identity",
  quantity_multiplier: "identity",
  ikea_logo: "identity",
  ean_barcode: "barcodes",
  ean_barcode_value: "barcodes",
  itf14_barcode: "barcodes",
  itf14_human_readable: "barcodes",
  datamatrix_barcode: "barcodes",
  ai_240: "barcodes",
  ai_13: "barcodes",
  ai_11: "barcodes",
  ai_10: "barcodes",
  human_readable_date: "date",
  date_stamp: "date",
  date_alpha_code: "date",
  date_numeric_prefix: "date",
  custom_identifier: "date",
  plant_identifier: "address",
  supplier_name: "address",
  supplier_address: "address",
  address_block: "address",
  origin_text: "address",
  copyright_notice: "compliance",
  compliance_marks: "compliance",
  age_rating: "compliance",
  gross_weight: "identity",
  net_weight: "identity",
  package_number: "identity",
  package_type: "identity",
  copy_count: "copy",
  copy_count_inline: "copy",
};

function fallbackCategory(id: string): FieldCategory {
  const s = id.toLowerCase();
  if (s.includes("barcode") || s.startsWith("ai_") || s.includes("ean") || s.includes("itf")) {
    return "barcodes";
  }
  if (s.includes("date") || s.includes("stamp") || s.includes("identifier")) {
    return "date";
  }
  if (s.includes("address") || s.includes("supplier") || s.includes("plant") || s.includes("origin")) {
    return "address";
  }
  if (s.includes("compliance") || s.includes("copyright") || s.includes("age")) {
    return "compliance";
  }
  if (s.includes("copy")) {
    return "copy";
  }
  return "identity";
}

export function fieldIdToCategory(fieldId: string): FieldCategory {
  return ID_TO_CATEGORY[fieldId] ?? fallbackCategory(fieldId);
}

/** Tinted “glass” fill (most of the color) + thin edge for definition */
export function boxStyleForField(
  fieldId: string
): { stroke: string; fill: string; strokeWidth: number } {
  const c = fieldIdToCategory(fieldId);
  const { r, g, b } = RGB[c];
  return {
    fill: `rgba(${r},${g},${b},0.4)`,
    stroke: `rgba(${r},${g},${b},0.72)`,
    strokeWidth: 1.5,
  };
}

/** Draft uses same palette but a bit more fill so the active draw is obvious */
export function draftBoxStyle(
  fieldId: string
): { stroke: string; fill: string; strokeWidth: number } {
  const c = fieldIdToCategory(fieldId);
  const { r, g, b } = RGB[c];
  return {
    fill: `rgba(${r},${g},${b},0.48)`,
    stroke: `rgba(${r},${g},${b},0.88)`,
    strokeWidth: 2,
  };
}

/** Current dropdown field: obvious focus ring; label stays visible through fill */
export function selectedFieldHighlightStyle(fieldId: string): {
  fill: string;
  stroke: string;
  strokeWidth: number;
  shadowColor: string;
  shadowBlur: number;
} {
  const c = fieldIdToCategory(fieldId);
  const { r, g, b } = RGB[c];
  return {
    fill: `rgba(${r},${g},${b},0.28)`,
    stroke: "rgba(248, 250, 252, 0.95)",
    strokeWidth: 2.5,
    shadowColor: `rgba(${r},${g},${b},0.75)`,
    shadowBlur: 10,
  };
}

export function ocrWordColor(
  index: number,
  pdfStyle: boolean
): { stroke: string; fill: string; strokeWidth: number } {
  if (pdfStyle) {
    const h = (index * 47) % 200;
    const hue = 180 + h;
    return {
      fill: `hsla(${hue}, 70%, 52%, 0.38)`,
      stroke: `hsla(${hue}, 80%, 42%, 0.55)`,
      strokeWidth: 1,
    };
  }
  const h = (index * 41) % 300;
  return {
    fill: `hsla(${h}, 50%, 50%, 0.36)`,
    stroke: `hsla(${h}, 55%, 38%, 0.5)`,
    strokeWidth: 1,
  };
}

export const regionOverlayStyle = {
  stroke: "rgba(130, 70, 220, 0.65)",
  fill: "rgba(168, 85, 247, 0.28)",
  strokeWidth: 1.5,
};
