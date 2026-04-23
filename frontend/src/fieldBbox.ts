/**
 * Resolve a bounding box for the active field: human correction first, else model extraction.
 */
export type BBox = { x: number; y: number; w: number; h: number };

type CorMap = Record<string, { value: string; bbox: BBox | null }>;

const AI_KEY: Record<string, string> = {
  ai_240: "240",
  ai_13: "13",
  ai_11: "11",
  ai_10: "10",
};

function pickBbox(
  el: unknown
): BBox | null {
  if (!el || typeof el !== "object") return null;
  const b = (el as { bbox?: { x?: number; y?: number; w?: number; h?: number } }).bbox;
  if (!b || b.w == null || b.h == null) return null;
  const w = Number(b.w);
  const h = Number(b.h);
  if (w <= 0 || h <= 0) return null;
  return {
    x: Number(b.x),
    y: Number(b.y),
    w,
    h,
  };
}

/**
 * Bbox to outline for the current dropdown field: corrections[fieldId].bbox, else
 * `label.elements[...]` (text, barcode, datamatrix_block).
 */
export function getBboxForSelectedField(
  label: unknown,
  fieldId: string,
  corrections: CorMap
): BBox | null {
  const cor = corrections[fieldId];
  if (cor?.bbox && cor.bbox.w > 0 && cor.bbox.h > 0) {
    return {
      x: cor.bbox.x,
      y: cor.bbox.y,
      w: cor.bbox.w,
      h: cor.bbox.h,
    };
  }
  const L = label as { elements?: Record<string, unknown> } | null;
  const e = L?.elements;
  if (!e) return null;

  if (fieldId === "datamatrix_barcode") {
    const dmb = e.datamatrix_block as
      | { barcode?: Record<string, unknown> }
      | undefined;
    return pickBbox(dmb?.barcode);
  }
  const ai = AI_KEY[fieldId];
  if (ai) {
    const dmb = e.datamatrix_block as
      | { ai_fields?: Record<string, unknown> }
      | undefined;
    return pickBbox(dmb?.ai_fields?.[ai]);
  }
  if (fieldId === "ean_barcode" || fieldId === "itf14_barcode") {
    return pickBbox(e[fieldId]);
  }
  return pickBbox(e[fieldId]);
}
