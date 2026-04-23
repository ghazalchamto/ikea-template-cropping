/**
 * Read display string for a field from `label` (ExtractedLabel.to_dict() shape).
 */
export function getFieldValueFromLabel(label: unknown, fieldId: string): string {
  const L = label as { elements?: Record<string, unknown> };
  const el = L?.elements;
  if (!el) return "";
  if (fieldId === "ean_barcode" || fieldId === "itf14_barcode") {
    const b = el[fieldId] as { raw_data?: string } | undefined;
    return b?.raw_data != null ? String(b.raw_data) : "";
  }
  if (fieldId === "datamatrix_barcode") {
    const dmb = el.datamatrix_block as
      | { barcode?: { raw_data?: string } }
      | undefined;
    return dmb?.barcode?.raw_data != null ? String(dmb.barcode.raw_data) : "";
  }
  const ai = { ai_240: "240", ai_13: "13", ai_11: "11", ai_10: "10" } as const;
  const g = ai[fieldId as keyof typeof ai];
  if (g) {
    const dmb = el.datamatrix_block as
      | { ai_fields?: Record<string, { value?: string | null }> }
      | undefined;
    const f = dmb?.ai_fields?.[g];
    return f?.value != null && f.value !== "" ? String(f.value) : "";
  }
  const fe = el[fieldId] as { value?: string | null } | undefined;
  if (fe && "value" in fe && fe.value != null) return String(fe.value);
  return "";
}
