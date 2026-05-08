// Shared API helpers + types for the extractor (port 8000 via /api) and the
// visual-validation microservice (port 8001 via /validation-api).

const API = "";
const VAL_API = "/validation-api";

export type Field = { id: string; label: string };
export type BBox = { x: number; y: number; w: number; h: number };
export type CorMap = Record<string, { value: string; bbox: BBox | null }>;

export type ValRep = {
  compliance_score?: number;
  label_type_name?: string;
  n_error?: number;
  n_warning?: number;
};
export type RepLayout = { n_error?: number; n_warning?: number };
export type RepOvl = { n_error?: number; n_warning?: number };

export type ExtractResponse = {
  label: Record<string, unknown>;
  file_hash: string;
  file_name?: string;
  dpi?: number;
  is_pdf?: boolean;
  preview_png_base64: string | null;
  image_width: number;
  image_height: number;
};

export type ProductSummary = {
  product_code: string;
  versions: string[];
  has_manual_schema: boolean;
};

export type RegionRow = {
  name: string;
  type?: string;
  method?: string;
  status: "PASS" | "FAIL" | "OPTIONAL" | string;
  mandatory?: boolean;
  weight?: number;
  score: number | null;
  threshold?: number | null;
  notes?: string[];
  shift_x_mm?: number | null;
  shift_y_mm?: number | null;
  width_delta_mm?: number | null;
  height_delta_mm?: number | null;
  severity?: string | null;
  failure_reasons?: string[];
};

export type FailedRegion = {
  name: string;
  type?: string;
  method?: string;
  combined_score?: number | null;
  threshold?: number | null;
  failure_reasons?: string[];
  notes?: string[];
  shift_x_mm?: number | null;
  shift_y_mm?: number | null;
  width_delta_mm?: number | null;
  height_delta_mm?: number | null;
  severity?: string | null;
};

export type ValidationResponse = {
  verdict: "VALID" | "INVALID" | string;
  final_score: number;
  global_threshold: number;
  schema_source: string;
  product_code: string;
  template_version: string;
  page_index: number;
  validation_mode: "layout" | "legacy" | string;
  layout_tolerance_mm: number | null;
  metrics: Record<string, number>;
  region_results: RegionRow[];
  failed_regions: FailedRegion[];
  warning_regions: FailedRegion[];
  skipped_optional_regions: string[];
  candidate_overview_png_base64: string | null;
  template_overview_png_base64: string | null;
  report_html: string;
};

async function expectOk(r: Response): Promise<Response> {
  if (!r.ok) {
    const text = await r.text().catch(() => r.statusText);
    throw new Error(text || `HTTP ${r.status}`);
  }
  return r;
}

// ── Extractor API ───────────────────────────────────────────────────────────

export async function fetchFields(): Promise<Field[]> {
  const r = await expectOk(await fetch(`${API}/api/fields`));
  return (await r.json()) as Field[];
}

export async function postExtract(file: File, dpi: number): Promise<ExtractResponse> {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("dpi", String(dpi));
  const r = await expectOk(await fetch(`${API}/api/extract`, { method: "POST", body: fd }));
  return (await r.json()) as ExtractResponse;
}

/**
 * Hydrate state from the server-side upload cache (no re-upload, no re-OCR).
 * Returns null if the hash isn't in the cache (404).
 */
export async function fetchCachedExtraction(
  fileHash: string
): Promise<ExtractResponse | null> {
  const r = await fetch(
    `${API}/api/cached/${encodeURIComponent(fileHash)}`
  );
  if (r.status === 404) return null;
  await expectOk(r);
  return (await r.json()) as ExtractResponse;
}

export async function deleteCachedExtraction(fileHash: string): Promise<void> {
  const r = await fetch(
    `${API}/api/cached/${encodeURIComponent(fileHash)}`,
    { method: "DELETE" }
  );
  if (r.status !== 404) await expectOk(r);
}

export type MergeValidateResponse = {
  validation: ValRep;
  layout: RepLayout;
  overlap: RepOvl;
  completeness: { n_missing?: number; n_consistency_issues?: number };
};

export async function postMergeValidate(
  label: Record<string, unknown>,
  corrections: CorMap
): Promise<MergeValidateResponse> {
  const r = await expectOk(
    await fetch(`${API}/api/merge-validate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ label, corrections }),
    })
  );
  return (await r.json()) as MergeValidateResponse;
}

export async function postExportJsonl(
  fileHash: string,
  dpi: number,
  label: Record<string, unknown>,
  corrections: CorMap
): Promise<{ ndjson: string }> {
  const r = await expectOk(
    await fetch(`${API}/api/export-jsonl`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ file_hash: fileHash, dpi, label, corrections }),
    })
  );
  return (await r.json()) as { ndjson: string };
}

// ── Validation API ──────────────────────────────────────────────────────────

export async function fetchProducts(): Promise<ProductSummary[]> {
  const r = await expectOk(await fetch(`${VAL_API}/api/products`));
  return (await r.json()) as ProductSummary[];
}

export type ValidateOptions = {
  productCode: string;
  version?: string | null;
  page?: number;
  layoutOnly?: boolean;
  layoutToleranceMm?: number;
  allowAutoSchema?: boolean;
};

/**
 * Run validation. Either ``file`` (multipart) or ``fileHash`` (cached upload)
 * must be provided. ``fileHash`` is preferred — it skips re-uploading bytes
 * and lets a refreshed page still validate.
 */
export async function postValidate(
  source: { file?: File | null; fileHash?: string | null },
  opts: ValidateOptions
): Promise<ValidationResponse> {
  const fd = new FormData();
  if (source.fileHash) {
    fd.append("file_hash", source.fileHash);
  } else if (source.file) {
    fd.append("file", source.file);
  } else {
    throw new Error("postValidate: provide either file or fileHash");
  }
  fd.append("product_code", opts.productCode);
  if (opts.version) fd.append("version", opts.version);
  fd.append("page", String(opts.page ?? 0));
  fd.append("layout_only", String(opts.layoutOnly ?? true));
  fd.append("layout_tolerance_mm", String(opts.layoutToleranceMm ?? 1.5));
  fd.append("allow_auto_schema", String(opts.allowAutoSchema ?? false));
  const r = await expectOk(
    await fetch(`${VAL_API}/api/validate`, { method: "POST", body: fd })
  );
  return (await r.json()) as ValidationResponse;
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/** Extract the AI(240) product code from an extracted-label payload, if any. */
export function getDetectedProductCode(
  label: Record<string, unknown> | null
): string | null {
  if (!label) return null;
  const elements = (label as { elements?: Record<string, unknown> }).elements;
  if (!elements) return null;
  const ai240 = elements["ai_240"] as { value?: string } | undefined;
  const v = ai240?.value;
  return typeof v === "string" && v.trim() ? v.trim() : null;
}
