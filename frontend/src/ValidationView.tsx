import { useEffect, useMemo, useState } from "react";
import {
  fetchProducts,
  postValidate,
  type ProductSummary,
  type RegionRow,
  type ValidationResponse,
} from "./api";

type Props = {
  fileBlob: File | null;
  fileHash: string | null;
  detectedProductCode: string | null;
  result: ValidationResponse | null;
  setResult: (v: ValidationResponse | null) => void;
  setBusy: (b: boolean) => void;
  setBusyMessage: (m: string) => void;
  setErr: (e: string | null) => void;
};

const STATUS_CLASS: Record<string, string> = {
  PASS: "status-pass",
  FAIL: "status-fail",
  OPTIONAL: "status-opt",
};

function fmtMm(v: number | null | undefined): string {
  if (v == null || Number.isNaN(Number(v))) return "—";
  const n = Number(v);
  return `${n >= 0 ? "+" : ""}${n.toFixed(2)}`;
}

function fmtScore(v: number | null | undefined): string {
  if (v == null || Number.isNaN(Number(v))) return "—";
  return Number(v).toFixed(4);
}

// ── Metadata-file parsing ──────────────────────────────────────────────────
// Production labels usually arrive with a sidecar metadata file that names the
// template. We accept a flexible JSON shape so the user can drop whatever
// they have without renaming fields.

type LabelMetadata = {
  productCode?: string;
  templateVersion?: string;
  pageIndex?: number;
  layoutOnly?: boolean;
  toleranceMm?: number;
  allowAutoSchema?: boolean;
};

/** Pick the first present key from a dict regardless of camelCase / snake_case. */
function _pick<T>(obj: Record<string, unknown>, keys: string[]): T | undefined {
  for (const k of keys) {
    if (obj[k] !== undefined && obj[k] !== null) return obj[k] as T;
  }
  return undefined;
}

function parseMetadataJson(text: string): LabelMetadata {
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch (e) {
    throw new Error(
      `Could not parse metadata JSON: ${e instanceof Error ? e.message : e}`
    );
  }
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
    throw new Error(
      "Metadata file must be a JSON object (e.g. { product_code: 'L10555-MIADM' })."
    );
  }
  const o = raw as Record<string, unknown>;
  const pc = _pick<string>(o, [
    "product_code",
    "productCode",
    "template",
    "label_template",
    "ikea_template",
    "ikea_label_template",
  ]);
  const ver = _pick<string>(o, [
    "template_version",
    "templateVersion",
    "version",
  ]);
  const page = _pick<number>(o, ["page", "page_index", "pageIndex"]);
  const layoutOnly = _pick<boolean>(o, [
    "layout_only",
    "layoutOnly",
  ]);
  const tol = _pick<number>(o, [
    "layout_tolerance_mm",
    "tolerance_mm",
    "toleranceMm",
  ]);
  const allowAuto = _pick<boolean>(o, [
    "allow_auto_schema",
    "allowAutoSchema",
  ]);
  if (!pc) {
    throw new Error(
      "Metadata file must contain a product code under one of: " +
        "product_code, productCode, template, label_template."
    );
  }
  return {
    productCode: typeof pc === "string" ? pc.trim() : undefined,
    templateVersion: typeof ver === "string" ? ver.trim() : undefined,
    pageIndex: typeof page === "number" ? page : undefined,
    layoutOnly: typeof layoutOnly === "boolean" ? layoutOnly : undefined,
    toleranceMm: typeof tol === "number" ? tol : undefined,
    allowAutoSchema:
      typeof allowAuto === "boolean" ? allowAuto : undefined,
  };
}

function MetadataChip({
  applied,
  onClear,
}: {
  applied: { fileName: string; fields: LabelMetadata };
  onClear: () => void;
}) {
  return (
    <span className="metadata-chip">
      From <code>{applied.fileName}</code>
      {applied.fields.productCode && (
        <>
          {" "}
          → <strong>{applied.fields.productCode}</strong>
        </>
      )}
      {applied.fields.templateVersion && (
        <> · {applied.fields.templateVersion}</>
      )}
      <button
        type="button"
        className="link-remove"
        onClick={onClear}
        aria-label="Clear metadata"
      >
        ✕
      </button>
    </span>
  );
}

export function ValidationView(props: Props) {
  const {
    fileBlob,
    fileHash,
    detectedProductCode,
    result,
    setResult,
    setBusy,
    setBusyMessage,
    setErr,
  } = props;

  const [products, setProducts] = useState<ProductSummary[]>([]);
  const [productCode, setProductCode] = useState("");
  const [version, setVersion] = useState("");
  const [pageIndex, setPageIndex] = useState(0);
  const [layoutOnly, setLayoutOnly] = useState(true);
  const [tolMm, setTolMm] = useState(1.5);
  const [allowAuto, setAllowAuto] = useState(false);
  const [showHtml, setShowHtml] = useState(false);
  const [productsErr, setProductsErr] = useState<string | null>(null);
  const [metaApplied, setMetaApplied] = useState<{
    fileName: string;
    fields: LabelMetadata;
  } | null>(null);
  const [metaErr, setMetaErr] = useState<string | null>(null);
  const [metaSource, setMetaSource] = useState<"file" | "paste">("file");
  const [pasted, setPasted] = useState("");

  useEffect(() => {
    fetchProducts()
      .then((p) => setProducts(p))
      .catch((e) =>
        setProductsErr(
          e instanceof Error ? e.message : `Could not load products: ${e}`
        )
      );
  }, []);

  // Auto-pick from extractor's ai_240 if it matches a known product.
  useEffect(() => {
    if (!detectedProductCode || products.length === 0) return;
    const match = products.find(
      (p) =>
        p.product_code.toUpperCase() === detectedProductCode.toUpperCase()
    );
    if (match && !productCode) {
      setProductCode(match.product_code);
    }
  }, [detectedProductCode, products, productCode]);

  const selectedProduct = useMemo(
    () => products.find((p) => p.product_code === productCode) || null,
    [products, productCode]
  );

  // When the product changes, default to the latest version (last entry).
  useEffect(() => {
    if (!selectedProduct) return;
    if (selectedProduct.versions.length === 0) {
      setVersion("");
      return;
    }
    if (!selectedProduct.versions.includes(version)) {
      setVersion(selectedProduct.versions[selectedProduct.versions.length - 1]);
    }
  }, [selectedProduct, version]);

  const detectedMatchesAny =
    !!detectedProductCode &&
    products.some(
      (p) =>
        p.product_code.toUpperCase() === detectedProductCode.toUpperCase()
    );

  const applyMetadata = (md: LabelMetadata, fileName: string) => {
    if (md.productCode) setProductCode(md.productCode);
    if (md.templateVersion) setVersion(md.templateVersion);
    if (typeof md.pageIndex === "number") setPageIndex(md.pageIndex);
    if (typeof md.layoutOnly === "boolean") setLayoutOnly(md.layoutOnly);
    if (typeof md.toleranceMm === "number") setTolMm(md.toleranceMm);
    if (typeof md.allowAutoSchema === "boolean") setAllowAuto(md.allowAutoSchema);
    setMetaApplied({ fileName, fields: md });
    setMetaErr(null);
  };

  const onMetadataFile = async (file: File | null | undefined) => {
    if (!file) return;
    try {
      const text = await file.text();
      const md = parseMetadataJson(text);
      applyMetadata(md, file.name);
    } catch (e) {
      setMetaErr(e instanceof Error ? e.message : String(e));
      setMetaApplied(null);
    }
  };

  const onApplyPasted = () => {
    if (!pasted.trim()) return;
    try {
      const md = parseMetadataJson(pasted);
      applyMetadata(md, "(pasted)");
    } catch (e) {
      setMetaErr(e instanceof Error ? e.message : String(e));
      setMetaApplied(null);
    }
  };

  const clearMetadata = () => {
    setMetaApplied(null);
    setMetaErr(null);
    setPasted("");
  };

  const canRun = !!(fileHash || fileBlob) && !!productCode;

  const runValidation = async () => {
    if (!fileHash && !fileBlob) {
      setErr("Upload a label PDF first (Extractor tab) before validating.");
      return;
    }
    if (!productCode) {
      setErr("Pick a product code before running validation.");
      return;
    }
    setErr(null);
    setBusyMessage(
      fileHash
        ? "Running layout validation (cached upload)…"
        : "Running layout validation…"
    );
    setBusy(true);
    setResult(null);
    try {
      // Prefer the cached hash so the bytes are not re-uploaded over the wire.
      const r = await postValidate(
        fileHash ? { fileHash } : { file: fileBlob },
        {
          productCode,
          version: version || null,
          page: pageIndex,
          layoutOnly,
          layoutToleranceMm: tolMm,
          allowAutoSchema: allowAuto,
        }
      );
      setResult(r);
    } catch (x) {
      setErr(x instanceof Error ? x.message : String(x));
    } finally {
      setBusy(false);
      setBusyMessage("Working…");
    }
  };

  const verdict = result?.verdict;
  const verdictClass =
    verdict === "VALID"
      ? "verdict-pill verdict-valid"
      : verdict === "INVALID"
        ? "verdict-pill verdict-invalid"
        : "verdict-pill";

  return (
    <div className="validation-root">
      <section className="validation-controls panel-dark">
        <h3>Layout validation against approved template</h3>
        {productsErr && <p className="alert-err">{productsErr}</p>}
        {detectedProductCode && detectedMatchesAny && (
          <p className="hint-inline" style={{ marginBottom: 8 }}>
            Auto-detected from <code>ai_240</code>:{" "}
            <strong>{detectedProductCode}</strong>
          </p>
        )}
        {detectedProductCode && !detectedMatchesAny && products.length > 0 && (
          <p className="badge-warn" style={{ marginBottom: 8 }}>
            Extractor read <code>{detectedProductCode}</code>, but no template
            with that name exists. Pick one manually below.
          </p>
        )}

        <div className="metadata-block">
          <div className="metadata-block__header">
            <span className="metadata-block__title">
              Metadata (optional)
            </span>
            <span className="hint-inline">
              Pre-fills product, version, page, tolerance.
            </span>
          </div>
          <div className="metadata-block__tabs" role="tablist">
            <button
              type="button"
              role="tab"
              aria-selected={metaSource === "file"}
              className={`metadata-block__tab ${
                metaSource === "file" ? "metadata-block__tab--active" : ""
              }`}
              onClick={() => setMetaSource("file")}
            >
              Upload file
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={metaSource === "paste"}
              className={`metadata-block__tab ${
                metaSource === "paste" ? "metadata-block__tab--active" : ""
              }`}
              onClick={() => setMetaSource("paste")}
            >
              Paste JSON
            </button>
          </div>

          {metaSource === "file" && (
            <div className="metadata-block__row">
              <label className="btn metadata-block__btn">
                Choose JSON
                <input
                  type="file"
                  accept=".json,application/json,text/plain"
                  style={{ display: "none" }}
                  onChange={(e) => {
                    const f = e.target.files?.[0];
                    void onMetadataFile(f);
                    // allow re-selecting the same file
                    e.target.value = "";
                  }}
                />
              </label>
              {metaApplied && (
                <MetadataChip
                  applied={metaApplied}
                  onClear={clearMetadata}
                />
              )}
            </div>
          )}

          {metaSource === "paste" && (
            <div className="metadata-block__paste">
              <textarea
                className="text-field metadata-block__textarea"
                rows={6}
                spellCheck={false}
                placeholder='{"product_code":"L10555-MIADM","template_version":"v1"}'
                value={pasted}
                onChange={(e) => setPasted(e.target.value)}
              />
              <div className="metadata-block__row">
                <button
                  type="button"
                  className="btn btn-primary"
                  onClick={onApplyPasted}
                  disabled={!pasted.trim()}
                >
                  Apply pasted JSON
                </button>
                <button
                  type="button"
                  className="btn"
                  onClick={() => setPasted("")}
                  disabled={!pasted}
                >
                  Clear text
                </button>
                {metaApplied && (
                  <MetadataChip
                    applied={metaApplied}
                    onClear={clearMetadata}
                  />
                )}
              </div>
            </div>
          )}

          {metaErr && (
            <p className="alert-err" style={{ marginTop: 6 }}>
              {metaErr}
            </p>
          )}
          <details className="metadata-help">
            <summary>Accepted JSON shape</summary>
            <pre>{`{
  "product_code": "L10555-MIADM",   // required
  "template_version": "v1",          // optional
  "page": 0,                         // optional
  "layout_only": true,               // optional
  "layout_tolerance_mm": 1.5,        // optional
  "allow_auto_schema": false         // optional
}`}</pre>
            <p className="hint-inline" style={{ marginTop: 4 }}>
              Aliases accepted: <code>productCode</code>, <code>template</code>,{" "}
              <code>label_template</code>; <code>version</code>;{" "}
              <code>page_index</code>; <code>tolerance_mm</code>. Anything you
              load is still overridable below.
            </p>
          </details>
        </div>

        <div className="validation-grid">
          <label className="label-row">
            Product code
            <select
              className="select-field"
              value={productCode}
              onChange={(e) => setProductCode(e.target.value)}
            >
              <option value="">— select —</option>
              {products.map((p) => (
                <option key={p.product_code} value={p.product_code}>
                  {p.product_code}
                  {p.has_manual_schema ? "" : " (no schema)"}
                </option>
              ))}
            </select>
          </label>
          <label className="label-row">
            Template version
            <select
              className="select-field"
              value={version}
              onChange={(e) => setVersion(e.target.value)}
              disabled={
                !selectedProduct || selectedProduct.versions.length === 0
              }
            >
              {selectedProduct?.versions.map((v) => (
                <option key={v} value={v}>
                  {v}
                </option>
              ))}
              {(!selectedProduct ||
                selectedProduct.versions.length === 0) && (
                <option value="">—</option>
              )}
            </select>
          </label>
          <label className="label-row">
            Page index
            <input
              className="text-field"
              type="number"
              min={0}
              value={pageIndex}
              onChange={(e) => setPageIndex(Number(e.target.value) || 0)}
            />
          </label>
          <label className="label-row">
            Tolerance (mm)
            <input
              className="text-field"
              type="number"
              min={0}
              step={0.1}
              value={tolMm}
              onChange={(e) => setTolMm(Number(e.target.value) || 0)}
              disabled={!layoutOnly}
            />
          </label>
        </div>
        <div className="validation-toggles">
          <label className="checkbox-row">
            <input
              type="checkbox"
              checked={layoutOnly}
              onChange={(e) => setLayoutOnly(e.target.checked)}
            />
            <span>Layout-only mode (geometry/mm; ignore text payload)</span>
          </label>
          <label className="checkbox-row">
            <input
              type="checkbox"
              checked={allowAuto}
              onChange={(e) => setAllowAuto(e.target.checked)}
            />
            <span>
              Allow auto-generated schema (when no manual schema exists)
            </span>
          </label>
        </div>
        <div style={{ display: "flex", gap: 8, marginTop: 12 }}>
          <button
            type="button"
            className="btn btn-primary"
            onClick={() => {
              void runValidation();
            }}
            disabled={!canRun}
          >
            Run validation
          </button>
          {!canRun && !fileHash && !fileBlob && (
            <span className="hint-inline">
              Drop a label on the Extractor tab to enable this.
            </span>
          )}
          {fileHash && (
            <span className="hint-inline">
              Using cached upload <code>{fileHash}</code> — no re-upload.
            </span>
          )}
        </div>
      </section>

      {result && (
        <section className="validation-result">
          <div className="validation-summary">
            <span className={verdictClass}>{result.verdict}</span>
            <div className="metric">
              <span>Final score</span>
              <span className="metric__value">
                {fmtScore(result.final_score)}
              </span>
            </div>
            <div className="metric">
              <span>Threshold</span>
              <span className="metric__value">
                {fmtScore(result.global_threshold)}
              </span>
            </div>
            <div className="metric">
              <span>Mode</span>
              <span className="metric__value">{result.validation_mode}</span>
            </div>
            <div className="metric">
              <span>Schema</span>
              <span className="metric__value">{result.schema_source}</span>
            </div>
            <div className="metric">
              <span>Template</span>
              <span className="metric__value">
                {result.product_code} · {result.template_version}
              </span>
            </div>
          </div>

          <div className="overview-pair">
            <figure>
              <figcaption>Approved template</figcaption>
              {result.template_overview_png_base64 ? (
                <img
                  alt="Template overview"
                  src={`data:image/png;base64,${result.template_overview_png_base64}`}
                />
              ) : (
                <div className="overview-missing">no image</div>
              )}
            </figure>
            <figure>
              <figcaption>Candidate (your upload)</figcaption>
              {result.candidate_overview_png_base64 ? (
                <img
                  alt="Candidate overview"
                  src={`data:image/png;base64,${result.candidate_overview_png_base64}`}
                />
              ) : (
                <div className="overview-missing">no image</div>
              )}
            </figure>
          </div>

          <RegionTable rows={result.region_results} />

          {result.failed_regions.length > 0 && (
            <FailedRegionsList failed={result.failed_regions} />
          )}

          <div style={{ marginTop: 12, display: "flex", gap: 8 }}>
            <button
              type="button"
              className="btn"
              onClick={() => setShowHtml((s) => !s)}
              disabled={!result.report_html}
            >
              {showHtml ? "Hide" : "View"} full HTML report
            </button>
            {result.report_html && (
              <button
                type="button"
                className="btn"
                onClick={() => {
                  const blob = new Blob([result.report_html], {
                    type: "text/html",
                  });
                  const url = URL.createObjectURL(blob);
                  window.open(url, "_blank", "noopener");
                  setTimeout(() => URL.revokeObjectURL(url), 60_000);
                }}
              >
                Open report in new tab
              </button>
            )}
          </div>
          {showHtml && result.report_html && (
            <iframe
              title="Validation report"
              className="report-iframe"
              srcDoc={result.report_html}
            />
          )}
        </section>
      )}
    </div>
  );
}

function RegionTable({ rows }: { rows: RegionRow[] }) {
  if (!rows.length) return null;
  return (
    <div className="region-table-wrap">
      <h3>Per-region results</h3>
      <table className="region-table">
        <thead>
          <tr>
            <th>Region</th>
            <th>Type</th>
            <th>Status</th>
            <th>Score</th>
            <th>Threshold</th>
            <th>Δx (mm)</th>
            <th>Δy (mm)</th>
            <th>Δw (mm)</th>
            <th>Δh (mm)</th>
            <th>Notes</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.name}>
              <td>
                <code>{r.name}</code>
              </td>
              <td>{r.type ?? "—"}</td>
              <td>
                <span
                  className={`pill ${STATUS_CLASS[r.status] ?? ""}`}
                >
                  {r.status}
                </span>
              </td>
              <td>{fmtScore(r.score)}</td>
              <td>{fmtScore(r.threshold)}</td>
              <td>{fmtMm(r.shift_x_mm)}</td>
              <td>{fmtMm(r.shift_y_mm)}</td>
              <td>{fmtMm(r.width_delta_mm)}</td>
              <td>{fmtMm(r.height_delta_mm)}</td>
              <td className="region-notes">
                {r.notes && r.notes.length ? r.notes.join("; ") : "—"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function FailedRegionsList({
  failed,
}: {
  failed: ValidationResponse["failed_regions"];
}) {
  return (
    <div className="failed-regions">
      <h3>Failed regions ({failed.length})</h3>
      <ul>
        {failed.map((f) => (
          <li key={f.name}>
            <strong>
              <code>{f.name}</code>
            </strong>
            {f.severity ? (
              <span className="pill pill-warn">{f.severity}</span>
            ) : null}
            <div className="hint-inline">
              score {fmtScore(f.combined_score)} · threshold{" "}
              {fmtScore(f.threshold)} · Δx {fmtMm(f.shift_x_mm)} · Δy{" "}
              {fmtMm(f.shift_y_mm)} · Δw {fmtMm(f.width_delta_mm)} · Δh{" "}
              {fmtMm(f.height_delta_mm)}
            </div>
            {f.failure_reasons && f.failure_reasons.length > 0 && (
              <ul className="reasons">
                {f.failure_reasons.map((reason, i) => (
                  <li key={i}>{reason}</li>
                ))}
              </ul>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}
