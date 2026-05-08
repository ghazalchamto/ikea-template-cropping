import type { ValidationResponse } from "./api";

type Props = {
  label: Record<string, unknown> | null;
  preview: string | null;
  validationResult: ValidationResponse | null;
  fileHash: string;
  detectedProductCode: string | null;
};

function pct(v: number | undefined | null): string {
  if (v == null || Number.isNaN(Number(v))) return "—";
  return `${Math.round(Number(v) * 100)}%`;
}

function countRegionsByStatus(result: ValidationResponse | null, status: string): number {
  if (!result) return 0;
  return result.region_results.filter((r) => r.status === status).length;
}

export function QADashboardView({
  label,
  preview,
  validationResult,
  fileHash,
  detectedProductCode,
}: Props) {
  const extractionMeta = (
    label as { extraction_metadata?: { overall_confidence?: number } } | null
  )?.extraction_metadata;
  const conf = extractionMeta?.overall_confidence;

  if (!label && !validationResult) {
    return (
      <div className="empty-state">
        <div className="empty-icon" aria-hidden>
          ✅
        </div>
        <div className="empty-title">
          Upload a label and run validation to populate the QA dashboard
        </div>
      </div>
    );
  }

  return (
    <section className="qa-dashboard">
      <div className="qa-header panel-dark">
        <h3>QA Dashboard</h3>
        <p className="hint-inline" style={{ marginTop: 0 }}>
          Single-screen QA view for production labels: extraction quality, template
          validation verdict, and visual side-by-side evidence.
        </p>
        <div className="qa-pill-row">
          <span className="pill">file_hash: {fileHash || "—"}</span>
          <span className="pill">
            detected_template: {detectedProductCode ?? "—"}
          </span>
          <span className="pill">extract_confidence: {pct(conf)}</span>
          <span
            className={`pill ${
              validationResult?.verdict === "VALID"
                ? "status-pass"
                : validationResult?.verdict === "INVALID"
                  ? "status-fail"
                  : ""
            }`}
          >
            verdict: {validationResult?.verdict ?? "not-run"}
          </span>
        </div>
      </div>

      <div className="qa-summary-grid">
        <div className="metric">
          <span>Validation score</span>
          <span className="metric__value">
            {validationResult
              ? `${(validationResult.final_score ?? 0).toFixed(4)}`
              : "—"}
          </span>
        </div>
        <div className="metric">
          <span>Threshold</span>
          <span className="metric__value">
            {validationResult
              ? `${(validationResult.global_threshold ?? 0).toFixed(4)}`
              : "—"}
          </span>
        </div>
        <div className="metric">
          <span>Passed regions</span>
          <span className="metric__value">
            {countRegionsByStatus(validationResult, "PASS")}
          </span>
        </div>
        <div className="metric">
          <span>Failed regions</span>
          <span className="metric__value">
            {countRegionsByStatus(validationResult, "FAIL")}
          </span>
        </div>
      </div>

      <div className="qa-compare-grid">
        <figure className="qa-figure">
          <figcaption>Extractor preview</figcaption>
          {preview ? (
            <img src={preview} alt="Extractor preview" />
          ) : (
            <div className="overview-missing">no extraction preview</div>
          )}
        </figure>
        <figure className="qa-figure">
          <figcaption>Approved template overview</figcaption>
          {validationResult?.template_overview_png_base64 ? (
            <img
              src={`data:image/png;base64,${validationResult.template_overview_png_base64}`}
              alt="Template overview"
            />
          ) : (
            <div className="overview-missing">run validation first</div>
          )}
        </figure>
        <figure className="qa-figure">
          <figcaption>Candidate validation overview</figcaption>
          {validationResult?.candidate_overview_png_base64 ? (
            <img
              src={`data:image/png;base64,${validationResult.candidate_overview_png_base64}`}
              alt="Candidate overview"
            />
          ) : (
            <div className="overview-missing">run validation first</div>
          )}
        </figure>
      </div>
    </section>
  );
}

