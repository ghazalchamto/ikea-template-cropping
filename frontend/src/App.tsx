import { useEffect, useMemo, useState } from "react";
import { FileDropzone } from "./FileDropzone";
import { LoadingOverlay } from "./LoadingOverlay";
import { ExtractorView } from "./ExtractorView";
import { ValidationView } from "./ValidationView";
import { QADashboardView } from "./QADashboardView";
import {
  deleteCachedExtraction,
  fetchCachedExtraction,
  fetchFields,
  getDetectedProductCode,
  postExtract,
  postMergeValidate,
  type CorMap,
  type Field,
  type RepLayout,
  type RepOvl,
  type ValidationResponse,
  type ValRep,
} from "./api";

const HASH_KEY = "ikea.upload.file_hash";

const LEGEND: { key: string; r: number; g: number; b: number }[] = [
  { key: "identity", r: 59, g: 130, b: 246 },
  { key: "barcodes", r: 234, g: 88, b: 12 },
  { key: "address", r: 168, g: 85, b: 247 },
  { key: "compliance", r: 236, g: 72, b: 153 },
  { key: "date", r: 20, g: 184, b: 166 },
  { key: "copy", r: 245, g: 158, b: 11 },
];

type Tab = "extractor" | "validation" | "dashboard";

export function App() {
  const [tab, setTab] = useState<Tab>("extractor");
  const [fields, setFields] = useState<Field[]>([]);
  const [dpi, setDpi] = useState(600);
  const [busy, setBusy] = useState(false);
  const [busyMessage, setBusyMessage] = useState("Working…");
  const [err, setErr] = useState<string | null>(null);
  const [label, setLabel] = useState<Record<string, unknown> | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [fileHash, setFileHash] = useState("");
  const [fileBlob, setFileBlob] = useState<File | null>(null);
  const [w, setW] = useState(0);
  const [h, setH] = useState(0);
  const [corrections, setCorrections] = useState<CorMap>({});
  const [valid, setValid] = useState<ValRep | null>(null);
  const [layoutRep, setLayoutRep] = useState<RepLayout | null>(null);
  const [ovRep, setOvRep] = useState<RepOvl | null>(null);
  const [validationResult, setValidationResult] =
    useState<ValidationResponse | null>(null);
  const [showOcrWords, setShowOcrWords] = useState(true);
  const [showRegions, setShowRegions] = useState(false);
  const [showPdfText, setShowPdfText] = useState(false);
  const [showClean, setShowClean] = useState(false);
  const [showJson, setShowJson] = useState(false);

  useEffect(() => {
    fetchFields()
      .then((d) => setFields(d))
      .catch((e) => setErr(String(e)));
  }, []);

  // Restore the previous upload from server-side cache (no re-upload, no
  // re-OCR) so a page refresh keeps the file usable for the Validation tab.
  useEffect(() => {
    const stored = localStorage.getItem(HASH_KEY);
    if (!stored) return;
    let cancelled = false;
    setBusyMessage("Restoring previous upload…");
    setBusy(true);
    fetchCachedExtraction(stored)
      .then(async (j) => {
        if (cancelled) return;
        if (!j) {
          localStorage.removeItem(HASH_KEY);
          return;
        }
        setLabel(j.label);
        setFileHash(j.file_hash);
        setW(j.image_width);
        setH(j.image_height);
        setPreview(
          j.preview_png_base64
            ? `data:image/png;base64,${j.preview_png_base64}`
            : null
        );
        setValidationResult(null);
        try {
          const m = await postMergeValidate(j.label, {});
          if (cancelled) return;
          setValid(m.validation);
          setLayoutRep(m.layout);
          setOvRep(m.overlap);
        } catch {
          // merge-validate is best-effort on restore
        }
      })
      .catch(() => {
        // Cache miss / network — drop the stale hash.
        localStorage.removeItem(HASH_KEY);
      })
      .finally(() => {
        if (!cancelled) {
          setBusy(false);
          setBusyMessage("Working…");
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const detectedProductCode = useMemo(
    () => getDetectedProductCode(label),
    [label]
  );

  const runFile = async (f: File) => {
    setErr(null);
    setBusyMessage("Uploading and extracting…");
    setBusy(true);
    setValid(null);
    setLayoutRep(null);
    setOvRep(null);
    setCorrections({});
    setPreview(null);
    setLabel(null);
    setFileBlob(f);
    try {
      const j = await postExtract(f, dpi);
      setLabel(j.label);
      setFileHash(j.file_hash);
      setW(j.image_width);
      setH(j.image_height);
      setPreview(
        j.preview_png_base64
          ? `data:image/png;base64,${j.preview_png_base64}`
          : null
      );
      try {
        localStorage.setItem(HASH_KEY, j.file_hash);
      } catch {
        // localStorage may be disabled — non-fatal.
      }
      const m = await postMergeValidate(j.label, {});
      setValid(m.validation);
      setLayoutRep(m.layout);
      setOvRep(m.overlap);
    } catch (x) {
      setErr(x instanceof Error ? x.message : String(x));
    } finally {
      setBusy(false);
      setBusyMessage("Working…");
    }
  };

  const clearCache = async () => {
    const h = fileHash;
    setLabel(null);
    setPreview(null);
    setFileHash("");
    setFileBlob(null);
    setW(0);
    setH(0);
    setCorrections({});
    setValid(null);
    setLayoutRep(null);
    setOvRep(null);
    setValidationResult(null);
    try {
      localStorage.removeItem(HASH_KEY);
    } catch {
      // ignore
    }
    if (h) {
      try {
        await deleteCachedExtraction(h);
      } catch {
        // best-effort
      }
    }
  };

  const hasExtraction = preview !== null && label !== null && w > 0;

  return (
    <div className="app-root">
      {busy && <LoadingOverlay message={busyMessage} />}
      <aside className="sidebar">
        <h2>⚙️ Options</h2>
        {tab === "extractor" && (
          <>
            <div className="option-block">
              <span className="option-label">Render DPI (PDF)</span>
              <input
                className="range-dpi"
                type="range"
                min={150}
                max={600}
                step={50}
                value={dpi}
                onChange={(e) => setDpi(Number(e.target.value))}
              />
              <div className="dpi-value">{dpi}</div>
            </div>
            <label className="checkbox-row">
              <input
                type="checkbox"
                checked={showOcrWords}
                onChange={(e) => setShowOcrWords(e.target.checked)}
              />
              <span>Show OCR word boxes</span>
            </label>
            <label className="checkbox-row">
              <input
                type="checkbox"
                checked={showRegions}
                onChange={(e) => setShowRegions(e.target.checked)}
              />
              <span>Show visual regions (Pass 4)</span>
            </label>
            <label className="checkbox-row">
              <input
                type="checkbox"
                checked={showPdfText}
                onChange={(e) => setShowPdfText(e.target.checked)}
              />
              <span>Show PDF text spans (Pass 1)</span>
            </label>
            <label className="checkbox-row">
              <input
                type="checkbox"
                checked={showClean}
                onChange={(e) => setShowClean(e.target.checked)}
              />
              <span>Show clean image too</span>
            </label>
            <label className="checkbox-row">
              <input
                type="checkbox"
                checked={showJson}
                onChange={(e) => setShowJson(e.target.checked)}
              />
              <span>Show full JSON payload</span>
            </label>
            <hr />
            <h3>Legend</h3>
            {LEGEND.map((L) => (
              <div key={L.key} className="legend-row">
                <span
                  className="legend-swatch"
                  style={{ background: `rgb(${L.r},${L.g},${L.b})` }}
                />
                <span>{L.key} (field boxes)</span>
              </div>
            ))}
            <p className="legend-note">
              Selected field: white dashed ring (model or saved correction).
              Other corrections: solid category tints. OCR: varied hues.
              Regions: purple. New draw: category dashed box on top.
            </p>
            {fileHash && (
              <>
                <hr />
                <h3>Cached upload</h3>
                <p className="legend-note" style={{ marginBottom: 8 }}>
                  This file is cached server-side keyed by{" "}
                  <code>{fileHash}</code>. The Validation tab can re-run
                  without re-uploading, even after a refresh.
                </p>
                <button
                  type="button"
                  className="btn"
                  onClick={() => {
                    void clearCache();
                  }}
                >
                  Clear cached upload
                </button>
              </>
            )}
          </>
        )}
        {tab === "validation" && (
          <>
            <p className="legend-note" style={{ fontSize: "0.8rem" }}>
              Layout validation compares your uploaded label against the
              approved template (selected by the AI(240) product code or
              picked manually).
            </p>
            <hr />
            <h3>Pipeline summary</h3>
            <ol className="legend-note" style={{ paddingLeft: 14 }}>
              <li>Extract the candidate page + label rect.</li>
              <li>Load the per-product region schema.</li>
              <li>Run typed validators per region.</li>
              <li>Aggregate weighted score → VALID/INVALID.</li>
              <li>Render side-by-side overviews + report.</li>
            </ol>
          </>
        )}
        {tab === "dashboard" && (
          <>
            <p className="legend-note" style={{ fontSize: "0.8rem" }}>
              QA dashboard consolidates extraction + validation for production
              label testers in one clean screen.
            </p>
            <hr />
            <h3>Checklist</h3>
            <ol className="legend-note" style={{ paddingLeft: 14 }}>
              <li>Upload candidate label</li>
              <li>Review extraction confidence</li>
              <li>Run validation against template</li>
              <li>Inspect side-by-side visual evidence</li>
            </ol>
          </>
        )}
      </aside>
      <main className="main">
        <h1>🔬 IKEA Label Validation</h1>
        <p className="subtitle">
          Upload a production label — every element is extracted, validated
          against the label-rules schemas, and checked for spatial compliance.
          Switch to the <strong>Validation</strong> tab to compare the same
          label against the approved template (visual layout report).
        </p>
        <FileDropzone
          onSelectFile={(f) => {
            void runFile(f);
          }}
          busy={busy}
        />
        {err && <p className="alert-err">{err}</p>}

        <nav className="tabbar" role="tablist">
          <button
            type="button"
            role="tab"
            aria-selected={tab === "extractor"}
            className={`tab ${tab === "extractor" ? "tab--active" : ""}`}
            onClick={() => setTab("extractor")}
          >
            Extractor
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={tab === "validation"}
            className={`tab ${tab === "validation" ? "tab--active" : ""}`}
            onClick={() => setTab("validation")}
          >
            Validation
            {detectedProductCode && (
              <span className="tab-badge">{detectedProductCode}</span>
            )}
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={tab === "dashboard"}
            className={`tab ${tab === "dashboard" ? "tab--active" : ""}`}
            onClick={() => setTab("dashboard")}
          >
            QA Dashboard
            {validationResult && (
              <span className="tab-badge">{validationResult.verdict}</span>
            )}
          </button>
        </nav>

        {tab === "extractor" && (
          <>
            {!hasExtraction && !busy && (
              <div className="empty-state">
                <div className="empty-icon" aria-hidden>
                  📄
                </div>
                <div className="empty-title">
                  Drop a label file above to begin
                </div>
              </div>
            )}
            {hasExtraction && preview && label && (
              <ExtractorView
                fields={fields}
                label={label}
                preview={preview}
                fileHash={fileHash}
                dpi={dpi}
                imageWidth={w}
                imageHeight={h}
                corrections={corrections}
                setCorrections={setCorrections}
                validation={valid}
                setValidation={setValid}
                layoutRep={layoutRep}
                setLayoutRep={setLayoutRep}
                ovRep={ovRep}
                setOvRep={setOvRep}
                showOcrWords={showOcrWords}
                showRegions={showRegions}
                showPdfText={showPdfText}
                showClean={showClean}
                showJson={showJson}
                setBusy={setBusy}
                setBusyMessage={setBusyMessage}
                setErr={setErr}
              />
            )}
          </>
        )}

        {tab === "validation" && (
          <ValidationView
            fileBlob={fileBlob}
            fileHash={fileHash || null}
            detectedProductCode={detectedProductCode}
            result={validationResult}
            setResult={setValidationResult}
            setBusy={setBusy}
            setBusyMessage={setBusyMessage}
            setErr={setErr}
          />
        )}

        {tab === "dashboard" && (
          <QADashboardView
            label={label}
            preview={preview}
            validationResult={validationResult}
            fileHash={fileHash}
            detectedProductCode={detectedProductCode}
          />
        )}
      </main>
    </div>
  );
}
