import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { FileDropzone } from "./FileDropzone";
import { LabelStage } from "./LabelStage";
import { LoadingOverlay } from "./LoadingOverlay";
import { getBboxForSelectedField } from "./fieldBbox";
import { getFieldValueFromLabel } from "./fieldValue";
import { countExtractedFields, getRawWords, getRegionBoxes } from "./metrics";

type Field = { id: string; label: string };
type BBox = { x: number; y: number; w: number; h: number };
type CorMap = Record<string, { value: string; bbox: BBox | null }>;
type ValRep = {
  compliance_score?: number;
  label_type_name?: string;
  n_error?: number;
  n_warning?: number;
};
type RepLayout = { n_error?: number; n_warning?: number };
type RepOvl = { n_error?: number; n_warning?: number };

const API = "";
const LEGEND: { key: string; r: number; g: number; b: number }[] = [
  { key: "identity", r: 59, g: 130, b: 246 },
  { key: "barcodes", r: 234, g: 88, b: 12 },
  { key: "address", r: 168, g: 85, b: 247 },
  { key: "compliance", r: 236, g: 72, b: 153 },
  { key: "date", r: 20, g: 184, b: 166 },
  { key: "copy", r: 245, g: 158, b: 11 },
];

export function App() {
  const [fields, setFields] = useState<Field[]>([]);
  const [dpi, setDpi] = useState(600);
  const [busy, setBusy] = useState(false);
  const [busyMessage, setBusyMessage] = useState("Working…");
  const [err, setErr] = useState<string | null>(null);
  const [label, setLabel] = useState<Record<string, unknown> | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [fileHash, setFileHash] = useState("");
  const [w, setW] = useState(0);
  const [h, setH] = useState(0);
  const [corrections, setCorrections] = useState<CorMap>({});
  const [selected, setSelected] = useState("product_name");
  const [valInput, setValInput] = useState("");
  const [draft, setDraft] = useState<BBox | null>(null);
  const [valid, setValid] = useState<ValRep | null>(null);
  const [layoutRep, setLayoutRep] = useState<RepLayout | null>(null);
  const [ovRep, setOvRep] = useState<RepOvl | null>(null);
  const [showOcrWords, setShowOcrWords] = useState(true);
  const [showRegions, setShowRegions] = useState(false);
  const [showPdfText, setShowPdfText] = useState(false);
  const [showClean, setShowClean] = useState(false);
  const [showJson, setShowJson] = useState(false);
  const valRef = useRef("");
  const selectedRef = useRef("product_name");
  const corRef = useRef<CorMap>({});

  useEffect(() => { valRef.current = valInput; }, [valInput]);
  useEffect(() => { selectedRef.current = selected; }, [selected]);
  useEffect(() => { corRef.current = corrections; }, [corrections]);

  useEffect(() => {
    fetch(`${API}/api/fields`)
      .then((r) => r.json())
      .then((d: Field[]) => setFields(d))
      .catch((e) => setErr(String(e)));
  }, []);

  useEffect(() => {
    if (!label) {
      setValInput("");
      valRef.current = "";
      return;
    }
    const fromModel = getFieldValueFromLabel(label, selected);
    const c = corRef.current[selected];
    const t = c?.value ?? fromModel;
    setValInput(t);
    valRef.current = t;
  }, [label, selected]);

  const applyMerge = useCallback(
    async (lab: Record<string, unknown>, cor: CorMap) => {
      const r = await fetch(`${API}/api/merge-validate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ label: lab, corrections: cor }),
      });
      if (!r.ok) throw new Error(await r.text());
      return (await r.json()) as {
        validation: ValRep;
        layout: RepLayout;
        overlap: RepOvl;
        completeness: { n_missing?: number; n_consistency_issues?: number };
      };
    },
    []
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
    const fd = new FormData();
    fd.append("file", f);
    fd.append("dpi", String(dpi));
    try {
      const ex = await fetch(`${API}/api/extract`, { method: "POST", body: fd });
      if (!ex.ok) throw new Error(await ex.text());
      const j = (await ex.json()) as {
        label: Record<string, unknown>;
        file_hash: string;
        preview_png_base64: string | null;
        image_width: number;
        image_height: number;
      };
      setLabel(j.label);
      setFileHash(j.file_hash);
      setW(j.image_width);
      setH(j.image_height);
      if (j.preview_png_base64) {
        setPreview(`data:image/png;base64,${j.preview_png_base64}`);
      } else {
        setPreview(null);
      }
      const m = await applyMerge(j.label, {});
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

  const saveWithoutDraw = useCallback(() => {
    if (!label) return;
    const k = selectedRef.current;
    setCorrections((c) => ({
      ...c,
      [k]: { value: valRef.current, bbox: c[k]?.bbox ?? null },
    }));
  }, [label]);

  const onBox = useCallback(
    (bbox: BBox) => {
      setCorrections((c) => ({
        ...c,
        [selectedRef.current]: { value: valRef.current, bbox },
      }));
    },
    []
  );

  const mergeValidate = async () => {
    if (!label) return;
    setBusyMessage("Re-validating…");
    setBusy(true);
    setErr(null);
    try {
      const m = await applyMerge(label, corrections);
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

  const exportJsonl = async () => {
    if (!label) return;
    setBusyMessage("Exporting training data…");
    setBusy(true);
    setErr(null);
    try {
      const r = await fetch(`${API}/api/export-jsonl`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ file_hash: fileHash, dpi, label, corrections }),
      });
      if (!r.ok) throw new Error(await r.text());
      const j = (await r.json()) as { ndjson: string };
      if (j.ndjson) {
        const a = document.createElement("a");
        a.href = URL.createObjectURL(new Blob([j.ndjson], { type: "application/x-ndjson" }));
        a.download = "finetune.jsonl";
        a.click();
        URL.revokeObjectURL(a.href);
      }
    } catch (x) {
      setErr(x instanceof Error ? x.message : String(x));
    } finally {
      setBusy(false);
      setBusyMessage("Working…");
    }
  };

  const otherBoxes = Object.entries(corrections)
    .filter(([id]) => id !== selected)
    .map(([id, c]) =>
      c.bbox ? { id, bbox: c.bbox, value: c.value } : null
    )
    .filter((x): x is { id: string; bbox: BBox; value: string } => x !== null);

  const selectedFieldBbox = useMemo(
    () =>
      label
        ? getBboxForSelectedField(label, selected, corrections)
        : null,
    [label, selected, corrections]
  );

  const ocrBox = getRawWords(label);
  const regBox = getRegionBoxes(label);
  const em = (label as { extraction_metadata?: { overall_confidence?: number } } | null)?.extraction_metadata;
  const conf = em?.overall_confidence != null ? Math.round(100 * Number(em.overall_confidence)) : 0;
  const fc = countExtractedFields(label);
  const lIssues = (layoutRep?.n_error ?? 0) + (layoutRep?.n_warning ?? 0);
  const oIssues = (ovRep?.n_error ?? 0) + (ovRep?.n_warning ?? 0);

  return (
    <div className="app-root">
      {busy && <LoadingOverlay message={busyMessage} />}
      <aside className="sidebar">
        <h2>⚙️ Options</h2>
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
          Selected field: white dashed ring (model or saved correction). Other corrections: solid
          category tints. OCR: varied hues. Regions: purple. New draw: category dashed box on top.
        </p>
      </aside>
      <main className="main">
        <h1>🔬 IKEA Label Validation</h1>
        <p className="subtitle">
          Upload a production label — every element is extracted, validated against the
          label-rules schemas, and checked for spatial compliance. Human review and drawing:
          this web UI. Backup: <code>streamlit run app.py</code>
        </p>
        <FileDropzone
          onSelectFile={(f) => { void runFile(f); }}
          busy={busy}
        />
        {err && <p className="alert-err">{err}</p>}

        {!preview && !busy && (
          <div className="empty-state">
            <div className="empty-icon" aria-hidden>
              📄
            </div>
            <div className="empty-title">Drop a label file above to begin</div>
          </div>
        )}

        {preview && label && w > 0 && (
          <>
            <div className="scorecard">
              <div className="metric">
                <span>Fields extracted</span>
                <span className="metric__value">
                  {fc.found}/{fc.total}
                </span>
              </div>
              <div className="metric">
                <span>Confidence</span>
                <span className="metric__value">{conf}%</span>
              </div>
              <div className="metric">
                <span>Compliance score</span>
                <span className="metric__value">
                  {valid
                    ? `${Math.round(100 * Number((valid as ValRep).compliance_score ?? 0))}%`
                    : "—"}
                </span>
              </div>
              <div className="metric">
                <span>Field rule errors</span>
                <span className="metric__value">{valid?.n_error ?? "—"}</span>
              </div>
              <div className="metric">
                <span>Layout issues</span>
                <span className="metric__value">{valid ? lIssues : "—"}</span>
              </div>
              <div className="metric">
                <span>Overlaps detected</span>
                <span className="metric__value">{valid ? oIssues : "—"}</span>
              </div>
            </div>
            {showJson && (
              <div className="json-details">
                <strong>Extraction JSON</strong>
                <pre>{JSON.stringify(label, null, 2)}</pre>
              </div>
            )}
            <div className="split">
              <div>
                <div className="label-card">
                  <p className="label-card__title">Label</p>
                  <LabelStage
                    dataUrl={preview}
                    naturalWidth={w}
                    naturalHeight={h}
                    otherBoxes={otherBoxes}
                    onNewBox={onBox}
                    draft={draft}
                    setDraft={setDraft}
                    selectedFieldId={selected}
                    selectedHighlight={selectedFieldBbox}
                    ocrWordBoxes={ocrBox}
                    showOcrWords={showOcrWords || showPdfText}
                    ocrAsPdfStyle={showPdfText}
                    regionBoxes={regBox}
                    showRegions={showRegions}
                    showHumanOverlays={!showClean}
                  />
                  {showClean && (
                    <p className="label-card__title" style={{ marginTop: "0.9rem" }}>
                      Original
                    </p>
                  )}
                  {showClean && (
                    <img
                      src={preview}
                      alt="Clean"
                      style={{ width: "100%", borderRadius: 8, display: "block" }}
                    />
                  )}
                </div>
                <p className="hint-inline">
                  White dashed ring = current field in the dropdown (from extraction or your
                  correction). Other solid boxes = other corrected fields. While dragging, the ring
                  hides until you release. {w}×{h} px.
                </p>
              </div>
              <div className="panel-dark">
                <h3>Extracted elements — human review</h3>
                <label className="label-row">
                  Field
                  <select
                    className="select-field"
                    value={selected}
                    onChange={(e) => setSelected(e.target.value)}
                  >
                    {fields.map((f) => (
                      <option key={f.id} value={f.id}>
                        {f.label}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="label-row">
                  Corrected value
                  <input
                    className="text-field"
                    type="text"
                    value={valInput}
                    onChange={(e) => {
                      setValInput(e.target.value);
                      valRef.current = e.target.value;
                    }}
                  />
                </label>
                <button type="button" className="btn" onClick={saveWithoutDraw} style={{ marginTop: 6 }}>
                  Save text only (no new box)
                </button>
                <h3 style={{ marginTop: 18, fontSize: "0.9rem" }}>Corrections</h3>
                <ul className="corr-list">
                  {Object.keys(corrections).length === 0 && <li>None yet</li>}
                  {Object.entries(corrections).map(([k, v]) => (
                    <li key={k}>
                      <code>{k}</code>: {v.value} — {v.bbox ? JSON.stringify(v.bbox) : "no box"}{" "}
                      <button
                        type="button"
                        className="link-remove"
                        onClick={() => {
                          setCorrections((c) => {
                            const n = { ...c };
                            delete n[k];
                            return n;
                          });
                        }}
                      >
                        remove
                      </button>
                    </li>
                  ))}
                </ul>
                <div style={{ display: "flex", flexDirection: "column", gap: 8, marginTop: 12 }}>
                  <button type="button" className="btn btn-primary" onClick={() => { void mergeValidate(); }}>
                    Re-validate with corrections
                  </button>
                  <button type="button" className="btn" onClick={() => { void exportJsonl(); }}>
                    Export JSONL (server + download)
                  </button>
                </div>
                {valid && (
                  <div style={{ marginTop: 16 }}>
                    <h3 style={{ fontSize: "0.95rem" }}>Validation (merged)</h3>
                    <p className="metric" style={{ marginTop: 8 }}>
                      <span>Compliance</span>
                      <strong>
                        {String(Math.round(100 * Number(valid.compliance_score ?? 0)))}%
                      </strong>
                    </p>
                    <p className="metric">
                      <span>Label type</span>
                      <strong>{String(valid.label_type_name)}</strong>
                    </p>
                    <p className="hint-inline">
                      Errors: {valid.n_error ?? 0} · Warnings: {valid.n_warning ?? 0}
                    </p>
                  </div>
                )}
              </div>
            </div>
          </>
        )}
      </main>
    </div>
  );
}
