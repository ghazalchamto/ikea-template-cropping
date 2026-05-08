import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { LabelStage } from "./LabelStage";
import { getBboxForSelectedField } from "./fieldBbox";
import { getFieldValueFromLabel } from "./fieldValue";
import { countExtractedFields, getRawWords, getRegionBoxes } from "./metrics";
import {
  postMergeValidate,
  postExportJsonl,
  type BBox,
  type CorMap,
  type Field,
  type RepLayout,
  type RepOvl,
  type ValRep,
} from "./api";

type Props = {
  fields: Field[];
  label: Record<string, unknown>;
  preview: string;
  fileHash: string;
  dpi: number;
  imageWidth: number;
  imageHeight: number;
  corrections: CorMap;
  setCorrections: React.Dispatch<React.SetStateAction<CorMap>>;
  validation: ValRep | null;
  setValidation: (v: ValRep | null) => void;
  layoutRep: RepLayout | null;
  setLayoutRep: (v: RepLayout | null) => void;
  ovRep: RepOvl | null;
  setOvRep: (v: RepOvl | null) => void;
  showOcrWords: boolean;
  showRegions: boolean;
  showPdfText: boolean;
  showClean: boolean;
  showJson: boolean;
  setBusy: (b: boolean) => void;
  setBusyMessage: (m: string) => void;
  setErr: (e: string | null) => void;
};

export function ExtractorView(props: Props) {
  const {
    fields,
    label,
    preview,
    fileHash,
    dpi,
    imageWidth: w,
    imageHeight: h,
    corrections,
    setCorrections,
    validation: valid,
    setValidation,
    layoutRep,
    setLayoutRep,
    ovRep,
    setOvRep,
    showOcrWords,
    showRegions,
    showPdfText,
    showClean,
    showJson,
    setBusy,
    setBusyMessage,
    setErr,
  } = props;

  const [selected, setSelected] = useState("product_name");
  const [valInput, setValInput] = useState("");
  const [draft, setDraft] = useState<BBox | null>(null);
  const valRef = useRef("");
  const selectedRef = useRef("product_name");
  const corRef = useRef<CorMap>({});

  useEffect(() => {
    valRef.current = valInput;
  }, [valInput]);
  useEffect(() => {
    selectedRef.current = selected;
  }, [selected]);
  useEffect(() => {
    corRef.current = corrections;
  }, [corrections]);

  useEffect(() => {
    const fromModel = getFieldValueFromLabel(label, selected);
    const c = corRef.current[selected];
    const t = c?.value ?? fromModel;
    setValInput(t);
    valRef.current = t;
  }, [label, selected]);

  const saveWithoutDraw = useCallback(() => {
    const k = selectedRef.current;
    setCorrections((c) => ({
      ...c,
      [k]: { value: valRef.current, bbox: c[k]?.bbox ?? null },
    }));
  }, [setCorrections]);

  const onBox = useCallback(
    (bbox: BBox) => {
      setCorrections((c) => ({
        ...c,
        [selectedRef.current]: { value: valRef.current, bbox },
      }));
    },
    [setCorrections]
  );

  const mergeValidate = async () => {
    setBusyMessage("Re-validating…");
    setBusy(true);
    setErr(null);
    try {
      const m = await postMergeValidate(label, corrections);
      setValidation(m.validation);
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
    setBusyMessage("Exporting training data…");
    setBusy(true);
    setErr(null);
    try {
      const j = await postExportJsonl(fileHash, dpi, label, corrections);
      if (j.ndjson) {
        const a = document.createElement("a");
        a.href = URL.createObjectURL(
          new Blob([j.ndjson], { type: "application/x-ndjson" })
        );
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
    .filter(
      (x): x is { id: string; bbox: BBox; value: string } => x !== null
    );

  const selectedFieldBbox = useMemo(
    () => getBboxForSelectedField(label, selected, corrections),
    [label, selected, corrections]
  );

  const ocrBox = getRawWords(label);
  const regBox = getRegionBoxes(label);
  const em = (
    label as { extraction_metadata?: { overall_confidence?: number } } | null
  )?.extraction_metadata;
  const conf =
    em?.overall_confidence != null
      ? Math.round(100 * Number(em.overall_confidence))
      : 0;
  const fc = countExtractedFields(label);
  const lIssues = (layoutRep?.n_error ?? 0) + (layoutRep?.n_warning ?? 0);
  const oIssues = (ovRep?.n_error ?? 0) + (ovRep?.n_warning ?? 0);

  return (
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
              ? `${Math.round(100 * Number(valid.compliance_score ?? 0))}%`
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
              <p
                className="label-card__title"
                style={{ marginTop: "0.9rem" }}
              >
                Original
              </p>
            )}
            {showClean && (
              <img
                src={preview}
                alt="Clean"
                style={{
                  width: "100%",
                  borderRadius: 8,
                  display: "block",
                }}
              />
            )}
          </div>
          <p className="hint-inline">
            White dashed ring = current field in the dropdown (from extraction
            or your correction). Other solid boxes = other corrected fields.
            While dragging, the ring hides until you release. {w}×{h} px.
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
          <button
            type="button"
            className="btn"
            onClick={saveWithoutDraw}
            style={{ marginTop: 6 }}
          >
            Save text only (no new box)
          </button>
          <h3 style={{ marginTop: 18, fontSize: "0.9rem" }}>Corrections</h3>
          <ul className="corr-list">
            {Object.keys(corrections).length === 0 && <li>None yet</li>}
            {Object.entries(corrections).map(([k, v]) => (
              <li key={k}>
                <code>{k}</code>: {v.value} —{" "}
                {v.bbox ? JSON.stringify(v.bbox) : "no box"}{" "}
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
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              gap: 8,
              marginTop: 12,
            }}
          >
            <button
              type="button"
              className="btn btn-primary"
              onClick={() => {
                void mergeValidate();
              }}
            >
              Re-validate with corrections
            </button>
            <button
              type="button"
              className="btn"
              onClick={() => {
                void exportJsonl();
              }}
            >
              Export JSONL (server + download)
            </button>
          </div>
          {valid && (
            <div style={{ marginTop: 16 }}>
              <h3 style={{ fontSize: "0.95rem" }}>Validation (merged)</h3>
              <p className="metric" style={{ marginTop: 8 }}>
                <span>Compliance</span>
                <strong>
                  {String(
                    Math.round(100 * Number(valid.compliance_score ?? 0))
                  )}
                  %
                </strong>
              </p>
              <p className="metric">
                <span>Label type</span>
                <strong>{String(valid.label_type_name)}</strong>
              </p>
              <p className="hint-inline">
                Errors: {valid.n_error ?? 0} · Warnings:{" "}
                {valid.n_warning ?? 0}
              </p>
            </div>
          )}
        </div>
      </div>
    </>
  );
}
