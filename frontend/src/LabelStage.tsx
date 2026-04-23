import { useRef, useState, useEffect, useCallback, useMemo } from "react";
import { Image as KonvaImage, Layer, Rect, Stage, Group } from "react-konva";
import type { KonvaEventObject } from "konva/lib/Node";
import type { BBox as SelBBox } from "./fieldBbox";
import {
  boxStyleForField,
  draftBoxStyle,
  ocrWordColor,
  regionOverlayStyle,
  selectedFieldHighlightStyle,
} from "./fieldColors";

const MAX_D = 920;

type BBox = { x: number; y: number; w: number; h: number };

type Props = {
  dataUrl: string;
  naturalWidth: number;
  naturalHeight: number;
  otherBoxes: { id: string; bbox: BBox; value: string }[];
  onNewBox: (bbox: BBox) => void;
  draft: BBox | null;
  setDraft: (b: BBox | null) => void;
  /** Streamlit-style: grey OCR word boxes */
  ocrWordBoxes?: { x: number; y: number; w: number; h: number }[];
  showOcrWords?: boolean;
  /** Slightly different stroke when “PDF text spans” is toggled in UI (same `raw_words` data). */
  ocrAsPdfStyle?: boolean;
  /** Contour / layout regions (Pass 4) */
  regionBoxes?: { x: number; y: number; w: number; h: number }[];
  showRegions?: boolean;
  /** If false, no green/orange/draft (clean image only) */
  showHumanOverlays?: boolean;
  /** Field id for the in-progress box — draft uses the same category color (with emphasis). */
  selectedFieldId?: string;
  /** Bbox for the field selected in the dropdown (correction or model); hidden while drafting. */
  selectedHighlight: SelBBox | null;
};

export function LabelStage({
  dataUrl,
  naturalWidth,
  naturalHeight,
  otherBoxes,
  onNewBox,
  draft,
  setDraft,
  ocrWordBoxes = [],
  showOcrWords = false,
  ocrAsPdfStyle = false,
  regionBoxes = [],
  showRegions = false,
  showHumanOverlays = true,
  selectedFieldId = "product_name",
  selectedHighlight = null,
}: Props) {
  const [image, setImage] = useState<HTMLImageElement | null>(null);
  const [drawW, setDrawW] = useState(1);
  const [drawH, setDrawH] = useState(1);
  const drawRef = useRef({ w: 1, h: 1, natW: 1, natH: 1 });
  const isDrag = useRef(false);
  const startRef = useRef({ x: 0, y: 0 });
  const liveRef = useRef<BBox | null>(null);

  useEffect(() => {
    const im = new window.Image();
    im.crossOrigin = "anonymous";
    im.onload = () => {
      setImage(im);
      const scale = Math.min(1, MAX_D / Math.max(im.naturalWidth, im.naturalHeight));
      const dw = Math.max(1, Math.round(im.naturalWidth * scale));
      const dh = Math.max(1, Math.round(im.naturalHeight * scale));
      setDrawW(dw);
      setDrawH(dh);
      drawRef.current = { w: dw, h: dh, natW: im.naturalWidth, natH: im.naturalHeight };
    };
    im.src = dataUrl;
  }, [dataUrl]);

  const toNat = useCallback((sx: number, sy: number) => {
    const { w, h, natW, natH } = drawRef.current;
    const x = Math.round((sx * natW) / w);
    const y = Math.round((sy * natH) / h);
    return {
      x: Math.max(0, Math.min(natW - 1, x)),
      y: Math.max(0, Math.min(natH - 1, y)),
    };
  }, []);

  const onDown = (e: KonvaEventObject<MouseEvent>) => {
    e.evt.stopPropagation();
    const p = e.target.getStage()?.getPointerPosition();
    if (!p) return;
    isDrag.current = true;
    const n = toNat(p.x, p.y);
    startRef.current = n;
    const b = { x: n.x, y: n.y, w: 1, h: 1 };
    liveRef.current = b;
    setDraft(b);
  };

  const onMove = (e: KonvaEventObject<MouseEvent>) => {
    if (!isDrag.current) return;
    const p = e.target.getStage()?.getPointerPosition();
    if (!p) return;
    const c = toNat(p.x, p.y);
    const s = startRef.current;
    const x0 = Math.min(s.x, c.x);
    const y0 = Math.min(s.y, c.y);
    const x1 = Math.max(s.x, c.x);
    const y1 = Math.max(s.y, c.y);
    const b = { x: x0, y: y0, w: Math.max(1, x1 - x0), h: Math.max(1, y1 - y0) };
    liveRef.current = b;
    setDraft(b);
  };

  const endDrag = useCallback(() => {
    if (!isDrag.current) return;
    isDrag.current = false;
    const b = liveRef.current;
    liveRef.current = null;
    setDraft(null);
    if (b && b.w > 2 && b.h > 2) onNewBox(b);
  }, [onNewBox, setDraft]);

  useEffect(() => {
    const w = () => endDrag();
    window.addEventListener("mouseup", w);
    return () => window.removeEventListener("mouseup", w);
  }, [endDrag]);

  const draftStyle = useMemo(
    () => draftBoxStyle(selectedFieldId),
    [selectedFieldId]
  );
  const focusStyle = useMemo(
    () => selectedFieldHighlightStyle(selectedFieldId),
    [selectedFieldId]
  );

  if (!image || !naturalWidth) {
    return <p style={{ color: "#94a3b8" }}>Loading image…</p>;
  }

  const scaleW = drawW / naturalWidth;
  const scaleH = drawH / naturalHeight;

  return (
    <div
      onMouseLeave={endDrag}
      className="label-stage-wrap"
    >
      <Stage
        width={drawW}
        height={drawH}
        onMouseDown={showHumanOverlays ? onDown : undefined}
        onMouseMove={showHumanOverlays ? onMove : undefined}
        onMouseUp={showHumanOverlays ? endDrag : undefined}
        style={{ cursor: showHumanOverlays ? "crosshair" : "default" }}
      >
        <Layer>
          <KonvaImage image={image} x={0} y={0} width={drawW} height={drawH} />
          {showRegions && regionBoxes.length > 0 && (
            <Group listening={false}>
              {regionBoxes.map((r, i) => (
                <Rect
                  key={`reg-${i}`}
                  x={r.x * scaleW}
                  y={r.y * scaleH}
                  width={r.w * scaleW}
                  height={r.h * scaleH}
                  fill={regionOverlayStyle.fill}
                  stroke={regionOverlayStyle.stroke}
                  strokeWidth={regionOverlayStyle.strokeWidth}
                />
              ))}
            </Group>
          )}
          {showOcrWords && ocrWordBoxes.length > 0 && (
            <Group listening={false}>
              {ocrWordBoxes.map((r, i) => {
                const oc = ocrWordColor(i, ocrAsPdfStyle);
                return (
                <Rect
                  key={`w-${i}`}
                  x={r.x * scaleW}
                  y={r.y * scaleH}
                  width={r.w * scaleW}
                  height={r.h * scaleH}
                  fill={oc.fill}
                  stroke={oc.stroke}
                  strokeWidth={oc.strokeWidth}
                />
                );
              })}
            </Group>
          )}
          {showHumanOverlays && (
            <Group listening={false}>
              {otherBoxes.map((o) => {
                const st = boxStyleForField(o.id);
                return (
                <Rect
                  key={o.id + String(o.bbox.x)}
                  x={o.bbox.x * scaleW}
                  y={o.bbox.y * scaleH}
                  width={o.bbox.w * scaleW}
                  height={o.bbox.h * scaleH}
                  fill={st.fill}
                  stroke={st.stroke}
                  strokeWidth={st.strokeWidth}
                />
                );
              })}
              {selectedHighlight &&
                selectedHighlight.w > 0 &&
                selectedHighlight.h > 0 &&
                !draft && (
                <Rect
                  x={selectedHighlight.x * scaleW}
                  y={selectedHighlight.y * scaleH}
                  width={Math.max(1, selectedHighlight.w * scaleW)}
                  height={Math.max(1, selectedHighlight.h * scaleH)}
                  fill={focusStyle.fill}
                  stroke={focusStyle.stroke}
                  strokeWidth={focusStyle.strokeWidth}
                  shadowBlur={focusStyle.shadowBlur}
                  shadowColor={focusStyle.shadowColor}
                  lineJoin="round"
                  dash={[12, 7]}
                />
              )}
              {draft && draft.w > 0 && (
                <Rect
                  x={draft.x * scaleW}
                  y={draft.y * scaleH}
                  width={Math.max(1, draft.w * scaleW)}
                  height={Math.max(1, draft.h * scaleH)}
                  fill={draftStyle.fill}
                  stroke={draftStyle.stroke}
                  strokeWidth={draftStyle.strokeWidth}
                  dash={[6, 4]}
                />
              )}
            </Group>
          )}
        </Layer>
      </Stage>
    </div>
  );
}
