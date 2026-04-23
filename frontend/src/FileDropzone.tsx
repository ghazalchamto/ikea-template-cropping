import { useRef, useState } from "react";

type Props = {
  onSelectFile: (file: File) => void;
  accept?: string;
  busy?: boolean;
};

export function FileDropzone({ onSelectFile, accept = ".pdf,image/*", busy }: Props) {
  const [drag, setDrag] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const pick = (e: { target: HTMLInputElement }) => {
    const f = e.target.files?.[0];
    if (f) onSelectFile(f);
    e.target.value = "";
  };

  return (
    <div
      className={`file-drop ${drag ? "file-drop--active" : ""}`}
      onDragOver={(e) => {
        e.preventDefault();
        e.stopPropagation();
        setDrag(true);
      }}
      onDragLeave={() => setDrag(false)}
      onDrop={(e) => {
        e.preventDefault();
        e.stopPropagation();
        setDrag(false);
        const f = e.dataTransfer.files?.[0];
        if (f) onSelectFile(f);
      }}
    >
      <div className="file-drop__inner">
        <div className="file-drop__left">
          <div className="file-drop__icon" aria-hidden>
            ☁
          </div>
          <div>
            <div className="file-drop__title">Drag and drop file here</div>
            <div className="file-drop__hint">Limit 200MB per file · PDF, PNG, JPG, JPEG, TIFF, TIF, BMP</div>
          </div>
        </div>
        <div>
          <input
            ref={inputRef}
            type="file"
            accept={accept}
            className="file-drop__input"
            onChange={pick}
            disabled={busy}
          />
          <button
            type="button"
            className="file-drop__browse"
            onClick={() => inputRef.current?.click()}
            disabled={busy}
          >
            Browse files
          </button>
        </div>
      </div>
    </div>
  );
}
