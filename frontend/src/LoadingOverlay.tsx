type Props = {
  message: string;
};

export function LoadingOverlay({ message }: Props) {
  return (
    <div className="loading-overlay" role="status" aria-live="polite" aria-busy="true">
      <div className="loading-overlay__panel">
        <div className="loading-spinner" aria-hidden />
        <p className="loading-overlay__message">{message}</p>
      </div>
    </div>
  );
}
