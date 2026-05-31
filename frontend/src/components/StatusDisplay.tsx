interface StatusDisplayProps {
  decodedText: string;
  statusMessage: string;
  isScanning: boolean;
  confidence: number | null;
}

export function StatusDisplay({
  decodedText,
  statusMessage,
  isScanning,
  confidence,
}: StatusDisplayProps) {
  const confidenceLabel =
    confidence !== null
      ? confidence >= 70
        ? 'High confidence'
        : confidence >= 30
          ? 'Low confidence'
          : 'Very low confidence'
      : '';

  return (
    <section
      className="status-display"
      aria-label="Scan results and status"
      role="region"
    >
      <div
        className="status-message"
        role="status"
        aria-live="polite"
        aria-atomic="true"
      >
        <span className="status-indicator" aria-hidden="true">
          {isScanning ? '🔴' : '⚪'}
        </span>
        <span>{statusMessage}</span>
      </div>

      {decodedText && (
        <div className="decoded-text-container">
          <h2 className="decoded-heading">Decoded Text</h2>
          <div
            className="decoded-text"
            role="log"
            aria-live="assertive"
            aria-atomic="false"
            aria-label={`Decoded Braille text. ${confidenceLabel}`}
          >
            {confidence !== null && confidence < 70 && confidence >= 30 && (
              <span className="confidence-warning" aria-label="Low confidence result">
                ⚠ Low confidence:
              </span>
            )}
            <p className="decoded-content">{decodedText}</p>
          </div>
          {confidence !== null && (
            <div
              className="confidence-display"
              aria-label={`Recognition confidence: ${Math.round(confidence)} percent`}
            >
              <span className="confidence-bar">
                <span
                  className="confidence-fill"
                  style={{ width: `${confidence}%` }}
                  role="progressbar"
                  aria-valuenow={Math.round(confidence)}
                  aria-valuemin={0}
                  aria-valuemax={100}
                  aria-label={`Confidence: ${Math.round(confidence)}%`}
                />
              </span>
              <span className="confidence-text">{Math.round(confidence)}%</span>
            </div>
          )}
        </div>
      )}
    </section>
  );
}
