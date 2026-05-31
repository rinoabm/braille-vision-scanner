import { useState, useCallback } from 'react';

interface ScanButtonProps {
  onStartScan: () => void;
  onStopScan: () => void;
  isScanning: boolean;
}

export function ScanButton({ onStartScan, onStopScan, isScanning }: ScanButtonProps) {
  const [pressed, setPressed] = useState(false);

  const handleClick = useCallback(() => {
    setPressed(true);
    if (isScanning) {
      onStopScan();
    } else {
      onStartScan();
    }
    setTimeout(() => setPressed(false), 150);
  }, [isScanning, onStartScan, onStopScan]);

  const label = isScanning ? 'Stop scan' : 'Start scan';
  const description = isScanning
    ? 'Stop the current Braille scanning session'
    : 'Start scanning for Braille text using the camera';

  return (
    <button
      className={`scan-button ${isScanning ? 'scanning' : ''} ${pressed ? 'pressed' : ''}`}
      onClick={handleClick}
      aria-label={label}
      aria-pressed={isScanning}
      aria-describedby="scan-button-desc"
      role="button"
      type="button"
    >
      <span className="button-icon" aria-hidden="true">
        {isScanning ? '⏹' : '▶'}
      </span>
      <span className="button-text">{label}</span>
      <span id="scan-button-desc" className="visually-hidden">
        {description}
      </span>
    </button>
  );
}
