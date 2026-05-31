import { useCallback } from 'react';

interface SpeedControlProps {
  currentSpeed: number;
  onSpeedChange: (newSpeed: number) => void;
  minSpeed?: number;
  maxSpeed?: number;
  step?: number;
}

export function SpeedControl({
  currentSpeed,
  onSpeedChange,
  minSpeed = 80,
  maxSpeed = 250,
  step = 10,
}: SpeedControlProps) {
  const handleSlower = useCallback(() => {
    const newSpeed = Math.max(minSpeed, currentSpeed - step);
    onSpeedChange(newSpeed);
  }, [currentSpeed, minSpeed, step, onSpeedChange]);

  const handleFaster = useCallback(() => {
    const newSpeed = Math.min(maxSpeed, currentSpeed + step);
    onSpeedChange(newSpeed);
  }, [currentSpeed, maxSpeed, step, onSpeedChange]);

  const canSlowDown = currentSpeed > minSpeed;
  const canSpeedUp = currentSpeed < maxSpeed;

  return (
    <div
      className="speed-control"
      role="group"
      aria-label="Speech speed adjustment"
    >
      <span className="speed-label" aria-live="polite" aria-atomic="true">
        Speed: {currentSpeed} words per minute
      </span>
      <div className="speed-buttons">
        <button
          className="speed-button slower"
          onClick={handleSlower}
          aria-label={`Decrease speech speed to ${Math.max(minSpeed, currentSpeed - step)} words per minute`}
          aria-disabled={!canSlowDown}
          disabled={!canSlowDown}
          role="button"
          type="button"
        >
          <span className="button-icon" aria-hidden="true">−</span>
          <span className="button-text">Slower</span>
        </button>
        <button
          className="speed-button faster"
          onClick={handleFaster}
          aria-label={`Increase speech speed to ${Math.min(maxSpeed, currentSpeed + step)} words per minute`}
          aria-disabled={!canSpeedUp}
          disabled={!canSpeedUp}
          role="button"
          type="button"
        >
          <span className="button-icon" aria-hidden="true">+</span>
          <span className="button-text">Faster</span>
        </button>
      </div>
    </div>
  );
}
