import { useCallback } from 'react';

interface RepeatButtonProps {
  onRepeat: () => void;
  disabled: boolean;
}

export function RepeatButton({ onRepeat, disabled }: RepeatButtonProps) {
  const handleClick = useCallback(() => {
    if (!disabled) {
      onRepeat();
    }
  }, [onRepeat, disabled]);

  return (
    <button
      className="repeat-button"
      onClick={handleClick}
      aria-label="Repeat last utterance"
      aria-disabled={disabled}
      disabled={disabled}
      role="button"
      type="button"
    >
      <span className="button-icon" aria-hidden="true">🔁</span>
      <span className="button-text">Repeat</span>
    </button>
  );
}
