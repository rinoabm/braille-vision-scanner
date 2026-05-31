import { useEffect, useRef, useState } from 'react';

const TUTORIAL_TEXT =
  'Welcome to Braille Vision Scanner. ' +
  'Available actions: ' +
  'Start scan to begin scanning Braille text. ' +
  'Stop scan to end the current scan. ' +
  'Repeat to hear the last result again. ' +
  'Adjust settings to change speech speed. ' +
  'You can also use voice commands for all actions.';

const TUTORIAL_STORAGE_KEY = 'braille-scanner-tutorial-played';

interface AudioTutorialProps {
  onTutorialComplete?: () => void;
}

export function AudioTutorial({ onTutorialComplete }: AudioTutorialProps) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [hasPlayed, setHasPlayed] = useState(false);
  const utteranceRef = useRef<SpeechSynthesisUtterance | null>(null);

  useEffect(() => {
    const alreadyPlayed = sessionStorage.getItem(TUTORIAL_STORAGE_KEY);
    if (alreadyPlayed) {
      setHasPlayed(true);
      return;
    }

    // Play tutorial on first load
    if ('speechSynthesis' in window) {
      const utterance = new SpeechSynthesisUtterance(TUTORIAL_TEXT);
      utterance.rate = 1.2; // Slightly faster to stay within 30 seconds
      utterance.pitch = 1.0;
      utterance.volume = 1.0;
      utterance.lang = 'en-US';

      utterance.onstart = () => setIsPlaying(true);
      utterance.onend = () => {
        setIsPlaying(false);
        setHasPlayed(true);
        sessionStorage.setItem(TUTORIAL_STORAGE_KEY, 'true');
        onTutorialComplete?.();
      };
      utterance.onerror = () => {
        setIsPlaying(false);
        setHasPlayed(true);
        sessionStorage.setItem(TUTORIAL_STORAGE_KEY, 'true');
        onTutorialComplete?.();
      };

      utteranceRef.current = utterance;

      // Small delay to ensure page is ready
      const timer = setTimeout(() => {
        window.speechSynthesis.speak(utterance);
      }, 500);

      return () => {
        clearTimeout(timer);
        window.speechSynthesis.cancel();
      };
    } else {
      setHasPlayed(true);
      onTutorialComplete?.();
    }
  }, [onTutorialComplete]);

  const handleReplayTutorial = () => {
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(TUTORIAL_TEXT);
      utterance.rate = 1.2;
      utterance.pitch = 1.0;
      utterance.volume = 1.0;
      utterance.lang = 'en-US';
      utterance.onstart = () => setIsPlaying(true);
      utterance.onend = () => setIsPlaying(false);
      utterance.onerror = () => setIsPlaying(false);
      window.speechSynthesis.speak(utterance);
    }
  };

  const handleSkipTutorial = () => {
    window.speechSynthesis.cancel();
    setIsPlaying(false);
    setHasPlayed(true);
    sessionStorage.setItem(TUTORIAL_STORAGE_KEY, 'true');
    onTutorialComplete?.();
  };

  return (
    <div
      className="audio-tutorial"
      role="region"
      aria-label="Audio tutorial"
    >
      {isPlaying && (
        <div
          className="tutorial-active"
          role="alert"
          aria-live="assertive"
        >
          <span className="tutorial-indicator" aria-hidden="true">🔊</span>
          <span>Tutorial playing...</span>
          <button
            className="skip-tutorial-button"
            onClick={handleSkipTutorial}
            aria-label="Skip audio tutorial"
            role="button"
            type="button"
          >
            Skip
          </button>
        </div>
      )}
      {hasPlayed && !isPlaying && (
        <button
          className="replay-tutorial-button"
          onClick={handleReplayTutorial}
          aria-label="Replay audio tutorial"
          role="button"
          type="button"
        >
          <span className="button-icon" aria-hidden="true">ℹ️</span>
          <span className="button-text">Replay Tutorial</span>
        </button>
      )}
    </div>
  );
}
