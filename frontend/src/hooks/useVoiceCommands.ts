import { useEffect, useRef, useCallback, useState } from 'react';

export type VoiceCommand =
  | 'start_scan'
  | 'stop_scan'
  | 'repeat'
  | 'faster'
  | 'slower';

export interface UseVoiceCommandsOptions {
  onCommand: (command: VoiceCommand) => void;
  enabled?: boolean;
}

interface SpeechRecognitionEvent {
  results: SpeechRecognitionResultList;
  resultIndex: number;
}

interface SpeechRecognitionErrorEvent {
  error: string;
}

const AVAILABLE_COMMANDS = 'start scan, stop scan, repeat, faster, slower';

const UNRECOGNIZED_MESSAGE = `Command not recognized. Available commands are: ${AVAILABLE_COMMANDS}.`;

function matchCommand(transcript: string): VoiceCommand | null {
  const normalized = transcript.toLowerCase().trim();

  if (normalized.includes('start scan') || normalized === 'scan') {
    return 'start_scan';
  }
  if (normalized.includes('stop scan') || normalized === 'stop') {
    return 'stop_scan';
  }
  if (normalized.includes('repeat')) {
    return 'repeat';
  }
  if (normalized.includes('faster')) {
    return 'faster';
  }
  if (normalized.includes('slower')) {
    return 'slower';
  }

  return null;
}

function speakFeedback(message: string): void {
  if (window.speechSynthesis && typeof SpeechSynthesisUtterance !== 'undefined') {
    const utterance = new SpeechSynthesisUtterance(message);
    utterance.rate = 1.0;
    utterance.pitch = 1.0;
    window.speechSynthesis.speak(utterance);
  }
}

export function useVoiceCommands({ onCommand, enabled = true }: UseVoiceCommandsOptions) {
  const recognitionRef = useRef<SpeechRecognition | null>(null);
  const [isListening, setIsListening] = useState(false);
  const [supported, setSupported] = useState(true);
  const [lastTranscript, setLastTranscript] = useState('');

  const startListening = useCallback(() => {
    if (recognitionRef.current && enabled) {
      try {
        recognitionRef.current.start();
      } catch {
        // Already started, ignore
      }
    }
  }, [enabled]);

  const stopListening = useCallback(() => {
    if (recognitionRef.current) {
      recognitionRef.current.stop();
      setIsListening(false);
    }
  }, []);

  useEffect(() => {
    const SpeechRecognitionAPI =
      window.SpeechRecognition || window.webkitSpeechRecognition;

    if (!SpeechRecognitionAPI) {
      setSupported(false);
      return;
    }

    const recognition = new SpeechRecognitionAPI();
    recognition.continuous = true;
    recognition.interimResults = false;
    recognition.lang = 'en-US';

    recognition.onstart = () => {
      setIsListening(true);
    };

    recognition.onend = () => {
      // Don't flicker the state — keep showing "Listening" during auto-restart
      // Only set to false if we're actually stopping
      if (!enabled) {
        setIsListening(false);
        return;
      }
      // Auto-restart without flickering the UI
      try {
        setTimeout(() => {
          if (recognitionRef.current) {
            recognitionRef.current.start();
          }
        }, 100);
      } catch {
        setIsListening(false);
      }
    };

    recognition.onresult = (event: SpeechRecognitionEvent) => {
      const lastResult = event.results[event.results.length - 1];
      if (!lastResult.isFinal) return;

      const transcript = lastResult[0].transcript;
      setLastTranscript(transcript);

      const command = matchCommand(transcript);
      if (command) {
        onCommand(command);
      }
      // Silently ignore unrecognized speech — don't spam the user
    };

    recognition.onerror = (event: SpeechRecognitionErrorEvent) => {
      if (event.error === 'not-allowed') {
        setSupported(false);
      }
      // For other errors (no-speech, network), let onend handle restart
    };

    recognitionRef.current = recognition;

    if (enabled) {
      try {
        recognition.start();
      } catch {
        // Ignore
      }
    }

    return () => {
      recognition.stop();
      recognitionRef.current = null;
    };
  }, [enabled, onCommand]);

  return {
    isListening,
    supported,
    lastTranscript,
    startListening,
    stopListening,
  };
}

export { matchCommand, speakFeedback, UNRECOGNIZED_MESSAGE, AVAILABLE_COMMANDS };
