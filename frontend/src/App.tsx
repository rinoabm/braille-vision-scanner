import { useState, useCallback, useRef, useEffect } from 'react';
import { useVoiceCommands, VoiceCommand, speakFeedback } from './hooks/useVoiceCommands';
import { useScanWebSocket, ScanResult } from './hooks/useScanWebSocket';
import {
  ScanButton,
  RepeatButton,
  SpeedControl,
  StatusDisplay,
  AudioTutorial,
  UploadButton,
} from './components';

function App() {
  const [scanning, setScanning] = useState(false);
  const [decodedText, setDecodedText] = useState('');
  const [ttsRate, setTtsRate] = useState(150);
  const [statusMessage, setStatusMessage] = useState('Ready');
  const [confidence, setConfidence] = useState<number | null>(null);
  const [audioAvailable, setAudioAvailable] = useState(true);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const captureIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const lastDetectedTextRef = useRef<string>('');
  const consecutiveMatchRef = useRef<number>(0);

  // Check audio availability
  useEffect(() => {
    setAudioAvailable('speechSynthesis' in window);
  }, []);

  const handleScanResult = useCallback((result: ScanResult) => {
    // Only process if the backend confirmed Braille was detected
    if (result.text && result.text.trim()) {
      const conf = result.confidence * 100;
      const cleanText = result.text.trim();
      setConfidence(conf);
      setStatusMessage(`Braille detected - Confidence: ${Math.round(conf)}%`);

      // Debounce: show result immediately (single frame is enough)
      // The backend validation already filters noise
      setDecodedText(cleanText);
      if (audioAvailable) {
        window.speechSynthesis?.cancel();
        speakFeedback(cleanText);
      }
    } else {
      // No Braille detected - reset consecutive counter
      lastDetectedTextRef.current = '';
      consecutiveMatchRef.current = 0;
      setStatusMessage('Scanning... (no Braille detected)');
      setConfidence(null);
    }
  }, [audioAvailable]);

  const handleWsError = useCallback((error: string) => {
    setStatusMessage(`Connection error: ${error}`);
  }, []);

  const { connected, connect, disconnect, sendFrame } = useScanWebSocket({
    onResult: handleScanResult,
    onError: handleWsError,
  });

  const startScanning = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: 'environment' }, width: { ideal: 640 }, height: { ideal: 480 } },
      });
      streamRef.current = stream;

      // Set scanning first so the video element renders
      setScanning(true);
      setStatusMessage('Scanning...');
      speakFeedback('Scan started');
      connect();
    } catch (err) {
      console.error('Camera error:', err);
      setStatusMessage('Camera access denied or unavailable');
      speakFeedback('Camera permission denied. Please grant access in settings.');
    }
  }, [connect]);

  // Attach stream to video element once scanning starts and video is in DOM
  useEffect(() => {
    if (scanning && streamRef.current && videoRef.current) {
      videoRef.current.srcObject = streamRef.current;
      videoRef.current.play().catch(console.error);

      // Start capturing frames and sending via WebSocket
      const canvas = document.createElement('canvas');
      canvas.width = 640;
      canvas.height = 480;
      const ctx = canvas.getContext('2d');

      captureIntervalRef.current = setInterval(() => {
        if (videoRef.current && ctx && videoRef.current.readyState >= 2) {
          // Flip horizontally to undo webcam mirror effect
          ctx.save();
          ctx.scale(-1, 1);
          ctx.drawImage(videoRef.current, -640, 0, 640, 480);
          ctx.restore();
          const base64 = canvas.toDataURL('image/jpeg', 0.7).split(',')[1];
          sendFrame(base64);
        }
      }, 500); // ~2 FPS
    }

    return () => {
      if (captureIntervalRef.current) {
        clearInterval(captureIntervalRef.current);
        captureIntervalRef.current = null;
      }
    };
  }, [scanning, sendFrame]);

  const stopScanning = useCallback(() => {
    if (captureIntervalRef.current) {
      clearInterval(captureIntervalRef.current);
      captureIntervalRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    // Cancel all queued speech immediately
    window.speechSynthesis?.cancel();
    disconnect();
    setScanning(false);
    setStatusMessage('Scan stopped');
    speakFeedback('Scan stopped');
  }, [disconnect]);

  const handleRepeat = useCallback(async () => {
    try {
      await fetch('/api/repeat', { method: 'POST' });
      speakFeedback(decodedText || 'No text to repeat');
    } catch {
      speakFeedback('Unable to repeat');
    }
  }, [decodedText]);

  const adjustSpeed = useCallback(
    async (newRate: number) => {
      setTtsRate(newRate);

      try {
        await fetch('/api/settings', {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ rate: newRate }),
        });
      } catch {
        // Settings update failed silently, local state still updated
      }

      speakFeedback(`Speed set to ${newRate} words per minute`);
    },
    []
  );

  const handleVoiceCommand = useCallback(
    (command: VoiceCommand) => {
      switch (command) {
        case 'start_scan':
          if (!scanning) startScanning();
          break;
        case 'stop_scan':
          if (scanning) stopScanning();
          break;
        case 'repeat':
          handleRepeat();
          break;
        case 'faster':
          adjustSpeed(Math.min(250, ttsRate + 10));
          break;
        case 'slower':
          adjustSpeed(Math.max(80, ttsRate - 10));
          break;
      }
    },
    [scanning, startScanning, stopScanning, handleRepeat, adjustSpeed, ttsRate]
  );

  const { isListening, supported: voiceSupported } = useVoiceCommands({
    onCommand: handleVoiceCommand,
    enabled: true,
  });

  return (
    <div className="app" role="application" aria-label="Braille Vision Scanner">
      <header className="app-header">
        <h1>Braille Vision Scanner</h1>
        <AudioTutorial />
      </header>

      <main className="app-main">
        <section className="camera-section" aria-label="Camera preview">
          <div
            className="camera-preview"
            role="img"
            aria-label={
              scanning
                ? 'Live camera preview showing Braille scanning area'
                : 'Camera preview inactive'
            }
          >
            <video
              ref={videoRef}
              className="camera-video"
              aria-hidden="true"
              playsInline
              muted
              autoPlay
              style={{ display: scanning ? 'block' : 'none' }}
            />
            {!scanning && (
              <div className="camera-placeholder" aria-hidden="true">
                <span className="placeholder-icon">📷</span>
                <span className="placeholder-text">Camera inactive</span>
              </div>
            )}
          </div>
          <span className="visually-hidden" role="status" aria-live="polite">
            {scanning ? 'Camera is active and scanning' : 'Camera is not active'}
          </span>
        </section>

        <StatusDisplay
          decodedText={decodedText}
          statusMessage={statusMessage}
          isScanning={scanning}
          confidence={confidence}
        />

        {!audioAvailable && (
          <p className="audio-fallback" role="alert" aria-live="assertive">
            Audio output unavailable. Text is displayed above.
          </p>
        )}

        <nav
          className="controls"
          role="toolbar"
          aria-label="Scanner controls"
        >
          <div className="primary-controls">
            <ScanButton
              onStartScan={startScanning}
              onStopScan={stopScanning}
              isScanning={scanning}
            />
            <UploadButton
              onUploadResult={(result) => {
                setDecodedText(result.text);
                setConfidence(result.confidence * 100);
                setStatusMessage(`Upload processed - Confidence: ${Math.round(result.confidence * 100)}%`);
                speakFeedback(result.text || 'No text detected');
              }}
              onError={(error) => {
                setStatusMessage(error);
                speakFeedback(error);
              }}
              disabled={scanning}
            />
            <RepeatButton
              onRepeat={handleRepeat}
              disabled={!decodedText}
            />
          </div>

          <div className="settings-controls">
            <SpeedControl
              currentSpeed={ttsRate}
              onSpeedChange={adjustSpeed}
            />
          </div>
        </nav>

        <section className="info-section" aria-label="Status information" role="region">
          <p>
            Voice commands:{' '}
            {voiceSupported ? (
              <span className="voice-status" aria-live="polite">
                {isListening ? '🎤 Listening' : '⏸ Paused'}
              </span>
            ) : (
              <span className="voice-status">Not supported in this browser</span>
            )}
          </p>
          {voiceSupported && (
            <ul className="voice-commands-list" aria-label="Available voice commands">
              <li><strong>"Start scan"</strong> — begin scanning</li>
              <li><strong>"Stop scan"</strong> — stop scanning</li>
              <li><strong>"Repeat"</strong> — hear last result again</li>
              <li><strong>"Faster"</strong> — increase speech speed</li>
              <li><strong>"Slower"</strong> — decrease speech speed</li>
            </ul>
          )}
          <p>
            WebSocket:{' '}
            <span className="ws-status" aria-live="polite">
              {connected ? '🟢 Connected' : '⚪ Disconnected'}
            </span>
          </p>
          <p aria-live="polite">Speech rate: {ttsRate} WPM</p>
        </section>
      </main>

      <footer className="app-footer" role="contentinfo">
        <p aria-label="Application information">
          Braille Vision Scanner — Accessible Braille-to-speech converter
        </p>
      </footer>
    </div>
  );
}

export default App;
