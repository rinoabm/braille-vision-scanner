import { useRef, useEffect } from 'react';

interface CameraPreviewProps {
  isActive: boolean;
  streamUrl?: string;
}

export function CameraPreview({ isActive, streamUrl }: CameraPreviewProps) {
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    if (!isActive || !videoRef.current) return;

    let stream: MediaStream | null = null;

    async function startCamera() {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: 'environment', width: 640, height: 480 },
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch {
        // Camera errors are handled by the parent via WebSocket
      }
    }

    if (!streamUrl) {
      startCamera();
    }

    return () => {
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }
    };
  }, [isActive, streamUrl]);

  return (
    <div
      className="camera-preview"
      role="img"
      aria-label={
        isActive
          ? 'Live camera preview showing Braille scanning area'
          : 'Camera preview inactive'
      }
    >
      {isActive ? (
        <video
          ref={videoRef}
          className="camera-video"
          autoPlay
          playsInline
          muted
          aria-hidden="true"
        />
      ) : (
        <div className="camera-placeholder" aria-hidden="true">
          <span className="placeholder-icon">📷</span>
          <span className="placeholder-text">Camera inactive</span>
        </div>
      )}
      <span className="visually-hidden" role="status" aria-live="polite">
        {isActive ? 'Camera is active and scanning' : 'Camera is not active'}
      </span>
    </div>
  );
}
