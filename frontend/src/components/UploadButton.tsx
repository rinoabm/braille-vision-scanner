import { useCallback, useRef } from 'react';

interface UploadButtonProps {
  onUploadResult: (result: { text: string; confidence: number }) => void;
  onError: (error: string) => void;
  disabled?: boolean;
}

export function UploadButton({ onUploadResult, onError, disabled }: UploadButtonProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleClick = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  const handleFileChange = useCallback(
    async (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      if (!file) return;

      const formData = new FormData();
      formData.append('file', file);

      try {
        const response = await fetch('/api/upload', {
          method: 'POST',
          body: formData,
        });

        if (!response.ok) {
          onError(`Upload failed: ${response.statusText}`);
          return;
        }

        const data = await response.json();

        if (data.alignment?.error) {
          onError(data.alignment.error);
          return;
        }

        onUploadResult({
          text: data.text || '(no text detected)',
          confidence: data.confidence,
        });
      } catch (err) {
        onError('Upload failed. Is the backend running?');
      }

      // Reset input so same file can be uploaded again
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    },
    [onUploadResult, onError]
  );

  return (
    <>
      <input
        ref={fileInputRef}
        type="file"
        accept="image/jpeg,image/png,image/jpg"
        onChange={handleFileChange}
        style={{ display: 'none' }}
        aria-hidden="true"
      />
      <button
        className="upload-button"
        onClick={handleClick}
        aria-label="Upload a Braille image file for recognition"
        disabled={disabled}
        role="button"
        type="button"
      >
        <span className="button-icon" aria-hidden="true">📁</span>
        <span className="button-text">Upload Image</span>
      </button>
    </>
  );
}
