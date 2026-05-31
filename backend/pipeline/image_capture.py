"""Image Capture Module for the Braille Vision Scanner.

Acquires frames from camera sources (webcam, mobile rear, mobile front)
or accepts uploaded image files. Validates sharpness using Laplacian variance
and provides continuous preview streaming at ≥15 FPS.

Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import AsyncGenerator

import cv2
import numpy as np

from backend.models.data_models import CameraSource, CapturedFrame


# Minimum sharpness threshold (Laplacian variance) for accepting a frame
_DEFAULT_SHARPNESS_THRESHOLD = 100.0

# Target preview frame interval for ≥15 FPS
_PREVIEW_FRAME_INTERVAL = 1.0 / 15.0

# Maximum acquisition time for a single frame capture (200ms)
_MAX_CAPTURE_TIME_MS = 200

# Minimum acceptable resolution
_MIN_RESOLUTION = (640, 480)

# Camera index mapping for OpenCV
_CAMERA_INDEX_MAP = {
    CameraSource.WEBCAM: 0,
    CameraSource.MOBILE_REAR: 0,
    CameraSource.MOBILE_FRONT: 1,
}


class CameraInitializationError(Exception):
    """Raised when the camera fails to initialize."""

    def __init__(self, message: str, audio_message: str):
        super().__init__(message)
        self.audio_message = audio_message


class CameraPermissionError(Exception):
    """Raised when camera permission is denied by the OS."""

    def __init__(self, message: str, audio_message: str):
        super().__init__(message)
        self.audio_message = audio_message


class ImageSharpnessError(Exception):
    """Raised when a captured frame is too blurry."""

    def __init__(self, sharpness_score: float, threshold: float):
        super().__init__(
            f"Image too blurry: sharpness {sharpness_score:.1f} "
            f"below threshold {threshold:.1f}"
        )
        self.sharpness_score = sharpness_score
        self.threshold = threshold
        self.audio_message = "Image is blurry. Please hold steady and try again."


class ImageCaptureModule:
    """Captures frames from camera sources or uploaded image files.

    Supports webcam, mobile rear camera, and mobile front camera sources.
    Validates frame sharpness using Laplacian variance and provides
    continuous preview streaming at ≥15 FPS.
    """

    def __init__(
        self,
        source: CameraSource,
        sharpness_threshold: float = _DEFAULT_SHARPNESS_THRESHOLD,
    ) -> None:
        """Initialize with the specified camera source.

        Args:
            source: Camera source type (webcam, mobile_rear, mobile_front).
            sharpness_threshold: Minimum Laplacian variance for accepting a frame.

        Raises:
            CameraInitializationError: If no camera is detected on the device.
            CameraPermissionError: If the OS denies camera access.
        """
        self._source = source
        self._sharpness_threshold = sharpness_threshold
        self._capture: cv2.VideoCapture | None = None
        self._is_initialized = False

    @property
    def source(self) -> CameraSource:
        """The configured camera source."""
        return self._source

    @property
    def is_initialized(self) -> bool:
        """Whether the camera has been successfully initialized."""
        return self._is_initialized

    def initialize(self) -> None:
        """Initialize the camera device.

        Raises:
            CameraInitializationError: If no camera is detected.
            CameraPermissionError: If camera permission is denied.
        """
        camera_index = _CAMERA_INDEX_MAP.get(self._source, 0)

        try:
            self._capture = cv2.VideoCapture(camera_index)
        except Exception as e:
            raise CameraPermissionError(
                f"Camera permission denied for {self._source.value}: {e}",
                audio_message=(
                    "Camera permission denied. "
                    "Please grant camera access in your device settings."
                ),
            )

        if self._capture is None or not self._capture.isOpened():
            raise CameraInitializationError(
                f"No camera detected for source: {self._source.value}",
                audio_message=(
                    "No camera available. You can upload an image instead."
                ),
            )

        # Set resolution to at least 640x480
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, max(640, _MIN_RESOLUTION[0]))
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, max(480, _MIN_RESOLUTION[1]))

        self._is_initialized = True

    def release(self) -> None:
        """Release the camera device."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self._is_initialized = False

    def check_sharpness(self, frame: np.ndarray) -> float:
        """Compute sharpness score using Laplacian variance.

        The Laplacian operator highlights regions of rapid intensity change.
        The variance of the Laplacian response indicates overall image sharpness:
        higher values mean sharper images.

        Args:
            frame: Input image as a numpy array (BGR or grayscale).

        Returns:
            Laplacian variance as a float sharpness score.
        """
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = float(laplacian.var())
        return variance

    async def capture_frame(self) -> CapturedFrame:
        """Capture a single frame within 200ms and validate sharpness.

        Acquires a frame from the initialized camera, checks that it meets
        the minimum sharpness threshold, and returns it with metadata.

        Returns:
            CapturedFrame with image data, timestamp, resolution, and sharpness score.

        Raises:
            CameraInitializationError: If camera is not initialized.
            ImageSharpnessError: If the captured frame is too blurry.
            RuntimeError: If frame acquisition exceeds 200ms or fails.
        """
        if not self._is_initialized or self._capture is None:
            raise CameraInitializationError(
                "Camera not initialized. Call initialize() first.",
                audio_message="Camera is not ready. Please wait and try again.",
            )

        start_time = time.perf_counter()

        # Acquire frame
        ret, frame = self._capture.read()

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        if not ret or frame is None:
            raise RuntimeError(
                "Failed to acquire frame from camera. "
                "The device may have been disconnected."
            )

        if elapsed_ms > _MAX_CAPTURE_TIME_MS:
            raise RuntimeError(
                f"Frame acquisition took {elapsed_ms:.0f}ms, "
                f"exceeding the {_MAX_CAPTURE_TIME_MS}ms limit."
            )

        # Validate sharpness
        sharpness = self.check_sharpness(frame)
        if sharpness < self._sharpness_threshold:
            raise ImageSharpnessError(sharpness, self._sharpness_threshold)

        height, width = frame.shape[:2]
        timestamp = time.time()

        return CapturedFrame(
            image=frame,
            timestamp=timestamp,
            resolution=(width, height),
            sharpness_score=sharpness,
            source=self._source,
        )

    async def start_preview(self) -> AsyncGenerator[CapturedFrame, None]:
        """Yield continuous preview frames at ≥15 FPS.

        Generates a stream of captured frames suitable for live preview display.
        Frames are yielded at a target rate of 15+ FPS. Sharpness is computed
        but frames are not rejected based on sharpness (preview is informational).

        Yields:
            CapturedFrame objects at ≥15 FPS.

        Raises:
            CameraInitializationError: If camera is not initialized.
        """
        if not self._is_initialized or self._capture is None:
            raise CameraInitializationError(
                "Camera not initialized. Call initialize() first.",
                audio_message="Camera is not ready. Please wait and try again.",
            )

        while True:
            frame_start = time.perf_counter()

            ret, frame = self._capture.read()
            if not ret or frame is None:
                # Brief pause before retrying on transient failure
                await asyncio.sleep(0.01)
                continue

            height, width = frame.shape[:2]
            sharpness = self.check_sharpness(frame)
            timestamp = time.time()

            yield CapturedFrame(
                image=frame,
                timestamp=timestamp,
                resolution=(width, height),
                sharpness_score=sharpness,
                source=self._source,
            )

            # Maintain ≥15 FPS by sleeping for the remainder of the frame interval
            elapsed = time.perf_counter() - frame_start
            sleep_time = _PREVIEW_FRAME_INTERVAL - elapsed
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    async def upload_image(self, file_path: str) -> CapturedFrame:
        """Accept a pre-existing image file as alternative input.

        Loads an image from disk, validates it meets minimum resolution,
        computes sharpness, and returns it as a CapturedFrame.

        Args:
            file_path: Path to the image file (JPEG or PNG).

        Returns:
            CapturedFrame with the loaded image and metadata.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file cannot be read as an image or resolution is too low.
            ImageSharpnessError: If the image is too blurry.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Image file not found: {file_path}"
            )

        # Load image using OpenCV
        frame = cv2.imread(file_path)
        if frame is None:
            raise ValueError(
                f"Cannot read image file: {file_path}. "
                "Ensure it is a valid JPEG or PNG image."
            )

        height, width = frame.shape[:2]

        # Validate minimum resolution (640x480)
        if width < _MIN_RESOLUTION[0] or height < _MIN_RESOLUTION[1]:
            raise ValueError(
                f"Image resolution {width}x{height} is below the minimum "
                f"required resolution of {_MIN_RESOLUTION[0]}x{_MIN_RESOLUTION[1]}."
            )

        # Validate sharpness
        sharpness = self.check_sharpness(frame)
        if sharpness < self._sharpness_threshold:
            raise ImageSharpnessError(sharpness, self._sharpness_threshold)

        timestamp = time.time()

        return CapturedFrame(
            image=frame,
            timestamp=timestamp,
            resolution=(width, height),
            sharpness_score=sharpness,
            source=self._source,
        )
