"""
Pipeline interface contracts for the Braille Vision Scanner.

Defines Protocol types for each pipeline stage, specifying the expected
input/output types and method signatures that concrete implementations must satisfy.

Requirements: 8.1, 8.2 - Real-time pipeline performance contracts.
"""

from __future__ import annotations

from typing import AsyncGenerator, Protocol, runtime_checkable

import numpy as np

from backend.models.data_models import (
    AlignmentResult,
    BrailleCell,
    CameraSource,
    CapturedFrame,
    DecodedText,
    DotClassification,
    DotDetectionResult,
    DotPosition,
    PreprocessedImage,
    SegmentationResult,
    ValidationResult,
)


@runtime_checkable
class ImageCaptureProtocol(Protocol):
    """Protocol for the Image Capture Module.

    Responsible for acquiring frames from camera sources or uploaded files.

    Input: Camera device handle or image file path.
    Output: CapturedFrame (numpy array + metadata: timestamp, resolution, sharpness_score).
    """

    def __init__(self, source: CameraSource) -> None:
        """Initialize with webcam, mobile rear, or mobile front camera."""
        ...

    async def capture_frame(self) -> CapturedFrame:
        """Capture a single frame within 200ms. Validates sharpness."""
        ...

    async def start_preview(self) -> AsyncGenerator[CapturedFrame, None]:
        """Yield continuous preview frames at >= 15 FPS."""
        ...

    def check_sharpness(self, frame: np.ndarray) -> float:
        """Return Laplacian variance as sharpness score."""
        ...

    async def upload_image(self, file_path: str) -> CapturedFrame:
        """Accept pre-existing image file as alternative input."""
        ...


@runtime_checkable
class AlignmentGuideProtocol(Protocol):
    """Protocol for the Alignment Guide.

    Responsible for providing real-time positioning feedback to the user.

    Input: Raw camera frame (numpy array).
    Output: AlignmentResult (is_aligned, direction, coverage_percent, braille_detected).
    """

    def analyze_frame(self, frame: np.ndarray) -> AlignmentResult:
        """Analyze frame for Braille text position and coverage.

        Returns direction cues within 500ms.
        """
        ...

    def get_audio_cue(self, result: AlignmentResult) -> str:
        """Convert alignment result to audio instruction string."""
        ...


@runtime_checkable
class PreprocessingPipelineProtocol(Protocol):
    """Protocol for the Preprocessing Pipeline.

    Responsible for normalizing images for consistent dot detection.

    Input: Raw captured frame (numpy array).
    Output: PreprocessedImage (binary numpy array + preprocessing metadata).
    """

    def process(self, frame: np.ndarray) -> PreprocessedImage:
        """Full preprocessing: normalize -> deskew -> denoise -> threshold.

        Completes within 100ms.
        """
        ...

    def normalize_brightness(self, frame: np.ndarray) -> np.ndarray:
        """Adjust to target mean intensity 128 +/- 20."""
        ...

    def correct_rotation(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        """Correct skew up to +/-15 degrees.

        Returns corrected image and angle in degrees.
        """
        ...

    def apply_noise_reduction(self, frame: np.ndarray) -> np.ndarray:
        """Gaussian/median filter with kernel <= 5x5."""
        ...

    def adaptive_threshold(self, frame: np.ndarray) -> np.ndarray:
        """Adaptive thresholding for uneven lighting.

        Produces a binary image where every pixel is 0 or 255.
        """
        ...

    def validate_image(self, frame: np.ndarray) -> ValidationResult:
        """Check if image can be corrected.

        Returns ValidationResult.OK if usable, or an error code if not.
        """
        ...


@runtime_checkable
class DotDetectorProtocol(Protocol):
    """Protocol for the Dot Detector.

    Responsible for identifying individual Braille dot positions in preprocessed images.

    Input: PreprocessedImage (binary image).
    Output: DotDetectionResult (list of DotPosition with x, y, confidence, radius).
    """

    def detect(self, image: PreprocessedImage) -> DotDetectionResult:
        """Detect dot positions with >= 90% precision/recall.

        Completes within 150ms for up to 2000x2000px.
        """
        ...

    def classify_dot(self, region: np.ndarray) -> DotClassification:
        """Classify a candidate region as dot or noise."""
        ...


@runtime_checkable
class CellSegmenterProtocol(Protocol):
    """Protocol for the Cell Segmenter.

    Responsible for grouping detected dots into 2x3 Braille cell grids.

    Input: DotDetectionResult (list of dot positions).
    Output: SegmentationResult (list of BrailleCell objects in reading order, flagged ambiguous regions).
    """

    def segment(self, dots: DotDetectionResult) -> SegmentationResult:
        """Group dots into cells in reading order.

        Completes within 100ms for up to 200 cells.
        """
        ...

    def detect_lines(self, dots: list[DotPosition]) -> list[list[DotPosition]]:
        """Separate dots into lines based on vertical gaps.

        A new line is detected when the vertical gap exceeds 2x intra-cell row spacing.
        """
        ...

    def group_into_cells(self, line_dots: list[DotPosition]) -> list[BrailleCell]:
        """Group a line's dots into individual cells.

        Assigns dots to nearest grid position within 40% of intra-cell spacing.
        """
        ...


@runtime_checkable
class BrailleDecoderProtocol(Protocol):
    """Protocol for the Braille Decoder.

    Responsible for mapping Braille cells to English characters and serializing cell patterns.

    Input: SegmentationResult (ordered list of BrailleCell).
    Output: DecodedText (text string + per-character confidence + unrecognized markers).
    """

    def decode(self, cells: SegmentationResult) -> DecodedText:
        """Decode cells to English text within 50ms per line of 40 cells."""
        ...

    def decode_cell(self, cell: BrailleCell) -> str:
        """Map a single cell to its English character."""
        ...

    def serialize_cell(self, cell: BrailleCell) -> str:
        """Serialize cell to JSON with dot positions and confidence scores."""
        ...

    def deserialize_cell(self, json_str: str) -> BrailleCell:
        """Deserialize JSON back to BrailleCell. Validates structure."""
        ...


@runtime_checkable
class TTSEngineProtocol(Protocol):
    """Protocol for the TTS Engine.

    Responsible for converting decoded text to speech output.

    Input: Decoded English text string.
    Output: Audio speech (via system audio device).
    """

    def __init__(self, rate: int = 150) -> None:
        """Initialize with default 150 WPM."""
        ...

    async def speak(self, text: str) -> None:
        """Speak text within 500ms of receiving it."""
        ...

    def adjust_rate(self, delta: int) -> int:
        """Adjust rate by +/-10 WPM within 80-250 range.

        Returns new rate.
        """
        ...

    async def repeat_last(self) -> None:
        """Re-speak the last fully decoded sentence."""
        ...

    async def queue_text(self, text: str) -> None:
        """Queue text if currently speaking."""
        ...
