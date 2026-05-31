"""Data models and enums for the Braille Vision Scanner pipeline.

Contains all dataclasses and enums used across pipeline stages:
Image Capture → Preprocessing → Dot Detection → Cell Segmentation →
Braille Decoding → Text-to-Speech.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class CameraSource(Enum):
    """Supported camera input sources."""

    WEBCAM = "webcam"
    MOBILE_REAR = "mobile_rear"
    MOBILE_FRONT = "mobile_front"


class ValidationResult(Enum):
    """Result codes for image validation in the preprocessing pipeline."""

    OK = "ok"
    ROTATION_EXCEEDED = "rotation_exceeded"
    OVEREXPOSED = "overexposed"
    RESOLUTION_TOO_LOW = "resolution_too_low"


class DotClassification(Enum):
    """Classification result for a candidate dot region."""

    DOT = "dot"
    NOISE = "noise"


@dataclass
class CapturedFrame:
    """A single frame captured from a camera source with metadata."""

    image: np.ndarray
    timestamp: float
    resolution: tuple[int, int]
    sharpness_score: float
    source: CameraSource


@dataclass
class AlignmentResult:
    """Result of analyzing a frame for Braille text alignment."""

    is_aligned: bool
    direction: str  # "centered", "left", "right", "too_close", "too_far"
    coverage_percent: float
    braille_detected: bool


@dataclass
class PreprocessedImage:
    """Output of the preprocessing pipeline with binary image and metadata."""

    binary_image: np.ndarray
    rotation_corrected: float  # degrees corrected
    brightness_adjusted: bool
    original_resolution: tuple[int, int]


@dataclass
class DotPosition:
    """A single detected Braille dot with position and confidence."""

    x: float
    y: float
    confidence: float
    radius: float


@dataclass
class DotDetectionResult:
    """Result of dot detection on a preprocessed image."""

    dots: list[DotPosition]
    low_confidence_regions: list[tuple[float, float, float, float]]  # x, y, w, h
    processing_time_ms: float


@dataclass
class BrailleCell:
    """A single Braille cell with active dot positions and confidence scores."""

    dot_positions: list[int]  # integers 1-8 representing active dots
    confidence_scores: list[float]  # 0.0 to 1.0 per dot
    grid_position: tuple[int, int]  # (line_index, cell_index)
    is_ambiguous: bool = False


@dataclass
class SegmentationResult:
    """Result of grouping detected dots into Braille cells."""

    cells: list[BrailleCell]
    lines: list[list[BrailleCell]]
    ambiguous_regions: list[tuple[float, float, float, float]]
    processing_time_ms: float


@dataclass
class DecodedText:
    """Result of decoding Braille cells into English text."""

    text: str
    per_character_confidence: list[float]
    unrecognized_indices: list[int]
    line_confidences: list[float]


@dataclass
class BrailleCellJSON:
    """JSON serialization schema for BrailleCell."""

    dot_positions: list[int]  # integers in range 1-8
    confidence_scores: list[float]  # decimals in range 0.0-1.0


@dataclass
class TTSState:
    """State of the text-to-speech engine."""

    current_rate: int = 150  # words per minute
    is_speaking: bool = False
    queue: list[str] = field(default_factory=list)
    last_sentence: str = ""
