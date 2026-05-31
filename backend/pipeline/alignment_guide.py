"""Alignment Guide for the Braille Vision Scanner.

Provides real-time positioning feedback to help visually impaired users
align their camera over Braille text. Uses OpenCV contour detection to
identify Braille text regions and compute alignment metrics.

Requirements: 2.1, 2.2, 2.3, 2.4, 2.5
"""

import cv2
import numpy as np

from backend.models.data_models import AlignmentResult


class AlignmentGuide:
    """Analyzes camera frames to provide alignment feedback for Braille scanning.

    Uses contour detection to find Braille text regions, then computes
    coverage percentage and center offset to generate directional cues.
    """

    # Alignment thresholds from requirements
    COVERAGE_THRESHOLD = 70.0  # Minimum coverage percentage for "aligned"
    CENTER_TOLERANCE = 0.10  # 10% tolerance for center offset

    # Area thresholds for filtering contours (relative to frame area)
    MIN_CONTOUR_AREA_RATIO = 0.005  # Minimum contour area as fraction of frame
    MIN_BRAILLE_REGION_RATIO = 0.02  # Minimum combined region to count as Braille

    def analyze_frame(self, frame: np.ndarray) -> AlignmentResult:
        """Analyze frame for Braille text position and coverage.

        Detects Braille text bounding box using contour analysis, computes
        coverage percentage and center offset relative to the frame.

        Args:
            frame: Raw camera frame as numpy array (BGR or grayscale).

        Returns:
            AlignmentResult with alignment status, direction, coverage, and detection flag.
            Completes within 500ms per frame.
        """
        if frame is None or frame.size == 0:
            return AlignmentResult(
                is_aligned=False,
                direction="centered",
                coverage_percent=0.0,
                braille_detected=False,
            )

        frame_height, frame_width = frame.shape[:2]
        frame_area = frame_height * frame_width

        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        # Detect Braille text region using adaptive thresholding and contour detection
        bounding_box = self._detect_braille_region(gray, frame_area)

        if bounding_box is None:
            return AlignmentResult(
                is_aligned=False,
                direction="centered",
                coverage_percent=0.0,
                braille_detected=False,
            )

        # Compute coverage and center offset
        bx, by, bw, bh = bounding_box
        coverage_percent = (bw * bh) / frame_area * 100.0

        # Compute center offset as fraction of frame dimensions
        box_center_x = bx + bw / 2.0
        box_center_y = by + bh / 2.0
        frame_center_x = frame_width / 2.0
        frame_center_y = frame_height / 2.0

        offset_x = (box_center_x - frame_center_x) / frame_width
        offset_y = (box_center_y - frame_center_y) / frame_height

        # Determine direction
        direction = self._compute_direction(offset_x, offset_y, coverage_percent)

        # Check alignment: coverage >= 70% and center within 10% tolerance
        is_aligned = (
            coverage_percent >= self.COVERAGE_THRESHOLD
            and abs(offset_x) <= self.CENTER_TOLERANCE
            and abs(offset_y) <= self.CENTER_TOLERANCE
        )

        if is_aligned:
            direction = "centered"

        return AlignmentResult(
            is_aligned=is_aligned,
            direction=direction,
            coverage_percent=coverage_percent,
            braille_detected=True,
        )

    def get_audio_cue(self, result: AlignmentResult) -> str:
        """Convert alignment result to audio instruction string.

        Args:
            result: AlignmentResult from analyze_frame().

        Returns:
            Human-readable directional instruction string for audio output.
        """
        if not result.braille_detected:
            return "No Braille text detected. Please reposition the device."

        if result.is_aligned:
            return "Aligned. Ready to scan."

        direction_cues = {
            "left": "Move the device to the right.",
            "right": "Move the device to the left.",
            "too_close": "Move the device farther away.",
            "too_far": "Move the device closer.",
            "centered": "Aligned. Ready to scan.",
        }

        return direction_cues.get(result.direction, "Adjust device position.")

    def _detect_braille_region(
        self, gray: np.ndarray, frame_area: int
    ) -> tuple[int, int, int, int] | None:
        """Detect the bounding box of Braille text in a grayscale frame.

        Uses adaptive thresholding and contour detection to find regions
        with dot-like patterns characteristic of Braille text.

        Args:
            gray: Grayscale image as numpy array.
            frame_area: Total frame area in pixels.

        Returns:
            Bounding box (x, y, w, h) of detected Braille region, or None if not found.
        """
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Adaptive thresholding to handle varying lighting
        binary = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=11,
            C=2,
        )

        # Morphological operations to connect nearby dots into regions
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated = cv2.dilate(binary, kernel, iterations=3)

        # Find contours
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        # Filter contours by minimum area
        min_area = frame_area * self.MIN_CONTOUR_AREA_RATIO
        valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]

        if not valid_contours:
            return None

        # Combine all valid contours into a single bounding box
        all_points = np.vstack(valid_contours)
        x, y, w, h = cv2.boundingRect(all_points)

        # Check if the combined region is large enough to be Braille text
        region_area = w * h
        if region_area < frame_area * self.MIN_BRAILLE_REGION_RATIO:
            return None

        return (x, y, w, h)

    def _compute_direction(
        self, offset_x: float, offset_y: float, coverage_percent: float
    ) -> str:
        """Compute the primary direction cue based on offsets and coverage.

        Args:
            offset_x: Horizontal offset as fraction of frame width (-0.5 to 0.5).
            offset_y: Vertical offset as fraction of frame height (-0.5 to 0.5).
            coverage_percent: Percentage of frame covered by Braille region.

        Returns:
            Direction string: "left", "right", "too_close", "too_far", or "centered".
        """
        # Check coverage-based directions first
        if coverage_percent > 90.0:
            return "too_close"
        if coverage_percent < 30.0:
            return "too_far"

        # Check horizontal offset (dominant direction)
        if offset_x < -self.CENTER_TOLERANCE:
            return "left"
        if offset_x > self.CENTER_TOLERANCE:
            return "right"

        # Check vertical offset - map to too_close/too_far
        if offset_y < -self.CENTER_TOLERANCE:
            return "too_far"
        if offset_y > self.CENTER_TOLERANCE:
            return "too_close"

        return "centered"
