"""Quality Assessment module for the Braille Vision Scanner.

Evaluates image quality and recognition confidence to provide user feedback:
- Quality score computation from image metrics (brightness, contrast, blur, noise)
- Corrective suggestion generation based on detected quality issues
- Partial occlusion detection and percentage reporting
- Low-confidence prefix logic for lines with 30-70% confidence
- Failure condition detection for unrecoverable scenarios

Requirements: 10.1, 10.2, 10.3, 10.4, 10.5
"""

from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np

from backend.models.data_models import DecodedText, PreprocessedImage


class QualityIssue(Enum):
    """Detected quality issues in the captured image."""

    LOW_BRIGHTNESS = "low_brightness"
    HIGH_BRIGHTNESS = "high_brightness"
    LOW_CONTRAST = "low_contrast"
    BLUR = "blur"
    NOISE = "noise"
    GLARE = "glare"


class FailureReason(Enum):
    """Reasons for recognition failure."""

    QUALITY_BELOW_THRESHOLD_NO_BRAILLE = "quality_below_threshold_no_braille"
    OCCLUSION_ABOVE_80_PERCENT = "occlusion_above_80_percent"
    ALL_LINES_BELOW_30_CONFIDENCE = "all_lines_below_30_confidence"


@dataclass
class QualityAssessmentResult:
    """Result of quality assessment on a captured image."""

    quality_score: float  # 0.0 to 1.0
    detected_issues: list[QualityIssue] = field(default_factory=list)
    corrective_suggestions: list[str] = field(default_factory=list)
    occlusion_percentage: float = 0.0  # 0.0 to 100.0
    is_failure: bool = False
    failure_reason: FailureReason | None = None
    failure_message: str = ""


# Mapping from quality issues to corrective suggestions
CORRECTIVE_SUGGESTIONS: dict[QualityIssue, str] = {
    QualityIssue.LOW_BRIGHTNESS: "Move to a brighter area or enable the device flashlight.",
    QualityIssue.HIGH_BRIGHTNESS: "Reduce lighting or move to shade.",
    QualityIssue.LOW_CONTRAST: "Reposition the page to improve contrast between dots and background.",
    QualityIssue.BLUR: "Stabilize the device and hold it steady.",
    QualityIssue.NOISE: "Reduce glare and ensure even lighting.",
    QualityIssue.GLARE: "Reduce glare by repositioning the page or adjusting the angle.",
}


class QualityAssessor:
    """Assesses image quality and recognition confidence for user feedback.

    Evaluates captured images for brightness, contrast, blur, and noise
    to produce a quality score and actionable corrective suggestions.
    Also handles confidence-based output prefixing and failure detection.
    """

    # Quality thresholds
    MIN_QUALITY_THRESHOLD = 0.3  # Below this, recognition is unreliable

    # Brightness thresholds (on 0-255 scale)
    LOW_BRIGHTNESS_THRESHOLD = 60
    HIGH_BRIGHTNESS_THRESHOLD = 200

    # Contrast threshold (standard deviation of pixel intensities)
    LOW_CONTRAST_THRESHOLD = 30

    # Blur threshold (Laplacian variance)
    BLUR_THRESHOLD = 50.0

    # Noise threshold (high-frequency energy ratio)
    NOISE_THRESHOLD = 0.4

    # Glare threshold (fraction of near-max pixels)
    GLARE_PIXEL_THRESHOLD = 0.15

    # Confidence thresholds
    LOW_CONFIDENCE_MIN = 0.30  # 30%
    LOW_CONFIDENCE_MAX = 0.70  # 70%
    FAILURE_CONFIDENCE_THRESHOLD = 0.30  # 30%

    # Occlusion thresholds
    PARTIAL_OCCLUSION_MIN = 10.0  # 10%
    PARTIAL_OCCLUSION_MAX = 80.0  # 80%
    FAILURE_OCCLUSION_THRESHOLD = 80.0  # >80%

    def compute_quality_score(self, image: np.ndarray) -> QualityAssessmentResult:
        """Compute a quality score from image metrics.

        Analyzes brightness, contrast, blur, and noise levels to produce
        a 0-1 quality score. Also detects specific quality issues and
        generates corrective suggestions.

        Args:
            image: Input image as numpy array (BGR or grayscale).

        Returns:
            QualityAssessmentResult with score, issues, and suggestions.
        """
        gray = self._to_grayscale(image)

        # Compute individual metrics (each 0.0 to 1.0, higher is better)
        brightness_score = self._compute_brightness_score(gray)
        contrast_score = self._compute_contrast_score(gray)
        blur_score = self._compute_blur_score(gray)
        noise_score = self._compute_noise_score(gray)

        # Detect specific issues
        detected_issues = self._detect_issues(gray)

        # Compute overall quality score as weighted average
        quality_score = (
            0.20 * brightness_score
            + 0.25 * contrast_score
            + 0.35 * blur_score
            + 0.20 * noise_score
        )

        # Clamp to [0, 1]
        quality_score = max(0.0, min(1.0, quality_score))

        # Generate corrective suggestions based on detected issues
        suggestions = self._generate_suggestions(detected_issues)

        return QualityAssessmentResult(
            quality_score=quality_score,
            detected_issues=detected_issues,
            corrective_suggestions=suggestions,
        )

    def estimate_occlusion(
        self, image: np.ndarray, preprocessed: PreprocessedImage
    ) -> float:
        """Estimate what percentage of the Braille page is occluded.

        Compares the expected content area with the area that contains
        detectable Braille-like features.

        Args:
            image: Original captured image as numpy array.
            preprocessed: The preprocessed binary image.

        Returns:
            Occlusion percentage (0.0 to 100.0).
        """
        binary = preprocessed.binary_image

        # Divide the image into a grid and check which regions have content
        rows, cols = binary.shape[:2]
        grid_rows = 10
        grid_cols = 10
        cell_h = rows // grid_rows
        cell_w = cols // grid_cols

        if cell_h == 0 or cell_w == 0:
            return 0.0

        total_cells = grid_rows * grid_cols
        empty_cells = 0

        for r in range(grid_rows):
            for c in range(grid_cols):
                y_start = r * cell_h
                y_end = (r + 1) * cell_h
                x_start = c * cell_w
                x_end = (c + 1) * cell_w

                cell_region = binary[y_start:y_end, x_start:x_end]

                # A cell is considered "empty" (occluded) if it has very few
                # white pixels (potential dot content)
                white_ratio = np.sum(cell_region > 0) / cell_region.size
                if white_ratio < 0.01:
                    empty_cells += 1

        occlusion_percentage = (empty_cells / total_cells) * 100.0
        return occlusion_percentage

    def assess_partial_occlusion(
        self, occlusion_percentage: float, decoded_text: DecodedText
    ) -> tuple[bool, str]:
        """Determine if partial occlusion should be reported.

        When between 10% and 80% of the page is occluded, report the
        approximate percentage that was successfully read.

        Args:
            occlusion_percentage: Estimated occlusion (0-100).
            decoded_text: The decoded text result.

        Returns:
            Tuple of (should_report, message).
        """
        if (
            self.PARTIAL_OCCLUSION_MIN
            <= occlusion_percentage
            <= self.PARTIAL_OCCLUSION_MAX
        ):
            read_percentage = 100.0 - occlusion_percentage
            message = (
                f"Partial content detected. Approximately "
                f"{read_percentage:.0f}% of the page was successfully read."
            )
            return True, message

        return False, ""

    def apply_confidence_prefix(self, decoded_text: DecodedText) -> list[str]:
        """Apply low-confidence prefix to lines with 30-70% confidence.

        For each line in the decoded text, if the line confidence is
        between 30% and 70% (inclusive of 30%, exclusive of 70%),
        prepend "low confidence" to the spoken output for that line.

        Args:
            decoded_text: The decoded text with line confidences.

        Returns:
            List of output strings for each line, with prefix applied
            where appropriate.
        """
        lines = decoded_text.text.split("\n") if decoded_text.text else []
        line_confidences = decoded_text.line_confidences

        output_lines: list[str] = []

        for i, line in enumerate(lines):
            if i < len(line_confidences):
                confidence = line_confidences[i]
                if (
                    self.LOW_CONFIDENCE_MIN <= confidence < self.LOW_CONFIDENCE_MAX
                ):
                    output_lines.append(f"low confidence: {line}")
                else:
                    output_lines.append(line)
            else:
                output_lines.append(line)

        return output_lines

    def detect_failure(
        self,
        quality_result: QualityAssessmentResult,
        decoded_text: DecodedText | None,
        braille_detected: bool,
        occlusion_percentage: float,
    ) -> tuple[bool, FailureReason | None, str]:
        """Detect failure conditions that prevent successful recognition.

        Failure conditions:
        1. Quality below threshold AND no Braille content identified
        2. More than 80% of the page is occluded
        3. All decoded lines have confidence below 30%

        Args:
            quality_result: The quality assessment result.
            decoded_text: The decoded text (None if decoding failed).
            braille_detected: Whether any Braille content was detected.
            occlusion_percentage: Estimated occlusion percentage.

        Returns:
            Tuple of (is_failure, failure_reason, failure_message).
        """
        # Condition 1: Quality below threshold with no Braille
        if (
            quality_result.quality_score < self.MIN_QUALITY_THRESHOLD
            and not braille_detected
        ):
            reason = FailureReason.QUALITY_BELOW_THRESHOLD_NO_BRAILLE
            message = (
                "Recognition failed: Image quality is too low and no Braille "
                "content could be identified. "
                + self._format_suggestions(quality_result.corrective_suggestions)
            )
            return True, reason, message

        # Condition 2: More than 80% occluded
        if occlusion_percentage > self.FAILURE_OCCLUSION_THRESHOLD:
            reason = FailureReason.OCCLUSION_ABOVE_80_PERCENT
            message = (
                "Recognition failed: More than 80% of the page is occluded. "
                "Please reposition the page to ensure the Braille text is fully visible."
            )
            return True, reason, message

        # Condition 3: All lines below 30% confidence
        if decoded_text is not None and decoded_text.line_confidences:
            all_below_threshold = all(
                conf < self.FAILURE_CONFIDENCE_THRESHOLD
                for conf in decoded_text.line_confidences
            )
            if all_below_threshold:
                reason = FailureReason.ALL_LINES_BELOW_30_CONFIDENCE
                message = (
                    "Recognition failed: Confidence is too low for all detected "
                    "lines. Please adjust conditions and try again. "
                    + self._format_suggestions(quality_result.corrective_suggestions)
                )
                return True, reason, message

        return False, None, ""

    def suggest_low_light_action(self, image: np.ndarray) -> str | None:
        """Suggest enabling flashlight or moving to brighter area for low light.

        When ambient light is below the minimum for reliable dot detection,
        provides a specific suggestion.

        Args:
            image: Input image as numpy array.

        Returns:
            Suggestion string if low light detected, None otherwise.
        """
        gray = self._to_grayscale(image)
        mean_brightness = float(np.mean(gray))

        if mean_brightness < self.LOW_BRIGHTNESS_THRESHOLD:
            return (
                "Low light detected. Please enable the device flashlight "
                "or move to a brighter area."
            )
        return None

    # --- Private helper methods ---

    def _compute_brightness_score(self, gray: np.ndarray) -> float:
        """Compute brightness quality score (0-1).

        Optimal brightness is around 128. Score decreases as brightness
        moves away from optimal in either direction.
        """
        mean_brightness = float(np.mean(gray))
        # Score is highest at 128, decreases toward 0 or 255
        deviation = abs(mean_brightness - 128.0) / 128.0
        return max(0.0, 1.0 - deviation)

    def _compute_contrast_score(self, gray: np.ndarray) -> float:
        """Compute contrast quality score (0-1).

        Uses standard deviation of pixel intensities. Higher std = better contrast.
        """
        std_dev = float(np.std(gray))
        # Normalize: std of 60+ is considered excellent contrast
        score = min(1.0, std_dev / 60.0)
        return score

    def _compute_blur_score(self, gray: np.ndarray) -> float:
        """Compute blur quality score (0-1).

        Uses Laplacian variance as a measure of image sharpness.
        Higher variance = sharper image = better score.
        """
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = float(laplacian.var())
        # Normalize: variance of 500+ is considered sharp
        score = min(1.0, variance / 500.0)
        return score

    def _compute_noise_score(self, gray: np.ndarray) -> float:
        """Compute noise quality score (0-1).

        Estimates noise by comparing the image with a smoothed version.
        Lower noise difference = better score.
        """
        smoothed = cv2.GaussianBlur(gray, (5, 5), 0)
        diff = cv2.absdiff(gray, smoothed)
        noise_level = float(np.mean(diff))
        # Normalize: noise_level of 0 is perfect, 20+ is very noisy
        score = max(0.0, 1.0 - (noise_level / 20.0))
        return score

    def _detect_issues(self, gray: np.ndarray) -> list[QualityIssue]:
        """Detect specific quality issues in the image."""
        issues: list[QualityIssue] = []

        mean_brightness = float(np.mean(gray))
        std_dev = float(np.std(gray))
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # Check brightness
        if mean_brightness < self.LOW_BRIGHTNESS_THRESHOLD:
            issues.append(QualityIssue.LOW_BRIGHTNESS)
        elif mean_brightness > self.HIGH_BRIGHTNESS_THRESHOLD:
            issues.append(QualityIssue.HIGH_BRIGHTNESS)

        # Check contrast
        if std_dev < self.LOW_CONTRAST_THRESHOLD:
            issues.append(QualityIssue.LOW_CONTRAST)

        # Check blur
        if laplacian_var < self.BLUR_THRESHOLD:
            issues.append(QualityIssue.BLUR)

        # Check for glare (high percentage of near-max pixels)
        glare_pixels = np.sum(gray >= 240)
        total_pixels = gray.size
        if total_pixels > 0 and (glare_pixels / total_pixels) > self.GLARE_PIXEL_THRESHOLD:
            issues.append(QualityIssue.GLARE)

        # Check noise
        smoothed = cv2.GaussianBlur(gray, (5, 5), 0)
        diff = cv2.absdiff(gray, smoothed)
        noise_level = float(np.mean(diff))
        if noise_level > (20.0 * self.NOISE_THRESHOLD):
            issues.append(QualityIssue.NOISE)

        return issues

    def _generate_suggestions(
        self, detected_issues: list[QualityIssue]
    ) -> list[str]:
        """Generate corrective suggestions based on detected issues."""
        suggestions: list[str] = []
        for issue in detected_issues:
            suggestion = CORRECTIVE_SUGGESTIONS.get(issue)
            if suggestion:
                suggestions.append(suggestion)
        return suggestions

    def _format_suggestions(self, suggestions: list[str]) -> str:
        """Format suggestions into a single message string."""
        if not suggestions:
            return "Please adjust conditions before retrying."
        return "Suggestions: " + " ".join(suggestions)

    def _to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale if it's a color image."""
        if len(image.shape) == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
