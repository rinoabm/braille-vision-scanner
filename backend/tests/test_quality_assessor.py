"""Unit tests for the QualityAssessor module.

Tests quality score computation, corrective suggestion generation,
partial occlusion detection, low-confidence prefix logic, and
failure condition detection.

Requirements: 10.1, 10.2, 10.3, 10.4, 10.5
"""

import numpy as np
import pytest

from backend.models.data_models import DecodedText, PreprocessedImage
from backend.pipeline.quality_assessor import (
    CORRECTIVE_SUGGESTIONS,
    FailureReason,
    QualityAssessor,
    QualityAssessmentResult,
    QualityIssue,
)


@pytest.fixture
def assessor():
    """Create a QualityAssessor instance."""
    return QualityAssessor()


@pytest.fixture
def good_quality_image():
    """Create a well-lit, sharp image with good contrast."""
    # Create an image with varied content (good contrast, good brightness)
    img = np.zeros((480, 640), dtype=np.uint8)
    # Add structured content for sharpness
    for y in range(0, 480, 20):
        img[y : y + 10, :] = 200
    # Add some dots for texture
    for x in range(50, 600, 30):
        for y in range(50, 430, 30):
            img[y - 3 : y + 3, x - 3 : x + 3] = 180
    return img


@pytest.fixture
def dark_image():
    """Create a very dark image (low brightness)."""
    return np.full((480, 640), 20, dtype=np.uint8)


@pytest.fixture
def bright_image():
    """Create a very bright image (high brightness)."""
    return np.full((480, 640), 240, dtype=np.uint8)


@pytest.fixture
def blurry_image():
    """Create a blurry image with no sharp edges."""
    img = np.random.randint(100, 150, (480, 640), dtype=np.uint8)
    # Heavy blur to remove all edges
    import cv2

    return cv2.GaussianBlur(img, (31, 31), 10)


@pytest.fixture
def sample_decoded_text():
    """Create a sample DecodedText with multiple lines."""
    return DecodedText(
        text="hello world\nthis is braille\ntest line",
        per_character_confidence=[0.9] * 30,
        unrecognized_indices=[],
        line_confidences=[0.85, 0.50, 0.25],
    )


@pytest.fixture
def sample_preprocessed_image():
    """Create a sample PreprocessedImage with some content."""
    binary = np.zeros((480, 640), dtype=np.uint8)
    # Add some white regions to simulate detected content
    binary[50:200, 50:600] = 255
    binary[250:400, 50:600] = 255
    return PreprocessedImage(
        binary_image=binary,
        rotation_corrected=0.0,
        brightness_adjusted=True,
        original_resolution=(640, 480),
    )


class TestComputeQualityScore:
    """Tests for quality score computation."""

    def test_good_image_high_score(self, assessor, good_quality_image):
        """A well-lit, sharp image should get a high quality score."""
        result = assessor.compute_quality_score(good_quality_image)
        assert result.quality_score > 0.5

    def test_dark_image_low_score(self, assessor, dark_image):
        """A very dark image should get a lower quality score."""
        result = assessor.compute_quality_score(dark_image)
        assert result.quality_score < 0.5

    def test_score_in_valid_range(self, assessor, good_quality_image):
        """Quality score should always be between 0 and 1."""
        result = assessor.compute_quality_score(good_quality_image)
        assert 0.0 <= result.quality_score <= 1.0

    def test_dark_image_detects_low_brightness(self, assessor, dark_image):
        """A dark image should detect low brightness issue."""
        result = assessor.compute_quality_score(dark_image)
        assert QualityIssue.LOW_BRIGHTNESS in result.detected_issues

    def test_bright_image_detects_high_brightness(self, assessor, bright_image):
        """A very bright image should detect high brightness issue."""
        result = assessor.compute_quality_score(bright_image)
        assert QualityIssue.HIGH_BRIGHTNESS in result.detected_issues

    def test_blurry_image_detects_blur(self, assessor, blurry_image):
        """A blurry image should detect blur issue."""
        result = assessor.compute_quality_score(blurry_image)
        assert QualityIssue.BLUR in result.detected_issues

    def test_color_image_handled(self, assessor):
        """Color images should be handled correctly."""
        color_img = np.full((480, 640, 3), 128, dtype=np.uint8)
        result = assessor.compute_quality_score(color_img)
        assert 0.0 <= result.quality_score <= 1.0

    def test_returns_quality_assessment_result(self, assessor, good_quality_image):
        """Should return a QualityAssessmentResult dataclass."""
        result = assessor.compute_quality_score(good_quality_image)
        assert isinstance(result, QualityAssessmentResult)


class TestCorrectiveSuggestions:
    """Tests for corrective suggestion generation."""

    def test_dark_image_suggests_brighter_area(self, assessor, dark_image):
        """Dark image should suggest moving to brighter area or flashlight."""
        result = assessor.compute_quality_score(dark_image)
        assert len(result.corrective_suggestions) > 0
        suggestions_text = " ".join(result.corrective_suggestions)
        assert "brighter" in suggestions_text.lower() or "flashlight" in suggestions_text.lower()

    def test_bright_image_suggests_reduce_lighting(self, assessor, bright_image):
        """Bright image should suggest reducing lighting."""
        result = assessor.compute_quality_score(bright_image)
        assert len(result.corrective_suggestions) > 0
        suggestions_text = " ".join(result.corrective_suggestions)
        assert "reduce" in suggestions_text.lower() or "shade" in suggestions_text.lower()

    def test_blurry_image_suggests_stabilize(self, assessor, blurry_image):
        """Blurry image should suggest stabilizing the device."""
        result = assessor.compute_quality_score(blurry_image)
        assert len(result.corrective_suggestions) > 0
        suggestions_text = " ".join(result.corrective_suggestions)
        assert "stabilize" in suggestions_text.lower() or "steady" in suggestions_text.lower()

    def test_no_issues_no_suggestions(self, assessor, good_quality_image):
        """An image with no issues should have no suggestions."""
        result = assessor.compute_quality_score(good_quality_image)
        # If no issues detected, no suggestions
        if not result.detected_issues:
            assert len(result.corrective_suggestions) == 0

    def test_each_issue_has_suggestion(self, assessor):
        """Every QualityIssue should have a corresponding suggestion."""
        for issue in QualityIssue:
            assert issue in CORRECTIVE_SUGGESTIONS


class TestPartialOcclusion:
    """Tests for partial occlusion detection and reporting."""

    def test_no_occlusion(self, assessor, sample_decoded_text):
        """No occlusion should not trigger a report."""
        should_report, message = assessor.assess_partial_occlusion(
            5.0, sample_decoded_text
        )
        assert should_report is False
        assert message == ""

    def test_partial_occlusion_reported(self, assessor, sample_decoded_text):
        """Occlusion between 10-80% should be reported."""
        should_report, message = assessor.assess_partial_occlusion(
            40.0, sample_decoded_text
        )
        assert should_report is True
        assert "60%" in message  # 100 - 40 = 60% read
        assert "partial" in message.lower()

    def test_high_occlusion_not_partial(self, assessor, sample_decoded_text):
        """Occlusion above 80% should not be reported as partial."""
        should_report, message = assessor.assess_partial_occlusion(
            85.0, sample_decoded_text
        )
        assert should_report is False

    def test_boundary_10_percent(self, assessor, sample_decoded_text):
        """Exactly 10% occlusion should be reported."""
        should_report, message = assessor.assess_partial_occlusion(
            10.0, sample_decoded_text
        )
        assert should_report is True

    def test_boundary_80_percent(self, assessor, sample_decoded_text):
        """Exactly 80% occlusion should be reported."""
        should_report, message = assessor.assess_partial_occlusion(
            80.0, sample_decoded_text
        )
        assert should_report is True

    def test_estimate_occlusion_empty_image(self, assessor):
        """A completely empty binary image should show high occlusion."""
        empty_binary = np.zeros((480, 640), dtype=np.uint8)
        preprocessed = PreprocessedImage(
            binary_image=empty_binary,
            rotation_corrected=0.0,
            brightness_adjusted=True,
            original_resolution=(640, 480),
        )
        original = np.zeros((480, 640), dtype=np.uint8)
        occlusion = assessor.estimate_occlusion(original, preprocessed)
        assert occlusion > 80.0

    def test_estimate_occlusion_full_content(self, assessor):
        """A binary image full of content should show low occlusion."""
        full_binary = np.full((480, 640), 255, dtype=np.uint8)
        preprocessed = PreprocessedImage(
            binary_image=full_binary,
            rotation_corrected=0.0,
            brightness_adjusted=True,
            original_resolution=(640, 480),
        )
        original = np.full((480, 640), 128, dtype=np.uint8)
        occlusion = assessor.estimate_occlusion(original, preprocessed)
        assert occlusion < 10.0


class TestConfidencePrefix:
    """Tests for low-confidence prefix logic."""

    def test_high_confidence_no_prefix(self, assessor):
        """Lines with confidence >= 70% should not get a prefix."""
        decoded = DecodedText(
            text="hello world",
            per_character_confidence=[0.9] * 11,
            unrecognized_indices=[],
            line_confidences=[0.85],
        )
        result = assessor.apply_confidence_prefix(decoded)
        assert result == ["hello world"]

    def test_low_confidence_gets_prefix(self, assessor):
        """Lines with confidence 30-70% should get 'low confidence' prefix."""
        decoded = DecodedText(
            text="hello world",
            per_character_confidence=[0.5] * 11,
            unrecognized_indices=[],
            line_confidences=[0.50],
        )
        result = assessor.apply_confidence_prefix(decoded)
        assert result == ["low confidence: hello world"]

    def test_very_low_confidence_no_prefix(self, assessor):
        """Lines with confidence < 30% should not get the prefix."""
        decoded = DecodedText(
            text="hello world",
            per_character_confidence=[0.2] * 11,
            unrecognized_indices=[],
            line_confidences=[0.20],
        )
        result = assessor.apply_confidence_prefix(decoded)
        assert result == ["hello world"]

    def test_boundary_30_percent_gets_prefix(self, assessor):
        """Exactly 30% confidence should get the prefix."""
        decoded = DecodedText(
            text="test line",
            per_character_confidence=[0.3] * 9,
            unrecognized_indices=[],
            line_confidences=[0.30],
        )
        result = assessor.apply_confidence_prefix(decoded)
        assert result == ["low confidence: test line"]

    def test_boundary_70_percent_no_prefix(self, assessor):
        """Exactly 70% confidence should NOT get the prefix (exclusive upper bound)."""
        decoded = DecodedText(
            text="test line",
            per_character_confidence=[0.7] * 9,
            unrecognized_indices=[],
            line_confidences=[0.70],
        )
        result = assessor.apply_confidence_prefix(decoded)
        assert result == ["test line"]

    def test_multiple_lines_mixed_confidence(self, assessor, sample_decoded_text):
        """Multiple lines with mixed confidence should be handled correctly."""
        result = assessor.apply_confidence_prefix(sample_decoded_text)
        # Line 1: 0.85 -> no prefix
        assert result[0] == "hello world"
        # Line 2: 0.50 -> prefix
        assert result[1] == "low confidence: this is braille"
        # Line 3: 0.25 -> no prefix (below 30%)
        assert result[2] == "test line"

    def test_empty_text(self, assessor):
        """Empty text should return empty list."""
        decoded = DecodedText(
            text="",
            per_character_confidence=[],
            unrecognized_indices=[],
            line_confidences=[],
        )
        result = assessor.apply_confidence_prefix(decoded)
        assert result == []


class TestFailureDetection:
    """Tests for failure condition detection."""

    def test_quality_below_threshold_no_braille(self, assessor):
        """Low quality + no Braille should trigger failure."""
        quality_result = QualityAssessmentResult(
            quality_score=0.2,
            detected_issues=[QualityIssue.BLUR, QualityIssue.LOW_BRIGHTNESS],
            corrective_suggestions=["Stabilize the device."],
        )
        is_failure, reason, message = assessor.detect_failure(
            quality_result=quality_result,
            decoded_text=None,
            braille_detected=False,
            occlusion_percentage=0.0,
        )
        assert is_failure is True
        assert reason == FailureReason.QUALITY_BELOW_THRESHOLD_NO_BRAILLE
        assert "failed" in message.lower()

    def test_quality_below_threshold_with_braille(self, assessor):
        """Low quality but Braille detected should NOT trigger this failure."""
        quality_result = QualityAssessmentResult(
            quality_score=0.2,
            detected_issues=[QualityIssue.BLUR],
            corrective_suggestions=["Stabilize the device."],
        )
        is_failure, reason, message = assessor.detect_failure(
            quality_result=quality_result,
            decoded_text=None,
            braille_detected=True,
            occlusion_percentage=0.0,
        )
        assert is_failure is False

    def test_occlusion_above_80_percent(self, assessor):
        """More than 80% occlusion should trigger failure."""
        quality_result = QualityAssessmentResult(quality_score=0.8)
        is_failure, reason, message = assessor.detect_failure(
            quality_result=quality_result,
            decoded_text=None,
            braille_detected=True,
            occlusion_percentage=85.0,
        )
        assert is_failure is True
        assert reason == FailureReason.OCCLUSION_ABOVE_80_PERCENT
        assert "80%" in message

    def test_occlusion_at_80_percent_no_failure(self, assessor):
        """Exactly 80% occlusion should NOT trigger failure (>80% required)."""
        quality_result = QualityAssessmentResult(quality_score=0.8)
        is_failure, reason, message = assessor.detect_failure(
            quality_result=quality_result,
            decoded_text=None,
            braille_detected=True,
            occlusion_percentage=80.0,
        )
        assert is_failure is False

    def test_all_lines_below_30_confidence(self, assessor):
        """All lines below 30% confidence should trigger failure."""
        quality_result = QualityAssessmentResult(quality_score=0.6)
        decoded = DecodedText(
            text="line1\nline2\nline3",
            per_character_confidence=[0.2] * 15,
            unrecognized_indices=[],
            line_confidences=[0.20, 0.15, 0.25],
        )
        is_failure, reason, message = assessor.detect_failure(
            quality_result=quality_result,
            decoded_text=decoded,
            braille_detected=True,
            occlusion_percentage=0.0,
        )
        assert is_failure is True
        assert reason == FailureReason.ALL_LINES_BELOW_30_CONFIDENCE
        assert "confidence" in message.lower()

    def test_some_lines_above_30_no_failure(self, assessor):
        """If at least one line is >= 30% confidence, no failure."""
        quality_result = QualityAssessmentResult(quality_score=0.6)
        decoded = DecodedText(
            text="line1\nline2",
            per_character_confidence=[0.5] * 10,
            unrecognized_indices=[],
            line_confidences=[0.20, 0.50],
        )
        is_failure, reason, message = assessor.detect_failure(
            quality_result=quality_result,
            decoded_text=decoded,
            braille_detected=True,
            occlusion_percentage=0.0,
        )
        assert is_failure is False

    def test_no_failure_good_conditions(self, assessor, sample_decoded_text):
        """Good conditions should not trigger failure."""
        quality_result = QualityAssessmentResult(quality_score=0.8)
        is_failure, reason, message = assessor.detect_failure(
            quality_result=quality_result,
            decoded_text=sample_decoded_text,
            braille_detected=True,
            occlusion_percentage=20.0,
        )
        assert is_failure is False
        assert reason is None
        assert message == ""


class TestSuggestLowLightAction:
    """Tests for low light suggestion."""

    def test_dark_image_suggests_flashlight(self, assessor, dark_image):
        """A dark image should suggest flashlight or brighter area."""
        suggestion = assessor.suggest_low_light_action(dark_image)
        assert suggestion is not None
        assert "flashlight" in suggestion.lower() or "brighter" in suggestion.lower()

    def test_normal_image_no_suggestion(self, assessor, good_quality_image):
        """A normally lit image should not trigger low light suggestion."""
        suggestion = assessor.suggest_low_light_action(good_quality_image)
        assert suggestion is None
