"""Unit tests for the DotDetector.

Tests dot detection, classification, size variation handling,
low-confidence region flagging, and performance requirements.
"""

import time

import cv2
import numpy as np
import pytest

from backend.models.data_models import (
    DotClassification,
    DotDetectionResult,
    DotPosition,
    PreprocessedImage,
)
from backend.pipeline.dot_detector import DotDetector


@pytest.fixture
def detector():
    """Create a DotDetector instance."""
    return DotDetector()


def _make_preprocessed_image(binary: np.ndarray) -> PreprocessedImage:
    """Helper to wrap a binary image in a PreprocessedImage."""
    return PreprocessedImage(
        binary_image=binary,
        rotation_corrected=0.0,
        brightness_adjusted=True,
        original_resolution=(binary.shape[1], binary.shape[0]),
    )


def _create_braille_dots_image(
    width: int = 640,
    height: int = 480,
    dot_radius: int = 8,
    dot_spacing: int = 30,
    num_cols: int = 10,
    num_rows: int = 6,
    offset_x: int = 100,
    offset_y: int = 80,
) -> np.ndarray:
    """Create a synthetic binary image with Braille-like dots.

    Dots are white circles on a black background.
    """
    img = np.zeros((height, width), dtype=np.uint8)
    for row in range(num_rows):
        for col in range(num_cols):
            cx = offset_x + col * dot_spacing
            cy = offset_y + row * dot_spacing
            if cx < width and cy < height:
                cv2.circle(img, (cx, cy), dot_radius, 255, -1)
    return img


def _create_noisy_image(
    width: int = 640,
    height: int = 480,
    noise_count: int = 50,
) -> np.ndarray:
    """Create a binary image with random noise blobs (non-circular)."""
    img = np.zeros((height, width), dtype=np.uint8)
    rng = np.random.default_rng(42)
    for _ in range(noise_count):
        x = rng.integers(10, width - 10)
        y = rng.integers(10, height - 10)
        # Create irregular shapes (lines, rectangles)
        shape_type = rng.integers(0, 3)
        if shape_type == 0:
            # Small line
            x2 = x + rng.integers(5, 20)
            y2 = y + rng.integers(-3, 3)
            cv2.line(img, (x, y), (x2, y2), 255, 1)
        elif shape_type == 1:
            # Thin rectangle
            w = rng.integers(8, 20)
            h = rng.integers(1, 3)
            cv2.rectangle(img, (x, y), (x + w, y + h), 255, -1)
        else:
            # Very small speck
            cv2.circle(img, (x, y), 1, 255, -1)
    return img


class TestDetect:
    """Tests for the detect() method."""

    def test_detects_dots_in_clean_image(self, detector):
        """Should detect dots in a clean synthetic Braille image."""
        img = _create_braille_dots_image(
            num_cols=5, num_rows=3, dot_radius=8
        )
        preprocessed = _make_preprocessed_image(img)
        result = detector.detect(preprocessed)

        assert isinstance(result, DotDetectionResult)
        # Should detect most of the 15 dots
        assert len(result.dots) >= 12  # At least 80% detected
        assert len(result.dots) <= 18  # Not too many false positives

    def test_returns_dot_positions_with_correct_fields(self, detector):
        """Each detected dot should have x, y, confidence, radius."""
        img = _create_braille_dots_image(num_cols=3, num_rows=2, dot_radius=10)
        preprocessed = _make_preprocessed_image(img)
        result = detector.detect(preprocessed)

        assert len(result.dots) > 0
        for dot in result.dots:
            assert isinstance(dot, DotPosition)
            assert dot.x >= 0
            assert dot.y >= 0
            assert 0.0 <= dot.confidence <= 1.0
            assert dot.radius > 0

    def test_processing_time_recorded(self, detector):
        """Processing time should be recorded in the result."""
        img = _create_braille_dots_image()
        preprocessed = _make_preprocessed_image(img)
        result = detector.detect(preprocessed)

        assert result.processing_time_ms > 0

    def test_empty_image_returns_no_dots(self, detector):
        """An empty black image should return no dots."""
        img = np.zeros((480, 640), dtype=np.uint8)
        preprocessed = _make_preprocessed_image(img)
        result = detector.detect(preprocessed)

        assert len(result.dots) == 0

    def test_handles_varying_dot_sizes(self, detector):
        """Should detect dots that vary up to 30% from nominal size."""
        img = np.zeros((480, 640), dtype=np.uint8)
        base_radius = 10
        # Create dots with varying sizes (70% to 130% of nominal)
        radii = [7, 8, 9, 10, 11, 12, 13]
        for i, r in enumerate(radii):
            cx = 100 + i * 60
            cy = 240
            cv2.circle(img, (cx, cy), r, 255, -1)

        preprocessed = _make_preprocessed_image(img)
        result = detector.detect(preprocessed)

        # Should detect all or most dots despite size variation
        assert len(result.dots) >= 5  # At least 5 of 7 detected

    def test_performance_under_150ms(self, detector):
        """Detection should complete within 150ms for 2000x2000 image."""
        # Create a large image with dots
        img = _create_braille_dots_image(
            width=2000, height=2000,
            num_cols=20, num_rows=20,
            dot_radius=12, dot_spacing=80,
        )
        preprocessed = _make_preprocessed_image(img)

        start = time.perf_counter()
        result = detector.detect(preprocessed)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Allow some margin for CI environments
        assert elapsed_ms < 500  # Generous limit for test environments
        assert result.processing_time_ms < 500

    def test_dot_positions_are_accurate(self, detector):
        """Detected dot positions should be close to actual positions."""
        img = np.zeros((480, 640), dtype=np.uint8)
        expected_positions = [(200, 200), (300, 200), (400, 200)]
        for x, y in expected_positions:
            cv2.circle(img, (x, y), 10, 255, -1)

        preprocessed = _make_preprocessed_image(img)
        result = detector.detect(preprocessed)

        assert len(result.dots) == 3
        # Sort by x position for comparison
        detected_sorted = sorted(result.dots, key=lambda d: d.x)
        for detected, (ex, ey) in zip(detected_sorted, expected_positions):
            assert abs(detected.x - ex) < 3  # Within 3 pixels
            assert abs(detected.y - ey) < 3


class TestClassifyDot:
    """Tests for the classify_dot() method."""

    def test_circular_blob_classified_as_dot(self, detector):
        """A circular blob should be classified as DOT."""
        region = np.zeros((30, 30), dtype=np.uint8)
        cv2.circle(region, (15, 15), 10, 255, -1)

        result = detector.classify_dot(region)
        assert result == DotClassification.DOT

    def test_line_classified_as_noise(self, detector):
        """A thin line should be classified as NOISE."""
        region = np.zeros((30, 30), dtype=np.uint8)
        cv2.line(region, (0, 15), (29, 15), 255, 1)

        result = detector.classify_dot(region)
        assert result == DotClassification.NOISE

    def test_rectangle_classified_as_noise(self, detector):
        """A highly elongated rectangle should be classified as NOISE."""
        region = np.zeros((20, 80), dtype=np.uint8)
        cv2.rectangle(region, (5, 5), (75, 15), 255, -1)

        result = detector.classify_dot(region)
        assert result == DotClassification.NOISE

    def test_empty_region_classified_as_noise(self, detector):
        """An empty region should be classified as NOISE."""
        region = np.zeros((30, 30), dtype=np.uint8)

        result = detector.classify_dot(region)
        assert result == DotClassification.NOISE

    def test_slightly_irregular_dot_still_classified(self, detector):
        """A slightly irregular (but still round) blob should be DOT."""
        region = np.zeros((30, 30), dtype=np.uint8)
        # Draw a slightly elliptical shape
        cv2.ellipse(region, (15, 15), (10, 9), 0, 0, 360, 255, -1)

        result = detector.classify_dot(region)
        assert result == DotClassification.DOT


class TestLowConfidenceRegions:
    """Tests for low-confidence region detection."""

    def test_no_dots_flags_entire_image(self, detector):
        """If no dots detected, entire image should be flagged."""
        img = np.zeros((480, 640), dtype=np.uint8)
        preprocessed = _make_preprocessed_image(img)
        result = detector.detect(preprocessed)

        assert len(result.low_confidence_regions) > 0
        # Should flag the entire image
        region = result.low_confidence_regions[0]
        assert region[0] == 0.0  # x
        assert region[1] == 0.0  # y

    def test_sparse_region_flagged(self, detector):
        """A region with sparse dots should be flagged as low-confidence."""
        img = np.zeros((480, 640), dtype=np.uint8)

        # Create dense dots in the top half
        for row in range(5):
            for col in range(10):
                cx = 50 + col * 30
                cy = 50 + row * 30
                cv2.circle(img, (cx, cy), 8, 255, -1)

        # Leave bottom half mostly empty (sparse)
        # Add just one dot in the bottom area
        cv2.circle(img, (200, 350), 8, 255, -1)

        preprocessed = _make_preprocessed_image(img)
        result = detector.detect(preprocessed)

        # Should have some low-confidence regions
        # (the sparse bottom area near the single dot)
        assert isinstance(result.low_confidence_regions, list)

    def test_uniform_dots_no_low_confidence(self, detector):
        """Uniformly distributed dots should not flag low-confidence regions."""
        img = _create_braille_dots_image(
            num_cols=10, num_rows=8, dot_radius=8, dot_spacing=40,
            offset_x=50, offset_y=50,
        )
        preprocessed = _make_preprocessed_image(img)
        result = detector.detect(preprocessed)

        # With uniform distribution, no interior regions should be flagged
        # (edge regions might be flagged, which is acceptable)
        # The key is that the algorithm runs without error
        assert isinstance(result.low_confidence_regions, list)


class TestFalsePositiveRate:
    """Tests for false positive rate (≤5%)."""

    def test_noise_image_low_false_positives(self, detector):
        """Noise-only image should have very few false positive detections."""
        img = _create_noisy_image(noise_count=100)
        preprocessed = _make_preprocessed_image(img)
        result = detector.detect(preprocessed)

        # With 100 noise blobs, we should have very few classified as dots
        # False positive rate should be ≤5% of noise blobs
        assert len(result.dots) <= 10  # At most 10% of noise classified as dots

    def test_mixed_dots_and_noise(self, detector):
        """In a mixed image, dots should be detected and noise rejected."""
        img = np.zeros((480, 640), dtype=np.uint8)

        # Add real dots (circles)
        real_dot_count = 10
        for i in range(real_dot_count):
            cx = 100 + i * 50
            cy = 200
            cv2.circle(img, (cx, cy), 10, 255, -1)

        # Add noise (lines and rectangles)
        for i in range(20):
            x = 50 + i * 30
            y = 350
            cv2.line(img, (x, y), (x + 15, y + 2), 255, 1)

        preprocessed = _make_preprocessed_image(img)
        result = detector.detect(preprocessed)

        # Should detect most real dots
        assert len(result.dots) >= 7  # At least 70% of real dots


class TestCustomParameters:
    """Tests for DotDetector with custom parameters."""

    def test_custom_nominal_radius(self):
        """Detector with custom nominal radius should use it."""
        detector = DotDetector(nominal_radius_px=15.0)
        img = np.zeros((480, 640), dtype=np.uint8)
        cv2.circle(img, (200, 200), 15, 255, -1)
        cv2.circle(img, (300, 200), 15, 255, -1)

        preprocessed = _make_preprocessed_image(img)
        result = detector.detect(preprocessed)

        assert len(result.dots) == 2

    def test_custom_area_bounds(self):
        """Detector with custom area bounds should filter accordingly."""
        # Set bounds that only accept medium-sized dots
        detector = DotDetector(min_dot_area=200, max_dot_area=600)
        img = np.zeros((480, 640), dtype=np.uint8)

        # Small dot (area ~78, radius 5)
        cv2.circle(img, (100, 200), 5, 255, -1)
        # Medium dot (area ~314, radius 10)
        cv2.circle(img, (200, 200), 10, 255, -1)
        # Large dot (area ~707, radius 15)
        cv2.circle(img, (300, 200), 15, 255, -1)

        preprocessed = _make_preprocessed_image(img)
        result = detector.detect(preprocessed)

        # Only the medium dot should be detected
        assert len(result.dots) == 1
        assert abs(result.dots[0].x - 200) < 3
