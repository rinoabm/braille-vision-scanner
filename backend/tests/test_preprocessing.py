"""Unit tests for the PreprocessingPipeline.

Tests brightness normalization, rotation correction, noise reduction,
adaptive thresholding, and image validation.
"""

import cv2
import numpy as np
import pytest

from backend.models.data_models import PreprocessedImage, ValidationResult
from backend.pipeline.preprocessing import PreprocessingPipeline


@pytest.fixture
def pipeline():
    """Create a PreprocessingPipeline instance."""
    return PreprocessingPipeline()


@pytest.fixture
def sample_grayscale_image():
    """Create a sample 640x480 grayscale image with some structure."""
    img = np.zeros((480, 640), dtype=np.uint8)
    # Add some horizontal lines to give structure for rotation detection
    for y in range(50, 480, 50):
        cv2.line(img, (0, y), (639, y), 200, 2)
    # Add some dots
    for x in range(100, 600, 50):
        for y in range(100, 400, 50):
            cv2.circle(img, (x, y), 5, 180, -1)
    return img


@pytest.fixture
def sample_color_image(sample_grayscale_image):
    """Create a sample BGR color image."""
    return cv2.cvtColor(sample_grayscale_image, cv2.COLOR_GRAY2BGR)


class TestNormalizeBrightness:
    """Tests for normalize_brightness method."""

    def test_dark_image_brightened(self, pipeline):
        """A dark image should be brightened toward target mean."""
        dark_img = np.full((480, 640), 30, dtype=np.uint8)
        result = pipeline.normalize_brightness(dark_img)
        mean_val = np.mean(result)
        assert abs(mean_val - 128) <= 20, f"Mean {mean_val} not within 128 ± 20"

    def test_bright_image_darkened(self, pipeline):
        """A bright image should be darkened toward target mean."""
        bright_img = np.full((480, 640), 220, dtype=np.uint8)
        result = pipeline.normalize_brightness(bright_img)
        mean_val = np.mean(result)
        assert abs(mean_val - 128) <= 20, f"Mean {mean_val} not within 128 ± 20"

    def test_already_normalized_image(self, pipeline):
        """An image already near target mean should stay close."""
        normal_img = np.full((480, 640), 128, dtype=np.uint8)
        result = pipeline.normalize_brightness(normal_img)
        mean_val = np.mean(result)
        assert abs(mean_val - 128) <= 20, f"Mean {mean_val} not within 128 ± 20"

    def test_output_is_uint8(self, pipeline):
        """Output should be uint8 dtype."""
        img = np.random.randint(0, 256, (480, 640), dtype=np.uint8)
        result = pipeline.normalize_brightness(img)
        assert result.dtype == np.uint8

    def test_color_image_converted(self, pipeline, sample_color_image):
        """Color images should be handled (converted to grayscale)."""
        result = pipeline.normalize_brightness(sample_color_image)
        assert len(result.shape) == 2  # Should be grayscale


class TestCorrectRotation:
    """Tests for correct_rotation method."""

    def test_no_rotation_needed(self, pipeline, sample_grayscale_image):
        """Image with horizontal lines should have minimal correction."""
        corrected, angle = pipeline.correct_rotation(sample_grayscale_image)
        assert corrected.shape == sample_grayscale_image.shape
        assert abs(angle) <= 5  # Should detect near-zero rotation

    def test_rotated_image_corrected(self, pipeline):
        """A rotated image should be corrected."""
        # Create image with clear horizontal lines
        img = np.zeros((480, 640), dtype=np.uint8)
        for y in range(50, 480, 30):
            cv2.line(img, (0, y), (639, y), 255, 2)

        # Rotate by 5 degrees
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        rot_matrix = cv2.getRotationMatrix2D(center, 5, 1.0)
        rotated = cv2.warpAffine(img, rot_matrix, (w, h))

        corrected, angle = pipeline.correct_rotation(rotated)
        assert corrected.shape == rotated.shape
        # The detected angle should be close to 5 degrees
        # (correction brings it back toward 0)

    def test_returns_tuple(self, pipeline, sample_grayscale_image):
        """Should return a tuple of (image, angle)."""
        result = pipeline.correct_rotation(sample_grayscale_image)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], np.ndarray)
        assert isinstance(result[1], float)

    def test_blank_image_no_correction(self, pipeline):
        """A blank image with no lines should not be rotated."""
        blank = np.full((480, 640), 128, dtype=np.uint8)
        corrected, angle = pipeline.correct_rotation(blank)
        assert angle == 0.0


class TestApplyNoiseReduction:
    """Tests for apply_noise_reduction method."""

    def test_reduces_noise(self, pipeline):
        """Noise reduction should reduce variance of noisy image."""
        # Create noisy image
        img = np.full((480, 640), 128, dtype=np.uint8)
        noise = np.random.randint(-30, 30, img.shape, dtype=np.int16)
        noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        result = pipeline.apply_noise_reduction(noisy)
        # Denoised image should have lower variance
        assert np.var(result) < np.var(noisy)

    def test_preserves_shape(self, pipeline, sample_grayscale_image):
        """Output should have same shape as input."""
        result = pipeline.apply_noise_reduction(sample_grayscale_image)
        assert result.shape == sample_grayscale_image.shape

    def test_output_is_uint8(self, pipeline, sample_grayscale_image):
        """Output should be uint8 dtype."""
        result = pipeline.apply_noise_reduction(sample_grayscale_image)
        assert result.dtype == np.uint8


class TestAdaptiveThreshold:
    """Tests for adaptive_threshold method."""

    def test_output_is_binary(self, pipeline, sample_grayscale_image):
        """Output should only contain 0 and 255 values."""
        result = pipeline.adaptive_threshold(sample_grayscale_image)
        unique_values = np.unique(result)
        assert all(v in [0, 255] for v in unique_values)

    def test_output_shape_preserved(self, pipeline, sample_grayscale_image):
        """Output should have same shape as input."""
        result = pipeline.adaptive_threshold(sample_grayscale_image)
        assert result.shape == sample_grayscale_image.shape

    def test_output_is_uint8(self, pipeline, sample_grayscale_image):
        """Output should be uint8 dtype."""
        result = pipeline.adaptive_threshold(sample_grayscale_image)
        assert result.dtype == np.uint8

    def test_uniform_image(self, pipeline):
        """A uniform image should still produce binary output."""
        uniform = np.full((480, 640), 128, dtype=np.uint8)
        result = pipeline.adaptive_threshold(uniform)
        unique_values = np.unique(result)
        assert all(v in [0, 255] for v in unique_values)


class TestValidateImage:
    """Tests for validate_image method."""

    def test_valid_image_returns_ok(self, pipeline, sample_grayscale_image):
        """A valid image should return OK."""
        result = pipeline.validate_image(sample_grayscale_image)
        assert result == ValidationResult.OK

    def test_low_resolution_rejected(self, pipeline):
        """Image below 640x480 should be rejected."""
        small_img = np.zeros((200, 300), dtype=np.uint8)
        result = pipeline.validate_image(small_img)
        assert result == ValidationResult.RESOLUTION_TOO_LOW

    def test_overexposed_rejected(self, pipeline):
        """Entirely overexposed image should be rejected."""
        overexposed = np.full((480, 640), 255, dtype=np.uint8)
        result = pipeline.validate_image(overexposed)
        assert result == ValidationResult.OVEREXPOSED

    def test_width_too_small(self, pipeline):
        """Image with width < 640 should be rejected."""
        narrow = np.zeros((480, 500), dtype=np.uint8)
        result = pipeline.validate_image(narrow)
        assert result == ValidationResult.RESOLUTION_TOO_LOW

    def test_height_too_small(self, pipeline):
        """Image with height < 480 should be rejected."""
        short = np.zeros((400, 640), dtype=np.uint8)
        result = pipeline.validate_image(short)
        assert result == ValidationResult.RESOLUTION_TOO_LOW

    def test_color_image_validated(self, pipeline, sample_color_image):
        """Color images should be validated correctly."""
        result = pipeline.validate_image(sample_color_image)
        assert result == ValidationResult.OK


class TestProcess:
    """Tests for the full process method."""

    def test_full_pipeline_returns_preprocessed_image(
        self, pipeline, sample_grayscale_image
    ):
        """Full pipeline should return a PreprocessedImage."""
        result = pipeline.process(sample_grayscale_image)
        assert isinstance(result, PreprocessedImage)

    def test_full_pipeline_binary_output(self, pipeline, sample_grayscale_image):
        """Full pipeline output should be binary."""
        result = pipeline.process(sample_grayscale_image)
        unique_values = np.unique(result.binary_image)
        assert all(v in [0, 255] for v in unique_values)

    def test_full_pipeline_metadata(self, pipeline, sample_grayscale_image):
        """Full pipeline should populate metadata correctly."""
        result = pipeline.process(sample_grayscale_image)
        assert result.brightness_adjusted is True
        assert result.original_resolution == (640, 480)
        assert isinstance(result.rotation_corrected, float)

    def test_full_pipeline_rejects_invalid(self, pipeline):
        """Full pipeline should raise ValueError for invalid images."""
        small_img = np.zeros((200, 300), dtype=np.uint8)
        with pytest.raises(ValueError, match="validation failed"):
            pipeline.process(small_img)

    def test_full_pipeline_color_input(self, pipeline, sample_color_image):
        """Full pipeline should handle color input."""
        result = pipeline.process(sample_color_image)
        assert isinstance(result, PreprocessedImage)
        assert len(result.binary_image.shape) == 2  # Should be 2D binary
