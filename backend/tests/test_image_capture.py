"""Unit tests for the ImageCaptureModule.

Tests sharpness validation, camera source selection, upload functionality,
and error handling for missing camera and permission denial.
"""

import os
import tempfile

import cv2
import numpy as np
import pytest
import pytest_asyncio

from backend.models.data_models import CameraSource, CapturedFrame
from backend.pipeline.image_capture import (
    CameraInitializationError,
    ImageCaptureModule,
    ImageSharpnessError,
    _DEFAULT_SHARPNESS_THRESHOLD,
)


class TestCheckSharpness:
    """Tests for the check_sharpness method."""

    def test_sharp_image_has_high_score(self):
        """A sharp image with clear edges should have a high sharpness score."""
        module = ImageCaptureModule(CameraSource.WEBCAM)
        # Create a sharp synthetic image with clear edges
        image = np.zeros((480, 640), dtype=np.uint8)
        # Add sharp edges (checkerboard pattern)
        for i in range(0, 480, 20):
            for j in range(0, 640, 20):
                if (i // 20 + j // 20) % 2 == 0:
                    image[i : i + 20, j : j + 20] = 255

        score = module.check_sharpness(image)
        assert score > _DEFAULT_SHARPNESS_THRESHOLD

    def test_blurry_image_has_low_score(self):
        """A heavily blurred image should have a low sharpness score."""
        module = ImageCaptureModule(CameraSource.WEBCAM)
        # Create a uniform gray image (no edges = very blurry)
        image = np.full((480, 640), 128, dtype=np.uint8)

        score = module.check_sharpness(image)
        assert score < _DEFAULT_SHARPNESS_THRESHOLD

    def test_bgr_image_converted_to_grayscale(self):
        """A BGR color image should be handled correctly."""
        module = ImageCaptureModule(CameraSource.WEBCAM)
        # Create a 3-channel BGR image with edges
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        image[100:200, 100:200] = [255, 255, 255]

        score = module.check_sharpness(image)
        assert score > 0

    def test_sharpness_increases_with_more_edges(self):
        """More edges in an image should produce a higher sharpness score."""
        module = ImageCaptureModule(CameraSource.WEBCAM)

        # Image with few edges
        simple = np.zeros((480, 640), dtype=np.uint8)
        simple[200:280, 200:440] = 255

        # Image with many edges (fine checkerboard)
        complex_img = np.zeros((480, 640), dtype=np.uint8)
        for i in range(0, 480, 10):
            for j in range(0, 640, 10):
                if (i // 10 + j // 10) % 2 == 0:
                    complex_img[i : i + 10, j : j + 10] = 255

        simple_score = module.check_sharpness(simple)
        complex_score = module.check_sharpness(complex_img)
        assert complex_score > simple_score


class TestCameraSourceSelection:
    """Tests for camera source configuration."""

    def test_webcam_source(self):
        """Module should accept webcam source."""
        module = ImageCaptureModule(CameraSource.WEBCAM)
        assert module.source == CameraSource.WEBCAM

    def test_mobile_rear_source(self):
        """Module should accept mobile rear camera source."""
        module = ImageCaptureModule(CameraSource.MOBILE_REAR)
        assert module.source == CameraSource.MOBILE_REAR

    def test_mobile_front_source(self):
        """Module should accept mobile front camera source."""
        module = ImageCaptureModule(CameraSource.MOBILE_FRONT)
        assert module.source == CameraSource.MOBILE_FRONT

    def test_not_initialized_by_default(self):
        """Module should not be initialized until initialize() is called."""
        module = ImageCaptureModule(CameraSource.WEBCAM)
        assert not module.is_initialized


class TestUploadImage:
    """Tests for the upload_image method."""

    @pytest.fixture
    def sharp_image_path(self, tmp_path):
        """Create a temporary sharp image file."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        # Create a checkerboard pattern for high sharpness
        for i in range(0, 480, 20):
            for j in range(0, 640, 20):
                if (i // 20 + j // 20) % 2 == 0:
                    image[i : i + 20, j : j + 20] = [255, 255, 255]
        path = str(tmp_path / "sharp_test.png")
        cv2.imwrite(path, image)
        return path

    @pytest.fixture
    def blurry_image_path(self, tmp_path):
        """Create a temporary blurry (uniform) image file."""
        image = np.full((480, 640, 3), 128, dtype=np.uint8)
        path = str(tmp_path / "blurry_test.png")
        cv2.imwrite(path, image)
        return path

    @pytest.fixture
    def small_image_path(self, tmp_path):
        """Create a temporary image below minimum resolution."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        path = str(tmp_path / "small_test.png")
        cv2.imwrite(path, image)
        return path

    @pytest.mark.asyncio
    async def test_upload_valid_sharp_image(self, sharp_image_path):
        """Uploading a valid sharp image should return a CapturedFrame."""
        module = ImageCaptureModule(CameraSource.WEBCAM)
        result = await module.upload_image(sharp_image_path)

        assert isinstance(result, CapturedFrame)
        assert result.resolution == (640, 480)
        assert result.sharpness_score >= _DEFAULT_SHARPNESS_THRESHOLD
        assert result.source == CameraSource.WEBCAM
        assert result.image is not None

    @pytest.mark.asyncio
    async def test_upload_nonexistent_file(self):
        """Uploading a nonexistent file should raise FileNotFoundError."""
        module = ImageCaptureModule(CameraSource.WEBCAM)
        with pytest.raises(FileNotFoundError):
            await module.upload_image("/nonexistent/path/image.png")

    @pytest.mark.asyncio
    async def test_upload_invalid_file(self, tmp_path):
        """Uploading a non-image file should raise ValueError."""
        # Create a text file that isn't a valid image
        path = str(tmp_path / "not_an_image.txt")
        with open(path, "w") as f:
            f.write("this is not an image")

        module = ImageCaptureModule(CameraSource.WEBCAM)
        with pytest.raises(ValueError, match="Cannot read image file"):
            await module.upload_image(path)

    @pytest.mark.asyncio
    async def test_upload_low_resolution_image(self, small_image_path):
        """Uploading an image below 640x480 should raise ValueError."""
        module = ImageCaptureModule(CameraSource.WEBCAM)
        with pytest.raises(ValueError, match="below the minimum"):
            await module.upload_image(small_image_path)

    @pytest.mark.asyncio
    async def test_upload_blurry_image(self, blurry_image_path):
        """Uploading a blurry image should raise ImageSharpnessError."""
        module = ImageCaptureModule(CameraSource.WEBCAM)
        with pytest.raises(ImageSharpnessError) as exc_info:
            await module.upload_image(blurry_image_path)
        assert exc_info.value.audio_message == (
            "Image is blurry. Please hold steady and try again."
        )


class TestErrorHandling:
    """Tests for error handling scenarios."""

    @pytest.mark.asyncio
    async def test_capture_without_initialization(self):
        """Calling capture_frame without initialize() should raise an error."""
        module = ImageCaptureModule(CameraSource.WEBCAM)
        with pytest.raises(CameraInitializationError) as exc_info:
            await module.capture_frame()
        assert "not initialized" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_preview_without_initialization(self):
        """Calling start_preview without initialize() should raise an error."""
        module = ImageCaptureModule(CameraSource.WEBCAM)
        with pytest.raises(CameraInitializationError):
            async for _ in module.start_preview():
                break

    def test_camera_initialization_error_has_audio_message(self):
        """CameraInitializationError should carry an audible error message."""
        error = CameraInitializationError(
            "No camera detected",
            audio_message="No camera available. You can upload an image instead.",
        )
        assert error.audio_message == (
            "No camera available. You can upload an image instead."
        )

    def test_camera_permission_error_has_audio_message(self):
        """CameraPermissionError should carry an audible error message."""
        from backend.pipeline.image_capture import CameraPermissionError

        error = CameraPermissionError(
            "Permission denied",
            audio_message=(
                "Camera permission denied. "
                "Please grant camera access in your device settings."
            ),
        )
        assert "permission denied" in error.audio_message.lower()

    def test_sharpness_error_has_audio_message(self):
        """ImageSharpnessError should carry an audible prompt to re-capture."""
        error = ImageSharpnessError(sharpness_score=50.0, threshold=100.0)
        assert error.audio_message == (
            "Image is blurry. Please hold steady and try again."
        )
        assert error.sharpness_score == 50.0
        assert error.threshold == 100.0

    def test_release_without_initialization(self):
        """Calling release() without initialization should not raise."""
        module = ImageCaptureModule(CameraSource.WEBCAM)
        module.release()  # Should not raise
        assert not module.is_initialized


class TestCustomSharpnessThreshold:
    """Tests for custom sharpness threshold configuration."""

    @pytest.mark.asyncio
    async def test_custom_threshold_accepts_sharp_enough_image(self, tmp_path):
        """A lower threshold should accept images that default would reject."""
        # Create an image with moderate sharpness
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        image[200:280, 200:440] = [255, 255, 255]
        path = str(tmp_path / "moderate.png")
        cv2.imwrite(path, image)

        # Use a very low threshold
        module = ImageCaptureModule(CameraSource.WEBCAM, sharpness_threshold=1.0)
        result = await module.upload_image(path)
        assert isinstance(result, CapturedFrame)

    @pytest.mark.asyncio
    async def test_high_threshold_rejects_moderate_image(self, tmp_path):
        """A very high threshold should reject moderately sharp images."""
        # Create an image with moderate sharpness
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        image[200:280, 200:440] = [255, 255, 255]
        path = str(tmp_path / "moderate.png")
        cv2.imwrite(path, image)

        # Use a very high threshold
        module = ImageCaptureModule(
            CameraSource.WEBCAM, sharpness_threshold=10000.0
        )
        with pytest.raises(ImageSharpnessError):
            await module.upload_image(path)
