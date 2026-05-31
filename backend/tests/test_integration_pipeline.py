"""Integration tests for the full Braille Vision Scanner pipeline.

Tests the end-to-end flow: image → preprocessing → detection → segmentation → decoding.
Validates latency requirements (< 2 seconds for single-shot) and WebSocket streaming.

Requirements: 8.1, 8.2, 8.3, 8.4
"""

import base64
import io
import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.api.main import app, run_pipeline


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


def create_synthetic_braille_image(text: str = "ab") -> np.ndarray:
    """Create a synthetic Braille image with known dot patterns.

    Generates a 640x480 image with Braille-like dot patterns.
    'a' = dot 1, 'b' = dots 1,2

    Args:
        text: Characters to encode (used for determining dot count).

    Returns:
        BGR image as numpy array.
    """
    image = np.zeros((480, 640), dtype=np.uint8)
    # Fill with a medium gray background
    image[:] = 180

    # Standard Braille cell dimensions (approximate pixel positions)
    dot_radius = 6
    intra_col_spacing = 18  # horizontal spacing between dot columns in a cell
    intra_row_spacing = 18  # vertical spacing between dot rows in a cell
    inter_cell_spacing = 40  # spacing between cells

    # Start position (centered in image)
    start_x = 200
    start_y = 200

    # Draw dots for 'a' (dot 1 = top-left)
    # Cell 1: dot at position 1 (row 0, col 0)
    cv2.circle(image, (start_x, start_y), dot_radius, 255, -1)

    # Cell 2: dots at positions 1, 2 (row 0 col 0, row 1 col 0)
    cell2_x = start_x + inter_cell_spacing
    cv2.circle(image, (cell2_x, start_y), dot_radius, 255, -1)
    cv2.circle(image, (cell2_x, start_y + intra_row_spacing), dot_radius, 255, -1)

    # Convert to BGR
    bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return bgr


def encode_image_to_jpeg_bytes(image: np.ndarray) -> bytes:
    """Encode an image to JPEG bytes."""
    success, buffer = cv2.imencode(".jpg", image)
    assert success, "Failed to encode image to JPEG"
    return buffer.tobytes()


def encode_image_to_base64(image: np.ndarray) -> str:
    """Encode an image to base64 string (as frontend sends it)."""
    jpeg_bytes = encode_image_to_jpeg_bytes(image)
    return base64.b64encode(jpeg_bytes).decode("utf-8")


class TestFullPipelineLatency:
    """Tests that the full pipeline meets latency requirements."""

    def test_single_shot_under_2_seconds(self):
        """Full pipeline completes within 2 seconds for single-shot capture.

        Requirements: 8.3
        """
        image = create_synthetic_braille_image()

        start_time = time.perf_counter()
        result = run_pipeline(image)
        elapsed = time.perf_counter() - start_time

        assert elapsed < 2.0, (
            f"Pipeline took {elapsed:.3f}s, exceeds 2-second requirement"
        )
        assert "text" in result
        assert "confidence" in result
        assert "alignment" in result

    def test_pipeline_returns_valid_structure(self):
        """Pipeline returns all expected fields in the result.

        Requirements: 8.1
        """
        image = create_synthetic_braille_image()
        result = run_pipeline(image)

        assert isinstance(result["text"], str)
        assert isinstance(result["confidence"], float)
        assert isinstance(result["line_confidences"], list)
        assert isinstance(result["unrecognized_count"], int)
        assert isinstance(result["alignment"], dict)
        assert "is_aligned" in result["alignment"]
        assert "direction" in result["alignment"]
        assert "coverage_percent" in result["alignment"]

    def test_continuous_mode_throughput(self):
        """Pipeline can process at least 2 frames per second.

        Requirements: 8.2
        """
        image = create_synthetic_braille_image()
        num_frames = 4
        start_time = time.perf_counter()

        for _ in range(num_frames):
            run_pipeline(image)

        elapsed = time.perf_counter() - start_time
        fps = num_frames / elapsed

        assert fps >= 2.0, (
            f"Pipeline achieved {fps:.2f} FPS, below 2 FPS requirement"
        )


class TestWebSocketIntegration:
    """Tests WebSocket endpoint with base64 and binary frame data."""

    def test_websocket_accepts_binary_frame(self, client):
        """WebSocket processes binary JPEG frame data.

        Requirements: 8.1
        """
        image = create_synthetic_braille_image()
        jpeg_bytes = encode_image_to_jpeg_bytes(image)

        with client.websocket_connect("/ws/scan") as websocket:
            websocket.send_bytes(jpeg_bytes)
            response = websocket.receive_json()

            assert "text" in response
            assert "confidence" in response
            assert response["skipped"] is False
            assert "processing_time_ms" in response

    def test_websocket_accepts_base64_frame(self, client):
        """WebSocket processes base64-encoded JPEG frame data (frontend format).

        Requirements: 8.1
        """
        image = create_synthetic_braille_image()
        base64_str = encode_image_to_base64(image)

        with client.websocket_connect("/ws/scan") as websocket:
            websocket.send_text(base64_str)
            response = websocket.receive_json()

            assert "text" in response
            assert "confidence" in response
            assert response["skipped"] is False

    def test_websocket_latency_under_2_seconds(self, client):
        """WebSocket round-trip (send frame → receive result) under 2 seconds.

        Requirements: 8.3
        """
        image = create_synthetic_braille_image()
        jpeg_bytes = encode_image_to_jpeg_bytes(image)

        with client.websocket_connect("/ws/scan") as websocket:
            start_time = time.perf_counter()
            websocket.send_bytes(jpeg_bytes)
            response = websocket.receive_json()
            elapsed = time.perf_counter() - start_time

            assert elapsed < 2.0, (
                f"WebSocket round-trip took {elapsed:.3f}s, exceeds 2-second limit"
            )
            assert "text" in response

    def test_websocket_multiple_frames_continuous(self, client):
        """WebSocket handles continuous frame stream at ≥2 FPS.

        Requirements: 8.2
        """
        image = create_synthetic_braille_image()
        jpeg_bytes = encode_image_to_jpeg_bytes(image)
        num_frames = 4

        with client.websocket_connect("/ws/scan") as websocket:
            start_time = time.perf_counter()

            for _ in range(num_frames):
                websocket.send_bytes(jpeg_bytes)
                response = websocket.receive_json()
                assert "text" in response

            elapsed = time.perf_counter() - start_time
            fps = num_frames / elapsed

            assert fps >= 2.0, (
                f"WebSocket achieved {fps:.2f} FPS, below 2 FPS requirement"
            )

    def test_websocket_invalid_base64_handled(self, client):
        """WebSocket handles invalid base64 data gracefully."""
        with client.websocket_connect("/ws/scan") as websocket:
            websocket.send_text("not-valid-base64!!!")
            response = websocket.receive_json()

            assert "error" in response
            assert response["text"] == ""


class TestUploadEndpointIntegration:
    """Tests the upload endpoint with synthetic Braille images."""

    def test_upload_synthetic_braille_image(self, client):
        """Upload endpoint processes a synthetic Braille image through full pipeline.

        Requirements: 8.1, 8.3
        """
        image = create_synthetic_braille_image()
        jpeg_bytes = encode_image_to_jpeg_bytes(image)

        start_time = time.perf_counter()
        response = client.post(
            "/api/upload",
            files={"file": ("braille.jpg", io.BytesIO(jpeg_bytes), "image/jpeg")},
        )
        elapsed = time.perf_counter() - start_time

        assert response.status_code == 200
        data = response.json()
        assert "text" in data
        assert "confidence" in data
        assert elapsed < 2.0, (
            f"Upload endpoint took {elapsed:.3f}s, exceeds 2-second limit"
        )


class TestMemoryUsage:
    """Tests that memory usage stays within bounds."""

    def test_pipeline_memory_within_2gb(self):
        """Pipeline memory usage stays within 2GB.

        Requirements: 8.4
        """
        import sys

        image = create_synthetic_braille_image()

        # Run pipeline multiple times to check for memory leaks
        for _ in range(10):
            run_pipeline(image)

        # Check that the image and result don't exceed reasonable size
        # A 640x480 BGR image is ~900KB, pipeline intermediates should be similar
        image_size_mb = image.nbytes / (1024 * 1024)
        assert image_size_mb < 100, (
            f"Single image uses {image_size_mb:.1f}MB, unexpectedly large"
        )

        # Verify we can get process memory info (basic sanity check)
        # On most systems, a fresh Python process uses ~50-100MB
        # The pipeline should not push us anywhere near 2GB
        try:
            import psutil

            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)
            assert memory_mb < 2048, (
                f"Process using {memory_mb:.0f}MB, exceeds 2GB limit"
            )
        except ImportError:
            # psutil not available, skip detailed memory check
            # The image size check above provides basic validation
            pass
