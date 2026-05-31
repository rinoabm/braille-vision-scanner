"""Tests for the FastAPI backend API endpoints.

Verifies REST endpoints and WebSocket functionality for the
Braille Vision Scanner pipeline.
"""

import io
import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.api.main import app, _tts_engine


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def sample_braille_image():
    """Create a synthetic image that can pass through the pipeline.

    Creates a 640x480 image with some dot-like features.
    """
    image = np.zeros((480, 640), dtype=np.uint8)
    # Add some white dots to simulate Braille
    dot_positions = [
        (100, 100), (100, 120), (100, 140),
        (115, 100), (115, 120), (115, 140),
        (150, 100), (150, 120),
        (165, 100), (165, 140),
    ]
    for x, y in dot_positions:
        cv2.circle(image, (x, y), 5, 255, -1)

    # Convert to BGR for encoding
    bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return bgr


@pytest.fixture
def encoded_image(sample_braille_image):
    """Encode sample image as JPEG bytes."""
    success, buffer = cv2.imencode(".jpg", sample_braille_image)
    assert success
    return buffer.tobytes()


class TestSettingsEndpoints:
    """Tests for GET/PUT /api/settings."""

    def test_get_settings_returns_current_rate(self, client):
        """GET /api/settings returns the current TTS rate."""
        response = client.get("/api/settings")
        assert response.status_code == 200
        data = response.json()
        assert "tts_rate" in data
        assert isinstance(data["tts_rate"], int)
        assert 80 <= data["tts_rate"] <= 250

    def test_put_settings_updates_rate(self, client):
        """PUT /api/settings updates the TTS rate."""
        # Set to a specific rate
        response = client.put("/api/settings", json={"tts_rate": 180})
        assert response.status_code == 200
        data = response.json()
        assert data["tts_rate"] == 180

        # Verify it persisted
        response = client.get("/api/settings")
        assert response.status_code == 200
        assert response.json()["tts_rate"] == 180

    def test_put_settings_clamps_rate_high(self, client):
        """PUT /api/settings clamps rate to max 250."""
        response = client.put("/api/settings", json={"tts_rate": 300})
        assert response.status_code == 200
        data = response.json()
        assert data["tts_rate"] <= 250

    def test_put_settings_clamps_rate_low(self, client):
        """PUT /api/settings clamps rate to min 80."""
        response = client.put("/api/settings", json={"tts_rate": 50})
        assert response.status_code == 200
        data = response.json()
        assert data["tts_rate"] >= 80


class TestRepeatEndpoint:
    """Tests for POST /api/repeat."""

    def test_repeat_with_no_previous_utterance(self, client):
        """POST /api/repeat returns repeated=False when nothing was spoken."""
        # Reset state
        _tts_engine._state.last_sentence = ""
        response = client.post("/api/repeat")
        assert response.status_code == 200
        data = response.json()
        assert data["repeated"] is False
        assert data["last_sentence"] == ""

    def test_repeat_with_previous_utterance(self, client):
        """POST /api/repeat returns repeated=True when there's a last sentence."""
        _tts_engine._state.last_sentence = "Hello world"
        response = client.post("/api/repeat")
        assert response.status_code == 200
        data = response.json()
        assert data["repeated"] is True
        assert data["last_sentence"] == "Hello world"


class TestUploadEndpoint:
    """Tests for POST /api/upload."""

    def test_upload_valid_image(self, client, encoded_image):
        """POST /api/upload with a valid image returns pipeline results."""
        response = client.post(
            "/api/upload",
            files={"file": ("test.jpg", io.BytesIO(encoded_image), "image/jpeg")},
        )
        assert response.status_code == 200
        data = response.json()
        assert "text" in data
        assert "confidence" in data
        assert "line_confidences" in data
        assert "unrecognized_count" in data

    def test_upload_invalid_file(self, client):
        """POST /api/upload with invalid data returns error info."""
        response = client.post(
            "/api/upload",
            files={"file": ("test.txt", io.BytesIO(b"not an image"), "text/plain")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["text"] == ""
        assert data["confidence"] == 0.0
        assert "error" in data.get("alignment", {})

    def test_upload_small_image(self, client):
        """POST /api/upload with too-small image returns error."""
        # Create a tiny image below minimum resolution
        tiny = np.zeros((100, 100, 3), dtype=np.uint8)
        success, buffer = cv2.imencode(".jpg", tiny)
        assert success

        response = client.post(
            "/api/upload",
            files={"file": ("tiny.jpg", io.BytesIO(buffer.tobytes()), "image/jpeg")},
        )
        assert response.status_code == 200
        data = response.json()
        # Should fail validation (resolution too low)
        assert data["text"] == ""
        assert "error" in data.get("alignment", {})


class TestWebSocket:
    """Tests for WebSocket /ws/scan."""

    def test_websocket_connection(self, client):
        """WebSocket /ws/scan accepts connections."""
        with client.websocket_connect("/ws/scan") as websocket:
            # Connection established successfully
            assert websocket is not None

    def test_websocket_processes_frame(self, client, encoded_image):
        """WebSocket processes a valid frame and returns results."""
        with client.websocket_connect("/ws/scan") as websocket:
            websocket.send_bytes(encoded_image)
            response = websocket.receive_json()

            assert "text" in response
            assert "confidence" in response
            assert "skipped" in response
            assert response["skipped"] is False

    def test_websocket_handles_invalid_frame(self, client):
        """WebSocket handles invalid frame data gracefully."""
        with client.websocket_connect("/ws/scan") as websocket:
            websocket.send_bytes(b"not a valid image")
            response = websocket.receive_json()

            assert "error" in response
            assert response["text"] == ""

    def test_websocket_multiple_frames(self, client, encoded_image):
        """WebSocket can process multiple frames sequentially."""
        with client.websocket_connect("/ws/scan") as websocket:
            for _ in range(3):
                websocket.send_bytes(encoded_image)
                response = websocket.receive_json()
                assert "text" in response
                assert "confidence" in response
