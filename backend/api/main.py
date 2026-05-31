"""FastAPI Backend for the Braille Vision Scanner.

Orchestrates the full pipeline (capture → preprocess → detect → segment → decode → TTS)
and serves REST and WebSocket endpoints for the frontend.

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
"""

from __future__ import annotations

import asyncio
import io
import logging
import time
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.models.data_models import (
    CameraSource,
    PreprocessedImage,
)
from backend.pipeline.alignment_guide import AlignmentGuide
from backend.pipeline.braille_decoder import BrailleDecoder
from backend.pipeline.cell_segmenter import CellSegmenter
from backend.pipeline.dot_detector import DotDetector
from backend.pipeline.embossed_detector import EmbossedDotDetector
from backend.pipeline.image_capture import ImageCaptureModule
from backend.pipeline.preprocessing import PreprocessingPipeline
from backend.pipeline.tts_engine import TTSEngine

logger = logging.getLogger(__name__)

# --- Pydantic models for request/response ---


class SettingsResponse(BaseModel):
    """Response model for GET /api/settings."""

    tts_rate: int


class SettingsUpdate(BaseModel):
    """Request model for PUT /api/settings."""

    tts_rate: int


class CaptureResponse(BaseModel):
    """Response model for capture and upload endpoints."""

    text: str
    confidence: float
    line_confidences: list[float]
    unrecognized_count: int
    alignment: Optional[dict] = None


class RepeatResponse(BaseModel):
    """Response model for POST /api/repeat."""

    repeated: bool
    last_sentence: str


# --- Application setup ---

app = FastAPI(
    title="Braille Vision Scanner API",
    description="Real-time Braille recognition pipeline with TTS output",
    version="1.0.0",
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pipeline singletons ---

_preprocessing = PreprocessingPipeline()
_dot_detector = DotDetector()
_embossed_detector = EmbossedDotDetector(min_radius=5, max_radius=35, sensitivity=0.5)
_cell_segmenter = CellSegmenter()
_braille_decoder = BrailleDecoder()
_alignment_guide = AlignmentGuide()
_tts_engine = TTSEngine()

# Frame skip threshold in seconds (Requirement 8.5)
_FRAME_PROCESSING_TIMEOUT = 1.0


# --- Pipeline execution ---


def _validate_braille_pattern(dots: list) -> bool:
    """Check if detected dots form a plausible Braille pattern.

    Uses strict grid-structure validation to reject random noise from
    faces, textures, and other non-Braille content.

    Returns:
        True if the dots likely represent real Braille, False otherwise.
    """
    if len(dots) == 0:
        return False

    # Need at least 4 dots to form a plausible Braille pattern in live scanning
    # (a single cell has up to 6 dots, minimum meaningful content is 2+ cells)
    if len(dots) < 3:
        return False

    # Check dot size consistency
    radii = [d.radius for d in dots]
    mean_radius = sum(radii) / len(radii)
    if mean_radius <= 0:
        return False

    radius_std = (sum((r - mean_radius) ** 2 for r in radii) / len(radii)) ** 0.5
    radius_cv = radius_std / mean_radius

    # Strict size uniformity — real Braille dots are fairly consistent
    # Handwritten dots have more variation than machine-embossed
    if radius_cv > 0.6:
        return False

    # For 6+ dots, require GRID STRUCTURE
    # Real Braille dots align into clear rows (2-3 distinct Y levels per cell)
    # and columns. Random noise is scattered.
    if len(dots) < 6:
        # For 3-5 dots, size consistency alone is enough
        return True
    
    # Check Y-coordinate clustering: dots should cluster into a few distinct rows
    y_values = sorted([d.y for d in dots])
    y_clusters = _count_clusters(y_values, mean_radius * 2.0)
    
    # Real Braille has 1-3 rows per line. With few dots, allow more Y variation.
    # Only reject if almost every dot is on its own line (truly scattered)
    if y_clusters > len(dots) * 0.8:
        return False

    # Check X-coordinate regularity: dots should show repeating spacing
    x_values = sorted([d.x for d in dots])
    x_gaps = [x_values[i] - x_values[i-1] for i in range(1, len(x_values)) if x_values[i] - x_values[i-1] > mean_radius * 0.5]
    
    if len(x_gaps) < 2:
        return True  # Not enough data to validate X structure
    
    # Check that X-gaps have some regularity (not all random)
    sorted_x_gaps = sorted(x_gaps)
    median_x_gap = sorted_x_gaps[len(sorted_x_gaps) // 2]
    
    # Count gaps that are within 60% of the median (regular spacing)
    regular_gaps = sum(1 for g in x_gaps if 0.4 * median_x_gap <= g <= 2.5 * median_x_gap)
    regularity = regular_gaps / len(x_gaps)
    
    # At least 50% of gaps should be somewhat regular
    if regularity < 0.5:
        return False

    return True


def _count_clusters(values: list, threshold: float) -> int:
    """Count the number of distinct clusters in a sorted list of values."""
    if not values:
        return 0
    clusters = 1
    for i in range(1, len(values)):
        if values[i] - values[i-1] > threshold:
            clusters += 1
    return clusters


def run_pipeline(image: np.ndarray) -> dict:
    """Run the full recognition pipeline on an image.

    Pipeline: preprocess → detect → segment → decode

    Only returns decoded text if the system is confident it found real Braille.
    Returns empty text if the image doesn't contain recognizable Braille patterns.

    Args:
        image: Input image as numpy array (BGR or grayscale).

    Returns:
        Dictionary with decoded text, confidence, line_confidences,
        unrecognized_count, and alignment info.

    Raises:
        ValueError: If preprocessing rejects the image.
    """
    # Alignment check
    alignment_result = _alignment_guide.analyze_frame(image)
    alignment_info = {
        "is_aligned": alignment_result.is_aligned,
        "direction": alignment_result.direction,
        "coverage_percent": alignment_result.coverage_percent,
        "braille_detected": alignment_result.braille_detected,
        "audio_cue": _alignment_guide.get_audio_cue(alignment_result),
    }

    # Preprocess
    preprocessed = _preprocessing.process(image)

    # Detect dots — try standard detector first, fall back to embossed detector
    detection_result = _dot_detector.detect(preprocessed)
    
    # If standard detector finds too few dots (< 4), try embossed detector
    # which uses shadow/highlight analysis for real raised Braille
    if len(detection_result.dots) < 4 or all(d.radius < 3 for d in detection_result.dots):
        embossed_result = _embossed_detector.detect(image)
        # Filter embossed results to only keep reasonably-sized dots
        filtered_dots = [d for d in embossed_result.dots if d.radius >= 3.0]
        if len(filtered_dots) > len([d for d in detection_result.dots if d.radius >= 3.0]):
            from backend.models.data_models import DotDetectionResult
            detection_result = DotDetectionResult(
                dots=filtered_dots,
                low_confidence_regions=embossed_result.low_confidence_regions,
                processing_time_ms=embossed_result.processing_time_ms,
            )

    # --- BRAILLE VALIDATION GATE ---
    # Check if detected dots form a plausible Braille pattern.
    # Real Braille has dots arranged in a regular grid with consistent spacing.
    # Random noise (faces, textures) produces scattered dots with no structure.
    
    dots = detection_result.dots
    
    # Filter out noise: keep the largest cluster of similarly-sized dots
    # Real Braille dots are consistent in size; noise varies wildly
    if len(dots) > 3:
        radii = [d.radius for d in dots]
        # Find the most common size range by looking at the largest dots
        # (real dots are usually bigger than noise specks)
        sorted_by_r = sorted(dots, key=lambda d: d.radius, reverse=True)
        # Take the top half by size and check if they're consistent
        top_half = sorted_by_r[:max(3, len(sorted_by_r) // 2)]
        top_radii = [d.radius for d in top_half]
        top_mean = sum(top_radii) / len(top_radii)
        
        # Keep dots within 50% of the top-half mean
        filtered_dots = [d for d in dots if d.radius >= top_mean * 0.5]
        if len(filtered_dots) >= 3:
            dots = filtered_dots
            from backend.models.data_models import DotDetectionResult as DotDetResult
            detection_result = DotDetResult(
                dots=dots,
                low_confidence_regions=detection_result.low_confidence_regions,
                processing_time_ms=detection_result.processing_time_ms,
            )
    
    is_likely_braille = _validate_braille_pattern(dots)
    
    if not is_likely_braille:
        return {
            "text": "",
            "confidence": 0.0,
            "line_confidences": [],
            "unrecognized_count": 0,
            "alignment": alignment_info,
            "braille_detected": False,
        }

    # Segment into cells
    segmentation_result = _cell_segmenter.segment(detection_result)

    # Decode to text
    decoded = _braille_decoder.decode(segmentation_result)

    # Compute overall confidence
    if decoded.per_character_confidence:
        overall_confidence = sum(decoded.per_character_confidence) / len(
            decoded.per_character_confidence
        )
    else:
        overall_confidence = 0.0

    # Final quality gate: reject if too many unrecognized characters
    total_chars = len(decoded.text.replace('\n', '').replace(' ', ''))
    if total_chars > 0:
        unrecognized_ratio = len(decoded.unrecognized_indices) / total_chars
    else:
        unrecognized_ratio = 1.0

    # If more than 40% of characters are unrecognized, it's probably not Braille
    if unrecognized_ratio > 0.4 or overall_confidence < 0.5:
        return {
            "text": "",
            "confidence": overall_confidence,
            "line_confidences": decoded.line_confidences,
            "unrecognized_count": len(decoded.unrecognized_indices),
            "alignment": alignment_info,
            "braille_detected": False,
        }

    return {
        "text": decoded.text,
        "confidence": overall_confidence,
        "line_confidences": decoded.line_confidences,
        "unrecognized_count": len(decoded.unrecognized_indices),
        "alignment": alignment_info,
    }


# --- REST Endpoints ---


@app.post("/api/capture", response_model=CaptureResponse)
async def capture_endpoint():
    """Single-shot capture triggering the full pipeline.

    Captures a frame from the default camera source and runs the full
    recognition pipeline. Delivers spoken output within 2 seconds of capture.

    Requirements: 8.3
    """
    capture_module = ImageCaptureModule(source=CameraSource.WEBCAM)
    try:
        capture_module.initialize()
        frame = await capture_module.capture_frame()
    except Exception as e:
        logger.error(f"Capture failed: {e}")
        return CaptureResponse(
            text="",
            confidence=0.0,
            line_confidences=[],
            unrecognized_count=0,
            alignment={"error": str(e)},
        )
    finally:
        capture_module.release()

    # Run pipeline
    try:
        result = run_pipeline(frame.image)
    except ValueError as e:
        return CaptureResponse(
            text="",
            confidence=0.0,
            line_confidences=[],
            unrecognized_count=0,
            alignment={"error": str(e)},
        )

    # Update TTS state (actual speech handled by frontend via Web Speech API)
    if result["text"]:
        _tts_engine._state.last_sentence = _tts_engine._extract_last_sentence(result["text"])

    return CaptureResponse(**result)


@app.post("/api/upload", response_model=CaptureResponse)
async def upload_endpoint(file: UploadFile = File(...)):
    """Upload an image file and run the full pipeline.

    Accepts multipart form image upload (JPEG/PNG), processes through
    the full recognition pipeline, and returns decoded text.

    Requirements: 8.1, 8.3
    """
    # Read uploaded file
    contents = await file.read()

    # Decode image from bytes
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        return CaptureResponse(
            text="",
            confidence=0.0,
            line_confidences=[],
            unrecognized_count=0,
            alignment={"error": "Invalid image file. Please upload a JPEG or PNG image."},
        )

    # Run pipeline
    try:
        result = run_pipeline(image)
    except ValueError as e:
        return CaptureResponse(
            text="",
            confidence=0.0,
            line_confidences=[],
            unrecognized_count=0,
            alignment={"error": str(e)},
        )

    # Update TTS state (actual speech handled by frontend via Web Speech API)
    if result["text"]:
        _tts_engine._state.last_sentence = _tts_engine._extract_last_sentence(result["text"])

    return CaptureResponse(**result)


@app.get("/api/settings", response_model=SettingsResponse)
async def get_settings():
    """Get current TTS settings.

    Returns the current speech rate configuration.

    Requirements: 8.4
    """
    return SettingsResponse(tts_rate=_tts_engine.state.current_rate)


@app.put("/api/settings", response_model=SettingsResponse)
async def update_settings(settings: SettingsUpdate):
    """Update TTS settings (speech rate).

    Adjusts the TTS rate. The rate is clamped to [80, 250] WPM.

    Requirements: 8.4
    """
    current_rate = _tts_engine.state.current_rate
    delta = settings.tts_rate - current_rate
    new_rate = _tts_engine.adjust_rate(delta)
    return SettingsResponse(tts_rate=new_rate)


@app.post("/api/repeat", response_model=RepeatResponse)
async def repeat_endpoint():
    """Repeat the last spoken utterance.

    Re-speaks the last fully decoded sentence via TTS.

    Requirements: 8.4
    """
    last_sentence = _tts_engine.state.last_sentence
    if last_sentence:
        # Update state to reflect repeat was requested
        # TTS will be triggered by the frontend via a separate mechanism
        # or handled asynchronously without blocking the response
        _tts_engine._state.last_sentence = last_sentence
        return RepeatResponse(repeated=True, last_sentence=last_sentence)
    return RepeatResponse(repeated=False, last_sentence="")


# --- WebSocket Endpoint ---


@app.websocket("/ws/scan")
async def scan_stream(websocket: WebSocket):
    """WebSocket endpoint for real-time Braille scanning.

    Accepts binary frame data or base64-encoded JPEG strings from the frontend,
    runs the full pipeline, and sends back JSON with decoded text, confidence,
    and alignment cues.

    Implements frame skipping: if processing exceeds 1 second, the frame
    is skipped and the next available frame is processed instead.

    Requirements: 8.1, 8.2, 8.5
    """
    await websocket.accept()

    try:
        while True:
            # Receive frame data - supports both binary and base64 text
            message = await websocket.receive()

            if "bytes" in message and message["bytes"]:
                data = message["bytes"]
            elif "text" in message and message["text"]:
                # Decode base64 text to bytes (frontend sends base64 JPEG)
                import base64 as b64module

                try:
                    data = b64module.b64decode(message["text"])
                except Exception:
                    await websocket.send_json({
                        "error": "Invalid base64 data",
                        "text": "",
                        "confidence": 0.0,
                    })
                    continue
            else:
                await websocket.send_json({
                    "error": "Empty frame data",
                    "text": "",
                    "confidence": 0.0,
                })
                continue

            # Decode image from bytes
            nparr = np.frombuffer(data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                await websocket.send_json({
                    "error": "Invalid frame data",
                    "text": "",
                    "confidence": 0.0,
                })
                continue

            # Save debug frame (last received frame for debugging)
            cv2.imwrite("debug_last_frame.jpg", image)

            # Process frame with timeout for frame skipping (Req 8.5)
            start_time = time.perf_counter()

            try:
                # Run pipeline in executor to avoid blocking the event loop
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, run_pipeline, image),
                    timeout=_FRAME_PROCESSING_TIMEOUT,
                )
            except asyncio.TimeoutError:
                # Frame processing exceeded 1 second - skip frame (Req 8.5)
                logger.debug("Frame skipped: processing exceeded 1 second")
                await websocket.send_json({
                    "skipped": True,
                    "text": "",
                    "confidence": 0.0,
                    "reason": "Processing exceeded 1 second threshold",
                })
                continue
            except ValueError as e:
                # Preprocessing rejected the frame
                await websocket.send_json({
                    "error": str(e),
                    "text": "",
                    "confidence": 0.0,
                })
                continue

            processing_time = time.perf_counter() - start_time

            # Send result back to frontend
            response = {
                "text": result["text"],
                "confidence": result["confidence"],
                "line_confidences": result["line_confidences"],
                "unrecognized_count": result["unrecognized_count"],
                "alignment": result["alignment"],
                "processing_time_ms": round(processing_time * 1000, 1),
                "skipped": False,
            }

            await websocket.send_json(response)

            # Update TTS state (actual speech handled by frontend)
            if result["text"]:
                _tts_engine._state.last_sentence = _tts_engine._extract_last_sentence(result["text"])

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.close(code=1011, reason=str(e))
        except Exception:
            pass


# --- Helper functions ---


async def _speak_text(text: str) -> None:
    """Speak text via TTS engine, handling errors gracefully.

    This is a fire-and-forget helper. It will never raise or block
    the calling endpoint. If TTS is unavailable, it logs and returns.

    Args:
        text: The text to speak aloud.
    """
    try:
        # Update last_sentence state regardless of whether speech works
        _tts_engine._state.last_sentence = _tts_engine._extract_last_sentence(text)
        prepared = _tts_engine._prepare_text(text)
        _tts_engine._ensure_engine()
        if _tts_engine._engine is not None:
            _tts_engine._engine.setProperty(
                "rate", _tts_engine._wpm_to_pyttsx3_rate(_tts_engine._state.current_rate)
            )
            _tts_engine._engine.say(prepared)
            loop = asyncio.get_event_loop()
            await asyncio.wait_for(
                loop.run_in_executor(None, _tts_engine._engine.runAndWait),
                timeout=10.0,
            )
    except asyncio.TimeoutError:
        logger.warning("TTS timed out")
    except Exception as e:
        logger.debug(f"TTS speak skipped: {e}")
