"""Text-to-Speech Engine for the Braille Vision Scanner.

Converts decoded English text to speech output using pyttsx3 for offline TTS.
Supports rate adjustment, queuing, repeat, and handles audio device errors.

Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from backend.models.data_models import TTSState

logger = logging.getLogger(__name__)

# Unicode replacement character used for unrecognized Braille patterns
UNKNOWN_CHAR = "\ufffd"
# Word to announce in place of unrecognized characters
UNKNOWN_REPLACEMENT = "unrecognized"

# Rate constraints
MIN_RATE = 80
MAX_RATE = 250
DEFAULT_RATE = 150
RATE_INCREMENT = 10

# Retry delay for audio device unavailability (seconds)
AUDIO_RETRY_DELAY = 2.0

# Maximum latency for speech output after receiving text (seconds)
SPEECH_LATENCY_LIMIT = 0.5


class AudioDeviceUnavailableError(Exception):
    """Raised when the audio output device is not available."""

    pass


class TTSEngine:
    """Text-to-Speech engine with rate control, queuing, and repeat.

    Uses pyttsx3 for offline speech synthesis. Handles audio device
    unavailability with retry after 2 seconds.
    """

    def __init__(self, rate: int = DEFAULT_RATE) -> None:
        """Initialize with the given speech rate in WPM.

        Args:
            rate: Speech rate in words per minute, clamped to [80, 250].
                  Defaults to 150 WPM.
        """
        self._state = TTSState(
            current_rate=max(MIN_RATE, min(MAX_RATE, rate)),
            is_speaking=False,
            queue=[],
            last_sentence="",
        )
        self._engine = None
        self._lock = asyncio.Lock()
        self._initialized = False

    @property
    def state(self) -> TTSState:
        """Return the current TTS state."""
        return self._state

    def _ensure_engine(self) -> None:
        """Initialize the pyttsx3 engine lazily.

        If pyttsx3 is not available or audio device fails, logs a warning
        and continues without speech (graceful degradation).
        """
        if self._engine is None:
            try:
                import pyttsx3

                self._engine = pyttsx3.init()
                self._engine.setProperty("rate", self._wpm_to_pyttsx3_rate(self._state.current_rate))
                self._initialized = True
            except Exception as e:
                logger.warning(f"TTS engine initialization failed (speech disabled): {e}")
                # Don't raise - allow the system to continue without audio
                self._engine = None
                self._initialized = False

    def _wpm_to_pyttsx3_rate(self, wpm: int) -> int:
        """Convert WPM to pyttsx3 rate property value.

        pyttsx3 uses a rate property that roughly corresponds to WPM.
        """
        return wpm

    def _prepare_text(self, text: str) -> str:
        """Prepare text for speech by replacing U+FFFD with 'unrecognized'.

        Args:
            text: The text to prepare for speech output.

        Returns:
            Text with all U+FFFD characters replaced with 'unrecognized'.
        """
        return text.replace(UNKNOWN_CHAR, UNKNOWN_REPLACEMENT)

    def _extract_last_sentence(self, text: str) -> str:
        """Extract the last fully decoded sentence from text.

        If the text contains fewer than two sentences, returns the entire text.

        Args:
            text: The decoded text to extract from.

        Returns:
            The last sentence, or the entire text if fewer than two sentences.
        """
        # Split on sentence-ending punctuation followed by space or end
        import re

        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        sentences = [s for s in sentences if s.strip()]

        if len(sentences) < 2:
            return text.strip()

        return sentences[-1].strip()

    async def speak(self, text: str) -> None:
        """Speak text aloud within 500ms of receiving it.

        Replaces U+FFFD placeholder characters with 'unrecognized'.
        If currently speaking, queues the text instead.
        Handles audio device unavailability with retry after 2 seconds.

        Args:
            text: The text to speak.

        Requirements: 7.1, 7.5, 7.6, 7.7
        """
        if not text:
            return

        prepared_text = self._prepare_text(text)

        async with self._lock:
            if self._state.is_speaking:
                # Queue text if currently speaking (Req 7.6)
                self._state.queue.append(prepared_text)
                return

            await self._do_speak(prepared_text)

    async def _do_speak(self, text: str) -> None:
        """Perform the actual speech output with error handling.

        Args:
            text: Pre-processed text ready for speech.
        """
        self._state.is_speaking = True
        self._state.last_sentence = self._extract_last_sentence(text)

        try:
            await self._speak_with_retry(text)
        finally:
            self._state.is_speaking = False

        # Process queued text (Req 7.6)
        await self._process_queue()

    async def _speak_with_retry(self, text: str) -> None:
        """Attempt to speak text, retrying once on audio device failure.

        Args:
            text: Text to speak.
        """
        self._ensure_engine()
        if self._engine is None:
            # No audio device available - log and continue silently
            logger.info(f"TTS (no audio): {text}")
            return
        try:
            await self._run_speech(text)
        except Exception as e:
            # Retry once after 2 seconds (Req 7.7)
            logger.warning(f"TTS failed, retrying in 2 seconds: {e}")
            await asyncio.sleep(AUDIO_RETRY_DELAY)
            try:
                self._engine = None  # Reset engine for retry
                self._ensure_engine()
                if self._engine is not None:
                    await self._run_speech(text)
                else:
                    logger.info(f"TTS (no audio after retry): {text}")
            except Exception as retry_err:
                logger.error(f"TTS retry failed: {retry_err}")

    async def _run_speech(self, text: str) -> None:
        """Execute speech synthesis on the engine.

        Args:
            text: Text to speak.
        """
        if self._engine is None:
            return
        self._engine.setProperty("rate", self._wpm_to_pyttsx3_rate(self._state.current_rate))
        self._engine.say(text)
        loop = asyncio.get_event_loop()
        try:
            await asyncio.wait_for(
                loop.run_in_executor(None, self._engine.runAndWait),
                timeout=30.0,
            )
        except asyncio.TimeoutError:
            logger.warning("TTS runAndWait timed out after 30 seconds")
        except Exception as e:
            logger.warning(f"TTS speech execution error: {e}")

    async def _process_queue(self) -> None:
        """Process any queued text after current speech completes.

        Speaks queued text within 500ms of the current utterance completing.
        """
        while self._state.queue:
            next_text = self._state.queue.pop(0)
            await self._do_speak(next_text)

    def adjust_rate(self, delta: int) -> int:
        """Adjust speech rate by ±10 WPM within [80, 250] range.

        The delta is clamped to multiples of RATE_INCREMENT (10 WPM).
        The resulting rate is clamped to [MIN_RATE, MAX_RATE].

        Args:
            delta: The adjustment amount. Positive increases rate,
                   negative decreases rate. Applied in 10 WPM increments.

        Returns:
            The new speech rate in WPM.

        Requirements: 7.3
        """
        # Apply delta in 10 WPM increments (truncate toward zero)
        increments = int(delta / RATE_INCREMENT)
        adjustment = increments * RATE_INCREMENT

        new_rate = self._state.current_rate + adjustment
        # Clamp to valid range
        new_rate = max(MIN_RATE, min(MAX_RATE, new_rate))

        self._state.current_rate = new_rate

        # Update engine rate if initialized
        if self._engine is not None:
            self._engine.setProperty("rate", self._wpm_to_pyttsx3_rate(new_rate))

        return new_rate

    async def repeat_last(self) -> None:
        """Re-speak the last fully decoded sentence.

        If no previous sentence exists, does nothing.

        Requirements: 7.4
        """
        if not self._state.last_sentence:
            return

        async with self._lock:
            await self._do_speak(self._state.last_sentence)

    async def queue_text(self, text: str) -> None:
        """Queue text for speaking when currently speaking.

        If not currently speaking, speaks immediately.
        Replaces U+FFFD placeholder characters with 'unrecognized'.

        Args:
            text: The text to queue or speak.

        Requirements: 7.6
        """
        if not text:
            return

        prepared_text = self._prepare_text(text)

        async with self._lock:
            if self._state.is_speaking:
                self._state.queue.append(prepared_text)
            else:
                await self._do_speak(prepared_text)
