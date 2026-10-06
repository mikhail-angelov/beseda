"""Records the whole dialog as it sounded: microphone and assistant voice on one wall-clock timeline."""

import logging
import threading
import time
import wave
from pathlib import Path

import numpy as np
import resampy

log = logging.getLogger("pogo.recording")

OUTPUT_RATE = 22050
# A chunk arriving later than this after the previous one starts a new segment (mic was off, assistant was silent).
GAP_SECONDS = 0.25
# Audio kept before the first detected speech: VAD fires a bit after the first syllable.
PRE_ROLL_SECONDS = 0.5


class Track:
    """Chunks of one source grouped into continuous segments positioned by arrival time."""

    def __init__(self, rate: int):
        self.rate = rate
        self.segments: list[tuple[float, list[bytes]]] = []
        self.end = 0.0

    def add(self, chunk: bytes, now: float) -> None:
        duration = len(chunk) / 2 / self.rate
        # Both sources deliver a chunk right after it was captured/played, so it started `duration` ago.
        start = now - duration
        if self.segments and start - self.end < GAP_SECONDS:
            self.segments[-1][1].append(chunk)
            self.end += duration
        else:
            self.segments.append((max(start, self.end), [chunk]))
            self.end = max(start, self.end) + duration


class DialogRecorder:
    def __init__(self, path: Path, voice_rate: int):
        self.path = path
        self.t0 = time.monotonic()
        self.mic = Track(16000)  # RealtimeSTT delivers 16 kHz int16 mono
        self.voice = Track(voice_rate)
        self.lock = threading.Lock()
        self.first_speech: float | None = None
        log.info("recording dialog to %s", path)

    def on_mic(self, chunk: bytes) -> None:
        with self.lock:
            self.mic.add(chunk, time.monotonic() - self.t0)

    def on_speech_start(self) -> None:
        """The recording starts here: the mic delivers background noise from startup on."""
        if self.first_speech is None:
            self.first_speech = time.monotonic() - self.t0
            log.info("first speech at %.2fs since start", self.first_speech)

    def on_voice(self, chunk: bytes) -> None:
        with self.lock:
            self.voice.add(chunk, time.monotonic() - self.t0)

    def save(self) -> None:
        with self.lock:
            tracks = [self.mic, self.voice]
            if self.first_speech is None:
                log.warning("no speech detected, %s not written", self.path)
                return
            shift = max(0.0, self.first_speech - PRE_ROLL_SECONDS)
            length = max(track.end for track in tracks) - shift
            mix = np.zeros(int(length * OUTPUT_RATE) + OUTPUT_RATE, dtype=np.float32)
            for track in tracks:
                for start, chunks in track.segments:
                    samples = np.frombuffer(b"".join(chunks), dtype=np.int16).astype(np.float32)
                    if track.rate != OUTPUT_RATE:
                        samples = resampy.resample(samples, track.rate, OUTPUT_RATE)
                    offset = int((start - shift) * OUTPUT_RATE)
                    if offset < 0:  # audio before the first phrase
                        samples, offset = samples[-offset:], 0
                    mix[offset : offset + len(samples)] += samples[: len(mix) - offset]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(self.path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(OUTPUT_RATE)
            wf.writeframes(np.clip(mix, -32768, 32767).astype(np.int16).tobytes())
        log.info(
            "saved dialog %s duration=%.1fs mic_segments=%d voice_segments=%d",
            self.path, len(mix) / OUTPUT_RATE, len(self.mic.segments), len(self.voice.segments),
        )
