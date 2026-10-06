"""Microphone with macOS voice processing (the echo canceller FaceTime uses).

Whatever the speakers play — a video, music, the assistant itself — is subtracted from the input, so the voice
activity detector hears only the person and finds the pause at the end of a phrase. Other apps keep their volume:
ducking is turned off.
"""

import logging
from typing import Callable

import numpy as np
import soxr
from AVFoundation import (
    AVAudioEngine,
    AVAudioVoiceProcessingOtherAudioDuckingConfiguration,
    AVAudioVoiceProcessingOtherAudioDuckingLevelMin,
)

log = logging.getLogger("beseda.microphone")

RATE = 16000  # what RealtimeSTT expects


class Microphone:
    """Streams 16 kHz int16 mono PCM to `feed` while `on` is true."""

    def __init__(self, feed: Callable[[bytes], None]):
        self.feed = feed
        self.on = False
        self.engine = AVAudioEngine.alloc().init()
        node = self.engine.inputNode()
        ok, error = node.setVoiceProcessingEnabled_error_(True, None)
        if not ok:
            raise RuntimeError(f"macOS voice processing is unavailable: {error}")
        node.setVoiceProcessingOtherAudioDuckingConfiguration_(
            AVAudioVoiceProcessingOtherAudioDuckingConfiguration(False, AVAudioVoiceProcessingOtherAudioDuckingLevelMin)
        )
        audio_format = node.outputFormatForBus_(0)
        # Voice processing reports several identical channels; the first one is the processed signal.
        self.resampler = soxr.ResampleStream(audio_format.sampleRate(), RATE, 1, dtype="float32")
        node.installTapOnBus_bufferSize_format_block_(0, 2048, audio_format, self._on_buffer)
        self.engine.prepare()
        ok, error = self.engine.startAndReturnError_(None)
        if not ok:
            raise RuntimeError(f"microphone did not start: {error}")
        log.info(
            "voice processing microphone rate=%g channels=%d", audio_format.sampleRate(), audio_format.channelCount()
        )

    def _on_buffer(self, buffer, when) -> None:
        if not self.on:
            return
        frames = buffer.frameLength()
        samples = np.frombuffer(buffer.floatChannelData()[0].as_buffer(frames), dtype=np.float32)
        audio = self.resampler.resample_chunk(samples)
        self.feed((np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes())

    def close(self) -> None:
        self.on = False
        self.engine.stop()
        self.engine.inputNode().removeTapOnBus_(0)
