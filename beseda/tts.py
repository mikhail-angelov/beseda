"""Pluggable TTS engines for RealtimeTTS. Every engine emits 16-bit mono PCM, so playback,
logging and dialog recording work the same for all of them.

SayEngine replaces RealtimeTTS's SystemEngine: its pyttsx3 driver silently produces empty audio
on macOS after the first runAndWait(), so only the first sentence of the session was ever spoken.
"""

import asyncio
import io
import logging
import os
import re
import subprocess
import tempfile
import time
import urllib.request
import wave
from pathlib import Path
from typing import Iterator

import pyaudio
from RealtimeTTS.engines.base_engine import BaseEngine

from beseda.language import Language

log = logging.getLogger("beseda.tts")

MODELS_DIR = Path.home() / ".beseda" / "models"


def speakable(deltas: Iterator[str]) -> Iterator[str]:
    """Drop fenced code blocks and markdown symbols so the TTS reads only prose."""
    buf, in_code = "", False
    for delta in deltas:
        buf += delta
        out = ""
        while (i := buf.find("```")) >= 0:
            if not in_code:
                out += buf[:i]
            buf, in_code = buf[i + 3 :], not in_code
        # Hold trailing backticks back: they may be the start of a fence split across deltas.
        tail = len(buf) - len(buf.rstrip("`"))
        if not in_code:
            out += buf[: len(buf) - tail]
        buf = buf[len(buf) - tail :]
        out = re.sub(r"[*_#`>|]", "", out)
        if out:
            yield out


class PcmEngine(BaseEngine):
    """Base for engines that render a whole sentence to PCM at `rate` Hz."""

    rate = 22050

    def __init__(self, voice: str, language: Language):
        self.voice = voice
        self.language = language

    def post_init(self) -> None:
        self.engine_name = type(self).__name__

    def get_stream_info(self):
        return pyaudio.paInt16, 1, self.rate

    def render(self, text: str) -> bytes:
        raise NotImplementedError

    def synthesize(self, text: str, sentence_count: int = 0) -> bool:
        super().synthesize(text, sentence_count)
        started = time.monotonic()
        try:
            audio = self.render(text)
        except Exception:
            log.exception("%s failed text=%r", self.engine_name, text)
            return False
        elapsed = time.monotonic() - started
        if not audio:
            log.error("%s produced empty audio in %.2fs text=%r", self.engine_name, elapsed, text)
            return False
        log.info(
            "%s synthesized sentence=%d in %.2fs audio=%.2fs chars=%d",
            self.engine_name, sentence_count, elapsed, len(audio) / 2 / self.rate, len(text),
        )
        self.queue.put(audio)
        return True

    def set_voice(self, voice: str) -> None:
        self.voice = voice


class SayEngine(PcmEngine):
    """macOS `say`: local, instant, robotic unless an Enhanced/Premium voice is installed."""

    rate = 22050

    def render(self, text: str) -> bytes:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "speech.wav")
            subprocess.run(
                ["say", "-v", self.voice, "-o", path, f"--data-format=LEI16@{self.rate}", text],
                check=True, capture_output=True,
            )
            with wave.open(path, "rb") as wf:
                return wf.readframes(wf.getnframes())


class SileroEngine(PcmEngine):
    """Silero: local neural voices on CPU, ~0.05 s per sentence. Non-commercial license."""

    rate = 24000

    def __init__(self, voice: str, language: Language):
        super().__init__(voice, language)
        import torch

        url = language.silero_model
        path = MODELS_DIR / Path(url).name
        if not path.exists():
            log.info("downloading %s", url)
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(url, path)
        torch.set_num_threads(4)
        self.model = torch.package.PackageImporter(str(path)).load_pickle("tts_models", "model")
        self.render(language.greeting)  # warm-up: the first call is ~10x slower

    def render(self, text: str) -> bytes:
        audio = self.model.apply_tts(text=text, speaker=self.voice, sample_rate=self.rate)
        return (audio.clamp(-1, 1) * 32767).short().numpy().tobytes()


class EdgeEngine(PcmEngine):
    """Microsoft Edge "Read aloud" neural voices: very natural, but cloud, unofficial and experimental."""

    rate = 24000

    def render(self, text: str) -> bytes:
        import edge_tts
        from pydub import AudioSegment

        async def fetch() -> bytes:
            stream = edge_tts.Communicate(text, self.voice).stream()
            return b"".join([chunk["data"] async for chunk in stream if chunk["type"] == "audio"])

        mp3 = asyncio.run(fetch())
        audio = AudioSegment.from_file(io.BytesIO(mp3), format="mp3")
        return audio.set_frame_rate(self.rate).set_channels(1).set_sample_width(2).raw_data


# Voices per engine come from the language pack; the first one is the default.
TTS_ENGINES: dict[str, type[PcmEngine]] = {"silero": SileroEngine, "edge": EdgeEngine, "say": SayEngine}


def create_engine(name: str, language: Language, voice: str | None = None) -> PcmEngine:
    started = time.monotonic()
    engine = TTS_ENGINES[name](voice or language.voices[name][0], language)
    log.info("tts engine=%s voice=%s ready in %.1fs", name, engine.voice, time.monotonic() - started)
    return engine
