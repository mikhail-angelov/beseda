"""Pluggable TTS engines for RealtimeTTS. Every engine emits 16-bit mono PCM, so playback,
logging and dialog recording work the same for all of them.

SayEngine replaces RealtimeTTS's SystemEngine: its pyttsx3 driver silently produces empty audio
on macOS after the first runAndWait(), so only the first sentence of the session was ever spoken.
"""

import asyncio
import io
import logging
import os
import subprocess
import tempfile
import time
import urllib.request
import wave
from pathlib import Path

import pyaudio
from RealtimeTTS.engines.base_engine import BaseEngine

log = logging.getLogger("pogo.tts")

MODELS_DIR = Path.home() / ".pogo" / "models"


class PcmEngine(BaseEngine):
    """Base for engines that render a whole sentence to PCM at `rate` Hz."""

    rate = 22050

    def __init__(self, voice: str):
        self.voice = voice

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
    """Silero v5.5: local neural Russian on CPU, ~0.1 s per sentence. Non-commercial license."""

    rate = 24000
    model_url = "https://models.silero.ai/models/tts/ru/v5_5_ru.pt"

    def __init__(self, voice: str):
        super().__init__(voice)
        import torch

        path = MODELS_DIR / Path(self.model_url).name
        if not path.exists():
            log.info("downloading %s", self.model_url)
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(self.model_url, path)
        torch.set_num_threads(4)
        self.model = torch.package.PackageImporter(str(path)).load_pickle("tts_models", "model")
        self.render("Привет.")  # warm-up: the first call is ~10x slower

    def render(self, text: str) -> bytes:
        audio = self.model.apply_tts(text=text, speaker=self.voice, sample_rate=self.rate)
        return (audio.clamp(-1, 1) * 32767).short().numpy().tobytes()


class PiperEngine(PcmEngine):
    """Piper (ONNX): local and very fast, noticeably more synthetic than Silero."""

    rate = 22050

    def __init__(self, voice: str):
        super().__init__(voice)
        from piper import PiperVoice
        from piper.download_voices import download_voice

        path = MODELS_DIR / f"{voice}.onnx"
        if not path.exists():
            log.info("downloading piper voice %s", voice)
            download_voice(voice, MODELS_DIR)
        self.model = PiperVoice.load(str(path))
        self.rate = self.model.config.sample_rate
        self.render("Привет.")

    def render(self, text: str) -> bytes:
        return b"".join(chunk.audio_int16_bytes for chunk in self.model.synthesize(text))


class EdgeEngine(PcmEngine):
    """Microsoft Edge "Read aloud" neural voices: very natural, but cloud and unofficial."""

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


# engine name -> (class, voices; the first voice is the default)
TTS_ENGINES: dict[str, tuple[type[PcmEngine], list[str]]] = {
    "silero": (SileroEngine, ["xenia", "baya", "kseniya", "aidar", "eugene"]),
    "edge": (EdgeEngine, ["ru-RU-SvetlanaNeural", "ru-RU-DmitryNeural"]),
    "piper": (PiperEngine, ["ru_RU-irina-medium", "ru_RU-denis-medium", "ru_RU-dmitri-medium", "ru_RU-ruslan-medium"]),
    "say": (SayEngine, ["Milena"]),
}


def create_engine(name: str, voice: str | None = None) -> PcmEngine:
    cls, voices = TTS_ENGINES[name]
    started = time.monotonic()
    engine = cls(voice or voices[0])
    log.info("tts engine=%s voice=%s ready in %.1fs", name, engine.voice, time.monotonic() - started)
    return engine
