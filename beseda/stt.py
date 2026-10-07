"""Speech recognition: GigaAM for Russian, whisper.cpp with the settings Vadic measured on real dictations."""

import logging
import re
from pathlib import Path

from RealtimeSTT.transcription_engines.factory import ENGINE_CLASS_PATHS

from beseda import gigaam
from beseda.language import Language
from beseda.models import MODELS_DIR, cached

log = logging.getLogger("beseda.stt")

# RealtimeSTT creates engines by name from this table, in a worker process on macOS; that process imports the app
# again, so the entry is there too.
ENGINE_CLASS_PATHS["gigaam"] = ("beseda.gigaam", "GigaAMEngine")

SUPPORT = Path.home() / "Library" / "Application Support"

# Models other apps on this Mac already downloaded are reused instead of fetched again.
MODELS = {
    "turbo": (
        "ggml-large-v3-turbo-q5_0.bin",
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin",
        [SUPPORT / "com.prakashjoshipax.VoiceInk" / "WhisperModels"],
    ),
    "vad": (
        "ggml-silero-v5.1.2.bin",
        "https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v5.1.2.bin",
        [SUPPORT / "Vadic" / "Models"],
    ),
}


def model_path(alias: str) -> str:
    name, url, elsewhere = MODELS[alias]
    for directory in elsewhere:
        if (directory / name).exists():
            return str(directory / name)
    return str(cached(url))


def whisper_model(name: str) -> str:
    """`turbo` is large-v3-turbo-q5_0: far more accurate, ~4x slower than `small` on M1."""
    return model_path(name) if name in MODELS else name


def recorder_options(stt: str, vocabulary: list[str], language: Language) -> dict:
    if stt == "gigaam":
        if language.code != "ru":
            raise SystemExit("--stt gigaam recognizes Russian only")
        if vocabulary:
            log.warning("vocabulary is ignored: GigaAM takes no prompt")
        gigaam.download(MODELS_DIR)
        return {
            "transcription_engine": "gigaam",
            "model": gigaam.MODEL,
            "download_root": str(MODELS_DIR),
            "language": language.code,
            "normalize_audio": True,
        }
    # The style prompt is a neutral sentence in the spoken language: it brings punctuation, capitals and "ё"
    # without biasing the words; the vocabulary keeps the spelling of terms.
    terms = ", ".join(vocabulary) + "." if vocabulary else ""
    return {
        "transcription_engine": "whisper_cpp",  # Metal on Apple Silicon, ~2x faster than faster-whisper on CPU
        "model": whisper_model(stt),
        "download_root": str(MODELS_DIR),  # Whisper models next to the others, not in pywhispercpp's cache
        "language": language.code,
        "beam_size": 1,  # greedy, as in Vadic: same text as beam 5 on real phrases
        "normalize_audio": True,  # quiet microphones make Whisper drop words
        "initial_prompt": f"{language.style_prompt} {terms}".strip(),
        "transcription_engine_options": {
            "model": {"redirect_whispercpp_logs_to": None},
            # Whisper's own VAD: without it a cough or a click becomes "Спасибо." or "Пока." (a stop phrase).
            "transcribe": {"temperature": 0.0, "vad": True, "vad_model_path": model_path("vad")},
        },
    }


def clean(text: str) -> str:
    """Drop Whisper's non-speech annotations: "[музыка]", "*звук*", "[BLANK_AUDIO]"."""
    return re.sub(r"\s+", " ", re.sub(r"\[[^\]]*\]|\*[^*]*\*", "", text)).strip()
