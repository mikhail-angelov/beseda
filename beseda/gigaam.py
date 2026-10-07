"""GigaAM v3 (Sber) as a RealtimeSTT transcription engine: Russian only, with punctuation and capitals.

On an M1 CPU it transcribes a phrase in 0.1–0.3 s, 2–5x faster than whisper.cpp `small` on Metal, and it doesn't
pad every phrase to Whisper's 30-second window. CoreML can't run it, so it stays on the CPU.
"""

from pathlib import Path

import numpy as np
import onnx_asr
from huggingface_hub import snapshot_download
from RealtimeSTT.transcription_engines.base import BaseTranscriptionEngine, TranscriptionInfo, TranscriptionResult

MODEL = "gigaam-v3-e2e-ctc"  # same accuracy as the RNNT variant here, simpler decoding
REPO = "istupakov/gigaam-v3-onnx"
FILES = ["config.json", "v3_e2e_ctc.onnx", "v3_e2e_ctc_vocab.txt"]


def download(models_dir: Path) -> None:
    """Fetches the model (~890 MB) in the app's own process, before the recorder starts.

    RealtimeSTT loads engines in a worker process that only logs a failure, so a download failing there would leave
    the app loading forever; here it stops the app with the error."""
    path = models_dir / MODEL
    if not all((path / name).exists() for name in FILES):
        snapshot_download(REPO, local_dir=path, allow_patterns=FILES)


class GigaAMEngine(BaseTranscriptionEngine):
    engine_name = "gigaam"

    def __init__(self, config):
        super().__init__(config)
        path = Path(config.download_root) / config.model if config.download_root else None
        self.model = onnx_asr.load_model(config.model, path, providers=["CPUExecutionProvider"])

    def transcribe(self, audio, language=None, use_prompt=True, **kwargs) -> TranscriptionResult:
        audio = self._normalize_audio(np.asarray(audio, dtype=np.float32))
        text = self.model.recognize(audio, sample_rate=16000)
        return TranscriptionResult(text=text, info=TranscriptionInfo(language="ru", language_probability=1.0))
