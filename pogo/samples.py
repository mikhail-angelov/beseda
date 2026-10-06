"""Speak the same phrase with every TTS engine and voice to compare them by ear.

uv run python -m pogo.samples [--engines silero edge] [--text "..."]
"""

import argparse
import time
import wave
from pathlib import Path

from rich.console import Console
from rich.table import Table

from pogo.tts import TTS_ENGINES

OUT_DIR = Path.home() / "Downloads" / "pogo-tts-samples"
TEXT = (
    "Привет! Я посмотрел проект: в папке четыре файла, и все тесты проходят. "
    "Хочешь, расскажу подробнее, что именно я поменял и почему?"
)


def main() -> None:
    parser = argparse.ArgumentParser(prog="pogo.samples", description=__doc__)
    parser.add_argument("--engines", nargs="+", choices=TTS_ENGINES, default=list(TTS_ENGINES))
    parser.add_argument("--text", default=TEXT)
    args = parser.parse_args()

    console = Console()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table = Table("файл", "загрузка", "синтез", "длина аудио", "скорость")
    for name in args.engines:
        cls, voices = TTS_ENGINES[name]
        for voice in voices:
            try:
                with console.status(f"{name} / {voice}…"):
                    started = time.monotonic()
                    engine = cls(voice)
                    loaded = time.monotonic() - started
                    started = time.monotonic()
                    audio = engine.render(args.text)
                    synth = time.monotonic() - started
            except Exception as error:
                console.print(f"[red]{name} / {voice}: {error}[/]")
                continue
            path = OUT_DIR / f"{name}-{voice}.wav"
            with wave.open(str(path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(engine.rate)
                wf.writeframes(audio)
            duration = len(audio) / 2 / engine.rate
            table.add_row(path.name, f"{loaded:.1f}s", f"{synth:.2f}s", f"{duration:.1f}s", f"×{duration / synth:.0f}")
    console.print(table)
    console.print(f"Файлы: {OUT_DIR}")


if __name__ == "__main__":
    main()
