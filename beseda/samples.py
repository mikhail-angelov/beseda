"""Speak the same phrase with every TTS engine and voice to compare them by ear.

beseda-samples [--language en] [--engines silero edge] [--text "..."]
"""

import argparse
import time
import wave
from pathlib import Path

from rich.console import Console
from rich.table import Table

from beseda.language import available, load
from beseda.tts import TTS_ENGINES

OUT_DIR = Path.home() / "Downloads" / "beseda-tts-samples"


def main() -> None:
    parser = argparse.ArgumentParser(prog="beseda.samples", description=__doc__)
    parser.add_argument("--language", choices=available(), default="ru")
    parser.add_argument("--engines", nargs="+", choices=TTS_ENGINES, default=list(TTS_ENGINES))
    parser.add_argument("--text", help="phrase to speak (default: the language pack's sample)")
    args = parser.parse_args()
    language = load(args.language)
    text = args.text or language.sample_text

    console = Console()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table = Table("file", "load", "synthesis", "audio", "speed")
    for name in args.engines:
        for voice in language.voices[name]:
            try:
                with console.status(f"{name} / {voice}…"):
                    started = time.monotonic()
                    engine = TTS_ENGINES[name](voice, language)
                    loaded = time.monotonic() - started
                    started = time.monotonic()
                    audio = engine.render(text)
                    synth = time.monotonic() - started
            except Exception as error:
                console.print(f"[red]{name} / {voice}: {error}[/]")
                continue
            path = OUT_DIR / f"{language.code}-{name}-{voice}.wav"
            with wave.open(str(path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(engine.rate)
                wf.writeframes(audio)
            duration = len(audio) / 2 / engine.rate
            table.add_row(path.name, f"{loaded:.1f}s", f"{synth:.2f}s", f"{duration:.1f}s", f"×{duration / synth:.0f}")
    console.print(table)
    console.print(f"Files: {OUT_DIR}")


if __name__ == "__main__":
    main()
