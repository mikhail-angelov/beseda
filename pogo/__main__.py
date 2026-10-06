"""Voice chat in the terminal: Space starts/stops the conversation, Esc interrupts, q quits."""

import argparse
import logging
import os
import queue
import re
import sys
import termios
import threading
import time
import tty
from datetime import datetime
from pathlib import Path
from typing import Iterator

from rich.console import Console, Group
from rich.live import Live
from rich.markup import escape
from rich.text import Text
from RealtimeSTT import AudioToTextRecorder
from RealtimeTTS import TextToAudioStream

from pogo.brains import BRAINS
from pogo.recording import DialogRecorder
from pogo.tts import TTS_ENGINES, create_engine

log = logging.getLogger("pogo")

STATUS = {
    "idle": "⏸  пауза — Пробел: начать разговор",
    "listening": "🎙  слушаю…",
    "hearing": "🎙  слышу вас…",
    "transcribing": "✍️  распознаю…",
    "thinking": "💭  думаю…",
    "tool": "🔧  выполняю…",
    "speaking": "🔊  говорю…",
}
HINT = "[dim]Пробел — старт/стоп · Esc — перебить · q — выход[/dim]"
LOG_DIR = Path.home() / ".pogo" / "logs"
RECORDINGS_DIR = Path.home() / "Downloads"


def setup_logging(debug: bool) -> Path:
    """One file per session; nothing goes to the terminal so the UI stays intact."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / f"pogo-{datetime.now():%Y%m%d-%H%M%S}.log"
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setLevel(logging.DEBUG if debug else logging.INFO)
    handler.setFormatter(
        logging.Formatter("%(asctime)s.%(msecs)03d %(levelname)-7s [%(threadName)s] %(name)s: %(message)s", "%H:%M:%S")
    )
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.WARNING)
    log.setLevel(logging.DEBUG if debug else logging.INFO)
    threading.excepthook = lambda a: log.critical(
        "uncaught exception in thread %s", a.thread.name, exc_info=(a.exc_type, a.exc_value, a.exc_traceback)
    )
    return path


class Turn:
    """Timeline of one exchange, measured from the moment the user starts speaking."""

    count = 0

    def __init__(self):
        Turn.count += 1
        self.id = Turn.count
        self.t0 = time.monotonic()
        self.marks: dict[str, float] = {}
        self.tools = 0
        self.mark("speech_start")

    def mark(self, name: str, detail: str = "") -> None:
        elapsed = time.monotonic() - self.t0
        first = name not in self.marks
        self.marks.setdefault(name, elapsed)
        if first or detail:
            log.info("turn=%d +%.2fs %s %s", self.id, elapsed, name, detail)

    def span(self, start: str, end: str) -> str:
        if start in self.marks and end in self.marks:
            return f"{self.marks[end] - self.marks[start]:.2f}s"
        return "-"

    def summary(self) -> str:
        return (
            f"stt={self.span('speech_end', 'transcribed')} "
            f"llm_first_token={self.span('prompt_sent', 'first_text')} "
            f"tts_first_audio={self.span('first_text', 'audio_start')} "
            f"voice_to_voice={self.span('speech_end', 'audio_start')} "
            f"tools={self.tools} total={self.span('speech_start', 'done')}"
        )


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


class App:
    def __init__(self, args: argparse.Namespace, log_path: Path):
        self.console = Console()
        self.status = "idle"
        self.detail = ""
        self.partial = ""
        self.live = Live(self._render(), console=self.console, auto_refresh=False)
        self.active = threading.Event()
        self.interrupted = threading.Event()
        self.in_listen = False
        self.turn: Turn | None = None
        self.spoken_chars = 0

        log.info("start args=%s", vars(args))
        with self.console.status("Загрузка…"):
            started = time.monotonic()
            self.brain = BRAINS[args.brain](args.model)
            engine = create_engine(args.tts, args.voice)
            self.dialog = DialogRecorder(Path(args.record), engine.rate) if args.record else None
            self.tts = TextToAudioStream(
                engine,
                language="ru",
                on_audio_stream_start=self._on_audio_start,
                on_audio_stream_stop=self._on_audio_stop,
            )
            self.recorder = AudioToTextRecorder(
                transcription_engine="whisper_cpp",  # Metal on Apple Silicon, ~2x faster than faster-whisper on CPU
                transcription_engine_options={"model": {"redirect_whispercpp_logs_to": None}},
                model=args.whisper,
                language="ru",
                spinner=False,
                level=logging.ERROR,
                no_log_file=True,
                post_speech_silence_duration=0.7,
                on_recording_start=self._on_speech_start,
                on_recording_stop=self._on_speech_end,
                on_recorded_chunk=self.dialog.on_mic if self.dialog else None,
            )
            self.recorder.set_microphone(False)
            log.info("components ready in %.1fs", time.monotonic() - started)
        self.console.print(f"[dim]Лог: {log_path}[/dim]")

    # --- UI -----------------------------------------------------------------

    def _render(self) -> Group:
        status = Text(STATUS[self.status], style="bold")
        if self.detail:
            status.append(f"  {self.detail}", style="dim")
        parts = [Text(self.partial[-400:], style="cyan")] if self.partial else []
        return Group(*parts, status, Text.from_markup(HINT))

    def refresh(self) -> None:
        self.live.update(self._render(), refresh=True)

    def set_status(self, status: str, detail: str = "") -> None:
        if status != self.status:
            log.debug("status %s -> %s %s", self.status, status, detail)
        self.status, self.detail = status, detail
        self.refresh()

    def show(self, markup: str) -> None:
        self.live.console.print(markup)

    # --- recorder / tts callbacks -------------------------------------------

    def _on_speech_start(self) -> None:
        self.turn = Turn()
        if self.dialog:
            self.dialog.on_speech_start()
        self.set_status("hearing")

    def _on_speech_end(self) -> None:
        if self.turn:
            self.turn.mark("speech_end")
        self.set_status("transcribing")

    def _on_audio_start(self) -> None:
        if self.turn:
            self.turn.mark("audio_start")
        self.set_status("speaking")

    def _on_audio_stop(self) -> None:
        if self.turn:
            self.turn.mark("audio_end")

    def _before_sentence(self, sentence: str) -> None:
        self.spoken_chars += len(sentence)
        if self.turn:
            self.turn.mark("sentence_synth_start", f"chars={len(sentence)} {sentence[:60]!r}")

    def _after_sentence(self, sentence: str) -> None:
        if self.turn:
            self.turn.mark("sentence_synth_end", f"chars={len(sentence)}")

    # --- conversation -------------------------------------------------------

    def conversation(self) -> None:
        while True:
            self.active.wait()
            try:
                self.listen_and_reply()
            except Exception as error:
                log.exception("turn failed")
                self.show(f"[red]Ошибка: {escape(str(error))} (подробности в логе)[/]")

    def listen_and_reply(self) -> None:
        self.set_status("listening")
        self.recorder.clear_audio_queue()
        self.recorder.set_microphone(True)
        log.info("listening")
        self.in_listen = True
        text = self.recorder.text()
        self.in_listen = False
        self.recorder.set_microphone(False)  # the assistant must not hear itself
        if not (self.active.is_set() and text.strip()):
            log.info("nothing to answer: active=%s text=%r", self.active.is_set(), text)
            self.turn = None
            return
        self.turn = self.turn or Turn()
        self.turn.mark("transcribed", f"{text!r}")
        self.reply(text.strip())

    def reply(self, text: str) -> None:
        turn = self.turn or Turn()
        self.turn = turn
        self.show(f"[bold green]Вы:[/] {escape(text)}")
        self.interrupted.clear()
        self.spoken_chars = 0
        self.set_status("thinking")

        to_speak: queue.Queue[str | None] = queue.Queue()
        self.tts.feed(speakable(iter(to_speak.get, None)))
        player = threading.Thread(
            target=self.tts.play,
            name="tts-play",
            kwargs={
                "language": "ru",
                "minimum_sentence_length": 15,
                "before_sentence_synthesized": self._before_sentence,
                "on_sentence_synthesized": self._after_sentence,
                "on_audio_chunk": self.dialog.on_voice if self.dialog else None,
            },
        )
        player.start()

        # Drain the brain fully even after an interrupt so its next turn starts clean.
        answer = ""
        turn.mark("prompt_sent")
        for kind, value in self.brain.ask(text):
            turn.mark("brain_first_event")
            if self.interrupted.is_set():
                continue
            if kind == "text":
                turn.mark("first_text")
                answer += value
                self.partial = answer
                to_speak.put(value)
                if self.status != "speaking":
                    self.set_status("thinking")
                else:
                    self.refresh()
            elif kind == "tool":
                turn.tools += 1
                turn.mark("tool", value)
                self.show(f"[yellow]🔧 {escape(value)}[/]")
                self.set_status("tool", value[:60])
            elif kind == "error":
                turn.mark("error", value)
                self.show(f"[red]Ошибка: {escape(value)}[/]")
        turn.mark("brain_done", f"chars={len(answer)}")
        to_speak.put(None)
        player.join()
        turn.mark("done")

        self.partial = ""
        interrupted = self.interrupted.is_set()
        if answer.strip():
            suffix = " [dim](прервано)[/]" if interrupted else ""
            self.show(f"[bold cyan]Ассистент:[/] {escape(answer.strip())}{suffix}")
            if not interrupted and "audio_start" not in turn.marks:
                log.error(
                    "turn=%d answer has %d chars, %d sent to TTS, but no audio was played",
                    turn.id, len(answer), self.spoken_chars,
                )
                self.show("[red]Ответ не озвучен (подробности в логе)[/]")
        log.info("turn=%d summary %s interrupted=%s", turn.id, turn.summary(), interrupted)
        self.show(f"[dim]⏱ {turn.summary()}[/dim]")
        self.turn = None
        self.set_status("listening" if self.active.is_set() else "idle")

    def interrupt(self) -> None:
        log.info("interrupt requested status=%s", self.status)
        self.interrupted.set()
        self.brain.abort()
        self.tts.stop()

    def toggle(self) -> None:
        if self.active.is_set():
            log.info("conversation off")
            self.active.clear()
            self.interrupt()
            self.recorder.set_microphone(False)
            if self.in_listen:
                self.recorder.abort()  # unblock recorder.text(); blocks if called outside of it
            self.set_status("idle")
        else:
            log.info("conversation on")
            self.active.set()

    # --- main loop ----------------------------------------------------------

    def run(self) -> None:
        fd = sys.stdin.fileno()
        saved = termios.tcgetattr(fd)
        tty.setcbreak(fd)  # single keypresses without breaking Rich's line output
        threading.Thread(target=self.conversation, name="conversation", daemon=True).start()
        try:
            with self.live:
                while (key := os.read(fd, 1)) not in (b"q", b"Q"):
                    if key == b" ":
                        self.toggle()
                    elif key == b"\x1b":
                        self.interrupt()
        finally:
            log.info("shutdown")
            termios.tcsetattr(fd, termios.TCSADRAIN, saved)
            self.brain.close()
            self.tts.stop()
            self.recorder.shutdown()
            if self.dialog:
                self.dialog.save()
                if self.dialog.path.exists():
                    self.console.print(f"Запись диалога: {self.dialog.path}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="pogo", description=__doc__)
    parser.add_argument("--brain", choices=BRAINS, default="pi")
    parser.add_argument("--model", help="модель для выбранного brain (по умолчанию DeepSeek V4 Flash)")
    parser.add_argument("--whisper", default="small", help="модель Whisper: small, medium, large-v3-turbo…")
    parser.add_argument("--tts", choices=TTS_ENGINES, default="silero", help="движок синтеза речи")
    parser.add_argument("--voice", help="голос движка (по умолчанию первый из списка в pogo/tts.py)")
    parser.add_argument(
        "--record",
        nargs="?",
        const=str(RECORDINGS_DIR / f"pogo-{datetime.now():%Y%m%d-%H%M%S}.wav"),
        metavar="PATH",
        help="записать весь диалог в WAV с реальными паузами (по умолчанию в ~/Downloads/)",
    )
    parser.add_argument("--debug", action="store_true", help="подробный лог: все события pi и смены статуса")
    args = parser.parse_args()
    App(args, setup_logging(args.debug)).run()


if __name__ == "__main__":
    main()
