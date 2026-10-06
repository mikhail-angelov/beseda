"""Voice conversations with your coding agent: say the wake word, talk, listen to the answer."""

import argparse
import logging
import math
import os
import queue
import subprocess
import sys
import termios
import threading
import time
import tomllib
import tty
from datetime import datetime
from pathlib import Path

from rich.console import Console, Group
from rich.live import Live
from rich.markup import escape
from rich.text import Text
from RealtimeSTT import AudioToTextRecorder
from RealtimeTTS import TextToAudioStream

from beseda.brains import BRAINS
from beseda.language import Language, available
from beseda.language import load as load_language
from beseda.recording import DialogRecorder
from beseda.stt import clean, recorder_options
from beseda.tts import TTS_ENGINES, create_engine, speakable
from beseda.wake import is_hold, is_stop, strip_wake, wake_pattern

log = logging.getLogger("beseda")

LOG_DIR = Path.home() / ".beseda" / "logs"
RECORDINGS_DIR = Path.home() / "Downloads"
CONFIG_PATH = Path.home() / ".beseda" / "config.toml"
SESSION_START_SOUND = "/System/Library/Sounds/Tink.aiff"
SESSION_END_SOUND = "/System/Library/Sounds/Bottle.aiff"


def setup_logging(debug: bool, keep_days: float) -> Path:
    """One file per session; nothing goes to the terminal so the UI stays intact.
    Logs hold everything said in the dialog, so sessions older than `keep_days` are deleted."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    cutoff = time.time() - keep_days * 86400
    for old in LOG_DIR.glob("beseda-*.log"):
        if old.stat().st_mtime < cutoff:
            old.unlink()
    path = LOG_DIR / f"beseda-{datetime.now():%Y%m%d-%H%M%S}.log"
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


class App:
    def __init__(self, args: argparse.Namespace, language: Language, log_path: Path):
        self.lang = language
        self.console = Console()
        self.status = "idle"
        self.detail = ""
        self.partial = ""
        self.live = Live(self._render(), console=self.console, auto_refresh=False)
        self.active = threading.Event()
        self.interrupted = threading.Event()
        self.stopping = threading.Event()
        self.in_listen = False
        self.turn: Turn | None = None
        self.spoken_chars = 0
        self.wake = wake_pattern(args.wake_word, language.wake_endings) if args.wake_word else None
        self.wake_word = args.wake_word
        self.follow_up = args.follow_up
        self.hold = args.hold
        # None: standby, waiting for the wake word; inf: answering; otherwise the follow-up deadline.
        self.session_until: float | None = None
        self.speech_started_at = 0.0

        log.info("start args=%s", vars(args))
        with self.console.status(language.text("loading")):
            started = time.monotonic()
            self.brain = BRAINS[args.brain](args.model, language.voice_prompt)
            engine = create_engine(args.tts, language, args.voice)
            self.dialog = DialogRecorder(Path(args.record), engine.rate) if args.record else None
            self.tts = TextToAudioStream(
                engine,
                language=language.code,
                on_audio_stream_start=self._on_audio_start,
                on_audio_stream_stop=self._on_audio_stop,
            )
            self.recorder = AudioToTextRecorder(
                **recorder_options(args.whisper, args.vocabulary, language),
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
        self.console.print(f"[dim]{escape(language.text('log', path=log_path))}[/dim]")
        if self.wake:
            self.console.print(language.text("intro", wake=self.wake_word.capitalize(), follow_up=f"{self.follow_up:g}"))
            self.active.set()  # wake word mode listens from the start, like a smart speaker

    # --- UI -----------------------------------------------------------------

    def _render(self) -> Group:
        status = Text(self.lang.text(f"status.{self.status}"), style="bold")
        if self.detail:
            status.append(f"  {self.detail}", style="dim")
        parts = [Text(self.partial[-400:], style="cyan")] if self.partial else []
        return Group(*parts, status, Text(self.lang.text("hint"), style="dim"))

    def refresh(self) -> None:
        self.live.update(self._render(), refresh=True)

    def set_status(self, status: str, detail: str = "") -> None:
        if status != self.status:
            log.debug("status %s -> %s %s", self.status, status, detail)
        self.status, self.detail = status, detail
        self.refresh()

    def show(self, markup: str) -> None:
        self.live.console.print(markup)

    def show_user(self, text: str, note: str = "") -> None:
        suffix = f" [dim]{escape(note)}[/dim]" if note else ""
        self.show(f"[bold green]{escape(self.lang.text('you'))}[/] {escape(text)}{suffix}")

    def duration(self, seconds: float) -> str:
        return self.lang.text("minutes", n=seconds / 60) if seconds >= 60 else self.lang.text("seconds", n=seconds)

    # --- recorder / tts callbacks -------------------------------------------

    def _on_speech_start(self) -> None:
        self.speech_started_at = time.monotonic()
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
        while not self.stopping.is_set():
            self.active.wait()
            try:
                self.listen_and_reply()
            except Exception as error:
                log.exception("turn failed")
                self.show(f"[red]{escape(self.lang.text('error_see_log', error=error))}[/]")

    def idle_status(self) -> str:
        return "standby" if self.wake and self.session_until is None else "listening"

    def listen_and_reply(self) -> None:
        self.set_status(self.idle_status())
        self.recorder.clear_audio_queue()
        self.recorder.set_microphone(True)
        log.info("listening session=%s", self.session_until is not None)
        self.in_listen = True
        text = clean(self.recorder.text())
        self.in_listen = False
        self.recorder.set_microphone(False)  # the assistant must not hear itself
        if self.stopping.is_set():
            return
        if not (self.active.is_set() and text):
            log.info("nothing to answer: active=%s text=%r", self.active.is_set(), text)
            self.turn = None
            return
        self.turn = self.turn or Turn()
        self.turn.mark("transcribed", f"{text!r}")

        if self.wake:
            # Speech that began after the follow-up window closed needs the wake word again.
            if self.session_until is not None and self.speech_started_at > self.session_until:
                self.end_session("timeout")
            addressed = strip_wake(text, self.wake)
            if self.session_until is None:
                if addressed is None:
                    log.info("ignored (no wake word): %r", text)
                    self.show(f"[dim]{escape(self.lang.text('ignored', text=text[:80]))}[/dim]")
                    self.turn = None
                    return
                self.start_session()
            if addressed is not None:
                text = addressed
            if not text:  # just the wake word: wait for the request itself
                self.session_until = time.monotonic() + self.follow_up
                self.turn = None
                return
            if is_stop(text, self.lang):
                self.show_user(text)
                self.end_session("stop")
                self.turn = None
                return
            if is_hold(text, self.lang):
                log.info("hold for %gs: %r", self.hold, text)
                self.show_user(text, self.lang.text("waiting", duration=self.duration(self.hold)))
                self.session_until = time.monotonic() + self.hold
                self.turn = None
                return
            self.session_until = math.inf

        self.reply(text)
        if self.session_until is not None:
            self.session_until = time.monotonic() + self.follow_up

    def start_session(self) -> None:
        log.info("session start")
        self.session_until = math.inf
        subprocess.run(["afplay", SESSION_START_SOUND])

    def end_session(self, reason: str) -> None:
        if self.session_until is None:
            return
        log.info("session end reason=%s", reason)
        self.session_until = None
        self.show(f"[dim]{escape(self.lang.text('session_end', reason=self.lang.text(f'end_reason.{reason}')))}[/dim]")
        subprocess.Popen(["afplay", SESSION_END_SOUND])
        if self.status == "listening":
            self.set_status("standby")

    def session_timer(self) -> None:
        """Closes the follow-up window and shows the countdown while waiting for the next phrase."""
        while True:
            time.sleep(0.25)
            until = self.session_until
            if until is None or math.isinf(until) or self.status != "listening":
                continue
            remaining = until - time.monotonic()
            if remaining <= 0:
                self.end_session("timeout")
            else:
                seconds = math.ceil(remaining)
                left = f"{seconds // 60}:{seconds % 60:02d}" if seconds > 60 else self.lang.text("seconds", n=seconds)
                self.set_status("listening", self.lang.text("remaining", time=left))

    def reply(self, text: str) -> None:
        turn = self.turn or Turn()
        self.turn = turn
        self.show_user(text)
        self.interrupted.clear()
        self.spoken_chars = 0
        self.set_status("thinking")

        to_speak: queue.Queue[str | None] = queue.Queue()
        self.tts.feed(speakable(iter(to_speak.get, None)))
        player = threading.Thread(
            target=self.tts.play,
            name="tts-play",
            kwargs={
                "language": self.lang.code,
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
                self.show(f"[red]{escape(self.lang.text('error', error=value))}[/]")
        turn.mark("brain_done", f"chars={len(answer)}")
        to_speak.put(None)
        player.join()
        turn.mark("done")

        self.partial = ""
        interrupted = self.interrupted.is_set()
        if answer.strip():
            suffix = f" [dim]{escape(self.lang.text('interrupted'))}[/]" if interrupted else ""
            self.show(f"[bold cyan]{escape(self.lang.text('assistant'))}[/] {escape(answer.strip())}{suffix}")
            if not interrupted and "audio_start" not in turn.marks:
                log.error(
                    "turn=%d answer has %d chars, %d sent to TTS, but no audio was played",
                    turn.id, len(answer), self.spoken_chars,
                )
                self.show(f"[red]{escape(self.lang.text('not_spoken'))}[/]")
        log.info("turn=%d summary %s interrupted=%s", turn.id, turn.summary(), interrupted)
        self.show(f"[dim]⏱ {turn.summary()}[/dim]")
        self.turn = None

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
            self.end_session("pause")
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
        if self.wake:
            threading.Thread(target=self.session_timer, name="session-timer", daemon=True).start()
        try:
            with self.live:
                while (key := os.read(fd, 1)) not in (b"q", b"Q"):
                    if key == b" ":
                        self.toggle()
                    elif key == b"\x1b":
                        self.interrupt()
        finally:
            log.info("shutdown")
            self.stopping.set()
            termios.tcsetattr(fd, termios.TCSADRAIN, saved)
            self.brain.close()
            self.tts.stop()
            # RealtimeSTT stops its mic reader process only while the mic flag is on; otherwise the orphaned
            # reader keeps the microphone (and the macOS mic indicator) busy after exit.
            self.recorder.set_microphone(True)
            self.recorder.shutdown()
            if self.dialog:
                self.dialog.save()
                if self.dialog.path.exists():
                    self.console.print(escape(self.lang.text("recording_saved", path=self.dialog.path)))


def main() -> None:
    parser = argparse.ArgumentParser(prog="beseda", description=__doc__, epilog=f"Defaults for any option: {CONFIG_PATH}")
    parser.add_argument("--language", choices=available(), default="ru", help="language pack: what you speak and hear")
    parser.add_argument("--brain", choices=BRAINS, default="pi")
    parser.add_argument("--model", help="model for the brain (default: DeepSeek V4 Flash)")
    parser.add_argument("--whisper", default="small", help="Whisper model: small (fast) or turbo (more accurate, ~2 s per phrase)")
    parser.add_argument("--vocabulary", nargs="*", default=[], metavar="TERM", help="terms Whisper should spell exactly like this")
    parser.add_argument("--tts", choices=TTS_ENGINES, default="silero", help="speech synthesis engine")
    parser.add_argument("--voice", help="TTS voice (default: the first one in the language pack)")
    parser.add_argument("--wake-word", help='wake word (default: from the language pack); "" answers everything')
    parser.add_argument("--follow-up", type=float, default=8, help="seconds after an answer to keep talking without the wake word")
    parser.add_argument("--hold", type=float, default=120, help="seconds a hold phrase («подожди», “hold on”) extends the wait")
    parser.add_argument(
        "--record",
        nargs="?",
        const=True,
        metavar="PATH",
        help="record the whole dialog to a WAV with the real pauses (default: ~/Downloads/)",
    )
    parser.add_argument("--log-days", type=float, default=14, help="days to keep logs (they hold the full dialog text)")
    parser.add_argument("--debug", action="store_true", help="verbose log: every pi event and status change")
    if CONFIG_PATH.exists():
        config = {key.replace("-", "_"): value for key, value in tomllib.loads(CONFIG_PATH.read_text()).items()}
        known = {action.dest for action in parser._actions}
        if unknown := set(config) - known:
            parser.error(f"{CONFIG_PATH}: unknown keys {sorted(unknown)}")
        parser.set_defaults(**config)
    args = parser.parse_args()
    language = load_language(args.language)
    if args.wake_word is None:
        args.wake_word = language.wake_word
    if args.record is True:
        args.record = str(RECORDINGS_DIR / f"beseda-{datetime.now():%Y%m%d-%H%M%S}.wav")
    App(args, language, setup_logging(args.debug, args.log_days)).run()


if __name__ == "__main__":
    main()
