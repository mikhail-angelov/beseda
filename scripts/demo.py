"""Records the README demo: the real app (recognition, pi, Silero) in a terminal, with the user's phrases
spoken by a second Silero voice instead of a microphone.

    brew install agg ffmpeg
    uv run python scripts/demo.py [--language ru] [--project PATH]

Writes docs/demo-<language>.gif (silent, long pauses shortened) and docs/demo-<language>.mp4 (real time, with sound).
The assistant's answers play through the speakers while it records.
"""

import argparse
import codecs
import fcntl
import functools
import json
import os
import pty
import select
import signal
import struct
import subprocess
import sys
import tempfile
import termios
import threading
import time
import traceback
from pathlib import Path

COLS, ROWS = 100, 24
DEADLINE_SECONDS = 300  # the whole scenario; a stuck agent or VAD must not hang the script
DOCS = Path(__file__).resolve().parent.parent / "docs"

# (pause before speaking, phrase): the pause is the "user" thinking after the assistant finishes.
SCENARIOS = {
    "ru": [
        (1.5, "Какая сегодня погода?"),
        (1.5, "Вика, сколько файлов с кодом на питоне в этом проекте?"),
        (1.2, "Подожди, дай подумать."),
        (3.0, "А какой из них самый большой?"),
        (1.2, "Спасибо, всё."),
    ],
    "en": [
        (1.5, "What's the weather like today?"),
        (1.5, "Alice, how many Python files are in this project?"),
        (1.2, "Hold on, let me think."),
        (3.0, "Which one is the biggest?"),
        (1.2, "Thanks, that's all."),
    ],
}
USER_VOICE = {"ru": "aidar", "en": "en_12"}
END_MARK = {"ru": "разговор завершён", "en": "conversation ended"}


def child(language_code: str, marker: Path, recording: Path) -> None:
    """Runs inside the pseudo-terminal: the app with a recorder fed by synthesized speech."""
    import numpy as np
    import resampy
    from RealtimeSTT import AudioToTextRecorder

    import beseda.__main__ as app_module
    from beseda.language import load
    from beseda.recording import PRE_ROLL_SECONDS
    from beseda.tts import SileroEngine

    language = load(language_code)
    app_module.AudioToTextRecorder = functools.partial(AudioToTextRecorder, use_microphone=False)
    args = argparse.Namespace(
        language=language_code, brain="pi", model=None, whisper="small", vocabulary=[], tts="silero", voice=None,
        wake_word=language.wake_word, follow_up=8.0, hold=120.0, record=str(recording), log_days=14, debug=False,
    )
    app = app_module.App(args, language, app_module.setup_logging(False, 14))
    voice = SileroEngine(USER_VOICE[language_code], language)

    pending = bytearray()  # phrase audio waiting to go into the "microphone"
    lock = threading.Lock()

    def microphone() -> None:
        """Feeds audio continuously like a real microphone: the phrase if one is pending, silence otherwise.
        Without a steady stream the VAD never sees the silence that ends a phrase."""
        chunk = 1024 * 2
        while True:
            with lock:
                data = bytes(pending[:chunk])
                del pending[:chunk]
            app.recorder.feed_audio(data + bytes(chunk - len(data)))
            time.sleep(chunk / 2 / 16000)

    def speak(text: str) -> None:
        audio = np.frombuffer(voice.render(text), np.int16).astype(np.float32)
        pcm = resampy.resample(audio, voice.rate, 16000).astype(np.int16).tobytes()
        with lock:
            pending.extend(pcm)
        while pending:
            time.sleep(0.05)

    def idle() -> bool:
        return app.status in ("standby", "listening") and app.turn is None

    def feeder() -> None:
        try:
            for pause, phrase in SCENARIOS[language_code]:
                waited = time.monotonic()
                while not idle():
                    if time.monotonic() - waited > 120:
                        raise TimeoutError(f"the app stayed in {app.status!r} before {phrase!r}")
                    time.sleep(0.1)
                time.sleep(pause)
                speak(phrase)
                time.sleep(0.5)
            # Wall-clock time where the dialog recording starts, to line the audio up with the terminal video.
            dialog = app.dialog
            start = dialog.t0 + dialog.first_speech - PRE_ROLL_SECONDS
            marker.write_text(json.dumps({"audio_start": time.time() - (time.monotonic() - start)}))
        except Exception:
            traceback.print_exc()
            os._exit(1)  # the parent sees the child exit without a marker and reports the failure

    threading.Thread(target=microphone, daemon=True).start()
    threading.Thread(target=feeder, daemon=True).start()
    app.run()


def record(language: str, project: Path, work: Path) -> tuple[Path, float, float]:
    """Runs the child in a pseudo-terminal and writes an asciicast; returns it with the cast and audio start times."""
    marker, recording, cast = work / "marker.json", work / "dialog.wav", work / "demo.cast"
    started = time.time()
    pid, fd = pty.fork()
    if pid == 0:
        os.chdir(project)
        os.environ.update(TERM="xterm-256color", COLUMNS=str(COLS), LINES=str(ROWS))
        os.execv(sys.executable, [sys.executable, __file__, "--child", language, str(marker), str(recording)])
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", ROWS, COLS, 0, 0))

    events, output, quit_at, quit_sent = [], "", None, None
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")  # a read may split a multibyte character
    while True:
        now = time.time()
        if now - started > DEADLINE_SECONDS:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
            raise SystemExit(f"demo: no end of the conversation after {DEADLINE_SECONDS} s; see ~/.beseda/logs")
        if quit_at is not None and now >= quit_at:
            os.write(fd, b"q")  # the app saves the dialog recording on exit
            quit_sent, quit_at = now, float("inf")
        if not select.select([fd], [], [], 0.2)[0]:
            continue
        try:
            data = os.read(fd, 65536)
        except OSError:  # the child exited
            break
        if not data:
            break
        text = decoder.decode(data)
        events.append([round(now - started, 3), "o", text])
        output += text
        if quit_at is None and END_MARK[language] in output:
            quit_at = now + 2.5
    os.waitpid(pid, 0)
    events = [event for event in events if event[0] <= quit_sent - started]  # the exit itself isn't part of the demo
    header = {"version": 2, "width": COLS, "height": ROWS, "timestamp": int(started)}
    cast.write_text("\n".join(json.dumps(line, ensure_ascii=False) for line in [header, *events]) + "\n")
    if not marker.exists():
        raise SystemExit("demo: the scenario failed:\n" + output[-2000:])
    audio_start = json.loads(marker.read_text())["audio_start"] - started
    intro = next(t for t, _, text in events if "Вика" in text or "Alice" in text)
    return cast, intro, audio_start


def trim(cast: Path, start: float, out: Path) -> None:
    """Drops the model loading before `start`, keeping later events with shifted times."""
    lines = cast.read_text().splitlines()
    events = [json.loads(line) for line in lines[1:]]
    kept = [[round(t - start, 3), kind, text] for t, kind, text in events if t >= start]
    out.write_text("\n".join([lines[0], *(json.dumps(e, ensure_ascii=False) for e in kept)]) + "\n")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child(sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]))
        return
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--language", choices=SCENARIOS, default="ru")
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent.parent, help="folder the agent works in")
    args = parser.parse_args()

    DOCS.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        cast, intro, audio_start = record(args.language, args.project, work)
        start = intro - 1.0  # a moment before the intro line, after the model loading
        trimmed = work / "trimmed.cast"
        trim(cast, start, trimmed)
        agg = ["agg", "--font-size", "18", "--theme", "monokai", "--last-frame-duration", "3"]
        gif, video_frames = DOCS / f"demo-{args.language}.gif", work / "frames.gif"
        subprocess.run([*agg, "--idle-time-limit", "2.5", str(trimmed), str(gif)], check=True)
        subprocess.run([*agg, str(trimmed), str(video_frames)], check=True)
        delay_ms = int((audio_start - start) * 1000)
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error", "-i", str(video_frames), "-i", str(work / "dialog.wav"),
                "-filter_complex", f"[1:a]adelay={delay_ms}:all=1,apad[a]", "-map", "0:v", "-map", "[a]",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-c:a", "aac", "-b:a", "96k", "-shortest", "-movflags", "+faststart", str(DOCS / f"demo-{args.language}.mp4"),
            ],
            check=True,
        )
    for path in sorted(DOCS.glob(f"demo-{args.language}.*")):
        print(f"{path}  {path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
