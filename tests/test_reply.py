import threading

import pytest

import beseda.__main__ as app
from beseda.language import load


class FakeTTS:
    """Plays like RealtimeTTS: consumes the fed iterator until it ends."""

    def __init__(self):
        self.stopped = False

    def feed(self, iterator):
        self.iterator = iterator

    def play(self, **kwargs):
        for _ in self.iterator:
            pass

    def stop(self):
        self.stopped = True


class FailingBrain:
    def ask(self, text):
        yield "text", "Начинаю отвечать"
        raise ConnectionError("network down")


def test_brain_failure_releases_playback():
    a = object.__new__(app.App)
    a.lang, a.tts, a.brain, a.dialog = load("ru"), FakeTTS(), FailingBrain(), None
    a.turn, a.partial, a.status, a.spoken_chars = None, "", "listening", 0
    a.interrupted, a.lock = threading.Event(), threading.RLock()
    a.show = a.set_status = a.refresh = lambda *args, **kwargs: None

    with pytest.raises(ConnectionError):
        a.reply("привет")
    assert a.tts.stopped
    assert not [t for t in threading.enumerate() if t.name == "tts-play"]  # the player finished
    assert a.turn is None
