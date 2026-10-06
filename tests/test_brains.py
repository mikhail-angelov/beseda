import threading

from beseda.brains import OpenAICompatBrain


class BlockedStream:
    """A streaming response whose next chunk never arrives until the stream is closed."""

    def __init__(self):
        self.closed = threading.Event()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self.closed.set()

    def __iter__(self):
        self.closed.wait()
        raise ConnectionError("stream closed")


def test_abort_releases_a_blocked_read():
    stream = BlockedStream()
    brain = object.__new__(OpenAICompatBrain)
    brain.model, brain.history, brain._aborted, brain._response = "m", [], threading.Event(), None
    brain.client = type("Client", (), {})()
    brain.client.chat = type("Chat", (), {})()
    brain.client.chat.completions = type("Completions", (), {"create": staticmethod(lambda **kw: stream)})()

    events = []
    worker = threading.Thread(target=lambda: events.extend(brain.ask("hi")))
    worker.start()
    worker.join(0.3)
    assert worker.is_alive()  # stuck waiting for the network
    brain.abort()
    worker.join(2)
    assert not worker.is_alive()
    assert events == []
