import wave

import numpy as np

from beseda.recording import OUTPUT_RATE, DialogRecorder, Track


def tone(seconds: float, rate: int) -> bytes:
    t = np.arange(int(seconds * rate)) / rate
    return (np.sin(2 * np.pi * 440 * t) * 10000).astype(np.int16).tobytes()


def test_track_joins_close_chunks_and_splits_on_gaps():
    track = Track(16000)
    chunk = tone(0.1, 16000)
    track.add(chunk, 1.1)
    track.add(chunk, 1.2)  # right after the previous chunk: same segment
    track.add(chunk, 3.0)  # after a pause: new segment
    assert [round(start, 2) for start, _ in track.segments] == [1.0, 2.9]
    assert len(track.segments[0][1]) == 2


def test_track_keeps_short_pauses():
    track = Track(16000)
    chunk = tone(0.1, 16000)
    track.add(chunk, 1.1)
    track.add(chunk, 1.3)  # 0.1 s of silence between the chunks: below GAP_SECONDS, still a real pause
    assert len(track.segments) == 1
    assert round(track.end, 2) == 1.3
    assert len(b"".join(track.segments[0][1])) == int(0.3 * 16000) * 2


def test_save_starts_at_first_speech_and_keeps_pauses(tmp_path):
    recorder = DialogRecorder(tmp_path / "dialog.wav", voice_rate=22050)
    recorder.mic.add(tone(3.0, 16000), 3.0)  # background noise from startup on
    recorder.mic.add(tone(1.0, 16000), 11.0)  # the user's phrase, 10..11 s
    recorder.first_speech = 10.3  # VAD fires a bit after the first syllable
    recorder.voice.add(tone(1.0, 22050), 13.0)  # the answer, 12..13 s
    recorder.save()

    with wave.open(str(tmp_path / "dialog.wav")) as wf:
        audio = np.frombuffer(wf.readframes(wf.getnframes()), np.int16)
    window = OUTPUT_RATE // 10
    loud = np.abs(audio[: len(audio) // window * window].reshape(-1, window)).max(axis=1) > 1000
    spans = np.flatnonzero(np.diff(np.r_[0, loud.astype(int), 0])).reshape(-1, 2) / 10
    # The file starts 0.5 s before the first speech: phrase at 0.2..1.2 s, answer at 2.2..3.2 s.
    assert spans.tolist() == [[0.2, 1.2], [2.2, 3.2]]


def test_save_without_speech_writes_nothing(tmp_path):
    recorder = DialogRecorder(tmp_path / "dialog.wav", voice_rate=22050)
    recorder.mic.add(tone(1.0, 16000), 1.0)
    recorder.save()
    assert not (tmp_path / "dialog.wav").exists()
