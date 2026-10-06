from beseda.stt import clean
from beseda.tts import speakable


def test_speakable_drops_code_and_markdown():
    deltas = ["Вот **код**:\n``", "`py\nprint(1)\n`", "``\nГотово #1"]
    assert "".join(speakable(iter(deltas))) == "Вот код:\n\nГотово 1"


def test_speakable_keeps_plain_text():
    assert "".join(speakable(iter(["Привет, ", "как дела?"]))) == "Привет, как дела?"


def test_clean_drops_whisper_annotations():
    assert clean("[музыка]") == ""
    assert clean("*звук*") == ""
    assert clean(" Привет [BLANK_AUDIO] как дела ") == "Привет как дела"
