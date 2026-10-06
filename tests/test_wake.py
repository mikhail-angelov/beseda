import pytest

from beseda.language import load
from beseda.wake import is_hold, is_stop, strip_wake, wake_pattern

RU = load("ru")
EN = load("en")
VIKA_RU = wake_pattern(RU.wake_word, RU.wake_endings)
VIKA_EN = wake_pattern(EN.wake_word, EN.wake_endings)


@pytest.mark.parametrize(
    "pattern, text, expected",
    [
        (VIKA_RU, "Вика, какая погода?", "какая погода?"),
        (VIKA_RU, "Эй, Вика! Расскажи анекдот.", "Расскажи анекдот."),
        (VIKA_RU, "вике скажи", "скажи"),
        (VIKA_RU, "Слушай Вику", ""),
        (VIKA_RU, "Вика.", ""),
        (VIKA_RU, "Привет, как дела?", None),
        (VIKA_RU, "Виктор пришёл", None),
        (VIKA_RU, "Ну вот, викарий", None),
        (VIKA_EN, "Vika, what's the weather?", "what's the weather?"),
        (VIKA_EN, "Hey Vika, hold on.", "hold on."),
        (VIKA_EN, "Okay vika", ""),
        (VIKA_EN, "What's the weather?", None),
        (VIKA_EN, "Vikings are coming", None),
    ],
)
def test_strip_wake(pattern, text, expected):
    assert strip_wake(text, pattern) == expected


@pytest.mark.parametrize(
    "language, text, expected",
    [
        (RU, "Стоп!", True),
        (RU, "Спасибо, всё.", True),
        (RU, "Стоп, подожди, а сколько стоит?", False),
        (EN, "Stop.", True),
        (EN, "Thanks, that's all.", True),
        (EN, "Stop the server", False),
    ],
)
def test_stop(language, text, expected):
    assert is_stop(text, language) is expected


@pytest.mark.parametrize(
    "language, text, expected",
    [
        (RU, "Подожди, дай подумать.", True),
        (RU, "Дай мне минутку", True),
        (RU, "Подожди, а сколько это стоит?", False),
        (RU, "Дай подумать над задачей", False),
        (RU, "Сейчас", False),
        (EN, "Hold on, let me think.", True),
        (EN, "Wait a second", True),
        (EN, "Give me a minute, please", True),
        (EN, "Wait, what's the price?", False),
        (EN, "Let me think about the API", False),
    ],
)
def test_hold(language, text, expected):
    assert is_hold(text, language) is expected
