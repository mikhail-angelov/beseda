import pytest

from beseda.wake import is_hold, is_stop, strip_wake, wake_pattern

VIKA = wake_pattern("вика")


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Вика, какая погода?", "какая погода?"),
        ("Эй, Вика! Расскажи анекдот.", "Расскажи анекдот."),
        ("вике скажи", "скажи"),
        ("Вика.", ""),
        ("Слушай Вику", ""),
        ("Привет, как дела?", None),
        ("Виктор пришёл", None),
        ("Ну вот, викарий", None),
    ],
)
def test_strip_wake(text, expected):
    assert strip_wake(text, VIKA) == expected


@pytest.mark.parametrize("text", ["Стоп!", "Хватит.", "Спасибо, всё.", "Всё, спасибо"])
def test_stop(text):
    assert is_stop(text)


@pytest.mark.parametrize("text", ["Стоп, подожди, а сколько стоит?", "Пока не надо"])
def test_not_stop(text):
    assert not is_stop(text)


@pytest.mark.parametrize("text", ["Подожди, дай подумать.", "Погоди.", "Секунду!", "Дай мне минутку", "Одну секунду"])
def test_hold(text):
    assert is_hold(text)


@pytest.mark.parametrize("text", ["Подожди, а сколько это стоит?", "Дай подумать над задачей", "Сейчас", "Ну так"])
def test_not_hold(text):
    assert not is_hold(text)
