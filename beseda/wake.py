"""Wake word and stop phrase detection on transcribed text."""

import re

STOP_PHRASES = {
    "стоп", "хватит", "отбой", "отмена", "всё", "все", "пока", "до свидания",
    "спасибо", "спасибо всё", "всё спасибо", "спасибо все", "все спасибо", "стоп стоп",
}
# A phrase made only of these words asks to wait; it needs at least one of HOLD_TRIGGERS.
HOLD_TRIGGERS = {
    "подожди", "подождите", "погоди", "погодите", "подумать", "подумаю", "секунду", "секундочку",
    "минуту", "минутку", "минуточку", "момент",
}
HOLD_WORDS = HOLD_TRIGGERS | {"дай", "дайте", "мне", "одну", "сейчас", "щас", "так", "ну", "ещё", "еще", "немного", "чуть", "я"}
VOWELS = "аяоеёиыуюэ"


def wake_pattern(word: str) -> re.Pattern:
    """Matches the wake word in any case form (Вика, Вику, Вике, Викой) within the first three words."""
    word = word.lower()
    stem = word[:-1] if word[-1] in VOWELS else word
    return re.compile(rf"^\W*(?:\w+\W+){{0,2}}?{stem}[{VOWELS}й]{{0,2}}\b\W*", re.IGNORECASE)


def strip_wake(text: str, pattern: re.Pattern) -> str | None:
    """Text after the wake word, or None if the phrase isn't addressed to the assistant."""
    match = pattern.match(text)
    return text[match.end():].strip() if match else None


def is_stop(text: str) -> bool:
    return " ".join(re.findall(r"\w+", text.lower())) in STOP_PHRASES


def is_hold(text: str) -> bool:
    """"Подожди", "дай подумать", "секунду": the user needs time, not an answer."""
    words = set(re.findall(r"\w+", text.lower()))
    return bool(words & HOLD_TRIGGERS) and words <= HOLD_WORDS
