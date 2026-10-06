"""Wake word, stop phrase and hold phrase detection on transcribed text."""

import re

from beseda.language import Language, normalize


def wake_pattern(word: str, endings: list[str]) -> re.Pattern:
    """Matches the wake word with any of its endings (Вика, Вику, Вике) within the first three words."""
    word = word.lower()
    by_length = sorted(endings, key=len, reverse=True)
    stem = next((word[: -len(ending)] for ending in by_length if ending and word.endswith(ending)), word)
    forms = "|".join(re.escape(ending) for ending in by_length)
    return re.compile(rf"^\W*(?:\w+\W+){{0,2}}?{re.escape(stem)}(?:{forms})\b\W*", re.IGNORECASE)


def strip_wake(text: str, pattern: re.Pattern) -> str | None:
    """Text after the wake word, or None if the phrase isn't addressed to the assistant."""
    match = pattern.match(text)
    return text[match.end():].strip() if match else None


def is_stop(text: str, language: Language) -> bool:
    return normalize(text) in language.stop_phrases


def is_hold(text: str, language: Language) -> bool:
    """"Подожди", "дай подумать", "hold on": the user needs time, not an answer."""
    words = set(normalize(text).split())
    return bool(words & language.hold_triggers) and words <= language.hold_triggers | language.hold_fillers
