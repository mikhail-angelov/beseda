"""Language packs: everything that depends on the spoken language, as data.

Built-in packs live in beseda/languages/<code>.toml. A file with the same name in ~/.beseda/languages/
overrides a built-in pack, and a new name adds a language.
"""

import re
import tomllib
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

USER_DIR = Path.home() / ".beseda" / "languages"


def normalize(text: str) -> str:
    """Lowercase words without punctuation: "Спасибо, всё!" -> "спасибо всё", "that's all" -> "that s all"."""
    return " ".join(re.findall(r"\w+", text.lower()))


@dataclass(frozen=True)
class Language:
    code: str  # Whisper language and sentence splitting
    wake_word: str
    wake_endings: list[str]  # grammatical endings the wake word may take: Вика, Вику, Вике
    style_prompt: str  # a neutral sentence that primes Whisper's punctuation
    voice_prompt: str  # tells the model its answers are spoken
    stop_phrases: set[str]
    hold_triggers: set[str]
    hold_fillers: set[str]
    greeting: str  # short phrase to warm up TTS models
    sample_text: str
    silero_model: str
    voices: dict[str, list[str]]  # TTS engine -> voices, the first one is the default
    ui: dict

    def text(self, key: str, **values) -> str:
        """A UI string: "status.listening", "session_end" with {placeholders} filled."""
        node = self.ui
        for part in key.split("."):
            node = node[part]
        return node.format(**values)


def available() -> list[str]:
    builtin = {path.name.removesuffix(".toml") for path in files("beseda").joinpath("languages").iterdir()}
    custom = {path.stem for path in USER_DIR.glob("*.toml")} if USER_DIR.exists() else set()
    return sorted(builtin | custom)


def load(code: str) -> Language:
    user = USER_DIR / f"{code}.toml"
    builtin = files("beseda").joinpath("languages", f"{code}.toml")
    if user.exists():
        data = tomllib.loads(user.read_text(encoding="utf-8"))
    elif builtin.is_file():
        data = tomllib.loads(builtin.read_text(encoding="utf-8"))
    else:
        raise SystemExit(f"Unknown language {code!r}; available: {', '.join(available())}")
    return Language(
        code=data["code"],
        wake_word=data["wake_word"],
        wake_endings=data["wake_endings"],
        style_prompt=data["style_prompt"],
        voice_prompt=" ".join(data["voice_prompt"].split()),
        stop_phrases={normalize(phrase) for phrase in data["stop_phrases"]},
        hold_triggers={normalize(word) for word in data["hold_triggers"]},
        hold_fillers={normalize(word) for word in data["hold_fillers"]},
        greeting=data["greeting"],
        sample_text=data["sample_text"],
        silero_model=data["silero_model"],
        voices=data["voices"],
        ui=data["ui"],
    )
