from beseda import language as lang


def keys(node: dict, prefix: str = "") -> set[str]:
    out = set()
    for key, value in node.items():
        out |= keys(value, f"{prefix}{key}.") if isinstance(value, dict) else {prefix + key}
    return out


def test_builtin_packs_have_the_same_ui_strings_and_voices():
    packs = [lang.load(code) for code in lang.available()]
    assert {"ru", "en"} <= {pack.code for pack in packs}
    reference = packs[0]
    for pack in packs:
        assert keys(pack.ui) == keys(reference.ui), pack.code
        assert pack.voices.keys() == reference.voices.keys(), pack.code
        assert pack.text("session_end", reason=pack.text("end_reason.stop"))


def test_user_pack_overrides_builtin(tmp_path, monkeypatch):
    monkeypatch.setattr(lang, "USER_DIR", tmp_path)
    builtin = (lang.files("beseda") / "languages" / "en.toml").read_text(encoding="utf-8")
    (tmp_path / "en.toml").write_text(builtin.replace('wake_word = "vika"', 'wake_word = "jarvis"'), encoding="utf-8")
    (tmp_path / "xx.toml").write_text(builtin, encoding="utf-8")
    assert lang.load("en").wake_word == "jarvis"
    assert "xx" in lang.available()
