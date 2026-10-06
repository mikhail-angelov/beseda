import pytest

import beseda.__main__ as app


@pytest.mark.parametrize(
    "config, ok",
    [
        ('brain = "deepseek"\nfollow-up = 10\nrecord = true\nvocabulary = ["JS"]', True),
        ('brain = "deepssek"', False),
        ("follow-up = true", False),
        ('vocabulary = ["ok", 1]', False),
        ("unknown-key = 1", False),
    ],
)
def test_config_values_are_validated(tmp_path, monkeypatch, config, ok):
    path = tmp_path / "config.toml"
    path.write_text(config)
    monkeypatch.setattr(app, "CONFIG_PATH", path)
    monkeypatch.setattr("sys.argv", ["beseda", "--help"])
    with pytest.raises(SystemExit) as exit:
        app.main()
    assert (exit.value.code == 0) is ok
