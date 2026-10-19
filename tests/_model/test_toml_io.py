import tomllib
from pathlib import Path

import pytest

from swarm_arena._errors import ConfigError
from swarm_arena._model.toml_io import dumps_toml, read_toml

SAMPLE = {
    "a": 1,
    "b": 'x\ny "q" \x7f',
    "c": [1, 2.5, True],
    "t": {"k": "v", "sub": {"z": 0}},
    "arr": [{"n": 1}, {"n": 2, "m": {"q": []}}],
    "e": {},
    "u": "é",
    "f": 1e-05,
}


def test_roundtrip() -> None:
    assert tomllib.loads(dumps_toml(SAMPLE)) == SAMPLE


def test_scalars_first_and_none_dropped() -> None:
    assert dumps_toml({"t": {"x": 1}, "a": 1, "n": None}) == "a = 1\n\n[t]\nx = 1\n"


def test_quoted_keys() -> None:
    assert dumps_toml({"a b": 1, "ok_key-1": 2}) == '"a b" = 1\nok_key-1 = 2\n'


def test_deterministic() -> None:
    # C25: pin text rather than comparing the writer with itself.
    data = {"t": {"x": 1}, "a b": 2, "arr": [{"n": 1}, {"n": 2}], "f": 1e-05}
    assert dumps_toml(data) == (
        '"a b" = 2\nf = 1e-05\n\n[t]\nx = 1\n\n[[arr]]\nn = 1\n\n[[arr]]\nn = 2\n'
    )


def test_rejects_bad_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        dumps_toml({"x": float("inf")})
    with pytest.raises(TypeError, match="bytes"):
        dumps_toml({"x": b"1"})


def test_read_toml_error_names_file(tmp_path: Path) -> None:
    (tmp_path / "bad.toml").write_text("a = ")
    with pytest.raises(ConfigError, match=r"bad\.toml"):
        read_toml(tmp_path / "bad.toml")


def test_read_toml(tmp_path: Path) -> None:
    path = tmp_path / "sample.toml"
    path.write_text(dumps_toml(SAMPLE), encoding="utf-8")
    assert read_toml(path) == SAMPLE


def test_inline_tables_and_nested_array_tables() -> None:
    data = {
        "a.b": {"x y": [{"k.l": 1, "nested": {"a": True}}, 2]},
        "rows": [{"empty": {}, "drop": None, "nested": [{"value": "é"}]}, {}],
        "drop": None,
    }
    expected = {
        "a.b": {"x y": [{"k.l": 1, "nested": {"a": True}}, 2]},
        "rows": [{"empty": {}, "nested": [{"value": "é"}]}, {}],
    }
    assert tomllib.loads(dumps_toml(data)) == expected


def test_control_characters_are_escaped() -> None:
    value = "".join(chr(i) for i in range(32)) + '\x7f\\"é'
    text = dumps_toml({value: value})
    assert "\x7f" not in text
    assert "\\u007f" in text
    assert tomllib.loads(text) == {value: value}


@pytest.mark.parametrize("value", [float("nan"), float("-inf")])
def test_rejects_nonfinite_nested_values(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        dumps_toml({"a": [{"x": [value]}]})
