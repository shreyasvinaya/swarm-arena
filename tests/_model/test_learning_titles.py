import json
from typing import Any

import pytest
from _fakes.secrecy import assert_secret_absent
from pydantic import ValidationError

from swarm_arena.board import Learning

_SECRET = "8bc35aae-2c20-4a52-9ccf-32a78bcfdd60"


def _learning(title: str, route: str) -> Learning:
    data: dict[str, Any] = {
        "id": "learning",
        "namespace": "demo",
        "run_id": "run",
        "goal_id": "goal",
        "round": 1,
        "author": "human:alice",
        "title": title,
        "text": "Try this.",
    }
    if route == "constructor":
        return Learning(**data)
    data["schema_version"] = 1
    if route == "mapping":
        return Learning.model_validate(data)
    return Learning.model_validate_json(json.dumps(data))


@pytest.mark.parametrize("route", ["constructor", "mapping", "json"])
@pytest.mark.parametrize("position", ["leading", "embedded", "trailing"])
@pytest.mark.parametrize(
    "codepoint",
    [*range(0x20), *range(0x7F, 0xA0), 0x2028, 0x2029],
    ids=lambda value: f"U+{value:04X}",
)
def test_learning_title_rejects_controls_before_trimming(
    codepoint: int, position: str, route: str
) -> None:
    char = chr(codepoint)
    titles = {
        "leading": char + _SECRET,
        "embedded": _SECRET + char + "title",
        "trailing": _SECRET + char,
    }
    with pytest.raises(ValidationError) as caught:
        _learning(titles[position], route)
    assert caught.value.errors()[0]["loc"] == ("title",)
    assert_secret_absent(caught.value, _SECRET)


@pytest.mark.parametrize("route", ["constructor", "mapping", "json"])
@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("  ordinary ASCII title  ", "ordinary ASCII title"),
        ("Δοκιμή café 中文 🧪", "Δοκιμή café 中文 🧪"),
        ("\u2003A useful title\u00a0", "A useful title"),
        ("  " + "é" * 120 + "  ", "é" * 120),
    ],
)
def test_learning_title_accepts_printable_text(title: str, expected: str, route: str) -> None:
    learning = _learning(title, route)
    assert learning.title == expected
    assert learning.from_human
    assert learning.evidence == []
    assert Learning.model_validate_json(learning.model_dump_json()) == learning


@pytest.mark.parametrize("route", ["constructor", "mapping", "json"])
@pytest.mark.parametrize("title", ["", " \u2003 ", "x" * 121])
def test_learning_title_preserves_length_bounds(title: str, route: str) -> None:
    with pytest.raises(ValidationError):
        _learning(title, route)
