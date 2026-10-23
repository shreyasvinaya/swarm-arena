"""Bound printable gateway observations without changing recorder fallback values."""

import pytest

from swarm_arena._gateway.labels import _observation_label, _profile_failures


@pytest.mark.parametrize("value", [None, "", " ", "x\n", "\x1b[2J", "x" * 257])
def test_invalid_observation_uses_explicit_fallback(value: object) -> None:
    assert _observation_label(value, 256) == "unknown"
    assert _observation_label(value, 256, fallback="<unknown>") == "<unknown>"


@pytest.mark.parametrize("value", ["unknown", "chat/model", "x" * 256])
def test_printable_observation_preserves_identity(value: str) -> None:
    assert _observation_label(value, 256) == value


def test_profile_failure_bound_is_idempotent() -> None:
    failures = [f"model-{index}: no price" for index in range(30)]
    expected = (*failures[:10], "and 20 more")
    assert _profile_failures(failures) == expected
    assert _profile_failures(expected) == expected
    assert _profile_failures(()) == ()


def test_profile_failure_summary_is_bounded_too() -> None:
    result = _profile_failures(["missing"] * 10 + ["and " + "9" * 5000 + " more"])
    assert all(len(value) <= 512 for value in result)
