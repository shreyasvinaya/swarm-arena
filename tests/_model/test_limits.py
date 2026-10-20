from typing import Any

import pytest
from _fakes.secrecy import assert_secret_absent
from pydantic import ValidationError

from swarm_arena._model.limits import Limits


def test_limits_defaults_match_spec() -> None:
    assert Limits().model_dump() == {
        "max_teams": 4,
        "max_agents_per_team": 4,
        "max_agents_per_goal": 12,
        "max_concurrent_goals": 4,
        "max_agents_total": 32,
        "max_agents_per_fingerprint": 2,
        "max_rounds": 3,
        "max_turns_per_round": 20,
        "max_tool_runs_per_round": 3,
        "max_tokens": None,
        "max_usd": None,
        "default_run_reservation": {"seconds": None, "tokens": 100_000, "usd": 1.0},
        "llm_tranche_usd": 0.25,
        "llm_tranche_tokens": 25_000,
        "tool_timeout_s": 3600.0,
        "round_timeout_s": None,
        "run_timeout_s": 86_400.0,
        "max_learnings_per_agent_round": 3,
        "max_board_reads_per_round": 10,
        "digest_tokens": 1500,
        "max_edit_bytes": 262_144,
        "max_consecutive_failures": 3,
        "max_refused_repeats": 3,
        "watchdog_idle_rounds": 2,
        "max_notes_bytes": 16_384,
    }


def test_effective_round_timeout() -> None:
    assert Limits().effective_round_timeout_s() == 14_400.0
    assert Limits(tool_timeout_s=100).effective_round_timeout_s() == 400.0
    assert Limits(round_timeout_s=50).effective_round_timeout_s() == 50.0


@pytest.mark.parametrize(
    "bad",
    [{"max_rounds": 0}, {"max_usd": 0}, {"tool_timeout_s": 0}, {"max_board_reads_per_round": -1}],
)
def test_limits_bounds(bad: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        Limits.model_validate(bad)


@pytest.mark.parametrize(
    "field",
    [
        "max_teams",
        "max_agents_per_team",
        "max_agents_per_goal",
        "max_concurrent_goals",
        "max_agents_total",
        "max_agents_per_fingerprint",
        "max_rounds",
        "max_turns_per_round",
        "max_tool_runs_per_round",
        "max_tokens",
        "max_usd",
        "llm_tranche_usd",
        "llm_tranche_tokens",
        "tool_timeout_s",
        "round_timeout_s",
        "run_timeout_s",
        "digest_tokens",
        "max_edit_bytes",
        "max_consecutive_failures",
        "max_refused_repeats",
        "watchdog_idle_rounds",
        "max_notes_bytes",
    ],
)
def test_all_positive_bounds(field: str) -> None:
    for value in (0, -1):
        with pytest.raises(ValidationError):
            Limits.model_validate({field: value})


@pytest.mark.parametrize("field", ["max_learnings_per_agent_round", "max_board_reads_per_round"])
def test_zero_allowed_counts(field: str) -> None:
    assert getattr(Limits.model_validate({field: 0}), field) == 0
    with pytest.raises(ValidationError):
        Limits.model_validate({field: -1})


def test_limits_frozen_extra_and_reservation() -> None:
    with pytest.raises(ValidationError):
        Limits.model_validate({"typo": 1})
    limits = Limits()
    with pytest.raises(ValidationError):
        limits.max_rounds = 9  # ty: ignore[invalid-assignment]
    assert limits.default_run_reservation.tokens == 100_000


def test_limits_errors_do_not_leak_values() -> None:
    secret = "542371d6-031b-544d-8244-19c4c8a32fc3"
    with pytest.raises(ValidationError) as exc:
        Limits.model_validate({"max_rounds": secret})
    assert_secret_absent(exc.value, secret)


@pytest.mark.parametrize(
    ("values", "location"),
    [
        ({"max_roundz": "542371d6-031b-544d-8244-19c4c8a32fc3"}, "max_roundz"),
        (
            {"default_run_reservation": {"tokenz": "542371d6-031b-544d-8244-19c4c8a32fc3"}},
            "default_run_reservation.tokenz",
        ),
    ],
)
def test_limits_unknown_fields_show_identifiers_without_values(
    values: dict[str, Any], location: str
) -> None:
    with pytest.raises(ValidationError) as exc:
        Limits.model_validate(values)
    assert location in str(exc.value)
    assert_secret_absent(exc.value, "542371d6-031b-544d-8244-19c4c8a32fc3")
