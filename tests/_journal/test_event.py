import json
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pytest
from _fakes.secrecy import assert_secret_absent
from pydantic import ValidationError

from swarm_arena._journal.event import WRITE_THROUGH, Event, EventDraft, EventType
from swarm_arena._model.base import FrozenModel, StoredModel
from swarm_arena._model.canonical import canonical_json


def _raw_event() -> dict[str, Any]:
    return {
        "epoch": 1,
        "seq": 7,
        "ts": "2026-10-18T13:00:00.000Z",
        "run_id": "r",
        "context_id": "r",
        "task_id": "c-1",
        "goal_id": "g",
        "type": "tool_end",
        "payload": {"a": 1},
    }


def test_event_types_are_exact() -> None:
    values = {
        "run_started",
        "run_config",
        "limits_changed",
        "takeover",
        "stop_requested",
        "goal_queued",
        "goal_started",
        "goal_finished",
        "plan_revision",
        "frontier",
        "round_started",
        "agent_status",
        "agent_counters",
        "notes_saved",
        "reserve",
        "settle",
        "tool_dispatch",
        "tool_start",
        "tool_end",
        "command_start",
        "command_end",
        "run_lost",
        "score",
        "judge_error",
        "learning",
        "decision",
        "no_result",
        "part_kept",
        "join_failed",
        "reserve_exhausted",
        "provider_failed",
        "watchdog",
        "round_closed",
        "converge_started",
        "final_order",
        "solution_written",
        "llm_call",
        "gateway_key",
        "reconcile",
        "curated",
        "review_imported",
    }
    assert {event_type.value for event_type in EventType} == values
    assert len(EventType.__members__) == len(values)
    assert all(event_type.name == event_type.value.upper() for event_type in EventType)
    assert isinstance(EventType.TOOL_END, str)


def test_write_through_set_is_exact() -> None:
    assert isinstance(WRITE_THROUGH, frozenset)
    assert {t.value for t in WRITE_THROUGH} == {
        "reserve",
        "settle",
        "tool_dispatch",
        "tool_start",
        "tool_end",
        "command_start",
        "frontier",
        "round_closed",
    }
    assert EventType.COMMAND_END not in WRITE_THROUGH


def test_event_round_trips_with_schema_version() -> None:
    ev = Event(
        epoch=1,
        seq=7,
        ts=datetime(2026, 10, 18, 13, 0, tzinfo=UTC),
        run_id="r",
        context_id="r",
        task_id="c-1",
        goal_id="g",
        type=EventType.TOOL_END,
        payload={"a": 1},
    )
    raw = json.loads(canonical_json(ev.model_dump(mode="json")))
    assert raw["schema_version"] == 1
    assert raw["ts"] == "2026-10-18T13:00:00.000Z"
    assert Event.model_validate(raw) == ev
    assert Event.model_validate_json(canonical_json(raw)) == ev
    assert isinstance(ev, StoredModel)
    assert Event.schema_kind == "event"


def test_newer_event_schema_refused() -> None:
    with pytest.raises(ValueError, match="newer"):
        Event.model_validate({**_raw_event(), "schema_version": 2})


def test_event_defaults_and_draft_are_frozen() -> None:
    draft = EventDraft(type=EventType.RUN_STARTED)
    assert isinstance(draft, FrozenModel)
    assert not isinstance(draft, StoredModel)
    assert draft.model_dump() == {
        "type": EventType.RUN_STARTED,
        "goal_id": "",
        "team": "",
        "agent": "",
        "round": 0,
        "task_id": "",
        "payload": {},
    }
    ev = Event.model_validate(
        {
            key: value
            for key, value in _raw_event().items()
            if key not in {"goal_id", "task_id", "payload"}
        }
    )
    assert (ev.goal_id, ev.team, ev.agent, ev.round, ev.task_id, ev.payload) == (
        "",
        "",
        "",
        0,
        "",
        {},
    )
    for model, field in [(draft, "type"), (ev, "seq")]:
        with pytest.raises(ValidationError, match="frozen_instance"):
            setattr(model, field, None)
    other = EventDraft(type=EventType.RUN_STARTED)
    draft.payload["x"] = 1
    assert other.payload == {}


def test_event_normalizes_utc_milliseconds_and_refuses_naive_time() -> None:
    ts = datetime(2026, 10, 18, 18, 30, 0, 123456, tzinfo=timezone(timedelta(hours=5, minutes=30)))
    ev = Event.model_validate({**_raw_event(), "ts": ts})
    assert ev.ts == datetime(2026, 10, 18, 13, 0, 0, 123000, tzinfo=UTC)
    assert ev.model_dump(mode="json")["ts"] == "2026-10-18T13:00:00.123Z"
    with pytest.raises(ValueError, match="value_error"):
        Event.model_validate({**_raw_event(), "ts": ts.replace(tzinfo=None)})


@pytest.mark.parametrize("model", [Event, EventDraft])
def test_payload_preserves_finite_json_and_null(model: type[Event] | type[EventDraft]) -> None:
    payload = {"nested": [None, True, 1, 1.5, {"text": "λ\u2028\u2029\n"}]}
    raw = _raw_event() if model is Event else {"type": "tool_end"}
    value = model.model_validate({**raw, "payload": payload})
    assert value.payload == payload
    assert model.model_validate_json(canonical_json(value.model_dump(mode="json"))) == value


@pytest.mark.parametrize("model", [Event, EventDraft])
@pytest.mark.parametrize(
    "bad",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
        b"private-payload",
        object(),
        (1,),
        {1: 2},
        "\ud800",
    ],
    ids=["nan", "inf", "negative-inf", "bytes", "object", "tuple", "nonstring-key", "surrogate"],
)
def test_payload_rejects_non_json_with_safe_errors(
    model: type[Event] | type[EventDraft], bad: object
) -> None:
    secret = "private-payload"
    raw = _raw_event() if model is Event else {"type": "tool_end"}
    with pytest.raises(ValidationError) as caught:
        model.model_validate({**raw, "payload": {secret: [secret, {"bad": bad}]}})
    assert_secret_absent(caught.value, secret, sanitized=True)


@pytest.mark.parametrize("model", [Event, EventDraft])
def test_unknown_fields_and_event_types_are_sanitized(
    model: type[Event] | type[EventDraft],
) -> None:
    secret = "private-event-type"
    raw = _raw_event() if model is Event else {"type": "tool_end"}
    for invalid in [{**raw, "type": secret}, {**raw, secret: secret}]:
        with pytest.raises(ValidationError) as caught:
            model.model_validate(invalid)
        assert_secret_absent(caught.value, secret, sanitized=True)
