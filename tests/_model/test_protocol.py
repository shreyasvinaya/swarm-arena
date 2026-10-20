import dataclasses
import pickle
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pytest
from _fakes.secrecy import assert_secret_absent

from swarm_arena._model.protocol import (
    Artifact,
    Message,
    Part,
    Role,
    Task,
    TaskState,
    TaskStatus,
    task_state_for,
)
from swarm_arena._model.tooling import RunStatus


def test_enum_values_are_proto_names() -> None:
    assert {s.value for s in TaskState} == {
        f"TASK_STATE_{n}"
        for n in (
            "UNSPECIFIED",
            "SUBMITTED",
            "WORKING",
            "COMPLETED",
            "FAILED",
            "CANCELED",
            "REJECTED",
            "INPUT_REQUIRED",
            "AUTH_REQUIRED",
        )
    }
    assert {r.value for r in Role} == {"ROLE_UNSPECIFIED", "ROLE_USER", "ROLE_AGENT"}


def test_part_exactly_one_content() -> None:
    assert Part(data=None).data is None
    assert Part(raw=b"x").raw == b"x"
    cases: tuple[dict[str, Any], ...] = ({}, {"text": "a", "url": "u"})
    for kwargs in cases:
        with pytest.raises(ValueError, match="exactly one"):
            Part(**kwargs)


def test_message_and_artifact_need_parts() -> None:
    with pytest.raises(ValueError, match="part"):
        Artifact(artifact_id="a", parts=())
    with pytest.raises(ValueError, match="part"):
        Message(message_id="m", role=Role.AGENT, parts=())


def test_task_status_needs_aware_timestamp() -> None:
    with pytest.raises(ValueError, match="aware"):
        TaskStatus(state=TaskState.WORKING, timestamp=datetime(2026, 1, 1))


def test_frozen_slots() -> None:
    def mutate(obj: Any) -> None:
        obj.text = "b"

    with pytest.raises(dataclasses.FrozenInstanceError):
        mutate(Part(text="a"))
    assert hasattr(Part, "__slots__")


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (RunStatus.OK, (TaskState.COMPLETED, {})),
        (RunStatus.CANCELLED, (TaskState.CANCELED, {})),
        *[
            (s, (TaskState.FAILED, {"swarm_arena.reason": s.value}))
            for s in (
                RunStatus.BUDGET_EXCEEDED,
                RunStatus.TIMEOUT,
                RunStatus.LOST,
                RunStatus.ERROR,
                RunStatus.OOM,
                RunStatus.PROVIDER_FAILED,
            )
        ],
    ],
)
def test_task_state_for(status: RunStatus, expected: tuple[TaskState, dict[str, str]]) -> None:
    assert task_state_for(status) == expected


@pytest.mark.parametrize(
    "fields",
    [
        {"text": ""},
        {"raw": b""},
        {"url": ""},
        {"data": False},
        {"data": 0},
        {"data": []},
        {"data": {}},
        {"data": None},
    ],
)
def test_part_content_presence_and_pickle(fields: dict[str, Any]) -> None:
    part = Part(**fields)
    assert pickle.loads(pickle.dumps(part)) == part
    with pytest.raises(ValueError, match="exactly one"):
        dataclasses.replace(part, **({"raw": b"x"} if "text" in fields else {"text": "x"}))


def test_task_status_normalizes_utc_milliseconds() -> None:
    stamp = datetime(2026, 1, 1, 5, 30, 0, 123456, tzinfo=timezone(timedelta(hours=5, minutes=30)))
    status = TaskStatus(state=TaskState.WORKING, timestamp=stamp)
    assert status.timestamp == datetime(2026, 1, 1, 0, 0, 0, 123000, tzinfo=UTC)
    assert status.timestamp.tzinfo is UTC


def test_protocol_record_fields_defaults_and_frozen_slots() -> None:
    part = Part(
        data={"answer": 42},
        media_type="application/json",
        filename="answer.json",
        metadata={"source": "local"},
    )
    msg = Message(
        message_id="m",
        role=Role.AGENT,
        parts=(part,),
        context_id="r",
        task_id="t",
        reference_task_ids=("earlier",),
        extensions=("urn:ext",),
    )
    artifact = Artifact(
        artifact_id="a", parts=(part,), name="answer", description="result", extensions=("urn:ext",)
    )
    status = TaskStatus(
        state=TaskState.COMPLETED, timestamp=datetime(2026, 1, 1, tzinfo=UTC), message=msg
    )
    task = Task(id="t", context_id="r", status=status, artifacts=(artifact,), history=(msg,))
    for record in (part, msg, artifact, status, task):
        assert pickle.loads(pickle.dumps(record)) == record
        assert not hasattr(record, "__dict__")
        name = dataclasses.fields(record)[0].name
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(record, name, "changed")
    assert task.metadata == {}
    assert artifact.metadata == {}
    assert msg.metadata == {}
    assert isinstance(task.metadata, dict)
    task.metadata["x"] = 1  # frozen records follow the declared shallow Mapping contract
    assert Task(id="other", context_id="r", status=status).metadata == {}


def test_protocol_validation_errors_do_not_expose_content() -> None:
    with pytest.raises(ValueError, match="exactly one") as caught:
        Part(
            text="1f7f7213-02d5-57ac-9da0-42a305305986", data="1f7f7213-02d5-57ac-9da0-42a305305986"
        )
    assert_secret_absent(caught.value, "1f7f7213-02d5-57ac-9da0-42a305305986")
