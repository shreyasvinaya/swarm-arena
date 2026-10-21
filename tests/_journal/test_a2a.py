from datetime import UTC, datetime, timedelta, timezone

import pytest

from swarm_arena._journal.a2a import agent_round_task, agent_task_id, task_to_json, tool_run_task
from swarm_arena._model.base import JsonValue
from swarm_arena._model.results import AgentStatus, Learning
from swarm_arena.protocol import Artifact, Message, Part, Role, Task, TaskState, TaskStatus
from swarm_arena.testing import make_candidate
from swarm_arena.tool import RunStatus, ToolOutput

AT = datetime(2026, 10, 18, 13, 0, 0, 123456, tzinfo=UTC)


def _learning(id: str) -> Learning:
    return Learning(
        id=id,
        namespace="n",
        run_id="r",
        goal_id="g",
        round=2,
        author="g/t/a",
        title="A useful result",
        text="Try this.",
        evidence=["c-1"],
    )


def test_tool_run_task_maps_status_and_parents() -> None:
    cand = make_candidate(
        ToolOutput(status=RunStatus.TIMEOUT, partial=True), parent_ids=["c-p"], id="c-x"
    )
    task = tool_run_task(cand, context_id="run-1", at=AT)
    assert task.id == "c-x"
    assert task.context_id == "run-1"
    assert task.status.state == TaskState.FAILED
    assert task.metadata["swarm_arena.reason"] == "timeout"
    assert task_to_json(task)["status"]["state"] == "TASK_STATE_FAILED"
    assert len(task.history) == 1
    assert task.history[0].reference_task_ids == ("c-p",)
    assert task.history[0].task_id == "c-x"
    assert task.history[0].context_id == "run-1"
    assert task_to_json(task)["history"][0]["referenceTaskIds"] == ["c-p"]
    assert "referenceTaskIds" not in task_to_json(task)


@pytest.mark.parametrize("status", list(RunStatus))
def test_tool_status_mapping(status: RunStatus) -> None:
    task = tool_run_task(make_candidate(ToolOutput(status=status), id="c"), context_id="r", at=AT)
    expected = {RunStatus.OK: TaskState.COMPLETED, RunStatus.CANCELLED: TaskState.CANCELED}
    assert task.status.state == expected.get(status, TaskState.FAILED)
    assert dict(task.metadata) == (
        {} if status in expected else {"swarm_arena.reason": status.value}
    )


def test_agent_round_task_has_one_artifact_per_learning() -> None:
    task = agent_round_task(
        context_id="r",
        address="g/t/a",
        round=2,
        learnings=[_learning("L-1")],
        status=AgentStatus.FINISHED,
        summary="done",
        at=AT,
    )
    assert task.id == "g/t/a#r2"
    assert len(task.artifacts) == 1
    assert {p.text is not None for p in task.artifacts[0].parts} == {True, False}
    assert task.artifacts[0].artifact_id == "L-1"
    assert task.artifacts[0].parts[0].data == _learning("L-1").model_dump(mode="json")
    assert task.artifacts[0].parts[1].text == "A useful result"
    assert task.status.message is not None
    assert task.status.message.parts == (Part(text="done"),)
    assert agent_task_id("g/t/a", 2) == "g/t/a#r2"


@pytest.mark.parametrize(
    ("status", "state", "reason"),
    [
        (AgentStatus.ACTIVE, TaskState.WORKING, None),
        (AgentStatus.FINISHED, TaskState.COMPLETED, None),
        (AgentStatus.PRUNED, TaskState.CANCELED, None),
        (AgentStatus.STOPPED, TaskState.CANCELED, None),
        (AgentStatus.STARVED, TaskState.FAILED, "starved"),
        (AgentStatus.WATCHDOG_STOPPED, TaskState.FAILED, "watchdog_stopped"),
    ],
)
def test_agent_status_mapping(status: AgentStatus, state: TaskState, reason: str | None) -> None:
    task = agent_round_task(
        context_id="r", address="g/t/a", round=1, learnings=[], status=status, summary="", at=AT
    )
    assert task.status.state == state
    assert dict(task.metadata) == ({} if reason is None else {"swarm_arena.reason": reason})
    assert task.artifacts == ()


def test_task_json_uses_proto_names_and_copies_nested_data() -> None:
    at = AT.astimezone(timezone(timedelta(hours=5, minutes=30)))
    task = Task(
        id="c",
        context_id="r",
        status=TaskStatus(state=TaskState.WORKING, timestamp=at),
        history=(
            Message(
                message_id="m",
                role=Role.AGENT,
                parts=(Part(text=""),),
                context_id="r",
                task_id="c",
                reference_task_ids=("p",),
                extensions=("ext",),
                metadata={"nested": [1]},
            ),
        ),
        artifacts=(
            Artifact(
                artifact_id="a",
                name="name",
                description="description",
                parts=(
                    Part(data=None),
                    Part(raw=b"\x00\xff", media_type="application/octet-stream", filename="x"),
                    Part(url="https://example.test/x"),
                ),
                extensions=("ext",),
            ),
        ),
        metadata={"nested": {"values": [1]}},
    )
    result = task_to_json(task)
    assert result["contextId"] == "r"
    assert result["status"] == {
        "state": "TASK_STATE_WORKING",
        "timestamp": "2026-10-18T13:00:00.123Z",
    }
    assert result["history"][0]["messageId"] == "m"
    assert result["history"][0]["role"] == "ROLE_AGENT"
    assert result["history"][0]["parts"] == [{"text": ""}]
    assert result["artifacts"][0]["parts"] == [
        {"data": None},
        {"raw": "AP8=", "mediaType": "application/octet-stream", "filename": "x"},
        {"url": "https://example.test/x"},
    ]
    result["metadata"]["nested"]["values"].append(2)
    assert task.metadata == {"nested": {"values": [1]}}


def test_agent_json_keeps_all_learning_artifacts_and_summary() -> None:
    task = agent_round_task(
        context_id="r",
        address="g/t/a",
        round=2,
        learnings=[_learning("L-1"), _learning("L-2")],
        status=AgentStatus.FINISHED,
        summary="done",
        at=AT,
    )
    result = task_to_json(task)
    assert [a["artifactId"] for a in result["artifacts"]] == ["L-1", "L-2"]
    assert result["status"]["message"]["parts"] == [{"text": "done"}]


@pytest.mark.parametrize("data", [{}, [], None, False, 0, ""])
def test_json_preserves_present_empty_data(data: JsonValue) -> None:
    task = Task(
        id="t",
        context_id="r",
        status=TaskStatus(state=TaskState.COMPLETED, timestamp=AT),
        artifacts=(Artifact(artifact_id="a", parts=(Part(data=data),)),),
    )
    assert task_to_json(task)["artifacts"][0]["parts"] == [{"data": data}]
