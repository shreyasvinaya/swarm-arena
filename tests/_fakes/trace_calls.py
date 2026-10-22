"""Core-only committed LLM provenance fixtures."""

from typing import Any

from _fakes.journals import FakeRun

from swarm_arena._journal.a2a import task_to_json, tool_run_task
from swarm_arena._journal.event import EventDraft, EventType
from swarm_arena._model.results import Candidate, LLMCall
from swarm_arena.tool import Usage


def _call(run: FakeRun, identifier: str, role: str = "manager", **fields: Any) -> LLMCall:
    """Build a core call with explicit recorded token and cost data."""
    data = dict(
        id=identifier,
        run_id=run.run_id,
        goal_id=run.goal.id,
        role=role,
        model="local/model",
        provider="local",
        cost_usd=0.000140,
        usage=Usage(input_tokens=25, output_tokens=20, cache_read_tokens=50, cache_write_tokens=25),
    )
    return LLMCall.model_validate({**data, **fields})


def _record(call: LLMCall) -> EventDraft:
    """Build the same payload envelope that the core recorder publishes."""
    return EventDraft(
        type=EventType.LLM_CALL,
        goal_id=call.goal_id,
        team=call.team,
        agent=call.agent,
        round=call.round,
        payload={"call": call.model_dump(mode="json"), "request_sha256": None},
    )


def _candidate(run: FakeRun, identifier: str = "c-llm") -> EventDraft:
    """Keep the tool completion and its A2A envelope mutually consistent."""
    draft = run.candidate(identifier)
    candidate = Candidate.model_validate(draft.payload["candidate"]).model_copy(
        update={"requested_by_call": "manager-call"}
    )
    return draft.model_copy(
        update={
            "payload": {
                **draft.payload,
                "candidate": candidate.model_dump(mode="json"),
                "a2a": task_to_json(tool_run_task(candidate, context_id=run.run_id, at=run._start)),
            }
        }
    )


def _seed(run: FakeRun) -> None:
    """Publish plan, manager/tool/judge calls and a linked learning."""
    draft = run.plan(revision=1 if run._events else 0, first_round=1)
    draft = draft.model_copy(
        update={
            "payload": {
                **draft.payload,
                "planner_call_ids": ["planner-call"],
                "origin": "llm",
            }
        }
    )
    score = run.score("s-llm", "c-llm", 1, panel_hash="surviving-panel")
    score.payload["score"]["llm_call_ids"] = ["judge-call", "judge-child", "judge-call"]
    run.commit(
        _record(_call(run, "planner-call", "planner")),
        draft,
        _record(_call(run, "manager-call", team="fd-solver", agent="tuner-1")),
        _record(_call(run, "tool-call", "tool", candidate_id="c-llm")),
        _candidate(run),
        _record(_call(run, "judge-call", "judge")),
        _record(_call(run, "judge-child", "judge", parent_call_id="judge-call")),
        score,
        run.learning("L-llm", "Linked learning", evidence=["c-llm"], posted_by_call="manager-call"),
        _record(_call(run, "failed-probe", status="error", error="ProviderFailed")),
    )
