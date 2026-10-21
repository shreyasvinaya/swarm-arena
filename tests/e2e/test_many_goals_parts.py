"""Exercise public multi-goal runs and inspect their committed audit records.

Only the journal readers use private interfaces: public results establish the
join behavior, while committed shares and reservation charges establish the
shared-budget assertions without relying on goal-completion timing.
"""

from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import anyio
import pytest

from swarm_arena import (
    AgentSpec,
    ConvergeSpec,
    FunctionTool,
    Goal,
    KeepTopTeams,
    PartSpec,
    Plan,
    ProgramJudge,
    Ref,
    RoundSpec,
    RunConfig,
    ScriptedManager,
    Swarm,
    TeamSpec,
)
from swarm_arena._journal.event import Event
from swarm_arena._journal.reader import read_committed
from swarm_arena._store.local import LocalStore
from swarm_arena.board import Candidate
from swarm_arena.tool import Estimate, RunStatus


def _value_judge() -> Ref:
    return Ref.of(ProgramJudge("_fakes.tools:score_value")).model_copy(update={"name": "value"})


def _line_plan(goal_id: str, *, rounds: int) -> Plan:
    return Plan(
        goal=Goal(id=goal_id, namespace="e2e"),
        teams=[
            TeamSpec(
                name="line",
                rationale="Evaluate successive values",
                agents=[
                    AgentSpec(
                        name="worker",
                        rationale="Evaluate the line",
                        tool=Ref.of(FunctionTool("_fakes.tools:line_value")),
                        manager=Ref.of(
                            ScriptedManager(
                                [
                                    {"round": r, "inputs": {"a": float(r)}}
                                    for r in range(1, rounds + 1)
                                ]
                            )
                        ),
                    )
                ],
            )
        ],
        judges=[_value_judge()],
        ranking="value",
        rounds=[RoundSpec() for _ in range(rounds)],
    )


def _parts_plan(goal_id: str, *, parts: list[str], rounds: int) -> Plan:
    return Plan(
        goal=Goal(id=goal_id, namespace="e2e"),
        parts=[
            PartSpec(name=part, goal=Goal(id=f"part-{part}", namespace="e2e")) for part in parts
        ],
        teams=[
            TeamSpec(
                name=part,
                part=part,
                rationale="Evaluate one part",
                agents=[
                    AgentSpec(
                        name="worker",
                        rationale="Improve the part each round",
                        tool=Ref.of(FunctionTool("_fakes.tools:part_value")),
                        manager=Ref.of(
                            ScriptedManager(
                                [
                                    {"round": r, "inputs": {"part": part, "v": float(10 * i + r)}}
                                    for r in range(1, rounds + 1)
                                ]
                            )
                        ),
                    )
                ],
            )
            for i, part in enumerate(parts, 1)
        ],
        judges=[_value_judge()],
        ranking="value",
        policy=Ref.of(KeepTopTeams(1)),
        rounds=[RoundSpec() for _ in range(rounds)],
        converge=ConvergeSpec(
            mode="team",
            team=TeamSpec(
                name="join",
                rationale="Sum the best parts",
                agents=[
                    AgentSpec(
                        name="joiner",
                        rationale="Read and combine assigned parents",
                        tool=Ref.of(FunctionTool("_fakes.tools:join_sum")),
                        manager=Ref.of(ScriptedManager([{"inputs": {}}])),
                    )
                ],
            ),
        ),
    )


def _events(cfg: RunConfig, type: str | None = None) -> list[Event]:
    """Read only the single run's authoritative committed journal prefix."""
    roots = list(Path(cfg.run_root).iterdir())
    assert len(roots) == 1
    committed = read_committed(LocalStore(roots[0]))
    assert committed is not None
    return [event for event in committed[1] if type is None or event.type == type]


def _first_agent_share(cfg: RunConfig, goal_id: str, *, round: int) -> Decimal:
    """Read the same part-agent's USD grant, excluding convergence shares."""
    [event] = [
        event
        for event in _events(cfg, "round_started")
        if event.goal_id == goal_id and event.round == round and not event.payload.get("converge")
    ]
    return Decimal(str(event.payload["shares"][f"{goal_id}/a/worker"]["usd"]))


def _peak_charged_usd(events: Sequence[Event]) -> Decimal:
    """Express the integer micro-dollar oracle in dollars for existing checks."""
    return Decimal(_peak_charged_micro_usd(events)) / 1_000_000


def _peak_charged_micro_usd(events: Sequence[Event]) -> int:
    """Measure charged exposure after each committed event, allowing refunds.

    Open, estimated and keep-charged reservations retain the full estimate.
    Ordinary settlements replace it with actual usage, including overspend;
    terminal usage overflow overrides even conservative charging with actuals.
    Integer micro-dollar sums do not round or clamp aggregate overflow.
    """

    def micro_usd(usd: float) -> int:
        units = Decimal(str(usd)) * 1_000_000
        assert units == units.to_integral_value()
        return int(units)

    charges: dict[str, int] = {}
    keep_charged: set[str] = set()
    peak = 0
    for event in events:
        payload = event.payload
        if event.type == "reserve":
            reservation = payload["reservation"]
            id = reservation["id"]
            assert id not in charges
            charges[id] = micro_usd(reservation["amount"]["usd"])
            if reservation["keep_charged"]:
                keep_charged.add(id)
        elif event.type == "settle":
            id = payload["id"]
            assert id in charges
            if payload.get("usage_overflow") or (
                not payload["estimated"] and id not in keep_charged
            ):
                charges[id] = micro_usd(payload["actual"]["usd"])
        peak = max(peak, sum(charges.values()))
    return peak


@pytest.mark.parametrize(
    ("estimated", "keep_charged", "overflow", "actual", "expected"),
    [
        (False, False, False, 0.0, "6"),
        (False, False, False, 8.0, "14"),
        (True, False, False, 0.0, "12"),
        (False, True, False, 0.0, "12"),
        (True, True, True, 1_000_000_000.0, "1000000006"),
    ],
    ids=["refund-reused", "overspend", "estimated", "keep-charged", "usage-overflow"],
)
def test_peak_charged_usd_tracks_committed_prefixes(
    estimated: bool, keep_charged: bool, overflow: bool, actual: float, expected: str
) -> None:
    def reserve(id: str) -> dict[str, Any]:
        return {
            "type": "reserve",
            "payload": {
                "reservation": {
                    "id": id,
                    "amount": {"tokens": 0, "usd": 6.0},
                    "keep_charged": keep_charged,
                }
            },
        }

    settled = {
        "type": "settle",
        "payload": {
            "id": "first",
            "actual": {"tokens": 0, "usd": actual},
            "estimated": estimated,
            "usage_overflow": overflow,
        },
    }
    # Reused funding can exceed $10 in gross reservations while never exposing
    # more than $6. Overflow instead retains the other outstanding reservation,
    # and its sum must not be clamped to the single-Amount USD range.
    records = (
        [reserve("first"), reserve("second"), settled]
        if overflow
        else [reserve("first"), settled, reserve("second")]
    )
    events = [
        Event.model_validate(
            {
                "epoch": 1,
                "seq": seq,
                "ts": datetime(2026, 1, 1, tzinfo=UTC),
                "run_id": "example",
                "context_id": "example",
                **record,
            }
        )
        for seq, record in enumerate(records, 1)
    ]
    assert _peak_charged_usd(events) == Decimal(expected)


async def test_two_goals_shared_budget_and_parts_join(run_config: RunConfig) -> None:
    cfg = run_config.model_copy(
        update={
            "limits": run_config.limits.model_copy(
                update={
                    "max_usd": 10.0,
                    "default_run_reservation": Estimate(tokens=0, usd=0.1),
                }
            )
        }
    )
    out = await Swarm(cfg).run(
        [_line_plan("g-line", rounds=1), _parts_plan("g-parts", parts=["a", "b"], rounds=2)]
    )
    assert set(out) == {"g-line", "g-parts"}
    assert out["g-line"]
    assert out["g-parts"]
    assert out["g-line"][0].output.data == {"value": 1.0}
    [join] = out["g-parts"]
    assert (join.goal_id, join.team, join.part, join.output.status) == (
        "g-parts",
        "join",
        None,
        RunStatus.OK,
    )

    # Journal inspection establishes parent provenance and budget sharing; the
    # public output proves that the real executor staged both parents' data.
    events = await anyio.to_thread.run_sync(_events, cfg)
    candidates = {
        candidate.id: candidate
        for event in events
        if event.type == "tool_end"
        for candidate in [Candidate.model_validate(event.payload["candidate"])]
    }
    parents = [candidates[id] for id in join.parent_ids]
    assert len(parents) == 2
    assert {parent.part for parent in parents} == {"a", "b"}
    assert all(parent.goal_id == "g-parts" and parent.round == 2 for parent in parents)
    assert {
        (candidate.part, candidate.round)
        for candidate in candidates.values()
        if candidate.goal_id == "g-parts" and candidate.team != "join"
    } == {("a", 1), ("b", 1), ("a", 2), ("b", 2)}
    assert all(candidate.output.status is RunStatus.OK for candidate in candidates.values())
    assert join.output.data == {"value": 34.0}
    assert join.output.data["value"] == sum(parent.output.data["value"] for parent in parents)
    assert {event.goal_id for event in events if event.type == "goal_started"} == {
        "g-line",
        "g-parts",
    }

    first = await anyio.to_thread.run_sync(lambda: _first_agent_share(cfg, "g-parts", round=1))
    later = await anyio.to_thread.run_sync(lambda: _first_agent_share(cfg, "g-parts", round=2))
    assert later > first > 0
    # This does not depend on when g-line returns its unused pool: g-parts has
    # one remaining round and zero actual spend, with the same surviving agents.
    assert Decimal(0) < _peak_charged_usd(events) <= Decimal("10.0")
    reservations = {
        event.payload["reservation"]["id"] for event in events if event.type == "reserve"
    }
    settlements = [event for event in events if event.type == "settle"]
    assert len(settlements) == len(reservations) > 0
    assert {event.payload["id"] for event in settlements} == reservations
    assert all(
        event.payload["actual"] == {"tokens": 0, "usd": 0.0} and event.payload["estimated"] is False
        for event in settlements
    )


async def test_two_goals_refuse_demand_above_shared_budget(run_config: RunConfig) -> None:
    """Offer $12 of paid requests and admit only one $3 request per goal."""
    cfg = run_config.model_copy(
        update={"limits": run_config.limits.model_copy(update={"max_usd": 10.0})}
    )
    plans = []
    for goal_id in ("g-first", "g-second"):
        plan = _line_plan(goal_id, rounds=1)
        team = plan.teams[0]
        agent = team.agents[0].model_copy(
            update={
                "tool": Ref(path="_fakes.tools:ThreeDollarTool"),
                "manager": Ref(path="_fakes.tools:ReportRefusalManager"),
            }
        )
        plans.append(
            plan.model_copy(update={"teams": [team.model_copy(update={"agents": [agent]})]})
        )

    out = await Swarm(cfg).run(plans)
    assert set(out) == {"g-first", "g-second"}
    assert all(len(candidates) == 1 for candidates in out.values())
    assert all(
        candidates[0].output.status is RunStatus.OK
        and candidates[0].output.data == {"value": 1}
        and candidates[0].output.usage.usd == 3.0
        for candidates in out.values()
    )

    events = await anyio.to_thread.run_sync(_events, cfg)
    reservations = [event.payload["reservation"] for event in events if event.type == "reserve"]
    assert len(reservations) == 2
    assert {reservation["holder"] for reservation in reservations} == {
        "g-first/line/worker",
        "g-second/line/worker",
    }
    assert all(reservation["amount"] == {"tokens": 0, "usd": 3.0} for reservation in reservations)
    # The manager records the public LimitReached.limit via post_learning;
    # envelope identities establish which two agents were actually refused.
    refusals = [event for event in events if event.type == "learning"]
    assert len(refusals) == 2
    assert {
        (event.goal_id, event.team, event.agent, event.payload["learning"]["text"])
        for event in refusals
    } == {("g-first", "line", "worker", "budget"), ("g-second", "line", "worker", "budget")}
    starts = [event for event in events if event.type == "tool_start"]
    assert len(starts) == 2
    assert {(event.goal_id, event.team, event.agent) for event in starts} == {
        ("g-first", "line", "worker"),
        ("g-second", "line", "worker"),
    }
    settlements = [event.payload for event in events if event.type == "settle"]
    assert len(settlements) == 2
    assert {settled["id"] for settled in settlements} == {
        reservation["id"] for reservation in reservations
    }
    assert all(
        settled["actual"] == {"tokens": 0, "usd": 3.0} and settled["estimated"] is False
        for settled in settlements
    )
    assert _peak_charged_micro_usd(events) == 6_000_000 <= 10_000_000


def _cross_namespace_plan() -> Plan:
    plan = _parts_plan("g-parts", parts=["a", "b"], rounds=2)
    plan = plan.model_copy(
        update={
            "parts": [
                part.model_copy(
                    update={
                        "goal": part.goal.model_copy(update={"namespace": f"other-{part.name}"})
                    }
                )
                for part in plan.parts
            ]
        }
    )
    for team in plan.teams:
        for agent in team.agents:
            agent.manager.config["post_learnings"] = True
    return plan


async def cross_namespace_crash(config_path: str) -> None:
    """Exit after the committed second-round frontier of the split plan."""
    import os

    from swarm_arena._journal.writer import JournalWriter

    original = JournalWriter._publish

    async def publish(self: JournalWriter) -> None:
        hit = any(event.type == "frontier" and event.round == 2 for event in self._queue)
        await original(self)
        if hit:
            os._exit(91)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(JournalWriter, "_publish", publish)
        await Swarm(RunConfig.load(config_path)).run(_cross_namespace_plan(), run_id="split")


@pytest.mark.parametrize("resume", [False, True])
async def test_cross_namespace_parts_run_and_resume(
    run_config: RunConfig, tmp_path: Path, resume: bool
) -> None:
    import os
    import sys

    config_path = tmp_path / "run.toml"
    config_path.write_text(run_config.dumps())
    if resume:
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join(
            [str(Path(__file__).parents[1]), str(Path(__file__).parents[2] / "src")]
        )
        child = await anyio.run_process(
            [
                sys.executable,
                "-c",
                "import sys, anyio; from e2e.test_many_goals_parts import cross_namespace_crash; "
                "anyio.run(cross_namespace_crash, sys.argv[1])",
                str(config_path),
            ],
            env=environment,
            check=False,
        )
        assert child.returncode == 91, child.stderr.decode()
        result = await Swarm(run_config).resume("split", force=True)
    else:
        result = await Swarm(run_config).run(_cross_namespace_plan(), run_id="split")
    assert len(result["g-parts"]) == 1
    winner = result["g-parts"][0]
    assert (winner.team, winner.agent, winner.round, winner.output.data) == (
        "join",
        "joiner",
        3,
        {"value": 34.0},
    )
    assert len(_events(run_config, "tool_end")) == 5
    learnings = _events(run_config, "learning")
    assert len(learnings) == 2
    assert {event.payload["learning"]["namespace"] for event in learnings} == {"other-a", "other-b"}
    assert not _events(run_config, "no_result")
