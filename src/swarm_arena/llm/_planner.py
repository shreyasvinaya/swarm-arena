"""Generate catalog-authorized plans inside owned, metered planning scopes."""

import json
from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any, Final

import anyio

from swarm_arena._board.digest import untrusted
from swarm_arena._budget.llm_scope import current_llm_scope, llm_scope
from swarm_arena._engine.run import mark_planning_session, planning_session
from swarm_arena._errors import _core_category, _core_error, _is_core_control, _SignalCategory
from swarm_arena._model.base import CatalogKind
from swarm_arena._model.catalog import PlannerCatalog
from swarm_arena._model.context_strings import _context_string
from swarm_arena._model.detach import _detach_model
from swarm_arena._model.goal import FileSet, Goal
from swarm_arena._model.limits import Limits
from swarm_arena._model.plan import AgentSpec, ConvergeSpec, PartSpec, Plan, RoundSpec, TeamSpec
from swarm_arena._model.planning_types import History
from swarm_arena._model.ref import Ref, _validate_config
from swarm_arena._model.run_config import RunConfig
from swarm_arena._outcomes import _Outcome
from swarm_arena._planning.planner import Planner
from swarm_arena._planning.provenance import _detach_plan, stamp
from swarm_arena._planning.validate import _admit_draft, _freeze_draft_goal, check_models
from swarm_arena.llm._hooks import BudgetExceeded, _prepare
from swarm_arena.llm._manager import _finish_accounting
from swarm_arena.llm._reviewer import Approve, Edited, Reviewer, _detach_decision
from swarm_arena.llm._structured import DictOutputSchema, structured_call

_ENTRY: Final[dict[str, Any]] = {
    "type": "object",
    "required": ["entry"],
    "properties": {"entry": {"type": "string"}, "settings": {"type": "object"}},
}
_TEAM: Final[dict[str, Any]] = {
    "type": "object",
    "required": ["name", "rationale", "agents"],
    "properties": {
        "name": {"type": "string"},
        "rationale": {"type": "string"},
        "part": {"type": ["string", "null"]},
        "agents": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "rationale", "tool", "manager"],
                "properties": {
                    "name": {"type": "string"},
                    "rationale": {"type": "string"},
                    "tool": _ENTRY,
                    "manager": _ENTRY,
                    "replicates": {"type": "integer"},
                },
            },
        },
    },
}
PLANNER_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "rationale": {"type": "string"},
        "keep_teams": {"type": "array", "items": {"type": "string"}},
        "keep_parts": {"type": "array", "items": {"type": "string"}},
        "parts": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name"],
                "properties": {
                    "name": {"type": "string"},
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "inputs": {"type": "object"},
                    "files": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "teams": {"type": "array", "items": _TEAM},
        "checks": {"type": "array", "items": _ENTRY},
        "judges": {
            "type": "array",
            "items": {
                **_ENTRY,
                "properties": {**_ENTRY["properties"], "name": {"type": "string"}},
            },
        },
        "ranking": {"type": "string"},
        "policy": {**_ENTRY, "type": ["object", "null"]},
        "rounds": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "tool_overrides": {"type": "object"},
                    "fidelity": {"type": "string"},
                },
            },
        },
        "converge": {
            "type": ["object", "null"],
            "properties": {
                "mode": {"type": "string"},
                "top_n": {"type": "integer"},
                "team": {**_TEAM, "type": ["object", "null"]},
            },
        },
    },
}


def plan_from_output(
    output: Mapping[str, Any],
    *,
    goal: Goal,
    catalog: PlannerCatalog,
    current: Plan | None,
    rounds_done: int,
    limits: Limits,
) -> Plan:
    """Convert finite structured output without granting reference authority.

    Parameters
    ----------
    output : Mapping[str, Any]
        Catalog selections and planner-settable fields.
    goal : Goal
        Frozen human main goal.
    catalog : PlannerCatalog
        Human-approved component entries.
    current : Plan or None
        Current revision, whose explicitly kept components retain identity.
    rounds_done : int
        Completed ordinary rounds, including implicit schedules.
    limits : Limits
        Run limits used to interpret implicit round schedules.

    Returns
    -------
    Plan
        Proposed plan. The caller stamps and admits it before returning success.

    Examples
    --------
    >>> goal = Goal(id="g", namespace="n")
    >>> plan_from_output({}, goal=goal, catalog=PlannerCatalog(), current=None,
    ...                  rounds_done=0, limits=Limits()).goal == goal
    True
    """
    if type(rounds_done) is not int or rounds_done < 0:
        raise _core_error(ValueError, "invalid rounds_done")
    try:
        data = DictOutputSchema("planner", PLANNER_SCHEMA).validate_json(json.dumps(dict(output)))
    except (ValueError, TypeError):
        raise _core_error(ValueError, "invalid planner output") from None

    def reference(row: dict[str, Any], kind: CatalogKind) -> Ref:
        name = _context_string(row["entry"], "entry")
        try:
            entry = catalog.get(name)
        except ValueError:
            raise _core_error(ValueError, f"unknown catalog entry {name}") from None
        if entry.kind != kind:
            raise _core_error(ValueError, f"catalog entry {name}: invalid kind")
        settings = row.get("settings", {})
        _validate_config(entry.settable_model(), settings, where=f"catalog entry {name}")
        return Ref(path=entry.ref, config={**entry.config, **settings})

    def team(row: dict[str, Any]) -> TeamSpec:
        return TeamSpec(
            name=row["name"],
            rationale=row["rationale"],
            part=row.get("part"),
            agents=[
                AgentSpec(
                    name=agent["name"],
                    rationale=agent["rationale"],
                    tool=reference(agent["tool"], "tool"),
                    manager=reference(agent["manager"], "manager"),
                    replicates=agent.get("replicates", 1),
                )
                for agent in row["agents"]
            ],
        )

    def kept(kind: str) -> list[Any]:
        names = data.get(f"keep_{kind}", [])
        available = {item.name: item for item in getattr(current, kind)} if current else {}
        defined = {row["name"] for row in data.get(kind, [])}
        if len(set(names)) != len(names) or any(
            name not in available or name in defined for name in names
        ):
            raise _core_error(ValueError, f"invalid keep_{kind}")
        return [available[name] for name in names]

    try:
        teams = kept("teams") + [team(row) for row in data.get("teams", [])]
        parts = kept("parts") + [
            PartSpec(
                name=row["name"],
                goal=Goal(
                    id=f"{goal.id}.{row['name']}",
                    namespace=goal.namespace,
                    title=row.get("title", ""),
                    description=row.get("description", ""),
                    inputs=row.get("inputs", {}),
                    files=FileSet(globs=row.get("files", [])),
                ),
            )
            for row in data.get("parts", [])
        ]
        checks = [reference(row, "check") for row in data.get("checks", [])]
        rounds = [
            RoundSpec(tool_overrides=row.get("tool_overrides", {}), fidelity=row.get("fidelity"))
            for row in data.get("rounds", [])
        ]
        converge_data = data.get("converge")
        converge = (
            (current.converge if current else ConvergeSpec())
            if converge_data is None
            else ConvergeSpec(
                mode=converge_data.get("mode", "select"),
                top_n=converge_data.get("top_n", 2),
                team=team(converge_data["team"]) if converge_data.get("team") is not None else None,
            )
        )
        if current is not None:
            checks = [*current.checks, *checks]
            judges, ranking, policy = current.judges, current.ranking, current.policy
            rounds = (
                (
                    [
                        current.rounds[i] if i < len(current.rounds) else RoundSpec()
                        for i in range(rounds_done)
                    ]
                    + rounds
                )
                if rounds
                else current.rounds
            )
        else:
            judges = [
                reference(row, "judge").model_copy(update={"name": row.get("name")})
                for row in data.get("judges", [])
            ]
            ranking = data.get("ranking", "")
            policy = (
                reference(data["policy"], "policy")
                if data.get("policy") is not None
                else Ref(path="swarm_arena:SuccessiveHalving")
            )
        result = Plan(
            goal=goal,
            teams=teams,
            parts=parts,
            checks=checks,
            judges=judges,
            ranking=ranking,
            policy=policy,
            rounds=rounds,
            converge=converge,
            rationale=data.get("rationale", ""),
        )
    except (ValueError, TypeError) as error:
        # Registered entry diagnostics are already value-free; model errors are not.
        from swarm_arena._errors import _rebuild_core_signal

        safe = _rebuild_core_signal(error)
        if safe is not None:
            raise safe from None
        raise _core_error(ValueError, "invalid planner output") from None
    if current is not None:
        result._base_dir = current.base_dir
    return result


def _history_copy(history: History | None) -> History | None:
    if history is None:
        return None
    detached = _detach_model(History, history)
    return detached.model_copy(update={"revisions": [_detach_plan(p) for p in history.revisions]})


class LLMPlanner(Planner):
    """Generate admitted plans using the human-approved component catalog.

    Parameters
    ----------
    model : str
        Model alias with the planner role.
    instructions : str, default=''
        Additional trusted planning instructions.

    Examples
    --------
    >>> from swarm_arena.llm import LLMPlanner
    >>> LLMPlanner("planner").models()
    ('planner',)
    """

    origin = "llm"

    def __init__(self, model: str, *, instructions: str = "") -> None:
        self._model = _context_string(model, "alias")
        self._instructions = _context_string(instructions, "instructions")

    def models(self) -> tuple[str, ...]:
        """Return the declared planner alias.

        Returns
        -------
        tuple[str, ...]
            One alias, checked before paid planning.

        Examples
        --------
        >>> LLMPlanner("planner").models()
        ('planner',)
        """
        return (self._model,)

    async def plan(self, goal: Goal, config: RunConfig, history: History | None) -> Plan:
        """Return an admitted draft with core-assigned LLM provenance.

        Parameters
        ----------
        goal : Goal
            Human goal; initial relative files resolve against cwd.
        config : RunConfig
            Human model and component catalogs and planning limits.
        history : History or None
            Fixed run evidence; r0 supplies the source directory when present.

        Returns
        -------
        Plan
            Admitted revision-zero proposal with completed planner accounting.

        Examples
        --------
        >>> plan = await LLMPlanner("planner").plan(goal, config, None)  # doctest: +SKIP
        """
        if current_llm_scope() is None:
            async with planning_session(config, goal, self) as scope:
                result = await self._draft(goal, config, history)
                result = stamp(result, origin=type(self).origin, parent=None)
                result = mark_planning_session(result, scope)
            return result
        return stamp(
            await self._draft(goal, config, history), origin=type(self).origin, parent=None
        )

    async def _draft(
        self,
        goal: Goal,
        config: RunConfig,
        history: History | None,
        *,
        feedback: Sequence[str] = (),
        previous: Plan | None = None,
    ) -> Plan:
        history = await _prepare(lambda: _history_copy(history))
        base_dir = history.revisions[0].base_dir if history and history.revisions else Path.cwd()
        return await self._draft_at(
            goal, config, history, base_dir=base_dir, feedback=feedback, previous=previous
        )

    async def _draft_at(
        self,
        goal: Goal,
        config: RunConfig,
        history: History | None,
        *,
        base_dir: Path,
        feedback: Sequence[str] = (),
        previous: Plan | None = None,
    ) -> Plan:
        """Generate and admit with an explicit source directory for file callers."""
        parent = current_llm_scope()
        if parent is None:
            raise _core_error(RuntimeError, "LLM planner outside an LLM scope")
        scope = parent.child()
        boundary = _Outcome()
        failure: BaseException | None = None
        interrupted = False
        result: Plan | None = None
        try:
            with llm_scope(scope):
                errors, _ = await _prepare(lambda: check_models({"planner": self.models()}, config))
                if errors:
                    raise _core_error(ValueError, "\n".join(errors))
                frozen = await _freeze_draft_goal(
                    goal, base_dir=base_dir, artifacts=scope.artifacts
                )
                revisions = history.revisions if history else []
                current = revisions[-1] if revisions else None
                prompt = await _prepare(
                    lambda: untrusted(
                        json.dumps(
                            {
                                "goal": frozen.model_dump(mode="json"),
                                "catalog": [
                                    {
                                        "name": e.name,
                                        "kind": e.kind,
                                        "description": e.description,
                                        "settable": e.settable,
                                    }
                                    for e in config.planner_catalog.entries
                                ],
                                "models": {
                                    alias: {"roles": spec.roles}
                                    for alias, spec in config.models.root.items()
                                },
                                "limits": config.limits.model_dump(mode="json"),
                                "current": None
                                if current is None
                                else {
                                    "teams": [t.name for t in current.teams],
                                    "parts": [p.name for p in current.parts],
                                },
                                "history": None
                                if history is None
                                else {
                                    "scores": [s.model_dump(mode="json") for s in history.scores],
                                    "learning_titles": [item.title for item in history.learnings],
                                    "prune_log": [
                                        (r, d.model_dump(mode="json")) for r, d in history.decisions
                                    ],
                                    "budget_left": history.budget_left.model_dump(mode="json"),
                                    "rounds_done": history.rounds_done,
                                    "join_failures": history.join_failures,
                                },
                                "feedback": list(feedback),
                                "previous": previous.model_dump(mode="json") if previous else None,
                            }
                        )
                    )
                )
                output, _ = await structured_call(
                    alias=self._model,
                    role="planner",
                    models=config.models,
                    namespace=frozen.namespace,
                    system=(
                        "Select catalog entries and only their settable settings. "
                        "Keep human components with keep_teams and keep_parts. "
                        + self._instructions
                    ),
                    user=prompt,
                    schema=PLANNER_SCHEMA,
                    name="planner",
                    no_cache_read=False,
                    temperature=None,
                )
                result = await _prepare(
                    partial(
                        plan_from_output,
                        output,
                        goal=frozen,
                        catalog=config.planner_catalog,
                        current=current,
                        rounds_done=history.rounds_done if history else 0,
                        limits=config.limits,
                    )
                )
                result = await _admit_draft(
                    result,
                    config,
                    scope.artifacts,
                    frozen_goal=frozen,
                    base_dir=base_dir,
                    history=revisions,
                    origin=type(self).origin,
                )
        except BaseException as error:
            failure = error
            try:
                boundary.classify(error)
            except anyio.get_cancelled_exc_class() as cancelled:
                interrupted, failure = True, cancelled
            except BaseException as control:
                failure = control
        finally:
            try:
                await _finish_accounting(scope, interrupted=interrupted)
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as cleanup_error:
                failure = (
                    BaseExceptionGroup("planner cleanup", [failure, cleanup_error])
                    if failure
                    else cleanup_error
                )
        if failure is not None:
            classified = boundary.classify(failure)
            if _is_core_control(classified):
                if (
                    type(failure) is BudgetExceeded
                    or _core_category(failure) is _SignalCategory.LIMIT
                ):
                    raise failure from None
                raise classified from None
            boundary.contain(classified, ValueError, "LLM planner")
        boundary.propagate_control(RuntimeError())
        assert result is not None
        return result


class CollaborativePlanner(Planner):
    """Draft and review plans inside one bounded planning session.

    Parameters
    ----------
    planner : LLMPlanner
        Metered draft generator.
    reviewer : Reviewer
        Cooperatively cancellable human review adapter.
    max_iterations : int, default=5
        Positive maximum number of reviews.

    Examples
    --------
    >>> from swarm_arena import Plan
    >>> class Human:
    ...     async def review(self, plan: Plan) -> Approve:
    ...         return Approve()
    >>> CollaborativePlanner(LLMPlanner("planner"), Human()).models()
    ('planner',)
    """

    origin = "together"

    def __init__(self, planner: LLMPlanner, reviewer: Reviewer, *, max_iterations: int = 5) -> None:
        if type(max_iterations) is not int or max_iterations <= 0:
            raise _core_error(ValueError, "max_iterations must be a positive integer")
        self._planner = planner
        self._reviewer = reviewer
        self._max_iterations = max_iterations

    def models(self) -> tuple[str, ...]:
        """Return the underlying planner's model aliases.

        Returns
        -------
        tuple[str, ...]
            Declared planning aliases.

        Examples
        --------
        >>> from swarm_arena.llm import Approve
        >>> from swarm_arena import Plan
        >>> class Human:
        ...     async def review(self, plan: Plan) -> Approve:
        ...         return Approve()
        >>> CollaborativePlanner(LLMPlanner("planner"), Human()).models()
        ('planner',)
        """
        return self._planner.models()

    async def plan(self, goal: Goal, config: RunConfig, history: History | None) -> Plan:
        """Return a reviewed plan admitted with together provenance.

        Parameters
        ----------
        goal : Goal
            Human main goal frozen before the first draft.
        config : RunConfig
            Catalog, budget, storage and session allowance.
        history : History or None
            Previous admitted revisions and run evidence.

        Returns
        -------
        Plan
            Approved or edited admitted proposal.

        Raises
        ------
        RuntimeError
            No approval within the configured review count.

        Examples
        --------
        >>> plan = await planner.plan(goal, config, None)  # doctest: +SKIP
        """
        return await self._plan_at(goal, config, history, base_dir=Path.cwd())

    async def _plan_at(
        self,
        goal: Goal,
        config: RunConfig,
        history: History | None,
        *,
        base_dir: Path,
    ) -> Plan:
        """Review with the explicit source root supplied by file-based callers."""
        if current_llm_scope() is None:
            async with planning_session(config, goal, self) as scope:
                result = mark_planning_session(
                    await self._review(goal, config, history, base_dir=base_dir), scope
                )
            return result
        return await self._review(goal, config, history, base_dir=base_dir)

    async def _review(
        self,
        goal: Goal,
        config: RunConfig,
        history: History | None,
        *,
        base_dir: Path,
    ) -> Plan:
        scope = current_llm_scope()
        assert scope is not None
        boundary = _Outcome()
        history = await _prepare(lambda: _history_copy(history))
        revisions = history.revisions if history else []
        frozen = await _freeze_draft_goal(goal, base_dir=base_dir, artifacts=scope.artifacts)
        feedback: list[str] = []
        previous = None
        for _ in range(self._max_iterations):
            draft = await self._planner._draft_at(
                frozen, config, history, base_dir=base_dir, feedback=feedback, previous=previous
            )
            try:
                offered = await _prepare(partial(_detach_plan, draft))
                decision = await self._reviewer.review(offered)
                decision = await _prepare(partial(_detach_decision, decision))
                if isinstance(decision, Approve | Edited):
                    proposed = draft if isinstance(decision, Approve) else decision.plan
                    return await _admit_draft(
                        proposed,
                        config,
                        scope.artifacts,
                        frozen_goal=frozen,
                        base_dir=proposed.base_dir,
                        history=revisions,
                        origin="together",
                    )
                feedback.append(decision.text)
            except BaseException as error:
                boundary.contain(error, RuntimeError, "reviewer")
            previous = draft
        raise _core_error(RuntimeError, f"plan not approved after {self._max_iterations} reviews")
