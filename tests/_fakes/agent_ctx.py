"""In-memory AgentContext fields and actions for optional manager tests."""

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel

from swarm_arena import AgentSpec, Goal, Ref, RoundSpec, Score
from swarm_arena._budget.llm_scope import current_requesting_call
from swarm_arena._errors import SoftError
from swarm_arena._model.base import JsonValue
from swarm_arena._scoring.runner import for_manager
from swarm_arena.board import Candidate, Learning, Row
from swarm_arena.tool import Action, Estimate, ParentRef, ToolOutput, ToolRequest


def _spec() -> AgentSpec:
    return AgentSpec(
        name="agent",
        rationale="Explore",
        tool=Ref(path="example:Tool"),
        manager=Ref(path="example:Manager"),
    )


@dataclass
class FakeAgentContext:
    """Hold R4.4 fields and deterministic action results without core services.

    Tests pass this fixture as ``cast(AgentContext, fake)``. It intentionally
    does not model frontier authorization or accounting; those use real core
    contexts. ``rows`` is fixed own-group membership; ``part`` names that
    group (None for main). ``sibling_rows`` explicitly declares additional
    groups, using the empty string for main, independently of producer labels.
    ``feedback_by_candidate`` supplies each producing candidate's check policy;
    absent/unknown policies hide private feedback through the core projection.
    ``learnings`` holds lookup-visible items in snapshot order, while
    ``own_learnings`` records own publications for newest-first search.
    ``raise_on`` injects an exact exception at any action/read and
    ``requesting_calls`` records the active call before tool and learning work.

    Examples
    --------
    >>> fake = FakeAgentContext()
    >>> fake.my_candidates()
    []
    """

    goal: Goal = field(default_factory=lambda: Goal(id="g", namespace="n"))
    spec: AgentSpec = field(default_factory=_spec)
    round: int = 1
    round_spec: RoundSpec = field(default_factory=RoundSpec)
    digest: str = ""
    notes: str = ""
    budget: Estimate = field(default_factory=Estimate)
    resumed: bool = False
    seed_parents: tuple[str, ...] = ()
    tool_description: str = "Run a trial."
    input_model: type[BaseModel] = BaseModel
    editable: tuple[str, ...] = ()
    actions: tuple[Action[Any], ...] = ()
    raise_on: dict[str, BaseException] = field(default_factory=dict)
    requesting_calls: list[tuple[str, str | None]] = field(default_factory=list)
    candidates: list[Candidate] = field(default_factory=list)
    learnings: list[Learning] = field(default_factory=list)
    scores: list[Score] = field(default_factory=list)
    rows: list[Row] = field(default_factory=list)
    part: str | None = None
    sibling_rows: dict[str, list[Row]] = field(default_factory=dict)
    feedback_by_candidate: dict[str, dict[str, str]] = field(default_factory=dict)
    own_learnings: list[Learning] = field(default_factory=list)
    artifact_bytes: dict[str, bytes] = field(default_factory=dict)
    action_handlers: dict[str, Callable[[Mapping[str, Any]], Awaitable[JsonValue]]] = field(
        default_factory=dict
    )

    def _raise(self, name: str) -> None:
        if error := self.raise_on.get(name):
            raise error

    def _manager(self, candidate: Candidate) -> Candidate:
        """Apply the shared producing-policy projection to a detached copy."""
        return for_manager(
            candidate.model_copy(deep=True), self.feedback_by_candidate.get(candidate.id, {})
        )

    def add_candidate(self, id: str) -> Candidate:
        """Seed an in-memory prior result.

        Examples
        --------
        >>> FakeAgentContext().add_candidate("c-1").id
        'c-1'
        """
        candidate = Candidate(
            id=id,
            run_id="run-1",
            goal_id=self.goal.id,
            team="team",
            agent=self.spec.name,
            round=self.round,
            request=ToolRequest(),
            output=ToolOutput(status="ok"),
        )
        self.candidates.append(candidate)
        return candidate

    async def run_tool(
        self,
        inputs: Mapping[str, Any],
        *,
        parents: Sequence[str] = (),
        replicate: int | None = None,
        edits: Mapping[str, str] | None = None,
    ) -> Candidate:
        """Record a tool request and its active requesting call in memory.

        Examples
        --------
        >>> import anyio
        >>> anyio.run(FakeAgentContext().run_tool, {"x": 1}).request.inputs
        {'x': 1}
        """
        call = current_requesting_call()
        self.requesting_calls.append(("run_tool", call))
        self._raise("run_tool")
        candidate = self.add_candidate(f"c-fake-{len(self.candidates) + 1:04d}")
        candidate = candidate.model_copy(
            update={
                "request": ToolRequest(
                    inputs=dict(inputs),
                    parents=[
                        ParentRef(candidate_id=id, run_id="run-1", tool_identity="fake")
                        for id in dict.fromkeys([*self.seed_parents, *parents])
                    ],
                    replicate=0 if replicate is None else replicate,
                    round=self.round,
                    round_overrides=self.round_spec.tool_overrides,
                    edits=dict(edits or {}),
                ),
                "requested_by_call": call,
            }
        )
        self.candidates[-1] = candidate
        return self._manager(candidate)

    def my_candidates(self) -> list[Candidate]:
        """Return detached seeded and newly made results.

        Examples
        --------
        >>> FakeAgentContext().my_candidates()
        []
        """
        self._raise("my_candidates")
        return [self._manager(item) for item in self.candidates]

    def scores_of(self, candidate_id: str) -> list[Score]:
        """Return seeded scores for an identifier.

        Examples
        --------
        >>> FakeAgentContext().scores_of("c-1")
        []
        """
        self._raise("scores_of")
        return [
            item.model_copy(deep=True) for item in self.scores if item.candidate_id == candidate_id
        ]

    def search_learnings(
        self,
        query: str,
        *,
        tags: Sequence[str] = (),
        scope: Literal["goal", "namespace"] = "goal",
        limit: int = 10,
    ) -> list[Learning]:
        """Search newest own items before snapshot items in the selected scope.

        Examples
        --------
        >>> FakeAgentContext().search_learnings("")
        []
        """
        self._raise("search_learnings")
        if scope not in ("goal", "namespace"):
            raise SoftError("invalid learning scope")

        def matches(item: Learning) -> bool:
            return (
                not item.hidden
                and item.namespace == self.goal.namespace
                and (scope == "namespace" or item.goal_id == self.goal.id)
                and set(tags).issubset(item.tags)
                and (
                    query.casefold() in item.title.casefold()
                    or query.casefold() in item.text.casefold()
                )
            )

        own = [item for item in reversed(self.own_learnings) if matches(item)]
        found = [item for item in self.learnings if matches(item)][: max(0, limit)]
        unique = {item.id: item for item in [*own, *found]}
        return [item.model_copy(deep=True) for item in list(unique.values())[: max(0, limit)]]

    def read_learning(self, id: str) -> Learning:
        """Return a detached seeded learning.

        Examples
        --------
        >>> fake = FakeAgentContext()
        >>> import anyio
        >>> hint = anyio.run(fake.post_learning, "Hint", "Try this", ["c-1"])
        >>> fake.read_learning(hint.id) == hint
        True
        """
        self._raise("read_learning")
        return next(item.model_copy(deep=True) for item in self.learnings if item.id == id)

    def get_candidate(self, id: str) -> Candidate:
        """Return a detached seeded candidate.

        Examples
        --------
        >>> fake = FakeAgentContext()
        >>> _ = fake.add_candidate("c-1")
        >>> fake.get_candidate("c-1").id
        'c-1'
        """
        self._raise("get_candidate")
        return next(self._manager(item) for item in self.candidates if item.id == id)

    def _read_artifact(self, sha256: str) -> bytes:
        self._raise("_read_artifact")
        return self.artifact_bytes[sha256]

    def best(self, k: int, *, part: str | None = None) -> list[Row]:
        """Return fixed own/sibling membership with the real context's selectors.

        None selects the own group; the empty string selects main. A named
        sibling must be explicitly declared, even when its rows are empty.
        Recorded producer part labels do not determine membership.

        Examples
        --------
        >>> FakeAgentContext().best(1)
        []
        """
        self._raise("best")
        group = (self.part or "") if part is None else part
        if group in self.sibling_rows:
            rows = self.sibling_rows[group]
        elif group == (self.part or ""):
            rows = self.rows
        else:
            raise SoftError("part outside board snapshot")
        return [item.model_copy(deep=True) for item in rows[: max(0, k)]]

    async def post_learning(
        self,
        title: str,
        text: str,
        evidence: Sequence[str],
        *,
        tags: Sequence[str] = (),
        refines: Sequence[str] = (),
        duplicate_reason: str | None = None,
    ) -> Learning:
        """Record a learning and its active requesting call in memory.

        Examples
        --------
        >>> import anyio
        >>> anyio.run(FakeAgentContext().post_learning, "Hint", "Try this", ["c-1"]).title
        'Hint'
        """
        call = current_requesting_call()
        self.requesting_calls.append(("post_learning", call))
        self._raise("post_learning")
        hint = Learning(
            id=f"L-fake-{len(self.learnings) + 1:04d}",
            namespace=self.goal.namespace,
            run_id="run-1",
            goal_id=self.goal.id,
            round=self.round,
            author=f"team/{self.spec.name}",
            title=title,
            text=text,
            evidence=list(evidence),
            tags=list(tags),
            refines=list(refines),
            posted_by_call=call,
        )
        self.learnings.append(hint)
        self.own_learnings.append(hint)
        return hint.model_copy(deep=True)

    async def save_notes(self, text: str) -> None:
        """Save carried notes in memory.

        Examples
        --------
        >>> import anyio
        >>> fake = FakeAgentContext()
        >>> anyio.run(fake.save_notes, "Try again")
        >>> fake.notes
        'Try again'
        """
        self._raise("save_notes")
        self.notes = text

    async def call_action(self, name: str, args: Mapping[str, Any]) -> JsonValue:
        """Invoke a test-supplied in-memory async action.

        Examples
        --------
        >>> fake = FakeAgentContext()
        >>> async def echo(args: Mapping[str, Any]) -> JsonValue:
        ...     return dict(args)
        >>> fake.action_handlers["echo"] = echo
        >>> import anyio
        >>> anyio.run(fake.call_action, "echo", {"x": 1})
        {'x': 1}
        """
        self._raise("call_action")
        return await self.action_handlers[name](args)
