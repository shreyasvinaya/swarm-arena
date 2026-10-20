"""Identify repeated requests and reserve bounded nondeterministic replicates."""

from collections.abc import Iterable
from collections.abc import Set as AbstractSet
from typing import TYPE_CHECKING, Literal

from swarm_arena._agent.manager import LimitReached
from swarm_arena._errors import Fenced, SoftError
from swarm_arena._journal.event import EventDraft, EventType
from swarm_arena._model.base import FrozenModel, JsonValue
from swarm_arena._model.canonical import canonical_json, sha256_hex
from swarm_arena._model.goal import GoalKey
from swarm_arena._model.ref import Ref
from swarm_arena._model.results import Candidate
from swarm_arena._model.tooling import RunStatus, ToolRequest

if TYPE_CHECKING:
    from swarm_arena._agent.runtime import AgentRuntime

COUNTED = frozenset({RunStatus.OK, RunStatus.ERROR})
"""Complete outcomes that consume repeat and replicate slots."""


class RepeatOutcome(FrozenModel):
    """
    Describe a repeat decision without changing counters or budget.

    Parameters
    ----------
    action : {'run', 'refuse', 'return_own', 'link'}
        Whether to run anew, refuse an own repeat, recover, or link a result.
    candidate : Candidate or None, default=None
        Matched same-run candidate for a refusal, recovery, or link.
    hint : Candidate or None, default=None
        Past-run candidate that permits a new run.

    Examples
    --------
    >>> RepeatOutcome(action="run").candidate is None
    True
    """

    action: Literal["run", "refuse", "return_own", "link"]
    candidate: Candidate | None = None
    hint: Candidate | None = None


def repeat_key(
    *, goal_key: GoalKey, tool_ref: Ref, request: ToolRequest, deterministic: bool
) -> str:
    """
    Hash goal, base tool, overrides, inputs, edit hashes, and ordered parent ids.

    Parameters
    ----------
    goal_key : GoalKey
        Snapshotted main-goal or part-goal identity.
    tool_ref : Ref
        Snapshotted base tool reference, without merged round overrides.
    request : ToolRequest
        Live or journal-loaded request. Round is excluded; edits are normalized
        using their serializer rather than the live-content hasher.
    deterministic : bool
        Whether to exclude the replicate index.

    Returns
    -------
    str
        SHA-256 digest of the canonical repeat fields.

    Examples
    --------
    >>> from swarm_arena import Goal
    >>> ref = Ref(path="swarm_arena:ScriptTool", config={"command": ["true"]})
    >>> req = ToolRequest(edits={"p.txt": "new"})
    >>> loaded = ToolRequest.model_validate_json(req.model_dump_json())
    >>> repeat_key(goal_key=Goal(id="g", namespace="n").key(), tool_ref=ref,
    ...            request=req, deterministic=True) == repeat_key(
    ...     goal_key=Goal(id="g", namespace="n").key(), tool_ref=ref,
    ...     request=loaded, deterministic=True)
    True
    """
    data: dict[str, JsonValue] = {
        "goal_key": goal_key.model_dump(mode="json"),
        "tool_ref": tool_ref.path,
        "tool_config": tool_ref.config,
        "round_overrides": request.round_overrides,
        "inputs": request.inputs,
        "edits": {
            path: ref.removeprefix("sha256:")
            for path, ref in request.model_dump(mode="json")["edits"].items()
        },
        "parents": [parent.candidate_id for parent in request.parents],
    }
    if not deterministic:
        data["replicate"] = request.replicate
    return sha256_hex(canonical_json(data))


class RepeatIndex:
    """
    Index caller-selected earlier results and this agent's current-round items.

    Parameters
    ----------
    run_id : str
        Current run identifier.
    entries : Iterable[tuple[str, Candidate]], default=()
        Initial repeat-key/candidate pairs, including recovered own results.

    Attributes
    ----------
    added : list[tuple[str, Candidate]]
        Pairs added after construction, in insertion order.

    Notes
    -----
    The round runner creates an index per agent. It supplies only earlier-round
    results and that agent's resumed entries, preserving snapshot visibility.
    Reservations and lookups run on the loop with no intervening await.

    Examples
    --------
    >>> idx = RepeatIndex("r")
    >>> idx.lookup("absent", agent="t/a", deterministic=True).action
    'run'
    >>> idx.added
    []
    """

    def __init__(self, run_id: str, entries: Iterable[tuple[str, Candidate]] = ()) -> None:
        """Copy the initial entries without recording them as new additions."""
        self.run_id = run_id
        self.added: list[tuple[str, Candidate]] = []
        self._entries: dict[str, list[Candidate]] = {}
        self._candidates: list[Candidate] = []
        self._pending: set[str] = set()
        # Keep the original objects alive until release, so identity cannot be reused.
        self._allocations: dict[tuple[int, int, int], tuple[AgentRuntime, ToolRequest, str]] = {}
        for key, candidate in entries:
            self._insert(key, candidate)

    def _insert(self, key: str, candidate: Candidate) -> None:
        """Retain both keyed lookup order and overall candidate insertion order."""
        self._entries.setdefault(key, []).append(candidate)
        self._candidates.append(candidate)

    def add(self, key: str, candidate: Candidate) -> None:
        """
        Add a completed result, release its pending key, and record the pair.

        Parameters
        ----------
        key : str
            Final request's repeat key.
        candidate : Candidate
            Completed result, including failed or partial outcomes.

        Examples
        --------
        >>> c = Candidate(id="c1", run_id="r", goal_id="g", team="t", agent="a",
        ...     round=1, request=ToolRequest(), output={"status": "ok"})
        >>> idx = RepeatIndex("r")
        >>> idx.add("k", c)
        >>> idx.added == [("k", c)]
        True
        """
        self._insert(key, candidate)
        self._pending.discard(key)
        for token, (_, _, pending_key) in tuple(self._allocations.items()):
            if pending_key == key:
                del self._allocations[token]
        self.added.append((key, candidate))

    def key(self, rt: "AgentRuntime", request: ToolRequest) -> str:
        """
        Compute a repeat key using the base ref and effective determinism hook.

        Parameters
        ----------
        rt : AgentRuntime
            Agent's runtime.
        request : ToolRequest
            Final request to identify.

        Returns
        -------
        str
            Canonical repeat key.

        Examples
        --------
        >>> from types import SimpleNamespace
        >>> from swarm_arena import Goal, ScriptTool
        >>> rt = SimpleNamespace(goal_key=Goal(id="g", namespace="n").key(),
        ...     tool_ref=Ref(path="swarm_arena:ScriptTool"),
        ...     tool=ScriptTool(command=["true"]))
        >>> len(RepeatIndex("r").key(rt, ToolRequest()))
        64
        """
        return repeat_key(
            goal_key=rt.goal_key,
            tool_ref=rt.tool_ref,
            request=request,
            deterministic=rt.tool.is_deterministic(request),
        )

    def lookup(
        self,
        key: str,
        *,
        agent: str,
        deterministic: bool,
        resumed: AbstractSet[str] = frozenset(),
    ) -> RepeatOutcome:
        """
        Prefer an own result, then another same-run result, then a past-run hint.

        Parameters
        ----------
        key : str
            Repeat key to find.
        agent : str
            Team/agent name, without the goal prefix.
        deterministic : bool
            False always permits a new run.
        resumed : AbstractSet[str], default=frozenset()
            Own candidate ids actually recovered in this restarted round.

        Returns
        -------
        RepeatOutcome
            Decision and optional matching candidate or past-run hint.

        Examples
        --------
        >>> RepeatIndex("r").lookup("k", agent="t/a", deterministic=False).action
        'run'
        """
        if not deterministic:
            return RepeatOutcome(action="run")
        other = hint = None
        for candidate in self._entries.get(key, ()):
            if not self._counted(candidate):
                continue
            if candidate.run_id != self.run_id:
                if hint is None:
                    hint = candidate
            elif self._own(candidate, agent):
                return RepeatOutcome(
                    action="return_own" if candidate.id in resumed else "refuse",
                    candidate=candidate,
                )
            elif other is None:
                other = candidate
        if other is not None:
            return RepeatOutcome(action="link", candidate=other)
        return RepeatOutcome(action="run", hint=hint)

    def candidates_of(self, agent: str) -> list[Candidate]:
        """
        Return this run's own candidates once per id, in insertion order.

        Parameters
        ----------
        agent : str
            Team/agent name, without the goal prefix.

        Returns
        -------
        list[Candidate]
            Own results of any status, including partials and earlier rounds.

        Examples
        --------
        >>> RepeatIndex("r").candidates_of("t/a")
        []
        """
        seen: set[str] = set()
        result: list[Candidate] = []
        for candidate in self._candidates:
            if self._own(candidate, agent) and candidate.id not in seen:
                seen.add(candidate.id)
                result.append(candidate)
        return result

    def next_replicate(
        self,
        rt: "AgentRuntime",
        request: ToolRequest,
        *,
        requested: int | None = None,
    ) -> int:
        """
        Reserve an unused replicate by key, ignoring request.replicate.

        Parameters
        ----------
        rt : AgentRuntime
            Agent runtime with a nondeterministic tool and its replicate bound.
        request : ToolRequest
            Request whose canonical fields identify the replicate pool.
        requested : int or None, default=None
            Explicit caller-selected index, or None for the smallest available.

        Returns
        -------
        int
            Index held pending until add or release.

        Raises
        ------
        SoftError
            If an explicit index is invalid or used, or all indices are used.
        LimitReached
            If automatic exhaustion exceeds the refusal watchdog threshold.

        Examples
        --------
        >>> from types import SimpleNamespace
        >>> from swarm_arena import Goal
        >>> from unittest.mock import Mock
        >>> rt = SimpleNamespace(goal_key=Goal(id="g", namespace="n").key(),
        ...     tool_ref=Ref(path="swarm_arena:ScriptTool"),
        ...     tool=Mock(is_deterministic=lambda req: False),
        ...     spec=SimpleNamespace(replicates=2, name="a"),
        ...     team=SimpleNamespace(name="t"))
        >>> idx = RepeatIndex("r")
        >>> idx.next_replicate(rt, ToolRequest(replicate=99))
        0
        >>> idx.next_replicate(rt, ToolRequest())
        1
        """
        n = rt.spec.replicates
        if requested is not None and (requested < 0 or requested >= n):
            raise SoftError("replicate index out of range")
        agent = f"{rt.team.name}/{rt.spec.name}"
        matched: Candidate | None = None
        for index in range(n) if requested is None else (requested,):
            try:
                key = self.key(rt, request.model_copy(update={"replicate": index}))
            except Fenced:
                raise
            except Exception as error:
                raise SoftError(f"invalid tool request: {type(error).__name__}") from None
            own = next(
                (c for c in self._entries.get(key, ()) if self._own(c, agent) and self._counted(c)),
                None,
            )
            if own is None and key not in self._pending:
                self._pending.add(key)
                self._allocations[(id(rt), id(request), index)] = (rt, request, key)
                return index
            if requested is not None:
                raise SoftError("replicate index already used")
            if matched is None:
                matched = own
        rt.watchdog.record_refusal()
        rt.journal.append(
            EventDraft(
                type=EventType.WATCHDOG,
                goal_id=rt.goal_id,
                team=rt.team.name,
                agent=rt.spec.name,
                round=rt.round,
                payload={"refused": None if matched is None else matched.id},
            )
        )
        if rt.watchdog.reason is not None:
            raise LimitReached("max_refused_repeats", rt.watchdog.reason)
        raise SoftError(f"repeat refused: replicates used up ({n} of {n})")

    def release(self, rt: "AgentRuntime", request: ToolRequest, index: int) -> None:
        """
        Release a retained allocation without calling tool hooks.

        Parameters
        ----------
        rt : AgentRuntime
            Agent whose replicate was reserved.
        request : ToolRequest
            Exact request object used to reserve the replicate; it may have
            changed since allocation.
        index : int
            Reserved index; release is idempotent and preserves committed use.

        Examples
        --------
        >>> from types import SimpleNamespace
        >>> from swarm_arena import Goal
        >>> from unittest.mock import Mock
        >>> rt = SimpleNamespace(goal_key=Goal(id="g", namespace="n").key(),
        ...     tool_ref=Ref(path="swarm_arena:ScriptTool"),
        ...     tool=Mock(is_deterministic=lambda req: False),
        ...     spec=SimpleNamespace(replicates=1, name="a"),
        ...     team=SimpleNamespace(name="t"))
        >>> idx, req = RepeatIndex("r"), ToolRequest()
        >>> index = idx.next_replicate(rt, req)
        >>> idx.release(rt, req, index)
        >>> idx.next_replicate(rt, req)
        0
        """
        allocation = self._allocations.pop((id(rt), id(request), index), None)
        if allocation is not None:
            self._pending.discard(allocation[2])

    def _own(self, candidate: Candidate, agent: str) -> bool:
        """Match run and team/agent, rather than an agent name alone."""
        return candidate.run_id == self.run_id and f"{candidate.team}/{candidate.agent}" == agent

    @staticmethod
    def _counted(candidate: Candidate) -> bool:
        """Exclude partial results and retryable execution outcomes."""
        return candidate.output.status in COUNTED and not candidate.output.partial
