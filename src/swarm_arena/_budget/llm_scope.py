"""Carry core LLM funding and request provenance through task-local contexts."""

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

from swarm_arena._budget.tranche import Tranche
from swarm_arena._journal.writer import JournalWriter
from swarm_arena._model.base import CallRole
from swarm_arena._model.catalog import ModelCatalog
from swarm_arena._model.settings import CachePolicy, TracingConfig
from swarm_arena._store.artifacts import ArtifactStore


@dataclass(frozen=True, kw_only=True)
class LLMScope:
    """
    Carry the funding, metadata and services for one component's LLM calls.

    Parameters
    ----------
    role : CallRole
        Manager, judge, planner, curator or tool.
    run_id : str
        Run identifier.
    goal_id : str
        Goal identifier.
    team : str
        Team name, empty when not applicable.
    agent : str
        Agent name, empty when not applicable.
    round : int
        Current round number.
    candidate_id : str or None
        Candidate for tool-internal calls, otherwise None.
    tranche : Tranche
        Holder's call funding.
    journal : JournalWriter
        Durable run journal.
    artifacts : ArtifactStore
        Content-addressed payload store.
    models : ModelCatalog
        Human-approved model aliases.
    tracing : TracingConfig
        Human tracing configuration.
    cache : CachePolicy or None
        Response-cache policy.
    force_no_cache_read : bool, default=False
        Disable cache reads for anchors and final judging.
    call_ids : list[str], optional
        Independently allocated list recording calls made within this scope.

    Examples
    --------
    >>> from unittest.mock import Mock
    >>> from swarm_arena._budget.ledger import Ledger
    >>> from swarm_arena._budget.amount import Amount
    >>> tranche = Tranche(Mock(spec=Ledger), "g/t/a", "g", size=Amount())
    >>> scope = LLMScope(role="manager", run_id="r", goal_id="g", team="t",
    ...     agent="a", round=1, candidate_id=None, tranche=tranche,
    ...     journal=Mock(spec=JournalWriter),
    ...     artifacts=Mock(spec=ArtifactStore), models=ModelCatalog({}), tracing=TracingConfig(),
    ...     cache=None)
    >>> with llm_scope(scope):
    ...     current_llm_scope() is scope
    True
    >>> current_llm_scope() is None
    True
    """

    role: CallRole
    run_id: str
    goal_id: str
    team: str
    agent: str
    round: int
    candidate_id: str | None
    tranche: Tranche
    journal: JournalWriter
    artifacts: ArtifactStore
    models: ModelCatalog
    tracing: TracingConfig
    cache: CachePolicy | None
    force_no_cache_read: bool = False
    call_ids: list[str] = field(default_factory=list)


_scope: ContextVar[LLMScope | None] = ContextVar("llm_scope", default=None)
_requesting_call: ContextVar[str | None] = ContextVar("requesting_call", default=None)


def current_llm_scope() -> LLMScope | None:
    """
    Return the current task's component scope.

    Returns
    -------
    LLMScope or None
        Innermost active scope, or None outside a component call.

    Examples
    --------
    >>> current_llm_scope() is None
    True
    """
    return _scope.get()


@contextmanager
def llm_scope(scope: LLMScope) -> Generator[None]:
    """
    Enter a component scope and restore its predecessor on every exit.

    Parameters
    ----------
    scope : LLMScope
        Funding and metadata inherited by child tasks.

    Yields
    ------
    None
        The scope remains active while inside the context.

    Examples
    --------
    The :class:`LLMScope` example constructs a scope and enters this context
    without performing I/O.
    """
    token = _scope.set(scope)
    try:
        yield
    finally:
        _scope.reset(token)


@contextmanager
def requesting_call(call_id: str | None) -> Generator[None]:
    """
    Associate actions in this task with the LLM call that requested them.

    Parameters
    ----------
    call_id : str or None
        Requesting call's identifier, or None to clear it temporarily.

    Yields
    ------
    None
        The identifier remains active while inside the context.

    Examples
    --------
    >>> with requesting_call("llm-1"):
    ...     current_requesting_call()
    'llm-1'
    >>> current_requesting_call() is None
    True
    """
    token = _requesting_call.set(call_id)
    try:
        yield
    finally:
        _requesting_call.reset(token)


def current_requesting_call() -> str | None:
    """
    Return the requesting LLM call's identifier for the current task.

    Returns
    -------
    str or None
        Innermost requesting call, or None outside its action context.

    Examples
    --------
    >>> with requesting_call("llm-2"):
    ...     current_requesting_call()
    'llm-2'
    """
    return _requesting_call.get()
