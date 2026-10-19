"""Define scored candidates, learnings, and recorded model calls."""

import math
import re
from enum import StrEnum
from typing import Any, ClassVar, Literal, Self

from pydantic import (
    field_validator,
    model_validator,
)

from swarm_arena._model.base import CallRole, FrozenModel, StoredModel
from swarm_arena._model.tooling import ToolOutput, ToolRequest, Usage
from swarm_arena._tooling.paths import check_key


def _finite_key(key: tuple[float, ...]) -> bool:
    """Return whether a ranking key is nonempty and contains only finite values."""
    return bool(key) and all(math.isfinite(value) for value in key)


class CheckResult(FrozenModel):
    """
    Record whether a candidate passes a named check.

    Parameters
    ----------
    check : str
        Check name.
    passed : bool
        Whether the check passed.
    reasons : list[str], default=[]
        Feedback explaining the outcome.
    details : dict[str, Any], default={}
        Structured check details.

    Examples
    --------
    >>> from swarm_arena.scoring import CheckResult
    >>> CheckResult(check="valid", passed=True).passed
    True
    """

    check: str
    passed: bool
    reasons: list[str] = []
    details: dict[str, Any] = {}


class Score(FrozenModel):
    """
    Rank a candidate using a nonempty finite lexicographic key.

    Larger keys are better. The core stamps identity and provenance fields.

    Parameters
    ----------
    id : str, default=""
        Score identifier.
    candidate_id : str, default=""
        Scored candidate identifier.
    judge : str, default=""
        Judge name.
    rubric_hash : str, default=""
        Rubric identity.
    panel_hash : str, default=""
        Panel membership identity.
    key : tuple[float, ...]
        Nonempty ranking key containing only finite values.
    value : float or None, default=None
        Human-facing value, before ranking-direction conversion.
    verdict : str or None, default=None
        Optional verdict.
    notes : str, default=""
        Judge feedback.
    batch_id : str, default=""
        Judging batch identifier.
    llm_call_ids : list[str], default=[]
        Calls that produced the score.

    Examples
    --------
    >>> from swarm_arena import Score
    >>> Score.minimize(0.25).key
    (-0.25,)
    """

    id: str = ""
    candidate_id: str = ""
    judge: str = ""
    rubric_hash: str = ""
    panel_hash: str = ""
    key: tuple[float, ...]
    value: float | None = None
    verdict: str | None = None
    notes: str = ""
    batch_id: str = ""
    llm_call_ids: list[str] = []

    @field_validator("key")
    @classmethod
    def _check_key(cls, key: tuple[float, ...]) -> tuple[float, ...]:
        """Validate the shared finite-key contract."""
        if not key:
            raise ValueError("key needs at least one value")
        if not _finite_key(key):
            raise ValueError("score key values must be finite")
        return key

    @classmethod
    def minimize(cls, x: float, *, verdict: str | None = None, notes: str = "") -> Self:
        """
        Create a score that prefers smaller values.

        Parameters
        ----------
        x : float
            Finite human-facing value.
        verdict : str or None, default=None
            Optional verdict.
        notes : str, default=""
            Judge feedback.

        Returns
        -------
        Score
            Score with key ``(-x,)`` and value ``x``.

        Examples
        --------
        >>> from swarm_arena import Score
        >>> Score.minimize(2.0).value
        2.0
        """
        return cls(key=(-x,), value=x, verdict=verdict, notes=notes)

    @classmethod
    def maximize(cls, x: float, *, verdict: str | None = None, notes: str = "") -> Self:
        """
        Create a score that prefers larger values.

        Parameters
        ----------
        x : float
            Finite human-facing value.
        verdict : str or None, default=None
            Optional verdict.
        notes : str, default=""
            Judge feedback.

        Returns
        -------
        Score
            Score with key ``(x,)`` and value ``x``.

        Examples
        --------
        >>> from swarm_arena import Score
        >>> Score.maximize(2.0).key
        (2.0,)
        """
        return cls(key=(x,), value=x, verdict=verdict, notes=notes)

    @property
    def identity(self) -> tuple[str, str, str]:
        """
        Return the judge, rubric, and panel identity.

        Returns
        -------
        tuple[str, str, str]
            Identity required when comparing scores.

        Examples
        --------
        >>> from swarm_arena import Score
        >>> Score(key=(1.0,), judge="quality").identity
        ('quality', '', '')
        """
        return self.judge, self.rubric_hash, self.panel_hash


class AgentStatus(StrEnum):
    """
    Describe an agent's status within a round.

    Examples
    --------
    >>> from swarm_arena.manager import AgentStatus
    >>> AgentStatus.ACTIVE.value
    'active'
    """

    ACTIVE = "active"
    FINISHED = "finished"
    STARVED = "starved"
    PRUNED = "pruned"
    STOPPED = "stopped"
    WATCHDOG_STOPPED = "watchdog_stopped"


class Candidate(StoredModel):
    """
    Store a tool result and its provenance, checks, and effective file overlay.

    Parameters
    ----------
    schema_version : int, default=1
        Stored-format version.
    id : str
        Candidate identifier.
    run_id : str
        Producing run identifier.
    goal_id : str
        Goal identifier.
    part : str or None, default=None
        Goal part used for ranking groups.
    team : str
        Producing team.
    agent : str
        Producing agent.
    round : int
        Producing round.
    fidelity : str, default=""
        Effort-level label copied from the round.
    tool_identity : str, default=""
        Producing tool identity; empty identities never match for overlays.
    request : ToolRequest
        Tool invocation.
    output : ToolOutput
        Execution outcome and result.
    checks : list[CheckResult], default=[]
        Check outcomes.
    parent_ids : list[str], default=[]
        Parent candidate identifiers.
    overlay : dict[str, str], default={}
        Normalized workspace paths mapped to lowercase SHA-256 digests.
    read_ids : list[str], default=[]
        Results and learnings read by the manager.
    requested_by_call : str or None, default=None
        Manager call that requested the invocation.
    llm_call_ids : list[str], default=[]
        Calls made inside the tool run.
    llm_calls_source : str, default="none"
        Call provenance: gateway, reported, langfuse, none, or unknown.
    origin : str, default="agent"
        Agent or human origin.
    warnings : list[str], default=[]
        Soft staging warnings and past-run hints.

    Examples
    --------
    >>> from swarm_arena.board import Candidate
    >>> from swarm_arena.tool import RunStatus, ToolOutput, ToolRequest
    >>> c = Candidate(id="c1", run_id="r", goal_id="g", team="t", agent="a",
    ...     round=1, request=ToolRequest(), output=ToolOutput(status=RunStatus.OK))
    >>> c.gated_out
    False
    """

    schema_kind: ClassVar[str] = "candidate"
    id: str
    run_id: str
    goal_id: str
    part: str | None = None
    team: str
    agent: str
    round: int
    fidelity: str = ""
    tool_identity: str = ""
    request: ToolRequest
    output: ToolOutput
    checks: list[CheckResult] = []
    parent_ids: list[str] = []
    overlay: dict[str, str] = {}
    read_ids: list[str] = []
    requested_by_call: str | None = None
    llm_call_ids: list[str] = []
    llm_calls_source: Literal["gateway", "reported", "langfuse", "none", "unknown"] = "none"
    origin: Literal["agent", "human"] = "agent"
    warnings: list[str] = []

    @field_validator("overlay")
    @classmethod
    def _check_overlay(cls, overlay: dict[str, str]) -> dict[str, str]:
        """Require canonical paths and lowercase SHA-256 digests."""
        for path, digest in overlay.items():
            try:
                check_key(path)
            except ValueError:
                raise ValueError("overlay keys must be normalized workspace paths") from None
            if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise ValueError("overlay values must be 64 lowercase hex characters")
        return overlay

    @property
    def gated_out(self) -> bool:
        """
        Return whether any check failed without storing a derived field.

        Returns
        -------
        bool
            Whether at least one check failed.

        Examples
        --------
        >>> from swarm_arena.board import Candidate
        >>> from swarm_arena.scoring import CheckResult
        >>> from swarm_arena.tool import RunStatus, ToolOutput, ToolRequest
        >>> c = Candidate(id="c1", run_id="r", goal_id="g", team="t", agent="a",
        ...     round=1, request=ToolRequest(), output=ToolOutput(status=RunStatus.OK),
        ...     checks=[CheckResult(check="valid", passed=False)])
        >>> c.gated_out
        True
        """
        return any(not check.passed for check in self.checks)


class Learning(StoredModel):
    """
    Store a short, evidence-backed learning for a problem namespace.

    Parameters
    ----------
    schema_version : int, default=1
        Stored-format version.
    id : str
        Learning identifier.
    namespace : str
        Problem family.
    run_id : str
        Producing run identifier.
    goal_id : str
        Goal identifier.
    round : int
        Producing round.
    author : str
        Team/agent address or human-prefixed author.
    title : str
        Stripped, nonempty single line of at most 120 characters.
    text : str
        Markdown body of at most 1500 characters.
    evidence : list[str], default=[]
        Candidate identifiers; required for non-human authors.
    tags : list[str], default=[]
        Search tags.
    verified : bool, default=False
        Whether complete evidence passed its checks.
    pinned : bool, default=False
        Human-selected pinned status.
    hidden : bool, default=False
        Human-selected hidden status.
    refines : list[str], default=[]
        Learnings refined by this learning.
    supersedes : list[str], default=[]
        Learnings replaced by this learning.
    posted_by_call : str or None, default=None
        Call that posted the learning.

    Examples
    --------
    >>> from swarm_arena.board import Learning
    >>> learning = Learning(id="l1", namespace="n", run_id="r", goal_id="g",
    ...     round=1, author="human:alice", title="Use the baseline", text="Start here.")
    >>> learning.from_human
    True
    """

    schema_kind: ClassVar[str] = "learning"
    id: str
    namespace: str
    run_id: str
    goal_id: str
    round: int
    author: str
    title: str
    text: str
    evidence: list[str] = []
    tags: list[str] = []
    verified: bool = False
    pinned: bool = False
    hidden: bool = False
    refines: list[str] = []
    supersedes: list[str] = []
    posted_by_call: str | None = None

    @field_validator("title")
    @classmethod
    def _check_title(cls, title: str) -> str:
        """Strip surrounding whitespace and enforce the single-line title limit."""
        stripped = title.strip()
        if not stripped or len(stripped) > 120 or "\n" in title or "\r" in title:
            raise ValueError("title must be nonempty, single-line, and at most 120 characters")
        return stripped

    @field_validator("text")
    @classmethod
    def _check_text(cls, text: str) -> str:
        """Enforce the learning body size limit."""
        if len(text) > 1500:
            raise ValueError("text must be at most 1500 characters")
        return text

    @model_validator(mode="after")
    def _check_evidence(self) -> Self:
        """Require candidate evidence for non-human authors."""
        if not self.from_human and not self.evidence:
            raise ValueError("evidence needs at least one id unless the author is human")
        return self

    @property
    def from_human(self) -> bool:
        """
        Return whether the author has the human prefix.

        Returns
        -------
        bool
            Whether the author starts with ``human:``.

        Examples
        --------
        >>> from swarm_arena.board import Learning
        >>> Learning(id="l", namespace="n", run_id="r", goal_id="g", round=1,
        ...     author="human:alice", title="A hint", text="Try this.").from_human
        True
        """
        return self.author.startswith("human:")


class LLMCall(StoredModel):
    """
    Store model-call usage, cost, provenance, and optional payload references.

    Parameters
    ----------
    schema_version : int, default=1
        Stored-format version.
    id : str
        Call identifier.
    run_id : str
        Run identifier.
    goal_id : str
        Goal identifier.
    team : str, default=""
        Team address.
    agent : str, default=""
        Agent address.
    round : int, default=0
        Round number.
    role : str
        Manager, judge, planner, curator, or tool role.
    candidate_id : str or None, default=None
        Candidate for tool-internal calls.
    model_alias : str or None, default=None
        Model catalog alias.
    model : str
        Model identifier.
    provider : str
        Provider identifier.
    request_ref : str or None, default=None
        Request payload hash, if stored.
    response_ref : str or None, default=None
        Response payload hash, if stored.
    tool_calls : list[dict[str, str]], default=[]
        Tool call identifiers and names.
    usage : Usage, default=Usage()
        Token consumption and reported cost.
    cache_hit : bool, default=False
        Whether the response cache supplied this call.
    cached_from_call_id : str or None, default=None
        Original call for a cached response.
    reservation_id : str or None, default=None
        Budget reservation identifier.
    cost_usd : float, default=0.0
        Recorded call cost.
    latency_ms : float or None, default=None
        Call latency in milliseconds.
    status : str, default="ok"
        Call outcome, ok or error.
    error : str or None, default=None
        Sanitized error description.
    trace_url : str or None, default=None
        Trace URL.
    parent_call_id : str or None, default=None
        Call that caused this call.
    litellm_call_id : str or None, default=None
        Gateway deduplication identifier.

    Examples
    --------
    >>> from swarm_arena.board import LLMCall
    >>> call = LLMCall(id="llm1", run_id="r", goal_id="g", role="judge",
    ...     model="local/model", provider="local")
    >>> call.cost_usd
    0.0
    """

    schema_kind: ClassVar[str] = "llm_call"
    id: str
    run_id: str
    goal_id: str
    team: str = ""
    agent: str = ""
    round: int = 0
    role: CallRole
    candidate_id: str | None = None
    model_alias: str | None = None
    model: str
    provider: str
    request_ref: str | None = None
    response_ref: str | None = None
    tool_calls: list[dict[str, str]] = []
    usage: Usage = Usage()
    cache_hit: bool = False
    cached_from_call_id: str | None = None
    reservation_id: str | None = None
    cost_usd: float = 0.0
    latency_ms: float | None = None
    status: Literal["ok", "error"] = "ok"
    error: str | None = None
    trace_url: str | None = None
    parent_call_id: str | None = None
    litellm_call_id: str | None = None
