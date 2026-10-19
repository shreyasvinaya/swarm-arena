"""Define human-controlled execution limits and budget defaults."""

from typing import ClassVar, Literal

from pydantic import Field

from swarm_arena._model.base import _SafeModel
from swarm_arena._model.tooling import Estimate


class Limits(_SafeModel):
    """
    Bound agent activity, resource spending, and execution time.

    Parameters
    ----------
    max_teams : int, default=4
        Maximum teams per goal, including parts.
    max_agents_per_team : int, default=4
        Maximum agents within one team.
    max_agents_per_goal : int, default=12
        Maximum agents assigned to one goal.
    max_concurrent_goals : int, default=4
        Maximum goals active at once.
    max_agents_total : int, default=32
        Maximum agents working concurrently across the run.
    max_agents_per_fingerprint : int, default=2
        Maximum agents retaining the same result fingerprint.
    max_rounds : int, default=3
        Maximum rounds and the default for plans without explicit rounds.
    max_turns_per_round : int, default=20
        Maximum manager turns per agent round.
    max_tool_runs_per_round : int, default=3
        Maximum tool runs per agent round.
    max_tokens : int or None, default=None
        Optional run token cap; LLM plans require a cap during preflight.
    max_usd : float or None, default=None
        Optional positive run dollar cap.
    default_run_reservation : Estimate, optional
        Fallback reservation of 100,000 tokens and 1 USD for unestimated runs.
    llm_tranche_usd : float, default=0.25
        Dollars reserved per LLM tranche.
    llm_tranche_tokens : int, default=25000
        Tokens reserved per LLM tranche.
    tool_timeout_s : float, default=3600.0
        Maximum duration of an individual tool run in seconds.
    round_timeout_s : float or None, default=None
        Round timeout, or four times the tool timeout when omitted.
    run_timeout_s : float, default=86400.0
        Maximum duration of the whole run in seconds.
    max_learnings_per_agent_round : int, default=3
        Learning publication cap; zero disables publication.
    max_board_reads_per_round : int, default=10
        Board read cap per agent round; zero disables reads.
    digest_tokens : int, default=1500
        Maximum tokens in the board digest.
    max_edit_bytes : int, default=262144
        Maximum size of each editable file submission.
    max_consecutive_failures : int, default=3
        Consecutive failure threshold.
    max_refused_repeats : int, default=3
        Repeated refusal threshold.
    watchdog_idle_rounds : int, default=2
        Idle round threshold for the watchdog.
    max_notes_bytes : int, default=16384
        Maximum size of an agent's notes.

    Examples
    --------
    >>> from swarm_arena import Limits
    >>> Limits(tool_timeout_s=100).effective_round_timeout_s()
    400.0
    """

    max_teams: int = Field(default=4, ge=1)
    max_agents_per_team: int = Field(default=4, ge=1)
    max_agents_per_goal: int = Field(default=12, ge=1)
    max_concurrent_goals: int = Field(default=4, ge=1)
    max_agents_total: int = Field(default=32, ge=1)
    max_agents_per_fingerprint: int = Field(default=2, ge=1)
    max_rounds: int = Field(default=3, ge=1)
    max_turns_per_round: int = Field(default=20, ge=1)
    max_tool_runs_per_round: int = Field(default=3, ge=1)
    max_tokens: int | None = Field(default=None, ge=1)
    max_usd: float | None = Field(default=None, gt=0)
    default_run_reservation: Estimate = Estimate(tokens=100_000, usd=1.0)
    llm_tranche_usd: float = Field(default=0.25, gt=0)
    llm_tranche_tokens: int = Field(default=25_000, ge=1)
    tool_timeout_s: float = Field(default=3600.0, gt=0)
    round_timeout_s: float | None = Field(default=None, gt=0)
    run_timeout_s: float = Field(default=86_400.0, gt=0)
    max_learnings_per_agent_round: int = Field(default=3, ge=0)
    max_board_reads_per_round: int = Field(default=10, ge=0)
    digest_tokens: int = Field(default=1500, ge=1)
    max_edit_bytes: int = Field(default=262_144, ge=1)
    max_consecutive_failures: int = Field(default=3, ge=1)
    max_refused_repeats: int = Field(default=3, ge=1)
    watchdog_idle_rounds: int = Field(default=2, ge=1)
    max_notes_bytes: int = Field(default=16_384, ge=1)

    def effective_round_timeout_s(self) -> float:
        """
        Return the explicit round timeout or four times the tool timeout.

        Returns
        -------
        float
            Effective round duration cap in seconds.

        Examples
        --------
        >>> from swarm_arena import Limits
        >>> Limits().effective_round_timeout_s()
        14400.0
        """
        return self.round_timeout_s if self.round_timeout_s is not None else 4 * self.tool_timeout_s

    _unknown_fields: ClassVar[Literal["show", "hide"]] = "show"
