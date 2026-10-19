"""Define judge contexts, scoring identity, and core score stamping."""

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Sequence
from typing import ClassVar

from swarm_arena._exec.executor import Executor, Job
from swarm_arena._model.canonical import canonical_json, sha256_hex
from swarm_arena._model.goal import Goal
from swarm_arena._model.ids import IdGen
from swarm_arena._model.ref import Configurable
from swarm_arena._model.results import Candidate, Score, _finite_key
from swarm_arena._scoring.check import CheckContext


class JudgeContext(CheckContext):
    """
    Execute judging commands and retain per-candidate batch errors.

    Parameters
    ----------
    goal : Goal
        Goal being judged.
    executor : Executor
        Executor receiving command jobs.
    artifact_root : str
        Root holding content-addressed workspace artifacts.
    run_id : str, default='local'
        Run identity for environment variables and default ids.
    round : int, default=0
        Current round number.
    part : str or None, default=None
        Part being judged, if any.
    fidelity : str, default=''
        Fidelity of this judging batch.
    batch_id : str, default=''
        Judging batch identifier.
    anchors : frozenset[str], optional
        Candidate ids used as anchors; defaults to an empty set.
    final : bool, default=False
        Whether this is final judging.
    read_cache : bool, default=True
        Whether the judge may read cached results.
    on_started : Callable[[Job.Started], Awaitable[None]] or None, default=None
        Callback placed on each command job before execution.
    secrets_file : str or None, default=None
        Executor-local file resolving allowlisted environment names.
    ids : IdGen or None, default=None
        Shared run id source; None creates ``IdGen(run_id, 0)``.
    tool_timeout_s : float, default=3600.0
        Human tool-time limit applied to every command.

    Attributes
    ----------
    errors : dict[str, str]
        First error reason recorded for each candidate in this context.

    Examples
    --------
    >>> from swarm_arena.scoring import JudgeContext
    >>> context = JudgeContext(goal=goal, executor=executor,  # doctest: +SKIP
    ...                        artifact_root='/artifacts', batch_id='b1')
    >>> context.judge_error('c1', 'non-finite key')  # doctest: +SKIP
    """

    def __init__(
        self,
        *,
        goal: Goal,
        executor: Executor,
        artifact_root: str,
        run_id: str = "local",
        round: int = 0,
        part: str | None = None,
        fidelity: str = "",
        batch_id: str = "",
        anchors: frozenset[str] = frozenset(),
        final: bool = False,
        read_cache: bool = True,
        on_started: Callable[[Job.Started], Awaitable[None]] | None = None,
        secrets_file: str | None = None,
        ids: IdGen | None = None,
        tool_timeout_s: float = 3600.0,
    ) -> None:
        """Retain batch settings and initialize an independent error channel."""
        super().__init__(
            goal=goal,
            executor=executor,
            artifact_root=artifact_root,
            run_id=run_id,
            round=round,
            part=part,
            on_started=on_started,
            secrets_file=secrets_file,
            ids=ids,
            tool_timeout_s=tool_timeout_s,
        )
        self.fidelity = fidelity
        self.batch_id = batch_id
        self.anchors = anchors
        self.final = final
        self.read_cache = read_cache
        self.errors: dict[str, str] = {}

    def judge_error(self, candidate_id: str, reason: str) -> None:
        """
        Record a candidate's first judging error.

        Parameters
        ----------
        candidate_id : str
            Candidate that could not be judged.
        reason : str
            Sanitized diagnostic supplied by the judge, without secret values.

        Examples
        --------
        >>> context.judge_error('c1', 'non-finite key')  # doctest: +SKIP
        """
        self.errors.setdefault(candidate_id, reason)


class Judge(Configurable, ABC):
    """
    Score candidates with a reproducible rubric and optional panel identity.

    Parameters
    ----------
    config : BaseModel or Mapping[str, Any] or None, optional
        Settings validated by the subclass's nested ``Config`` model.
        None selects that model's defaults.

    Attributes
    ----------
    name : ClassVar[str]
        Judge name.

    Examples
    --------
    >>> from swarm_arena.scoring import Judge
    >>> sorted(Judge.__abstractmethods__)
    ['judge']
    """

    name: ClassVar[str]

    def preflight(self) -> None:
        """
        Check readiness before scoring candidates.

        The default implementation performs no I/O or validation.

        Examples
        --------
        >>> judge.preflight()  # doctest: +SKIP
        """

    def rubric_hash(self) -> str:
        """
        Hash the judge's import path and canonical configuration.

        Returns
        -------
        str
            SHA-256 digest identifying the scoring rubric.

        Raises
        ------
        ValueError
            If the configuration cannot be serialized as canonical JSON.

        Examples
        --------
        >>> digest = judge.rubric_hash()  # doctest: +SKIP
        """
        cls = type(self)
        try:
            return sha256_hex(
                canonical_json(
                    {
                        "path": f"{cls.__module__}:{cls.__qualname__}",
                        "config": self.config.model_dump(mode="json"),
                    }
                )
            )
        except Exception:
            raise ValueError("cannot encode judge rubric") from None

    def panel_hash(self) -> str:
        """
        Return the default panel membership identity.

        Returns
        -------
        str
            Empty string by default; a score's nonempty panel hash wins.

        Examples
        --------
        >>> panel = judge.panel_hash()  # doctest: +SKIP
        """
        return ""

    def anchor_count(self) -> int:
        """
        Return the required number of anchor candidates.

        Returns
        -------
        int
            Zero by default.

        Examples
        --------
        >>> anchors = judge.anchor_count()  # doctest: +SKIP
        """
        return 0

    def providers(self) -> frozenset[str]:
        """
        Return the language-model providers called by the judge.

        Returns
        -------
        frozenset[str]
            No providers by default.

        Examples
        --------
        >>> providers = judge.providers()  # doctest: +SKIP
        """
        return frozenset()

    def models(self) -> tuple[str, ...]:
        """
        Return the model catalog aliases called by the judge.

        Returns
        -------
        tuple[str, ...]
            No model aliases by default.

        Examples
        --------
        >>> models = judge.models()  # doctest: +SKIP
        """
        return ()

    @abstractmethod
    async def judge(
        self, goal: Goal, candidates: Sequence[Candidate], ctx: JudgeContext
    ) -> list[Score]:
        """
        Score a batch and record failures for individual candidates.

        Parameters
        ----------
        goal : Goal
            Goal defining the scoring problem.
        candidates : Sequence[Candidate]
            Candidates to judge.
        ctx : JudgeContext
            Batch state, command execution, and error channel.

        Returns
        -------
        list[Score]
            Raw scores with candidate ids set, ready for core stamping.

        Examples
        --------
        >>> scores = await judge.judge(goal, candidates, context)  # doctest: +SKIP
        """


def is_finite_score(score: Score) -> bool:
    """
    Check a score's key even when model validation was bypassed.

    Parameters
    ----------
    score : Score
        Score constructed normally or with ``model_construct`` or ``model_copy``.

    Returns
    -------
    bool
        True for a nonempty key containing only finite numbers.

    Examples
    --------
    >>> from swarm_arena import Score
    >>> is_finite_score(Score.maximize(1.0))
    True
    """
    return _finite_key(score.key)


def stamp_scores(
    scores: Sequence[Score], *, judge: Judge, judge_name: str, ids: IdGen, batch_id: str
) -> list[Score]:
    """
    Copy raw scores with core ids and judging provenance.

    Parameters
    ----------
    scores : Sequence[Score]
        Scores to stamp without modifying the originals.
    judge : Judge
        Judge supplying rubric and default panel identity.
    judge_name : str
        Configured judge name to stamp on every score.
    ids : IdGen
        Shared run id source, also used by the context's command jobs.
    batch_id : str
        Batch id used when a score has no batch id of its own.

    Returns
    -------
    list[Score]
        Copies with new ids, judge names, rubric hashes, and effective panel and
        batch ids; candidate ids and scoring payloads are preserved.

    Examples
    --------
    >>> stamped = stamp_scores(  # doctest: +SKIP
    ...     scores, judge=judge, judge_name='quality', ids=context.ids, batch_id='b1')
    """
    return [
        score.model_copy(
            update={
                "id": ids.next("s"),
                "judge": judge_name,
                "rubric_hash": judge.rubric_hash(),
                "panel_hash": score.panel_hash or judge.panel_hash(),
                "batch_id": score.batch_id or batch_id,
            }
        )
        for score in scores
    ]
