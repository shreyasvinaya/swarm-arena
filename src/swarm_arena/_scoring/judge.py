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
from swarm_arena._outcomes import _propagate_control
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
    >>> from unittest.mock import Mock
    >>> from swarm_arena import Goal
    >>> from swarm_arena.executor import Executor
    >>> from swarm_arena.scoring import JudgeContext
    >>> context = JudgeContext(goal=Goal(id='g', namespace='example'),
    ...     executor=Mock(spec=Executor), artifact_root='/artifacts', batch_id='b1')
    >>> context.judge_error('c1', 'non-finite key')
    >>> context.judge_error('c1', 'later reason')
    >>> (context.batch_id, context.errors)
    ('b1', {'c1': 'non-finite key'})
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
        The :class:`JudgeContext` example calls ``context.judge_error(...)`` twice
        and verifies that the first reason wins.
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
    >>> import asyncio
    >>> from collections.abc import Sequence
    >>> from unittest.mock import Mock
    >>> from swarm_arena import Goal, Score
    >>> from swarm_arena.board import Candidate
    >>> from swarm_arena.scoring import Judge, JudgeContext
    >>> from swarm_arena.tool import RunStatus, ToolOutput, ToolRequest
    >>> sorted(Judge.__abstractmethods__)
    ['judge']
    >>> class ExampleJudge(Judge):
    ...     name = 'example'
    ...     async def judge(self, goal: Goal, candidates: Sequence[Candidate],
    ...                     ctx: JudgeContext) -> list[Score]:
    ...         return [Score(key=(1.0,), candidate_id=c.id) for c in candidates]
    >>> judge = ExampleJudge()
    >>> judge.preflight()
    >>> len(judge.rubric_hash())
    64
    >>> (judge.panel_hash(), judge.anchor_count(), judge.providers(), judge.models())
    ('', 0, frozenset(), ())
    >>> goal = Goal(id='g', namespace='example')
    >>> candidate = Candidate(id='c1', run_id='r', goal_id=goal.id, team='t', agent='a',
    ...     round=1, request=ToolRequest(), output=ToolOutput(status=RunStatus.OK))
    >>> scores = asyncio.run(judge.judge(goal, [candidate], Mock(spec=JudgeContext)))
    >>> [(s.candidate_id, s.key) for s in scores]
    [('c1', (1.0,))]
    """

    name: ClassVar[str]

    def preflight(self) -> None:
        """
        Check readiness before scoring candidates.

        The default implementation performs no I/O or validation.

        Examples
        --------
        The :class:`Judge` example calls ``judge.preflight()`` without I/O.
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
        The :class:`Judge` example calls ``judge.rubric_hash()`` and checks its
        digest length.
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
        except BaseException as error:
            _propagate_control(error)
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
        The :class:`Judge` example calls ``judge.panel_hash()`` and gets ``''``.
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
        The :class:`Judge` example calls ``judge.anchor_count()`` and gets zero.
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
        The :class:`Judge` example calls ``judge.providers()`` and gets an empty set.
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
        The :class:`Judge` example calls ``judge.models()`` and gets an empty tuple.
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
        The :class:`Judge` example runs ``judge.judge(goal, candidates, context)``
        with ``asyncio.run`` and inspects the returned candidate scores.
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
        True for a nonempty tuple of finite int/float values, excluding bool.
        False for malformed or missing keys; validation bypasses never raise.

    Examples
    --------
    >>> from swarm_arena import Score
    >>> is_finite_score(Score.maximize(1.0))
    True
    """
    try:
        key = score.key
        return (
            isinstance(key, tuple)
            and all(
                isinstance(value, (int, float)) and not isinstance(value, bool) for value in key
            )
            and _finite_key(key)
        )
    except (TypeError, ValueError, OverflowError, AttributeError):
        return False


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
    >>> from collections.abc import Sequence
    >>> from swarm_arena import Goal, Score
    >>> from swarm_arena.board import Candidate
    >>> from swarm_arena.scoring import Judge, JudgeContext
    >>> from swarm_arena._model.ids import IdGen
    >>> class ExampleJudge(Judge):
    ...     name = 'example'
    ...     async def judge(self, goal: Goal, candidates: Sequence[Candidate],
    ...                     ctx: JudgeContext) -> list[Score]:
    ...         return [Score(key=(1.0,), candidate_id=c.id) for c in candidates]
    >>> raw = Score(key=(1.0,), candidate_id='c1')
    >>> [score] = stamp_scores([raw], judge=ExampleJudge(), judge_name='quality',
    ...                        ids=IdGen('r', 0), batch_id='b1')
    >>> (score.judge, score.candidate_id, score.batch_id, bool(score.id), raw.id)
    ('quality', 'c1', 'b1', True, '')
    """
    prepared = _prepare_score_stamps(scores, judge, judge_name, batch_id)
    return _stamp_score_ids(prepared, ids)


def _prepare_score_stamps(
    scores: Sequence[Score], judge: Judge, judge_name: str, batch_id: str
) -> list[Score]:
    """Validate complete Python payloads and prepare provenance without allocating ids."""
    validated = [
        Score.model_validate(score.model_dump(mode="python", warnings=False)) for score in scores
    ]
    rubric_hash = judge.rubric_hash()
    panel_hash: str | None = None
    prepared = []
    for score in validated:
        if not score.panel_hash and panel_hash is None:
            panel_hash = judge.panel_hash()
        prepared.append(
            Score.model_validate(
                {
                    **score.model_dump(mode="python"),
                    "judge": judge_name,
                    "rubric_hash": rubric_hash,
                    "panel_hash": score.panel_hash or panel_hash,
                    "batch_id": score.batch_id or batch_id,
                }
            )
        )
    return prepared


def _stamp_score_ids(scores: Sequence[Score], ids: IdGen) -> list[Score]:
    """Allocate loop-owned ids only after abandonable preparation has completed."""
    return [score.model_copy(update={"id": ids.next("s")}) for score in scores]
