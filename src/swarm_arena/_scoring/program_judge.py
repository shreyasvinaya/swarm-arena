"""Score candidates using trusted synchronous Python functions."""

import inspect
from collections.abc import Awaitable, Callable, Sequence
from functools import partial
from typing import ClassVar, cast

from swarm_arena._errors import _exception_type_name
from swarm_arena._exec.threads import run_sync_daemon
from swarm_arena._model.detach import _detach_model
from swarm_arena._model.goal import Goal
from swarm_arena._model.ref import _ConfigInputModel, import_path, load_object
from swarm_arena._model.results import Candidate, Score
from swarm_arena._outcomes import _Outcome, _propagate_control, _raise_terminal
from swarm_arena._scoring.judge import Judge, JudgeContext, is_finite_score


def _is_async_callable(fn: object) -> bool:
    """Recognize asynchronous functions, methods, and partial callable objects."""
    while isinstance(fn, partial):
        fn = fn.func
    return inspect.iscoroutinefunction(fn) or (
        callable(fn) and inspect.iscoroutinefunction(fn.__call__)
    )


def _score_copy(
    fn: Callable[[Goal, Candidate], Score], goal: Goal, candidate: Candidate
) -> tuple[Score | None, str | None]:
    """Evaluate and detach all returned-model operations in the same worker."""
    result = fn(goal, candidate.model_copy(deep=True))
    if inspect.isawaitable(result):
        _discard_awaitable(result)
        return None, "ProgramJudge needs a sync function"
    if not isinstance(result, Score):
        return None, f"returned {type(result).__name__}, expected Score"
    if not is_finite_score(result):
        return None, "score key must be nonempty and finite"
    try:
        score = _detach_model(Score, result)
    except KeyboardInterrupt:
        raise
    except BaseException as error:
        _propagate_control(error)
        return None, "invalid Score returned"
    return score.model_copy(update={"candidate_id": candidate.id}), None


def _discard_awaitable(result: object) -> None:
    """Close or cancel an awaitable a sync scoring function returned, so it never warns."""
    close = getattr(result, "close", None)
    if callable(close):
        close()
        return
    cancel = getattr(result, "cancel", None)
    if callable(cancel):
        cancel()
        return
    iterator = cast(Awaitable[object], result).__await__()
    close = getattr(iterator, "close", None)
    if callable(close):
        close()


class ProgramJudge(Judge):
    """
    Score candidates with an importable synchronous Python function.

    Functions run in daemon threads in the driver process with no time limit.
    Cancellation stops waiting promptly; the abandoned function keeps running.
    Each call receives a deep copy of its candidate, made in that thread, so a
    large output never stalls the event loop. The function sees the caller's
    context variables (a copy), but ``anyio.from_thread`` is not available to
    it. Exceptions (``<Type>: scoring function raised``), incorrect return
    types, invalid keys, and a returned Score that fails validation again
    (``invalid Score returned``) become per-candidate judge errors. Exception
    messages are omitted because they can contain candidate or configuration data.
    Construction stores the import path; preflight or first judging loads it.

    Parameters
    ----------
    function : str or Callable[[Goal, Candidate], Score] or ProgramJudge.Config
        Module-level function, import path, or validated configuration.

    Examples
    --------
    >>> from swarm_arena import ProgramJudge
    >>> judge = ProgramJudge('my_judges:quality')
    >>> judge.config.function
    'my_judges:quality'
    """

    name: ClassVar[str] = "program"

    class Config(_ConfigInputModel):
        """
        Store the import path of a scoring function.

        Parameters
        ----------
        function : str
            Import path of a synchronous ``(Goal, Candidate) -> Score`` function.

        Examples
        --------
        >>> from swarm_arena import ProgramJudge
        >>> ProgramJudge.Config(function='my_judges:quality').function
        'my_judges:quality'
        """

        function: str

    def __init__(self, function: str | Callable[[Goal, Candidate], Score] | Config, /) -> None:
        """Store validated settings without loading the referenced function."""
        if _is_async_callable(function):
            raise TypeError("ProgramJudge needs a sync function")
        if inspect.ismethod(function) and not isinstance(function.__self__, type):
            raise TypeError(
                "ProgramJudge needs an importable function, not a bound instance method"
            )
        data = (
            function
            if isinstance(function, self.Config)
            else {"function": function if isinstance(function, str) else import_path(function)}
        )
        super().__init__(data)
        self._function: Callable[[Goal, Candidate], Score] | None = None

    @property
    def config(self) -> Config:
        """
        Return the validated function reference.

        Returns
        -------
        ProgramJudge.Config
            Immutable configuration for this judge.

        Examples
        --------
        >>> from swarm_arena import ProgramJudge
        >>> ProgramJudge('my_judges:quality').config.function
        'my_judges:quality'
        """
        return cast(ProgramJudge.Config, self._config)

    def preflight(self) -> None:
        """
        Load the scoring function and reject asynchronous or noncallable objects.

        Raises
        ------
        ValueError
            If the referenced object cannot be imported.
        TypeError
            If the object is not a synchronous callable.

        Examples
        --------
        An in-memory module stands in for your own ``my_judges.py``:

        >>> import sys, types
        >>> from swarm_arena import ProgramJudge, Score
        >>> def quality(goal, candidate):
        ...     return Score.maximize(1.0)
        >>> module = types.ModuleType('preflight_demo')
        >>> module.quality = quality
        >>> sys.modules['preflight_demo'] = module
        >>> ProgramJudge('preflight_demo:quality').preflight()
        >>> ProgramJudge('preflight_demo:missing').preflight()
        Traceback (most recent call last):
        ...
        ValueError: cannot import preflight_demo:missing
        >>> del sys.modules['preflight_demo']
        """
        self._load_function()

    def _load_function(self) -> Callable[[Goal, Candidate], Score]:
        """Load and cache the callable without exposing import exception values."""
        if self._function is None:
            fn = load_object(self.config.function)
            if not callable(fn) or _is_async_callable(fn):
                raise TypeError("ProgramJudge needs a sync function")
            self._function = cast(Callable[[Goal, Candidate], Score], fn)
        return self._function

    async def judge(
        self, goal: Goal, candidates: Sequence[Candidate], ctx: JudgeContext
    ) -> list[Score]:
        """
        Score isolated candidate copies in daemon threads and retain failures.

        There is no scoring time limit. Cancellation abandons the running call.
        Registered TERMINAL/FENCED failures propagate before candidate errors.

        Parameters
        ----------
        goal : Goal
            Goal passed to the scoring function.
        candidates : Sequence[Candidate]
            Candidates to score in input order.
        ctx : JudgeContext
            Context receiving sanitized errors for candidates without scores.

        Returns
        -------
        list[Score]
            Successful scores with their candidate ids assigned.

        Raises
        ------
        ValueError
            If the scoring function cannot be imported.
        TypeError
            If the referenced object is not synchronous and callable.

        Examples
        --------
        An in-memory module stands in for your own ``my_judges.py``:

        >>> import sys, types
        >>> import anyio
        >>> from swarm_arena import Goal, ProgramJudge, Score
        >>> from swarm_arena.board import Candidate
        >>> from swarm_arena.executor import LocalExecutor
        >>> from swarm_arena.scoring import JudgeContext
        >>> from swarm_arena.tool import RunStatus, ToolOutput, ToolRequest
        >>> def quality(goal, candidate):
        ...     return Score.maximize(candidate.output.data['accuracy'])
        >>> module = types.ModuleType('judge_demo')
        >>> module.quality = quality
        >>> sys.modules['judge_demo'] = module
        >>> goal = Goal(id='g', namespace='demo')
        >>> candidate = Candidate(
        ...     id='c1', run_id='r', goal_id='g', team='t', agent='a', round=1,
        ...     request=ToolRequest(),
        ...     output=ToolOutput(status=RunStatus.OK, data={'accuracy': 0.9}))
        >>> ctx = JudgeContext(goal=goal, executor=LocalExecutor(), artifact_root='unused')
        >>> judge = ProgramJudge('judge_demo:quality')
        >>> [(s.candidate_id, s.key) for s in anyio.run(judge.judge, goal, [candidate], ctx)]
        [('c1', (0.9,))]
        >>> del sys.modules['judge_demo']
        """
        fn = await run_sync_daemon(self._load_function)
        scores: list[Score] = []
        for candidate in candidates:
            outcome = _Outcome()
            try:
                score, reason = await run_sync_daemon(_score_copy, fn, goal, candidate)
            except KeyboardInterrupt:
                raise
            except BaseException as error:
                name = _exception_type_name(error)
                error = outcome.classify(error)
                _raise_terminal(error)
                ctx.judge_error(candidate.id, f"{name}: scoring function raised")
                continue
            if reason is not None:
                ctx.judge_error(candidate.id, reason)
            elif score is not None:
                scores.append(score)
        return scores
