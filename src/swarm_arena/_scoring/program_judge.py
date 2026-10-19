"""Score candidates using trusted synchronous Python functions."""

import inspect
from collections.abc import Callable, Sequence
from typing import ClassVar, cast

import anyio

from swarm_arena._model.goal import Goal
from swarm_arena._model.ref import _ConfigInputModel, _validate_config, import_path, load_object
from swarm_arena._model.results import Candidate, Score
from swarm_arena._scoring.judge import Judge, JudgeContext, is_finite_score


class ProgramJudge(Judge):
    """
    Score candidates with an importable synchronous Python function.

    Functions run in worker threads in the driver process. Exceptions, incorrect
    return types, and invalid keys become per-candidate judge errors. Exception
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
        data = (
            function
            if isinstance(function, self.Config)
            else {"function": function if isinstance(function, str) else import_path(function)}
        )
        self._config = _validate_config(self.Config, data, where="ProgramJudge")
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
        >>> from swarm_arena import ProgramJudge
        >>> ProgramJudge('builtins:max').preflight()
        """
        self._load_function()

    def _load_function(self) -> Callable[[Goal, Candidate], Score]:
        """Load and cache the callable without exposing import exception values."""
        if self._function is None:
            try:
                fn = load_object(self.config.function)
            except Exception:
                raise ValueError(f"cannot import {self.config.function}") from None
            if not callable(fn) or inspect.iscoroutinefunction(fn):
                raise TypeError("ProgramJudge needs a sync function")
            self._function = cast(Callable[[Goal, Candidate], Score], fn)
        return self._function

    async def judge(
        self, goal: Goal, candidates: Sequence[Candidate], ctx: JudgeContext
    ) -> list[Score]:
        """
        Score each candidate in a worker thread and retain individual failures.

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

        Examples
        --------
        >>> import anyio
        >>> from swarm_arena import Goal, ProgramJudge
        >>> from swarm_arena.executor import LocalExecutor
        >>> from swarm_arena.scoring import JudgeContext
        >>> goal = Goal(id='g', namespace='demo')
        >>> ctx = JudgeContext(goal=goal, executor=LocalExecutor(), artifact_root='unused')
        >>> anyio.run(ProgramJudge('builtins:max').judge, goal, [], ctx)
        []
        """
        fn = self._load_function()
        scores: list[Score] = []
        for candidate in candidates:
            try:
                result = await anyio.to_thread.run_sync(fn, goal, candidate)
                if not isinstance(result, Score):
                    ctx.judge_error(
                        candidate.id, f"returned {type(result).__name__}, expected Score"
                    )
                elif not is_finite_score(result):
                    ctx.judge_error(candidate.id, "score key must be nonempty and finite")
                else:
                    scores.append(result.model_copy(update={"candidate_id": candidate.id}))
            except Exception as error:
                ctx.judge_error(
                    candidate.id, f"{type(error).__name__}: scoring failed; expected a finite Score"
                )
        return scores
