"""Offline search tools, bounds check, and judge for the Rosenbrock function."""

import math
import random
from typing import ClassVar, cast

from swarm_arena import Goal, Score
from swarm_arena.board import Candidate
from swarm_arena.scoring import Check, CheckContext, CheckResult


def gradient_descent(lr: float, steps: int, x0: float, y0: float) -> dict[str, float]:
    """
    Minimize Rosenbrock's function with simultaneous gradient updates.

    Parameters
    ----------
    lr : float
        Learning rate; 0.001 is a stable choice for the example's starting points.
    steps : int
        Number of gradient updates.
    x0, y0 : float
        Starting coordinates.

    Returns
    -------
    dict[str, float]
        Final coordinates ``x``, ``y`` and objective ``f``.

    Examples
    --------
    >>> gradient_descent(0.001, 2000, -1.5, 2.0)["f"] < 12.5
    True
    """
    x, y = x0, y0
    for _ in range(steps):
        dx = 2 * (x - 1) - 400 * x * (y - x * x)
        dy = 200 * (y - x * x)
        x, y = x - lr * dx, y - lr * dy
    return {"x": x, "y": y, "f": (1 - x) ** 2 + 100 * (y - x * x) ** 2}


def random_search(n: int, radius: float, seed: int | None = None) -> dict[str, float]:
    """
    Select the best of uniformly sampled points in a square around the origin.

    Parameters
    ----------
    n : int
        Positive number of points to sample.
    radius : float
        Half-width of the square in each coordinate.
    seed : int or None, default=None
        Local random generator seed; omit it for fresh samples.

    Returns
    -------
    dict[str, float]
        Best coordinates ``x``, ``y`` and objective ``f``.

    Examples
    --------
    >>> random_search(100, 2.0, seed=7) == random_search(100, 2.0, seed=7)
    True
    """
    rng = random.Random(seed)
    points = []
    for _ in range(n):
        x, y = rng.uniform(-radius, radius), rng.uniform(-radius, radius)
        points.append({"x": x, "y": y, "f": (1 - x) ** 2 + 100 * (y - x * x) ** 2})
    return min(points, key=lambda point: point["f"])


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


class BoundsCheck(Check):
    """
    Require finite result data with both coordinates inside the configured bounds.

    Parameters
    ----------
    config : BoundsCheck.Config or mapping or None, optional
        Bounds configuration; None uses a bound of 10.0.

    Examples
    --------
    >>> BoundsCheck().config.bound
    10.0
    """

    name: ClassVar[str] = "bounds"

    class Config(Check.Config):
        """
        Configure an immutable, extra-forbidding bounds check.

        Parameters
        ----------
        bound : float, default=10.0
            Maximum absolute coordinate value.

        Examples
        --------
        >>> BoundsCheck.Config().bound
        10.0
        """

        bound: float = 10.0

    async def check(self, candidate: Candidate, ctx: CheckContext) -> CheckResult:
        """
        Check coordinates and objective, failing closed on missing or invalid data.

        Parameters
        ----------
        candidate : Candidate
            Tool result to validate.
        ctx : CheckContext
            Execution context; this in-memory check does not execute commands.

        Returns
        -------
        CheckResult
            Pass/fail result with a fixed, value-free failure reason.

        Examples
        --------
        >>> import asyncio
        >>> from unittest.mock import Mock
        >>> from swarm_arena.tool import RunStatus, ToolOutput, ToolRequest
        >>> candidate = Candidate(id="c", run_id="r", goal_id="rosenbrock", team="descent",
        ...     agent="gd-1", round=1, request=ToolRequest(),
        ...     output=ToolOutput(status=RunStatus.OK, data={"x": 1.0, "y": 1.0, "f": 0.0}))
        >>> asyncio.run(BoundsCheck().check(candidate, Mock(spec=CheckContext))).passed
        True
        """
        data = candidate.output.data
        bound = cast(BoundsCheck.Config, self.config).bound
        passed = (
            isinstance(data, dict)
            and all(_finite_number(data.get(key)) for key in ("x", "y", "f"))
            and abs(data["x"]) <= bound
            and abs(data["y"]) <= bound
        )
        return CheckResult(
            check=self.name,
            passed=passed,
            reasons=[] if passed else ["result must be finite and inside the coordinate bounds"],
        )


def score(goal: Goal, candidate: Candidate) -> Score:
    """
    Rank a candidate by its Rosenbrock objective, preferring smaller values.

    Parameters
    ----------
    goal : Goal
        Goal being scored.
    candidate : Candidate
        Result containing a finite numeric ``f`` value.

    Returns
    -------
    Score
        Minimization score whose human-facing value is ``f``.

    Raises
    ------
    ValueError
        If objective data is missing or invalid.

    Examples
    --------
    >>> from swarm_arena.tool import RunStatus, ToolOutput, ToolRequest
    >>> candidate = Candidate(id="c", run_id="r", goal_id="rosenbrock", team="descent",
    ...     agent="gd-1", round=1, request=ToolRequest(),
    ...     output=ToolOutput(status=RunStatus.OK, data={"f": 0.25}))
    >>> score(Goal(id="rosenbrock", namespace="toy"), candidate).key
    (-0.25,)
    """
    data = candidate.output.data
    if not isinstance(data, dict) or not _finite_number(data.get("f")):
        raise ValueError("result requires a finite numeric f") from None
    return Score.minimize(data["f"])
