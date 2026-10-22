"""Rubric judging with metered panels and repeated batch anchors."""

import json
import statistics
from collections.abc import Sequence
from typing import Any, Final, Literal, Self, cast

import anyio
from pydantic import Field, JsonValue, model_validator

from swarm_arena._board.digest import untrusted
from swarm_arena._budget.llm_scope import current_llm_scope, llm_scope
from swarm_arena._engine.stops import report_provider_failure
from swarm_arena._errors import (
    ProviderFailed,
    StorageError,
    _core_category,
    _is_core_control,
    _SignalCategory,
)
from swarm_arena._model.canonical import canonical_json, sha256_hex
from swarm_arena._model.goal import Goal
from swarm_arena._model.ref import _ConfigInputModel, _validate_config
from swarm_arena._model.results import Candidate, Score
from swarm_arena._outcomes import _Outcome
from swarm_arena._scoring.judge import Judge, JudgeContext, _panel_drops, _PanelDrop
from swarm_arena.llm._hooks import BudgetExceeded, _prepare
from swarm_arena.llm._manager import _finish_accounting
from swarm_arena.llm._structured import _captured_call_ids, structured_call

JUDGE_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "required": ["scores"],
    "properties": {
        "scores": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["index", "value", "verdict", "notes"],
                "properties": {
                    "index": {"type": "integer"},
                    "value": {"type": "number"},
                    "verdict": {"type": "string"},
                    "notes": {"type": "string"},
                },
            },
        }
    },
}


def _panel(models: Sequence[str]) -> str:
    return "" if len(models) == 1 else sha256_hex(canonical_json(cast(JsonValue, sorted(models))))


class LLMJudge(Judge):
    """
    Score candidates against a fixed rubric with a panel of model aliases.

    Parameters
    ----------
    rubric : str or LLMJudge.Config
        Rubric text or complete validated configuration.
    models : Sequence[str], default=()
        Catalog aliases, retaining repeated votes.
    aggregate : str, default='mean'
        Mean, median, minimum or maximum of surviving votes.

    Examples
    --------
    >>> from swarm_arena.llm import LLMJudge
    >>> LLMJudge("Correctness", ["a", "b"], "median").models()
    ('a', 'b')
    """

    name = "llm"

    class Config(_ConfigInputModel):
        """
        Configure rubric, panel, scale and internal batching.

        Parameters
        ----------
        rubric : str
            Fixed scoring instructions.
        models : list[str]
            Nonempty catalog panel, preserving duplicates.
        aggregate : {'mean', 'median', 'min', 'max'}, default='mean'
            Vote aggregation.
        scale_min, scale_max : float, default=0.0, 10.0
            Inclusive finite scale endpoints.
        anchors : int, default=2
            Number of leaders requested for each batch.
        batch_size : int, default=8
            Total candidates per ordinary batch, including anchors.

        Examples
        --------
        >>> from swarm_arena.llm import LLMJudge
        >>> LLMJudge.Config(rubric="r", models=["a"]).batch_size
        8
        """

        rubric: str
        models: list[str] = Field(min_length=1)
        aggregate: Literal["mean", "median", "min", "max"] = "mean"
        scale_min: float = 0.0
        scale_max: float = 10.0
        anchors: int = Field(default=2, ge=0)
        batch_size: int = Field(default=8, gt=0)

        @model_validator(mode="after")
        def _bounds(self) -> Self:
            if self.anchors >= self.batch_size:
                raise ValueError("anchors must be fewer than batch_size")
            if self.scale_min >= self.scale_max:
                raise ValueError("scale_min must be below scale_max")
            return self

    def __init__(
        self, rubric: str | Config, models: Sequence[str] = (), aggregate: str = "mean", /
    ) -> None:
        self._config = _validate_config(
            self.Config,
            rubric
            if isinstance(rubric, self.Config)
            else {"rubric": rubric, "models": list(models), "aggregate": aggregate},
            where="LLMJudge",
        )

    def models(self) -> tuple[str, ...]:
        """
        Return one alias per panel vote, including repeated members.

        Returns
        -------
        tuple[str, ...]
            Model catalog aliases used for pricing and voting.

        Examples
        --------
        >>> from swarm_arena.llm import LLMJudge
        >>> LLMJudge("r", ["a", "a"]).models()
        ('a', 'a')
        """
        return tuple(cast(LLMJudge.Config, self.config).models)

    def rubric_hash(self) -> str:
        """
        Hash the rubric, scale, aggregation and single-model alias.

        Returns
        -------
        str
            Stable SHA-256 identity.

        Examples
        --------
        >>> from swarm_arena.llm import LLMJudge
        >>> len(LLMJudge("r", ["a"]).rubric_hash())
        64
        """
        config = cast(LLMJudge.Config, self.config)
        return sha256_hex(
            canonical_json(
                {
                    "rubric": config.rubric,
                    "scale": [config.scale_min, config.scale_max],
                    "aggregate": config.aggregate,
                    "model": config.models[0] if len(config.models) == 1 else None,
                }
            )
        )

    def panel_hash(self) -> str:
        """
        Hash the sorted panel, retaining repeated aliases.

        Returns
        -------
        str
            Empty for a single configured model; otherwise SHA-256.

        Examples
        --------
        >>> from swarm_arena.llm import LLMJudge
        >>> LLMJudge("r", ["a"]).panel_hash()
        ''
        """
        return _panel(self.models())

    def anchor_count(self) -> int:
        """
        Return the requested number of leader anchors.

        Returns
        -------
        int
            Configured anchor count.

        Examples
        --------
        >>> from swarm_arena.llm import LLMJudge
        >>> LLMJudge("r", ["a"]).anchor_count()
        2
        """
        return cast(LLMJudge.Config, self.config).anchors

    async def judge(
        self, goal: Goal, candidates: Sequence[Candidate], ctx: JudgeContext
    ) -> list[Score]:
        """
        Score batches concurrently and drain accounting before returning.

        Parameters
        ----------
        goal : Goal
            Goal supplying the cache namespace.
        candidates : Sequence[Candidate]
            Candidates including the context's leader anchors.
        ctx : JudgeContext
            Final-call, cache and anchor settings.

        Returns
        -------
        list[Score]
            Scores agreed by every surviving member across all batches.

        Raises
        ------
        BudgetExceeded
            If a full-call envelope cannot be reserved before dispatch.
        RuntimeError
            If no LLM scope is active.

        Examples
        --------
        >>> scores = await judge.judge(goal, candidates, ctx)  # doctest: +SKIP
        """
        scope = current_llm_scope()
        if scope is None:
            raise RuntimeError("LLM judge outside an LLM scope")
        scope = scope.child()
        interrupted = False
        failure: BaseException | None = None
        accounting_failure: BaseException | None = None
        scores: list[Score] = []
        provider_errors: dict[int, Exception] = {}
        survivors: list[str] = []
        boundary = _Outcome()
        try:
            with llm_scope(scope):
                scores = await self._judge(goal, candidates, ctx, provider_errors, survivors)
        except BaseException as error:
            failure = error
            try:
                boundary.classify(error)
            except anyio.get_cancelled_exc_class() as cancelled:
                interrupted = True
                failure = cancelled
            except BaseException as control:
                failure = control
        finally:
            try:
                await _finish_accounting(scope, interrupted=interrupted)
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as cleanup_error:
                accounting_failure = cleanup_error
                failure = (
                    BaseExceptionGroup("judge cleanup", [failure, cleanup_error])
                    if failure is not None
                    else cleanup_error
                )
        classified: Exception | None = None
        if failure is not None:
            classified = boundary.classify(failure)
            if _core_category(classified) is _SignalCategory.TERMINAL:
                raise classified from None
            if accounting_failure is not None:
                # Provider classification cannot establish successful settlement.
                boundary.contain(accounting_failure, StorageError, "LLM judge accounting")
            if _core_category(classified) not in {_SignalCategory.PROVIDER, _SignalCategory.LIMIT}:
                boundary.contain(classified, RuntimeError, "LLM judge")
        boundary.propagate_control(RuntimeError())
        # Accounting must finish before reporting. A sibling LIMIT cannot erase
        # genuine provider evidence, but failed cleanup never advances the latch.
        reported: set[str] = set()
        for error in provider_errors.values():
            assert isinstance(error, ProviderFailed)
            if error.provider not in reported:
                await report_provider_failure(scope, error, source="judge")
                reported.add(error.provider)
        if classified is not None:
            assert failure is not None
            if type(failure) is BudgetExceeded or _core_category(failure) is _SignalCategory.LIMIT:
                raise failure from None
            raise classified from None
        # The scoring runner carries these to the engine's atomic score commit.
        # Single-member and total failures have no surviving panel to describe.
        config = cast(LLMJudge.Config, self.config)
        drops = _panel_drops.get()
        if drops is not None and scores and survivors and len(config.models) > 1:
            panel = sha256_hex(canonical_json(cast(JsonValue, sorted(survivors))))
            for member_index, error in sorted(provider_errors.items()):
                assert isinstance(error, ProviderFailed)
                drops.append(
                    _PanelDrop(
                        provider=error.provider,
                        dropped_member=config.models[member_index],
                        member_index=member_index,
                        panel_hash=panel,
                        batch_id=ctx.batch_id,
                    )
                )
        return scores

    async def _judge(
        self,
        goal: Goal,
        candidates: Sequence[Candidate],
        ctx: JudgeContext,
        provider_errors: dict[int, Exception],
        survivors: list[str],
    ) -> list[Score]:
        scope = current_llm_scope()
        assert scope is not None
        config = cast(LLMJudge.Config, self.config)
        anchors = [c for c in candidates if c.id in ctx.anchors]
        others = [c for c in candidates if c.id not in ctx.anchors]
        if not candidates:
            return []
        if ctx.final:
            batches = [list(candidates)]
        else:
            capacity = config.batch_size - len(anchors)
            if capacity <= 0:
                raise ValueError("anchors must be fewer than batch_size")
            batches = [
                anchors + others[i : i + capacity] for i in range(0, len(others), capacity)
            ] or [anchors]
        results: dict[tuple[int, int], tuple[dict[int, dict[str, Any]], list[str]]] = {}
        calls: dict[tuple[int, int], list[str]] = {}
        failed: set[int] = set()
        controls: list[BaseException] = []
        first_limit: BaseException | None = None

        def check_admission() -> None:
            if first_limit is not None:
                raise first_limit

        async def member(member_index: int, batch_index: int) -> None:
            nonlocal first_limit
            outcome = _Outcome()
            batch = batches[batch_index]
            captured: list[str] = []
            token = _captured_call_ids.set(captured)
            try:
                prompt = await _prepare(
                    lambda: untrusted(
                        json.dumps(
                            [
                                {"index": i, "candidate": c.model_dump(mode="json")}
                                for i, c in enumerate(batch)
                            ]
                        )
                    )
                )
                output, ids = await structured_call(
                    alias=config.models[member_index],
                    role="judge",
                    models=scope.models,
                    namespace=goal.namespace,
                    system=(
                        f"Score each candidate independently. Rubric: {config.rubric}\n"
                        f"Scale: [{config.scale_min}, {config.scale_max}]. "
                        "Return scores with index, value, verdict and notes."
                    ),
                    user=prompt,
                    schema=JUDGE_SCHEMA,
                    name="judge",
                    temperature=0,
                    admission_check=check_admission,
                    no_cache_read=not ctx.read_cache
                    or ctx.final
                    or any(c.id in ctx.anchors for c in batch),
                )
                captured.extend(id for id in ids if id not in captured)
                values: dict[int, dict[str, Any]] = {}
                for row in output["scores"]:
                    index = row["index"]
                    if type(index) is not int or not 0 <= index < len(batch) or index in values:
                        raise ValueError("invalid index")
                    values[index] = row
                results[member_index, batch_index] = values, ids
            except BaseException as error:
                classified = outcome.classify(error)
                if _core_category(classified) is _SignalCategory.PROVIDER:
                    provider_errors.setdefault(member_index, classified)
                    failed.add(member_index)
                elif _is_core_control(classified):
                    if _core_category(classified) is _SignalCategory.LIMIT and first_limit is None:
                        first_limit = error
                    controls.append(error if isinstance(error, BudgetExceeded) else classified)
                else:
                    failed.add(member_index)
            finally:
                calls[member_index, batch_index] = captured
                _captured_call_ids.reset(token)

        async with anyio.create_task_group() as group:
            for m in range(len(config.models)):
                for b in range(len(batches)):
                    group.start_soon(member, m, b)
        if controls:
            classified = _Outcome().classify(BaseExceptionGroup("judge controls", controls))
            if _core_category(classified) is _SignalCategory.LIMIT and first_limit is not None:
                raise first_limit
            raise classified from None
        live = [m for m in range(len(config.models)) if m not in failed]
        survivors.extend(config.models[m] for m in live)
        if not live:
            if provider_errors:
                raise provider_errors[min(provider_errors)] from None
            return []
        panel = (
            sha256_hex(canonical_json(cast(JsonValue, sorted(config.models[m] for m in live))))
            if len(config.models) > 1
            else ""
        )
        aggregate = {"mean": statistics.mean, "median": statistics.mean, "min": min, "max": max}[
            config.aggregate
        ]
        scores: dict[str, Score] = {}
        later: dict[str, list[float | str]] = {candidate.id: [] for candidate in anchors}
        omitted: set[str] = set()
        for b, batch in enumerate(batches):
            for index, candidate in enumerate(batch):
                rows = [results[m, b][0].get(index) for m in live]
                if any(
                    row is None
                    or type(row["value"]) not in (int, float)
                    or not config.scale_min <= row["value"] <= config.scale_max
                    for row in rows
                ):
                    if candidate.id in scores and candidate.id in ctx.anchors:
                        later[candidate.id].append("invalid")
                        scores[candidate.id].llm_call_ids.extend(
                            id for m in range(len(config.models)) for id in calls[m, b]
                        )
                    else:
                        omitted.add(candidate.id)
                    continue
                valid = [row for row in rows if row is not None]
                votes = [row["value"] for row in valid]
                if config.aggregate == "median":
                    votes.sort()
                    votes = votes[(len(votes) - 1) // 2 : len(votes) // 2 + 1]
                # mean uses exact ratios, avoiding overflow and subnormal underflow.
                value = float(aggregate(votes))
                ids = [id for m in range(len(config.models)) for id in calls[m, b]]
                if candidate.id in scores:
                    later.setdefault(candidate.id, []).append(value)
                    scores[candidate.id].llm_call_ids.extend(ids)
                else:
                    scores[candidate.id] = Score.maximize(
                        value,
                        verdict=valid[0]["verdict"],
                        notes="; ".join(row["notes"] for row in valid),
                    ).model_copy(
                        update={
                            "candidate_id": candidate.id,
                            "llm_call_ids": ids,
                            "panel_hash": panel,
                        }
                    )
        for id, values in later.items():
            if id not in scores:
                continue
            scores[id] = scores[id].model_copy(
                update={"notes": scores[id].notes + f"; anchor values in later batches: {values}"}
            )
        return [scores[c.id] for c in candidates if c.id in scores and c.id not in omitted]
