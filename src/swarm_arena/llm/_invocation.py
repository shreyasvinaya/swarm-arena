"""Own one child LLM invocation and its accounting on every exit."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import anyio

from swarm_arena._budget.llm_scope import LLMScope, llm_scope
from swarm_arena._errors import StorageError, _core_category, _is_core_control, _SignalCategory
from swarm_arena._outcomes import _Outcome
from swarm_arena.llm._hooks import BudgetExceeded
from swarm_arena.llm._manager import _finish_accounting


@dataclass
class _InvocationAccounting:
    """Carry accounting origin separately from the classified control outcome."""

    failed: bool = False


@asynccontextmanager
async def _invocation(
    parent: LLMScope,
    *,
    name: str,
    error_type: type[Exception],
    accounting_name: str = "LLM accounting",
    accounting: _InvocationAccounting | None = None,
) -> AsyncGenerator[LLMScope]:
    """
    Enter a child scope, contain failures and finish accounting before exit.

    The caller owns the surrounding supervision and supplies the diagnostic
    name and ordinary error type. Caller cancellation wins over fencing,
    terminal cleanup and ordinary limits through the shared outcome classifier.
    Direct budget refusals retain their original SDK-compatible carrier.

    Parameters
    ----------
    parent : LLMScope
        Owner scope supplying funding and services.
    name : str
        Core component identifier, such as ``curator`` or ``planner``.
    error_type : type[Exception]
        Boundary type for an ordinary unregistered component failure.

    accounting_name : str, default="LLM accounting"
        Diagnostic site for an ordinary accounting failure.

    accounting : _InvocationAccounting or None, default=None
        Optional private status marking a failed accounting step independently
        of the control category propagated to the caller.

    Yields
    ------
    LLMScope
        Child with independent call identifiers and a shared parent tranche.

    Raises
    ------
    Exception
        Classified body or cleanup failure after accounting. Core controls
        retain their authority and genuine caller cancellation propagates.

    Examples
    --------
    An unbound in-memory scope has no calls to drain:

    >>> from swarm_arena import Limits
    >>> from swarm_arena.config import ModelCatalog, TracingConfig
    >>> import anyio
    >>> parent = LLMScope(role="curator", run_id="example", goal_id="g",
    ...     team="", agent="", round=1, candidate_id=None, tranche=None,
    ...     journal=None, artifacts=None, models=ModelCatalog({}),
    ...     tracing=TracingConfig(), cache=None, limits=Limits(), settle_timeout_s=2)
    >>> async def example():
    ...     async with _invocation(parent, name="curator", error_type=ValueError) as child:
    ...         return child.parent is parent and child.call_ids == []
    >>> anyio.run(example)
    True
    """
    scope = parent.child()
    boundary = _Outcome()
    failure: BaseException | None = None
    accounting_failure: BaseException | None = None
    interrupted = False
    try:
        with llm_scope(scope):
            yield scope
    except BaseException as error:
        failure = error
        try:
            boundary.classify(error)
        except anyio.get_cancelled_exc_class() as cancelled:
            interrupted, failure = True, cancelled
        except BaseException as control:
            failure = control
    finally:
        try:
            await _finish_accounting(scope, interrupted=interrupted)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as cleanup_error:
            if accounting is not None:
                accounting.failed = True
            accounting_failure = cleanup_error
            failure = (
                BaseExceptionGroup(f"{name} cleanup", [failure, cleanup_error])
                if failure is not None
                else cleanup_error
            )
    if failure is not None:
        # Keep the captured cancellation active for AnyIO scope attribution.
        try:
            raise failure
        except BaseException as error:
            classified = boundary.classify(error)
            if _core_category(classified) is _SignalCategory.TERMINAL:
                raise classified from None
            if accounting_failure is not None:
                boundary.contain(accounting_failure, StorageError, accounting_name)
            if _is_core_control(classified):
                if (
                    type(failure) is BudgetExceeded
                    or _core_category(failure) is _SignalCategory.LIMIT
                ):
                    raise failure from None
                raise classified from None
            boundary.contain(classified, error_type, f"LLM {name}")
    boundary.propagate_control(RuntimeError())
