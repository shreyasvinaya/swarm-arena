"""Define configurable curation and prepare evidence-backed summary learnings."""

import json
import sys
from abc import ABC, abstractmethod
from collections.abc import Sequence
from functools import partial
from inspect import isawaitable

from swarm_arena._board.base import BoardView
from swarm_arena._budget.llm_scope import current_llm_scope
from swarm_arena._errors import _exception_type_name, _is_core_control
from swarm_arena._exec.threads import run_sync_daemon
from swarm_arena._journal.llm_record import drain
from swarm_arena._model.context_strings import _context_string
from swarm_arena._model.detach import _detach_model
from swarm_arena._model.ids import IdGen
from swarm_arena._model.ref import Configurable
from swarm_arena._model.results import Learning
from swarm_arena._outcomes import _classify_outcome, _Outcome


class Curator(Configurable, ABC):
    """
    Merge related learnings into evidence-backed summaries at round end.

    Subclasses propose content and nonempty ``supersedes`` lists. The core
    validates proposals and supplies identity, routing, evidence and flags.

    Synchronous curator hooks run in abandonable daemon threads. The engine
    abandons their waits at stop-now and on timeouts; late results cannot
    publish summaries. Async curator hooks run on the engine loop and must
    cooperate with cancellation: they must not block the loop or shield their
    waits. Use a synchronous hook when its body requires abandonment.

    Parameters
    ----------
    config : BaseModel or Mapping[str, Any] or None, optional
        Settings validated by the curator's nested configuration model.

    Examples
    --------
    >>> from swarm_arena.board import Curator
    >>> class NoMerge(Curator):
    ...     async def curate(self, namespace, new, view):
    ...         return []
    >>> NoMerge().config.model_dump()
    {}
    """

    def models(self) -> tuple[str, ...]:
        """
        Declare model aliases used by this curator.

        Returns
        -------
        tuple[str, ...]
            Empty by default for a curator that makes no model calls.

        Examples
        --------
        >>> Curator.models(None)
        ()
        """
        return ()

    @abstractmethod
    async def curate(
        self, namespace: str, new: Sequence[Learning], view: BoardView
    ) -> list[Learning]:
        """
        Propose summaries of new or previously visible namespace learnings.

        Parameters
        ----------
        namespace : str
            Namespace whose learnings may be summarized.
        new : Sequence[Learning]
            This goal's learnings from the completed ordinary round.
        view : BoardView
            Fixed board snapshot containing visible namespace evidence.

        Returns
        -------
        list[Learning]
            Proposals with content and nonempty supersedes lists. Core-owned
            metadata is replaced after validation.

        Examples
        --------
        >>> import anyio
        >>> from swarm_arena.board import Curator
        >>> class NoMerge(Curator):
        ...     async def curate(self, namespace, new, view):
        ...         return []
        >>> anyio.run(NoMerge().curate, "demo", [], None)
        []
        """


def _curator_error(error: BaseException) -> str:
    """Preserve validated provider/control signals; return only type diagnostics.

    The owning goal settles the invocation before recording a provider skip.
    Ordinary hook failures remain component errors without provider authority.
    """
    error = _classify_outcome(error)
    if _is_core_control(error):
        raise error from None
    return f"curator error: {_exception_type_name(error)}"


def _normalize_summaries(
    result: object,
    sources: dict[str, tuple[tuple[str, ...], bool]],
    *,
    namespace: str,
    run_id: str,
    goal_id: str,
    round: int,
    call_ids: tuple[str, ...] = (),
) -> tuple[list[Learning], list[str]]:
    """Traverse and detach user objects in a daemon, without mutable services."""
    if not isinstance(result, Sequence) or isinstance(result, (str, bytes, bytearray)):
        raise TypeError("invalid curator result")
    summaries: list[Learning] = []
    errors: list[str] = []
    for proposal in result:
        if not isinstance(proposal, Learning):
            raise TypeError("invalid curator learning")
        # Detach nested containers before any authority-bearing update: even
        # a serializer-returned dict may override update or retain plugin aliases.
        data = json.loads(json.dumps(proposal.model_dump(mode="python", warnings=False)))
        if not isinstance(data, dict):
            raise TypeError("invalid curator learning payload")
        supersedes = data.get("supersedes")
        if (
            not isinstance(supersedes, list)
            or not supersedes
            or any(not isinstance(id, str) or id not in sources for id in supersedes)
        ):
            errors.append("invalid summary supersedes")
            continue
        evidence = list(dict.fromkeys(id for old in supersedes for id in sources[old][0]))
        if not evidence:
            errors.append("summary has no evidence")
            continue
        data.update(
            id="pending",
            namespace=namespace,
            run_id=run_id,
            goal_id=goal_id,
            round=round,
            author="curator",
            evidence=evidence,
            verified=any(sources[id][1] for id in supersedes),
            pinned=False,
            hidden=False,
            posted_by_call=(
                data.get("posted_by_call")
                if data.get("posted_by_call") in call_ids
                else call_ids[-1]
                if call_ids
                else None
            ),
        )
        # Production validators and a JSON round trip leave only core-owned values.
        summary = Learning.model_validate(data)
        summaries.append(Learning.model_validate_json(summary.model_dump_json()))
    return summaries, errors


async def _prepare_curator(
    curator: Curator,
    *,
    namespace: str,
    run_id: str,
    goal_id: str,
    round: int,
    new: Sequence[Learning],
    view: BoardView,
) -> tuple[list[Learning], list[str]]:
    """Run hooks and detach a complete batch; the owning goal supervises this await."""
    outcome = _Outcome()
    scope = current_llm_scope()
    previous_calls = (
        {_context_string(id, "call_id") for id in scope.call_ids} if scope is not None else set()
    )
    try:
        namespace = _context_string(namespace, "namespace")
        run_id = _context_string(run_id, "run_id")
        goal_id = _context_string(goal_id, "goal_id")

        def read_sources() -> dict[str, tuple[tuple[str, ...], bool]]:
            # Connector reads and returned models belong to the supervised worker.
            items = [
                _detach_model(Learning, item)
                for item in [*view.learnings(namespace, limit=sys.maxsize), *new]
            ]
            return {
                item.id: (tuple(item.evidence), item.verified)
                for item in items
                if item.namespace == namespace and not item.hidden
            }

        sources = await run_sync_daemon(read_sources)
        await run_sync_daemon(lambda: curator.models())
        result = await run_sync_daemon(lambda: curator.curate(namespace, new, view))
        if isawaitable(result):
            result = await result
        # The invocation owns all of its calls, including late custom callbacks.
        # Resolve provenance only after their bounded accounting has completed.
        if scope is not None:
            await drain(scope, wait=True)
        call_ids = (
            tuple(
                id
                for raw in scope.call_ids
                if (id := _context_string(raw, "call_id")) not in previous_calls
            )
            if scope is not None
            else ()
        )
        return await run_sync_daemon(
            partial(
                _normalize_summaries,
                result,
                sources,
                namespace=namespace,
                run_id=run_id,
                goal_id=goal_id,
                round=round,
                call_ids=call_ids,
            )
        )
    except BaseException as error:
        return [], [_curator_error(outcome.classify(error))]


def _stamp_summaries(summaries: Sequence[Learning], ids: IdGen) -> list[Learning]:
    """Assign IDs on the loop only after the complete batch is core-owned."""
    return [summary.model_copy(update={"id": ids.next("L")}) for summary in summaries]
