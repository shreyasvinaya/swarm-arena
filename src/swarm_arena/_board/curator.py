"""Define configurable curation and prepare evidence-backed summary learnings."""

import json
import sys
from abc import ABC, abstractmethod
from collections.abc import Sequence
from functools import partial
from inspect import isawaitable

from swarm_arena._agent.manager import LimitReached
from swarm_arena._board.base import BoardView
from swarm_arena._errors import Fenced, _core_error, _rebuild_core_signal
from swarm_arena._exec.threads import run_sync_daemon
from swarm_arena._model.ids import IdGen
from swarm_arena._model.ref import Configurable
from swarm_arena._model.results import Learning


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


def _curator_error(error: Exception) -> str:
    """Preserve rebuilt core controls and contain untrusted exception values."""
    safe = _rebuild_core_signal(error)
    if isinstance(safe, (Fenced, LimitReached)):
        raise safe from None
    if isinstance(error, Fenced):
        raise _core_error(Fenced) from None
    return f"curator error: {type(error).__name__}"


def _normalize_summaries(
    result: object,
    sources: dict[str, tuple[tuple[str, ...], bool]],
    *,
    namespace: str,
    run_id: str,
    goal_id: str,
    round: int,
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
            posted_by_call=None,
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
    try:
        # Board access stays on the loop. Snapshot evidence before exposing inputs
        # to the plugin, and never give the normalizing daemon a board or IdGen.
        sources = {
            item.id: (tuple(item.evidence), item.verified)
            for item in [*view.learnings(namespace, limit=sys.maxsize), *new]
            if item.namespace == namespace and not item.hidden
        }
        await run_sync_daemon(lambda: curator.models())
        result = await run_sync_daemon(lambda: curator.curate(namespace, new, view))
        if isawaitable(result):
            result = await result
        return await run_sync_daemon(
            partial(
                _normalize_summaries,
                result,
                sources,
                namespace=namespace,
                run_id=run_id,
                goal_id=goal_id,
                round=round,
            )
        )
    except Exception as error:
        return [], [_curator_error(error)]


def _stamp_summaries(summaries: Sequence[Learning], ids: IdGen) -> list[Learning]:
    """Assign IDs on the loop only after the complete batch is core-owned."""
    return [summary.model_copy(update={"id": ids.next("L")}) for summary in summaries]
