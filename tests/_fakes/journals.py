"""Committed, deterministic journals without an engine or JournalWriter."""

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from _fakes.leases import force_take_over

from swarm_arena import Goal, PartSpec, Plan, Ref, Score
from swarm_arena._board.pointers import RunPointer, write_pointer
from swarm_arena._journal.a2a import agent_task_id, task_to_json, tool_run_task
from swarm_arena._journal.event import Event, EventDraft, EventType
from swarm_arena._journal.lease import Head, Lease
from swarm_arena._journal.replay import RunState
from swarm_arena._journal.segment import encode_segment, segment_name
from swarm_arena._model.plan import ConvergeSpec
from swarm_arena._model.results import CheckResult, Learning
from swarm_arena._store.base import Store, open_store
from swarm_arena.testing import make_candidate
from swarm_arena.tool import RunStatus, ToolOutput, ToolRequest

if TYPE_CHECKING:
    from swarm_arena._board.cache import ArenaCache

START: Final = datetime(2026, 10, 18, tzinfo=UTC)


class FakeRun:
    """
    Write explicit committed segments using the production lease protocol.

    Parameters
    ----------
    run_root : str
        Local or S3 root containing runs.
    run_id : str
        New run identifier.
    namespace : str, default="toy"
        Default goal namespace.
    goal_id : str, default="g1"
        Default main goal identifier.
    start : datetime, default=START
        Timestamp origin; event sequence numbers add seconds.
    goal : Goal or None, default=None
        Explicit goal replacing namespace and goal_id defaults.

    Examples
    --------
    >>> run = FakeRun('/tmp/journal-fixture', 'example')  # doctest: +SKIP
    """

    def __init__(
        self,
        run_root: str,
        run_id: str,
        *,
        namespace: str = "toy",
        goal_id: str = "g1",
        start: datetime = START,
        goal: Goal | None = None,
    ) -> None:
        self.run_root = run_root
        self.run_id = run_id
        self.store: Store = open_store(run_root).child(run_id)
        self.goal = goal if goal is not None else Goal(id=goal_id, namespace=namespace)
        self.heads: list[Head] = []
        self._start = start
        self._lease = Lease.acquire_new(self.store, owner_id="fixture", host="local", ttl_s=3600)
        self._old_epoch = self._lease.doc.epoch
        self._events: list[Event] = []

    def _position(self, drafts: Sequence[EventDraft], epoch: int) -> list[Event]:
        seq = self.heads[-1].seq if self.heads else 0
        return [
            Event(
                **draft.model_dump(),
                run_id=self.run_id,
                context_id=self.run_id,
                epoch=epoch,
                seq=seq + i,
                ts=self._start + timedelta(seconds=seq + i),
            )
            for i, draft in enumerate(drafts, 1)
        ]

    def commit(self, *drafts: EventDraft) -> Head:
        """
        Publish one nonempty segment and advance authority with CAS.

        Parameters
        ----------
        *drafts : EventDraft
            Ordered records in this committed segment.

        Returns
        -------
        Head
            Published head, also appended to ``heads``.

        Examples
        --------
        >>> head = run.commit(run.plan(), run.candidate('c-1'))  # doctest: +SKIP
        """
        if not drafts:
            raise ValueError("commit requires events")
        events = self._position(drafts, self._lease.doc.epoch)
        head = Head(
            epoch=events[-1].epoch,
            seq=events[-1].seq,
            segment=segment_name(events[-1].epoch, events[-1].seq),
        )
        self.store.create(
            head.segment, encode_segment(self.heads[-1].segment if self.heads else None, events)
        )
        self._lease.advance(head)
        self.heads.append(head)
        self._events.extend(events)
        return head

    def take_over(self) -> None:
        """
        Fence the old writer, retaining its committed head.

        Examples
        --------
        >>> run.take_over()  # doctest: +SKIP
        """
        self._old_epoch = self._lease.doc.epoch
        self._lease = force_take_over(
            self.store,
            owner_id="replacement",
            host="local",
            ttl_s=3600,
            expect_epoch=self._old_epoch,
        )

    def orphan(self, *drafts: EventDraft) -> str:
        """
        Write a fenced old-epoch segment without publishing its head.

        Parameters
        ----------
        *drafts : EventDraft
            Records from the fenced writer.

        Returns
        -------
        str
            Orphan segment name, absent from committed authority.

        Examples
        --------
        >>> name = run.orphan(run.candidate('ghost'))  # doctest: +SKIP
        """
        if not drafts:
            raise ValueError("orphan requires events")
        events = self._position(drafts, self._old_epoch)
        name = segment_name(self._old_epoch, events[-1].seq)
        self.store.create(
            name, encode_segment(self.heads[-1].segment if self.heads else None, events)
        )
        return name

    def plan(
        self,
        *,
        revision: int = 0,
        ranking: str = "j",
        min_improvement: float | None = None,
        parts: Sequence[str] = (),
        first_round: int | None = None,
        join_retry: int | None = None,
    ) -> EventDraft:
        """
        Build a declared plan with producer activation and preflight metadata.

        Parameters
        ----------
        revision : int, default=0
            Committed revision number.
        ranking : str, default="j"
            Recorded ranking judge name.
        min_improvement : float or None, default=None
            Convergence improvement threshold.
        parts : Sequence[str], default=()
            Declared lowercase part names, each with its own goal id.
        first_round : int or None, default=None
            Activation boundary; None derives the next round from core replay.
        join_retry : int or None, default=None
            Optional completed join proposal count.

        Returns
        -------
        EventDraft
            Plan revision with final producer metadata.

        Examples
        --------
        >>> draft = run.plan(parts=['a'], first_round=1)  # doctest: +SKIP
        """
        if first_round is None:
            state = RunState.from_events(self._events)
            current = state.goals.get(self.goal.id)
            first_round = 1 if current is None else max(current.round, current.rounds_closed) + 1
        plan = Plan(
            goal=self.goal,
            teams=[],
            judges=[
                Ref(
                    path="swarm_arena:ProgramJudge",
                    name=ranking,
                    config={"function": "_fakes.tools:small_b"},
                )
            ],
            ranking=ranking,
            parts=[
                PartSpec(name=p, goal=self.goal.model_copy(update={"id": f"{self.goal.id}-{p}"}))
                for p in parts
            ],
            revision=revision,
            parent_revision=revision - 1 if revision else None,
            converge=ConvergeSpec(min_improvement=min_improvement),
        )
        payload: dict[str, Any] = {
            "plan": plan.model_dump(mode="json"),
            "origin": "human",
            "planner_call_ids": [],
            "first_round": first_round,
        }
        if revision == 0:
            payload["base_dir"] = str(Path.cwd())
        if join_retry is not None:
            payload["join_retry"] = join_retry
        return EventDraft(type=EventType.PLAN_REVISION, goal_id=self.goal.id, payload=payload)

    def candidate(
        self,
        cid: str,
        *,
        team: str = "t1",
        agent: str = "a1",
        round: int = 1,
        part: str | None = None,
        fidelity: str = "f1",
        partial: bool = False,
        passed: bool = True,
        parents: Sequence[str] = (),
        status: RunStatus = RunStatus.OK,
        repeat_key: str | None = None,
        fingerprint: str | None = None,
    ) -> EventDraft:
        """
        Build a tool completion, preserving task identity and optional hints.

        Parameters
        ----------
        cid : str
            Candidate and producer task identifier.
        team : str, default="t1"
            Producing team.
        agent : str, default="a1"
            Producing agent.
        round : int, default=1
            Producing round.
        part : str or None, default=None
            Declared part, or the main goal.
        fidelity : str, default="f1"
            Effort label.
        partial : bool, default=False
            Whether the tool output is partial.
        passed : bool, default=True
            Result of the fixture check.
        parents : Sequence[str], default=()
            Ordered parent identifiers.
        status : RunStatus, default=RunStatus.OK
            Tool completion status.
        repeat_key : str or None, default=None
            Already computed repeat identity; omitted when None.
        fingerprint : str or None, default=None
            Recorded effective-tool fingerprint, never recomputed by readers.

        Returns
        -------
        EventDraft
            Tool completion using the production A2A serializer.

        Examples
        --------
        >>> draft = run.candidate('c-1', part='a')  # doctest: +SKIP
        """
        candidate = make_candidate(
            ToolOutput(status=status, partial=partial),
            request=ToolRequest(round=round),
            id=cid,
            run_id=self.run_id,
            goal_id=self.goal.id,
            part=part,
            team=team,
            agent=agent,
            round=round,
            fidelity=fidelity,
            parent_ids=list(parents),
            checks=[CheckResult(check="c", passed=passed)],
            overlay={},
        )
        payload: dict[str, Any] = {
            "candidate": candidate.model_dump(mode="json"),
            "fingerprint": fingerprint,
            "a2a": task_to_json(tool_run_task(candidate, context_id=self.run_id, at=self._start)),
        }
        if repeat_key is not None:
            payload["repeat_key"] = repeat_key
        return EventDraft(
            type=EventType.TOOL_END,
            goal_id=self.goal.id,
            team=team,
            agent=agent,
            round=round,
            task_id=cid,
            payload=payload,
        )

    def score(
        self,
        sid: str,
        cid: str,
        x: float,
        *,
        judge: str = "j",
        rubric_hash: str = "r1",
        panel_hash: str = "",
        minimize: bool = True,
    ) -> EventDraft:
        """
        Build a recorded score with the larger-is-better key convention.

        Parameters
        ----------
        sid : str
            Score identifier.
        cid : str
            Scored candidate identifier.
        x : float
            Human-facing score value.
        judge : str, default="j"
            Recorded judge name.
        rubric_hash : str, default="r1"
            Rubric identity.
        panel_hash : str, default=""
            Panel identity.
        minimize : bool, default=True
            Whether smaller values rank higher.

        Returns
        -------
        EventDraft
            Score event routed to the owning main goal.

        Examples
        --------
        >>> draft = run.score('s-1', 'c-1', 0.25)  # doctest: +SKIP
        """
        score = Score(
            id=sid,
            candidate_id=cid,
            judge=judge,
            rubric_hash=rubric_hash,
            panel_hash=panel_hash,
            key=(-x if minimize else x,),
            value=x,
        )
        return EventDraft(
            type=EventType.SCORE,
            goal_id=self.goal.id,
            payload={"score": score.model_dump(mode="json")},
        )

    def learning(
        self,
        lid: str,
        title: str,
        *,
        text: str = "t",
        evidence: Sequence[str] = (),
        tags: Sequence[str] = (),
        verified: bool = False,
        refines: Sequence[str] = (),
        supersedes: Sequence[str] = (),
        round: int = 1,
        author: str = "t1/a1",
        posted_by_call: str | None = None,
    ) -> EventDraft:
        """
        Build a main-goal learning, defaulting to committed candidate evidence.

        Parameters
        ----------
        lid : str
            Learning identifier.
        title : str
            Single-line learning title.
        text : str, default="t"
            Learning body.
        evidence : Sequence[str], default=()
            Candidate evidence; a nonhuman default uses the last committed candidate.
        tags : Sequence[str], default=()
            Search tags.
        verified : bool, default=False
            Recorded verification status.
        refines : Sequence[str], default=()
            Referenced earlier learnings.
        supersedes : Sequence[str], default=()
            Replaced learning identifiers.
        round : int, default=1
            Posting round.
        author : str, default="t1/a1"
            Team/agent address or explicit human-prefixed author.
        posted_by_call : str or None, default=None
            Posting model call provenance.

        Returns
        -------
        EventDraft
            Learning event with the owning agent task and main goal routing.

        Examples
        --------
        >>> draft = run.learning('L-1', 'Try this', evidence=['c-1'])  # doctest: +SKIP
        """
        if not evidence and not author.startswith("human:"):
            evidence = [
                event.payload["candidate"]["id"]
                for event in self._events
                if event.type == EventType.TOOL_END and event.goal_id == self.goal.id
            ][-1:]
        learning = Learning(
            id=lid,
            namespace=self.goal.namespace,
            run_id=self.run_id,
            goal_id=self.goal.id,
            round=round,
            author=author,
            title=title,
            text=text,
            evidence=list(evidence),
            tags=list(tags),
            verified=verified,
            refines=list(refines),
            supersedes=list(supersedes),
            posted_by_call=posted_by_call,
        )
        team, _, agent = author.partition("/")
        return EventDraft(
            type=EventType.LEARNING,
            goal_id=self.goal.id,
            team=team,
            agent=agent,
            round=round,
            task_id=agent_task_id(f"{self.goal.id}/{author}", round),
            payload={"learning": learning.model_dump(mode="json")},
        )


def add_pointer(arena_root: str, run: FakeRun, namespace: str | None = None) -> None:
    """
    Publish the run through the production write-once pointer helper.

    Parameters
    ----------
    arena_root : str
        Arena root receiving the immutable pointer.
    run : FakeRun
        Run to publish.
    namespace : str or None, default=None
        Pointer namespace, defaulting to the run goal's namespace.

    Examples
    --------
    >>> add_pointer('/tmp/arena-fixture', run)  # doctest: +SKIP
    """
    write_pointer(
        open_store(arena_root),
        namespace or run.goal.namespace,
        RunPointer(run_id=run.run_id, run_root=run.run_root),
    )


def candidate_ids(cache: "ArenaCache") -> list[str]:
    """
    Inspect candidate IDs under the connection's complete-query lock.

    Parameters
    ----------
    cache : ArenaCache
        Open namespace cache.

    Returns
    -------
    list[str]
        Candidate identifiers in lexical order.

    Examples
    --------
    >>> candidate_ids(cache)  # doctest: +SKIP
    ['c-1']
    """
    with cache._lock:
        return [
            row[0]
            for row in cache._conn.execute("SELECT id FROM candidates ORDER BY id").fetchall()
        ]
