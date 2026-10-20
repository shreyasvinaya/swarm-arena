"""Define synchronous board indexing and snapshot read contracts."""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from swarm_arena._model.board_types import Frontier, LineageGraph, Row
from swarm_arena._model.goal import GoalKey
from swarm_arena._model.planning_types import Decision
from swarm_arena._model.ref import Configurable
from swarm_arena._model.results import Candidate, Learning, Score


class _Reads(ABC):
    """Share the board and snapshot read interface."""

    @abstractmethod
    def best(
        self,
        goal: GoalKey,
        k: int,
        *,
        judge: str,
        run_id: str | None = None,
        part: str | None = None,
        fidelity: str | None = None,
    ) -> list[Candidate]:
        """
        Return the best complete, gated-in candidates of one ranking group.

        Parameters
        ----------
        goal : GoalKey
            Registered goal identity.
        k : int
            Maximum number of candidates; nonpositive values return no results.
        judge : str
            Judge name; only its latest identity in the group is compared.
        run_id : str or None, default=None
            Optional producing run filter.
        part : str or None, default=None
            Part filter, defaulting to the registered goal's part.
        fidelity : str or None, default=None
            Effort label, or the highest-round group's fidelity when absent.

        Returns
        -------
        list[Candidate]
            Candidates in deterministic score order.

        Raises
        ------
        ValueError
            If the goal is unknown or outside this view.

        Examples
        --------
        Reuse the public in-memory setup in the :class:`BoardView` example:

        >>> from doctest import script_from_examples
        >>> from swarm_arena.board import BoardView
        >>> exec(script_from_examples(BoardView.__doc__))
        >>> view.best(key, 5, judge="program")
        []
        """

    @abstractmethod
    def learnings(
        self,
        namespace: str,
        *,
        tags: Sequence[str] = (),
        goal: GoalKey | None = None,
        limit: int = 50,
    ) -> list[Learning]:
        """
        List visible learnings in relevance order across a namespace.

        Parameters
        ----------
        namespace : str
            Problem family.
        tags : Sequence[str], default=()
            Tags used to prefer matching learnings.
        goal : GoalKey or None, default=None
            Goal whose learnings are preferred after pinned items.
        limit : int, default=50
            Maximum number of items; nonpositive values return no results.

        Returns
        -------
        list[Learning]
            Pinned, goal-relevant, verified, tag-matching, then newest items.
            Hidden items are excluded.

        Raises
        ------
        ValueError
            If namespace or goal lies outside this view.

        Examples
        --------
        Reuse the public in-memory setup in the :class:`BoardView` example:

        >>> from doctest import script_from_examples
        >>> from swarm_arena.board import BoardView
        >>> exec(script_from_examples(BoardView.__doc__))
        >>> view.learnings("n")[0].title
        'Use the baseline'
        """

    @abstractmethod
    def search_learnings(
        self,
        namespace: str,
        query: str,
        *,
        tags: Sequence[str] = (),
        goal: GoalKey | None = None,
        limit: int = 10,
    ) -> list[Learning]:
        """
        Search titles and bodies by case-insensitive substring, newest first.

        Parameters
        ----------
        namespace : str
            Problem family.
        query : str
            Substring to find in a title or body.
        tags : Sequence[str], default=()
            Required tags; every supplied tag must be present.
        goal : GoalKey or None, default=None
            Optional goal scope, validated against a snapshot's goal.
        limit : int, default=10
            Maximum number of matches; nonpositive values return no results.

        Returns
        -------
        list[Learning]
            Visible matches in descending journal order.

        Raises
        ------
        ValueError
            If namespace or goal lies outside this view.

        Examples
        --------
        Reuse the public in-memory setup in the :class:`BoardView` example:

        >>> from doctest import script_from_examples
        >>> from swarm_arena.board import BoardView
        >>> exec(script_from_examples(BoardView.__doc__))
        >>> view.search_learnings("n", "baseline")[0].id
        'L-1'
        """

    @abstractmethod
    def get_learning(self, id: str) -> Learning:
        """
        Read a visible learning by identifier.

        Parameters
        ----------
        id : str
            Learning identifier.

        Returns
        -------
        Learning
            An independent copy of the learning.

        Raises
        ------
        KeyError
            If the learning is absent or hidden.
        ValueError
            If its namespace lies outside this snapshot.

        Examples
        --------
        Reuse the public in-memory setup in the :class:`BoardView` example:

        >>> from doctest import script_from_examples
        >>> from swarm_arena.board import BoardView
        >>> exec(script_from_examples(BoardView.__doc__))
        >>> view.get_learning("L-1").title
        'Use the baseline'
        """

    @abstractmethod
    def get_candidate(self, id: str) -> Candidate:
        """
        Read a candidate by identifier within this goal's snapshot.

        Parameters
        ----------
        id : str
            Candidate identifier.

        Returns
        -------
        Candidate
            An independent copy of the candidate.

        Raises
        ------
        KeyError
            If the candidate is absent from the snapshot.
        ValueError
            If the candidate belongs to another goal or part.

        Examples
        --------
        Reuse the public in-memory setup in the :class:`BoardView` example:

        >>> from doctest import script_from_examples
        >>> from swarm_arena.board import BoardView
        >>> exec(script_from_examples(BoardView.__doc__))
        >>> view.get_candidate("c-1").id
        'c-1'
        """

    @abstractmethod
    def scores(self, candidate_id: str) -> list[Score]:
        """
        Read every recorded score of a visible candidate in journal order.

        Parameters
        ----------
        candidate_id : str
            Candidate identifier.

        Returns
        -------
        list[Score]
            Independent score copies, including older identities.

        Raises
        ------
        KeyError
            If the candidate is absent from the snapshot.
        ValueError
            If the candidate belongs to another goal or part.

        Examples
        --------
        Reuse the public in-memory setup in the :class:`BoardView` example:

        >>> from doctest import script_from_examples
        >>> from swarm_arena.board import BoardView
        >>> exec(script_from_examples(BoardView.__doc__))
        >>> view.scores("c-1")
        []
        """

    @abstractmethod
    def lineage(self, candidate_id: str) -> LineageGraph:
        """
        Traverse visible candidate ancestry in parent order.

        Parameters
        ----------
        candidate_id : str
            Root candidate identifier.

        Returns
        -------
        LineageGraph
            Visible same-goal nodes and child-to-parent edges; cycles terminate.

        Raises
        ------
        KeyError
            If the root is absent from the snapshot.
        ValueError
            If the root belongs to another goal or part.

        Examples
        --------
        Reuse the public in-memory setup in the :class:`BoardView` example:

        >>> from doctest import script_from_examples
        >>> from swarm_arena.board import BoardView
        >>> exec(script_from_examples(BoardView.__doc__))
        >>> view.lineage("c-1").parents("c-1")
        []
        """

    @abstractmethod
    def leaderboard(self, goal: GoalKey, *, judge: str) -> list[Row]:
        """
        Describe the goal's complete ranking with judge provenance.

        Parameters
        ----------
        goal : GoalKey
            Registered goal identity.
        judge : str
            Ranking judge name.

        Returns
        -------
        list[Row]
            One-based rows under the latest judge identity and newest fidelity.

        Raises
        ------
        ValueError
            If the goal is unknown or outside this view.

        Examples
        --------
        Reuse the public in-memory setup in the :class:`BoardView` example:

        >>> from doctest import script_from_examples
        >>> from swarm_arena.board import BoardView
        >>> exec(script_from_examples(BoardView.__doc__))
        >>> view.leaderboard(key, judge="program")
        []
        """


class BoardView(_Reads):
    """
    Read one goal's fixed round-start snapshot and its namespace's learnings.

    Identifier reads raise KeyError for items outside the recorded frontier.
    Other goal or namespace arguments raise ValueError. Implement all eight
    read methods and the private candidate-membership seam to provide a custom
    snapshot.

    Attributes
    ----------
    goal : GoalKey
        Goal identity admitted to this view.
    frontier : Frontier
        Exact committed heads used to construct the view.

    Examples
    --------
    This minimal single-goal view demonstrates unscored, parentless records.
    A production connector also implements ranking, learning relevance, ancestry,
    and selection by committed frontier. All setup uses public interfaces:

    >>> from collections.abc import Sequence
    >>> from swarm_arena import Goal, Score
    >>> from swarm_arena.board import BoardView, Candidate, Frontier, GoalKey, Learning
    >>> from swarm_arena.board import LineageGraph, Row
    >>> from swarm_arena.tool import ToolOutput, ToolRequest
    >>> class MemoryReads:
    ...     def __init__(self, goal: GoalKey) -> None:
    ...         self.goal = goal
    ...         self.candidates: dict[str, Candidate] = {}
    ...         self.learning_records: dict[str, Learning] = {}
    ...         self.score_records: dict[str, list[Score]] = {}
    ...     def best(self, goal: GoalKey, k: int, *, judge: str,
    ...              run_id: str | None = None, part: str | None = None,
    ...              fidelity: str | None = None) -> list[Candidate]:
    ...         if goal != self.goal:
    ...             raise ValueError("goal outside view")
    ...         return []
    ...     def learnings(self, namespace: str, *, tags: Sequence[str] = (),
    ...                   goal: GoalKey | None = None, limit: int = 50) -> list[Learning]:
    ...         if namespace != self.goal.namespace or goal not in (None, self.goal):
    ...             raise ValueError("scope outside view")
    ...         return [v.model_copy(deep=True) for v in self.learning_records.values()
    ...                 if not v.hidden and set(tags) <= set(v.tags)][:max(0, limit)]
    ...     def search_learnings(self, namespace: str, query: str, *,
    ...                         tags: Sequence[str] = (), goal: GoalKey | None = None,
    ...                         limit: int = 10) -> list[Learning]:
    ...         return [v for v in self.learnings(namespace, tags=tags, goal=goal)
    ...                 if query.casefold() in (v.title + " " + v.text).casefold()][:max(0, limit)]
    ...     def get_learning(self, id: str) -> Learning:
    ...         item = self.learning_records[id]
    ...         if item.hidden:
    ...             raise KeyError(id)
    ...         return item.model_copy(deep=True)
    ...     def get_candidate(self, id: str) -> Candidate:
    ...         return self.candidates[id].model_copy(deep=True)
    ...     def scores(self, candidate_id: str) -> list[Score]:
    ...         self.get_candidate(candidate_id)
    ...         return [v.model_copy(deep=True) for v in self.score_records.get(candidate_id, [])]
    ...     def lineage(self, candidate_id: str) -> LineageGraph:
    ...         candidate = self.get_candidate(candidate_id)
    ...         return LineageGraph(root=candidate_id, nodes={candidate_id: candidate}, edges=[])
    ...     def leaderboard(self, goal: GoalKey, *, judge: str) -> list[Row]:
    ...         self.best(goal, 1, judge=judge)
    ...         return []
    >>> class MemoryView(MemoryReads, BoardView):
    ...     def __init__(self, goal: GoalKey, frontier: Frontier) -> None:
    ...         MemoryReads.__init__(self, goal)
    ...         self.frontier = frontier
    >>> key = Goal(id="g", namespace="n").key()
    >>> view = MemoryView(key, Frontier())
    >>> candidate = Candidate(id="c-1", run_id="r", goal_id="g", team="search",
    ...     agent="a", round=1, request=ToolRequest(), output=ToolOutput(status="ok"))
    >>> learning = Learning(id="L-1", namespace="n", run_id="r", goal_id="g",
    ...     round=1, author="human:a", title="Use the baseline", text="Start here")
    >>> view.candidates[candidate.id] = candidate
    >>> view.learning_records[learning.id] = learning
    >>> view.get_candidate("c-1").id
    'c-1'
    """

    goal: GoalKey
    frontier: Frontier

    def _visible_candidate_ids(self) -> frozenset[str]:
        """Return fixed, goal-scoped candidate membership without I/O or charges."""
        raise NotImplementedError


class Board(Configurable, _Reads):
    """
    Index journaled run records and provide deterministic board snapshots.

    Read methods and put methods run on the event loop. Async engine callers
    offload open, close, frontier and snapshot to AnyIO workers. The board
    indexes already-journaled records and never writes the journal itself.

    Parameters
    ----------
    config : BaseModel or Mapping[str, Any] or None, optional
        Settings validated through the board's nested configuration model.

    Examples
    --------
    Reuse the unscored ``MemoryReads`` mixin defined with public interfaces in the
    :class:`BoardView` example. ``script_from_examples`` executes that displayed
    setup so each example also runs independently. This demonstration stores
    records in memory; a production connector selects committed journal heads.

    >>> from doctest import script_from_examples
    >>> from swarm_arena.board import Board, BoardView
    >>> from swarm_arena.planning import Decision
    >>> exec(script_from_examples(BoardView.__doc__))
    >>> class MemoryBoard(MemoryReads, Board):
    ...     def __init__(self) -> None:
    ...         Board.__init__(self)
    ...         MemoryReads.__init__(self, key)
    ...         self.decisions: dict[tuple[str, int], Decision] = {}
    ...     def put_candidate(self, candidate: Candidate, *, epoch: int, seq: int) -> None:
    ...         self.candidates[candidate.id] = candidate.model_copy(deep=True)
    ...     def put_score(self, score: Score, *, epoch: int, seq: int) -> None:
    ...         records = self.score_records.setdefault(score.candidate_id, [])
    ...         records.append(score.model_copy(deep=True))
    ...     def put_learning(self, learning: Learning, *, epoch: int, seq: int) -> None:
    ...         self.learning_records[learning.id] = learning.model_copy(deep=True)
    ...     def put_decision(self, goal_id: str, round: int, decision: Decision,
    ...                      *, epoch: int, seq: int) -> None:
    ...         self.decisions[goal_id, round] = decision.model_copy(deep=True)
    ...     def frontier(self) -> Frontier:
    ...         return Frontier()
    ...     def snapshot(self, goal: GoalKey, frontier: Frontier) -> BoardView:
    ...         from copy import deepcopy
    ...         if goal != self.goal:
    ...             raise ValueError("unknown goal")
    ...         result = MemoryView(goal, frontier.model_copy(deep=True))
    ...         result.candidates = deepcopy(self.candidates)
    ...         result.learning_records = deepcopy(self.learning_records)
    ...         result.score_records = deepcopy(self.score_records)
    ...         return result
    >>> board = MemoryBoard()
    >>> board.put_candidate(candidate, epoch=1, seq=1)
    >>> board.snapshot(key, board.frontier()).get_candidate("c-1").id
    'c-1'
    """

    def open(self, run_id: str, run_root: str, namespaces: Sequence[str]) -> None:
        """
        Open a run's board resources; the default does nothing.

        Parameters
        ----------
        run_id : str
            Producing run identifier.
        run_root : str
            Storage root containing the run.
        namespaces : Sequence[str]
            Namespaces used by this run.

        Examples
        --------
        Reuse the public in-memory setup in the :class:`Board` example:

        >>> from doctest import script_from_examples
        >>> from swarm_arena.board import Board
        >>> exec(script_from_examples(Board.__doc__))
        >>> board.open("r", "/runs", ["n"])
        """

    def close(self) -> None:
        """
        Close board resources; the default does nothing.

        Examples
        --------
        Reuse the public in-memory setup in the :class:`Board` example:

        >>> from doctest import script_from_examples
        >>> from swarm_arena.board import Board
        >>> exec(script_from_examples(Board.__doc__))
        >>> board.close()
        """

    @abstractmethod
    def put_candidate(self, candidate: Candidate, *, epoch: int, seq: int) -> None:
        """
        Index a journaled candidate at its recorded position.

        Parameters
        ----------
        candidate : Candidate
            Committed candidate record.
        epoch : int
            Ownership epoch of its event.
        seq : int
            Run-wide journal sequence number of its event.

        Examples
        --------
        Reuse the public in-memory setup in the :class:`Board` example:

        >>> from doctest import script_from_examples
        >>> from swarm_arena.board import Board
        >>> exec(script_from_examples(Board.__doc__))
        >>> board.put_candidate(candidate, epoch=1, seq=4)
        >>> board.get_candidate("c-1").id
        'c-1'
        """

    @abstractmethod
    def put_score(self, score: Score, *, epoch: int, seq: int) -> None:
        """
        Index a journaled score at its recorded position.

        Parameters
        ----------
        score : Score
            Stamped score with its judge identity.
        epoch : int
            Ownership epoch of its event.
        seq : int
            Run-wide journal sequence number of its event.

        Examples
        --------
        Reuse the public in-memory setup in the :class:`Board` example:

        >>> from doctest import script_from_examples
        >>> from swarm_arena.board import Board
        >>> exec(script_from_examples(Board.__doc__))
        >>> score = Score(id="s-1", candidate_id="c-1", judge="program",
        ...     rubric_hash="r", panel_hash="p", key=(1.0,))
        >>> board.put_score(score, epoch=1, seq=5)
        >>> board.scores("c-1")[0].key
        (1.0,)
        """

    @abstractmethod
    def put_learning(self, learning: Learning, *, epoch: int, seq: int) -> None:
        """
        Index a journaled learning at its recorded position.

        Parameters
        ----------
        learning : Learning
            Namespace learning, including its visibility flags.
        epoch : int
            Ownership epoch of its event.
        seq : int
            Run-wide journal sequence number of its event.

        Examples
        --------
        Reuse the public in-memory setup in the :class:`Board` example:

        >>> from doctest import script_from_examples
        >>> from swarm_arena.board import Board
        >>> exec(script_from_examples(Board.__doc__))
        >>> board.put_learning(learning, epoch=1, seq=6)
        >>> board.get_learning("L-1").title
        'Use the baseline'
        """

    @abstractmethod
    def put_decision(
        self,
        goal_id: str,
        round: int,
        decision: Decision,
        *,
        epoch: int,
        seq: int,
    ) -> None:
        """
        Index an already-journaled policy decision.

        Parameters
        ----------
        goal_id : str
            Main goal identifier.
        round : int
            Completed round number.
        decision : Decision
            Policy pruning result.
        epoch : int
            Ownership epoch of its event.
        seq : int
            Run-wide journal sequence number of its event.

        Examples
        --------
        Reuse the public in-memory setup in the :class:`Board` example:

        >>> from doctest import script_from_examples
        >>> from swarm_arena.board import Board
        >>> exec(script_from_examples(Board.__doc__))
        >>> decision = Decision(prune={"search/a": "stalled"}, policy="patience")
        >>> board.put_decision("g", 1, decision, epoch=1, seq=7)
        >>> board.decisions["g", 1].policy
        'patience'
        """

    @abstractmethod
    def frontier(self) -> Frontier:
        """
        Capture the current committed heads included by this board.

        Returns
        -------
        Frontier
            Run marks and exact human-object names in canonical order.

        Examples
        --------
        Reuse the public in-memory setup in the :class:`Board` example:

        >>> from doctest import script_from_examples
        >>> from swarm_arena.board import Board
        >>> exec(script_from_examples(Board.__doc__))
        >>> board.frontier().marks
        []
        """

    @abstractmethod
    def snapshot(self, goal: GoalKey, frontier: Frontier) -> BoardView:
        """
        Freeze a goal view at the supplied committed frontier.

        Parameters
        ----------
        goal : GoalKey
            Registered goal identity.
        frontier : Frontier
            Exact heads delimiting visible records.

        Returns
        -------
        BoardView
            An independent, fixed read view.

        Raises
        ------
        ValueError
            If the goal is not registered.

        Examples
        --------
        Reuse the public in-memory setup in the :class:`Board` example:

        >>> from doctest import script_from_examples
        >>> from swarm_arena.board import Board
        >>> exec(script_from_examples(Board.__doc__))
        >>> board.snapshot(key, board.frontier()).goal == key
        True
        """
