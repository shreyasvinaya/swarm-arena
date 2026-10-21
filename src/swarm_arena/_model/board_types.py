"""Define durable board frontiers and in-memory ranking and lineage views."""

from collections import deque
from typing import ClassVar

from swarm_arena._model.base import FrozenModel, StoredModel
from swarm_arena._model.canonical import canonical_json, sha256_hex
from swarm_arena._model.results import Candidate


class Frontier(StoredModel):
    """
    Store the exact committed records included in a board snapshot.

    Parameters
    ----------
    schema_version : int, default=1
        Stored-format version.
    marks : list[Frontier.Mark], default=[]
        Committed journal heads for each included run.
    reviews : list[str], default=[]
        Exact review object names.
    curation : list[str], default=[]
        Exact curation object names.

    Examples
    --------
    >>> from swarm_arena.board import Frontier
    >>> Frontier().marks
    []
    """

    class Mark(FrozenModel):
        """
        Identify one run's committed journal head.

        Parameters
        ----------
        run_id : str
            Run identifier.
        run_root : str
            Storage root containing the run.
        epoch : int
            Head segment's owner epoch.
        seq : int
            Head segment's final sequence number.
        segment : str
            Exact run-relative journal segment name.

        Examples
        --------
        >>> from swarm_arena.board import Frontier
        >>> mark = Frontier.Mark(run_id="r", run_root="/runs", epoch=1,
        ...     seq=4, segment="journal/e1-4.jsonl")
        >>> mark.seq
        4
        """

        run_id: str
        run_root: str
        epoch: int
        seq: int
        segment: str

    schema_kind: ClassVar[str] = "frontier"
    marks: list[Mark] = []
    reviews: list[str] = []
    curation: list[str] = []

    def digest(self) -> str:
        """
        Hash the canonical JSON-mode dump of this frontier.

        Returns
        -------
        str
            Lowercase SHA-256 digest including the stored version and list order.

        Examples
        --------
        >>> from swarm_arena.board import Frontier
        >>> len(Frontier().digest())
        64
        >>> Frontier().digest() == Frontier().digest()
        True
        """
        return sha256_hex(canonical_json(self.model_dump(mode="json")))


class Row(FrozenModel):
    """
    Present a candidate's leaderboard position and ranking provenance.

    Parameters
    ----------
    rank : int
        Leaderboard position.
    candidate_id : str
        Ranked candidate identifier.
    run_id : str
        Producing run identifier.
    goal_id : str
        Goal identifier.
    part : str or None
        Part name, or None for the main goal.
    team : str
        Producing team name.
    agent : str
        Producing agent name.
    fidelity : str
        Ranked effort level.
    key : tuple[float, ...]
        Lexicographic score key, larger being better.
    value : float or None
        Optional human-facing score value.
    judge : str
        Judge name.
    rubric_hash : str
        Rubric identity.
    panel_hash : str
        Panel membership identity.
    built_on : list[str]
        Parent candidate identifiers.
    partial : bool
        Whether the output is partial.

    Examples
    --------
    >>> from swarm_arena.board import Row
    >>> row = Row(rank=1, candidate_id="c", run_id="r", goal_id="g", part=None,
    ...     team="t", agent="a", fidelity="full", key=(2.0,), value=2.0,
    ...     judge="program", rubric_hash="rh", panel_hash="", built_on=[], partial=False)
    >>> row.rank
    1
    """

    rank: int
    candidate_id: str
    run_id: str
    goal_id: str
    part: str | None
    team: str
    agent: str
    fidelity: str
    key: tuple[float, ...]
    value: float | None
    judge: str
    rubric_hash: str
    panel_hash: str
    built_on: list[str]
    partial: bool


class LineageGraph(FrozenModel):
    """
    Represent candidate ancestry with ordered child-to-parent edges.

    Parameters
    ----------
    root : str
        Candidate identifier at the root of this view.
    nodes : dict[str, Candidate]
        Candidate records indexed by identifier.
    edges : list[tuple[str, str]]
        Ordered child and parent identifier pairs.

    Examples
    --------
    >>> from swarm_arena.board import LineageGraph
    >>> graph = LineageGraph(root="c", nodes={}, edges=[("c", "b"), ("b", "a")])
    >>> graph.ancestors("c")
    ['b', 'a']
    """

    root: str
    nodes: dict[str, Candidate]
    edges: list[tuple[str, str]]

    def parents(self, candidate_id: str) -> list[str]:
        """
        Return direct parent identifiers in edge order.

        Parameters
        ----------
        candidate_id : str
            Child candidate identifier.

        Returns
        -------
        list[str]
            Direct parents, or an empty list if there are no matching edges.

        Examples
        --------
        >>> from swarm_arena.board import LineageGraph
        >>> LineageGraph(root="c", nodes={}, edges=[("c", "b")]).parents("c")
        ['b']
        """
        return [parent for child, parent in self.edges if child == candidate_id]

    def ancestors(self, candidate_id: str) -> list[str]:
        """
        Traverse unique ancestors breadth-first in edge order.

        Parameters
        ----------
        candidate_id : str
            Starting candidate identifier, excluded from the result even in cycles.

        Returns
        -------
        list[str]
            Reachable parents and ancestors, each included once. Traversal follows
            edges even if a node record is unavailable and terminates on cycles.

        Examples
        --------
        >>> from swarm_arena.board import LineageGraph
        >>> graph = LineageGraph(root="c", nodes={},
        ...     edges=[("c", "b"), ("b", "a"), ("c", "a")])
        >>> graph.ancestors("c")
        ['b', 'a']
        """
        parents: dict[str, list[str]] = {}
        for child, parent in self.edges:
            parents.setdefault(child, []).append(parent)
        seen = {candidate_id}
        queue = deque(parents.get(candidate_id, []))
        result: list[str] = []
        while queue:
            parent = queue.popleft()
            if parent in seen:
                continue
            seen.add(parent)
            result.append(parent)
            queue.extend(parents.get(parent, []))
        return result
