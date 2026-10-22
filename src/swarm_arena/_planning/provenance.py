"""Track reference provenance without importing or constructing components."""

from collections.abc import Mapping
from typing import Any

from pydantic import PrivateAttr

from swarm_arena._model.base import CatalogKind, FrozenModel, Origin
from swarm_arena._model.canonical import canonical_json
from swarm_arena._model.data import json_equal
from swarm_arena._model.detach import _detach_model
from swarm_arena._model.plan import Plan
from swarm_arena._model.ref import Ref, is_ref_path


class RefSite(FrozenModel):
    """
    Describe a reference occurrence and its inherited human authority.

    Parameters
    ----------
    pointer : str
        RFC 6901 pointer into a plan; effective override sites use ``@`` to
        join the override pointer to the affected tool pointer.
    kind : str
        Component role inherited by any nested references.
    ref : Ref
        Raw reference, or the effective tool reference at an override site.
    carried_human : bool, default=False
        Whether this unchanged site retains human authority from its parent.

    Notes
    -----
    ``ref_sites`` attaches a private stable key to each occurrence. A directly
    constructed site has only its pointer as a scope and cannot inherit the
    authority of a site produced by traversal.

    Examples
    --------
    >>> site = RefSite(pointer="/judges/0", kind="judge", ref=Ref(path="demo:Judge"))
    >>> site.carried_human
    False
    """

    pointer: str
    kind: CatalogKind
    ref: Ref
    carried_human: bool = False
    _key: str | None = PrivateAttr(default=None)


def is_nested_ref(value: Any) -> bool:
    """
    Recognize a nested reference shape without resolving its import path.

    Parameters
    ----------
    value : Any
        Raw JSON value, potentially a dumped reference.

    Returns
    -------
    bool
        Whether a dictionary has a full Ref path and only path/config/name
        keys. Reference field validation happens separately during traversal.

    Examples
    --------
    >>> is_nested_ref({"path": "demo.tools:Tool", "config": {"steps": 2}})
    True
    >>> is_nested_ref({"path": "demo:Tool", "other": 1})
    False
    """
    return (
        isinstance(value, dict)
        and value.keys() <= {"path", "config", "name"}
        and is_ref_path(value.get("path"))
    )


def site_key(site: RefSite) -> str:
    """
    Return a collision-proof, import-free identity for a reference occurrence.

    Parameters
    ----------
    site : RefSite
        Reference site produced by ``ref_sites`` or ``diff_revision``.

    Returns
    -------
    str
        Canonical JSON encoding of kind, address scope, explicit-name versus
        path discriminator, raw label and occurrence. Nested sites append
        escaped configuration pointers. Directly constructed sites instead
        use a separate pointer scope.

    Examples
    --------
    >>> site = RefSite(pointer="/policy", kind="policy", ref=Ref(path="demo:Policy"))
    >>> site_key(site) == site_key(site.model_copy())
    True
    """
    if site._key is not None:
        return site._key
    return _root_key(site.kind, ["pointer", site.pointer], site.ref, 0)


def _root_key(kind: CatalogKind, scope: list[str | int], ref: Ref, occurrence: int) -> str:
    """Encode each identity dimension separately, including name-versus-path."""
    discriminator = "name" if ref.name is not None else "path"
    label = ref.name if ref.name is not None else ref.path
    return canonical_json([kind, [*scope], discriminator, label, occurrence]).decode("utf-8")


def _escape(token: str) -> str:
    """Escape a single RFC 6901 pointer token."""
    return token.replace("~", "~0").replace("/", "~1")


def ref_sites(plan: Plan) -> list[RefSite]:
    """
    Enumerate all raw and effective reference sites without importing code.

    Parameters
    ----------
    plan : Plan
        Plan containing main and converge agents, scoring refs, part scoring,
        policies and round overrides.

    Returns
    -------
    list[RefSite]
        Sites at every configuration depth with stable scoped keys. Explicit
        rounds and the explicit converge round each contribute an effective
        tool site per affected agent, including when overrides are empty.

    Examples
    --------
    >>> from swarm_arena import Goal
    >>> plan = Plan(goal=Goal(id="g", namespace="demo"), teams=[],
    ...             judges=[Ref(path="demo:Judge")])
    >>> [site.pointer for site in ref_sites(plan)]
    ['/judges/0', '/policy']
    """
    sites: list[RefSite] = []
    occurrences: dict[str, int] = {}

    def walk(value: Any, pointer: str, kind: CatalogKind, key: str) -> None:
        """Visit every nested mapping/list, recognizing only full Ref shapes."""
        if isinstance(value, dict):
            if is_nested_ref(value):
                add(Ref.model_validate(value), pointer, kind, key)
            else:
                for name, child in value.items():
                    suffix = "/" + _escape(name)
                    walk(child, pointer + suffix, kind, key + suffix)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                suffix = f"/{index}"
                walk(child, pointer + suffix, kind, key + suffix)

    def add(ref: Ref, pointer: str, kind: CatalogKind, key: str) -> None:
        """Attach a stable key and traverse this reference's raw configuration."""
        site = RefSite(pointer=pointer, kind=kind, ref=ref)
        site._key = key
        sites.append(site)
        walk(ref.config, pointer + "/config", kind, key + "/config")

    def root(ref: Ref, pointer: str, kind: CatalogKind, scope: list[str | int]) -> None:
        """Assign duplicate labels distinct structured occurrence numbers."""
        identity = _root_key(kind, scope, ref, 0)
        occurrence = occurrences.get(identity, 0)
        occurrences[identity] = occurrence + 1
        add(ref, pointer, kind, _root_key(kind, scope, ref, occurrence))

    for ti, team in enumerate(plan.teams):
        for ai, agent in enumerate(team.agents):
            pointer = f"/teams/{ti}/agents/{ai}"
            scope: list[str | int] = ["agent", team.name, agent.name]
            root(agent.tool, pointer + "/tool", "tool", scope)
            root(agent.manager, pointer + "/manager", "manager", scope)
    for kind, refs in (("check", plan.checks), ("judge", plan.judges)):
        for index, ref in enumerate(refs):
            root(ref, f"/{kind}s/{index}", kind, ["plan"])
    for pi, part in enumerate(plan.parts):
        for kind, refs in (("check", part.checks), ("judge", part.judges)):
            for index, ref in enumerate(refs or []):
                root(ref, f"/parts/{pi}/{kind}s/{index}", kind, ["part", part.name])
    root(plan.policy, "/policy", "policy", ["plan"])
    converge = plan.converge
    if converge.team is not None:
        for ai, agent in enumerate(converge.team.agents):
            pointer = f"/converge/team/agents/{ai}"
            scope = ["converge", converge.team.name, agent.name]
            root(agent.tool, pointer + "/tool", "tool", scope)
            root(agent.manager, pointer + "/manager", "manager", scope)
    for ri, round_spec in enumerate(plan.rounds):
        for ti, team in enumerate(plan.teams):
            for ai, agent in enumerate(team.agents):
                effective = agent.tool.model_copy(
                    update={"config": {**agent.tool.config, **round_spec.tool_overrides}}
                )
                root(
                    effective,
                    f"/rounds/{ri}/tool_overrides@/teams/{ti}/agents/{ai}/tool",
                    "tool",
                    ["round", ri, team.name, agent.name],
                )
    if converge.team is not None and converge.round is not None:
        for ai, agent in enumerate(converge.team.agents):
            effective = agent.tool.model_copy(
                update={"config": {**agent.tool.config, **converge.round.tool_overrides}}
            )
            root(
                effective,
                f"/converge/round/tool_overrides@/converge/team/agents/{ai}/tool",
                "tool",
                ["converge-round", converge.team.name, agent.name],
            )
    return sites


def diff_revision(
    parent: Plan | None, child: Plan, *, parent_human: Mapping[str, bool]
) -> list[RefSite]:
    """
    Carry human authority only across unchanged, type-equal reference sites.

    Parameters
    ----------
    parent : Plan or None
        Immediate preceding revision, if any.
    child : Plan
        Revision whose sites are returned.
    parent_human : Mapping[str, bool]
        Human-authority flags indexed by the parent's stable site keys.

    Returns
    -------
    list[RefSite]
        Every child site, marked carried only when its full dumped Ref equals
        the parent's Ref under typed JSON comparison and the parent flag is true.

    Examples
    --------
    >>> from swarm_arena import Goal
    >>> plan = Plan(goal=Goal(id="g", namespace="demo"), teams=[], judges=[])
    >>> diff_revision(None, plan, parent_human={})[0].carried_human
    False
    """
    previous = {site_key(s): s for s in ref_sites(parent)} if parent is not None else {}
    result: list[RefSite] = []
    for site in ref_sites(child):
        key = site_key(site)
        old = previous.get(key)
        carried = (
            old is not None
            and parent_human.get(key, False)
            and json_equal(old.ref.model_dump(mode="json"), site.ref.model_dump(mode="json"))
        )
        result.append(site.model_copy(update={"carried_human": carried}))
    return result


def stamp(plan: Plan, *, origin: Origin, parent: Plan | None) -> Plan:
    """
    Copy a plan with core-assigned provenance while retaining its base directory.

    Parameters
    ----------
    plan : Plan
        Planner output whose provenance fields will be replaced.
    origin : str
        Origin assigned by the core.
    parent : Plan or None
        Immediate predecessor, or None for revision zero.

    Returns
    -------
    Plan
        Copy with origin, revision and parent_revision overwritten.

    Examples
    --------
    >>> from swarm_arena import Goal
    >>> plan = Plan(goal=Goal(id="g", namespace="demo"), teams=[], judges=[])
    >>> stamped = stamp(plan, origin="llm", parent=None)
    >>> (stamped.origin, stamped.revision, stamped.parent_revision)
    ('llm', 0, None)
    """
    return plan.model_copy(
        update={
            "origin": origin,
            "revision": 0 if parent is None else parent.revision + 1,
            "parent_revision": None if parent is None else parent.revision,
        }
    )


def _detach_plan(value: object) -> Plan:
    """Detach a planner's model and preserve its nonserialized base directory."""
    if not isinstance(value, Plan):
        raise TypeError("invalid planner result")
    base_dir = value.base_dir
    plan = _detach_model(Plan, value)
    plan._base_dir = type(plan.base_dir)(str(base_dir))
    plan._set_planner_provenance(value.planning_session, value.planner_call_ids)
    return plan
