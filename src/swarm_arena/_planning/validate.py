"""Admit frozen revisions after import-free authorization and aggregate plan rules."""

import re
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from functools import partial
from typing import Any, TypeVar

from pydantic import BaseModel

from swarm_arena._agent.manager import Manager
from swarm_arena._errors import ConfigError, PlanError, _core_error
from swarm_arena._model.canonical import canonical_json, sha256_hex
from swarm_arena._model.catalog import CatalogEntry, _human_only_inputs
from swarm_arena._model.data import json_equal
from swarm_arena._model.goal import FileSet, Goal, GoalKey
from swarm_arena._model.locations import _safe_location
from swarm_arena._model.plan import AgentSpec, Plan, TeamSpec
from swarm_arena._model.ref import Configurable, Ref
from swarm_arena._model.run_config import RunConfig
from swarm_arena._model.tooling import ToolRequest
from swarm_arena._planning.catalog_check import _check_catalog_defaults, authorize_revision
from swarm_arena._planning.policy import Policy
from swarm_arena._planning.provenance import RefSite, is_nested_ref, ref_sites
from swarm_arena._scoring.check import Check
from swarm_arena._scoring.judge import Judge
from swarm_arena._store.artifacts import ArtifactStore
from swarm_arena._tooling.files import snapshot_fileset, snapshot_plan, snapshot_ref
from swarm_arena._tooling.paths import fold_key, matches
from swarm_arena._tooling.staging import tool_root
from swarm_arena._tooling.tool import Tool, effective_tool_ref

NAME_RE = re.compile(r"[a-z0-9][a-z0-9._-]*")
ModelCheck = Callable[[Plan, RunConfig], tuple[list[str], list[str]]]
_MODEL_CHECKS: list[ModelCheck] = []
_T = TypeVar("_T")
_BASES: dict[str, type[Configurable]] = {
    "tool": Tool,
    "manager": Manager,
    "check": Check,
    "judge": Judge,
    "policy": Policy,
}


def _teams(plan: Plan) -> list[tuple[str, TeamSpec]]:
    teams = [(f"/teams/{i}", team) for i, team in enumerate(plan.teams)]
    if plan.converge.team is not None:
        teams.append(("/converge/team", plan.converge.team))
    return teams


def _agents(plan: Plan) -> Iterator[tuple[str, AgentSpec]]:
    for pointer, team in _teams(plan):
        for i, agent in enumerate(team.agents):
            yield f"{pointer}/agents/{i}", agent


def _inherited_converge(plan: Plan) -> Plan:
    """Expose the last round's inherited overrides to raw authorization as well."""
    if plan.converge.team is not None and plan.converge.round is None and plan.rounds:
        return plan.model_copy(
            update={"converge": plan.converge.model_copy(update={"round": plan.rounds[-1]})}
        )
    return plan


def _authorize(plan: Plan, config: RunConfig, history: Sequence[Plan]) -> Plan:
    projected = _inherited_converge(plan)
    projected_history = [_inherited_converge(p) for p in history]
    deferred: list[tuple[RefSite, CatalogEntry, tuple[str, ...]]] = []
    authorized, errors = _safe_call(
        "/refs",
        partial(
            authorize_revision,
            projected,
            config,
            history=projected_history,
            deferred=deferred,
        ),
    )
    if errors:
        # Retired addresses are also import-free, so report them even when a
        # restored agent has already lost its inherited catalog authority.
        raise _core_error(PlanError, [*errors, *_rule_addresses(plan, config, history)]) from None
    default_errors = _safe_call("/refs", partial(_check_catalog_defaults, deferred))
    if default_errors:
        raise _core_error(PlanError, default_errors) from None
    if projected is not plan:
        authorized = authorized.model_copy(
            update={"converge": authorized.converge.model_copy(update={"round": None})}
        )
    return authorized


def _safe_call(pointer: str, action: Callable[[], _T]) -> _T:
    """Hide arbitrary connector values at the admission boundary."""
    try:
        return action()
    except Exception as error:
        raise _core_error(PlanError, [f"{pointer}: {type(error).__name__}"]) from None


def _site_pointers(sites: Sequence[RefSite]) -> dict[str, str]:
    """Retain schema fields and indices, hiding data-derived nested pointer keys."""
    pointers: dict[str, str] = {}
    for index, site in enumerate(sites):
        parent = next(
            (
                ancestor
                for ancestor in reversed(sites[:index])
                if site.pointer.startswith(ancestor.pointer + "/config/")
            ),
            None,
        )
        if parent is None:
            pointers[site.pointer] = site.pointer
            continue
        current: Any = parent.ref.config
        location: list[str | int] = []
        suffix = site.pointer[len(parent.pointer + "/config/") :]
        for token in suffix.split("/"):
            key = token.replace("~1", "/").replace("~0", "~")
            if isinstance(current, list) and key.isdecimal():
                location.append(int(key))
                current = current[int(key)] if int(key) < len(current) else None
            else:
                location.append(key)
                current = current.get(key) if isinstance(current, dict) else None
        try:
            cls = parent.ref.load()
            if not issubclass(cls, Configurable):
                raise TypeError
            safe = _safe_location(cls.Config, tuple(location))
        except Exception:
            safe = ("<key>",) * len(location)
        tokens = [
            "<key>" if part == "<…>" else str(part).replace("~", "~0").replace("/", "~1")
            for part in safe
        ]
        pointers[site.pointer] = pointers[parent.pointer] + "/config/" + "/".join(tokens)
    return pointers


def _rule_names(plan: Plan, config: RunConfig, history: Sequence[Plan]) -> list[str]:
    errors: list[str] = []

    def name(value: str, pointer: str) -> None:
        if NAME_RE.fullmatch(value) is None:
            errors.append(f"{pointer}: name must be a lowercase slug")

    def unique(values: Sequence[str], pointer: str, kind: str) -> None:
        if len(set(values)) != len(values):
            errors.append(f"{pointer}: duplicate {kind} names")

    teams = _teams(plan)
    unique([team.name for _, team in teams], "/teams", "team")
    unique([part.name for part in plan.parts], "/parts", "part")
    for pointer, team in teams:
        name(team.name, pointer + "/name")
        unique([a.name for a in team.agents], pointer + "/agents", "agent")
        for i, agent in enumerate(team.agents):
            name(agent.name, f"{pointer}/agents/{i}/name")
    for i, part in enumerate(plan.parts):
        name(part.name, f"/parts/{i}/name")
    sites = ref_sites(plan)
    pointers = _site_pointers(sites)
    for site in sites:
        if site.ref.name is not None:
            name(site.ref.name, pointers[site.pointer] + "/name")
    return errors


def _rule_rationales(plan: Plan, config: RunConfig, history: Sequence[Plan]) -> list[str]:
    return [
        f"{pointer}/rationale: rationale must not be blank"
        for pointer, obj in [*_teams(plan), *_agents(plan)]
        if not obj.rationale.strip()
    ]


def _peak(plan: Plan) -> int:
    return max(
        sum(len(team.agents) for team in plan.teams),
        len(plan.converge.team.agents) if plan.converge.team is not None else 0,
    )


def _rule_limits(plan: Plan, config: RunConfig, history: Sequence[Plan]) -> list[str]:
    limits = config.limits
    errors = []
    for pointer, value, limit, field in [
        ("/teams", len(plan.teams), limits.max_teams, "max_teams"),
        ("/teams", _peak(plan), limits.max_agents_per_goal, "max_agents_per_goal"),
        ("/rounds", len(plan.rounds), limits.max_rounds, "max_rounds"),
        *[
            (p + "/agents", len(t.agents), limits.max_agents_per_team, "max_agents_per_team")
            for p, t in _teams(plan)
        ],
    ]:
        if value > limit:
            errors.append(f"{pointer}: {field}: {value} > {limit}")
    return errors


def _rule_max_agents_total(
    plan: Plan, config: RunConfig, history: Sequence[Plan], *, other_active_agents: int
) -> list[str]:
    peak = _peak(plan)
    limit = config.limits.max_agents_total
    if peak + other_active_agents > limit:
        return [
            f"max_agents_total: peak {peak} agents + {other_active_agents}"
            f" active in other goals > {limit}"
        ]
    return []


def _goal_key_errors(plans: Sequence[Plan]) -> list[str]:
    """Refuse ambiguous board groups using only goal and part identifiers."""
    seen: dict[GoalKey, str] = {}
    errors: list[str] = []
    for plan in plans:
        groups = [(f"goal {plan.goal.id!r}", plan.goal)]
        groups.extend(
            (f"goal {plan.goal.id!r} part {part.name!r}", part.goal) for part in plan.parts
        )
        for group, goal in groups:
            key = _safe_call("/goal", partial(GoalKey.of, goal))
            if key in seen:
                errors.append(f"/parts: duplicate GoalKey for {seen[key]} and {group}")
            else:
                seen[key] = group
    return errors


def _rule_parts(plan: Plan, config: RunConfig, history: Sequence[Plan]) -> list[str]:
    errors = _goal_key_errors([plan])
    names = {p.name for p in plan.parts}
    for pointer, team in _teams(plan):
        if team.part is not None and team.part not in names:
            errors.append(f"{pointer}/part: unknown part {team.part!r}")
    assigned = {t.part for t in plan.teams}
    for i, part in enumerate(plan.parts):
        if part.name not in assigned:
            errors.append(f"/parts/{i}: part {part.name!r} has no team")
    if plan.parts and plan.converge.mode != "team":
        errors.append("/converge/mode: parts need converge mode 'team'")
    if plan.converge.mode == "team" and plan.converge.team is None:
        errors.append("/converge/team: converge mode 'team' requires a team")
    return errors


def _rule_main_goal(plan: Plan, config: RunConfig, history: Sequence[Plan]) -> list[str]:
    key = _safe_call("/goal", lambda: GoalKey.of(plan.goal))
    if history and key != _safe_call("/goal", lambda: GoalKey.of(history[0].goal)):
        return ["/goal: main goal differs from the frozen main goal"]
    return []


def _ref_name(ref: Ref, pointer: str) -> str:
    return _safe_call(pointer, ref.component_name)


def _scoring_names(refs: Sequence[Ref], pointer: str, kind: str) -> tuple[list[str], list[str]]:
    names: list[str] = []
    errors: list[str] = []
    for i, ref in enumerate(refs):
        try:
            names.append(_ref_name(ref, f"{pointer}/{i}"))
        except PlanError as error:
            errors.extend(error.errors)
    if len(set(names)) != len(names):
        errors.append(f"{pointer}: duplicate {kind} names")
    return names, errors


def _rule_scoring(plan: Plan, config: RunConfig, history: Sequence[Plan]) -> list[str]:
    errors: list[str] = []
    scopes = [("", plan.checks, plan.judges, plan.ranking)]
    scopes += [
        (
            f"/parts/{i}",
            plan.checks if p.checks is None else p.checks,
            plan.judges if p.judges is None else p.judges,
            plan.ranking if p.ranking is None else p.ranking,
        )
        for i, p in enumerate(plan.parts)
    ]
    for pointer, checks, judges, ranking in scopes:
        _, problems = _scoring_names(checks, pointer + "/checks", "check")
        errors.extend(problems)
        names, problems = _scoring_names(judges, pointer + "/judges", "judge")
        errors.extend(problems)
        if ranking and ranking not in names:
            errors.append(f"{pointer}/ranking: {ranking!r} does not name a declared judge")
        elif not ranking and len(names) != 1:
            errors.append(f"{pointer}/ranking: ranking requires one judge or an explicit name")
    return errors


def _same_refs(left: Sequence[Ref] | None, right: Sequence[Ref] | None) -> bool:
    return json_equal(
        None if left is None else [r.model_dump(mode="json") for r in left],
        None if right is None else [r.model_dump(mode="json") for r in right],
    )


def _rule_retained_scoring(plan: Plan, config: RunConfig, history: Sequence[Plan]) -> list[str]:
    if plan.origin == "human" or not history:
        return []
    errors: list[str] = []
    for field, kind in [("checks", "check"), ("judges", "judge")]:
        first: dict[str, Ref] = {}
        for old in history:
            for ref in getattr(old, field):
                first.setdefault(_ref_name(ref, f"/{field}"), ref)
        current = {_ref_name(ref, f"/{field}"): ref for ref in getattr(plan, field)}
        for name, ref in first.items():
            if name not in current or not json_equal(
                ref.model_dump(mode="json"), current[name].model_dump(mode="json")
            ):
                errors.append(f"/{field}: {kind} {name!r} must be retained unchanged")
        if field == "judges" and current.keys() - first.keys():
            errors.append("/judges: non-human revisions may add checks only")
    if any(plan.ranking != old.ranking for old in history):
        errors.append("/ranking: ranking must be retained unchanged")
    old_parts = {p.name: p for p in history[-1].parts}
    for i, part in enumerate(plan.parts):
        old_part = old_parts.get(part.name)
        for field in ("checks", "judges", "ranking"):
            value = getattr(part, field)
            if old_part is None:
                if value is not None:
                    errors.append(f"/parts/{i}/{field}: new part must leave scoring unset")
            else:
                previous = getattr(old_part, field)
                equal = (
                    json_equal(value, previous)
                    if field == "ranking"
                    else _same_refs(value, previous)
                )
                if not equal:
                    errors.append(f"/parts/{i}/{field}: retained part scoring must not change")
    return errors


def _addresses(plan: Plan) -> set[str]:
    return {
        address
        for _, team in _teams(plan)
        for address in [team.name, *(f"{team.name}/{a.name}" for a in team.agents)]
    }


def _rule_addresses(plan: Plan, config: RunConfig, history: Sequence[Plan]) -> list[str]:
    retired: set[str] = set()
    previous: set[str] = set()
    for old in history:
        current = _addresses(old)
        retired.update(previous - current)
        previous = current
    return [
        f"/teams: address {address} must not be reused"
        for address in sorted(retired & _addresses(plan))
    ]


def _overrides(plan: Plan, site: RefSite) -> dict[str, Any]:
    pointer, sep, target = site.pointer.partition("@")
    if not sep or "/config/" in target:
        return {}
    if pointer.startswith("/rounds/"):
        return plan.rounds[int(pointer.split("/")[2])].tool_overrides
    return {} if plan.converge.round is None else plan.converge.round.tool_overrides


def _rule_configs(
    plan: Plan, config: RunConfig, history: Sequence[Plan], *, built: dict[str, Configurable]
) -> list[str]:
    errors: list[str] = []
    sites = ref_sites(plan)
    pointers = _site_pointers(sites)
    for site in sites:
        pointer = pointers[site.pointer]
        base = Configurable if "/config/" in site.pointer else _BASES[site.kind]
        ref = site.ref
        try:
            cls = ref.load()
            if plan.origin != "human":
                protected = _human_only_inputs(cls)
                for entry in config.planner_catalog.entries:
                    if entry.kind == site.kind and entry.ref == ref.path:
                        for key in entry.settable.keys() & ref.config.keys() & protected.keys():
                            errors.append(
                                f"{pointer}: {ref.path}: human-only field "
                                f"{protected[key]} cannot be settable"
                            )
            if base is Tool:
                ref = effective_tool_ref(ref, _overrides(plan, site))
            else:
                ref._resolve_config(base)
        except ConfigError as error:
            errors.append(f"{pointer}: {error}")
            continue
        except Exception as error:
            errors.append(f"{pointer}: {ref.path}: {type(error).__name__}")
            continue
        try:
            built[site.pointer] = _safe_call(f"{pointer}: {ref.path}", partial(ref.build, base))
        except PlanError as error:
            errors.extend(error.errors)
    return errors


def _env_names(component: Configurable) -> list[tuple[str, str]]:
    """Inspect typed nested defaults as well as explicitly written references."""
    result: list[tuple[str, str]] = []

    def visit(value: Any, pointer: str, *, tool_config: bool = False) -> None:
        if isinstance(value, Ref):
            cls, cfg = value._resolve_config(Configurable)
            visit(cfg, pointer, tool_config=issubclass(cls, Tool))
        elif isinstance(value, BaseModel):
            if tool_config:
                names = getattr(value, "env", ())
                if not isinstance(names, (list, tuple)) or any(
                    not isinstance(name, str)
                    or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None
                    for name in names
                ):
                    raise ValueError("invalid env names")
                result.extend((pointer + "/env", name) for name in names)
            for field in type(value).model_fields:
                visit(getattr(value, field), pointer + "/config/" + field)
        elif is_nested_ref(value):
            visit(Ref.model_validate(value), pointer)
        elif isinstance(value, dict):
            for child in value.values():
                # Arbitrary mapping keys are not safe diagnostic identifiers.
                visit(child, pointer + "/<key>")
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, pointer + f"/{index}")

    visit(component.config, "", tool_config=isinstance(component, Tool))
    return result


def _rule_env(
    plan: Plan, config: RunConfig, history: Sequence[Plan], *, built: dict[str, Configurable]
) -> list[str]:
    errors: list[str] = []
    for pointer, component in built.items():
        if "/config/" not in pointer:
            try:
                requested = _safe_call(pointer, partial(_env_names, component))
            except PlanError as error:
                errors.extend(error.errors)
                continue
            for suffix, name in requested:
                if name not in config.secrets.allow:
                    errors.append(f"{pointer}{suffix}: env name {name!r} is not in secrets.allow")
    return errors


def _rule_replicates(
    plan: Plan, config: RunConfig, history: Sequence[Plan], *, built: dict[str, Configurable]
) -> list[str]:
    errors: list[str] = []
    for pointer, agent in _agents(plan):
        if agent.replicates <= 1:
            continue
        tool_pointer = pointer + "/tool"
        contexts: list[tuple[str, int, dict[str, Any]]] = []
        if pointer.startswith("/converge/"):
            settings = plan.converge.round
            context = (
                tool_pointer
                if settings is None
                else "/converge/round/tool_overrides@" + tool_pointer
            )
            contexts.append(
                (
                    context,
                    plan.rounds_total(config.limits) + 1,
                    {} if settings is None else settings.tool_overrides,
                )
            )
        elif plan.rounds:
            contexts.extend(
                (f"/rounds/{i}/tool_overrides@{tool_pointer}", i + 1, settings.tool_overrides)
                for i, settings in enumerate(plan.rounds)
            )
        else:
            contexts.extend((tool_pointer, i + 1, {}) for i in range(config.limits.max_rounds))
        for context_pointer, round_number, overrides in contexts:
            tool = built.get(context_pointer)
            if not isinstance(tool, Tool):
                continue
            deterministic = _safe_call(
                context_pointer,
                partial(
                    tool.is_deterministic,
                    ToolRequest(round=round_number, round_overrides=overrides),
                ),
            )
            if deterministic:
                errors.append(
                    f"{pointer}/replicates: deterministic tool cannot have replicates"
                    f" ({context_pointer})"
                )
    return errors


def agent_fingerprint(spec: AgentSpec, manager: Manager) -> str:
    """
    Hash the base tool path, raw settings, and manager model aliases.

    Parameters
    ----------
    spec : AgentSpec
        Agent whose tool configuration participates in duplicate admission.
    manager : Manager
        Built manager supplying model aliases through ``models()``.

    Returns
    -------
    str
        SHA-256 fingerprint independent of names, rationales and instructions.

    Examples
    --------
    >>> from swarm_arena._agent.manager import Manager
    >>> spec = AgentSpec(name="a", rationale="Explore", tool=Ref(path="demo:Tool"),
    ...                  manager=Ref(path="demo:Manager"))
    >>> class Quiet(Manager):
    ...     async def run_round(self, ctx):
    ...         raise NotImplementedError
    >>> len(agent_fingerprint(spec, Quiet()))
    64
    """
    return sha256_hex(
        canonical_json(
            {"tool": spec.tool.path, "config": spec.tool.config, "models": list(manager.models())}
        )
    )


def _rule_fingerprints(
    plan: Plan, config: RunConfig, history: Sequence[Plan], *, built: dict[str, Configurable]
) -> list[str]:
    counts: Counter[str] = Counter()
    errors: list[str] = []
    for pointer, agent in _agents(plan):
        manager = built.get(pointer + "/manager")
        if isinstance(manager, Manager):
            try:
                counts[
                    _safe_call(pointer + "/manager", partial(agent_fingerprint, agent, manager))
                ] += 1
            except PlanError as error:
                errors.extend(error.errors)
    if any(count > config.limits.max_agents_per_fingerprint for count in counts.values()):
        errors.append("/teams: max_agents_per_fingerprint exceeded")
    return errors


def _file_sets(value: Any) -> Iterator[FileSet]:
    """Find typed file declarations, including defaults and nested configurations."""
    if isinstance(value, FileSet):
        yield value
    elif isinstance(value, Ref):
        _, cfg = value._resolve_config(Configurable)
        yield from _file_sets(cfg)
    elif isinstance(value, BaseModel):
        for field in type(value).model_fields:
            yield from _file_sets(getattr(value, field))
    elif is_nested_ref(value):
        yield from _file_sets(Ref.model_validate(value))
    elif isinstance(value, dict):
        for child in value.values():
            yield from _file_sets(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _file_sets(child)


def _file_paths(value: Any) -> set[str]:
    """Collect frozen paths through the shared typed-file traversal."""
    return {path for files in _file_sets(value) for path in files.manifest}


def judge_private_paths(plan: Plan) -> frozenset[str]:
    """
    Collect check and judge file paths without changing their spellings.

    Parameters
    ----------
    plan : Plan
        Snapshotted plan, including explicit part scoring and nested refs.

    Returns
    -------
    frozenset[str]
        Raw workspace-relative spellings to protect from tools and goals.

    Raises
    ------
    PlanError
        If a scoring configuration cannot be inspected safely.

    Examples
    --------
    >>> judge_private_paths(Plan(goal=Goal(id="g", namespace="demo"), teams=[], judges=[]))
    frozenset()
    """
    paths: set[str] = set()
    errors: list[str] = []
    roots = [("/checks", plan.checks), ("/judges", plan.judges)]
    roots += [
        (f"/parts/{i}/{field}", getattr(part, field) or [])
        for i, part in enumerate(plan.parts)
        for field in ("checks", "judges")
    ]
    for pointer, refs in roots:
        for i, ref in enumerate(refs):
            try:
                paths.update(_safe_call(f"{pointer}/{i}: {ref.path}", partial(_file_paths, ref)))
            except PlanError as error:
                errors.extend(error.errors)
    if errors:
        raise _core_error(PlanError, errors) from None
    return frozenset(paths)


def _tool_goal(plan: Plan, pointer: str) -> Goal:
    target = pointer.split("@")[-1]
    if target.startswith("/teams/"):
        team = plan.teams[int(target.split("/")[2])]
        return next((p.goal for p in plan.parts if p.name == team.part), plan.goal)
    return plan.goal


def _rule_private_files(
    plan: Plan,
    config: RunConfig,
    history: Sequence[Plan],
    *,
    built: dict[str, Configurable],
    private: frozenset[str],
) -> list[str]:
    folded = {fold_key(path) for path in private}
    pointers = _site_pointers(ref_sites(plan))
    errors: list[str] = []

    def check(paths: Sequence[str], pointer: str) -> None:
        errors.extend(
            f"{pointer}: file {path!r} is private to a check or judge"
            for path in paths
            if fold_key(path) in folded
        )

    check(list(plan.goal.files.manifest), "/goal/files")
    for i, part in enumerate(plan.parts):
        check(list(part.goal.files.manifest), f"/parts/{i}/goal/files")
    for pointer, tool in built.items():
        if not isinstance(tool, Tool):
            continue
        try:
            diagnostic = pointers[pointer]
            own = _safe_call(diagnostic, tool._own_files)
            check(list(own), diagnostic)
            _safe_call(
                diagnostic, partial(tool_root, _tool_goal(plan, pointer).files.manifest, own)
            )
        except PlanError as error:
            errors.extend(error.errors)
    return errors


def _rule_part_files(plan: Plan, config: RunConfig, history: Sequence[Plan]) -> list[str]:
    if plan.origin == "human":
        return []
    main = plan.goal.files.manifest
    return [
        f"/parts/{i}/goal/files: file {path!r} is not an exact entry of the main goal manifest"
        for i, part in enumerate(plan.parts)
        for path, digest in part.goal.files.manifest.items()
        if main.get(path) != digest
    ]


def register_model_check(check: ModelCheck) -> None:
    """
    Register an optional model admission hook.

    Parameters
    ----------
    check : ModelCheck
        Callable accepting plan and config and returning errors and warnings.
        Hook diagnostics must contain identifiers only, never secret values.

    Examples
    --------
    >>> before = len(_MODEL_CHECKS)
    >>> register_model_check(lambda plan, config: ([], []))
    >>> len(_MODEL_CHECKS) == before + 1
    True
    >>> _ = _MODEL_CHECKS.pop()
    """
    _MODEL_CHECKS.append(check)


def validate_plan(
    plan: Plan, config: RunConfig, *, history: Sequence[Plan] = (), other_active_agents: int = 0
) -> list[str]:
    """
    Validate plan admission and return optional model warnings.

    Import-free raw authorization runs before typed catalog-default checks.
    Only catalog-matched Configs may resolve omitted FileSet defaults. The same
    two steps run in admission before snapshotting. Subsequent rule errors are
    collected. File identities must already be snapshotted; the engine uses
    ``admit_revision`` to resolve them first.

    Parameters
    ----------
    plan : Plan
        Revision to validate, with core-assigned provenance.
    config : RunConfig
        Human-controlled limits, planner catalog, and permitted secret names.
    history : Sequence[Plan], default=()
        Admitted revisions in chronological order, starting with r0.
    other_active_agents : int, default=0
        Active agents belonging to other goals, added to this plan's peak.

    Returns
    -------
    list[str]
        Warnings from registered model checks when every admission rule passes.

    Raises
    ------
    PlanError
        If raw authorization fails or any admission rule rejects the revision.
        Configuration and connector errors omit secret-bearing values.

    Examples
    --------
    >>> from swarm_arena import Goal, Plan, ProgramJudge, RunConfig, validate_plan
    >>> plan = Plan(goal=Goal(id="g", namespace="demo"), teams=[],
    ...             judges=[ProgramJudge("my_judges:quality")])
    >>> validate_plan(plan, RunConfig())
    []
    """
    _authorize(plan, config, history)
    # Validation inspects the supplied snapshot; authorization's cleaned copy is
    # consumed only by admission, which can resolve newly introduced FileSets.
    plan = _inherited_converge(plan)
    errors: list[str] = []
    warnings: list[str] = []
    for rule in (
        _rule_names,
        _rule_limits,
        _rule_rationales,
        _rule_parts,
        _rule_main_goal,
        _rule_scoring,
        _rule_retained_scoring,
        _rule_addresses,
        _rule_part_files,
    ):
        try:
            errors.extend(rule(plan, config, history))
        except PlanError as error:
            errors.extend(error.errors)
    errors.extend(
        _rule_max_agents_total(plan, config, history, other_active_agents=other_active_agents)
    )
    built: dict[str, Configurable] = {}
    errors.extend(_rule_configs(plan, config, history, built=built))
    try:
        _validate_frozen_files(plan)
    except PlanError as error:
        errors.extend(error.errors)
    for rule in (_rule_env, _rule_replicates, _rule_fingerprints):
        try:
            errors.extend(rule(plan, config, history, built=built))
        except PlanError as error:
            errors.extend(error.errors)
    try:
        private = judge_private_paths(plan)
        errors.extend(_rule_private_files(plan, config, history, built=built, private=private))
    except PlanError as error:
        errors.extend(error.errors)
    for hook in _MODEL_CHECKS:
        try:
            hook_errors, hook_warnings = _safe_call("/models", partial(hook, plan, config))
            errors.extend(hook_errors)
            warnings.extend(hook_warnings)
        except PlanError as error:
            errors.extend(error.errors)
    if errors:
        raise _core_error(PlanError, errors) from None
    return warnings


def _resolve_revision(child: Plan, r0: Plan, artifacts: ArtifactStore) -> Plan:
    """Resolve planner part globs exclusively within the frozen main manifest."""
    initial = child is r0
    child = child.model_copy(deep=True)
    child._base_dir = r0.base_dir
    main = snapshot_fileset(r0.goal.files, r0.base_dir, artifacts)
    if initial:
        child = child.model_copy(update={"goal": child.goal.model_copy(update={"files": main})})
    elif not child.goal.files.snapshotted:
        # The main goal itself is frozen and is checked by GoalKey below.
        child = child.model_copy(
            update={
                "goal": child.goal.model_copy(
                    update={"files": snapshot_fileset(child.goal.files, r0.base_dir, artifacts)}
                )
            }
        )
    parts = []
    for part in child.parts:
        files = part.goal.files
        if not files.snapshotted:
            for glob in files.globs:
                if not any(matches(path, [glob]) for path in main.manifest):
                    raise _core_error(
                        PlanError,
                        [f"part {part.name!r}: glob {glob!r} matches no file of the main goal"],
                    ) from None
            files = FileSet(
                globs=files.globs,
                manifest={
                    path: digest
                    for path, digest in main.manifest.items()
                    if matches(path, files.globs)
                },
            )
        parts.append(
            part.model_copy(update={"goal": part.goal.model_copy(update={"files": files})})
        )
    return child.model_copy(update={"parts": parts})


def _round_refs(plan: Plan) -> Iterator[tuple[str, Ref, dict[str, Any]]]:
    """Yield effective roots, retaining the original shared override storage."""
    projected = _inherited_converge(plan)
    for site in ref_sites(projected):
        pointer, separator, target = site.pointer.partition("@")
        if separator and "/config/" not in target:
            if pointer.startswith("/converge/") and plan.converge.round is None:
                pointer = f"/rounds/{len(plan.rounds) - 1}/tool_overrides"
            yield pointer, site.ref, _overrides(projected, site)


def _validate_round_refs(plan: Plan) -> None:
    """Validate merged raw overrides before any source files are snapshotted."""
    errors: list[str] = []
    for pointer, ref, overrides in _round_refs(plan):
        try:
            effective_tool_ref(ref, overrides)
        except ConfigError as error:
            errors.append(f"{pointer}: {error}")
        except Exception as error:
            errors.append(f"{pointer}: {ref.path}: {type(error).__name__}")
    if errors:
        raise _core_error(PlanError, errors) from None


def _snapshot_round_refs(plan: Plan, artifacts: ArtifactStore) -> Plan:
    """Freeze override FileSets without replacing the agents' base refs."""
    updates: dict[str, dict[str, Any]] = {}
    for pointer, ref, overrides in _round_refs(plan):
        frozen = snapshot_ref(effective_tool_ref(ref, overrides), plan.base_dir, artifacts)
        values = {key: frozen.config[key] for key in overrides}
        if pointer in updates and not json_equal(updates[pointer], values):
            raise _core_error(
                PlanError, [f"{pointer}: inconsistent snapshots across effective tools"]
            )
        updates[pointer] = values
    rounds = [
        round_spec.model_copy(
            update={
                "tool_overrides": updates.get(
                    f"/rounds/{i}/tool_overrides", round_spec.tool_overrides
                )
            }
        )
        for i, round_spec in enumerate(plan.rounds)
    ]
    converge = plan.converge
    if converge.round is not None:
        converge = converge.model_copy(
            update={
                "round": converge.round.model_copy(
                    update={
                        "tool_overrides": updates.get(
                            "/converge/round/tool_overrides", converge.round.tool_overrides
                        )
                    }
                )
            }
        )
    return plan.model_copy(update={"rounds": rounds, "converge": converge})


def _validate_frozen_files(plan: Plan) -> None:
    """Refuse unfrozen files in every goal, component root and effective tool."""
    roots: list[tuple[str, Any]] = [("/goal/files", plan.goal.files)]
    roots.extend((f"/parts/{i}/goal/files", part.goal.files) for i, part in enumerate(plan.parts))
    roots.extend(
        (site.pointer, site.ref)
        for site in ref_sites(_inherited_converge(plan))
        if "/config/" not in site.pointer
    )
    errors: list[str] = []
    for pointer, value in roots:
        try:
            files = _safe_call(pointer, partial(list, _file_sets(value)))
            if any(not file_set.snapshotted for file_set in files):
                errors.append(f"{pointer}: contains an unsnapshotted FileSet")
        except PlanError as error:
            errors.extend(error.errors)
    if errors:
        raise _core_error(PlanError, errors) from None


def admit_revision(
    plan: Plan,
    config: RunConfig,
    artifacts: ArtifactStore,
    *,
    history: Sequence[Plan] = (),
    other_active_agents: int = 0,
) -> tuple[Plan, list[str]]:
    """
    Authorize, snapshot, and validate one revision before execution.

    Refuse any goal or component configuration whose file declarations, including
    nested references and defaults, remain unsnapshotted after admission.

    Parameters
    ----------
    plan : Plan
        Human plan or core-stamped planner revision.
    config : RunConfig
        Human-controlled admission settings.
    artifacts : ArtifactStore
        Destination for newly snapshotted source files.
    history : Sequence[Plan], default=()
        Admitted revisions, with r0's original base directory restored on resume.
    other_active_agents : int, default=0
        Active agents in other goals.

    Returns
    -------
    tuple[Plan, list[str]]
        Snapshotted revision and model-check warnings.

    Raises
    ------
    PlanError
        If authorization, snapshot resolution, or admission fails.

    Notes
    -----
    This synchronous boundary performs file I/O; async callers offload it.
    Non-human component manifests are retained only through admitted history.
    Without history, even catalog-default files are snapshotted again from the
    human source directory; missing sources fail admission. A human may vouch
    for a saved snapshot by marking its origin as ``"human"``.

    Examples
    --------
    >>> from swarm_arena import ProgramJudge
    >>> from swarm_arena._store.base import open_store
    >>> plan = Plan(goal=Goal(id="g", namespace="demo"), teams=[],
    ...             judges=[ProgramJudge("my_judges:quality")])
    >>> artifacts = ArtifactStore(open_store("artifacts"))
    >>> frozen, warnings = admit_revision(plan, RunConfig(), artifacts)
    >>> warnings, frozen.goal.files.snapshotted
    ([], True)
    """
    authorized = _authorize(plan, config, history)
    _validate_round_refs(authorized)
    try:
        if plan.origin != "human":
            authorized = _resolve_revision(
                authorized, history[0] if history else authorized, artifacts
            )
        frozen = snapshot_plan(authorized, artifacts)
        frozen = _snapshot_round_refs(frozen, artifacts)
    except PlanError:
        raise
    except Exception as error:
        raise _core_error(PlanError, [f"/snapshot: {type(error).__name__}"]) from None
    warnings = validate_plan(
        frozen, config, history=history, other_active_agents=other_active_agents
    )
    return frozen, warnings
