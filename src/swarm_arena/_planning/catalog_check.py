"""Authorize raw planner references before component imports or file snapshots."""

from collections.abc import Mapping, Sequence
from typing import Any, get_args

from swarm_arena._errors import ConfigError
from swarm_arena._model.base import CatalogKind
from swarm_arena._model.catalog import CatalogEntry, PlannerCatalog, _field_input_keys
from swarm_arena._model.data import json_equal
from swarm_arena._model.goal import FileSet
from swarm_arena._model.plan import Plan
from swarm_arena._model.ref import Configurable, Ref, _validate_config
from swarm_arena._model.run_config import RunConfig
from swarm_arena._planning.provenance import (
    RefSite,
    diff_revision,
    is_nested_ref,
    ref_sites,
    site_key,
)

_ManifestPair = tuple[tuple[str, ...], tuple[tuple[str, str], ...]]


def _is_fileset(value: Any) -> bool:
    """Recognize dumped file sets without mistaking extended data for snapshots."""
    return (
        isinstance(value, dict)
        and value.keys() == {"globs", "manifest"}
        and isinstance(value["globs"], list)
        and all(isinstance(glob, str) for glob in value["globs"])
        and isinstance(value["manifest"], dict)
        and all(isinstance(digest, str) for digest in value["manifest"].values())
    )


def _catalog_value(value: Any) -> Any:
    """Normalize only snapshot representations and nested reference defaults."""
    if _is_fileset(value):
        return list(value["globs"])
    if is_nested_ref(value):
        return {
            "path": value["path"],
            "config": catalog_form(value.get("config") or {}),
            "name": value.get("name") or None,
        }
    if isinstance(value, dict):
        return catalog_form(value)
    if isinstance(value, list):
        return [_catalog_value(item) for item in value]
    return value


def catalog_form(config: Mapping[str, Any]) -> dict[str, Any]:
    """
    Normalize configuration representations for import-free catalog comparison.

    Parameters
    ----------
    config : Mapping[str, Any]
        Raw or snapshotted component configuration.

    Returns
    -------
    dict[str, Any]
        Independent configuration with dumped FileSets replaced by globs and
        nested Ref defaults made explicit, recursively. Scalar types remain
        unchanged; compare results with ``json_equal``.

    Examples
    --------
    >>> catalog_form({"files": {"globs": ["run.py"], "manifest": {}}})
    {'files': ['run.py']}
    >>> catalog_form({"child": {"path": "demo:Child"}})
    {'child': {'path': 'demo:Child', 'config': {}, 'name': None}}
    """
    return {key: _catalog_value(value) for key, value in config.items()}


def _check_catalog_defaults(
    deferred: Sequence[tuple[RefSite, CatalogEntry, tuple[str, ...]]],
) -> tuple[list[str], set[_ManifestPair]]:
    """Decide raw deferrals using only matched human Configs, without file I/O.

    A matching default declaration also preserves its supplied frozen manifest
    on reload, independently of revision history. All other manifests retain
    the ordinary history-based trust rule.
    """
    errors: list[str] = []
    trusted: set[_ManifestPair] = set()
    for site, entry, fields in deferred:
        try:
            component = Ref(path=entry.ref).load()
            schema = component.Config if issubclass(component, Configurable) else None
        except Exception:
            schema = None
        for key in fields:
            raw = site.ref.config[key]
            accepted = False
            try:
                field = schema.model_fields.get(key) if schema is not None else None
                if (
                    field is not None
                    and _field_input_keys(key, field).isdisjoint(entry.config)
                    and any(
                        isinstance(candidate, type) and issubclass(candidate, FileSet)
                        for candidate in (field.annotation, *get_args(field.annotation))
                    )
                ):
                    value = field.get_default(call_default_factory=True)
                    accepted = isinstance(value, FileSet) and json_equal(
                        catalog_form({key: raw}),
                        catalog_form({key: value.model_dump(mode="json")}),
                    )
            except Exception:
                accepted = False
            if not accepted:
                errors.append(f"{site.pointer}: field '<field>' is not in the planner catalog")
            elif _is_fileset(raw):
                trusted.add(_manifest_pair(raw))
    return errors, trusted


def find_entry(
    ref: Ref,
    kind: CatalogKind,
    catalog: PlannerCatalog,
    *,
    raw: bool = False,
    defer_defaults: bool = False,
) -> CatalogEntry | None:
    """
    Find the first fixed-config match, including approved FileSet defaults.

    Parameters
    ----------
    ref : Ref
        Raw or snapshotted base component reference, including human refs.
    kind : str
        Required component role.
    catalog : PlannerCatalog
        Human-approved catalog, searched in configured order.
    raw : bool, default=False
        Use import-free comparison for raw authorization only. Otherwise extra
        fields must match their omitted typed FileSet defaults, using the same
        decision as admission. Only already matched human paths may be loaded.
    defer_defaults : bool, default=False
        In raw mode, ignore extra fixed fields for candidate selection only.
        The caller must collect them with ``check_catalog`` and check their
        typed defaults before resolving or snapshotting references.

    Returns
    -------
    CatalogEntry or None
        First same-kind/path entry whose fixed configuration matches after
        normalization and removal of that entry's top-level settable keys
        from the reference. Matching preserves scalar types at every depth.
        Other extra fields are never ignored in the default mode.

    Examples
    --------
    >>> entry = CatalogEntry(name="tool", kind="tool", ref="demo:Tool", description="d")
    >>> catalog = PlannerCatalog(entries=[entry])
    >>> find_entry(Ref(path="demo:Tool"), "tool", catalog) is entry
    True
    """
    for entry in catalog.entries:
        if entry.kind == kind and entry.ref == ref.path:
            fixed = catalog_form(
                {key: value for key, value in ref.config.items() if key not in entry.settable}
            )
            expected = catalog_form(entry.config)
            extra = tuple(sorted(fixed.keys() - expected.keys()))
            compared = (
                {key: value for key, value in fixed.items() if key in expected}
                if not raw or defer_defaults
                else fixed
            )
            if not json_equal(compared, expected):
                continue
            if raw or not extra:
                return entry
            site = RefSite(pointer="/ref", kind=kind, ref=ref)
            if not _check_catalog_defaults([(site, entry, extra)])[0]:
                return entry
    return None


def check_catalog(
    site: RefSite,
    catalog: PlannerCatalog,
    *,
    overrides: Mapping[str, Any],
    deferred: list[tuple[RefSite, CatalogEntry, tuple[str, ...]]] | None = None,
) -> list[str]:
    """
    Check fixed fields and planner-settable values without resolving references.

    Parameters
    ----------
    site : RefSite
        Raw reference occurrence, possibly an effective tool override site.
    catalog : PlannerCatalog
        Human-approved component configurations and override schemas.
    overrides : Mapping[str, Any]
        Explicit round overrides; even an unchanged fixed value is forbidden.
    deferred : list or None, default=None
        Optional collector for omitted extra fields at catalog-matched paths.
        These require a typed default decision before resolution or snapshotting.
        Without a collector, extra fields are reported immediately.

    Returns
    -------
    list[str]
        Value-free errors prefixed by the site's pointer. Unknown planner keys
        render as ``<field>``; known catalog fields retain their names.

    Examples
    --------
    >>> site = RefSite(pointer="/policy", kind="policy", ref=Ref(path="demo:Policy"))
    >>> check_catalog(site, PlannerCatalog(), overrides={})
    ["/policy: demo:Policy is not in the planner catalog for kind 'policy'"]
    """
    entry = find_entry(site.ref, site.kind, catalog, raw=True, defer_defaults=deferred is not None)
    if entry is None:
        entry = next(
            (e for e in catalog.entries if e.kind == site.kind and e.ref == site.ref.path),
            None,
        )
    if entry is None:
        return [
            f"{site.pointer}: {site.ref.path} is not in the planner catalog for kind {site.kind!r}"
        ]

    errors: list[str] = []
    fixed = catalog_form({k: v for k, v in site.ref.config.items() if k not in entry.settable})
    expected = catalog_form(entry.config)
    for key, value in expected.items():
        if key not in fixed or not json_equal(fixed[key], value):
            errors.append(f"{site.pointer}: fixed field {key!r} differs from the planner catalog")
    extra = tuple(sorted(fixed.keys() - expected.keys()))
    if deferred is not None and extra:
        deferred.append((site, entry, extra))
    else:
        for key in extra:
            # Only the human catalog can authorize a diagnostic field name.
            label = key if key in entry.settable else "<field>"
            errors.append(f"{site.pointer}: field {label!r} is not in the planner catalog")
    for key in overrides:
        if key not in entry.settable:
            label = key if key in entry.config else "<field>"
            errors.append(f"{site.pointer}: override sets fixed field {label!r}")

    model = entry.settable_model()
    for values in (site.ref.config, overrides):
        selected = {key: value for key, value in values.items() if key in entry.settable}
        try:
            _validate_config(model, selected, where=site.pointer)
        except ConfigError as error:
            # _validate_config owns sanitization and the shared summary renderer.
            errors.append(str(error))
    return errors


def parent_human(history: Sequence[Plan]) -> dict[str, bool]:
    """
    Fold admitted history into the immediate parent's human-authority flags.

    Parameters
    ----------
    history : Sequence[Plan]
        Admitted revisions in chronological order. Human revisions grant all
        their sites authority; non-human revisions carry only unchanged sites.

    Returns
    -------
    dict[str, bool]
        Flags keyed by ``site_key`` for the last revision. Removed or changed
        sites lose their inherited authority, even if later restored. Raw
        override edits lose authority even when the effective Ref is unchanged.

    Examples
    --------
    >>> parent_human([])
    {}
    """
    flags: dict[str, bool] = {}
    parent: Plan | None = None
    for revision in history:
        if revision.origin == "human":
            flags = {site_key(site): True for site in ref_sites(revision)}
        else:
            flags = {
                site_key(site): site.carried_human
                for site in _revision_sites(parent, revision, flags)
            }
        parent = revision
    return flags


def _manifest_pair(value: dict[str, Any]) -> _ManifestPair:
    """Keep glob order and compare manifest mappings independently of key order."""
    return tuple(value["globs"]), tuple(sorted(value["manifest"].items()))


def _trusted_pairs(history: Sequence[Plan]) -> set[_ManifestPair]:
    """Collect frozen file pairs from every reference configuration in history."""
    trusted: set[_ManifestPair] = set()
    pending = [
        value for plan in history for site in ref_sites(plan) for value in site.ref.config.values()
    ]
    while pending:
        value = pending.pop()
        if _is_fileset(value):
            trusted.add(_manifest_pair(value))
        elif is_nested_ref(value):
            pending.extend(value.get("config", {}).values())
        elif isinstance(value, dict):
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
    return trusted


def _strip_manifests(value: Any, trusted: set[_ManifestPair]) -> Any:
    """Copy values, retaining a supplied manifest only if history holds its pair."""
    if _is_fileset(value) and _manifest_pair(value) not in trusted:
        return list(value["globs"])
    if is_nested_ref(value):
        return {
            key: (
                {field: _strip_manifests(item, trusted) for field, item in child.items()}
                if key == "config"
                else child
            )
            for key, child in value.items()
        }
    if isinstance(value, dict):
        return {key: _strip_manifests(child, trusted) for key, child in value.items()}
    if isinstance(value, list):
        return [_strip_manifests(child, trusted) for child in value]
    return value


def _at_pointer(plan: Plan, pointer: str) -> Any:
    """Resolve an actual RFC 6901 pointer within a copied plan without imports."""
    value: Any = plan
    for token in pointer.removeprefix("/").split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        if isinstance(value, dict):
            value = value[token]
        elif isinstance(value, list):
            value = value[int(token)]
        else:
            value = getattr(value, token)
    return value


def _override_pointer(pointer: str) -> tuple[str, bool] | None:
    """Locate synthetic effective-tool sites and distinguish their nested refs."""
    if pointer.startswith("/rounds/") or pointer.startswith("/converge/round/"):
        override, separator, tool = pointer.partition("@")
        if separator and override.endswith("/tool_overrides"):
            return override, tool.endswith("/tool") and "/config/" not in tool
    return None


def _revision_sites(parent: Plan | None, child: Plan, flags: Mapping[str, bool]) -> list[RefSite]:
    """Also require unchanged raw overrides before carrying effective-site authority."""
    sites = diff_revision(parent, child, parent_human=flags)
    if parent is not None:
        for index, site in enumerate(sites):
            override = _override_pointer(site.pointer)
            if site.carried_human and override is not None and override[1]:
                pointer = override[0]
                if not json_equal(_at_pointer(parent, pointer), _at_pointer(child, pointer)):
                    sites[index] = site.model_copy(update={"carried_human": False})
    return sites


def _diagnostic_pointers(sites: Sequence[RefSite], catalog: PlannerCatalog) -> dict[str, str]:
    """Retain configured nested pointer keys, hiding planner-invented locations."""
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
        entry = find_entry(parent.ref, parent.kind, catalog, raw=True) or next(
            (e for e in catalog.entries if e.ref == parent.ref.path and e.kind == parent.kind),
            None,
        )
        expected: Any = entry.config if entry is not None else {}
        actual: Any = parent.ref.config
        suffix = site.pointer[len(parent.pointer + "/config/") :].split("/")
        safe: list[str] = []
        for position, token in enumerate(suffix):
            key = token.replace("~1", "/").replace("~0", "~")
            if isinstance(actual, list):
                safe.append(token)
                expected = (
                    expected[int(key)]
                    if isinstance(expected, list) and int(key) < len(expected)
                    else None
                )
                actual = actual[int(key)]
            else:
                known = isinstance(expected, dict) and key in expected
                settable = position == 0 and entry is not None and key in entry.settable
                safe.append(token if known or settable else "<field>")
                expected = expected[key] if known else None
                actual = actual[key]
        pointers[site.pointer] = pointers[parent.pointer] + "/config/" + "/".join(safe)
    return pointers


def authorize_revision(
    plan: Plan,
    config: RunConfig,
    *,
    history: Sequence[Plan] = (),
    deferred: list[tuple[RefSite, CatalogEntry, tuple[str, ...]]] | None = None,
    _default_pairs: frozenset[_ManifestPair] = frozenset(),
) -> tuple[Plan, list[str]]:
    """
    Authorize a revision's raw references and discard untrusted frozen manifests.

    Parameters
    ----------
    plan : Plan
        Revision with core-assigned origin and provenance.
    config : RunConfig
        Human-controlled planner catalog.
    history : Sequence[Plan], default=()
        Previously admitted revisions in chronological order.
    deferred : list or None, default=None
        Collector for extra fields omitted by a matching catalog entry. Both
        admission and validation must decide these with the shared typed check
        after rejecting raw errors, before any resolution or snapshotting.
    _default_pairs : frozenset, default=frozenset()
        Frozen pairs approved by that typed default check for this invocation.
        Only the admission boundary supplies these on its second raw pass.

    Returns
    -------
    tuple[Plan, list[str]]
        Human input unchanged, or a deep copy with untrusted manifests replaced
        by globs in non-carried sites, plus catalog errors. Trusted pairs may
        come from any historical Ref site at any depth or from the typed default
        decision. No components are imported or built, and no files are read.
        Extra fields are deferred, not authorized, until the typed check passes.
        Diagnostic pointers redact
        unconfigured nested mapping keys as ``<field>``. Callers must reject
        errors before resolving components or snapshotting files.

    Examples
    --------
    >>> from swarm_arena import Goal
    >>> plan = Plan(goal=Goal(id="g", namespace="demo"), teams=[], judges=[])
    >>> authorized, errors = authorize_revision(plan, RunConfig())
    >>> authorized is plan and not errors
    True
    """
    if plan.origin == "human":
        return plan, []
    sites = _revision_sites(history[-1] if history else None, plan, parent_human(history))
    pointers = _diagnostic_pointers(sites, config.planner_catalog)
    trusted = _trusted_pairs(history) | _default_pairs
    deferred = [] if deferred is None else deferred
    result = plan.model_copy(deep=True)
    errors: list[str] = []
    for site in sites:
        if site.carried_human:
            continue
        override_site = _override_pointer(site.pointer)
        overrides = (
            _at_pointer(plan, override_site[0])
            if override_site is not None and override_site[1]
            else {}
        )
        diagnostic_site = site.model_copy(update={"pointer": pointers[site.pointer]})
        errors.extend(
            check_catalog(
                diagnostic_site,
                config.planner_catalog,
                overrides=overrides,
                deferred=deferred,
            )
        )
        if override_site is not None:
            target = _at_pointer(result, override_site[0])
        else:
            ref = _at_pointer(result, site.pointer)
            target = ref.config if isinstance(ref, Ref) else ref.get("config", {})
        cleaned = {key: _strip_manifests(value, trusted) for key, value in target.items()}
        target.clear()
        target.update(cleaned)
    return result, errors
