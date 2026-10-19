"""Define declarative plans, component specifications, and TOML loading."""

import tomllib
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal, Self, cast

from pydantic import (
    Field,
    PrivateAttr,
    StringConstraints,
    ValidationError,
    model_validator,
)

from swarm_arena._errors import ConfigError
from swarm_arena._model.base import JsonValue, Origin, StoredModel
from swarm_arena._model.canonical import canonical_json, sha256_hex
from swarm_arena._model.goal import Goal, _Slug
from swarm_arena._model.limits import Limits, _SafeModel, _sanitize_validation
from swarm_arena._model.ref import Ref, RefField, unique_names
from swarm_arena._model.toml_io import dumps_toml, read_toml

_Rationale = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Resources(_SafeModel):
    """
    Declare the resources required to run one agent replica.

    Parameters
    ----------
    cpus : float, default=1.0
        Positive CPU capacity, including fractional requests.
    gpus : float, default=0.0
        Nonnegative GPU capacity.
    memory_gib : float or None, default=None
        Positive memory request in GiB, if specified.
    custom : dict[str, float], default={}
        Named resources with nonnegative capacity requests.

    Examples
    --------
    >>> from swarm_arena import Resources
    >>> Resources(cpus=0.5).gpus
    0.0
    """

    cpus: float = Field(default=1.0, gt=0)
    gpus: float = Field(default=0.0, ge=0)
    memory_gib: float | None = Field(default=None, gt=0)
    custom: dict[str, Annotated[float, Field(ge=0)]] = {}

    _sanitize_errors = model_validator(mode="wrap")(_sanitize_validation)


class PartSpec(_SafeModel):
    """
    Describe a sub-goal with optional scoring overrides.

    Parameters
    ----------
    name : str
        Lowercase slug identifying the part.
    goal : Goal
        Sub-goal to solve.
    checks : list[Ref] or None, default=None
        Check references, or None to inherit the plan's checks.
    judges : list[Ref] or None, default=None
        Judge references, or None to inherit the plan's judges.
    ranking : str or None, default=None
        Ranking judge name, or None to inherit the plan's ranking.

    Examples
    --------
    >>> from swarm_arena import Goal, PartSpec
    >>> PartSpec(name="part-a", goal=Goal(id="a", namespace="demo")).checks is None
    True
    """

    name: _Slug
    goal: Goal
    checks: list[RefField] | None = None
    judges: list[RefField] | None = None
    ranking: str | None = None

    _sanitize_errors = model_validator(mode="wrap")(_sanitize_validation)


class AgentSpec(_SafeModel):
    """
    Describe an agent's approach, components, and resource needs.

    Parameters
    ----------
    name : str
        Lowercase slug unique within the team.
    rationale : str
        Nonblank explanation of the agent's approach, stripped of outer whitespace.
    tool : Ref
        Tool reference, also accepting a component class or instance.
    manager : Ref
        Manager reference, also accepting a component class or instance.
    replicates : int, default=1
        Positive number of replicas to run.
    resources : Resources, default=Resources()
        Resource request for each replica.

    Examples
    --------
    >>> from swarm_arena import AgentSpec, Ref
    >>> AgentSpec(name="a", rationale="Explore", tool=Ref(path="demo:Tool"),
    ...           manager=Ref(path="demo:Manager")).replicates
    1
    """

    name: _Slug
    rationale: _Rationale
    tool: RefField
    manager: RefField
    replicates: int = Field(default=1, ge=1)
    resources: Resources = Resources()

    _sanitize_errors = model_validator(mode="wrap")(_sanitize_validation)


class TeamSpec(_SafeModel):
    """
    Group agents pursuing a shared approach to a goal or part.

    Parameters
    ----------
    name : str
        Lowercase slug unique within the plan.
    rationale : str
        Nonblank explanation of the team's approach, stripped of outer whitespace.
    part : str or None, default=None
        Declared part name, or None to work on the main goal.
    agents : list[AgentSpec]
        Agents assigned to the team.

    Examples
    --------
    >>> from swarm_arena import TeamSpec
    >>> TeamSpec(name="search", rationale="Explore alternatives", agents=[]).part is None
    True
    """

    name: _Slug
    rationale: _Rationale
    part: str | None = None
    agents: list[AgentSpec]

    _sanitize_errors = model_validator(mode="wrap")(_sanitize_validation)


class RoundSpec(_SafeModel):
    """
    Configure tool effort and fidelity for one round.

    Parameters
    ----------
    tool_overrides : dict[str, Any], default={}
        Tool settings applied to each agent during the round.
    tool_timeout_s : float or None, default=None
        Positive per-tool timeout in seconds, if set.
    fidelity : str or None, default=None
        Effort label, or a derived hash when absent or empty.

    Examples
    --------
    >>> from swarm_arena import RoundSpec
    >>> RoundSpec(fidelity="full").effective_fidelity()
    'full'
    """

    tool_overrides: dict[str, Any] = {}
    tool_timeout_s: float | None = Field(default=None, gt=0)
    fidelity: str | None = None

    def effective_fidelity(self) -> str:
        """
        Return the fidelity label or a canonical tool-settings hash prefix.

        Returns
        -------
        str
            Explicit nonempty label or the first twelve SHA-256 hex digits.

        Raises
        ------
        ValueError
            If tool overrides cannot be encoded as finite JSON.

        Examples
        --------
        >>> from swarm_arena import RoundSpec
        >>> len(RoundSpec(tool_overrides={"epochs": 100}).effective_fidelity())
        12
        """
        if self.fidelity:
            return self.fidelity
        try:
            encoded = canonical_json(cast(JsonValue, self.tool_overrides))
        except (TypeError, ValueError):
            raise ValueError("tool_overrides: expected finite JSON values") from None
        return sha256_hex(encoded)[:12]

    _sanitize_errors = model_validator(mode="wrap")(_sanitize_validation)


class ConvergeSpec(_SafeModel):
    """
    Configure final selection or a dedicated convergence team.

    Parameters
    ----------
    mode : {"select", "team"}, default="select"
        Select existing results or run a convergence team.
    top_n : int, default=2
        Positive number of leading candidates to retain.
    distinct : bool, default=True
        Require distinct result fingerprints.
    allow_partial : bool, default=False
        Permit partial candidates during convergence.
    min_improvement : float or None, default=None
        Optional improvement threshold.
    team : TeamSpec or None, default=None
        Dedicated convergence team, when using team mode.
    round : RoundSpec or None, default=None
        Convergence round settings; None inherits the final round's settings.
    parents_per_part : int, default=1
        Positive number of parent candidates selected from each part.
    max_join_attempts : int, default=1
        Nonnegative number of join attempts.

    Examples
    --------
    >>> from swarm_arena import ConvergeSpec
    >>> ConvergeSpec().mode
    'select'
    """

    mode: Literal["select", "team"] = "select"
    top_n: int = Field(default=2, ge=1)
    distinct: bool = True
    allow_partial: bool = False
    min_improvement: float | None = None
    team: TeamSpec | None = None
    round: RoundSpec | None = None
    parents_per_part: int = Field(default=1, ge=1)
    max_join_attempts: int = Field(default=1, ge=0)

    _sanitize_errors = model_validator(mode="wrap")(_sanitize_validation)


class FlowSpec(_SafeModel):
    """
    Describe a messaging edge reserved for a future protocol version.

    Parameters
    ----------
    sender : str
        Sender's address.
    recipient : str
        Recipient agent, team, or broadcast marker ``*``.
    rationale : str
        Reason for allowing this messaging edge.

    Examples
    --------
    >>> from swarm_arena.planning import FlowSpec
    >>> FlowSpec(sender="a", recipient="*", rationale="Share progress").recipient
    '*'
    """

    sender: str
    recipient: str
    rationale: str

    _sanitize_errors = model_validator(mode="wrap")(_sanitize_validation)


class Plan(StoredModel, _SafeModel):
    """
    Describe a versioned goal-solving plan without importing its components.

    Parameters
    ----------
    schema_version : int, default=1
        Stored plan schema version, upgraded during validation.
    goal : Goal
        Main goal to solve.
    parts : list[PartSpec], default=[]
        Named sub-goals with optional scoring overrides.
    teams : list[TeamSpec]
        Teams with unique names and unique agent names within each team.
    checks : list[Ref], default=[]
        Plan-level integrity check references.
    judges : list[Ref]
        Judge references used to evaluate candidates.
    ranking : str, default=""
        Ranking judge name; empty selects the only judge.
    policy : Ref, optional
        Pruning policy, defaulting to ``swarm_arena:SuccessiveHalving``.
    rounds : list[RoundSpec], default=[]
        Explicit rounds, or empty to use the limit's default round count.
    converge : ConvergeSpec, default=ConvergeSpec()
        Final selection or convergence settings. Its team follows the same
        structure rules and must have a name distinct from every regular team.
    sharing : {"round"}, default="round"
        Timing of shared information between agents.
    flows : list[FlowSpec], default=[]
        Reserved messaging edges; must be empty in v1.
    rationale : str, default=""
        Explanation of the overall plan.
    origin : {"human", "llm", "together", "autonomous"}, default="human"
        Plan provenance, preserved when loading from TOML.
    revision : int, default=0
        Plan revision number.
    parent_revision : int or None, default=None
        Prior revision from which this plan was derived.

    Examples
    --------
    >>> from swarm_arena import Goal, Plan, Ref
    >>> plan = Plan(goal=Goal(id="g", namespace="demo"), teams=[],
    ...             judges=[Ref(path="demo:Judge", name="score")])
    >>> plan.ranking_name()
    'score'
    """

    schema_kind: ClassVar[str] = "plan"
    goal: Goal
    parts: list[PartSpec] = []
    teams: list[TeamSpec]
    checks: list[RefField] = []
    judges: list[RefField]
    ranking: str = ""
    policy: RefField = Ref(path="swarm_arena:SuccessiveHalving")
    rounds: list[RoundSpec] = []
    converge: ConvergeSpec = ConvergeSpec()
    sharing: Literal["round"] = "round"
    flows: list[FlowSpec] = []
    rationale: str = ""
    origin: Origin = "human"
    revision: int = 0
    parent_revision: int | None = None
    _base_dir: Path = PrivateAttr(default_factory=Path.cwd)

    @model_validator(mode="after")
    def _structure(self) -> Self:
        """Enforce v1 messaging and scoped structural uniqueness."""
        if self.flows:
            raise ValueError("Plan.flows is reserved for v2 messaging and must be empty")
        if len({part.name for part in self.parts}) != len(self.parts):
            raise ValueError("duplicate part names")
        teams = [*self.teams]
        if self.converge.team is not None:
            teams.append(self.converge.team)
        if len({team.name for team in teams}) != len(teams):
            raise ValueError("duplicate team names")
        parts = {part.name for part in self.parts}
        for team in teams:
            if len({agent.name for agent in team.agents}) != len(team.agents):
                raise ValueError("duplicate agent names within a team")
            if team.part is not None and team.part not in parts:
                raise ValueError("team.part: unknown part")
        return self

    @classmethod
    def _from_data(cls, data: dict[str, Any], *, base_dir: Path) -> Self:
        """Validate parsed TOML and capture its base directory."""
        try:
            plan = cls.model_validate(data)
        except ValidationError as err:
            errors = err.errors(include_input=False, include_url=False, include_context=False)
            summary = "; ".join(
                f"{'.'.join(str(part) for part in item['loc']) or 'plan'}: {item['msg']}"
                for item in errors
            )
            raise ConfigError(summary) from None
        plan._base_dir = base_dir
        return plan

    @classmethod
    def load(cls, path: str | Path) -> "Plan":
        """
        Load a TOML plan and capture the file's absolute parent directory.

        Parameters
        ----------
        path : str or Path
            UTF-8 TOML plan file to read.

        Returns
        -------
        Plan
            Validated plan preserving any explicit origin in the file.

        Raises
        ------
        ConfigError
            If TOML parsing or plan validation fails; includes the file path and
            invalid field identifiers, but omits rejected values.
        OSError
            If the plan file cannot be read.

        Examples
        --------
        >>> from swarm_arena import Plan
        >>> Plan.load("plan.toml").base_dir.is_absolute()  # doctest: +SKIP
        True
        """
        path = Path(path)
        try:
            data = read_toml(path)
        except ConfigError:
            # read_toml's parser message and exception chain can contain input keys.
            raise ConfigError(f"{path}: invalid TOML") from None
        try:
            return cls._from_data(data, base_dir=path.parent.resolve())
        except ConfigError as err:
            raise ConfigError(f"{path}: {err}") from None

    @classmethod
    def loads(cls, text: str, *, base_dir: Path) -> "Plan":
        r"""
        Parse a TOML plan while preserving its declared origin.

        Parameters
        ----------
        text : str
            TOML plan text; omitted origin defaults to human.
        base_dir : Path
            Directory against which the plan's relative paths are interpreted.

        Returns
        -------
        Plan
            Validated plan with its base directory captured privately.

        Raises
        ------
        ConfigError
            If TOML parsing or plan validation fails. Unknown field identifiers
            are included; rejected values and arbitrary mapping keys are omitted.

        Examples
        --------
        >>> from pathlib import Path
        >>> from swarm_arena import Plan
        >>> text = 'teams=[]\njudges=[]\n[goal]\nid="g"\nnamespace="demo"'
        >>> Plan.loads(text, base_dir=Path(".")).origin
        'human'
        """
        try:
            data = tomllib.loads(text)
        except tomllib.TOMLDecodeError:
            raise ConfigError("plan: invalid TOML") from None
        return cls._from_data(data, base_dir=base_dir)

    def dumps(self) -> str:
        """
        Serialize the plan as deterministic TOML including its origin.

        Returns
        -------
        str
            TOML for serialized model fields, excluding null fields and base directory.

        Examples
        --------
        >>> from swarm_arena import Goal, Plan
        >>> plan = Plan(goal=Goal(id="g", namespace="demo"), teams=[], judges=[])
        >>> 'origin = "human"' in plan.dumps()
        True
        """
        return dumps_toml(self.model_dump(mode="json", exclude_none=True))

    @property
    def base_dir(self) -> Path:
        """
        Return the captured directory for the plan's relative paths.

        Returns
        -------
        Path
            File parent for loaded plans, supplied directory for parsed text,
            or construction-time working directory for Python-built plans.

        Examples
        --------
        >>> from pathlib import Path
        >>> from swarm_arena import Goal, Plan
        >>> Plan(goal=Goal(id="g", namespace="demo"), teams=[], judges=[]).base_dir == Path.cwd()
        True
        """
        return self._base_dir

    def ranking_name(self) -> str:
        """
        Resolve the ranking judge after checking unique judge names.

        Returns
        -------
        str
            Explicit ranking name or the name of the sole judge.

        Raises
        ------
        ValueError
            If judges have invalid or duplicate names, the ranking is unknown,
            or no unique ranking judge can be selected. Duplicate and unknown
            names are included without exposing component configuration values.

        Examples
        --------
        >>> from swarm_arena import Goal, Plan, Ref
        >>> Plan(goal=Goal(id="g", namespace="demo"), teams=[],
        ...      judges=[Ref(path="demo:Judge", name="score")]).ranking_name()
        'score'
        """
        try:
            judges = [ref.model_copy(update={"name": ref.component_name()}) for ref in self.judges]
        except (ValueError, TypeError):
            raise ValueError("judges: invalid component names") from None
        # Name resolution can raise unsafe import errors; uniqueness errors contain
        # only the already-resolved identifiers and may retain their diagnostics.
        names = unique_names(judges)
        if not self.ranking:
            if len(names) == 1:
                return names[0]
            if len(names) > 1:
                raise ValueError("ranking is required with several judges")
            raise ValueError("judges: no ranking judge available")
        if self.ranking not in names:
            raise ValueError(f"ranking {self.ranking!r} does not name a declared judge")
        return self.ranking

    def rounds_total(self, limits: Limits) -> int:
        """
        Return the explicit round count or the configured default.

        Parameters
        ----------
        limits : Limits
            Limits supplying the default maximum round count.

        Returns
        -------
        int
            Number of rounds represented by this plan.

        Examples
        --------
        >>> from swarm_arena import Goal, Limits, Plan
        >>> Plan(goal=Goal(id="g", namespace="demo"), teams=[], judges=[]).rounds_total(Limits())
        3
        """
        return len(self.rounds) or limits.max_rounds

    _sanitize_errors = model_validator(mode="wrap")(_sanitize_validation)
