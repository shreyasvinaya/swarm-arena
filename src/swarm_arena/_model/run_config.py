"""Define versioned human-only run configuration and TOML persistence."""

from pathlib import Path
from typing import Annotated, ClassVar, Literal

from pydantic import Field, ValidationError

from swarm_arena._errors import ConfigError, _core_error
from swarm_arena._model.base import StoredModel
from swarm_arena._model.catalog import ModelCatalog, PlannerCatalog
from swarm_arena._model.diagnostics import validation_summary
from swarm_arena._model.limits import Limits
from swarm_arena._model.ref import Ref, RefField
from swarm_arena._model.settings import (
    CachePolicy,
    GatewayConfig,
    SecretsPolicy,
    TracingConfig,
)
from swarm_arena._model.toml_io import dumps_toml, loads_toml, read_toml


class RunConfig(StoredModel):
    """
    Configure a run's human-controlled limits, components, and storage.

    Parameters
    ----------
    schema_version : int, default=1
        Stored-format version, upgraded on load.
    limits : Limits, optional
        Execution and budget limits.
    reserve_fraction : float, default=0.15
        Reserved fraction of the budget, in the interval [0, 1).
    goal_weights : dict[str, float] or None, default=None
        Positive relative budget weights by goal identifier.
    executor : Ref, optional
        Executor reference, defaulting to LocalExecutor.
    board : Ref or None, default=None
        Optional board reference.
    curator : Ref or None, default=None
        Optional curator reference.
    run_root : str, default='.swarm-arena/runs'
        Run storage location, preserving relative spelling.
    arena_root : str, default='.swarm-arena/arena'
        Arena storage location, preserving relative spelling.
    artifact_root : str, default='.swarm-arena/artifacts'
        Artifact storage location, preserving relative spelling.
    secrets : SecretsPolicy, optional
        Secrets file and permitted environment variable names.
    schedule_timeout_s : float, default=600.0
        Scheduling timeout in seconds.
    sync_interval_s : float, default=60.0
        Synchronization interval in seconds.
    lease_ttl_s : float, default=60.0
        Ownership lease lifetime in seconds.
    settle_timeout_s : float, default=60.0
        Post-stop settlement timeout in seconds, covering pending call reports
        and durable publication of stop results, goal storage, solutions, and
        review exports.
    tracing : TracingConfig, optional
        Payload storage and tracing policy.
    llm_gateway : GatewayConfig or None, default=None
        Optional LiteLLM gateway settings.
    models : ModelCatalog, optional
        Human-approved model aliases, empty by default.
    planner_catalog : PlannerCatalog, optional
        Human-approved planner components, empty by default.
    cache : CachePolicy or None, default=None
        Optional response cache settings.

    Examples
    --------
    >>> from swarm_arena import RunConfig
    >>> RunConfig().reserve_fraction
    0.15
    """

    _unknown_fields: ClassVar[Literal["show", "hide"]] = "show"
    schema_kind: ClassVar[str] = "run_config"
    limits: Limits = Limits()
    reserve_fraction: float = Field(default=0.15, ge=0, lt=1)
    goal_weights: dict[str, Annotated[float, Field(gt=0)]] | None = None
    executor: RefField = Ref(path="swarm_arena.executor:LocalExecutor")
    board: RefField | None = None
    curator: RefField | None = None
    run_root: str = ".swarm-arena/runs"
    arena_root: str = ".swarm-arena/arena"
    artifact_root: str = ".swarm-arena/artifacts"
    secrets: SecretsPolicy = SecretsPolicy()
    schedule_timeout_s: float = Field(default=600.0, gt=0)
    sync_interval_s: float = Field(default=60.0, gt=0)
    lease_ttl_s: float = Field(default=60.0, gt=0)
    settle_timeout_s: float = Field(default=60.0, gt=0)
    tracing: TracingConfig = TracingConfig()
    llm_gateway: GatewayConfig | None = None
    models: ModelCatalog = ModelCatalog({})
    planner_catalog: PlannerCatalog = PlannerCatalog()
    cache: CachePolicy | None = None

    @classmethod
    def load(cls, path: str | Path) -> "RunConfig":
        """
        Load and validate a UTF-8 TOML run configuration.

        Parameters
        ----------
        path : str or Path
            Configuration file to read.

        Returns
        -------
        RunConfig
            Validated configuration with relative roots preserved.

        Raises
        ------
        ConfigError
            If reading, parsing, or validation fails. Diagnostics identify the
            path and safe field locations, including misspelled setting names,
            without rejected input values or arbitrary mapping keys.

        Examples
        --------
        >>> from swarm_arena import RunConfig
        >>> RunConfig.load('run.toml').reserve_fraction  # doctest: +SKIP
        0.15
        """
        path = Path(path)
        data = read_toml(path)
        return cls._from_data(data, where=str(path))

    @classmethod
    def loads(cls, text: str) -> "RunConfig":
        """
        Parse a UTF-8 TOML run configuration with safe diagnostics.

        Parameters
        ----------
        text : str
            TOML configuration source.

        Returns
        -------
        RunConfig
            Validated run configuration.

        Raises
        ------
        ConfigError
            If parsing or validation fails, with identifiers and safe positions.

        Examples
        --------
        >>> RunConfig.loads(RunConfig().dumps()) == RunConfig()
        True
        """
        return cls._from_data(loads_toml(text, where="run_config"), where="run_config")

    @classmethod
    def _from_data(cls, data: object, *, where: str) -> "RunConfig":
        """Validate parsed settings with one shared summary renderer."""
        try:
            return cls.model_validate(data)
        except ValidationError as error:
            raise _core_error(
                ConfigError, f"{where}: {validation_summary(error, root='run_config')}"
            ) from None

    def dumps(self) -> str:
        """
        Serialize the configuration as deterministic TOML without null fields.

        Returns
        -------
        str
            TOML containing configuration and secret names, never resolved secrets.

        Examples
        --------
        >>> import tomllib
        >>> from swarm_arena import RunConfig
        >>> cfg = RunConfig()
        >>> RunConfig.model_validate(tomllib.loads(cfg.dumps())) == cfg
        True
        """
        return dumps_toml(self.model_dump(mode="json", exclude_none=True))
