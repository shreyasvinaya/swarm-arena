"""Define human-controlled secrets, tracing, gateway, and cache settings."""

from typing import Annotated, ClassVar, Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from swarm_arena._model.base import LLMRole, _SafeModel
from swarm_arena._model.limits import Limits


class _ConfigModel(_SafeModel):
    """Inherit the shared hidden-key configuration validation policy."""


class _HumanConfigModel(_ConfigModel):
    """Show human-authored setting typo identifiers without rejected values."""

    _unknown_fields: ClassVar[Literal["show", "hide"]] = "show"


class SecretsPolicy(_HumanConfigModel):
    """
    Declare the secrets file and permitted environment variable names.

    Parameters
    ----------
    file : str or None, default=None
        Path to KEY=VALUE lines, present on each node.
    allow : list[str], default=[]
        Environment variable names permitted for tools, never their values.

    Examples
    --------
    >>> from swarm_arena.config import SecretsPolicy
    >>> SecretsPolicy(allow=['API_KEY']).file is None
    True
    """

    file: str | None = None
    allow: list[Annotated[str, Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")]] = []


class TracingConfig(_HumanConfigModel):
    """
    Select payload storage and optional Langfuse tracing.

    Parameters
    ----------
    store_payloads : bool, default=True
        Store LLM request and response payloads.
    langfuse : bool, default=False
        Enable Langfuse using its own environment variables.

    Examples
    --------
    >>> from swarm_arena.config import TracingConfig
    >>> TracingConfig().store_payloads
    True
    """

    store_payloads: bool = True
    langfuse: bool = False


class GatewayConfig(_HumanConfigModel):
    """
    Configure an optional LiteLLM gateway and its budget policy.

    Parameters
    ----------
    url : str
        Gateway address starting with http:// or https://.
    master_key_env : str, default='LITELLM_MASTER_KEY'
        Environment variable containing the master key.
    require_hard_cap : bool or None, default=None
        Explicit hard cap policy; None follows the presence of a dollar limit.
    request_timeout_s : float, default=600.0
        Gateway request timeout in seconds.
    key_duration_s : float, default=86400.0
        Virtual key lifetime in seconds.
    default_estimated_output_tokens : int, default=4096
        Output token estimate when a request supplies no override.

    Examples
    --------
    >>> from swarm_arena import Limits
    >>> from swarm_arena.config import GatewayConfig
    >>> GatewayConfig(url='https://gateway.example').hard_cap_required(Limits(max_usd=5))
    True
    """

    url: str = Field(pattern=r"^https?://")
    master_key_env: str = "LITELLM_MASTER_KEY"
    require_hard_cap: bool | None = None
    request_timeout_s: float = 600.0
    key_duration_s: float = 86_400.0
    default_estimated_output_tokens: int = 4096

    def hard_cap_required(self, limits: Limits) -> bool:
        """
        Resolve the hard cap policy for the supplied limits.

        Parameters
        ----------
        limits : Limits
            Run limits whose dollar budget supplies the default policy.

        Returns
        -------
        bool
            Explicit policy, or whether a dollar limit is configured.

        Examples
        --------
        >>> from swarm_arena import Limits
        >>> from swarm_arena.config import GatewayConfig
        >>> GatewayConfig(url='http://localhost:4000').hard_cap_required(Limits())
        False
        """
        return (
            self.require_hard_cap
            if self.require_hard_cap is not None
            else limits.max_usd is not None
        )


class CachePolicy(_HumanConfigModel):
    """
    Configure optional response caching with separate permissions per role.

    Parameters
    ----------
    backend : {'disk', 's3', 'redis'}
        Storage backend for cached responses.
    s3_path : str or None, default=None
        Required S3 location when using the S3 backend.
    disk_dir : str or None, default=None
        Optional directory for the disk backend.
    redis_url_env : str or None, default=None
        Required environment variable name for the Redis backend.
    namespace : str or None, default=None
        Namespace override; None uses the goal namespace.
    ttl_s : int, default=2592000
        Positive cache lifetime in seconds.
    roles : dict[str, RoleSwitch], optional
        Read and write switches. Judges and curators read and write;
        planners only write; managers do neither.

    Examples
    --------
    >>> from swarm_arena.config import CachePolicy
    >>> CachePolicy(backend='disk').roles['manager'].read
    False
    """

    class RoleSwitch(_HumanConfigModel):
        """
        Select cache read and write permissions for a role.

        Parameters
        ----------
        read : bool
            Allow reading cached responses.
        write : bool
            Allow writing responses to the cache.

        Examples
        --------
        >>> from swarm_arena.config import CachePolicy
        >>> CachePolicy.RoleSwitch(read=False, write=True).write
        True
        """

        read: bool
        write: bool

    backend: Literal["disk", "s3", "redis"]
    s3_path: str | None = None
    disk_dir: str | None = None
    redis_url_env: str | None = None
    namespace: str | None = None
    ttl_s: int = Field(default=2_592_000, ge=1)
    roles: dict[LLMRole, RoleSwitch] = {
        "judge": RoleSwitch(read=True, write=True),
        "planner": RoleSwitch(read=False, write=True),
        "manager": RoleSwitch(read=False, write=False),
        "curator": RoleSwitch(read=True, write=True),
    }

    @model_validator(mode="after")
    def _backend_requirements(self) -> Self:
        """Require the selected backend's connection setting."""
        if self.backend == "s3" and not self.s3_path:
            raise PydanticCustomError("arena_config", "s3 backend requires s3_path")
        if self.backend == "redis" and not self.redis_url_env:
            raise PydanticCustomError("arena_config", "redis backend requires redis_url_env")
        return self
