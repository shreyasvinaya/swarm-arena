"""Define goals, snapshotted file sets, and cross-run goal identities."""

from typing import Annotated, cast

from pydantic import AfterValidator, Field, model_validator

from swarm_arena._model.base import FrozenModel, JsonValue
from swarm_arena._model.canonical import canonical_json, sha256_hex
from swarm_arena._model.data import Manifest, Sha256, TomlJsonObject
from swarm_arena._tooling.paths import check_glob

_Slug = Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")]
_Sha256 = Sha256


class FileSet(FrozenModel):
    """
    Describe workspace files with globs and a snapshotted manifest.

    Snapshotting stores each matched regular file by SHA-256; a file larger
    than 1 GiB is refused by its size and named in the error.

    Parameters
    ----------
    globs : list[str], default=[]
        Workspace-relative file patterns; a list input is shorthand for this field.
    manifest : dict[str, str], default={}
        Normalized workspace paths mapped to lowercase SHA-256 digests.

    Examples
    --------
    >>> from swarm_arena.tool import FileSet
    >>> FileSet.model_validate(["problem/*.phyk"]).globs
    ['problem/*.phyk']
    """

    globs: list[Annotated[str, AfterValidator(check_glob)]] = []
    manifest: Manifest = {}

    @model_validator(mode="before")
    @classmethod
    def _from_list(cls, value: object) -> object:
        """Interpret a bare glob list as the globs field."""
        return {"globs": value} if isinstance(value, list) else value

    @property
    def snapshotted(self) -> bool:
        """
        Return whether the file set has no globs or has a manifest.

        Returns
        -------
        bool
            Whether file identity is available without reading source files.

        Examples
        --------
        >>> from swarm_arena.tool import FileSet
        >>> FileSet().snapshotted
        True
        """
        return not self.globs or bool(self.manifest)


class Goal(FrozenModel):
    """
    Describe a goal and the inputs that identify it across runs.

    Parameters
    ----------
    id : str
        Lowercase slug identifying the goal within its namespace.
    namespace : str
        Lowercase problem-family slug shared by related goals.
    title : str, default=""
        Short human-readable title.
    description : str, default=""
        Full problem description.
    inputs : dict[str, Any], default={}
        Structured problem inputs used in the goal identity.
    files : FileSet, default=FileSet()
        Source patterns and their snapshotted file manifest.
    metadata : dict[str, Any], default={}
        Additional descriptive information outside the goal identity.

    Examples
    --------
    >>> from swarm_arena import Goal
    >>> Goal(id="ns-viscosity", namespace="physika").title
    ''
    """

    id: _Slug
    namespace: _Slug
    title: str = ""
    description: str = ""
    inputs: TomlJsonObject = {}
    files: FileSet = FileSet()
    metadata: TomlJsonObject = {}

    def key(self) -> "GoalKey":
        """
        Compute this goal's identity from its inputs and file manifest.

        Returns
        -------
        GoalKey
            Namespace, id, and canonical hashes of inputs and files.

        Raises
        ------
        ValueError
            If files are not snapshotted or inputs cannot be canonically encoded.

        Examples
        --------
        >>> from swarm_arena import Goal
        >>> Goal(id="g", namespace="n").key().namespace
        'n'
        """
        return GoalKey.of(self)


class GoalKey(FrozenModel):
    """
    Identify a goal independently of a particular run.

    Parameters
    ----------
    namespace : str
        Problem family of the goal.
    id : str
        Goal identifier within the namespace.
    inputs_sha256 : str
        SHA-256 digest of the canonical structured inputs.
    files_sha256 : str
        SHA-256 digest of the canonical snapshotted manifest.

    Examples
    --------
    >>> from swarm_arena import Goal
    >>> from swarm_arena.board import GoalKey
    >>> GoalKey.of(Goal(id="g", namespace="n")).id
    'g'
    """

    namespace: str
    id: str
    inputs_sha256: str
    files_sha256: str

    @classmethod
    def of(cls, goal: Goal) -> "GoalKey":
        """
        Build an identity from a goal with snapshotted files.

        Parameters
        ----------
        goal : Goal
            Goal whose inputs and file manifest should be hashed.

        Returns
        -------
        GoalKey
            Stable identity, independent of dictionary insertion order.

        Raises
        ------
        ValueError
            If files are not snapshotted or inputs cannot be canonically encoded.

        Examples
        --------
        >>> from swarm_arena import Goal
        >>> from swarm_arena.board import GoalKey
        >>> GoalKey.of(Goal(id="g", namespace="n")) == Goal(id="g", namespace="n").key()
        True
        """
        if not goal.files.snapshotted:
            raise ValueError(f"goal {goal.id}: files are not snapshotted")
        return cls(
            namespace=goal.namespace,
            id=goal.id,
            inputs_sha256=sha256_hex(canonical_json(goal.inputs)),
            files_sha256=sha256_hex(canonical_json(cast(JsonValue, goal.files.manifest))),
        )
