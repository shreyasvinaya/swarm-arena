"""Check the optional distribution before importing Ray's runtime."""

import importlib.metadata

_MSG = "swarm_arena.ray needs the 'ray' extra: pip install 'swarm-arena[ray]'"

try:
    importlib.metadata.version("ray")
except importlib.metadata.PackageNotFoundError:
    raise ModuleNotFoundError(_MSG, name="ray") from None

try:
    import ray as _ray
except ModuleNotFoundError as _exc:
    if _exc.name == "ray":
        raise ModuleNotFoundError(_MSG, name="ray") from None
    raise

if getattr(_ray, "__file__", None) is None:
    raise ModuleNotFoundError(_MSG, name="ray") from None
