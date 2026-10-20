"""Make the names a public module exports report that module as their home."""

import inspect
import sys
from collections.abc import Sequence


def publish(module_name: str, names: Sequence[str]) -> None:
    """
    Set ``__module__`` of exported names to their public module.

    Every exported class and function gets ``__module__ = module_name``, so docs, pickling,
    and import paths show the public path, not the private module that defines the name.
    For each exported class, the same is done for its public nested classes (recursively)
    and for the functions defined in its body: methods, ``classmethod`` and
    ``staticmethod`` functions, and property getters. Without that step, doctest skips
    every method example: the private module skips the class, and the public module skips
    the method. Only functions whose ``__qualname__`` puts them in the class body change,
    so a function that a library injects into the class (pydantic does) is left alone.
    Original class line metadata is removed when publishing, as with Python 3.13's
    native ``__module__`` assignment, so source inspection cannot read unrelated
    lines from the facade module.

    Parameters
    ----------
    module_name : str
        Name of the public module that exports the names. It must be in ``sys.modules``.
    names : Sequence[str]
        The exported names, normally the module's ``__all__``.

    Examples
    --------
    >>> import sys, types
    >>> from swarm_arena._facade import publish
    >>> class Demo:
    ...     def run(self) -> None:
    ...         '''Run the demo.'''
    >>> facade = types.ModuleType("demo_public")
    >>> facade.Demo = Demo
    >>> sys.modules["demo_public"] = facade
    >>> publish("demo_public", ["Demo"])
    >>> Demo.__module__, Demo.run.__module__
    ('demo_public', 'demo_public')
    >>> del sys.modules["demo_public"]
    """
    module = sys.modules[module_name]
    for name in names:
        obj = getattr(module, name)
        if isinstance(obj, type):
            _publish_class(obj, module_name)
        elif inspect.isfunction(obj):
            obj.__module__ = module_name


def _publish_class(cls: type, module_name: str) -> None:
    cls.__module__ = module_name
    # Some metaclasses retain Python 3.13's original line in the new module.
    if "__firstlineno__" in vars(cls):
        delattr(cls, "__firstlineno__")
    prefix = cls.__qualname__ + "."
    for attr, value in vars(cls).items():
        if isinstance(value, type):
            if not attr.startswith("_") and value.__qualname__.startswith(prefix):
                _publish_class(value, module_name)
            continue
        if isinstance(value, classmethod | staticmethod):
            value = value.__func__
        elif isinstance(value, property):
            value = value.fget
        if inspect.isfunction(value) and value.__qualname__.startswith(prefix):
            value.__module__ = module_name
