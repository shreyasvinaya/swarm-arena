"""Importable deterministic failures used by recovery regressions."""


def fail_square(x: float) -> dict[str, float]:
    raise RuntimeError("tool failed")
