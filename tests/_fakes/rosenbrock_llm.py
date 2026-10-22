"""Data-only tool used to test a final judge's unfundable serialized inputs."""

from typing import Any


def large_output(size: int) -> dict[str, Any]:
    """
    Return a configurable candidate payload without making model calls.

    Parameters
    ----------
    size : int
        Number of characters included in the candidate data.

    Returns
    -------
    dict[str, Any]
        Numeric objective and deliberately large, finite JSON data.

    Examples
    --------
    >>> large_output(3)
    {'y': 0.0, 'payload': 'xxx'}
    """
    return {"y": 0.0, "payload": "x" * size}
