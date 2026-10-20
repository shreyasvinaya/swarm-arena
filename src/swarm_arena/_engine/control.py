"""Name epoch-scoped run control files."""


def stop_key(epoch: int, *, now: bool) -> str:
    """
    Return the graceful or immediate stop key for one lease epoch.

    Parameters
    ----------
    epoch : int
        Current lease epoch.
    now : bool
        Select immediate cancellation instead of graceful admission closure.

    Returns
    -------
    str
        Run-store-relative control key.

    Examples
    --------
    >>> stop_key(2, now=True)
    'control/stop-now-e2'
    """
    return f"control/stop{'-now' if now else ''}-e{epoch}"
