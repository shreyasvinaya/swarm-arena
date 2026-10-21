from typing import NoReturn


class HostileString(str):
    """Fail if core code dispatches any relevant string hook."""

    armed: bool = False

    def _check(self) -> None:
        if self.armed:
            raise AssertionError("user string hook executed")

    def __hash__(self) -> int:
        self._check()
        return str.__hash__(self)

    def __eq__(self, other: object) -> bool:
        self._check()
        return str.__eq__(self, other)

    def __ne__(self, other: object) -> bool:
        self._check()
        return str.__ne__(self, other)

    def __str__(self) -> str:
        self._check()
        return str.__str__(self)

    def __format__(self, spec: str) -> str:
        self._check()
        return str.__format__(self, spec)

    def encode(self, encoding: str = "utf-8", errors: str = "strict") -> bytes:
        self._check()
        return str.encode(self, encoding, errors)


class NotString:
    def __str__(self) -> NoReturn:
        raise AssertionError("invalid value formatted")

    def __repr__(self) -> NoReturn:
        raise AssertionError("invalid value represented")


def hostile(value: str) -> HostileString:
    result = HostileString(value)
    result.armed = True
    return result
