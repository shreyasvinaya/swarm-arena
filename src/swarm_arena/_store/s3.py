"""Store complete objects using conditional S3 requests and uncached reads."""

import time
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import PurePosixPath
from typing import Any, cast
from urllib.parse import urlsplit
from uuid import uuid4

from swarm_arena._errors import (
    AlreadyExists,
    ConditionalWritesUnsupported,
    NotFound,
    PreconditionFailed,
    StorageError,
    WriteOutcomeUnknown,
    _core_error,
)
from swarm_arena._store.base import Blob, Store, _check_key


class _S3Error(Exception):
    """Carry response classification without exposing backend error text."""

    status: int | None
    code: str

    def __init__(self, status: int | None, code: str) -> None:
        self.status = status
        self.code = code
        super().__init__("S3 request failed")


def _response_error(error: Exception) -> _S3Error:
    """Recover status and code from a raw or s3fs-translated ClientError."""
    from botocore.exceptions import ClientError

    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ClientError):
            response = current.response
            return _S3Error(
                response.get("ResponseMetadata", {}).get("HTTPStatusCode"),
                response.get("Error", {}).get("Code", ""),
            )
        current = current.__cause__
    return _S3Error(None, "")


def _single_attempt_http_session() -> type[Any]:
    """Disable aiohttp's internal retry without importing extras at module load."""
    from aiobotocore.httpsession import AIOHTTPSession

    class SingleAttemptSession(AIOHTTPSession):
        async def _get_session(self, proxy_url: str | None) -> Any:
            session: Any = await super()._get_session(proxy_url)
            # aiohttp can resend idempotent methods below the SDK retry layer.
            session._retry_connection = False
            return session

    return SingleAttemptSession


class S3Store(Store):
    """Store objects with server-enforced creation and version conditions.

    Parameters
    ----------
    url : str
        S3 bucket and optional root prefix.
    **storage_options : Any
        Options forwarded to the lazy s3fs client. Credentials and endpoints
        normally come from the AWS environment, never stored run configuration.

    Notes
    -----
    Construction performs no I/O. Reads bypass listing caches. ETags remain
    opaque quoted strings returned by S3.

    Examples
    --------
    >>> S3Store("s3://example/runs").url
    's3://example/runs'
    """

    def __init__(self, url: str, **storage_options: Any) -> None:
        """Validate the root and check the optional dependency without I/O."""
        try:
            import s3fs  # noqa: F401
        except ImportError:
            raise _core_error(
                ConditionalWritesUnsupported,
                's3:// storage needs the s3 extra: uv add "swarm-arena[s3]"; '
                "see https://github.com/shreyasvinaya/swarm-arena#install",
            ) from None
        try:
            parsed = urlsplit(url)
            if (
                parsed.scheme != "s3"
                or not parsed.netloc
                or any(c in parsed.netloc for c in "@: \\")
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError
            prefix = parsed.path.lstrip("/").rstrip("/")
            if prefix:
                _check_key(prefix)
        except ValueError:
            raise _core_error(ValueError, "invalid S3 store root") from None
        self._init_components(parsed.netloc, prefix, storage_options)

    def _init_components(self, bucket: str, prefix: str, options: dict[str, Any]) -> None:
        """Initialize a validated raw root without interpreting key characters."""
        self._bucket = bucket
        self._prefix = prefix + "/" if prefix else ""
        self._url = f"s3://{self._bucket}" + (f"/{prefix}" if prefix else "")
        self._options = dict(options)
        self._fs: Any = None
        self._write_fs: Any = None
        self._params: frozenset[str] | None = None

    @property
    def url(self) -> str:
        """Return the normalized store root.

        Returns
        -------
        str
            Bucket URL with its optional prefix.

        Examples
        --------
        >>> S3Store("s3://example/runs/").url
        's3://example/runs'
        """
        return self._url

    def _filesystem(self, *, write: bool = False) -> Any:
        """Build separate read and single-attempt PUT clients lazily."""
        fs = self._write_fs if write else self._fs
        if fs is None:
            import s3fs

            options: dict[str, Any] = {
                **self._options,
                "skip_instance_cache": True,
                "use_listings_cache": False,
            }
            if write:
                config = dict(options.get("config_kwargs") or {})
                config["retries"] = {"total_max_attempts": 1, "mode": "standard"}
                config["http_session_cls"] = _single_attempt_http_session()
                options["config_kwargs"] = config
            fs = s3fs.S3FileSystem(**options)
            if write:
                # Automatic PUT retries must not hide an ambiguous response.
                fs.retries = 1
                self._write_fs = fs
            else:
                self._fs = fs
        return fs

    def _key(self, key: str) -> str:
        """Validate a relative key using the common Store contract."""
        try:
            return self._prefix + _check_key(key)
        except ValueError:
            raise _core_error(
                ValueError, "bad store key: require a relative path without dot segments"
            ) from None

    def _conditional_params(self) -> frozenset[str]:
        """Cache the live client's PutObject input model."""
        if self._params is None:
            try:
                fs = self._filesystem(write=True)
                client = fs.s3
                self._params = frozenset(
                    client.meta.service_model.operation_model("PutObject").input_shape.members
                )
            except Exception:
                raise _core_error(StorageError, "S3 PutObject model inspection failed") from None
        return self._params

    def _call(self, method: str, **kwargs: Any) -> dict[str, Any]:
        """Call the read client without translating its errors."""
        return cast(dict[str, Any], self._filesystem().call_s3(method, **kwargs))

    @contextmanager
    def _reading(self, operation: str) -> Generator[None]:
        """Map read, stream-consumption and listing failures at one boundary."""
        try:
            yield
        except Exception as exc:
            response = _response_error(exc)
            if response.status == 404:
                raise _core_error(NotFound, f"S3 {operation}: object not found") from None
            raise _core_error(StorageError, f"S3 {operation} failed") from None

    def _put_object(self, key: str, data: bytes, **condition: str) -> str:
        """Send one conditional PUT and retain only response classification."""
        try:
            result = self._filesystem(write=True).call_s3(
                "put_object", Bucket=self._bucket, Key=key, Body=data, **condition
            )
            return cast(str, result["ETag"])
        except Exception as exc:
            response = _response_error(exc)
            if response.status is None:
                raise _core_error(WriteOutcomeUnknown, "S3 PutObject outcome unknown") from None
            raise response from None

    def _write(self, key: str, data: bytes, parameter: str, value: str) -> str:
        """Guard the model, then retry only conditional-request conflicts."""
        full_key = self._key(key)
        if parameter not in self._conditional_params():
            raise _core_error(
                ConditionalWritesUnsupported,
                f'S3 PutObject requires {parameter}; uv add --upgrade "swarm-arena[s3]"; '
                "see https://github.com/shreyasvinaya/swarm-arena#install",
            ) from None
        for attempt in range(4):
            try:
                return self._put_object(full_key, data, **{parameter: value})
            except _S3Error as exc:
                if exc.status == 409:
                    if attempt < 3:
                        time.sleep((0.1, 0.2, 0.4)[attempt])
                        continue
                    raise _core_error(
                        WriteOutcomeUnknown, "S3 PutObject conflict outcome unknown"
                    ) from None
                if exc.status == 412:
                    kind = AlreadyExists if parameter == "IfNoneMatch" else PreconditionFailed
                    raise _core_error(kind, f"S3 PutObject {parameter} condition failed") from None
                if exc.status == 404:
                    raise _core_error(NotFound, "S3 PutObject: object not found") from None
                raise _core_error(StorageError, "S3 PutObject failed") from None
        raise AssertionError("unreachable write attempt")

    def read(self, key: str) -> Blob:
        """Read bytes and the ETag from the same GET response.

        Parameters
        ----------
        key : str
            Relative object key.

        Returns
        -------
        Blob
            Complete contents and opaque ETag.

        Examples
        --------
        >>> store.read("example").data  # doctest: +SKIP
        b'hello'
        """
        from fsspec.asyn import sync

        full_key = self._key(key)
        with self._reading("get_object"):
            response = self._call("get_object", Bucket=self._bucket, Key=full_key)

            async def consume() -> bytes:
                async with response["Body"] as body:
                    return cast(bytes, await body.read())

            data = cast(bytes, sync(self._filesystem().loop, consume))
            return Blob(data, response["ETag"])

    def create(self, key: str, data: bytes) -> str:
        """Create an object only when no object occupies its key.

        Parameters
        ----------
        key : str
            Relative object key.
        data : bytes
            Complete object contents.

        Returns
        -------
        str
            Quoted ETag of the created object.

        Examples
        --------
        >>> etag = store.create("example", b"hello")  # doctest: +SKIP
        """
        return self._write(key, data, "IfNoneMatch", "*")

    def replace(self, key: str, data: bytes, *, if_match: str) -> str:
        """Replace an object only at the observed version.

        Parameters
        ----------
        key : str
            Relative object key.
        data : bytes
            Complete replacement contents.
        if_match : str
            Required current ETag, including its quotes.

        Returns
        -------
        str
            Quoted ETag of the replacement.

        Examples
        --------
        >>> etag = store.replace("example", b"updated", if_match=etag)  # doctest: +SKIP
        """
        return self._write(key, data, "IfMatch", if_match)

    def exists(self, key: str) -> bool:
        """Check object existence without masking unreachable storage.

        Parameters
        ----------
        key : str
            Relative object key.

        Returns
        -------
        bool
            False only for a missing object.

        Examples
        --------
        >>> store.exists("example")  # doctest: +SKIP
        True
        """
        full_key = self._key(key)
        try:
            with self._reading("head_object"):
                self._call("head_object", Bucket=self._bucket, Key=full_key)
        except NotFound:
            return False
        return True

    def list(self, prefix: str) -> list[str]:
        """List all pages of matching keys without a listing cache.

        Parameters
        ----------
        prefix : str
            Literal relative prefix; empty selects the entire store.

        Returns
        -------
        list[str]
            Sorted root-relative keys, excluding temporary and lock basenames.

        Examples
        --------
        >>> store.list("runs/")  # doctest: +SKIP
        ['runs/a.json']
        """
        full_prefix = self._key(prefix) if prefix else self._prefix
        kwargs: dict[str, Any] = {"Bucket": self._bucket, "Prefix": full_prefix}
        keys: list[str] = []
        with self._reading("list_objects_v2"):
            while True:
                page = self._call("list_objects_v2", **kwargs)
                for item in page.get("Contents", []):
                    key = item["Key"][len(self._prefix) :]
                    name = PurePosixPath(key).name
                    if ".tmp-" not in name and not name.endswith(".lock"):
                        keys.append(key)
                if not page.get("IsTruncated", False):
                    break
                kwargs["ContinuationToken"] = page["NextContinuationToken"]
        return sorted(keys)

    def child(self, prefix: str) -> "S3Store":
        """Open a child root with the same client options and no I/O.

        Parameters
        ----------
        prefix : str
            Non-empty relative key for the child root.

        Returns
        -------
        S3Store
            Independent store rooted at the child prefix.

        Examples
        --------
        >>> S3Store("s3://example").child("runs").url
        's3://example/runs'
        """
        key = self._key(prefix).rstrip("/")
        child = object.__new__(S3Store)
        child._init_components(self._bucket, key, self._options)
        return child

    def probe_conditional_writes(self) -> None:
        """Verify create, stale replace, current replace and readback behavior.

        The one small probe object remains under ``.swarm-arena-probe/``;
        bucket lifecycle rules can remove old probes.

        Raises
        ------
        ConditionalWritesUnsupported
            If any step fails to demonstrate the required behavior.

        Examples
        --------
        >>> store.probe_conditional_writes()  # doctest: +SKIP
        """
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%f")[:-3] + "Z"
        key = f".swarm-arena-probe/{stamp}-{uuid4().hex[:8]}"
        step = "create"
        try:
            etag = self.create(key, b"1")
            try:
                self.create(key, b"2")
            except AlreadyExists:
                pass
            else:
                raise ValueError
            step = "replace stale"
            try:
                self.replace(key, b"3", if_match="probe-mismatch")
            except PreconditionFailed:
                pass
            else:
                raise ValueError
            step = "replace current"
            self.replace(key, b"4", if_match=etag)
            step = "read"
            if self.read(key).data != b"4":
                raise ValueError
        except Exception:
            raise _core_error(
                ConditionalWritesUnsupported, f"S3 {step} did not behave conditionally"
            ) from None
