# Concepts

Written in M1a.

## Artifact storage

Artifacts are stored once under the SHA-256 digest of their bytes. Repeated
uploads of identical content return the same digest. Every retrieval checks the
bytes against that digest and raises a storage error if they have changed.

The internal storage layer currently supports local paths. Its conditional
create writes and syncs a temporary file before publishing it with a hard link;
concurrent writers cannot overwrite an existing key. Temporary and lock files
are omitted from sorted key listings. Store constructors do no filesystem I/O.
Conditional replacement arrives in M1b, and remote storage arrives in M2.

For internal integrations, a minimal example is:

```python
from pathlib import Path

from swarm_arena._store.artifacts import ArtifactStore
from swarm_arena._store.base import open_store

artifacts = ArtifactStore(open_store("./artifacts"))
digest = artifacts.put_bytes(b"hello")
assert artifacts.get_bytes(digest) == b"hello"
artifacts.fetch(digest, Path("workspace/input.txt"))
```

`fetch` creates missing destination directories and sets permissions to `0444`
by default, or `0644` with `read_only=False`. `put_file` returns a `FileRef` whose
name and path are the source basename. That basename must satisfy the `FileRef`
path rules; callers can use `model_copy` to set a workspace-relative path. The
snapshot layer uses `put_bytes` when only the digest is needed.

These storage modules are private implementation interfaces. Async callers run
their synchronous operations through `anyio.to_thread.run_sync`.
