# Use your own files and script

Written in M1a.

## File snapshots

A goal or component can declare a `FileSet` with globs relative to the plan file:

```toml
[goal]
id = "example"
namespace = "demo"
files = ["problem/*.txt", "problem/run.py"]
```

Snapshotting stores each selected regular file by SHA-256 and records a sorted
`path → sha256` manifest while keeping the original globs. Every glob must match
at least one regular file. Symlinks below the plan directory and reserved
top-level paths (`swarm`, `parents`, `candidate`) are rejected. Those names are
allowed as basenames below an ordinary directory, such as `problem/candidate`.

Once a file set has a manifest, snapshotting reuses it without reading the source
files again. Source edits or deletion therefore cannot change an existing
snapshot. A file set with no globs needs no source files.

The internal snapshot helpers cover the main goal, part goals, tools, managers,
checks, judges, policy, and convergence team. Component configs are validated
before traversing explicit `FileSet` and nested `Ref` fields, including lists of
refs. Omitted defaults stay omitted, unrelated settings keep their original
representation, and the returned plan retains its original base directory.
