# swarm_arena.board

```{eval-rst}
.. automodule:: swarm_arena.board
   :members:
```

`Candidate` stores a tool request, its output, provenance, and checks. Its
`gated_out` property is true when any check fails and is excluded from stored
JSON. Overlay keys must be normalized relative workspace paths, and values
must be lowercase SHA-256 digests.

`Learning` titles are stripped, single-line, nonempty, and at most 120
characters. Bodies are limited to 1500 characters. Non-human authors must cite
at least one evidence candidate; an author starting with `human:` may omit
that evidence. The `from_human` property is derived and excluded from JSON.

`LLMCall` records call usage, costs, and provenance, with optional references
to stored request and response payloads. All three records carry a
`schema_version` and reject a version newer than the library supports.
