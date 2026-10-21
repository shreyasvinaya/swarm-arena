# swarm_arena.testing

`OfflineHarness.run_rounds` honors `Limits.run_timeout_s`, including preparation
after lease acquisition. Reaching that deadline raises
`RuntimeError("offline harness stopped")`; owned services still receive bounded
cleanup. Set the limit to allow enough time for the rounds your test exercises.

```{eval-rst}
.. automodule:: swarm_arena.testing
   :members:
```
