# swarm_arena.scoring

```{eval-rst}
.. automodule:: swarm_arena.scoring
   :members:
```

Scores are available as `swarm_arena.Score`. Larger lexicographic keys rank
higher. Keys must be nonempty and contain only finite numbers.

```python
from swarm_arena import Score
from swarm_arena.scoring import CheckResult

loss = Score.minimize(0.25, notes="Lower loss is better")
assert loss.key == (-0.25,)
assert loss.value == 0.25
quality = Score.maximize(3.0, verdict="good")
passed = CheckResult(check="valid", passed=True)
```

`Score.identity` returns `(judge, rubric_hash, panel_hash)` for comparing
scores from the same judge configuration.
