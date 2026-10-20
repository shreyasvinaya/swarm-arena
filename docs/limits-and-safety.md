# Limits and safety

Local execution supports timeouts, resource limits, bounded output uploads,
and an environment allowlist. See the [executor API](api/executor.md) for
settings and the [isolation boundary](own-files.md#isolation-boundary) for
the trust required of scripts. Local execution is not a security sandbox.

The optional `Limits.max_usd` run cap must be positive and no greater than
1,000,000,000 USD. Leaving it unset means there is no configured dollar cap.
Budget balances still support only the inclusive range from -1,000,000,000
to 1,000,000,000 USD, preserving exact accounting in whole micro-dollars.
Construction and arithmetic refuse amounts outside that range. Allocations
round down to whole micro-dollars, with the remainder retained in the parent pool.

```python
from swarm_arena import Limits

limits = Limits(max_usd=100.0)
```
