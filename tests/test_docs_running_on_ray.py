"""Pin the cluster guide and its configuration-only example."""

from dataclasses import replace
from pathlib import Path
from textwrap import indent

import pytest
from _docs import block_runs, docs_bindir, page_path, parse_blocks, required_facts, run_blocks

REQUIRED_FACTS = [
    "swarm-arena[ray]",
    'ray.init(address="auto")',
    "swarm-arena submit",
    "RAY_AUTH_MODE=token",
    "RAY_AUTH_TOKEN",
    "runtime_env",
    '{"llm_keys": 1}',
    "max_retries=0",
    "max_calls=1",
    "s3://",
    "worker nodes",
    "strict profile",
    "require_hard_cap",
    "Blocking outbound traffic from workers to provider hosts",
    "swarm-arena reconcile",
    "observed (completeness unverified)",
    "soft caps and estimated spend",
    "RayContext.dashboard_url",
    "HTTPS, proxied or",
    "identity-matched",
    "worker-local disk",
    "entire reservation",
    "settle_timeout_s",
]


def test_required_facts() -> None:
    required_facts("Running on Ray", REQUIRED_FACTS)


def test_ray_configuration_example_never_initializes_ray(tmp_path: Path) -> None:
    page = page_path("Running on Ray")
    blocks = parse_blocks(page.read_text(encoding="utf-8"))
    examples = [
        block
        for block in blocks
        if block.kind == "requires" and block.arg == "ray" and "RunConfig(" in block.code
    ]
    assert len(examples) == 1
    block = examples[0]
    bindir = docs_bindir()
    if not block_runs(block, workdir=tmp_path, bindir=bindir):
        pytest.skip("Ray configuration example needs the target interpreter's ray extra")
    guarded = (
        "from unittest.mock import patch\n"
        "import tomllib\n"
        "with patch('ray.init', side_effect=AssertionError('cluster startup forbidden')):\n"
        + indent(block.code, "    ")
        + "\n    data = tomllib.loads(config.dumps())\n"
        "    assert data['executor']['path'] == 'swarm_arena.ray:RayExecutor'\n"
        "    roots = {'run': 'runs', 'arena': 'arena', 'artifact': 'artifacts'}\n"
        "    for name, suffix in roots.items():\n"
        "        assert data[name + '_root'] == 's3://bucket/' + suffix\n"
    )
    run_blocks([replace(block, code=guarded)], source=str(page), workdir=tmp_path, bindir=bindir)
