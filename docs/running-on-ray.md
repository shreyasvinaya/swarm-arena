# Running on Ray

## Install

Install the optional runtime with uv:

```sh
uv add "swarm-arena[ray] @ git+https://github.com/shreyasvinaya/swarm-arena"
```

The extra requires `ray[default]>=2.59,<3`. Ray 2.59 provides CPython
3.11–3.13 wheels on Linux x86_64/aarch64 and macOS arm64 (macOS 12 or newer).
Intel macOS has no wheel for this version. macOS CI checks the core package;
the dedicated Linux Ray jobs check the optional runtime.

Set `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0` before starting a Ray driver with `uv run`
to prevent Ray from automatically packaging the driver's uv environment.
Cluster authentication and worker dependency provisioning are separate setup steps.

Ray execution is planned for a later release. Use
[LocalExecutor](api/executor.md) to run tools on this host today.
