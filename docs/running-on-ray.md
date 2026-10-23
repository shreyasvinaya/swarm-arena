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

## Running the driver as a Ray Job

`swarm-arena submit` runs `swarm-arena run --yes` inside the cluster as a
Ray Job. The Ray executor connects from that driver with
`ray.init(address="auto")`; Ray Client (`ray://`) is not used. Check the
estimate locally before approving unattended execution:

```sh
swarm-arena plan run.toml plan.toml
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
swarm-arena submit --address http://<head>:8265 --working-dir . --yes run.toml plan.toml
```

Replace `<head>` with your head node's hostname. Without `--address`, Ray
resolves `RAY_ADDRESS`. Success prints the submission ID and exits zero;
Ray submission errors exit one. Without `--yes`, the command submits nothing
and exits two, even in a terminal. The ID is Ray's `submission_id`, not its
deprecated `job_id` argument.

The worker image or repeatable `--pip SPEC` options must provide
`swarm-arena[ray]` and the connector package, including the `swarm-arena`
console script. For example, before the first release:

```sh
swarm-arena submit --address http://<head>:8265 --working-dir . \
  --pip 'swarm-arena[ray] @ git+https://github.com/shreyasvinaya/swarm-arena' \
  --pip deepretro-swarm --yes run.toml plan.toml
```

Paths in the driver command are shell-quoted. Keep the run configuration,
plans and connector code accessible from the job's working directory.
Only an explicitly supplied `working_dir` and `pip` list enter `runtime_env`;
omitting both sends no runtime environment. Choose the upload directory so it
contains no secrets, and keep credentials out of dependency specifications.
Secrets never go in `runtime_env`, including `env_vars`: provision them
separately in the worker environment or through `RunConfig.secrets.file` on
the worker. The command does not copy credentials from the submitting shell.

Authentication also needs external provisioning. Same-user local
`ray.init(address="auto")` can use token authentication through Ray's default
token file. Explicit-address drivers and HTTP Jobs clients need
`RAY_AUTH_MODE=token` and an available token (for example, externally supplied
`RAY_AUTH_TOKEN` or a provisioned token file). Provision authentication for
the submitting client and the in-cluster driver separately; local discovery
does not establish remote access. Keep tokens out of job metadata and uploaded
files. Set `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0` in the launching environment
before Ray initialization to retain explicit packaging control.
