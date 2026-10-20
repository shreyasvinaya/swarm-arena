# The arena

The shared arena service is planned for a later release. For the records
available now, see the [board API](api/board.md).

## Storage on S3

Each of `run_root`, `arena_root`, and `artifact_root` can be a local path or an
`s3://bucket/prefix` root. Install S3 support with:

```console
uv add 'swarm-arena[s3]'
```

For example, set the roots in your run configuration:

```toml
run_root = "s3://my-bucket/runs"
arena_root = "s3://my-bucket/arena"
artifact_root = "s3://my-bucket/artifacts"
```

Credentials and the endpoint come from the standard AWS environment, including
`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`, and
`AWS_ENDPOINT_URL_S3`. Do not put them in `run.toml`. A local `run_root` with an
S3 `arena_root` is refused because other readers need access to the run journal.

The endpoint must enforce both `If-None-Match` creation and `If-Match`
replacement. Preflight verifies these conditions once per distinct S3 root.
`resume` probes again before taking over a run, since it may run on a different
host or endpoint. The planned `review import` command has the same requirement
before takeover. `S3Store` also refuses each write if the installed botocore model
would discard its required conditional parameter. Listings bypass s3fs caches
so newly published run pointers are visible on the next listing.

Each probe writes one small object under `.swarm-arena-probe/` within each S3
root. The objects are retained because `Store` has no delete operation. Configure
a bucket lifecycle rule for each root's probe prefix to expire old probes.
Local stores require no probe objects.
