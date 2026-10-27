# Releasing swarm-arena

1. **One-time setup (owner only).** On PyPI, configure a **trusted publisher**
   for project `swarm-arena`: owner `shreyasvinaya`, repository `swarm-arena`,
   workflow `release.yml`, environment `pypi`. The project name was reserved
   with `0.0.0a1`; that placeholder is not the usable release.

   On GitHub, create environment `pypi`, set the owner as a **required reviewer**,
   and under Deployment branches and tags choose **Selected branches and tags**
   with only the tag rule `v*`. Review this protection before each tag and each
   approval: naming a missing environment in YAML can create an unprotected one.
   The publish job alone has `id-token: write`; it uses OIDC trusted publishing
   without a PyPI token. PRs and manual dispatches build and validate but cannot
   publish. Keep repository, environment and PyPI settings changes owner-only.

   Import the repository on Read the Docs as `swarm-arena`. Verify project
   ownership, a successful main/latest build, intended release-version activation
   and working documentation links. Confirm `[project.urls]` matches the real
   repository and hosts. Local YAML tests and Sphinx success do not establish
   hosted setup or an enabled trusted publisher.

2. **Prepare the candidate.** Use a clean checkout of the intended release commit.
   Set the version with `uv version X.Y.Z`, then `uv lock`. Runtime `__version__`
   and docs `release` derive from installed metadata; do not hard-code them.

   For the first release, update the **already-prepared** `0.1.0` changelog
   section's date and content to the actual tag date; never add a **second 0.1.0**
   section. For a later release, create its section once, moving the Unreleased
   entries there. On a repeated preparation, update that existing section.
   Keep one leading, empty `## [Unreleased]` and exactly one dated heading
   `## [X.Y.Z] - YYYY-MM-DD` per version, with a real calendar date.
   In `CHANGELOG.md`, update both references:

   ```text
   [Unreleased]: https://github.com/shreyasvinaya/swarm-arena/compare/vX.Y.Z...HEAD
   [X.Y.Z]: https://github.com/shreyasvinaya/swarm-arena/releases/tag/vX.Y.Z
   ```

   Recheck whether LiteLLM 1.107.0 final is available. If the release-candidate
   floor remains, the owner explicitly chooses to ship with the documented rc
   dependency or wait. Any dependency change requires its affected checks again.
   The install forms already target the release; retain the README source fallback
   until the owner sees 0.1.0 on PyPI.

   Run the local common checks: `uv sync --all-extras --all-groups`,
   `uv run ruff check .`, `uv run ruff format --check .`, `uv run ty check`,
   focused changed/importer pytest files (including `tests/test_ci_config.py` and
   `tests/test_packaging.py`), the changed tests in an isolated core-only environment,
   and `uv run sphinx-build -W -q docs docs/_build/html`.
   Full-suite and supported-matrix acceptance comes from GitHub CI, not a local
   full-suite run. Record skips, deselections and unrun live checks separately.
   Commit with message `release: X.Y.Z` and a body explaining the preparation.

   Build the final committed candidate in fresh owned paths. Substitute the actual
   version below. Default `uv build` builds the sdist and then the wheel **from
   that sdist**; do not add `--wheel` or publish a separately rebuilt wheel.

   ```sh
   RELEASE_TMP=$(mktemp -d)
   RELEASE_VERSION=X.Y.Z
   RELEASE_COMMIT=$(git rev-parse HEAD)
   uv build --out-dir "$RELEASE_TMP/dist" --no-create-gitignore
   uvx twine check --strict "$RELEASE_TMP"/dist/*
   uv venv "$RELEASE_TMP/venv" --python 3.11
   uv pip install --python "$RELEASE_TMP/venv/bin/python" "$RELEASE_TMP"/dist/*.whl
   mkdir "$RELEASE_TMP/empty"
   (cd "$RELEASE_TMP/empty" && "$RELEASE_TMP/venv/bin/swarm-arena" --help)
   uv run --no-sync python tests/test_ci_config.py --check-release \
     --dist "$RELEASE_TMP/dist" --tag "v$RELEASE_VERSION" --commit "$RELEASE_COMMIT" \
     --wheel-python "$RELEASE_TMP/venv/bin/python" --record "$RELEASE_TMP/release.json"
   SWARM_ARENA_DOCS_VENV="$RELEASE_TMP/venv" uv run --no-sync pytest \
     tests/test_docs.py::test_readme_quickstart -v
   ```

   The shared guard checks the exact stable tag, project, dated changelog,
   wheel/sdist metadata, installed metadata, runtime version and docs release.
   Its wheel interpreter runs from an empty folder and must import from that
   venv's site-packages. The README test runs the installed core wheel with only
   the README's input files, no credentials and failing proxy settings; these
   proxies are not an OS network sandbox. Core smoke is not optional-extra or
   live-provider evidence. Missing build/Twine tools or skipped smoke checks
   leave the release unverified. Keep `release.json` with the commit and SHA256
   hashes; remove only your recorded temporary directory when finished.

3. **Check CI, then tag (owner's explicit go-ahead only).** CI and the weekly
   upstream job must be green on main. Before tagging, verify required CI success
   for the **exact commit** to be tagged, including all required matrix jobs.
   Record the run URL, `headSha`, `status: completed` and `conclusion: success`.
   A successful run on an earlier SHA or a PR merge SHA that differs from the
   candidate does not qualify. CI runs on branch pushes/PRs, not tag pushes;
   `needs: build` only establishes the release build, not the separate full CI.
   If the candidate changes, obtain new matching evidence before continuing.

   The owner can inspect runs with this read-only command, then check their job
   details in GitHub (the command alone does not assert success):

   ```sh
   gh run list --workflow ci.yml --commit "$RELEASE_COMMIT" \
     --json url,headSha,event,status,conclusion
   ```

   After verifying environment `pypi`, its required reviewer and `v*` restriction,
   and receiving the owner's go-ahead, the owner tags that exact commit:

   ```sh
   git tag -a vX.Y.Z "$RELEASE_COMMIT" -m "X.Y.Z"
   git push origin vX.Y.Z
   ```

   Before approving the waiting `pypi` environment run, verify again that the tag
   resolves to the owner-approved commit with successful exact-head CI. Compare
   the release run's SHA and downloaded `release-dist` artifact's `release.json`:
   its tag, commit, version, two filenames and SHA256 hashes must match the files
   built and checked by that run. The build job tests versions and the installed
   README recipe before uploading. The publish job downloads only that run's
   checked artifact, verifies its hashes and tag commit, and publishes those
   files without rebuilding. Local candidate hashes record local evidence;
   the run's manifest identifies the actual publication bytes. Inspect the
   release workflow changes before approval. Only the owner approves publication.

4. **After publication (owner only).** Verify PyPI's version and artifact hashes,
   the Read the Docs tag build, release-version activation and documentation URLs.
   In a fresh venv outside the checkout, use
   `uv pip install --python <venv>/bin/python "swarm-arena==X.Y.Z"`, assert both
   installed metadata and `swarm_arena.__version__`, and run the CLI and README
   smoke with that venv. Do not claim these live installation checks from local
   build evidence. Remove the README fallback line only after the owner sees
   0.1.0 on PyPI, updating its guard test in the same follow-up. Creating a GitHub
   release, uploads to PyPI/TestPyPI, live provider/gateway checks and hosting
   changes all require separate owner action; preparing this checklist authorizes
   none of them.

5. **A bad release (owner only).** Yank it on PyPI, then prepare and verify
   X.Y.(Z+1). A published version number can never be reused; do not replace or
   overwrite its artifacts or move its release tag.

6. **Deprecations.** Follow the [Versioning policy](docs/versioning.md), including
   the spec §9 rule 1 deprecation notice and versioning commitments.

7. **Pinned publish actions.** The publish job, the only job that can mint a
   PyPI token, uses `actions/download-artifact` and
   `pypa/gh-action-pypi-publish` pinned to full commit SHAs, with the tag in a
   comment. To update one, resolve the new tag with
   `git ls-remote https://github.com/<owner>/<repo> 'refs/tags/<tag>^{}'`
   (or `refs/tags/<tag>` for a lightweight tag), review the release notes, and
   change the SHA, the comment and `test_publish_job_is_least_privilege`
   together.
