"""Guard retained owner modules against non-abandonable storage workers."""

import ast
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1] / "src" / "swarm_arena"
# Exact module, enclosing function, and worker callback; no module-wide escapes.
_ALLOW = {
    ("_engine/run.py", "RunRunner.run", "admit_revision"): "Admission precedes acquisition.",
    ("_engine/run.py", "RunRunner.run", "_promote"): "Artifact promotion precedes acquisition.",
    ("_engine/manifest.py", "build_manifest_async", "_build_manifest"): "Preownership host probes.",
    ("_engine/control.py", "request_stop", "_request_stop"): "Stop command owns no lease.",
    ("_engine/control.py", "read_status", "_read_status"): "Status reader owns no lease.",
    ("_scoring/human_review.py", "HumanReview.export", "run_store"): "Export owns no lease.",
    ("_scoring/human_review.py", "HumanReview.export", "_read_state"): "Export owns no lease.",
    ("_scoring/human_review.py", "HumanReview.export", "_write_export"): "Export owns no lease.",
    (
        "_scoring/human_review.py",
        "HumanReview.import_csv",
        "_read_csv",
    ): "CSV read before takeover.",
    ("_scoring/human_review.py", "HumanReview.import_csv", "run_store"): "Store before takeover.",
    (
        "_scoring/human_review.py",
        "HumanReview.import_csv",
        "_read_state",
    ): "Pre-takeover validation.",
    ("_scoring/human_review.py", "HumanReview.import_csv", "check_storage"): "Pre-takeover probes.",
    ("_tooling/script_tool.py", "_list_files", "_list_files_sync"): "In-memory manifest/view only.",
    ("_exec/local_run.py", "_JobRun.run", "self._stopped_output"): "Local regular-file notes only.",
    ("_exec/local_run.py", "_JobRun._deliver_partial", "self._stopped_output"): "Local notes only.",
    ("_exec/local_run.py", "_JobRun._deliver_partial", "note_in_log"): "Local regular-file note.",
    ("_exec/local_run.py", "_JobRun._wait_post_collect", "note_in_log"): "Local regular-file note.",
    ("_exec/local_run.py", "_JobRun._finish", "note_in_log"): "Local regular-file note.",
    ("testing/_harness.py", "OfflineHarness.run_rounds", "self._run_folder"): "Before acquisition.",
    ("testing/_harness.py", "OfflineHarness.run_rounds", "_build"): "Preownership snapshots.",
    (
        "testing/_harness.py",
        "OfflineHarness.run_rounds",
        "execution_ref.build",
    ): "Before acquisition.",
    (
        "testing/_harness.py",
        "OfflineHarness.run_rounds",
        "Lease.acquire_new",
    ): "Initial acquisition.",
    ("testing/_harness.py", "OfflineHarness.run_rounds", "_failed_runs"): "After writer release.",
}


_STORE_METHODS = {"create", "read", "replace", "exists", "delete"}
_LEASE_METHODS = {"acquire_new", "take_over", "renew", "release", "advance"}


def _storage_method(node: ast.expr, *, lease: bool = False) -> bool:
    """Recognize the explicit store/lease receiver names used in owner modules."""
    if not isinstance(node, ast.Attribute):
        return False
    receiver = ast.unparse(node.value).rsplit(".", 1)[-1].lower().lstrip("_")
    store = receiver in {"store", "arena"} or receiver.endswith("_store")
    owner = receiver == "lease" or receiver.endswith("_lease")
    return (store and node.attr in _STORE_METHODS) or (
        lease and owner and node.attr in _LEASE_METHODS
    )


def _worker_calls(source: str) -> list[tuple[str, str, int]]:
    """Find bare workers and synchronous storage calls, ignoring docstrings."""
    tree = ast.parse(source)
    imports: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                imports[name.asname or name.name.split(".")[0]] = (
                    name.name if name.asname else name.name.split(".")[0]
                )
        elif isinstance(node, ast.ImportFrom):
            for name in node.names:
                imports[name.asname or name.name] = f"{node.module}.{name.name}"

    def resolved(node: ast.expr) -> str:
        name, *attributes = ast.unparse(node).split(".")
        return ".".join([imports.get(name, name), *attributes])

    found: list[tuple[str, str, int]] = []
    scope: list[str] = []
    asynchronous: list[bool] = []

    class Visitor(ast.NodeVisitor):
        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            scope.append(node.name)
            self.generic_visit(node)
            scope.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
            scope.append(node.name)
            asynchronous.append(isinstance(node, ast.AsyncFunctionDef))
            self.generic_visit(node)
            asynchronous.pop()
            scope.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Call(self, node: ast.Call) -> None:
            function = resolved(node.func)
            callback = node.args[0] if node.args else ast.Constant(value=None)
            if (
                isinstance(callback, ast.Call)
                and resolved(callback.func) in {"partial", "functools.partial"}
                and callback.args
            ):
                callback = callback.args[0]
            if function.endswith("to_thread.run_sync") or (
                function.endswith("run_sync_daemon") and _storage_method(callback, lease=True)
            ):
                found.append((".".join(scope), ast.unparse(callback), node.lineno))
            elif asynchronous and asynchronous[-1] and _storage_method(node.func):
                found.append((".".join(scope), ast.unparse(node.func), node.lineno))
            self.generic_visit(node)

    Visitor().visit(tree)
    return found


def test_lease_owned_modules_use_abandonable_workers() -> None:
    paths = [
        *_ROOT.joinpath("_engine").glob("*.py"),
        _ROOT / "_journal/writer.py",
        _ROOT / "_journal/views.py",
        _ROOT / "_journal/lease.py",
        _ROOT / "_agent/context.py",
        _ROOT / "testing/_harness.py",
        _ROOT / "_exec/local_run.py",
        _ROOT / "_scoring/human_review.py",
        _ROOT / "_agent/tool_run.py",
        _ROOT / "_tooling/script_tool.py",
    ]
    violations = []
    used = set()
    for path in sorted(paths):
        relative = path.relative_to(_ROOT).as_posix()
        for function, callback, line in _worker_calls(path.read_text()):
            key = relative, function, callback
            if key not in _ALLOW:
                violations.append(f"{relative}:{line}: {function}: {callback}")
            else:
                used.add(key)
    assert not violations, "unprotected worker calls:\n" + "\n".join(violations)
    assert used == set(_ALLOW), "remove obsolete worker exceptions"
    assert all(_ALLOW.values())


def test_guard_finds_nested_calls_and_ignores_docstrings() -> None:
    source = '''
"""anyio.to_thread.run_sync(store.read) is documentation."""
class Owner:
    async def publish(self):
        async def nested():
            return await anyio.to_thread.run_sync(store.create, "key", b"data")
'''
    assert _worker_calls(source) == [("Owner.publish.nested", "store.create", 6)]


@pytest.mark.parametrize(
    ("import_line", "worker"),
    [
        ("from anyio.to_thread import run_sync", "run_sync"),
        ("from anyio.to_thread import run_sync as worker", "worker"),
        ("from anyio import to_thread as threads", "threads.run_sync"),
        ("import anyio as aio", "aio.to_thread.run_sync"),
        ("import anyio.to_thread as threads", "threads.run_sync"),
    ],
)
def test_guard_resolves_worker_aliases(import_line: str, worker: str) -> None:
    source = f'{import_line}\nasync def publish():\n    await {worker}(store.read, "key")'
    assert _worker_calls(source) == [("publish", "store.read", 3)]


@pytest.mark.parametrize("method", ["create", "read", "replace", "exists", "delete"])
def test_guard_finds_direct_async_store_calls(method: str) -> None:
    source = f'async def publish():\n    store.{method}("key")'
    assert _worker_calls(source) == [("publish", f"store.{method}", 2)]


@pytest.mark.parametrize(
    "callback", ["store.create", "self.store.read", "lease.renew", "Lease.take_over"]
)
def test_guard_finds_uncontained_daemon_storage(callback: str) -> None:
    source = f'async def publish():\n    await run_sync_daemon({callback}, "key")'
    assert _worker_calls(source) == [("publish", callback, 2)]


def test_guard_accepts_storage_helpers_and_synchronous_primitives() -> None:
    source = """
async def publish():
    def primitive():
        return store.read("key")
    await _storage_io(store.create, "key", b"data")
    await _validated_io(read_chain, store, head)
    await _storage_io(lease.renew)
"""
    assert _worker_calls(source) == []
