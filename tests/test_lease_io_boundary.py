"""Guard retained owner modules against non-abandonable storage workers."""

import ast
from pathlib import Path

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
}


def _worker_calls(source: str) -> list[tuple[str, str, int]]:
    """Find executable worker calls, excluding documentation string mentions."""
    found: list[tuple[str, str, int]] = []
    scope: list[str] = []

    class Visitor(ast.NodeVisitor):
        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            scope.append(node.name)
            self.generic_visit(node)
            scope.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
            scope.append(node.name)
            self.generic_visit(node)
            scope.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Call(self, node: ast.Call) -> None:
            if ast.unparse(node.func).endswith("to_thread.run_sync"):
                callback = node.args[0] if node.args else ast.Constant(value=None)
                if (
                    isinstance(callback, ast.Call)
                    and ast.unparse(callback.func) == "partial"
                    and callback.args
                ):
                    callback = callback.args[0]
                found.append((".".join(scope), ast.unparse(callback), node.lineno))
            self.generic_visit(node)

    Visitor().visit(ast.parse(source))
    return found


def test_lease_owned_modules_use_abandonable_workers() -> None:
    paths = [
        *_ROOT.joinpath("_engine").glob("*.py"),
        _ROOT / "_journal/writer.py",
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
