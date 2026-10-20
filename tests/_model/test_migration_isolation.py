import subprocess
import sys
from pathlib import Path


def test_shared_migration_fixture_restores_registry(tmp_path: Path) -> None:
    shared = Path(__file__).with_name("conftest.py")
    assert shared.is_file(), "migration isolation must be available to all model tests"
    (tmp_path / "conftest.py").write_text(shared.read_text())
    (tmp_path / "test_registry.py").write_text(
        """
import pytest
from swarm_arena._model import migrate

original_functions = migrate._migration_functions
original_view = migrate._MIGRATIONS
migrate.migration("existing_fixture_probe", 1)(lambda raw: raw)
original_contents = dict(original_view)

@pytest.mark.parametrize("fail", [True, False])
def test_isolated(request, fail):
    assert migrate._migration_functions is original_functions
    assert migrate._MIGRATIONS is original_view
    assert dict(original_view) == original_contents
    request.getfixturevalue("isolated_migrations")
    assert not migrate._MIGRATIONS
    assert migrate._migration_functions is not original_functions
    with pytest.raises(TypeError):
        migrate._MIGRATIONS[("fixture_probe", 1)] = lambda raw: raw
    step = lambda raw: raw
    migrate.migration("fixture_probe", 1)(step)
    assert migrate._MIGRATIONS[("fixture_probe", 1)] is step
    assert ("fixture_probe", 1) not in original_view
    if fail:
        pytest.fail("intentional fixture teardown probe")

def test_restored():
    assert migrate._migration_functions is original_functions
    assert migrate._MIGRATIONS is original_view
    assert dict(original_view) == original_contents
"""
    )
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-c", "/dev/null", str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "intentional fixture teardown probe" in result.stdout
    assert "1 failed, 2 passed" in result.stdout, result.stdout + result.stderr
