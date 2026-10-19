from pathlib import Path

import pytest

from swarm_arena._tooling.paths import RESERVED, check_glob, check_key, check_relpath, matches


@pytest.mark.parametrize("path", ["SWARM/x", "Parents/x", "Candidate/x", "\u017fwarm/x"])
def test_folded_reserved_names_are_refused(path: str) -> None:
    with pytest.raises(ValueError, match="reserved"):
        check_relpath(path)
    with pytest.raises(ValueError, match="reserved"):
        check_key(path)


def test_fold_key_normalizes_unicode_then_casefolds() -> None:
    from swarm_arena._tooling.paths import fold_key

    assert fold_key("PROBLEM/Cafe\u0301.py") == fold_key("problem/Caf\u00e9.py")
    assert fold_key("Straße/x") == "strasse/x"
    assert fold_key("a.txt") != fold_key("b.txt")


def test_check_relpath_normalizes() -> None:
    assert check_relpath("./problem//run.py") == "problem/run.py"
    assert check_relpath("swarmy/x") == "swarmy/x"


@pytest.mark.parametrize(
    "bad",
    ["", ".", "/abs", "a/../b", "..", "swarm/result.json", "parents/x", "candidate", "a\x00b"],
)
def test_check_relpath_rejects(bad: str) -> None:
    with pytest.raises(ValueError, match="path"):
        check_relpath(bad)


def test_check_key_requires_normal_form() -> None:
    assert check_key("p/a.txt") == "p/a.txt"
    for bad in ("./p/a.txt", "p//a.txt", "p/a.txt/"):
        with pytest.raises(ValueError, match="normal form"):
            check_key(bad)
    with pytest.raises(ValueError, match="reserved"):
        check_key("swarm/x")


@pytest.mark.parametrize(
    ("path", "globs", "ok"),
    [
        ("problem/a.phyk", ["problem/*.phyk"], True),
        ("problem/sub/a.phyk", ["problem/*.phyk"], False),
        ("problem/sub/a.phyk", ["problem/**/*.phyk"], True),
        ("problem/a.phyk", ["problem/**/*.phyk"], True),
        ("a.txt", ["*.txt"], True),
        ("d/a.txt", ["*.txt"], False),
        (".hidden", ["*"], True),
        ("p/a1.txt", ["p/a?.txt"], True),
        ("p/a12.txt", ["p/a?.txt"], False),
        ("p/b.txt", ["p/[ab].txt"], True),
        ("x", [], False),
    ],
)
def test_matches_dialect(path: str, globs: list[str], ok: bool) -> None:
    assert matches(path, globs) is ok


def test_matches_agrees_with_pathlib_glob(tmp_path: Path) -> None:
    files = ["problem/a.phyk", "problem/sub/b.phyk", "problem/run.py", "obs.npz", "problem/.h"]
    for filename in files:
        path = tmp_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    for pattern in ["problem/*", "problem/**/*.phyk", "*.npz", "problem/*.p?yk"]:
        assert {
            p.relative_to(tmp_path).as_posix() for p in tmp_path.glob(pattern) if p.is_file()
        } == {f for f in files if matches(f, [pattern])}


@pytest.mark.parametrize("bad", ["", "/abs/*", "../x/*", "a/../*"])
def test_check_glob_rejects(bad: str) -> None:
    with pytest.raises(ValueError, match="glob"):
        check_glob(bad)


def test_reserved_names() -> None:
    assert frozenset({"swarm", "parents", "candidate"}) == RESERVED


@pytest.mark.parametrize(
    ("path", "reason"),
    [
        ("./", "empty"),
        ("//abs", "absolute"),
        ("a/./../b", "contains '..'"),
        ("a\x00b", "contains NUL"),
        ("./parents/x", "uses the reserved name 'parents'"),
    ],
)
def test_check_relpath_error_messages(path: str, reason: str) -> None:
    with pytest.raises(ValueError, match="path") as exc:
        check_relpath(path)
    assert str(exc.value) == f"path {path!r}: {reason}"


@pytest.mark.parametrize("pattern", ["problem/**/*.phyk", "./p//*.txt", "swarm/*", "*"])
def test_check_glob_preserves_valid_pattern(pattern: str) -> None:
    assert check_glob(pattern) == pattern


def test_check_glob_rejects_nul() -> None:
    with pytest.raises(ValueError, match="contains NUL"):
        check_glob("a\x00*")


@pytest.mark.parametrize(
    ("path", "pattern", "ok"),
    [
        ("p", "p/**", True),
        ("p/a/b", "p/**", True),
        ("p", "p/**/**", True),
        ("p/a/b", "p/**/**", True),
        ("p/a/b", "**", True),
        ("p/a/b", "**/**", True),
        ("prefix/a", "p/**", False),
        ("p/a/b/c.txt", "p/**/*.txt", True),
        ("p/a.txt", "**/*.txt", True),
        ("a.txt", "**/*.txt", True),
        ("p/a.txt", "./p//*.txt", True),
        ("p/a/b.txt", "p/**.txt", False),
        ("p/a.txt", "p/[!b].txt", True),
        ("p/b.txt", "p/[!b].txt", False),
        ("p//.txt", "p/[!b].txt", False),
        ("p/a.txt", "p/[a-c].txt", True),
        ("p/].txt", "p/[]].txt", True),
        ("p/[.txt", "p/[.txt", True),
        ("p/z.txt", "p/[z-a].txt", False),
        ("p/a/b", "p/?", False),
        ("p/a\nb", "p/*", True),
        ("p/.hidden/a.txt", "p/**/*.txt", True),
        ("a.txt.bak", "*.txt", False),
        ("a.TXT", "*.txt", False),
    ],
)
def test_matches_edge_cases(path: str, pattern: str, ok: bool) -> None:
    assert matches(path, [pattern]) is ok


def test_matches_accepts_any_matching_glob() -> None:
    assert matches("p/a.txt", ("*.txt", "p/*.txt"))


def test_matches_rejects_invalid_glob() -> None:
    with pytest.raises(ValueError, match="glob"):
        matches("a.txt", ["../*.txt"])


@pytest.mark.parametrize(
    ("files", "directories", "conflict"),
    [
        (["p/a", "p/b"], [], None),
        (["p/a", "P/b"], [], ("p/a", "P/b")),
        (["p/a", "p"], [], ("p/a", "p")),
        (["p", "p/a"], [], ("p", "p/a")),
        (["p/a", "P"], [], ("p/a", "P")),
        (["p/a", "p/a"], [], ("p/a", "p/a")),
        (["p"], ["p"], ("p", "p")),
        (["p/a"], ["P"], ("p/a", "P")),
        (["p/a"], ["p"], None),
    ],
)
def test_shared_folded_layout(
    files: list[str], directories: list[str], conflict: tuple[str, str] | None
) -> None:
    from swarm_arena._tooling.paths import fold_conflicts

    conflicts = fold_conflicts(files, directories=directories)
    assert conflicts == ([] if conflict is None else [conflict])


def test_fold_merge_keeps_winning_spelling_without_mutation() -> None:
    from swarm_arena._tooling.paths import fold_merge

    base = {"p/a": "base", "p/b": "keep", "caf\u00e9/a": "base"}
    overlay = {"p/A": "winner", "cafe\u0301/a": "unicode"}
    assert fold_merge(base, overlay) == {"p/A": "winner", "p/b": "keep", "cafe\u0301/a": "unicode"}
    assert base == {"p/a": "base", "p/b": "keep", "caf\u00e9/a": "base"}
    assert overlay == {"p/A": "winner", "cafe\u0301/a": "unicode"}
