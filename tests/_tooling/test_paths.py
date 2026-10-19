import sys
import unicodedata
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


def test_fold_merge_keeps_the_first_spelling_without_mutation() -> None:
    # One spelling rule for every stage: a later layer replaces the content, never the name.
    from swarm_arena._tooling.paths import fold_merge

    base = {"p/a": "base", "p/b": "keep", "caf\u00e9/a": "base"}
    overlay = {"p/A": "winner", "cafe\u0301/a": "unicode", "p/c": "new"}
    assert fold_merge(base, overlay) == {
        "p/a": "winner",
        "p/b": "keep",
        "caf\u00e9/a": "unicode",
        "p/c": "new",
    }
    assert base == {"p/a": "base", "p/b": "keep", "caf\u00e9/a": "base"}
    assert overlay == {"p/A": "winner", "cafe\u0301/a": "unicode", "p/c": "new"}


@pytest.mark.parametrize(
    ("base", "overlay"),
    [
        ({"p/a.txt": "x", "P/b.txt": "y"}, {}),
        ({}, {"p/a.txt": "x", "P/b.txt": "y"}),
        ({"p/a.txt": "x"}, {"P/b.txt": "y"}),
    ],
)
def test_fold_merge_errors_name_the_tree(base: dict[str, str], overlay: dict[str, str]) -> None:
    from swarm_arena._tooling.paths import fold_merge

    with pytest.raises(ValueError, match="conflicting workspace path") as exc:
        fold_merge(base, overlay, prefix="parents/c2/")
    assert str(exc.value) == (
        "conflicting workspace path: parents/c2/p/a.txt and parents/c2/P/b.txt"
    )


def test_parent_data_name_is_reserved_for_files_and_directories() -> None:
    from swarm_arena._tooling.paths import parent_data_conflicts, parent_tree

    paths = ["data.json/x", "Data.json/y", "DATA.JSON", "p/data.json/x", "data.jsonl/x"]
    assert parent_data_conflicts(paths) == ["data.json/x", "Data.json/y"]
    # The sink skips a whole-path alias of the generated file instead.
    assert parent_tree({"DATA.JSON": "o", "p/a": "o"}, {"Data.json": "f", "p/A": "f"}) == {
        "p/a": "f"
    }


# Canonical caseless classes (Unicode D145) that NFC-then-casefold split. APFS
# treats each pair as one file name; fold_key must give each pair one key.
CASELESS_PAIRS = [
    ("\u015a", "\u017f\u0301"),
    ("\u015b", "\u017f\u0301"),
    ("\u015c", "\u017f\u0302"),
    ("\u015e", "\u017f\u0327"),
    ("\u0160", "\u017f\u030c"),
    ("\u0161", "\u017f\u030c"),
    ("\u0218", "\u017f\u0326"),
    ("\u0388", "\u03f5\u0301"),
    ("\u0390", "\u0399\u0344"),
    ("\u03b0", "\u03a5\u0344"),
    ("\u04e6", "\u1c82\u0308"),
    ("\u1e62", "\u017f\u0323"),
    ("\u1e68", "\u1e9b\u0323"),
    ("\u1f10", "\u03f5\u0313"),
    ("\u1f11", "\u03f5\u0314"),
    ("\u1f72", "\u03f5\u0300"),
    ("\u1fb7", "\u1fbc\u0342"),
    ("\u1fc7", "\u1fcc\u0342"),
    ("\u1fd2", "\u03aa\u0300"),
    ("\u1fd7", "\u03aa\u0342"),
    ("\u1fe2", "\u03ab\u0300"),
    ("\u1fe5", "\u03f1\u0314"),
    ("\u1fe7", "\u03ab\u0342"),
    ("\u1ff7", "\u1ffc\u0342"),
    ("\u0124\u0331", "\u0125\u0331"),
    ("\u015e\u0307", "\u1e9b\u0327"),
    ("\u0218\u0307", "\u1e9b\u0326"),
    ("\u021e\u0331", "\u021f\u0331"),
    ("\u041e\u0344", "\u1c82\u0344"),
    ("\u1e22\u0331", "\u1e23\u0331"),
    ("\u1e26\u0331", "\u1e27\u0331"),
]


@pytest.mark.parametrize(("first", "second"), CASELESS_PAIRS)
def test_fold_key_joins_canonical_caseless_pairs(first: str, second: str) -> None:
    from swarm_arena._tooling.paths import fold_conflicts, fold_key

    assert fold_key(f"p/{first}.py") == fold_key(f"p/{second}.py")
    assert fold_conflicts([f"p/{first}.py", f"p/{second}.py"]) != []


def test_fold_key_is_canonical_caseless() -> None:
    """Property: one key per D145 class, so NFC, NFD and case folding never change it."""
    from swarm_arena._tooling.paths import fold_key

    def nfc(text: str) -> str:
        return unicodedata.normalize("NFC", text)

    def nfd(text: str) -> str:
        return unicodedata.normalize("NFD", text)

    singles = [chr(code) for code in range(0x80, 0x30000) if not 0xD800 <= code <= 0xDFFF]
    marks = [chr(code) for code in range(0x300, 0x370)] + ["\u0345"]
    cased = [c for c in singles if ord(c) < 0x2000 and (c.casefold() != c or nfd(c) != c)]
    split = []
    for text in [*singles, *(c + mark for c in cased for mark in marks)]:
        key = fold_key(text)
        variants = (key, nfc(text), nfd(text), nfd(text).casefold())
        if any(fold_key(variant) != key for variant in variants):
            split.append(ascii(text))
    assert split == []


def _case_insensitive(directory: Path) -> bool:
    probe = directory / "case-probe"
    probe.touch()
    try:
        return (directory / "CASE-PROBE").exists()
    finally:
        probe.unlink()


@pytest.mark.skipif(sys.platform != "darwin", reason="pins APFS name matching")
@pytest.mark.parametrize(("first", "second"), CASELESS_PAIRS)
def test_caseless_pairs_are_one_file_on_apfs(tmp_path: Path, first: str, second: str) -> None:
    if not _case_insensitive(tmp_path):
        pytest.skip("needs a case-insensitive volume")
    (tmp_path / first).write_text("x")
    assert (tmp_path / second).exists()
    assert [entry.name for entry in tmp_path.iterdir()] == [first]
