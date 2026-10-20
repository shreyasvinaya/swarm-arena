# Versioning policy

This package is in prerelease development. Stored records carry a
`schema_version` and reject newer unsupported formats. See the
[board API](api/board.md) for the current record contracts. A stable release
compatibility policy will accompany the first stable release.

## Python compatibility

The package requires Python 3.11 or newer. CI tests Python 3.11, 3.12, and 3.13.
Public classes report their facade module as their import path. Publishing clears
the original class line metadata on Python 3.13, so source inspection does not
mistake lines in that facade for the class definition. Docstring validation still
checks every public object.
