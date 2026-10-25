"""ds1000 — owner: Enea, due: M1 Thu 29 Oct.

TODO:
Load DS-1000 from the official release, keep Pandas, Numpy, Matplotlib, Sklearn, Scipy. Build Task objects
(never include reference solutions), execute with the official test code through the sandbox, --limit flag.
"""

from __future__ import annotations

from pathlib import Path

VERSIONS_TXT = Path(__file__).parent / "sandbox" / "versions.txt"

# DS-1000 metadata["library"] -> our Chunk.library names
LIBRARIES = {
    "Pandas": "pandas",
    "Numpy": "numpy",
    "Matplotlib": "matplotlib",
    "Sklearn": "scikit-learn",
    "Scipy": "scipy",
}


def pinned_versions(path: Path = VERSIONS_TXT) -> dict[str, str]:
    """Read `name==version` lines from the sandbox requirements, skipping comments and blank lines."""
    pins = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if "==" in line:
            name, version = line.split("==", 1)
            pins[name.strip()] = version.strip()
    return pins


def target_versions(ds1000_library: str) -> dict[str, str]:
    """Task.target_versions for a DS-1000 problem: its own library at the version DS-1000 pins."""
    library = LIBRARIES[ds1000_library]
    return {library: pinned_versions()[library]}
