"""Create GitHub labels, milestones and issues from docs/tasks.yaml.

Usage: python scripts/create_issues.py [--dry-run]
Needs the GitHub CLI (`gh`) logged in, run from inside the repo. Safe to re-run: existing issue titles are skipped.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

MILESTONES = {
    "M0": ("M0 Setup", "2026-10-26"),
    "M1": ("M1 Baseline + first retrieval", "2026-10-29"),
    "M2": ("M2 End-to-end on pandas slice", "2026-11-01"),
    "M3": ("M3 All configs run once", "2026-11-08"),
    "M4": ("M4 Final results frozen", "2026-11-15"),
    "M5": ("M5 Report and slides", "2026-11-22"),
}


def gh(*args: str, dry: bool) -> str:
    if dry:
        print("gh", " ".join(args))
        return ""
    return subprocess.run(["gh", *args], check=True, capture_output=True, text=True).stdout


def main() -> None:
    dry = "--dry-run" in sys.argv
    tasks = yaml.safe_load(Path("docs/tasks.yaml").read_text(encoding="utf-8"))

    labels = {f"owner:{t['owner']}" for t in tasks} | {f"area:{t['area']}" for t in tasks}
    for label in sorted(labels):
        subprocess.run(["gh", "label", "create", label, "--force"], capture_output=True) if not dry else print(
            "label", label
        )

    repo = "" if dry else json.loads(gh("repo", "view", "--json", "nameWithOwner", dry=dry))["nameWithOwner"]
    existing_ms = set() if dry else {
        m["title"] for m in json.loads(gh("api", f"repos/{repo}/milestones?state=all", dry=dry) or "[]")
    }
    for title, due in MILESTONES.values():
        if title not in existing_ms:
            gh("api", f"repos/{repo}/milestones", "-f", f"title={title}", "-f", f"due_on={due}T21:00:00Z", dry=dry)

    listing = [] if dry else json.loads(
        gh("issue", "list", "--state", "all", "--limit", "500", "--json", "title", dry=dry)
    )
    existing = {i["title"] for i in listing}
    for t in tasks:
        if t["title"] in existing:
            continue
        body = (
            f"Owner: **{t['owner']}** · Due: **{t['due']}**\n\n"
            "See docs/PLAN.md for details and the definition of done."
        )
        gh("issue", "create", "--title", t["title"], "--body", body,
           "--label", f"owner:{t['owner']}", "--label", f"area:{t['area']}",
           "--milestone", MILESTONES[t["milestone"]][0], dry=dry)
    print("done")


if __name__ == "__main__":
    main()
