#!/usr/bin/env python
"""Enforce the single-owner file map in OWNERS.yaml.

Failure modes (v2 plan §9.5):
  (a) a file owned by the other person changed without the `cross-owner` label;
  (b) a contract file changed without the `contract` label;
  (c) a new file is reached by no pattern in OWNERS.yaml.

The PR author lane is derived from the branch prefix (`qamran/...`,
`warisha/...`) or passed explicitly with --author. Locally, with no base
branch, the check is a no-op success.
"""

from __future__ import annotations

import argparse
import fnmatch
import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _glob_to_regex(pattern: str) -> re.Pattern[str]:
    """Translate an OWNERS glob (`**`, `*`, `{a,b}`) into a regex."""
    # Expand simple brace alternatives first.
    m = re.search(r"\{([^{}]+)\}", pattern)
    if m:
        alternatives = [
            pattern[: m.start()] + alt + pattern[m.end() :] for alt in m.group(1).split(",")
        ]
        return re.compile("|".join(f"(?:{_glob_to_regex(a).pattern})" for a in alternatives))

    out = []
    i = 0
    while i < len(pattern):
        if pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return re.compile("^" + "".join(out) + "$")


def matches(path: str, pattern: str) -> bool:
    path = path.replace("\\", "/")
    if fnmatch.fnmatchcase(path, pattern):
        return True
    return bool(_glob_to_regex(pattern).match(path))


def load_owners() -> tuple[dict[str, list[str]], list[str]]:
    data = yaml.safe_load((ROOT / "OWNERS.yaml").read_text(encoding="utf-8")) or {}
    owners = data.get("owners", {}) or {}
    contract = data.get("contract", []) or []
    # Any file under a directory wildcard is owned for coverage purposes.
    return {str(k): [str(p) for p in (v or [])] for k, v in owners.items()}, [
        str(p) for p in contract
    ]


def owner_of(path: str, owners: dict[str, list[str]]) -> str | None:
    for owner in ("qamran", "warisha"):
        for pattern in owners.get(owner, []):
            if matches(path, pattern):
                return owner
    return None


def changed_files(base: str | None) -> list[str]:
    if not base:
        return []
    try:
        out = subprocess.run(
            ["git", "diff", "--name-only", "--diff-filter=ACMR", base, "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except subprocess.CalledProcessError as exc:  # pragma: no cover - CI guard
        print(f"git diff failed: {exc.stderr}", file=sys.stderr)
        raise SystemExit(2) from exc
    return [line.strip() for line in out.splitlines() if line.strip()]


def author_from_branch(branch: str | None) -> str | None:
    if not branch:
        return None
    prefix = branch.split("/", 1)[0].lower()
    return prefix if prefix in {"qamran", "warisha"} else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=None, help="git revision to diff against")
    parser.add_argument("--branch", default=None, help="PR head branch")
    parser.add_argument("--author", default=None, choices=["qamran", "warisha"])
    parser.add_argument("--labels", default="", help="comma-separated PR labels")
    parser.add_argument("--files", nargs="*", default=None, help="explicit file list (tests)")
    args = parser.parse_args(argv)

    owners, contract = load_owners()
    labels = {label.strip() for label in args.labels.split(",") if label.strip()}

    author = args.author or author_from_branch(args.branch)
    files = args.files if args.files is not None else changed_files(args.base)

    if not files:
        print("ownership: nothing to check")
        return 0
    if author is None:
        print(
            "ownership: cannot derive author lane; pass --author or use a "
            "qamran/... or warisha/... branch",
            file=sys.stderr,
        )
        return 2

    other = "warisha" if author == "qamran" else "qamran"
    errors: list[str] = []

    for path in files:
        owner = owner_of(path, owners)
        if owner is None:
            errors.append(f"{path}: new file is not covered by OWNERS.yaml (add it in the same PR)")
            continue
        if owner == other and "cross-owner" not in labels:
            errors.append(f"{path}: owned by {owner}; {author} needs the `cross-owner` label")
        is_contract = any(matches(path, pattern) for pattern in contract)
        if is_contract and "contract" not in labels:
            errors.append(f"{path}: contract file changed without the `contract` label")

    if errors:
        print("ownership check FAILED:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print(f"ownership: {len(files)} file(s) OK for {author}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
