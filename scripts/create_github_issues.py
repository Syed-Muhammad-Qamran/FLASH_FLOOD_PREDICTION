"""Create the sprint issue set in a GitHub repository.

Reads docs/github/issues.json and files every issue, plus the labels it needs.

    # dry run: validates the manifest, touches nothing
    python scripts/create_github_issues.py --dry-run

    # real run (needs a token with repo scope)
    $env:GH_TOKEN = "ghp_..."          # PowerShell
    python scripts/create_github_issues.py

    # or, if the GitHub CLI is installed and logged in
    python scripts/create_github_issues.py --use-gh

The token is read from GH_TOKEN or GITHUB_TOKEN. It is never written to disk.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

API = "https://api.github.com"
MANIFEST = Path(__file__).resolve().parents[1] / "docs" / "github" / "issues.json"


def _token(use_gh: bool) -> str | None:
    if use_gh:
        try:
            out = subprocess.run(
                ["gh", "auth", "token"], capture_output=True, text=True, check=True
            )
            return out.stdout.strip() or None
        except Exception:
            print("Could not read a token from the GitHub CLI.", file=sys.stderr)
            return None
    return os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")


def _request(
    method: str,
    url: str,
    token: str,
    payload: dict[str, Any] | None = None,
) -> tuple[int, Any]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    req.add_header("User-Agent", "floodwatch-sprint-setup")
    req.add_header("Authorization", f"Bearer {token}")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            body = resp.read().decode("utf-8")
            return resp.status, (json.loads(body) if body else None)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = body
        return exc.code, parsed


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    for key in ("repo", "labels", "issues"):
        if key not in manifest:
            raise ValueError(f"manifest is missing '{key}'")
    return manifest


def dry_run(manifest: dict[str, Any]) -> int:
    print(f"repo   : {manifest['repo']}")
    print(f"labels : {len(manifest['labels'])}")
    print(f"issues : {len(manifest['issues'])}\n")
    for issue in manifest["issues"]:
        assignees = ", ".join(issue.get("assignees") or []) or "-"
        labels = ", ".join(issue.get("labels") or []) or "-"
        print(f"  {issue['key']:<10} {issue['title']}")
        print(f"             labels: {labels}")
        print(f"             assignees: {assignees}")
        body = issue.get("body", "")
        if not body.strip():
            print("             ERROR: empty body")
            return 1
    print("\nManifest is valid. Nothing was sent.")
    return 0


def ensure_labels(repo: str, labels: list[dict[str, Any]], token: str) -> None:
    for label in labels:
        status, body = _request(
            "POST", f"{API}/repos/{repo}/labels", token, payload=label
        )
        if status in (200, 201):
            print(f"  label created : {label['name']}")
        elif status == 422:
            print(f"  label exists  : {label['name']}")
        else:
            print(f"  label FAILED  : {label['name']} -> {status} {body}")


def create_issues(
    repo: str, issues: list[dict[str, Any]], token: str
) -> tuple[dict[str, int], int]:
    created: dict[str, int] = {}
    failures = 0
    for issue in issues:
        payload: dict[str, Any] = {
            "title": issue["title"],
            "body": issue.get("body", ""),
            "labels": issue.get("labels", []),
        }
        if issue.get("assignees"):
            payload["assignees"] = issue["assignees"]

        status, body = _request(
            "POST", f"{API}/repos/{repo}/issues", token, payload=payload
        )
        if status == 201:
            created[issue["key"]] = body["number"]
            print(f"  #{body['number']:<4} {issue['title']}")
        else:
            failures += 1
            print(f"  FAILED  {issue['title']} -> {status} {body}")
    return created, failures


def link_subissues(repo: str, epic_number: int, created: dict[str, int], token: str) -> None:
    """Append a sub-issue list to the epic body."""
    status, body = _request("GET", f"{API}/repos/{repo}/issues/{epic_number}", token)
    if status != 200:
        print(f"  could not read epic #{epic_number} to link sub-issues")
        return

    lines = ["", "## Sub-issues", ""]
    for key, number in created.items():
        if number == epic_number:
            continue
        lines.append(f"- [ ] #{number} - `{key}`")
    new_body = (body.get("body") or "") + "\n".join(lines)

    status, _ = _request(
        "PATCH", f"{API}/repos/{repo}/issues/{epic_number}", token, payload={"body": new_body}
    )
    if status == 200:
        print(f"  linked {len(created) - 1} sub-issues into the epic")
    else:
        print(f"  failed to link sub-issues -> {status}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="File the FloodWatch sprint issues.")
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--repo", default=None, help="owner/name; defaults to the manifest")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--use-gh", action="store_true", help="read the token from `gh auth token`")
    args = ap.parse_args(argv)

    manifest = load_manifest(Path(args.manifest))
    repo = args.repo or manifest["repo"]

    if args.dry_run:
        return dry_run(manifest)

    token = _token(args.use_gh)
    if not token:
        print(
            "No token found.\n"
            "  Set GH_TOKEN, or install the GitHub CLI and pass --use-gh.\n"
            "  Run with --dry-run to validate the manifest without a token.",
            file=sys.stderr,
        )
        return 2

    print(f"repo: {repo}\n")
    print("labels:")
    ensure_labels(repo, manifest["labels"], token)

    print("\nissues:")
    created, failures = create_issues(repo, manifest["issues"], token)

    epic = created.get("epic")
    if epic and len(created) > 1:
        print("\nlinking:")
        link_subissues(repo, epic, created, token)

    print(f"\ncreated {len(created)} issue(s), {failures} failure(s)")
    for key, number in created.items():
        print(f"  {key:<10} https://github.com/{repo}/issues/{number}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
