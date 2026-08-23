#!/usr/bin/env python3
"""Create the one official OpenHCS release tag from proven readiness evidence."""

from __future__ import annotations

import subprocess

from scripts.release_readiness import ReleaseReadiness, ReleaseReadinessError


def main() -> int:
    try:
        readiness = ReleaseReadiness.prove()
    except ReleaseReadinessError as exc:
        print(f"Release aborted: {exc}")
        return 1

    package_version = readiness.package_version
    response = input(
        f"Create release v{package_version} from "
        f"{readiness.repository.commit.sha}? [y/N] "
    )
    if response.lower() != "y":
        print("Aborted.")
        return 0

    tag = f"v{package_version}"
    try:
        subprocess.run(
            ["git", "tag", "-a", tag, "-m", f"Release version {package_version}"],
            check=True,
        )
        subprocess.run(
            ["git", "push", readiness.repository.remote, tag],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        print(f"Release failed: {exc}")
        return 1

    print(
        f"Created and pushed {tag} from proven commit {readiness.repository.commit.sha}."
    )
    print(
        "Monitor publication at: "
        f"https://github.com/{readiness.repository.github_repository}/actions"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
