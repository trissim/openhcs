#!/usr/bin/env python3
"""Summarize OpenHCS RUNTIME_PROFILE timing logs."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import re
import statistics


_PROFILE_RE = re.compile(
    r"^\s*RUNTIME_PROFILE\s+(?P<label>\S+)\s+"
    r"(?P<seconds>[0-9]+(?:\.[0-9]+)?)s\b"
)


def _read_profile(path: Path) -> dict[str, list[float]]:
    samples_by_label: dict[str, list[float]] = defaultdict(list)
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = _PROFILE_RE.match(line)
            if match is None:
                continue
            samples_by_label[match.group("label")].append(float(match.group("seconds")))
    return dict(samples_by_label)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()

    rows = []
    for label, samples in _read_profile(args.profile).items():
        total = sum(samples)
        rows.append(
            (
                total,
                label,
                len(samples),
                min(samples),
                statistics.median(samples),
                max(samples),
            )
        )

    print("total_seconds,count,min_seconds,median_seconds,max_seconds,label")
    for total, label, count, min_seconds, median_seconds, max_seconds in sorted(
        rows,
        reverse=True,
    )[: args.top]:
        print(
            f"{total:.6f},{count},{min_seconds:.6f},"
            f"{median_seconds:.6f},{max_seconds:.6f},{label}"
        )


if __name__ == "__main__":
    main()
