#!/usr/bin/env python3
"""Exercise one real staged desktop update without the graphical progress shell."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from packaging.version import Version

from openhcs.pyqt_gui.services.desktop_update_worker import (
    DesktopUpdatePlan,
    DesktopUpdateProgressAction,
    DesktopUpdateProgressReporterABC,
    ResolvedProcessLaunchSpec,
    _run_update,
)


class _ConsoleProgress(DesktopUpdateProgressReporterABC):
    def phase(self, phase) -> None:
        print(f"PHASE: {phase.value}", file=sys.stderr, flush=True)

    def output(self, message: str) -> None:
        print(message, file=sys.stderr, flush=True)

    def failure(self, message: str) -> DesktopUpdateProgressAction:
        print(f"FAILURE: {message}", file=sys.stderr, flush=True)
        return DesktopUpdateProgressAction.EXIT

    def complete(self) -> None:
        print("COMPLETE", file=sys.stderr, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    plan_source = parser.add_mutually_exclusive_group(required=True)
    plan_source.add_argument("--plan", type=Path)
    plan_source.add_argument("--latest-version", type=Version)
    arguments = parser.parse_args()
    if arguments.plan is not None:
        plan = DesktopUpdatePlan.read(arguments.plan)
    else:
        from openhcs.pyqt_gui.services.desktop_update import (
            DesktopRuntimeEnvironment,
        )

        assert arguments.latest_version is not None
        plan = DesktopRuntimeEnvironment.current().update_plan(arguments.latest_version)
    execution = _run_update(
        plan,
        launch_spec=ResolvedProcessLaunchSpec(
            creationflags=0,
            start_new_session=False,
        ),
        progress=_ConsoleProgress(),
    )
    print(json.dumps(asdict(execution), sort_keys=True))
    return int(execution.error_message is not None)


if __name__ == "__main__":
    raise SystemExit(main())
