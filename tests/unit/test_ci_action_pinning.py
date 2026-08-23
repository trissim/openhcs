from __future__ import annotations

import re
from pathlib import Path

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_ROOT = REPOSITORY_ROOT / ".github" / "workflows"
DEPENDABOT_PATH = REPOSITORY_ROOT / ".github" / "dependabot.yml"
USES_LINE = re.compile(r"^\s*(?:-\s*)?uses:\s*(?P<target>[^#\s]+)")
IMMUTABLE_ACTION = re.compile(r"[^@\s]+@[0-9a-f]{40}")
IMMUTABLE_CONTAINER = re.compile(r"docker://[^@\s]+@sha256:[0-9a-f]{64}")


def test_external_workflow_dependencies_are_immutable() -> None:
    mutable_dependencies: list[str] = []
    for workflow_path in sorted(WORKFLOW_ROOT.glob("*.y*ml")):
        for line_number, line in enumerate(
            workflow_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            match = USES_LINE.match(line)
            if match is None:
                continue
            target = match.group("target")
            if target.startswith("./"):
                continue
            if IMMUTABLE_ACTION.fullmatch(target) is not None:
                continue
            if IMMUTABLE_CONTAINER.fullmatch(target) is not None:
                continue
            relative_path = workflow_path.relative_to(REPOSITORY_ROOT)
            mutable_dependencies.append(f"{relative_path}:{line_number}: {target}")

    assert mutable_dependencies == []


def test_dependabot_owns_github_action_pin_updates() -> None:
    configuration = yaml.safe_load(DEPENDABOT_PATH.read_text(encoding="utf-8"))
    action_updates = next(
        update
        for update in configuration["updates"]
        if update["package-ecosystem"] == "github-actions"
    )

    assert action_updates["directory"] == "/"
    assert action_updates["schedule"] == {"interval": "weekly"}
    assert action_updates["groups"] == {
        "github-actions": {"patterns": ["*"]},
    }
