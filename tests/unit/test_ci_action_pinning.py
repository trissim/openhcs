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
RELEASE_VERSION = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
GITHUB_EXPRESSION_SOURCE_LIMIT = 21_000


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


def test_documentation_gate_owns_context_aware_workflow_validation() -> None:
    workflow = yaml.safe_load((WORKFLOW_ROOT / "docs.yml").read_text(encoding="utf-8"))
    lint_step = next(
        step
        for step in workflow["jobs"]["validate"]["steps"]
        if step.get("name") == "Validate GitHub Actions workflows"
    )

    assert RELEASE_VERSION.fullmatch(lint_step["env"]["ACTIONLINT_VERSION"])
    assert SHA256.fullmatch(lint_step["env"]["ACTIONLINT_LINUX_AMD64_SHA256"])
    assert "rhysd/actionlint/releases/download" in lint_step["run"]
    assert "sha256sum --check" in lint_step["run"]
    assert '"$executable" -shellcheck= -pyflakes=' in lint_step["run"]


def test_workflow_run_expressions_respect_hosted_parser_limit() -> None:
    oversized_expressions: list[str] = []
    for workflow_path in sorted(WORKFLOW_ROOT.glob("*.y*ml")):
        workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
        for job_name, job in workflow["jobs"].items():
            for step in job.get("steps", ()):
                run = step.get("run")
                if not isinstance(run, str) or "${{" not in run:
                    continue
                if len(run) > GITHUB_EXPRESSION_SOURCE_LIMIT:
                    oversized_expressions.append(
                        f"{workflow_path.name}:{job_name}:{step.get('name', '<unnamed>')}"
                    )

    assert oversized_expressions == []
