#!/usr/bin/env python3
"""Validate deployment build inputs from the declarations inside a wheel."""

from __future__ import annotations

import argparse
import fnmatch
import json
from pathlib import Path, PurePosixPath
import re
import shlex
from zipfile import ZipFile


DEVELOPER_HOME_PATH_PATTERN = re.compile(
    r"(?:/(?:home|Users)/[^/\s'\"]+|[A-Za-z]:\\Users\\[^\\\s'\"]+)"
)


def _member_path(base: PurePosixPath, value: str) -> PurePosixPath:
    """Resolve a wheel member path without allowing it to escape the wheel."""

    parts: list[str] = []
    for part in (base / value).parts:
        if part in {"", "."}:
            continue
        if part == "..":
            if not parts:
                raise ValueError(f"Deployment path escapes the wheel: {value}")
            parts.pop()
            continue
        parts.append(part)
    return PurePosixPath(*parts)


def _compose_build_inputs(document: str) -> tuple[tuple[str, str], ...]:
    """Project context and Dockerfile scalars from Compose build declarations."""

    builds: list[tuple[str, str]] = []
    lines = document.splitlines()
    for index, line in enumerate(lines):
        content = line.split("#", 1)[0].rstrip()
        if not content or content.lstrip() != "build:":
            continue
        build_indent = len(content) - len(content.lstrip())
        context = "."
        dockerfile = "Dockerfile"
        for nested in lines[index + 1 :]:
            nested_content = nested.split("#", 1)[0].rstrip()
            if not nested_content:
                continue
            nested_indent = len(nested_content) - len(nested_content.lstrip())
            if nested_indent <= build_indent:
                break
            key, separator, value = nested_content.strip().partition(":")
            if not separator:
                continue
            scalar = value.strip().strip("'\"")
            if key == "context" and scalar:
                context = scalar
            elif key == "dockerfile" and scalar:
                dockerfile = scalar
        builds.append((context, dockerfile))
    return tuple(builds)


def _logical_dockerfile_lines(document: str) -> tuple[str, ...]:
    logical_lines: list[str] = []
    current = ""
    for raw_line in document.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        current = f"{current} {stripped}".strip()
        if current.endswith("\\"):
            current = current[:-1].rstrip()
            continue
        logical_lines.append(current)
        current = ""
    if current:
        logical_lines.append(current)
    return tuple(logical_lines)


def _dockerfile_copy_sources(document: str) -> tuple[str, ...]:
    """Return local COPY/ADD sources declared by a Dockerfile."""

    sources: list[str] = []
    for line in _logical_dockerfile_lines(document):
        instruction, separator, arguments = line.partition(" ")
        if not separator or instruction.upper() not in {"ADD", "COPY"}:
            continue
        arguments = arguments.strip()
        if arguments.startswith("["):
            values = json.loads(arguments)
            if not isinstance(values, list) or not all(
                isinstance(value, str) for value in values
            ):
                raise ValueError(f"Invalid {instruction} JSON arguments: {arguments}")
            operands = values
        else:
            operands = shlex.split(arguments)
        while operands and operands[0].startswith("--"):
            flag = operands.pop(0)
            if flag.startswith("--from="):
                operands = []
                break
        if len(operands) < 2:
            continue
        for source in operands[:-1]:
            if "://" not in source:
                sources.append(source)
    return tuple(sources)


def _member_exists(members: set[str], path: PurePosixPath) -> bool:
    candidate = path.as_posix().rstrip("/")
    if any(character in candidate for character in "*?["):
        return any(fnmatch.fnmatch(member, candidate) for member in members)
    return candidate in members or any(
        member.startswith(f"{candidate}/") for member in members
    )


def validate_wheel_deployment(wheel_path: Path) -> tuple[str, ...]:
    """Return missing or unsafe deployment inputs declared inside ``wheel_path``."""

    errors: list[str] = []
    with ZipFile(wheel_path) as wheel:
        members = set(wheel.namelist())
        for member in sorted(members):
            member_path = PurePosixPath(member)
            if member_path.parts[:1] == ("openhcs",) and "build" in member_path.parts:
                errors.append(f"{member}: wheel contains nested build output")
            if member_path.parts[:1] != ("openhcs",) or member_path.suffix != ".py":
                continue
            try:
                source = wheel.read(member).decode("utf-8")
            except UnicodeDecodeError:
                errors.append(f"{member}: Python source is not UTF-8")
                continue
            developer_paths = sorted(set(DEVELOPER_HOME_PATH_PATTERN.findall(source)))
            if developer_paths:
                errors.append(
                    f"{member}: wheel contains developer-home paths: "
                    + ", ".join(developer_paths)
                )
        compose_members = tuple(
            member
            for member in members
            if PurePosixPath(member).name.startswith("docker-compose")
            and PurePosixPath(member).suffix in {".yaml", ".yml"}
        )
        for compose_member in compose_members:
            compose_path = PurePosixPath(compose_member)
            compose = wheel.read(compose_member).decode("utf-8")
            try:
                builds = _compose_build_inputs(compose)
            except ValueError as exc:
                errors.append(f"{compose_member}: {exc}")
                continue
            for context_value, dockerfile_value in builds:
                try:
                    context_path = _member_path(compose_path.parent, context_value)
                    dockerfile_path = _member_path(context_path, dockerfile_value)
                except ValueError as exc:
                    errors.append(f"{compose_member}: {exc}")
                    continue
                dockerfile_member = dockerfile_path.as_posix()
                if dockerfile_member not in members:
                    errors.append(
                        f"{compose_member}: missing declared Dockerfile "
                        f"{dockerfile_member}"
                    )
                    continue
                dockerfile = wheel.read(dockerfile_member).decode("utf-8")
                try:
                    copy_sources = _dockerfile_copy_sources(dockerfile)
                except (ValueError, json.JSONDecodeError) as exc:
                    errors.append(f"{dockerfile_member}: {exc}")
                    continue
                for source in copy_sources:
                    try:
                        source_path = _member_path(context_path, source)
                    except ValueError as exc:
                        errors.append(f"{dockerfile_member}: {exc}")
                        continue
                    if not _member_exists(members, source_path):
                        errors.append(
                            f"{dockerfile_member}: missing declared build source "
                            f"{source_path.as_posix()}"
                        )
    return tuple(errors)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    errors = validate_wheel_deployment(args.wheel)
    if errors:
        parser.error("\n".join(errors))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
