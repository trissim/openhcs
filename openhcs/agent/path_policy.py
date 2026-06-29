"""Filesystem access policy for agent-facing OpenHCS services."""

from __future__ import annotations

from dataclasses import dataclass
from os import environ
from pathlib import Path

from openhcs.agent.exceptions import AgentFacingErrorMixin


DEFAULT_AGENT_WINDOW_SNAPSHOT_DIR = Path("/tmp/openhcs-mcp-window-snapshots")


class AgentPathPolicyError(AgentFacingErrorMixin, ValueError):
    """Raised when a path is outside the agent policy boundary."""

    agent_error_code = "agent_path_policy_rejected"
    agent_error_hint = (
        "Pass a local path under OPENHCS_AGENT_READ_ROOTS or "
        "OPENHCS_AGENT_WRITE_ROOTS as appropriate. Use openhcs_inspect_plate_path "
        "first to validate a plate folder."
    )


@dataclass(frozen=True, slots=True)
class AgentPathRootSet:
    roots: tuple[Path, ...]

    @classmethod
    def from_paths(cls, paths: tuple[Path, ...]) -> "AgentPathRootSet":
        resolved: list[Path] = []
        seen: set[Path] = set()
        for path in paths:
            candidate = Path(path).expanduser().resolve(strict=False)
            if candidate not in seen:
                seen.add(candidate)
                resolved.append(candidate)
        return cls(tuple(resolved))

    @classmethod
    def from_environment(
        cls,
        name: str,
        fallback: "AgentPathRootSet",
    ) -> "AgentPathRootSet":
        raw = environ.get(name)
        if not raw:
            return fallback
        return cls.from_paths(
            tuple(Path(item) for item in raw.split(":") if item.strip())
        )

    def __contains__(self, path: Path) -> bool:
        candidate = path.expanduser().resolve(strict=False)
        return any(
            candidate == root or root in candidate.parents
            for root in self.roots
        )


@dataclass(frozen=True, slots=True)
class AgentPathPolicy:
    readable_roots: AgentPathRootSet
    writable_roots: AgentPathRootSet

    @classmethod
    def with_roots(
        cls,
        *,
        readable_roots: tuple[Path, ...],
        writable_roots: tuple[Path, ...],
    ) -> "AgentPathPolicy":
        return cls(
            AgentPathRootSet.from_paths(readable_roots),
            AgentPathRootSet.from_paths(writable_roots),
        )

    @classmethod
    def default(cls) -> "AgentPathPolicy":
        repo_root = Path(__file__).resolve().parents[2]
        return cls(
            readable_roots=AgentPathRootSet.from_paths(
                (
                    repo_root,
                    Path("/tmp"),
                )
            ),
            writable_roots=AgentPathRootSet.from_paths(
                (
                    Path("/tmp"),
                    repo_root / "mcp_outputs",
                )
            ),
        )

    @classmethod
    def from_environment(cls) -> "AgentPathPolicy":
        default = cls.default()
        readable = AgentPathRootSet.from_environment(
            "OPENHCS_AGENT_READ_ROOTS",
            default.readable_roots,
        )
        writable = AgentPathRootSet.from_environment(
            "OPENHCS_AGENT_WRITE_ROOTS",
            default.writable_roots,
        )
        return cls(
            readable_roots=readable,
            writable_roots=writable,
        )

    def assert_readable(self, path: str | Path) -> Path:
        candidate = Path(path).expanduser().resolve(strict=False)
        if not candidate.exists():
            raise AgentPathPolicyError(f"Readable path does not exist: {candidate}")
        if candidate not in self.readable_roots:
            raise AgentPathPolicyError(
                f"Readable path is outside allowed roots: {candidate}"
            )
        return candidate

    def assert_writable(self, path: str | Path) -> Path:
        candidate = Path(path).expanduser().resolve(strict=False)
        existing_parent = _nearest_existing_parent(candidate)
        if existing_parent not in self.writable_roots:
            raise AgentPathPolicyError(
                f"Writable path is outside allowed roots: {candidate}"
            )
        return candidate


def _nearest_existing_parent(path: Path) -> Path:
    candidate = path if path.exists() else path.parent
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate.resolve(strict=False)
