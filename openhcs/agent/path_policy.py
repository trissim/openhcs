"""Filesystem access policy for agent-facing OpenHCS services."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from openhcs.agent.exceptions import AgentFacingErrorMixin
from openhcs.agent.runtime_platform import AgentRuntimePlatformAuthority


class AgentPathLocationAuthority:
    """Own cross-platform default roots for agent-visible filesystem access."""

    @staticmethod
    def package_root() -> Path:
        return Path(__file__).resolve().parents[1]

    @classmethod
    def source_checkout_root(cls) -> Path | None:
        candidate = cls.package_root().parent
        if (candidate / "pyproject.toml").is_file():
            return candidate
        return None

    @staticmethod
    def temporary_root() -> Path:
        return AgentRuntimePlatformAuthority.current().temporary_root()

    @staticmethod
    def user_data_root() -> Path:
        return AgentRuntimePlatformAuthority.current().application_data_root("OpenHCS")

    @classmethod
    def default_output_root(cls) -> Path:
        return cls.user_data_root() / "mcp_outputs"

    @classmethod
    def default_window_snapshot_dir(cls) -> Path:
        return cls.temporary_root() / "openhcs-mcp-window-snapshots"


DEFAULT_AGENT_WINDOW_SNAPSHOT_DIR = (
    AgentPathLocationAuthority.default_window_snapshot_dir()
)


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
        raw = os.environ.get(name)
        if not raw:
            return fallback
        return cls.from_paths(
            tuple(Path(item) for item in raw.split(os.pathsep) if item.strip())
        )

    def __contains__(self, path: Path) -> bool:
        candidate = path.expanduser().resolve(strict=False)
        return any(
            candidate == root or root in candidate.parents for root in self.roots
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
        package_root = AgentPathLocationAuthority.package_root()
        source_checkout_root = AgentPathLocationAuthority.source_checkout_root()
        temporary_root = AgentPathLocationAuthority.temporary_root()
        user_data_root = AgentPathLocationAuthority.user_data_root()
        readable_roots = [package_root, temporary_root, user_data_root]
        writable_roots = [
            temporary_root,
            AgentPathLocationAuthority.default_output_root(),
        ]
        if source_checkout_root is not None:
            readable_roots.insert(0, source_checkout_root)
            writable_roots.append(source_checkout_root / "mcp_outputs")
        return cls(
            readable_roots=AgentPathRootSet.from_paths(tuple(readable_roots)),
            writable_roots=AgentPathRootSet.from_paths(tuple(writable_roots)),
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
        if candidate not in self.writable_roots:
            raise AgentPathPolicyError(
                f"Writable path is outside allowed roots: {candidate}"
            )
        return candidate
