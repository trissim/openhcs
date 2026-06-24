"""CellProfiler executable discovery for benchmark adapters."""

from __future__ import annotations

import os
import shutil
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from benchmark.contracts.tool_adapter import ToolNotInstalledError


CELLPROFILER_EXECUTABLE_ENV = "CELLPROFILER_EXECUTABLE"
OPENHCS_BENCHMARK_TOOL_ROOTS_ENV = "OPENHCS_BENCHMARK_TOOL_ROOTS"


class CellProfilerExecutableSource(StrEnum):
    """Ordered sources for resolving the native CellProfiler executable."""

    def __new__(
        cls,
        value: str,
        trusts_declared_path: bool,
    ) -> "CellProfilerExecutableSource":
        member = str.__new__(cls, value)
        member._value_ = value
        member.trusts_declared_path = trusts_declared_path
        return member

    CONFIGURED = ("configured", True)
    ENVIRONMENT = ("environment", True)
    PATH = ("path", True)
    CURRENT_PYTHON_ENVIRONMENT = ("current_python_environment", False)
    DECLARED_TOOL_ROOT = ("declared_tool_root", False)
    LOCAL_WORKSPACE_TOOL_ROOT = ("local_workspace_tool_root", False)


@dataclass(frozen=True, slots=True)
class CellProfilerExecutableCandidate:
    """One candidate executable path from a named discovery source."""

    source: CellProfilerExecutableSource
    path: Path

    def discovered_path(self) -> Path | None:
        """Return the path when this candidate is usable without subprocess probing."""
        if self.source.trusts_declared_path:
            return self.path
        if self.path.is_file():
            return self.path
        return None


@dataclass(frozen=True, slots=True)
class CellProfilerExecutableResolver:
    """Resolve CellProfiler through explicit config, environment, PATH, and dev roots."""

    configured_executable: Path | None = None
    environment: Mapping[str, str] | None = None
    repo_root: Path | None = None
    python_executable: Path = Path(sys.executable)

    def resolve(self) -> Path:
        """Return the first executable path accepted by the discovery contract."""
        for candidate in self.candidates():
            discovered_path = candidate.discovered_path()
            if discovered_path is not None:
                return discovered_path
        raise ToolNotInstalledError(self._not_found_message())

    def candidates(self) -> tuple[CellProfilerExecutableCandidate, ...]:
        """Return executable candidates in precedence order."""
        candidates: list[CellProfilerExecutableCandidate] = []
        if self.configured_executable is not None:
            candidates.append(
                CellProfilerExecutableCandidate(
                    CellProfilerExecutableSource.CONFIGURED,
                    self.configured_executable,
                )
            )
        env_value = self._environment.get(CELLPROFILER_EXECUTABLE_ENV)
        if env_value:
            candidates.append(
                CellProfilerExecutableCandidate(
                    CellProfilerExecutableSource.ENVIRONMENT,
                    Path(os.path.expandvars(env_value)).expanduser(),
                )
            )
        path_executable = shutil.which("cellprofiler")
        if path_executable is not None:
            candidates.append(
                CellProfilerExecutableCandidate(
                    CellProfilerExecutableSource.PATH,
                    Path(path_executable),
                )
            )
        candidates.extend(
            CellProfilerExecutableCandidate(
                CellProfilerExecutableSource.CURRENT_PYTHON_ENVIRONMENT,
                path,
            )
            for path in self._current_environment_candidates()
        )
        candidates.extend(
            CellProfilerExecutableCandidate(
                CellProfilerExecutableSource.DECLARED_TOOL_ROOT,
                path,
            )
            for path in self._declared_tool_root_candidates()
        )
        candidates.extend(
            CellProfilerExecutableCandidate(
                CellProfilerExecutableSource.LOCAL_WORKSPACE_TOOL_ROOT,
                path,
            )
            for path in self._local_workspace_tool_root_candidates()
        )
        return tuple(candidates)

    @property
    def _environment(self) -> Mapping[str, str]:
        return os.environ if self.environment is None else self.environment

    @property
    def _repo_root(self) -> Path:
        return (
            Path(__file__).resolve().parents[2]
            if self.repo_root is None
            else self.repo_root
        )

    def _current_environment_candidates(self) -> tuple[Path, ...]:
        scripts_dir = self.python_executable.parent
        return (
            scripts_dir / "cellprofiler",
            scripts_dir / "cellprofiler.exe",
        )

    def _declared_tool_root_candidates(self) -> tuple[Path, ...]:
        raw_roots = self._environment.get(OPENHCS_BENCHMARK_TOOL_ROOTS_ENV)
        if not raw_roots:
            return ()
        return tuple(
            candidate
            for root in raw_roots.split(os.pathsep)
            if root
            for candidate in self._tool_root_executable_candidates(
                Path(os.path.expandvars(root)).expanduser()
            )
        )

    def _local_workspace_tool_root_candidates(self) -> tuple[Path, ...]:
        roots = (self._repo_root, *self._workspace_sibling_roots())
        return tuple(
            candidate
            for root in roots
            for candidate in self._tool_root_executable_candidates(root)
        )

    def _workspace_sibling_roots(self) -> tuple[Path, ...]:
        workspace_root = self._repo_root.parent
        if not workspace_root.is_dir():
            return ()
        return tuple(
            candidate
            for candidate in sorted(workspace_root.iterdir())
            if candidate.is_dir() and candidate != self._repo_root
        )

    @staticmethod
    def _tool_root_executable_candidates(root: Path) -> tuple[Path, ...]:
        environment_roots = (
            root,
            root / ".venv-cellprofiler39",
            root / ".venv-cellprofiler",
            root / ".venv",
        )
        return tuple(
            executable_path
            for environment_root in environment_roots
            for executable_path in (
                environment_root / "bin" / "cellprofiler",
                environment_root / "Scripts" / "cellprofiler.exe",
            )
        )

    def _not_found_message(self) -> str:
        searched = "\n".join(f"- {candidate.path}" for candidate in self.candidates())
        return (
            "CellProfiler executable not configured and no local installation was "
            "discovered. Install CellProfiler in the current environment, expose it "
            "on PATH, pass an executable path, or set "
            f"{CELLPROFILER_EXECUTABLE_ENV} or {OPENHCS_BENCHMARK_TOOL_ROOTS_ENV}."
            f"\nSearched:\n{searched}"
        )
