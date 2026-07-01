"""Self-materializing benchmark manifest root contracts."""

from __future__ import annotations

import os
import shutil
import subprocess
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import ClassVar

from metaclass_registry import AutoRegisterMeta

ManifestRootRequirementMap = dict[str, list["ManifestRootRequirement"]]


class ManifestAcquisitionError(RuntimeError):
    """Raised when a benchmark manifest root cannot be materialized."""


class ManifestRootAcquisitionKind(Enum):
    """Supported benchmark-manifest root acquisition families."""

    DATASET_REGISTRY = "dataset_registry"
    GIT_SPARSE = "git_sparse"


@dataclass(frozen=True, slots=True)
class ManifestRootRequirement:
    """One path that a manifest case expects under a root."""

    case_name: str
    path_key: str
    relative_path: Path
    dataset_id: str | None = None


@dataclass(frozen=True, slots=True)
class ManifestRootAcquisitionSpec:
    """Declarative acquisition policy for one manifest path root."""

    kind: ManifestRootAcquisitionKind
    git_url: str | None = None
    git_ref: str = "HEAD"
    sparse_paths: tuple[str, ...] = ()
    dataset_ids: tuple[str, ...] = ()

    @classmethod
    def from_manifest(cls, raw_value: object) -> "ManifestRootAcquisitionSpec":
        """Parse an acquisition block from a benchmark manifest."""
        if not isinstance(raw_value, Mapping):
            raise ValueError("Manifest root acquisition must be an object.")
        raw_kind = raw_value.get("kind")
        if raw_kind is None:
            raise ValueError("Manifest root acquisition must declare kind.")
        try:
            kind = ManifestRootAcquisitionKind(str(raw_kind))
        except ValueError as exc:
            raise ValueError(
                f"Unsupported manifest root acquisition kind {raw_kind!r}."
            ) from exc
        raw_sparse_paths = raw_value.get("sparse_paths", ())
        raw_dataset_ids = raw_value.get("dataset_ids", ())
        git_url = raw_value.get("git_url")
        return cls(
            kind=kind,
            git_url=str(git_url) if git_url is not None else None,
            git_ref=str(raw_value.get("git_ref", "HEAD")),
            sparse_paths=_string_tuple(raw_sparse_paths, "sparse_paths"),
            dataset_ids=_string_tuple(raw_dataset_ids, "dataset_ids"),
        )


@dataclass(frozen=True, slots=True)
class ManifestRootAcquisitionRequest:
    """Resolved work request for one materializable manifest root."""

    root_name: str
    root_path: Path
    spec: ManifestRootAcquisitionSpec
    requirements: tuple[ManifestRootRequirement, ...]

    def missing_requirements(self) -> tuple[ManifestRootRequirement, ...]:
        """Return required root-relative paths not currently present on disk."""
        return tuple(
            requirement
            for requirement in self.requirements
            if not (self.root_path / requirement.relative_path).exists()
        )


class ManifestRootAcquisitionStrategy(ABC, metaclass=AutoRegisterMeta):
    """Registered materializer for one benchmark-manifest root source family."""

    __registry_key__ = "kind"
    __skip_if_no_key__ = True

    kind: ClassVar[ManifestRootAcquisitionKind | None] = None

    @classmethod
    def for_spec(
        cls,
        spec: ManifestRootAcquisitionSpec,
    ) -> "ManifestRootAcquisitionStrategy":
        try:
            strategy_type = cls.__registry__[spec.kind]
        except KeyError as exc:
            raise ManifestAcquisitionError(
                f"Unsupported manifest root acquisition kind {spec.kind.name}."
            ) from exc
        return strategy_type()

    @abstractmethod
    def materialize(self, request: ManifestRootAcquisitionRequest) -> None:
        """Materialize one manifest root."""


class DatasetRegistryRootAcquisitionStrategy(ManifestRootAcquisitionStrategy):
    """Materialize DatasetSpec-backed cases into a benchmark dataset cache root."""

    kind = ManifestRootAcquisitionKind.DATASET_REGISTRY

    def materialize(self, request: ManifestRootAcquisitionRequest) -> None:
        from benchmark.datasets.acquire import acquire_dataset
        from benchmark.datasets.registry import get_dataset_spec

        dataset_ids = request.spec.dataset_ids or tuple(
            dict.fromkeys(
                requirement.dataset_id
                for requirement in request.requirements
                if requirement.dataset_id is not None
            )
        )
        if not dataset_ids:
            raise ManifestAcquisitionError(
                f"Manifest root {request.root_name!r} has no dataset ids to acquire."
            )
        for dataset_id in dataset_ids:
            acquire_dataset(get_dataset_spec(dataset_id), cache_base=request.root_path)


class GitSparseRootAcquisitionStrategy(ManifestRootAcquisitionStrategy):
    """Materialize a manifest root from a sparse git checkout."""

    kind = ManifestRootAcquisitionKind.GIT_SPARSE

    def materialize(self, request: ManifestRootAcquisitionRequest) -> None:
        if not request.spec.git_url:
            raise ManifestAcquisitionError(
                f"Manifest root {request.root_name!r} git_sparse acquisition "
                "requires git_url."
            )
        needs_ref_fetch = request.spec.git_ref != "HEAD"
        if (request.root_path / ".git").exists():
            self._run_git(
                ["fetch", "--depth", "1", "origin", request.spec.git_ref],
                request.root_path,
            )
            needs_ref_fetch = False
        else:
            if request.root_path.exists() and any(request.root_path.iterdir()):
                missing = request.missing_requirements()
                raise ManifestAcquisitionError(
                    f"Manifest root {request.root_name!r} at {request.root_path} "
                    "exists but is not a git checkout and is missing "
                    f"{len(missing)} required paths. Move it or set a different root."
                )
            shutil.rmtree(request.root_path, ignore_errors=True)
            request.root_path.parent.mkdir(parents=True, exist_ok=True)
            self._run_git(
                [
                    "clone",
                    "--depth",
                    "1",
                    "--filter=blob:none",
                    "--sparse",
                    request.spec.git_url,
                    str(request.root_path),
                ],
                None,
            )
        if needs_ref_fetch:
            self._run_git(
                ["fetch", "--depth", "1", "origin", request.spec.git_ref],
                request.root_path,
            )

        sparse_paths = _git_sparse_paths(request)
        if sparse_paths:
            self._run_git(["sparse-checkout", "set", *sparse_paths], request.root_path)
        if request.spec.git_ref == "HEAD":
            self._run_git(["pull", "--ff-only"], request.root_path)
        else:
            self._run_git(["checkout", "FETCH_HEAD"], request.root_path)

    @staticmethod
    def _run_git(args: Sequence[str], cwd: Path | None) -> None:
        try:
            subprocess.run(
                ["git", *args],
                cwd=cwd,
                check=True,
                text=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr.strip() or exc.stdout.strip() or str(exc)
            raise ManifestAcquisitionError(
                f"git {' '.join(args)} failed: {detail}"
            ) from exc


def manifest_auto_acquire_enabled() -> bool:
    """Return whether manifest roots should materialize missing files."""
    raw_value = os.environ.get("OPENHCS_BENCHMARK_AUTO_ACQUIRE", "1").strip().lower()
    return raw_value not in {"0", "false", "no", "off"}


def materialize_manifest_root(
    *,
    root_name: str,
    root_path: Path,
    acquisition_spec: ManifestRootAcquisitionSpec,
    requirements: tuple[ManifestRootRequirement, ...],
) -> None:
    """Materialize one acquisition-enabled manifest root when required paths miss."""
    request = ManifestRootAcquisitionRequest(
        root_name=root_name,
        root_path=root_path,
        spec=acquisition_spec,
        requirements=requirements,
    )
    if not request.missing_requirements():
        return
    ManifestRootAcquisitionStrategy.for_spec(acquisition_spec).materialize(request)
    missing_after = request.missing_requirements()
    if missing_after:
        missing_lines = ", ".join(
            f"{item.case_name}:{item.path_key}={item.relative_path}"
            for item in missing_after[:5]
        )
        raise ManifestAcquisitionError(
            f"Manifest root {root_name!r} materialized but still misses "
            f"{len(missing_after)} required paths: {missing_lines}"
        )


def manifest_root_requirements_by_root(
    raw_cases: Sequence[object],
) -> ManifestRootRequirementMap:
    """Collect root-relative path requirements from manifest cases."""
    requirements: ManifestRootRequirementMap = {}
    for raw_case in raw_cases:
        if not isinstance(raw_case, Mapping):
            continue
        case_name = str(raw_case.get("name", "<unnamed>"))
        dataset_id = (
            str(raw_case["dataset_id"])
            if raw_case.get("dataset_id") is not None
            else None
        )
        for path_key in ("dataset_path", "cppipe_path"):
            root_name = raw_case.get(f"{path_key}_root")
            raw_path = raw_case.get(path_key)
            if root_name is None or raw_path is None:
                continue
            relative_path = Path(os.path.expandvars(str(raw_path))).expanduser()
            if relative_path.is_absolute():
                continue
            requirements.setdefault(str(root_name), []).append(
                ManifestRootRequirement(
                    case_name=case_name,
                    path_key=path_key,
                    relative_path=relative_path,
                    dataset_id=dataset_id,
                )
            )
    return requirements


def _git_sparse_paths(request: ManifestRootAcquisitionRequest) -> tuple[str, ...]:
    """Return declared and requirement-derived sparse checkout paths."""
    paths = list(request.spec.sparse_paths)
    paths.extend(
        requirement.relative_path.parts[0]
        for requirement in request.requirements
        if requirement.relative_path.parts
    )
    return tuple(dict.fromkeys(paths))


def _string_tuple(raw_value: object, field_name: str) -> tuple[str, ...]:
    """Parse a manifest string sequence."""
    if raw_value is None:
        return ()
    if isinstance(raw_value, str):
        return (raw_value,)
    if not isinstance(raw_value, Sequence):
        raise ValueError(f"Manifest acquisition {field_name} must be a sequence.")
    return tuple(str(item) for item in raw_value)
