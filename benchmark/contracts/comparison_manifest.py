"""Shared comparison-manifest path contracts."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

from benchmark.contracts.manifest_acquisition import (
    ManifestRootAcquisitionSpec,
    manifest_auto_acquire_enabled,
    materialize_manifest_roots,
)
from benchmark.datasets.cache import BenchmarkPathRootKind, resolve_benchmark_path_root


JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | Mapping[str, "JSONValue"] | Sequence["JSONValue"]
ManifestPayload: TypeAlias = Mapping[str, JSONValue]
ManifestCasePayload: TypeAlias = Mapping[str, JSONValue]


@dataclass(frozen=True, slots=True)
class ManifestPathRootDeclaration:
    """Serialized path-root declaration from a benchmark manifest."""

    env: str | None = None
    path: str | None = None
    default: str | None = None
    default_kind: BenchmarkPathRootKind | None = None
    acquisition: ManifestRootAcquisitionSpec | None = None

    @classmethod
    def from_raw(
        cls,
        name: str,
        raw_value: JSONValue,
    ) -> "ManifestPathRootDeclaration":
        """Parse one root declaration from manifest JSON."""
        if isinstance(raw_value, str):
            return cls(path=raw_value)
        if not isinstance(raw_value, Mapping):
            raise ValueError(f"Manifest path root {name!r} must be a string or object.")
        raw_acquisition = raw_value.get("acquisition")
        raw_default_kind = raw_value.get("default_kind")
        return cls(
            env=_optional_string(raw_value.get("env"), "env"),
            path=_optional_string(raw_value.get("path"), "path"),
            default=_optional_string(raw_value.get("default"), "default"),
            default_kind=(
                BenchmarkPathRootKind(str(raw_default_kind))
                if raw_default_kind is not None
                else None
            ),
            acquisition=(
                ManifestRootAcquisitionSpec.from_manifest(raw_acquisition)
                if raw_acquisition is not None
                else None
            ),
        )

    def resolve_path(self, root_name: str) -> Path:
        """Resolve this declaration to a filesystem path."""
        resolved_path = os.environ.get(self.env) if self.env is not None else None
        if resolved_path is not None:
            return Path(os.path.expandvars(resolved_path)).expanduser()
        if self.path is not None:
            return Path(os.path.expandvars(self.path)).expanduser()
        if self.default_kind is not None:
            return resolve_benchmark_path_root(self.default_kind)
        if self.default is not None:
            return Path(os.path.expandvars(self.default)).expanduser()
        raise ValueError(
            f"Manifest path root {root_name!r} must declare path, default, "
            "default_kind, or env."
        )


@dataclass(frozen=True, slots=True)
class ManifestPathRoot:
    """Named filesystem root declared by a benchmark comparison manifest."""

    name: str
    path: Path
    acquisition: ManifestRootAcquisitionSpec | None = None

    @classmethod
    def from_manifest(cls, name: str, raw_value: JSONValue) -> "ManifestPathRoot":
        declaration = ManifestPathRootDeclaration.from_raw(name, raw_value)
        return cls(
            name=name,
            path=declaration.resolve_path(name),
            acquisition=declaration.acquisition,
        )


@dataclass(frozen=True, slots=True)
class ComparisonManifestPathResolver:
    """Resolve manifest paths against declared roots."""

    roots: Mapping[str, ManifestPathRoot]

    @classmethod
    def from_payload(
        cls,
        payload: ManifestPayload,
    ) -> "ComparisonManifestPathResolver":
        raw_roots = payload.get("path_roots")
        if raw_roots is None:
            raw_roots = {}
        if not isinstance(raw_roots, Mapping):
            raise ValueError("Benchmark manifest path_roots must be an object.")
        return cls(
            roots={
                str(name): ManifestPathRoot.from_manifest(str(name), raw_value)
                for name, raw_value in raw_roots.items()
            },
        )

    def resolve(self, raw_case: ManifestCasePayload, path_key: str) -> Path:
        root_key = raw_case.get(f"{path_key}_root")
        raw_path = raw_case.get(path_key)
        if raw_path is None:
            raise ValueError(f"Benchmark case missing required path {path_key!r}.")
        path = Path(os.path.expandvars(str(raw_path))).expanduser()
        if root_key is not None:
            root = self.roots.get(str(root_key))
            if root is None:
                raise ValueError(
                    f"Benchmark case references unknown {path_key}_root "
                    f"{root_key!r}."
                )
            return root.path / path
        return path


@dataclass(frozen=True, slots=True)
class ComparisonManifest:
    """Parsed benchmark comparison manifest with shared path semantics."""

    path: Path
    payload: ManifestPayload
    path_resolver: ComparisonManifestPathResolver

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        materialize_roots: bool | None = None,
    ) -> "ComparisonManifest":
        manifest_path = Path(path)
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("Benchmark manifest must be a JSON object.")
        path_resolver = ComparisonManifestPathResolver.from_payload(payload)
        should_materialize = (
            manifest_auto_acquire_enabled()
            if materialize_roots is None
            else materialize_roots
        )
        if should_materialize:
            materialize_manifest_roots(path_resolver.roots, payload.get("cases"))
        return cls(
            path=manifest_path,
            payload=payload,
            path_resolver=path_resolver,
        )


def _optional_string(value: JSONValue | None, field_name: str) -> str | None:
    """Parse an optional manifest string field."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    raise ValueError(f"Manifest path root field {field_name!r} must be a string.")
