"""Shared comparison-manifest path contracts."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ManifestPathRoot:
    """Named filesystem root declared by a benchmark comparison manifest."""

    name: str
    path: Path

    @classmethod
    def from_manifest(cls, name: str, raw_value: object) -> "ManifestPathRoot":
        if isinstance(raw_value, str):
            return cls(name=name, path=Path(os.path.expandvars(raw_value)).expanduser())
        if not isinstance(raw_value, Mapping):
            raise ValueError(f"Manifest path root {name!r} must be a string or object.")
        env_name = raw_value.get("env")
        default_value = raw_value.get("default")
        path_value = raw_value.get("path")
        if env_name is not None:
            resolved = os.environ.get(str(env_name))
            if resolved is not None:
                return cls(name=name, path=Path(resolved).expanduser())
        if path_value is not None:
            return cls(
                name=name,
                path=Path(os.path.expandvars(str(path_value))).expanduser(),
            )
        if default_value is not None:
            return cls(
                name=name,
                path=Path(os.path.expandvars(str(default_value))).expanduser(),
            )
        raise ValueError(
            f"Manifest path root {name!r} must declare path, default, or env."
        )


@dataclass(frozen=True, slots=True)
class ComparisonManifestPathResolver:
    """Resolve manifest paths against declared roots."""

    roots: Mapping[str, ManifestPathRoot]

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, object],
    ) -> "ComparisonManifestPathResolver":
        raw_roots = payload.get("path_roots", {})
        if not isinstance(raw_roots, Mapping):
            raise ValueError("Benchmark manifest path_roots must be an object.")
        return cls(
            roots={
                str(name): ManifestPathRoot.from_manifest(str(name), raw_value)
                for name, raw_value in raw_roots.items()
            },
        )

    def resolve(self, raw_case: Mapping[str, object], path_key: str) -> Path:
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
    payload: Mapping[str, Any]
    path_resolver: ComparisonManifestPathResolver

    @classmethod
    def load(cls, path: Path) -> "ComparisonManifest":
        manifest_path = Path(path)
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("Benchmark manifest must be a JSON object.")
        return cls(
            path=manifest_path,
            payload=payload,
            path_resolver=ComparisonManifestPathResolver.from_payload(payload),
        )
