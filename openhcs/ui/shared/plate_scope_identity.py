"""Framework-neutral plate and pipeline scope identities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, unquote

CELLPROFILER_PIPELINE_SCOPE_MARKER = "#openhcs-cppipe="
SCOPE_SEGMENT_SEPARATOR = "::"
PIPELINE_SCOPE_SEGMENT = "pipeline"
PIPELINE_SCOPE_PATTERN = re.compile(
    rf"^.+{re.escape(SCOPE_SEGMENT_SEPARATOR)}{re.escape(PIPELINE_SCOPE_SEGMENT)}$"
)


@dataclass(frozen=True, slots=True)
class PlateScopeIdentity:
    """Parsed plate/orchestrator scope that keeps pipeline identity opaque."""

    scope_id: str
    plate_root: Path
    cppipe_path: Path | None = None

    @property
    def display_name(self) -> str:
        if self.cppipe_path is None:
            return self.plate_root.name
        return f"{self.plate_root.name} / {self.cppipe_path.stem}"

    def code_value(self) -> Path | str:
        """Return the public code value without parsing opaque pipeline scopes."""

        return self.plate_root if self.cppipe_path is None else self.scope_id

    @classmethod
    def from_plate_root(cls, plate_root: Path | str) -> "PlateScopeIdentity":
        root = Path(plate_root)
        return cls(scope_id=str(root), plate_root=root)

    @classmethod
    def from_cellprofiler_pipeline(
        cls,
        plate_root: Path | str,
        cppipe_path: Path | str,
    ) -> "PlateScopeIdentity":
        root = Path(plate_root)
        pipeline = Path(cppipe_path)
        encoded_pipeline_name = quote(pipeline.name, safe="")
        return cls(
            scope_id=f"{root}{CELLPROFILER_PIPELINE_SCOPE_MARKER}{encoded_pipeline_name}",
            plate_root=root,
            cppipe_path=pipeline,
        )

    @classmethod
    def from_scope_id(cls, scope_id: str) -> "PlateScopeIdentity":
        if CELLPROFILER_PIPELINE_SCOPE_MARKER not in scope_id:
            return cls.from_plate_root(scope_id)

        plate_root_text, encoded_pipeline_name = scope_id.rsplit(
            CELLPROFILER_PIPELINE_SCOPE_MARKER,
            maxsplit=1,
        )
        pipeline_name = unquote(encoded_pipeline_name)
        plate_root = Path(plate_root_text)
        return cls(
            scope_id=scope_id,
            plate_root=plate_root,
            cppipe_path=plate_root / pipeline_name,
        )


@dataclass(frozen=True, slots=True)
class PipelineScopeIdentity:
    """ObjectState scope for the pipeline under one plate scope."""

    scope_id: str
    plate_scope: str

    @classmethod
    def from_plate_scope(cls, plate_scope: str) -> "PipelineScopeIdentity":
        if not plate_scope:
            raise ValueError("Pipeline scope requires a non-empty plate scope.")
        return cls(
            scope_id=(
                f"{plate_scope}{SCOPE_SEGMENT_SEPARATOR}{PIPELINE_SCOPE_SEGMENT}"
            ),
            plate_scope=plate_scope,
        )

    @classmethod
    def from_scope_id(cls, scope_id: str) -> "PipelineScopeIdentity":
        if not cls.matches(scope_id):
            raise ValueError(f"Invalid pipeline scope id: {scope_id!r}")
        plate_scope, _pipeline_segment = scope_id.rsplit(
            SCOPE_SEGMENT_SEPARATOR,
            1,
        )
        return cls(scope_id=scope_id, plate_scope=plate_scope)

    @classmethod
    def matches(cls, scope_id: str) -> bool:
        return PIPELINE_SCOPE_PATTERN.fullmatch(scope_id) is not None

    @classmethod
    def handler_pattern(cls) -> str:
        return PIPELINE_SCOPE_PATTERN.pattern
