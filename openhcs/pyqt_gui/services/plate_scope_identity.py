"""Plate-level ObjectState scope identity helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, unquote


CELLPROFILER_PIPELINE_SCOPE_MARKER = "#openhcs-cppipe="


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
