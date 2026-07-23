"""Nominal source-binding context shared by runtime and UI surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from openhcs.core.source_bindings import (
    EMPTY_SOURCE_BINDINGS,
    SourceBindingsConfig,
    StepSourceBindingsConfig,
)
from openhcs.core.source_bindings_view import SourceInventory
from openhcs.core.vfs_protocol import FileManagerLike


@dataclass(frozen=True, slots=True)
class SourceBindingContext:
    """Complete public source-binding context for one logical plate row."""

    logical_plate_id: str
    display_plate_root: Path
    execution_plate_path: Path
    source_bindings: SourceBindingsConfig
    filemanager: FileManagerLike
    source_backend: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "display_plate_root", Path(self.display_plate_root))
        object.__setattr__(self, "execution_plate_path", Path(self.execution_plate_path))
        if not isinstance(self.source_bindings, SourceBindingsConfig):
            raise TypeError(
                "SourceBindingContext.source_bindings must be SourceBindingsConfig."
            )
        source_backend = str(self.source_backend).strip()
        if not source_backend:
            raise ValueError("SourceBindingContext.source_backend cannot be empty.")
        object.__setattr__(self, "source_backend", source_backend)

    def inventory(
        self,
        step_bindings: StepSourceBindingsConfig = EMPTY_SOURCE_BINDINGS,
    ) -> SourceInventory:
        """Return inventory for the declared source root and resolved step override."""
        if not isinstance(step_bindings, StepSourceBindingsConfig):
            raise TypeError(
                "SourceBindingContext.inventory requires StepSourceBindingsConfig."
            )
        return SourceInventory.from_filemanager(
            filemanager=self.filemanager,
            source_root=self.display_plate_root,
            backend=self.source_backend,
            source_bindings=self.source_bindings,
            step_bindings=step_bindings,
        )
