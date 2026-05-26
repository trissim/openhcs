"""Nominal source-binding context shared by runtime and UI surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openhcs.core.pipeline_image_schema import PipelineImageSchema
from openhcs.core.source_bindings import StepSourceBindingsConfig
from openhcs.core.source_bindings_view import SourceInventory, SourceInventoryProvider


@dataclass(frozen=True, slots=True)
class SourceBindingContext:
    """Complete source-binding context for one logical plate row."""

    logical_plate_id: str
    display_plate_root: Path
    execution_plate_path: Path
    source_schema: PipelineImageSchema
    inventory_provider: SourceInventoryProvider
    cppipe_path: Path | None = None
    import_result: Any | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "display_plate_root", Path(self.display_plate_root))
        object.__setattr__(self, "execution_plate_path", Path(self.execution_plate_path))
        if self.cppipe_path is not None:
            object.__setattr__(self, "cppipe_path", Path(self.cppipe_path))

    def inventory(
        self,
        bindings: StepSourceBindingsConfig = StepSourceBindingsConfig(),
    ) -> SourceInventory:
        """Return source inventory through the context's provider authority."""

        return self.inventory_provider.inventory(
            schema=self.source_schema,
            bindings=bindings,
        )
