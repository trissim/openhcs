"""CellProfiler source-metadata declarations shared by setup and export."""

from __future__ import annotations

from enum import Enum
from typing import cast
from openhcs.core.runtime_tabular_values import FieldSpec
from openhcs.core.source_metadata import SourceMetadataScalar


_NO_STATIC_DEFAULT = object()


class CellProfilerSourceMetadataField(Enum):
    """Source fields CellProfiler declares independently of extraction rules."""

    FILE_LOCATION = ("FileLocation", str)
    FRAME = ("Frame", int, 0)
    SERIES = ("Series", int, 0)

    def __init__(
        self,
        field_name: str,
        dtype: type[object],
        default: SourceMetadataScalar | object = _NO_STATIC_DEFAULT,
    ) -> None:
        self.field_name = field_name
        self.dtype = dtype
        self._static_default = default

    def field_spec(self) -> FieldSpec:
        """Return this field's exact source schema declaration."""

        return FieldSpec(
            self.field_name,
            self.dtype,
            required=False,
        )

    @classmethod
    def static_defaults(cls) -> dict[str, SourceMetadataScalar]:
        """Return defaults independent of an individual source image."""

        return {
            field.field_name: cast(SourceMetadataScalar, field._static_default)
            for field in cls
            if field._static_default is not _NO_STATIC_DEFAULT
        }
