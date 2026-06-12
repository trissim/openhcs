"""CellProfiler database table/column projection dialect."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

from metaclass_registry import AutoRegisterMeta

from openhcs.core.runtime_table_projection import (
    RuntimeProjectedColumnIdentity,
    RuntimeProjectedColumnRole,
    RuntimeTableProjectionDialect,
)


@dataclass(frozen=True, slots=True)
class CellProfilerDatabaseColumnDialect(RuntimeTableProjectionDialect):
    """Render runtime table semantics into CellProfiler/CPA database names."""

    table_prefix: str = ""

    def image_table_name(self) -> str:
        return f"{self.table_prefix}Per_Image"

    def object_table_name(self, object_name: str) -> str:
        self._require_name(object_name, "object_name")
        return f"{self.table_prefix}Per_{object_name}"

    def relationship_table_name(self, relationship_name: str) -> str:
        self._require_name(relationship_name, "relationship_name")
        return f"{self.table_prefix}{relationship_name}"

    def column_name(self, identity: RuntimeProjectedColumnIdentity) -> str:
        if not isinstance(identity, RuntimeProjectedColumnIdentity):
            raise TypeError(
                "CellProfilerDatabaseColumnDialect.column_name requires "
                "RuntimeProjectedColumnIdentity, got "
                f"{type(identity).__name__}."
            )
        return CellProfilerColumnNameRenderer.for_role(identity.role).column_name(
            self,
            identity,
        )

    def object_id_column(
        self,
        object_name: str | None,
        *,
        qualified: bool,
    ) -> str:
        if not qualified:
            return "ObjectNumber"
        required_object_name = self._required(object_name, "object_name")
        return f"{required_object_name}_Number_Object_Number"

    def object_location_column(self, object_name: str, axis_name: str) -> str:
        self._require_name(object_name, "object_name")
        normalized_axis = axis_name.strip().upper()
        if normalized_axis not in {"X", "Y", "Z"}:
            raise ValueError(
                "CellProfiler object location axis must be X, Y, or Z; "
                f"got {axis_name!r}."
            )
        return f"{object_name}_Location_Center_{normalized_axis}"

    @staticmethod
    def _required(value: str | None, field_name: str) -> str:
        if value is None or not value.strip():
            raise ValueError(f"CellProfiler column identity requires {field_name}.")
        return value.strip()

    @classmethod
    def _require_name(cls, value: str, field_name: str) -> None:
        cls._required(value, field_name)


class CellProfilerColumnNameRenderer(ABC, metaclass=AutoRegisterMeta):
    """Render one projected column role into CellProfiler naming."""

    __registry_key__ = "role"
    __skip_if_no_key__ = True
    role: ClassVar[RuntimeProjectedColumnRole | None] = None

    @classmethod
    def for_role(
        cls,
        role: RuntimeProjectedColumnRole,
    ) -> "CellProfilerColumnNameRenderer":
        renderer_type = cls.__registry__.get(role)
        if renderer_type is None:
            raise ValueError(f"Unsupported CellProfiler column role {role!r}.")
        return renderer_type()

    @abstractmethod
    def column_name(
        self,
        dialect: CellProfilerDatabaseColumnDialect,
        identity: RuntimeProjectedColumnIdentity,
    ) -> str:
        """Return the rendered CellProfiler column name."""


class CellProfilerImageIdColumnRenderer(CellProfilerColumnNameRenderer):
    role = RuntimeProjectedColumnRole.IMAGE_ID

    def column_name(
        self,
        dialect: CellProfilerDatabaseColumnDialect,
        identity: RuntimeProjectedColumnIdentity,
    ) -> str:
        return "ImageNumber"


class CellProfilerObjectIdColumnRenderer(CellProfilerColumnNameRenderer):
    role = RuntimeProjectedColumnRole.OBJECT_ID

    def column_name(
        self,
        dialect: CellProfilerDatabaseColumnDialect,
        identity: RuntimeProjectedColumnIdentity,
    ) -> str:
        return dialect.object_id_column(identity.object_name, qualified=False)


class CellProfilerMetadataColumnRenderer(CellProfilerColumnNameRenderer):
    role = RuntimeProjectedColumnRole.METADATA

    def column_name(
        self,
        dialect: CellProfilerDatabaseColumnDialect,
        identity: RuntimeProjectedColumnIdentity,
    ) -> str:
        return f"Image_Metadata_{dialect._required(identity.metadata_key, 'metadata_key')}"


class CellProfilerGroupColumnRenderer(CellProfilerColumnNameRenderer):
    role = RuntimeProjectedColumnRole.GROUP

    def column_name(
        self,
        dialect: CellProfilerDatabaseColumnDialect,
        identity: RuntimeProjectedColumnIdentity,
    ) -> str:
        return f"Image_Group_{dialect._required(identity.metadata_key, 'metadata_key')}"


class CellProfilerSourceImagePathColumnRenderer(CellProfilerColumnNameRenderer):
    role = RuntimeProjectedColumnRole.SOURCE_IMAGE_PATH

    def column_name(
        self,
        dialect: CellProfilerDatabaseColumnDialect,
        identity: RuntimeProjectedColumnIdentity,
    ) -> str:
        return f"Image_PathName_{dialect._required(identity.source_image_name, 'source_image_name')}"


class CellProfilerSourceImageFileColumnRenderer(CellProfilerColumnNameRenderer):
    role = RuntimeProjectedColumnRole.SOURCE_IMAGE_FILE

    def column_name(
        self,
        dialect: CellProfilerDatabaseColumnDialect,
        identity: RuntimeProjectedColumnIdentity,
    ) -> str:
        return f"Image_FileName_{dialect._required(identity.source_image_name, 'source_image_name')}"


class CellProfilerObjectLocationColumnRenderer(CellProfilerColumnNameRenderer):
    role = RuntimeProjectedColumnRole.OBJECT_LOCATION

    def column_name(
        self,
        dialect: CellProfilerDatabaseColumnDialect,
        identity: RuntimeProjectedColumnIdentity,
    ) -> str:
        return dialect.object_location_column(
            dialect._required(identity.object_name, "object_name"),
            dialect._required(identity.axis_name, "axis_name"),
        )


class CellProfilerMeasurementFeatureColumnRenderer(CellProfilerColumnNameRenderer):
    role = RuntimeProjectedColumnRole.MEASUREMENT_FEATURE

    def column_name(
        self,
        dialect: CellProfilerDatabaseColumnDialect,
        identity: RuntimeProjectedColumnIdentity,
    ) -> str:
        return dialect._required(identity.feature_name, "feature_name")
