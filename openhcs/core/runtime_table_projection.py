"""Generic runtime table projection contracts.

These contracts are the write-side counterpart to measurement lookup dialects:
runtime values keep semantic ownership, while projection dialects render those
semantics into an external tabular schema.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from openhcs.core.runtime_semantics import MeasurementSubject


class RuntimeProjectedColumnRole(str, Enum):
    """Semantic roles that can be rendered into external table columns."""

    IMAGE_ID = "image_id"
    OBJECT_ID = "object_id"
    METADATA = "metadata"
    GROUP = "group"
    SOURCE_IMAGE_PATH = "source_image_path"
    SOURCE_IMAGE_FILE = "source_image_file"
    OBJECT_LOCATION = "object_location"
    MEASUREMENT_FEATURE = "measurement_feature"


@dataclass(frozen=True, slots=True)
class RuntimeProjectedColumnIdentity:
    """Semantic identity for one projected table column."""

    role: RuntimeProjectedColumnRole
    subject: MeasurementSubject | None = None
    object_name: str | None = None
    feature_name: str | None = None
    source_image_name: str | None = None
    metadata_key: str | None = None
    axis_name: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "role",
            (
                self.role
                if isinstance(self.role, RuntimeProjectedColumnRole)
                else RuntimeProjectedColumnRole(self.role)
            ),
        )
        for field_name in (
            "object_name",
            "feature_name",
            "source_image_name",
            "metadata_key",
            "axis_name",
        ):
            value = getattr(self, field_name)
            if value == "":
                raise ValueError(
                    f"RuntimeProjectedColumnIdentity.{field_name} cannot be empty."
                )


@dataclass(frozen=True, slots=True)
class RuntimeProjectedTable:
    """External table projection derived from runtime semantics."""

    table_name: str
    rows: tuple[Mapping[str, Any], ...]
    columns: tuple[str, ...]
    subject: MeasurementSubject | None = None

    def __post_init__(self) -> None:
        if not self.table_name:
            raise ValueError("RuntimeProjectedTable.table_name cannot be empty.")
        if any(not column for column in self.columns):
            raise ValueError("RuntimeProjectedTable.columns cannot contain empties.")
        object.__setattr__(self, "rows", tuple(self.rows))
        object.__setattr__(self, "columns", tuple(dict.fromkeys(self.columns)))


class RuntimeTableProjectionDialect(ABC):
    """Render runtime table semantics into one external tabular dialect."""

    @abstractmethod
    def image_table_name(self) -> str:
        """Return the external table name for image-scope measurements."""

    @abstractmethod
    def object_table_name(self, object_name: str) -> str:
        """Return the external table name for one object-scope measurement table."""

    @abstractmethod
    def relationship_table_name(self, relationship_name: str) -> str:
        """Return the external table name for one relationship table."""

    @abstractmethod
    def column_name(self, identity: RuntimeProjectedColumnIdentity) -> str:
        """Return the external column name for one semantic column identity."""
