"""Typed OpenHCS virtual-workspace metadata carriers."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, TypeAlias

from polystore.atomic import LOCK_CONFIG, FileLockError, atomic_update_json
from polystore.virtual_workspace import SourcePixelRef

from openhcs.core.artifacts import ArtifactType
from openhcs.core.source_bindings import SourceProjectionRole
from openhcs.core.source_metadata import (
    SourceMetadataMapping,
    SourceMetadataScalar,
    SourceMetadataValue,
)
from openhcs.core.source_projection import (
    OpenHCSPlaneAddress,
    SourceArtifactProjection,
    SourcePlaneProjection,
    SourceProjection,
)

@dataclass(frozen=True)
class OpenHCSMetadataConfig:
    """Configuration owned by the OpenHCS metadata file contract."""

    METADATA_FILENAME: str = os.getenv(
        "OPENHCS_METADATA_FILENAME",
        "openhcs_metadata.json",
    )
    SUBDIRECTORIES_KEY: str = "subdirectories"
    AVAILABLE_BACKENDS_KEY: str = "available_backends"
    DEFAULT_TIMEOUT: float = LOCK_CONFIG.DEFAULT_TIMEOUT


METADATA_CONFIG = OpenHCSMetadataConfig()


class MetadataWriteError(Exception):
    """Raised when an OpenHCS metadata transaction fails."""


class AtomicMetadataWriter:
    """Atomically update subdirectory-keyed OpenHCS metadata."""

    def __init__(self, timeout: float = METADATA_CONFIG.DEFAULT_TIMEOUT):
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)

    def update_available_backends(
        self,
        metadata_path: str | Path,
        available_backends: dict[str, bool],
    ) -> None:
        def update(data: dict[str, Any] | None) -> dict[str, Any]:
            if data is None:
                raise MetadataWriteError(
                    "Cannot update backends: metadata file does not exist"
                )
            data[METADATA_CONFIG.AVAILABLE_BACKENDS_KEY] = available_backends
            return data

        self._execute_update(metadata_path, update)

    def merge_subdirectory_metadata(
        self,
        metadata_path: str | Path,
        subdirectory_updates: dict[str, dict[str, Any]],
    ) -> None:
        def update(data: dict[str, Any] | None) -> dict[str, Any]:
            data = self._ensure_subdirectories_structure(data)
            subdirectories = data[METADATA_CONFIG.SUBDIRECTORIES_KEY]
            for subdirectory_name, fields in subdirectory_updates.items():
                subdirectory = subdirectories.setdefault(subdirectory_name, {})
                for key, value in fields.items():
                    if (
                        key == METADATA_CONFIG.AVAILABLE_BACKENDS_KEY
                        and isinstance(value, dict)
                    ):
                        subdirectory[key] = {
                            **subdirectory.get(key, {}),
                            **value,
                        }
                    else:
                        subdirectory[key] = value
            return data

        self._execute_update(
            metadata_path,
            update,
            {METADATA_CONFIG.SUBDIRECTORIES_KEY: {}},
        )

    def replace_subdirectory_metadata(
        self,
        metadata_path: str | Path,
        subdirectory_name: str,
        subdirectory_metadata: dict[str, Any],
    ) -> None:
        def update(data: dict[str, Any] | None) -> dict[str, Any]:
            data = self._ensure_subdirectories_structure(data)
            data[METADATA_CONFIG.SUBDIRECTORIES_KEY][subdirectory_name] = dict(
                subdirectory_metadata
            )
            return data

        self._execute_update(
            metadata_path,
            update,
            {METADATA_CONFIG.SUBDIRECTORIES_KEY: {}},
        )

    def _execute_update(
        self,
        metadata_path: str | Path,
        update: Callable[[dict[str, Any] | None], dict[str, Any]],
        default_data: dict[str, Any] | None = None,
    ) -> None:
        try:
            atomic_update_json(metadata_path, update, self.timeout, default_data)
        except FileLockError as exc:
            raise MetadataWriteError(f"Failed to update metadata: {exc}") from exc

    @staticmethod
    def _ensure_subdirectories_structure(
        data: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if data is None:
            data = {}
        data.setdefault(METADATA_CONFIG.SUBDIRECTORIES_KEY, {})
        return data


def get_metadata_path(plate_root: str | Path) -> Path:
    """Return the canonical metadata path for one OpenHCS plate root."""

    return Path(plate_root) / METADATA_CONFIG.METADATA_FILENAME


@dataclass(frozen=True)
class OpenHCSMetadataFields:
    """Field identities declared by the OpenHCS metadata contract."""

    SUBDIRECTORIES: str = METADATA_CONFIG.SUBDIRECTORIES_KEY
    IMAGE_FILES: str = "image_files"
    AVAILABLE_BACKENDS: str = METADATA_CONFIG.AVAILABLE_BACKENDS_KEY
    SOURCE_METADATA: str = "source_metadata"
    SOURCE_DIAGNOSTICS: str = "source_diagnostics"
    WORKSPACE_MAPPING: str = "workspace_mapping"
    GRID_DIMENSIONS: str = "grid_dimensions"
    PIXEL_SIZE: str = "pixel_size"
    SOURCE_FILENAME_PARSER_NAME: str = "source_filename_parser_name"
    MICROSCOPE_HANDLER_NAME: str = "microscope_handler_name"
    CHANNELS: str = "channels"
    WELLS: str = "wells"
    SITES: str = "sites"
    Z_INDEXES: str = "z_indexes"
    TIMEPOINTS: str = "timepoints"
    OBJECTIVES: str = "objectives"
    ACQUISITION_DATETIME: str = "acquisition_datetime"
    PLATE_NAME: str = "plate_name"
    DEFAULT_SUBDIRECTORY: str = "."
    MICROSCOPE_TYPE: str = "openhcsdata"


FIELDS = OpenHCSMetadataFields()

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | Mapping[str, "JsonValue"] | Sequence["JsonValue"]
OpenHCSMetadataPayload: TypeAlias = Mapping[str, JsonValue]
OpenHCSSubdirectoryPayload: TypeAlias = Mapping[str, JsonValue]


@dataclass(frozen=True, slots=True)
class OpenHCSMetadataSubdirectories:
    """Typed view over OpenHCS metadata subdirectory payloads."""

    metadata: OpenHCSMetadataPayload

    def items(self) -> tuple[tuple[str, OpenHCSSubdirectoryPayload], ...]:
        subdirectories = self.metadata.get(FIELDS.SUBDIRECTORIES)
        if subdirectories is None:
            return ()
        if not isinstance(subdirectories, Mapping):
            raise RuntimeError("OpenHCS metadata subdirectories must be a mapping.")
        items: list[tuple[str, OpenHCSSubdirectoryPayload]] = []
        for name, subdirectory in subdirectories.items():
            if not isinstance(subdirectory, Mapping):
                raise RuntimeError(
                    f"OpenHCS metadata subdirectory {name!r} must be a mapping."
                )
            items.append((str(name), subdirectory))
        return tuple(items)

    def values(self) -> tuple[OpenHCSSubdirectoryPayload, ...]:
        return tuple(subdirectory for _, subdirectory in self.items())

    def has_workspace_mapping(self) -> bool:
        return any(
            VirtualWorkspaceMapping.from_subdirectory(subdirectory).has_entries
            for subdirectory in self.values()
        )


@dataclass(frozen=True, slots=True)
class VirtualWorkspaceMapping:
    """Validated virtual-workspace mapping entries for one subdirectory."""

    entries: Mapping[str, SourcePixelRef]

    @classmethod
    def from_subdirectory(
        cls,
        subdirectory: OpenHCSSubdirectoryPayload,
    ) -> "VirtualWorkspaceMapping":
        mapping = subdirectory.get(FIELDS.WORKSPACE_MAPPING)
        if mapping is None:
            return cls(MappingProxyType({}))
        if not isinstance(mapping, Mapping):
            raise RuntimeError("virtual_workspace workspace_mapping must be a mapping.")
        return cls(
            MappingProxyType(
                {
                    str(key): SourcePixelRef.from_workspace_mapping(value)
                    for key, value in mapping.items()
                }
            )
        )

    @property
    def has_entries(self) -> bool:
        return bool(self.entries)

    def source_ref_for(self, virtual_path: str) -> SourcePixelRef | None:
        return self.entries.get(virtual_path)

    def require_source_ref(self, virtual_path: str) -> SourcePixelRef:
        source_ref = self.source_ref_for(virtual_path)
        if source_ref is None:
            raise ValueError(
                "OpenHCS workspace metadata is missing a source mapping for "
                f"{virtual_path!r}."
            )
        return source_ref


@dataclass(frozen=True, slots=True)
class VirtualWorkspaceSourceProjectionEntries:
    """Validated nominal source projections keyed by canonical virtual path."""

    entries: Mapping[str, SourceProjection]

    @classmethod
    def from_subdirectory(
        cls,
        subdirectory: OpenHCSSubdirectoryPayload,
    ) -> "VirtualWorkspaceSourceProjectionEntries":
        records = subdirectory.get("source_projection")
        if records is None:
            return cls(MappingProxyType({}))
        if not isinstance(records, Sequence) or isinstance(records, str):
            raise RuntimeError("virtual_workspace source_projection must be a list.")
        entries: dict[str, SourceProjection] = {}
        for record in records:
            virtual_path, projection = cls._projection_record(record)
            if virtual_path in entries:
                raise RuntimeError(
                    "virtual_workspace source_projection contains duplicate path "
                    f"{virtual_path!r}."
                )
            entries[virtual_path] = projection
        return cls(MappingProxyType(entries))

    @classmethod
    def _projection_record(
        cls,
        record: JsonValue,
    ) -> tuple[str, SourceProjection]:
        if not isinstance(record, Mapping):
            raise RuntimeError(
                "virtual_workspace source_projection records must be mappings."
            )
        virtual_path = cls._required_text(record, "virtual_path")
        address_value = record.get("address")
        if not isinstance(address_value, Mapping):
            raise RuntimeError(
                "virtual_workspace source_projection address must be a mapping."
            )
        address = OpenHCSPlaneAddress(
            well=cls._required_text(address_value, "well"),
            site=cls._required_text(address_value, "site"),
            channel=cls._required_text(address_value, "channel"),
            z_index=cls._required_text(address_value, "z_index"),
            timepoint=cls._required_text(address_value, "timepoint"),
        )
        ref_value = record.get("ref")
        ref = SourcePixelRef.from_workspace_mapping(ref_value)
        try:
            projection_role = SourceProjectionRole(
                cls._required_text(record, "projection_role")
            )
        except ValueError as exc:
            raise RuntimeError(
                "virtual_workspace source_projection has an unknown projection_role."
            ) from exc
        source_metadata = cls._optional_metadata(record, "source_metadata")
        component_labels = cls._optional_component_labels(record)
        source_alias = cls._optional_text(record, "source_alias")
        if projection_role is SourceProjectionRole.PRIMARY_PLANE:
            projection: SourceProjection = SourcePlaneProjection(
                address=address,
                ref=ref,
                source_alias=source_alias,
                source_metadata=source_metadata,
                component_labels=component_labels,
            )
        else:
            if source_alias is None:
                raise RuntimeError(
                    "Source-artifact projection records require source_alias."
                )
            artifact_kind = cls._required_text(record, "artifact_kind")
            projection = SourceArtifactProjection(
                address=address,
                ref=ref,
                source_alias=source_alias,
                artifact_kind=ArtifactType.coerce(artifact_kind),
                source_metadata=source_metadata,
                component_labels=component_labels,
            )
        return virtual_path, projection

    @staticmethod
    def _required_text(record: Mapping[str, JsonValue], field: str) -> str:
        value = record.get(field)
        if not isinstance(value, (str, int)) or isinstance(value, bool):
            raise RuntimeError(
                f"virtual_workspace source_projection field {field!r} must be text."
            )
        text = str(value).strip()
        if not text:
            raise RuntimeError(
                f"virtual_workspace source_projection field {field!r} cannot be empty."
            )
        return text

    @classmethod
    def _optional_text(
        cls,
        record: Mapping[str, JsonValue],
        field: str,
    ) -> str | None:
        if field not in record:
            return None
        return cls._required_text(record, field)

    @staticmethod
    def _optional_metadata(
        record: Mapping[str, JsonValue],
        field: str,
    ) -> SourceMetadataMapping:
        if field not in record:
            return MappingProxyType({})
        return VirtualWorkspaceSourceMetadataEntries.normalize_metadata_fields(
            record[field]
        )

    @classmethod
    def _optional_component_labels(
        cls,
        record: Mapping[str, JsonValue],
    ) -> Mapping[str, str | None]:
        labels = record.get("component_labels")
        if labels is None:
            return MappingProxyType({})
        if not isinstance(labels, Mapping):
            raise RuntimeError(
                "virtual_workspace source_projection component_labels must be a mapping."
            )
        normalized: dict[str, str | None] = {}
        for key, value in labels.items():
            if value is not None and not isinstance(value, str):
                raise RuntimeError(
                    "virtual_workspace source_projection component label values "
                    "must be text or null."
                )
            normalized[str(key)] = value
        return MappingProxyType(normalized)


@dataclass(frozen=True, slots=True)
class VirtualWorkspaceSourceMetadataEntries:
    """Validated source metadata entries for one virtual-workspace subdirectory."""

    entries: Mapping[str, SourceMetadataMapping]

    @classmethod
    def from_subdirectory(
        cls,
        subdirectory: OpenHCSSubdirectoryPayload,
    ) -> "VirtualWorkspaceSourceMetadataEntries":
        source_metadata = subdirectory.get(FIELDS.SOURCE_METADATA)
        if source_metadata is None:
            return cls(MappingProxyType({}))
        if not isinstance(source_metadata, Mapping):
            raise RuntimeError(
                "virtual_workspace source metadata must be a path-keyed mapping."
            )
        return cls(
            MappingProxyType(
                {
                    str(virtual_path): cls.normalize_metadata_fields(metadata_fields)
                    for virtual_path, metadata_fields in source_metadata.items()
                }
            )
        )

    @staticmethod
    def normalize_metadata_fields(metadata_fields: JsonValue) -> SourceMetadataMapping:
        if not isinstance(metadata_fields, Mapping):
            raise RuntimeError("virtual_workspace source metadata values must be mappings.")
        return MappingProxyType(
            {
                str(key): VirtualWorkspaceSourceMetadataEntries.normalize_metadata_value(
                    value
                )
                for key, value in metadata_fields.items()
            }
        )

    @staticmethod
    def normalize_metadata_value(value: JsonValue) -> SourceMetadataValue:
        if isinstance(value, Mapping):
            return MappingProxyType(
                {
                    str(nested_key): VirtualWorkspaceSourceMetadataEntries.require_scalar_metadata_value(
                        nested_value
                    )
                    for nested_key, nested_value in value.items()
                }
            )
        return VirtualWorkspaceSourceMetadataEntries.require_scalar_metadata_value(value)

    @staticmethod
    def require_scalar_metadata_value(value: JsonValue) -> SourceMetadataScalar:
        if isinstance(value, Mapping) or (
            isinstance(value, Sequence) and not isinstance(value, str)
        ):
            raise RuntimeError(
                "virtual_workspace source metadata supports scalar values and "
                "one-level scalar mappings only."
            )
        if value is None:
            return None
        if not isinstance(value, (str, int, float, bool)):
            raise RuntimeError(
                "virtual_workspace source metadata scalar values must be strings, "
                "numbers, booleans, or null."
            )
        return value

    def metadata_for(self, virtual_path: str) -> SourceMetadataMapping:
        metadata = self.entries.get(virtual_path)
        if metadata is None:
            return MappingProxyType({})
        return metadata


@dataclass(frozen=True, slots=True)
class VirtualWorkspaceChannelLabels:
    """Validated channel labels for one virtual-workspace subdirectory."""

    entries: Mapping[str, str]

    @classmethod
    def from_subdirectory(
        cls,
        subdirectory: OpenHCSSubdirectoryPayload,
    ) -> "VirtualWorkspaceChannelLabels":
        channels = subdirectory.get(FIELDS.CHANNELS)
        if channels is None:
            return cls(MappingProxyType({}))
        if not isinstance(channels, Mapping):
            raise RuntimeError("virtual_workspace channels must be a mapping.")
        return cls(
            MappingProxyType({str(key): str(value) for key, value in channels.items()})
        )

    def label_for(self, channel_value: SourceMetadataScalar) -> str | None:
        return self.entries.get(str(channel_value))
