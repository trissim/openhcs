"""Typed OpenHCS virtual-workspace metadata carriers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TypeAlias

from openhcs.core.source_metadata import (
    SourceMetadataMapping,
    SourceMetadataScalar,
    SourceMetadataValue,
)
from openhcs.microscopes.openhcs import FIELDS

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | Mapping[str, "JsonValue"] | Sequence["JsonValue"]
OpenHCSMetadataPayload: TypeAlias = Mapping[str, JsonValue]
OpenHCSSubdirectoryPayload: TypeAlias = Mapping[str, JsonValue]
WorkspaceSourceRef: TypeAlias = JsonValue


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

    entries: Mapping[str, WorkspaceSourceRef]

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
        return cls(MappingProxyType({str(key): value for key, value in mapping.items()}))

    @property
    def has_entries(self) -> bool:
        return bool(self.entries)

    def source_ref_for(self, virtual_path: str) -> WorkspaceSourceRef | None:
        return self.entries.get(virtual_path)

    def require_source_ref(self, virtual_path: str) -> WorkspaceSourceRef:
        source_ref = self.source_ref_for(virtual_path)
        if source_ref is None:
            raise ValueError(
                "OpenHCS workspace metadata is missing a source mapping for "
                f"{virtual_path!r}."
            )
        return source_ref


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
        return str(value)

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
