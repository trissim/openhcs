"""Typed source-plane projection authority for OpenHCS workspaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from openhcs.core.pipeline_image_schema import SOURCE_IMAGE_TYPE_METADATA_FIELD


_COMPONENT_FIELDS = ("well", "site", "channel", "z_index", "timepoint")
_SOURCE_REF_FIELDS = (
    "backend",
    "reader",
    "source_path",
    "series_index",
    "plane_index",
    "c",
    "z",
    "t",
)


@dataclass(frozen=True, slots=True)
class OpenHCSPlaneAddress:
    """Canonical OpenHCS logical address for one image plane."""

    well: str
    site: str
    channel: str
    z_index: str
    timepoint: str

    def __post_init__(self) -> None:
        for name in _COMPONENT_FIELDS:
            value = getattr(self, name)
            if value is None or value == "":
                raise ValueError(f"{name} cannot be empty in an OpenHCS plane address.")
            object.__setattr__(self, name, str(value))

    def as_component_metadata(self) -> dict[str, str]:
        """Return parser-compatible component metadata."""

        return {
            "well": self.well,
            "site": self.site,
            "channel": self.channel,
            "z_index": self.z_index,
            "timepoint": self.timepoint,
        }

    @classmethod
    def from_parsed(cls, parsed: Mapping[str, Any]) -> "OpenHCSPlaneAddress":
        """Create an address from parser output."""

        missing = [name for name in _COMPONENT_FIELDS if parsed.get(name) is None]
        if missing:
            raise ValueError(
                "Parsed OpenHCS virtual filename lacks required components: "
                + ", ".join(missing)
            )
        return cls(
            well=str(parsed["well"]),
            site=str(parsed["site"]),
            channel=str(parsed["channel"]),
            z_index=str(parsed["z_index"]),
            timepoint=str(parsed["timepoint"]),
        )


@dataclass(frozen=True, slots=True)
class SourcePixelRef:
    """Backend-resolvable source pixel reference for one OpenHCS plane."""

    backend: str
    source_path: str
    reader: str | None = None
    series_index: int | None = None
    plane_index: int | None = None
    source_channel: int | None = None
    source_z_index: int | None = None
    source_timepoint: int | None = None

    def __post_init__(self) -> None:
        if not self.backend:
            raise ValueError("SourcePixelRef.backend cannot be empty.")
        if not self.source_path:
            raise ValueError("SourcePixelRef.source_path cannot be empty.")
        object.__setattr__(self, "backend", str(self.backend))
        object.__setattr__(self, "source_path", str(self.source_path))
        if self.reader is not None:
            object.__setattr__(self, "reader", str(self.reader))

    def to_legacy_workspace_mapping(self) -> dict[str, Any]:
        """Return the structured workspace_mapping payload for this source ref."""

        payload: dict[str, Any] = {
            "backend": self.backend,
            "source_path": self.source_path,
        }
        if self.reader is not None:
            payload["reader"] = self.reader
        if self.series_index is not None:
            payload["series_index"] = int(self.series_index)
        if self.plane_index is not None:
            payload["plane_index"] = int(self.plane_index)
        if self.source_channel is not None:
            payload["c"] = int(self.source_channel)
        if self.source_z_index is not None:
            payload["z"] = int(self.source_z_index)
        if self.source_timepoint is not None:
            payload["t"] = int(self.source_timepoint)
        return payload

    def source_metadata(self) -> dict[str, str]:
        """Return string metadata useful for provenance views."""

        payload = self.to_legacy_workspace_mapping()
        return {
            key: str(value)
            for key, value in payload.items()
            if key in _SOURCE_REF_FIELDS and key != "backend"
        }


@dataclass(frozen=True, slots=True)
class SourcePlaneProjection:
    """One canonical OpenHCS plane mapped to one source pixel reference."""

    address: OpenHCSPlaneAddress
    ref: SourcePixelRef
    source_alias: str | None = None
    image_type: str | None = None
    source_metadata: Mapping[str, str] = field(
        default_factory=lambda: MappingProxyType({})
    )
    component_labels: Mapping[str, str | None] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_metadata",
            MappingProxyType({str(k): str(v) for k, v in self.source_metadata.items()}),
        )
        object.__setattr__(
            self,
            "component_labels",
            MappingProxyType(
                {
                    str(k): None if v is None else str(v)
                    for k, v in self.component_labels.items()
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class SourceProjectionSet:
    """Validated source projection set for one OpenHCS source workspace."""

    projections: tuple[SourcePlaneProjection, ...]

    def __post_init__(self) -> None:
        projections = tuple(self.projections)
        if not projections:
            raise ValueError("SourceProjectionSet requires at least one projection.")
        seen: set[OpenHCSPlaneAddress] = set()
        for projection in projections:
            if projection.address in seen:
                raise ValueError(f"Duplicate source projection address: {projection.address}")
            seen.add(projection.address)
        object.__setattr__(self, "projections", projections)

    def metadata_dict(
        self,
        *,
        parser: Any,
        microscope_handler_name: str,
        source_filename_parser_name: str,
        grid_dimensions: list[int],
        pixel_size: float,
        available_backends: Mapping[str, bool] | None = None,
        main: bool | None = None,
        results_dir: str | None = None,
        image_extension: str = ".tif",
    ) -> dict[str, Any]:
        """Serialize the projection set to one OpenHCS subdirectory metadata dict."""

        serializer = SourceProjectionMetadataSerializer(
            parser=parser,
            image_extension=image_extension,
        )
        return serializer.metadata_dict(
            self,
            microscope_handler_name=microscope_handler_name,
            source_filename_parser_name=source_filename_parser_name,
            grid_dimensions=grid_dimensions,
            pixel_size=pixel_size,
            available_backends=available_backends,
            main=main,
            results_dir=results_dir,
        )


@dataclass(frozen=True, slots=True)
class SourceProjectionMetadataSerializer:
    """Serialize projection identity into OpenHCS metadata-compatible fields."""

    parser: Any
    image_extension: str = ".tif"

    def metadata_dict(
        self,
        projection_set: SourceProjectionSet,
        *,
        microscope_handler_name: str,
        source_filename_parser_name: str,
        grid_dimensions: list[int],
        pixel_size: float,
        available_backends: Mapping[str, bool] | None = None,
        main: bool | None = None,
        results_dir: str | None = None,
    ) -> dict[str, Any]:
        """Return an OpenHCS subdirectory metadata dictionary."""

        projection_paths = tuple(
            (projection, self.virtual_path(projection.address))
            for projection in projection_set.projections
        )
        metadata: dict[str, Any] = {
            "microscope_handler_name": microscope_handler_name,
            "source_filename_parser_name": source_filename_parser_name,
            "grid_dimensions": list(grid_dimensions),
            "pixel_size": pixel_size,
            "image_files": [path for _, path in projection_paths],
            "channels": self._component_values(projection_set, "channel"),
            "wells": self._component_values(projection_set, "well"),
            "sites": self._component_values(projection_set, "site"),
            "z_indexes": self._component_values(projection_set, "z_index"),
            "timepoints": self._component_values(projection_set, "timepoint"),
            "available_backends": dict(
                available_backends
                if available_backends is not None
                else self._available_backends(projection_set)
            ),
            "workspace_mapping": {
                path: projection.ref.to_legacy_workspace_mapping()
                for projection, path in projection_paths
            },
            "source_metadata": {
                path: self._source_metadata(projection)
                for projection, path in projection_paths
            },
            "source_projection": [
                self._source_projection_payload(projection, path)
                for projection, path in projection_paths
            ],
        }
        if main is not None:
            metadata["main"] = main
        if results_dir is not None:
            metadata["results_dir"] = results_dir
        return metadata

    def virtual_path(self, address: OpenHCSPlaneAddress) -> str:
        """Render and validate a canonical OpenHCS virtual image filename."""

        path = self.parser.construct_filename(
            well=address.well,
            site=_parser_component(address.site),
            channel=_parser_component(address.channel),
            z_index=_parser_component(address.z_index),
            timepoint=_parser_component(address.timepoint),
            extension=self.image_extension,
        )
        parsed = self.parser.parse_filename(path)
        if parsed is None:
            raise ValueError(f"Generated virtual filename is not parser-readable: {path!r}")
        parsed_address = OpenHCSPlaneAddress.from_parsed(parsed)
        if parsed_address != address:
            raise ValueError(
                "Generated virtual filename parsed to a different address: "
                f"{path!r} -> {parsed_address!r}, expected {address!r}."
            )
        return path

    def _component_values(
        self,
        projection_set: SourceProjectionSet,
        component: str,
    ) -> dict[str, str | None]:
        values: dict[str, str | None] = {}
        for projection in projection_set.projections:
            key = getattr(projection.address, component)
            label = projection.component_labels.get(component)
            previous = values.get(key)
            if previous is not None and label is not None and previous != label:
                raise ValueError(
                    f"Conflicting label for {component}={key!r}: "
                    f"{previous!r} vs {label!r}."
                )
            values[key] = label if label is not None else previous
        return values

    def _available_backends(
        self,
        projection_set: SourceProjectionSet,
    ) -> dict[str, bool]:
        return {
            projection.ref.backend: True
            for projection in projection_set.projections
        }

    def _source_metadata(
        self,
        projection: SourcePlaneProjection,
    ) -> dict[str, str]:
        metadata = {
            **projection.ref.source_metadata(),
            **dict(projection.source_metadata),
        }
        if projection.source_alias is not None:
            metadata.setdefault("source_alias", str(projection.source_alias))
        if projection.image_type is not None:
            metadata.setdefault(SOURCE_IMAGE_TYPE_METADATA_FIELD, str(projection.image_type))
        for component, value in projection.address.as_component_metadata().items():
            existing = metadata.get(component)
            if existing is not None and str(existing) != value:
                raise ValueError(
                    f"source_metadata for {projection.address!r} conflicts with "
                    f"canonical {component}: {existing!r} vs {value!r}."
                )
            metadata[component] = value
        return metadata

    def _source_projection_payload(
        self,
        projection: SourcePlaneProjection,
        path: str,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "virtual_path": path,
            "address": projection.address.as_component_metadata(),
            "ref": projection.ref.to_legacy_workspace_mapping(),
        }
        if projection.source_alias is not None:
            payload["source_alias"] = projection.source_alias
        if projection.image_type is not None:
            payload["image_type"] = projection.image_type
        if projection.source_metadata:
            payload["source_metadata"] = dict(projection.source_metadata)
        if projection.component_labels:
            payload["component_labels"] = dict(projection.component_labels)
        return payload


def _parser_component(value: str) -> str | int:
    """Preserve semantic numeric axes while letting parsers own formatting."""

    return int(value) if value.isdecimal() else value
