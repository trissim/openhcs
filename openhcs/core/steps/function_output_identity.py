"""Centralized output filename identity for FunctionStep main-flow writes."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
import re
from typing import TYPE_CHECKING, Mapping, TypeAlias

from openhcs.constants.constants import AllComponents
from openhcs.core.runtime_values import (
    ImagePayloadMetadata,
    RuntimeArrayData,
    image_payload_metadata,
)
from openhcs.core.source_image_provenance import (
    SourceComponentMetadata,
    SourceImageProvenancePlane,
)
from openhcs.core.source_matching import source_component_metadata_value
from openhcs.microscopes.microscope_interfaces import FilenameParser

if TYPE_CHECKING:
    from openhcs.core.context.processing_context import ProcessingContext

ParsedFilenameValue: TypeAlias = str | int | float | bool | None
FunctionOutputComponentValue: TypeAlias = str | int
FunctionOutputComponentValues: TypeAlias = Mapping[str, FunctionOutputComponentValue]

@dataclass(frozen=True, slots=True)
class FunctionOutputPathRequest:
    """Inputs needed to project one runtime output slice into a VFS path."""

    parser: FilenameParser
    output_dir: Path
    output_payload: RuntimeArrayData
    input_path: str | None
    source_identity_stack_axes: frozenset[str] = field(default_factory=frozenset)

@dataclass(frozen=True, slots=True)
class FunctionOutputIdentity:
    """Semantic component identity used for one output filename."""

    component_values: FunctionOutputComponentValues
    extension: str | None
    source: str
    filename_component_values: FunctionOutputComponentValues | None = field(
        default=None,
        kw_only=True,
    )
    filename_qualifier: str | None = field(default=None, kw_only=True)

    def component_metadata(self) -> SourceComponentMetadata:
        """Return parser-compatible component metadata for this output identity."""
        metadata = dict(self.component_values)
        if self.extension is not None:
            metadata["extension"] = self.extension
        return metadata

    def filename_component_metadata(self) -> SourceComponentMetadata:
        """Return parser-compatible metadata used to construct the output filename."""
        metadata = dict(
            self.filename_component_values
            if self.filename_component_values is not None
            else self.component_values
        )
        if self.extension is not None:
            metadata["extension"] = self.extension
        return metadata

    def with_filename_qualifier(self, qualifier: str) -> "FunctionOutputIdentity":
        """Return this identity qualified by a declared output surface name."""
        return replace(self, filename_qualifier=FilenameQualifier.from_value(qualifier))

    def without_filename_qualifier(self) -> "FunctionOutputIdentity":
        """Return this identity without its output-surface filename qualifier."""
        return replace(self, filename_qualifier=None)


class FilenameQualifier:
    """Normalize declared output surface names for filename disambiguation."""

    _unsafe_pattern = re.compile(r"[^A-Za-z0-9_.-]+")

    @classmethod
    def from_value(cls, value: str) -> str:
        qualifier = cls._unsafe_pattern.sub("_", str(value).strip())
        qualifier = qualifier.strip("_.-")
        if not qualifier:
            raise ValueError("Filename qualifier cannot be empty.")
        return qualifier

@dataclass(frozen=True, slots=True)
class FunctionOutputParserContext:
    """Parser-facing microscope context for FunctionStep output finalization."""

    parser: FilenameParser
    microscope_type: str

    @classmethod
    def from_processing_context(
        cls,
        context: "ProcessingContext",
    ) -> "FunctionOutputParserContext":
        handler = context.microscope_handler
        return cls(
            parser=handler.parser,
            microscope_type=handler.microscope_type,
        )

    @property
    def parser_name(self) -> str:
        return self.parser.__class__.__name__

    def parse_path_metadata(self, path: str | Path) -> SourceComponentMetadata | None:
        return self.parser.parse_filename(Path(path).name)

class FunctionOutputPathAuthority:
    """Construct output filenames from semantic identity through the parser."""

    @classmethod
    def output_path(cls, request: FunctionOutputPathRequest) -> Path:
        return cls.output_path_for_identity(
            request,
            FunctionOutputIdentityAuthority.identity(request),
        )

    @classmethod
    def output_filename(cls, request: FunctionOutputPathRequest) -> str:
        return cls.output_filename_for_identity(
            request,
            FunctionOutputIdentityAuthority.identity(request),
        )

    @classmethod
    def output_path_for_identity(
        cls,
        request: FunctionOutputPathRequest,
        identity: FunctionOutputIdentity,
    ) -> Path:
        return Path(request.output_dir) / cls.output_filename_for_identity(
            request,
            identity,
        )

    @classmethod
    def output_filename_for_identity(
        cls,
        request: FunctionOutputPathRequest,
        identity: FunctionOutputIdentity,
    ) -> str:
        return cls.filename_for_identity(request.parser, identity)

    @classmethod
    def filename_for_identity(
        cls,
        parser: FilenameParser,
        identity: FunctionOutputIdentity,
    ) -> str:
        component_values = dict(
            identity.filename_component_values
            if identity.filename_component_values is not None
            else identity.component_values
        )
        extension = identity.extension
        try:
            if extension is None:
                filename = parser.construct_filename(**component_values)
            else:
                filename = parser.construct_filename(
                    extension=extension,
                    **component_values,
                )
            return cls._qualified_filename(filename, identity.filename_qualifier)
        except Exception as exc:
            raise ValueError(
                "Cannot construct FunctionStep output filename from "
                f"{identity.source} identity. Components={component_values!r}, "
                f"extension={extension!r}."
            ) from exc

    @staticmethod
    def _qualified_filename(filename: str, qualifier: str | None) -> str:
        if qualifier is None:
            return filename
        path = Path(filename)
        suffixes = "".join(path.suffixes)
        stem = path.name[: -len(suffixes)] if suffixes else path.name
        return f"{stem}_{qualifier}{suffixes}"


class FunctionOutputExtensionAuthority:
    """Resolve filename extensions from source metadata or paths."""

    @classmethod
    def first(cls, *extensions: str | None) -> str | None:
        for extension in extensions:
            if extension is not None:
                return extension
        return None

    @classmethod
    def from_metadata(
        cls,
        metadata: SourceComponentMetadata | None,
    ) -> str | None:
        if metadata is None:
            return None
        return cls.from_raw(metadata.get("extension"))

    @classmethod
    def from_path(cls, path: str | None) -> str | None:
        if path is None:
            return None
        return cls.from_raw("".join(Path(path).suffixes))

    @staticmethod
    def from_raw(raw_extension: ParsedFilenameValue) -> str | None:
        if raw_extension is None:
            return None
        extension = str(raw_extension)
        if not extension:
            return None
        return extension


class FunctionOutputComponentIdentityAuthority:
    """Normalize parser/source component values for output identity."""

    @classmethod
    def from_source_metadata(
        cls,
        metadata: SourceComponentMetadata | None,
    ) -> dict[str, FunctionOutputComponentValue]:
        if metadata is None:
            return {}
        return {
            component.value: cls.coerce_component_value(component, value)
            for component in AllComponents
            if (value := source_component_metadata_value(metadata, component))
            is not None
        }

    @classmethod
    def from_parsed(
        cls,
        parsed: SourceComponentMetadata,
    ) -> dict[str, FunctionOutputComponentValue]:
        return {
            str(key): cls.coerce_component_value(
                AllComponents.from_value(str(key)),
                value,
            )
            for key, value in parsed.items()
            if str(key) != "extension"
        }

    @staticmethod
    def coerce_component_value(
        component: AllComponents | None,
        value: ParsedFilenameValue,
    ) -> FunctionOutputComponentValue:
        if component is None or not component.is_variable_axis():
            return str(value)
        value_text = str(value)
        return int(value_text) if value_text.isdecimal() else value_text


class FunctionOutputIdentityAuthority:
    """Resolve the single source of output identity for a runtime slice."""

    @classmethod
    def identity(cls, request: FunctionOutputPathRequest) -> FunctionOutputIdentity:
        metadata = image_payload_metadata(request.output_payload)
        payload_identity = cls.identity_from_metadata(
            request.parser,
            metadata,
            fallback_identity_path=request.input_path,
            source_identity_stack_axes=request.source_identity_stack_axes,
        )
        if payload_identity is not None:
            if payload_identity.extension is None:
                raise ValueError(
                    "FunctionStep output payload carries component identity but "
                    "no source path or extension. Refusing to borrow an input "
                    "extension for a semantic payload identity."
                )
            return payload_identity
        return cls._input_aligned_identity(request)

    @classmethod
    def identity_from_metadata(
        cls,
        parser: FilenameParser,
        metadata: ImagePayloadMetadata,
        *,
        fallback_identity_path: str | None = None,
        source_identity_stack_axes: frozenset[str] = frozenset(),
    ) -> FunctionOutputIdentity | None:
        """Return parser-backed identity carried by image payload metadata."""
        provenance_planes = metadata.source_image_provenance_planes
        if provenance_planes.count > 1:
            if source_identity_stack_axes:
                return cls._source_stack_identity_from_provenance(
                    parser,
                    metadata,
                    source_identity_stack_axes,
                    fallback_identity_path=fallback_identity_path,
                )
            if fallback_identity_path is not None:
                return None
            raise ValueError(
                "FunctionStep output slice carries multi-plane source "
                "provenance. Refusing to collapse multiple semantic identities "
                "into one output filename."
            )

        identity = cls._identity_from_metadata(
            metadata.source_component_metadata,
            extension=FunctionOutputExtensionAuthority.first(
                FunctionOutputExtensionAuthority.from_metadata(
                    metadata.source_component_metadata
                ),
                FunctionOutputExtensionAuthority.from_path(metadata.source_path),
            ),
            source="payload component metadata",
        )
        if identity is not None:
            return cls._complete_identity_from_paths(
                parser,
                identity,
                    (
                        (
                            metadata.source_path,
                            "payload source path",
                        ),
                        (
                            fallback_identity_path,
                            "fallback parsed filename",
                        ),
                    ),
                )

        if provenance_planes.count == 1:
            plane = provenance_planes.plane(0)
            identity = cls._identity_from_metadata(
                plane.component_metadata,
                extension=FunctionOutputExtensionAuthority.first(
                    FunctionOutputExtensionAuthority.from_metadata(
                        plane.component_metadata
                    ),
                    FunctionOutputExtensionAuthority.from_path(plane.path),
                ),
                source="single-plane payload provenance metadata",
            )
            if identity is not None:
                return cls._complete_identity_from_paths(
                    parser,
                    identity,
                    (
                        (
                            plane.path,
                            "single-plane payload provenance path",
                        ),
                        (
                            fallback_identity_path,
                            "fallback parsed filename",
                        ),
                    ),
                )
            if plane.path is not None:
                return cls._parsed_path_identity(
                    parser,
                    plane.path,
                    source="single-plane payload provenance path",
                )

        if metadata.source_path is not None:
            return cls._parsed_path_identity(
                parser,
                metadata.source_path,
                source="payload source path",
            )
        return None

    @classmethod
    def filename_identity_from_metadata(
        cls,
        parser: FilenameParser,
        metadata: ImagePayloadMetadata,
        *,
        fallback_identity_path: str | None = None,
    ) -> FunctionOutputIdentity | None:
        """Return the identity that should name a filename-addressed output."""
        identity = cls._identity_from_metadata(
            metadata.source_component_metadata,
            extension=FunctionOutputExtensionAuthority.first(
                FunctionOutputExtensionAuthority.from_metadata(
                    metadata.source_component_metadata
                ),
                FunctionOutputExtensionAuthority.from_path(metadata.source_path),
            ),
            source="payload component metadata",
        )
        if identity is not None:
            return cls._complete_identity_from_paths(
                parser,
                identity,
                (
                    (
                        metadata.source_path,
                        "payload source path",
                    ),
                    (
                        fallback_identity_path,
                        "fallback parsed filename",
                    ),
                ),
            )

        for path, source in (
            (
                metadata.source_path,
                "payload source path",
            ),
            (
                fallback_identity_path,
                "fallback parsed filename",
            ),
        ):
            if path is None:
                continue
            parsed_identity = cls._parsed_path_identity(
                parser,
                path,
                source=source,
            )
            if parsed_identity is not None:
                return parsed_identity

        provenance_planes = metadata.source_image_provenance_planes
        if provenance_planes.count:
            return cls._source_stack_plane_identity(
                parser,
                provenance_planes.plane(0),
                0,
            )
        return None

    @classmethod
    def _complete_identity_from_paths(
        cls,
        parser: FilenameParser,
        identity: FunctionOutputIdentity,
        candidates: tuple[tuple[str | None, str], ...],
    ) -> FunctionOutputIdentity:
        for path, source in candidates:
            if path is None:
                continue
            parsed_identity = cls._parsed_path_identity(
                parser,
                path,
                source=source,
            )
            if parsed_identity is None:
                continue
            component_values = dict(parsed_identity.component_values)
            component_values.update(identity.component_values)
            return FunctionOutputIdentity(
                component_values=component_values,
                extension=FunctionOutputExtensionAuthority.first(
                    identity.extension,
                    parsed_identity.extension,
                ),
                source=f"{identity.source} over {parsed_identity.source}",
            )
        return identity

    @classmethod
    def _parsed_path_identity(
        cls,
        parser: FilenameParser,
        path: str,
        *,
        source: str,
    ) -> FunctionOutputIdentity | None:
        parsed = parser.parse_filename(Path(path).name)
        if parsed is None:
            return None
        return FunctionOutputIdentity(
            component_values=FunctionOutputComponentIdentityAuthority.from_parsed(
                parsed
            ),
            extension=FunctionOutputExtensionAuthority.from_metadata(parsed),
            source=source,
        )

    @classmethod
    def _source_stack_identity_from_provenance(
        cls,
        parser: FilenameParser,
        metadata: ImagePayloadMetadata,
        source_identity_stack_axes: frozenset[str],
        *,
        fallback_identity_path: str | None,
    ) -> FunctionOutputIdentity:
        provenance_planes = metadata.source_image_provenance_planes
        plane_identities = tuple(
            cls._source_stack_plane_identity(
                parser,
                provenance_planes.plane(plane_index),
                plane_index,
            )
            for plane_index in range(provenance_planes.count)
        )
        semantic_component_values = cls._source_stack_semantic_component_values(
            plane_identities,
            source_identity_stack_axes,
        )
        extension = cls._source_stack_extension(
            plane_identities,
            source_identity_stack_axes,
            fallback_identity_path=fallback_identity_path,
        )
        filename_component_values = dict(plane_identities[0].component_values)
        return FunctionOutputIdentity(
            component_values=semantic_component_values,
            extension=extension,
            source=(
                "multi-plane payload provenance over source identity stack axes "
                f"{tuple(sorted(source_identity_stack_axes))!r}"
            ),
            filename_component_values=filename_component_values,
        )

    @classmethod
    def _source_stack_plane_identity(
        cls,
        parser: FilenameParser,
        plane: SourceImageProvenancePlane,
        plane_index: int,
    ) -> FunctionOutputIdentity:
        identity = cls._identity_from_metadata(
            plane.component_metadata,
            extension=FunctionOutputExtensionAuthority.first(
                FunctionOutputExtensionAuthority.from_metadata(
                    plane.component_metadata
                ),
                FunctionOutputExtensionAuthority.from_path(plane.path),
            ),
            source=f"source-stack provenance plane {plane_index} metadata",
        )
        if identity is not None:
            return cls._complete_identity_from_paths(
                parser,
                identity,
                (
                    (
                        plane.path,
                        f"source-stack provenance plane {plane_index} path",
                    ),
                ),
            )
        if plane.path is not None:
            parsed_identity = cls._parsed_path_identity(
                parser,
                plane.path,
                source=f"source-stack provenance plane {plane_index} path",
            )
            if parsed_identity is not None:
                return parsed_identity
        raise ValueError(
            "FunctionStep output source stack plane has no parser-resolved "
            f"identity at plane index {plane_index}."
        )

    @classmethod
    def _source_stack_semantic_component_values(
        cls,
        plane_identities: tuple[FunctionOutputIdentity, ...],
        source_identity_stack_axes: frozenset[str],
    ) -> dict[str, FunctionOutputComponentValue]:
        semantic_keys = cls._source_stack_semantic_component_keys(
            plane_identities,
            source_identity_stack_axes,
        )
        semantic_component_values: dict[str, FunctionOutputComponentValue] = {}
        for key in semantic_keys:
            values = tuple(identity.component_values.get(key) for identity in plane_identities)
            if any(key not in identity.component_values for identity in plane_identities):
                raise ValueError(
                    "FunctionStep output source stack provenance has inconsistent "
                    f"component coverage for non-stack component {key!r}."
                )
            if len(frozenset(values)) != 1:
                raise ValueError(
                    "FunctionStep output source stack provenance varies outside "
                    "declared source identity stack axes. "
                    f"Component {key!r} has values {values!r}; declared stack axes "
                    f"are {tuple(sorted(source_identity_stack_axes))!r}."
                )
            semantic_component_values[key] = values[0]
        return semantic_component_values

    @classmethod
    def _source_stack_semantic_component_keys(
        cls,
        plane_identities: tuple[FunctionOutputIdentity, ...],
        source_identity_stack_axes: frozenset[str],
    ) -> tuple[str, ...]:
        keys = (
            frozenset(
                key
                for identity in plane_identities
                for key in identity.component_values
            )
            - source_identity_stack_axes
        )
        ordered_keys = tuple(
            component.value
            for component in AllComponents
            if component.value in keys
        )
        extra_keys = tuple(sorted(keys - frozenset(ordered_keys)))
        return (*ordered_keys, *extra_keys)

    @staticmethod
    def _source_stack_extension(
        plane_identities: tuple[FunctionOutputIdentity, ...],
        source_identity_stack_axes: frozenset[str],
        *,
        fallback_identity_path: str | None,
    ) -> str | None:
        fallback_extension = FunctionOutputExtensionAuthority.from_path(
            fallback_identity_path
        )
        if fallback_extension is not None:
            return fallback_extension

        extensions = tuple(
            identity.extension
            for identity in plane_identities
            if identity.extension is not None
        )
        if not extensions:
            return None
        extension = extensions[0]
        if any(candidate != extension for candidate in extensions):
            raise ValueError(
                "FunctionStep output source stack provenance has multiple "
                f"extensions {extensions!r}; declared stack axes are "
                f"{tuple(sorted(source_identity_stack_axes))!r}."
            )
        return extension

    @classmethod
    def _input_aligned_identity(
        cls,
        request: FunctionOutputPathRequest,
    ) -> FunctionOutputIdentity:
        if request.input_path is None:
            raise ValueError(
                "FunctionStep output payload has no component identity and no "
                "input path is available for input-aligned output identity."
            )
        parsed = request.parser.parse_filename(Path(request.input_path).name)
        if parsed is None:
            raise ValueError(
                "FunctionStep output payload has no component identity and the "
                f"input path cannot be parsed: {request.input_path!r}."
            )
        return FunctionOutputIdentity(
            component_values=FunctionOutputComponentIdentityAuthority.from_parsed(
                parsed
            ),
            extension=FunctionOutputExtensionAuthority.from_metadata(parsed),
            source="input-aligned parsed filename",
        )

    @classmethod
    def _identity_from_metadata(
        cls,
        metadata: SourceComponentMetadata | None,
        *,
        extension: str | None,
        source: str,
    ) -> FunctionOutputIdentity | None:
        component_values = FunctionOutputComponentIdentityAuthority.from_source_metadata(
            metadata
        )
        if not component_values:
            return None
        return FunctionOutputIdentity(
            component_values=component_values,
            extension=extension,
            source=source,
        )
