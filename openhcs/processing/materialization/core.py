"""Materialization core (writer-based, greenfield).

Key idea: the abstraction boundary is the output *format* (writers), not per-analysis handlers.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import operator
import re
import string
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence, Sized
from dataclasses import dataclass, field, is_dataclass, replace
from functools import lru_cache, singledispatch
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import ClassVar, TYPE_CHECKING, TypeAlias

import pandas as pd
import numpy as np
from metaclass_registry import AutoRegisterMeta
from polystore.streaming.viewer_transport import (
    PathMappedViewerStreamSourceMetadata,
    ViewerStreamBackendKwargs,
)
from zmqruntime.viewer_protocol import ViewerWireField, ViewerWireValue

from openhcs.constants.constants import AllComponents, VariableComponents
from openhcs.core.artifacts import ArtifactMaterializationPayload
from openhcs.core.component_set import ComponentSet
from openhcs.core.image_file_serialization import ImageFileFormat
from openhcs.processing.materialization.constants import MaterializationFormat, WriteMode
from openhcs.processing.materialization.options import (
    CsvOptions,
    FileBundleOptions,
    FileOutputOptions,
    ImageFileOptions,
    JsonOptions,
    MaterializedFilenameIdentity,
    ROIOptions,
    SourceOptions,
    TabularExtractionOptions,
    TextOptions,
    TiffStackOptions,
)
from openhcs.core.runtime_tabular_values import (
    ColumnarRows,
)
from openhcs.core.runtime_image_values import (
    ImagePayloadMetadata,
    ImagePayloadMetadataCompositionMode,
    image_payload_data,
    image_payload_metadata,
)
from openhcs.core.runtime_array_values import runtime_array_operand
from openhcs.core.runtime_object_labels import (
    ObjectLabelValue,
)
from openhcs.core.source_image_provenance import (
    SourceComponentMetadata,
    VariableComponentAxisProjection,
    SourceImageIdentity,
    SourceImageProvenance,
    SourcePlaneIndexedMetadata,
)
from openhcs.core.source_matching import (
    source_component_metadata_value,
    source_metadata_value,
)
from openhcs.core.source_spatial_domain import SourceSpatialDomain
from openhcs.core.registry_strategies import (
    AlwaysMatchesContextMixin,
    MostDerivedContextStrategyMixin,
)
from openhcs.core.runtime_tabular_values import (
    measurement_row_mapping,
    supports_measurement_row_mapping,
)
from openhcs.core.runtime_slice_projection import (
    RuntimeProjectedPayloadItem,
    RuntimeProjectionPlaneMetadata,
    RuntimeProjectionSourceIdentityRequest,
    RuntimeProjectionSourceIdentityRequirement,
)
from openhcs.core.steps.stream_component_semantics import (
    StreamImagePayloadMetadataProjector,
    StreamViewerComponentMetadataProjector,
)
if TYPE_CHECKING:
    from polystore.filemanager import FileManager
    from openhcs.core.context.processing_context import ProcessingContext
    from openhcs.microscopes.microscope_interfaces import FilenameParser

logger = logging.getLogger(__name__)

MaterializationValue = (
    str
    | bytes
    | int
    | float
    | bool
    | list
    | tuple
    | dict
    | ColumnarRows
    | ObjectLabelValue
    | np.ndarray
    | pd.DataFrame
    | pd.Series
    | None
)
SourceComponentIdentity: TypeAlias = tuple[tuple[str, str], ...]

class BackendCallKwargs(ABC, metaclass=AutoRegisterMeta):
    """Nominal payload that knows how to build FileManager kwargs per output."""

    backend_kwargs_key_axis: ClassVar[str] = "backend_kwargs_kind"
    __registry_key__ = backend_kwargs_key_axis
    __skip_if_no_key__ = True
    stable_key_axis: ClassVar[str] = backend_kwargs_key_axis
    backend_kwargs_kind: ClassVar[str | None] = None

    @classmethod
    def registered_types(cls) -> tuple[type["BackendCallKwargs"], ...]:
        return tuple(cls.__registry__.values())

    @abstractmethod
    def to_filemanager_kwargs(
        self,
        output: "Output",
    ) -> dict:
        """Return kwargs for one FileManager save call."""

    @abstractmethod
    def filemanager_batches(
        self,
        outputs: Sequence["Output"],
    ) -> tuple[tuple[tuple["Output", ...], dict], ...]:
        """Partition outputs into exact FileManager batch calls."""

@dataclass(frozen=True, slots=True)
class RawBackendKwargs(BackendCallKwargs, Mapping[str, MaterializationValue]):
    """Raw FileManager backend kwargs at the materialization adapter edge."""

    backend_kwargs_kind = "raw"
    values: Mapping[str, MaterializationValue] = field(default_factory=dict)

    def __getitem__(self, key: str) -> MaterializationValue:
        return self.values[key]

    def __iter__(self):
        return iter(self.values)

    def __len__(self) -> int:
        return len(self.values)

    def to_filemanager_kwargs(
        self,
        output: "Output",
    ) -> dict:
        return dict(self.values)

    def filemanager_batches(
        self,
        outputs: Sequence["Output"],
    ) -> tuple[tuple[tuple["Output", ...], dict], ...]:
        return ((tuple(outputs), dict(self.values)),)

class TabularRow(dict[str, MaterializationValue]):
    """Nominal row emitted by materialization tabular extraction."""

class TabularRows(list[TabularRow]):
    """Nominal row collection emitted by materialization tabular extraction."""

def tabular_row_from_payload(
    item: MaterializationValue,
    field_names: Sequence[str] | None = None,
) -> TabularRow:
    """Project one payload item into a canonical tabular row."""
    if supports_measurement_row_mapping(item):
        row = measurement_row_mapping(item)
        if field_names:
            return TabularRow(
                {
                    field_name: row[field_name]
                    for field_name in field_names
                    if field_name in row
                }
            )
        return TabularRow(dict(row))

    return TabularRow({"value": item})

def expanded_tabular_rows(
    array_data: Sequence[MaterializationValue],
    base_row: TabularRow,
    row_columns: Mapping[str, str],
) -> TabularRows:
    """Expand an explicitly declared nested row field into tabular rows."""
    array_rows = tuple(array_data)
    if not array_rows:
        return TabularRows((base_row,))

    rows = TabularRows()
    for elem in array_rows:
        if isinstance(elem, (list, tuple)):
            cols = {
                col: elem[int(idx)]
                for idx, col in row_columns.items()
                if str(idx).isdigit() and int(idx) < len(elem)
            }
        else:
            cols = {}
        rows.append(TabularRow({**base_row, **cols}))
    return rows

@dataclass(frozen=True, slots=True)
class BackendKwargsAbsent:
    """Default backend kwargs variant for callers that do not provide overrides."""

BACKEND_KWARGS_ABSENT = BackendKwargsAbsent()
BackendKwargsValue: TypeAlias = (
    "BackendCallKwargs | Mapping[str, MaterializationValue]"
)
BackendKwargs: TypeAlias = dict[str, BackendKwargsValue]
NormalizedBackendKwargs: TypeAlias = dict[str, "BackendCallKwargs"]
BackendKwargsInput = BackendKwargs | BackendKwargsAbsent
PrimaryPathSelector = Callable[[list["Output"]], str]
CandidatePathSelector = Callable[[FileOutputOptions, str], tuple[str, ...]]

@dataclass(frozen=True)
class Output:
    path: str
    content: MaterializationValue
    metadata: ImagePayloadMetadata | None = None
    variable_components: tuple[AllComponents, ...] = ()

    @classmethod
    def from_metadata(
        cls,
        *,
        path: str,
        content: MaterializationValue,
        metadata: ImagePayloadMetadata | None,
    ) -> "Output":
        return cls(
            path=path,
            content=content,
            metadata=metadata,
        )

    @property
    def source_identity(self) -> SourceImageIdentity | None:
        if self.metadata is None:
            return None
        return self.metadata.source_provenance.scalar_source_identity

    @property
    def source_component_metadata(self) -> SourceComponentMetadata | None:
        if self.source_identity is None:
            return None
        return self.source_identity.component_metadata

    @property
    def viewer_stream_requires_source_metadata(self) -> bool:
        return not isinstance(self.content, str)

    def with_source_identity_fallback(
        self,
        fallback: SourceImageIdentity | None,
    ) -> "Output":
        if fallback is None:
            return self
        fallback_metadata = ImagePayloadMetadata(
            source_path=fallback.path,
            source_component_metadata=fallback.component_metadata,
        )
        metadata = (
            fallback_metadata
            if self.metadata is None
            else self.metadata.with_source_context_from(fallback_metadata)
        )
        return replace(
            self,
            metadata=metadata,
        )

    def with_variable_components(
        self,
        variable_components: Sequence[VariableComponents],
    ) -> "Output":
        """Carry compiler-resolved runtime plane components to backend saving."""

        return replace(
            self,
            variable_components=ComponentSet.coerce(variable_components).as_tuple(),
        )

class SourceSegmentAuthority:
    """Resolve one source selector segment against mappings or named fields."""

    @staticmethod
    def resolve(value: MaterializationValue, segment: str) -> MaterializationValue:
        if isinstance(value, dict):
            return value[segment]
        return operator.attrgetter(segment)(value)

class MaterializationSourceProjection(ABC, metaclass=AutoRegisterMeta):
    """Projection strategy for writer source options."""

    __registry_key__ = "source_type"
    __skip_if_no_key__ = True
    source_type: ClassVar[type | None] = None
    __registry__: ClassVar[dict[type, type["MaterializationSourceProjection"]]] = {}

    @classmethod
    def from_source(cls, source) -> "MaterializationSourceProjection":
        source_type = source.__class__
        try:
            projection_type = cls.__registry__[source_type]
        except KeyError as exc:
            raise TypeError(
                "Unsupported materialization source selector type: "
                f"{source_type.__name__}"
            ) from exc
        return projection_type.build(source)

    @classmethod
    @abstractmethod
    def build(cls, source) -> "MaterializationSourceProjection":
        """Build a projection from the declared source selector."""

    @abstractmethod
    def project(self, value: MaterializationValue) -> MaterializationValue:
        """Return the payload seen by materialization writers."""

class IdentityMaterializationSourceProjection(MaterializationSourceProjection):
    """No source option was declared; writers consume the payload directly."""

    source_type = type(None)

    @classmethod
    def build(cls, source: None) -> "IdentityMaterializationSourceProjection":
        del source
        return cls()

    def project(self, value: MaterializationValue) -> MaterializationValue:
        return value

@dataclass(frozen=True, slots=True)
class SourcePathSegments:
    """Validated dotted source path segments."""

    segments: tuple[str, ...]

    @classmethod
    def from_text(cls, source: str) -> "SourcePathSegments":
        segments = tuple(source.split("."))
        if "" in segments:
            raise ValueError("Materialization source path contains an empty segment.")
        return cls(segments)

@dataclass(frozen=True, slots=True)
class DottedMaterializationSourceProjection(MaterializationSourceProjection):
    """Source option declared as a dotted path into the input payload."""

    source_type = str
    source_path: SourcePathSegments

    @classmethod
    def build(cls, source: str) -> "DottedMaterializationSourceProjection":
        return cls(SourcePathSegments.from_text(source))

    def project(self, value: MaterializationValue) -> MaterializationValue:
        projected = value
        for segment in self.source_path.segments:
            projected = SourceSegmentAuthority.resolve(projected, segment)
        return projected

def _materialization_sequence(value: MaterializationValue) -> tuple[MaterializationValue, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return (value,)

def _materialization_sequence_type(value: MaterializationValue) -> type | None:
    if isinstance(value, (list, tuple)):
        return value.__class__
    return None

@singledispatch
def materialization_is_empty(value: MaterializationValue) -> bool:
    """Return whether a materialization payload carries no output data."""
    return False

@materialization_is_empty.register(type(None))
def none_materialization_is_empty(value: None) -> bool:
    return True

@materialization_is_empty.register(np.ndarray)
def numpy_materialization_is_empty(value: np.ndarray) -> bool:
    return value.size == 0

@materialization_is_empty.register(pd.DataFrame)
def dataframe_materialization_is_empty(value: pd.DataFrame) -> bool:
    return value.empty

@materialization_is_empty.register(pd.Series)
def series_materialization_is_empty(value: pd.Series) -> bool:
    return value.empty

@materialization_is_empty.register(ColumnarRows)
def columnar_rows_materialization_is_empty(value: ColumnarRows) -> bool:
    return value.row_count() == 0

@materialization_is_empty.register(list)
@materialization_is_empty.register(tuple)
@materialization_is_empty.register(dict)
@materialization_is_empty.register(str)
@materialization_is_empty.register(bytes)
def sized_materialization_is_empty(value: Sized) -> bool:
    return len(value) == 0

@dataclass(frozen=True, slots=True)
class MaterializationInputItem(RuntimeProjectedPayloadItem):
    """One materialization payload item with semantic metadata preserved."""

    @property
    def data(self) -> MaterializationValue:
        return runtime_array_operand(image_payload_data(self.value))

    @property
    def runtime_plane(self) -> RuntimeProjectionPlaneMetadata | None:
        """Return runtime display-plane identity for this materialized item."""
        return self.runtime_plane_metadata

class ROIMaterializationSourceSpatialDomain:
    """Resolve the source-spatial domain used by ROI extraction."""

    def domain_for_item(self, item: MaterializationInputItem) -> SourceSpatialDomain:
        metadata_domain = item.metadata.source_spatial_domain.with_value_name(
            "ROI materialization source image",
        )
        if metadata_domain.source_shape_yx is not None:
            return metadata_domain
        local_shape_yx = item.metadata.spatial_shape_yx(item.data)
        if local_shape_yx is None:
            return metadata_domain
        return SourceSpatialDomain(
            origin_yx=(
                metadata_domain.origin_yx
                if metadata_domain.origin_yx is not None
                else (0, 0)
            ),
            source_shape_yx=local_shape_yx,
            fill_value=metadata_domain.fill_value,
            value_name=metadata_domain.value_name,
        )

    def domain_for_target(
        self,
        target: "ROIMaterializationTarget",
    ) -> SourceSpatialDomain:
        """Return the one exact source domain represented by an ROI archive."""

        domains = tuple(self.domain_for_item(item) for item in target.items)
        if not domains:
            return SourceSpatialDomain(value_name="ROI materialization source image")
        domain_keys = {
            (domain.origin_yx, domain.source_shape_yx) for domain in domains
        }
        if len(domain_keys) != 1:
            raise ValueError(
                f"ROI archive {target.archive.path!r} cannot represent conflicting "
                f"source-spatial domains {tuple(domain_keys)!r}."
            )
        return domains[0]

@dataclass(frozen=True, slots=True)
class SourcePlaneProjectionContract:
    """Source-plane provenance declared by the payload before slice projection."""

    plane_count_sources: tuple[int, ...] = ()

    @classmethod
    def from_payloads(
        cls,
        payloads: tuple[MaterializationValue, ...],
    ) -> "SourcePlaneProjectionContract":
        return cls(
            tuple(
                plane_count_source
                for payload in payloads
                for plane_count_source in (
                    image_payload_metadata(payload)
                    .source_provenance
                    .plane_count_sources
                )
            )
        )

    @property
    def requires_source_identity(self) -> bool:
        return bool(self.plane_count_sources)

@dataclass(frozen=True, slots=True)
class SourceComponentMetadataStemAuthority:
    """Construct source stems from component metadata through the filename parser."""

    parser: "FilenameParser"

    def stem(self, metadata: SourceComponentMetadata) -> str:
        return Path(
            self.parser.construct_filename(
                extension=self.required_extension(metadata),
                **self.component_values(metadata),
            )
        ).stem

    @staticmethod
    def required_extension(metadata: SourceComponentMetadata) -> str:
        if "extension" not in metadata:
            raise ValueError(
                "Component-addressed ROI paths require source component metadata "
                "to include an extension."
            )
        return str(metadata["extension"])

    @staticmethod
    def component_values(
        metadata: Mapping[str, MaterializationValue],
    ) -> dict[str, MaterializationValue]:
        return {
            str(key): value
            for key, value in metadata.items()
            if str(key) != "extension"
        }

@dataclass(frozen=True, slots=True)
class SourceStemAuthoritySelection:
    """Available source-stem inputs for one materialization context."""

    parser: "FilenameParser | None"

    @classmethod
    def from_processing_context(
        cls,
        context: "ProcessingContext | None",
    ) -> "SourceStemAuthoritySelection":
        if context is None:
            return cls(parser=None)
        microscope_handler = context.microscope_handler
        if microscope_handler is None:
            return cls(parser=None)
        return cls(parser=microscope_handler.parser)

    def required_parser(self) -> "FilenameParser":
        if self.parser is None:
            raise ValueError("Parser-backed source-stem authority requires a parser.")
        return self.parser

class SourceStemAuthority(
    MostDerivedContextStrategyMixin[SourceStemAuthoritySelection],
    ABC,
):
    """Resolve the stream/materialization source stem for ROI path ownership."""

    authority_key_axis: ClassVar[str] = "authority_key"
    __registry_key__ = authority_key_axis
    __skip_if_no_key__ = True
    stable_key_axis: ClassVar[str] = authority_key_axis

    authority_key: ClassVar[str | None] = None
    strategy_key_attr: ClassVar[str] = authority_key_axis

    @classmethod
    def from_processing_context(
        cls,
        context: "ProcessingContext | None",
    ) -> "SourceStemAuthority":
        return cls.for_context(
            SourceStemAuthoritySelection.from_processing_context(context),
            error_subject="source-stem authority selection",
        )

    @classmethod
    def build_for_context(
        cls,
        selection: SourceStemAuthoritySelection,
    ) -> "SourceStemAuthority":
        """Build an authority after context selection."""
        del selection
        return cls()

    def required_source_stem(self, metadata: ImagePayloadMetadata) -> str:
        source_stem = SourceStemResolutionPolicy.for_context(
            metadata,
            error_subject="source-stem metadata resolution",
        ).source_stem(self, metadata)
        if source_stem is None:
            raise ValueError(
                "ROI source-stem path policy requires metadata.source_path or "
                "metadata.source_component_metadata."
            )
        return source_stem

    def source_replacement_suffix(
        self,
        path_name: str,
        metadata: ImagePayloadMetadata,
        reference_source_stem: str | None,
        reference_metadata: ImagePayloadMetadata | None,
    ) -> str | None:
        for source_stem in self.source_replacement_stems(
            path_name,
            metadata,
            reference_source_stem,
            reference_metadata,
        ):
            if path_name.startswith(source_stem):
                return path_name[len(source_stem):]
        return None

    def source_replacement_stems(
        self,
        path_name: str,
        metadata: ImagePayloadMetadata,
        reference_source_stem: str | None,
        reference_metadata: ImagePayloadMetadata | None,
    ) -> tuple[str, ...]:
        del path_name, metadata, reference_metadata
        if reference_source_stem is None:
            return ()
        return (reference_source_stem,)

    @abstractmethod
    def source_component_metadata_stem(
        self,
        metadata: SourceComponentMetadata,
    ) -> str:
        """Return the source stem for component-addressed metadata."""

class SourceStemResolutionPolicy(
    MostDerivedContextStrategyMixin[ImagePayloadMetadata],
    ABC,
):
    """Resolve source-stem inputs through MRO precedence over payload metadata."""

    @abstractmethod
    def source_stem(
        self,
        authority: SourceStemAuthority,
        metadata: ImagePayloadMetadata,
    ) -> str | None:
        """Return this source-stem case's projection."""

class MissingSourceStemResolutionPolicy(
    AlwaysMatchesContextMixin[ImagePayloadMetadata],
    SourceStemResolutionPolicy,
):
    """No addressable source identity is present."""

    strategy_key = "missing"

    def source_stem(
        self,
        authority: SourceStemAuthority,
        metadata: ImagePayloadMetadata,
    ) -> str | None:
        del authority, metadata
        return None

class ComponentMetadataSourceStemResolutionPolicy(MissingSourceStemResolutionPolicy):
    """Component metadata is addressable through the selected source-stem authority."""

    strategy_key = "component_metadata"

    def matches(self, context: ImagePayloadMetadata) -> bool:
        return (
            context.source_provenance.scalar_source_identity.component_metadata
            is not None
        )

    def source_stem(
        self,
        authority: SourceStemAuthority,
        metadata: ImagePayloadMetadata,
    ) -> str:
        component_metadata = (
            metadata.source_provenance.scalar_source_identity.component_metadata
        )
        if component_metadata is None:
            raise ValueError(
                "Component metadata source-stem policy requires scalar "
                "source_component_metadata."
            )
        return authority.source_component_metadata_stem(component_metadata)

class SourcePathSourceStemResolutionPolicy(
    ComponentMetadataSourceStemResolutionPolicy
):
    """Source path identity has precedence over component-derived filenames."""

    strategy_key = "source_path"

    def matches(self, context: ImagePayloadMetadata) -> bool:
        return bool(context.source_provenance.scalar_source_identity.path)

    def source_stem(
        self,
        authority: SourceStemAuthority,
        metadata: ImagePayloadMetadata,
    ) -> str:
        del authority
        return _cached_path_stem(
            metadata.source_provenance.scalar_source_identity.path
        )

@dataclass(frozen=True, slots=True)
class PathOnlySourceStemAuthority(
    AlwaysMatchesContextMixin[SourceStemAuthoritySelection],
    SourceStemAuthority,
):
    """Resolve ROI source stems when no filename parser exists in the context."""

    authority_key = "path_only"

    def source_component_metadata_stem(
        self,
        metadata: SourceComponentMetadata,
    ) -> str:
        del metadata
        raise ValueError(
            "Component-addressed ROI paths require a filename parser in the "
            "materialization context."
        )

@dataclass(frozen=True, slots=True)
class ParserBackedSourceStemAuthority(PathOnlySourceStemAuthority):
    """Resolve ROI source stems from paths or parser-readable component metadata."""

    authority_key = "parser_backed"
    parser: "FilenameParser"

    @classmethod
    def matches_context(
        cls,
        context: SourceStemAuthoritySelection,
    ) -> bool:
        return context.parser is not None

    @classmethod
    def build_for_context(
        cls,
        context: SourceStemAuthoritySelection,
    ) -> "ParserBackedSourceStemAuthority":
        return cls(parser=context.required_parser())

    def source_replacement_stems(
        self,
        path_name: str,
        metadata: ImagePayloadMetadata,
        reference_source_stem: str | None,
        reference_metadata: ImagePayloadMetadata | None,
    ) -> tuple[str, ...]:
        prefix_metadata = metadata
        if reference_metadata is not None:
            prefix_metadata = reference_metadata

        return (
            *super(ParserBackedSourceStemAuthority, self).source_replacement_stems(
                path_name,
                metadata,
                reference_source_stem,
                reference_metadata,
            ),
            *self.path_component_prefix_stems(path_name, prefix_metadata),
        )

    def path_component_prefix_stems(
        self,
        path_name: str,
        metadata: ImagePayloadMetadata,
    ) -> tuple[str, ...]:
        component_metadata = (
            metadata.source_provenance.scalar_source_identity.component_metadata
        )
        if component_metadata is None:
            return ()

        return tuple(
            SourceComponentMetadataStemAuthority(self.parser).stem(parsed)
            for extension in self.path_parse_extensions(metadata)
            for parsed in (self.parser.parse_filename(f"{path_name}{extension}"),)
            if parsed is not None
            and self.parsed_components_match_metadata(parsed, component_metadata)
        )

    @staticmethod
    def path_parse_extensions(metadata: ImagePayloadMetadata) -> tuple[str, ...]:
        source_identity = metadata.source_provenance.scalar_source_identity
        component_metadata = source_identity.component_metadata
        if component_metadata is not None and "extension" in component_metadata:
            return (str(component_metadata["extension"]),)

        source_path = source_identity.path
        if source_path is None:
            return ()
        suffixes = "".join(Path(source_path).suffixes)
        if not suffixes:
            return ()
        return (suffixes,)

    @staticmethod
    def parsed_components_match_metadata(
        parsed: Mapping[str, MaterializationValue],
        metadata: SourceComponentMetadata,
    ) -> bool:
        component_items = tuple(
            SourceComponentMetadataStemAuthority.component_values(parsed).items()
        )
        return all(
            key in metadata
            and ParserBackedSourceStemAuthority.component_values_match(
                parsed_value,
                metadata[key],
            )
            for key, parsed_value in component_items
        )

    @staticmethod
    def component_values_match(
        left: MaterializationValue,
        right: MaterializationValue,
    ) -> bool:
        return left == right or str(left) == str(right)

    def source_component_metadata_stem(
        self,
        metadata: SourceComponentMetadata,
    ) -> str:
        return SourceComponentMetadataStemAuthority(self.parser).stem(
            metadata,
        )

@dataclass(frozen=True, slots=True)
class ROIMaterializationArchiveIdentity:
    """Archive path plus stream-visible source identity for one ROI output."""

    path: str
    source_identity: SourceImageIdentity | None = None

    @classmethod
    def from_metadata(
        cls,
        *,
        path: str,
        metadata: ImagePayloadMetadata | None,
    ) -> "ROIMaterializationArchiveIdentity":
        return cls(
            path=path,
            source_identity=(
                None
                if metadata is None
                else metadata.source_provenance.scalar_source_identity
            ),
        )

    @property
    def component_address_identity(self) -> SourceComponentIdentity | None:
        if self.source_identity is None:
            return None
        return self.source_identity.identity[1]

    @property
    def source_stem_identity(self) -> str | None:
        if self.source_identity is None:
            return None
        if self.component_address_identity is not None:
            return None
        source_path = self.source_identity.path
        if source_path:
            return _cached_path_stem(source_path)
        return None

    @property
    def source_component_metadata(self) -> SourceComponentMetadata | None:
        if self.source_identity is None:
            return None
        return self.source_identity.component_metadata

    @property
    def source_identity_completeness(self) -> tuple[bool, bool]:
        """Rank stream identities by how much source identity they preserve."""
        return (
            self.source_identity is not None,
            self.component_address_identity is not None,
        )

    def compatible_with(
        self,
        other: "ROIMaterializationArchiveIdentity",
    ) -> bool:
        if self.path != other.path:
            return False
        if self.source_identity is None or other.source_identity is None:
            return True
        if (
            self.component_address_identity is not None
            and other.component_address_identity is not None
        ):
            return self.component_address_identity == other.component_address_identity
        if self.source_stem_identity is not None and other.source_stem_identity is not None:
            return self.source_stem_identity == other.source_stem_identity
        return True

    def merge_with(
        self,
        other: "ROIMaterializationArchiveIdentity",
    ) -> "ROIMaterializationArchiveIdentity":
        self._require_compatible(other)
        return max(
            (self, other),
            key=lambda identity: identity.source_identity_completeness,
        )

    def _require_compatible(
        self,
        other: "ROIMaterializationArchiveIdentity",
    ) -> None:
        if self.compatible_with(other):
            return
        raise ValueError(
            "ROI materialization produced conflicting stream identities for "
            f"archive path {self.path!r}."
        )

@dataclass(frozen=True, slots=True)
class ROIMaterializationTarget:
    """One ROI archive output with the label planes it owns."""

    archive: ROIMaterializationArchiveIdentity
    items: tuple[MaterializationInputItem, ...]

    def with_archive_source_identity(
        self,
        source_identity: SourceImageIdentity,
    ) -> "ROIMaterializationTarget":
        return type(self)(
            archive=ROIMaterializationArchiveIdentity(
                path=self.archive.path,
                source_identity=source_identity,
            ),
            items=self.items,
        )

    def merge_with(
        self,
        other: "ROIMaterializationTarget",
    ) -> "ROIMaterializationTarget":
        return ROIMaterializationTarget(
            archive=self.archive.merge_with(other.archive),
            items=self.items + other.items,
        )

ROIMaterializationTargets = tuple[ROIMaterializationTarget, ...]

@dataclass(frozen=True, slots=True)
class ROIPathAuthority:
    """Name addressable ROI archives from source metadata."""

    source_stem_authority: SourceStemAuthority

    @classmethod
    def from_processing_context(
        cls,
        context: "ProcessingContext | None",
    ) -> "ROIPathAuthority":
        return cls(SourceStemAuthority.from_processing_context(context))

    def path(
        self,
        *,
        paths: "PathHelper",
        options: ROIOptions,
        metadata: ImagePayloadMetadata,
        reference_source_stem: str | None,
        reference_metadata: ImagePayloadMetadata | None,
    ) -> str:
        source_stem = self.source_stem_authority.required_source_stem(metadata)
        suffix = self.source_stem_authority.source_replacement_suffix(
            paths.name,
            metadata,
            reference_source_stem,
            reference_metadata,
        )
        if suffix is not None:
            path = str(paths.parent / f"{source_stem}{suffix}{options.roi_suffix}")
        else:
            path = str(paths.parent / f"{paths.name}_{source_stem}{options.roi_suffix}")
        if not path:
            raise RuntimeError("ROI path authority produced an empty path.")
        return path

@dataclass(frozen=True, slots=True)
class ROIRequestBase(ROIPathAuthority):
    """Shared ROI materialization request facts."""

    paths: "PathHelper"
    options: ROIOptions

@dataclass(frozen=True, slots=True)
class ROIMaterializationTargetRequest(ROIRequestBase):
    """Input facts needed to derive ROI archive targets."""

    materialization_input: "MaterializationInput"
    artifact_source_identity: SourceImageIdentity | None = None
    output_key: str | None = None
    step_index: int | None = None

    @property
    def source_identity_set(self) -> "ROIPlaneSourceIdentitySet":
        return ROIPlaneSourceIdentitySet(self.materialization_input)

    @property
    def requires_projected_targeting(self) -> bool:
        return self.source_identity_set.requires_projected_targeting

@dataclass(frozen=True, slots=True)
class MaterializationInput:
    """Normalized writer input with data and metadata from one authority."""

    items: tuple[MaterializationInputItem, ...]
    unprojected_items: tuple[MaterializationInputItem, ...] = ()
    sequence_type: type | None = None
    source_plane_projection: SourcePlaneProjectionContract = field(
        default_factory=SourcePlaneProjectionContract
    )

    @classmethod
    def from_value(
        cls,
        value: MaterializationValue,
        options: SourceOptions,
    ) -> "MaterializationInput":
        projected = MaterializationSourceProjection.from_source(
            options.source
        ).project(value)
        source_payloads = _materialization_sequence(projected)
        items = tuple(
            MaterializationInputItem(
                value=item,
                source_description="materialization payload",
            )
            for item in source_payloads
        )
        return cls(
            items=items,
            unprojected_items=items,
            sequence_type=_materialization_sequence_type(projected),
        )

    @classmethod
    def from_runtime_slice_projected_value(
        cls,
        value: MaterializationValue,
        options: SourceOptions,
    ) -> "MaterializationInput":
        projected = MaterializationSourceProjection.from_source(
            options.source
        ).project(value)
        source_payloads = _materialization_sequence(projected)
        return cls(
            items=tuple(
                MaterializationInputItem(
                    value=item.value,
                    source_description=item.source_description,
                    runtime_plane_metadata=item.runtime_plane_metadata,
                )
                for source_payload in source_payloads
                for item in (
                    RuntimeProjectionSourceIdentityRequirement.OPTIONAL
                ).project_payload_items(
                    RuntimeProjectionSourceIdentityRequest(
                        value=source_payload,
                        source_description="materialization payload",
                        plane_projection=(
                            source_payload.declared_plane_projection()
                            if isinstance(source_payload, ObjectLabelValue)
                            else None
                        ),
                    )
                )
            ),
            unprojected_items=tuple(
                MaterializationInputItem(
                    value=source_payload,
                    source_description="materialization payload",
                )
                for source_payload in source_payloads
            ),
            sequence_type=_materialization_sequence_type(projected),
            source_plane_projection=SourcePlaneProjectionContract.from_payloads(
                source_payloads
            ),
        )

    @property
    def data(self) -> MaterializationValue:
        data_items = [item.data for item in self.items]
        if self.sequence_type is None:
            if not data_items:
                return None
            return data_items[0]
        return self.sequence_type(data_items)

class PathSuffixes:
    """Filename suffixes recognized by materialization path normalization."""

    ROI_ZIP: ClassVar[str] = ".roi.zip"
    ROI: ClassVar[str] = ".roi"
    PKL: ClassVar[str] = ".pkl"

class PathHelper:
    """Path helper parameterized by output options."""

    def __init__(self, base_path: str, options: FileOutputOptions):
        self.base_path = self._strip_path(base_path, options)
        self.parent = self.base_path.parent
        self.name = self.base_path.name

    @staticmethod
    def _strip_path(path: str, options: FileOutputOptions) -> Path:
        return Path(
            _cached_stripped_materialization_path(
                path,
                options.strip_pkl,
                options.strip_roi_suffix,
            )
        )

    def with_suffix(self, suffix: str) -> str:
        return str(self.parent / f"{self.name}{suffix}")

    def source_identity_base_path(self, options: FileOutputOptions) -> Path | None:
        """Return the source-identity base path when it already names this output."""
        suffix = options.primary_output_suffix
        if (
            options.filename_identity is MaterializedFilenameIdentity.SOURCE_IDENTITY
            and suffix
            and self.base_path.name.endswith(suffix)
        ):
            return self.base_path
        return None

    def primary_output_path(self, options: FileOutputOptions) -> str:
        """Return a writer output path, preserving matching source-identity paths."""
        source_path = self.source_identity_base_path(options)
        if source_path is not None:
            return str(source_path)
        return self.with_suffix(options.primary_output_suffix)

    def primary_output_basename(self, options: FileOutputOptions) -> str:
        """Return basename for derived outputs from the same writer."""
        source_path = self.source_identity_base_path(options)
        if source_path is None:
            return self.name
        return source_path.name[: -len(options.primary_output_suffix)]

    def related_output_path(
        self,
        options: FileOutputOptions,
        related_suffix: str,
    ) -> str:
        """Return a side output path sharing the primary output basename."""
        return str(
            self.parent
            / f"{self.primary_output_basename(options)}{related_suffix}"
        )

    def patterned_output_path(
        self,
        options: FileOutputOptions,
        pattern: str,
        *,
        index: int,
        preserve_single_source_output: bool = False,
    ) -> str:
        """Return an indexed output path using this writer's primary basename."""
        source_path = self.source_identity_base_path(options)
        if preserve_single_source_output and source_path is not None:
            return str(source_path)
        return str(
            self.parent
            / f"{self.primary_output_basename(options)}{pattern.format(index=index)}"
        )


@lru_cache(maxsize=65536)
def _cached_path_stem(path: str) -> str:
    """Return the filename stem for a frequently reused materialization path."""
    return Path(path).stem


@lru_cache(maxsize=65536)
def _cached_path_parent(path: str) -> str:
    """Return the parent directory for a materialization output path."""

    return str(Path(path).parent)


@lru_cache(maxsize=65536)
def _cached_stripped_materialization_path(
    path: str,
    strip_pkl: bool,
    strip_roi_suffix: bool,
) -> str:
    """Return a materialization base path stripped according to writer flags."""
    p = Path(path)
    name = p.name

    if name.endswith(PathSuffixes.ROI_ZIP):
        name = name[: -len(PathSuffixes.ROI_ZIP)]

    if strip_pkl and name.endswith(PathSuffixes.PKL):
        name = name[: -len(PathSuffixes.PKL)]
    if strip_roi_suffix and name.endswith(PathSuffixes.ROI):
        name = name[: -len(PathSuffixes.ROI)]

    return str(p.with_name(name))

class MaterializationCandidatePathAuthority:
    """Project writer options to the output paths a backend may see."""

    @staticmethod
    def single_output(options: FileOutputOptions, base_path: str) -> tuple[str, ...]:
        paths = PathHelper(base_path, options)
        return (paths.primary_output_path(options),)

class ROICandidatePathAuthority:
    """Candidate paths emitted by ROI materialization."""

    @staticmethod
    def paths(options: FileOutputOptions, base_path: str) -> tuple[str, ...]:
        if not isinstance(options, ROIOptions):
            raise TypeError(
                "ROICandidatePathAuthority requires ROIOptions, "
                f"got {options.__class__.__name__}."
            )
        paths = PathHelper(base_path, options)
        return (
            paths.primary_output_path(options),
            paths.related_output_path(options, options.summary_suffix),
        )

class TiffStackCandidatePathAuthority:
    """Candidate paths emitted by TIFF stack materialization."""

    @staticmethod
    def paths(options: FileOutputOptions, base_path: str) -> tuple[str, ...]:
        if not isinstance(options, TiffStackOptions):
            raise TypeError(
                "TiffStackCandidatePathAuthority requires TiffStackOptions, "
                f"got {options.__class__.__name__}."
            )
        paths = PathHelper(base_path, options)
        return (
            paths.patterned_output_path(
                options,
                options.slice_pattern,
                index=0,
                preserve_single_source_output=True,
            ),
            paths.related_output_path(options, options.summary_suffix),
        )


def _tiff_stack_slice_path(
    paths: PathHelper,
    options: TiffStackOptions,
    *,
    index: int,
    slice_count: int,
) -> str:
    return paths.patterned_output_path(
        options,
        options.slice_pattern,
        index=index,
        preserve_single_source_output=slice_count == 1,
    )


def _tiff_stack_summary_path(paths: PathHelper, options: TiffStackOptions) -> str:
    return paths.related_output_path(options, options.summary_suffix)


class WriteModePathPolicy(ABC, metaclass=AutoRegisterMeta):
    """Apply existing-path policy for one write mode."""

    __registry_key__ = "write_mode"
    __skip_if_no_key__ = True

    write_mode: ClassVar[WriteMode | None] = None

    @abstractmethod
    def prepare_existing_path(self, saver: "BackendSaver", backend: str, path: str) -> None:
        """Prepare an already-existing backend path before saving."""

class OverwritePathPolicy(WriteModePathPolicy):
    write_mode = WriteMode.OVERWRITE

    def prepare_existing_path(self, saver: "BackendSaver", backend: str, path: str) -> None:
        saver.filemanager.delete(path, backend)

class ErrorPathPolicy(WriteModePathPolicy):
    write_mode = WriteMode.ERROR

    def prepare_existing_path(self, saver: "BackendSaver", backend: str, path: str) -> None:
        raise FileExistsError(f"Refusing to overwrite existing path: {path} ({backend})")

@dataclass(frozen=True, slots=True)
class ViewerStreamBackendCallKwargs(BackendCallKwargs):
    """Viewer stream kwargs with per-output component metadata applied in-band."""

    backend_kwargs_kind = "viewer_stream"
    values: ViewerStreamBackendKwargs

    def to_filemanager_kwargs(
        self,
        output: Output,
    ) -> dict:
        output_fields = self._output_fields(output)
        if output_fields is None:
            return self.values.to_kwargs()
        component_metadata, item_fields = output_fields
        return self.values.with_single_item_source(
            component_metadata,
            item_fields,
        ).to_kwargs()

    def filemanager_batches(
        self,
        outputs: Sequence[Output],
    ) -> tuple[tuple[tuple[Output, ...], dict], ...]:
        grouped: list[
            tuple[
                dict[str, ViewerWireValue],
                list[Output],
                list[dict[str, MaterializationValue]],
            ]
        ] = []
        unprojected_outputs: list[Output] = []
        for output in outputs:
            output_fields = self._output_fields(output)
            if output_fields is None:
                unprojected_outputs.append(output)
                continue
            component_metadata, item_fields = output_fields
            group = next(
                (
                    candidate
                    for candidate in grouped
                    if candidate[0] == item_fields
                ),
                None,
            )
            if group is None:
                group = (item_fields, [], [])
                grouped.append(group)
            group[1].append(output)
            group[2].append(component_metadata)

        batches: list[tuple[tuple[Output, ...], dict]] = []
        if unprojected_outputs:
            batches.append(
                (tuple(unprojected_outputs), self.values.to_kwargs())
            )
        for item_fields, batch_outputs, component_metadata in grouped:
            metadata_by_path = dict(
                zip(
                    (output.path for output in batch_outputs),
                    component_metadata,
                    strict=True,
                )
            )
            if len(metadata_by_path) != len(batch_outputs):
                raise ValueError(
                    "Viewer materialization batch requires unique output paths."
                )
            stream_request = self.values.with_item_fields(
                item_fields
            ).stream_request
            stream_request = replace(
                stream_request,
                source=replace(
                    stream_request.source,
                    metadata=PathMappedViewerStreamSourceMetadata(
                        metadata_by_path=metadata_by_path
                    ),
                ),
            )
            batches.append(
                (
                    tuple(batch_outputs),
                    ViewerStreamBackendKwargs(stream_request).to_kwargs(),
                )
            )
        return tuple(batches)

    def _output_fields(
        self,
        output: Output,
    ) -> tuple[dict[str, MaterializationValue], dict[str, ViewerWireValue]] | None:
        source_identity = output.source_identity
        if source_identity is not None:
            source_identity = source_identity.with_parsed_path_components(
                self.values.stream_request.source.identity.microscope_handler.parser
            )
        component_metadata = (
            source_identity.component_metadata
            if source_identity is not None
            else None
        )
        if component_metadata is None:
            if not output.viewer_stream_requires_source_metadata:
                return None
            raise ValueError(
                "Viewer stream materialization requires output metadata with "
                "source_component_metadata."
            )
        item_fields = (
            StreamImagePayloadMetadataProjector.item_fields_for_plane_components(
                output.metadata,
                output.variable_components,
            )
        )
        plane_component_values = item_fields.get(
            ViewerWireField.PLANE_COMPONENT_VALUES.value,
            {},
        )
        if not isinstance(plane_component_values, Mapping):
            raise TypeError(
                "Viewer stream plane_component_values must be a mapping."
            )
        display_semantics = self.values.stream_request.display_semantics
        projected_metadata = StreamViewerComponentMetadataProjector(
            tuple(
                component
                for component in display_semantics.component_order
                if component not in plane_component_values
            )
        ).project_required(index=0, metadata=component_metadata)
        return projected_metadata, item_fields

EMPTY_BACKEND_CALL_KWARGS = RawBackendKwargs()

class BackendSaver:
    """Centralized multi-backend saving."""

    def __init__(
        self,
        backends: list[str],
        filemanager: "FileManager",
        backend_kwargs: NormalizedBackendKwargs,
        *,
        write_mode: WriteMode,
    ):
        self.backends = backends
        self.filemanager = filemanager
        self.backend_kwargs = backend_kwargs
        self.write_mode = write_mode

    def save_all(
        self,
        outputs: Sequence[Output],
    ) -> None:
        for backend in self.backends:
            backend_instance = self.filemanager._get_backend(backend)
            supported_outputs = tuple(
                output
                for output in outputs
                if backend_instance.supports_file_path(output.path)
            )
            skipped_outputs = tuple(
                output
                for output in outputs
                if not backend_instance.supports_file_path(output.path)
            )
            for output in skipped_outputs:
                logger.info(
                    "Skipping backend %s for unsupported materialization path %s",
                    backend,
                    output.path,
                )
            for output in supported_outputs:
                self._prepare_path(backend, backend_instance, output.path)
            if not supported_outputs:
                continue
            if backend in self.backend_kwargs:
                kwargs_payload = self.backend_kwargs[backend]
            else:
                kwargs_payload = EMPTY_BACKEND_CALL_KWARGS
            for batch_outputs, kwargs in kwargs_payload.filemanager_batches(
                supported_outputs
            ):
                self.filemanager.save_batch(
                    [output.content for output in batch_outputs],
                    [output.path for output in batch_outputs],
                    backend,
                    **kwargs,
                )

    def _prepare_path(self, backend: str, backend_instance, path: str) -> None:
        if not backend_instance.requires_filesystem_validation:
            return

        self.filemanager.ensure_directory(_cached_path_parent(path), backend)

        if not self.filemanager.exists(path, backend):
            return

        WriteModePathPolicy.__registry__[self.write_mode]().prepare_existing_path(
            self,
            backend,
            path,
        )

@dataclass(frozen=True)
class MaterializationContext:
    base_path: str
    backends: list[str]
    backend_kwargs: NormalizedBackendKwargs
    filemanager: "FileManager"
    extra_inputs: dict
    context: "ProcessingContext | None" = None
    artifact_source_identity: SourceImageIdentity | None = None
    variable_components: Sequence[VariableComponents] = field(default_factory=tuple)
    write_mode: WriteMode = WriteMode.OVERWRITE
    source_paths: tuple[str, ...] = ()
    output_key: str | None = None
    step_index: int | None = None

    def paths(self, options: FileOutputOptions) -> PathHelper:
        return PathHelper(self.base_path, options)

    @property
    def saver(self) -> BackendSaver:
        return BackendSaver(
            self.backends,
            self.filemanager,
            self.backend_kwargs,
            write_mode=self.write_mode,
        )

@dataclass(frozen=True)
class WriterSpec:
    format: MaterializationFormat
    options_type: type
    write: "WriterFunction"
    primary_path: PrimaryPathSelector
    candidate_paths: CandidatePathSelector

WriterFunction = Callable
_WRITERS_BY_OPTIONS: dict[type, WriterSpec] = {}

class PrimaryPathAuthority:
    """Resolve the primary materialized path for writer outputs."""

    @staticmethod
    def first_output_path(outs: list[Output]) -> str:
        if outs:
            return outs[0].path
        return ""

class BackendKwargsAuthority:
    """Normalize absent/present backend kwargs variants."""

    @staticmethod
    def normalize(backend_kwargs: BackendKwargsInput) -> NormalizedBackendKwargs:
        if isinstance(backend_kwargs, BackendKwargsAbsent):
            return {}
        return {
            backend: BackendKwargsAuthority.normalize_value(value)
            for backend, value in backend_kwargs.items()
        }

    @staticmethod
    def normalize_value(value: BackendKwargsValue) -> BackendCallKwargs:
        if isinstance(value, BackendCallKwargs):
            return value
        return RawBackendKwargs(dict(value))

def writer_for(
    options_type: type,
    fmt: MaterializationFormat,
    *,
    primary_path: PrimaryPathSelector | None = None,
    candidate_paths: CandidatePathSelector | None = None,
):
    """Register a writer for a given options type.

    This is intentionally metaprogramming-friendly: adding a new format is
    defining one options dataclass and one function.
    """

    def decorator(fn: WriterFunction):
        if options_type in _WRITERS_BY_OPTIONS:
            raise ValueError(f"Writer already registered for options type {options_type.__name__}")
        selected_primary_path = primary_path
        if selected_primary_path is None:
            selected_primary_path = PrimaryPathAuthority.first_output_path
        selected_candidate_paths = candidate_paths
        if selected_candidate_paths is None:
            selected_candidate_paths = MaterializationCandidatePathAuthority.single_output
        _WRITERS_BY_OPTIONS[options_type] = WriterSpec(
            format=fmt,
            options_type=options_type,
            write=fn,
            primary_path=selected_primary_path,
            candidate_paths=selected_candidate_paths,
        )
        return fn

    return decorator

@singledispatch
def tabular_rows_from_payload(
    data: MaterializationValue,
    options: TabularExtractionOptions,
) -> TabularRows | None:
    del data, options
    return None

@tabular_rows_from_payload.register(pd.DataFrame)
def dataframe_tabular_rows(
    data: pd.DataFrame,
    options: TabularExtractionOptions,
) -> TabularRows:
    del options
    return TabularRows(TabularRow(dict(row)) for row in data.to_dict(orient="records"))

@tabular_rows_from_payload.register(pd.Series)
def series_tabular_rows(
    data: pd.Series,
    options: TabularExtractionOptions,
) -> TabularRows:
    del options
    return TabularRows((TabularRow(dict(data.to_dict())),))

@tabular_rows_from_payload.register(ColumnarRows)
def columnar_rows_tabular_rows(
    data: ColumnarRows,
    options: TabularExtractionOptions,
) -> TabularRows:
    del options
    return TabularRows(
        TabularRow(dict(row))
        for row in data.row_mappings()
    )


def _mapping_column_length(value: MaterializationValue) -> int | None:
    """Return row count for a mapping value that behaves like a table column."""
    if isinstance(value, Mapping):
        return None
    if isinstance(value, (str, bytes, bytearray)):
        return None
    shape = getattr(value, "shape", None)
    if shape:
        return int(shape[0])
    if isinstance(value, Sequence):
        return len(value)
    try:
        return len(value)  # type: ignore[arg-type]
    except TypeError:
        return None


def _mapping_column_value(
    value: MaterializationValue,
    row_index: int,
) -> MaterializationValue:
    """Return one row value from a column-like mapping value."""
    length = _mapping_column_length(value)
    if length is None:
        return value
    if row_index >= length:
        return value
    return value[row_index]  # type: ignore[index]


@tabular_rows_from_payload.register(Mapping)
def mapping_tabular_rows(
    data: Mapping,
    options: TabularExtractionOptions,
) -> TabularRows:
    """Project mapping-of-columns payloads into row mappings."""
    columns = {str(column): value for column, value in data.items()}
    field_names = tuple(options.fields) if options.fields else tuple(columns)
    column_lengths = tuple(
        length
        for field_name in field_names
        if field_name in columns
        for length in (_mapping_column_length(columns[field_name]),)
        if length is not None
    )
    row_count = max(column_lengths) if column_lengths else 1
    rows = TabularRows()
    for row_index in range(row_count):
        row = TabularRow(
            {
                field_name: _mapping_column_value(columns[field_name], row_index)
                for field_name in field_names
                if field_name in columns
            }
        )
        if TabularRowsAuthority.should_add_slice_index(row, options.fields):
            row[TabularRowsAuthority.SLICE_INDEX_FIELD] = row_index
        rows.append(row)
    return rows


class TabularRowsAuthority:
    """Build canonical tabular rows for CSV and JSON materialization."""

    SLICE_INDEX_FIELD: ClassVar[str] = "slice_index"

    @staticmethod
    def requested(options: TabularExtractionOptions) -> bool:
        return bool(
            options.fields
            or options.row_field
            or options.row_unpacker
            or options.row_columns
        )

    @classmethod
    def build(
        cls,
        data: MaterializationValue,
        options: TabularExtractionOptions,
    ) -> TabularRows:
        direct_rows = tabular_rows_from_payload(data, options)
        if direct_rows is not None:
            return direct_rows

        rows = TabularRows()
        for idx, item in enumerate(_materialization_sequence(data)):
            base_row = cls.base_row(item, idx, options)

            if options.row_unpacker:
                for exp_row in options.row_unpacker(item):
                    rows.append(TabularRow({**base_row, **exp_row}))
                continue

            if options.row_field:
                array_data = FieldValueAuthority.value(item, options.row_field)
                rows.extend(
                    expanded_tabular_rows(array_data, base_row, options.row_columns)
                )
                continue

            rows.append(base_row)
        return rows

    @classmethod
    def base_row(
        cls,
        item: MaterializationValue,
        index: int,
        options: TabularExtractionOptions,
    ) -> TabularRow:
        field_names = options.fields
        base_row = tabular_row_from_payload(item, field_names)
        if cls.should_add_slice_index(base_row, field_names):
            base_row[cls.SLICE_INDEX_FIELD] = index
        return base_row

    @classmethod
    def should_add_slice_index(
        cls,
        base_row: Mapping,
        field_names: Sequence[str] | None,
    ) -> bool:
        return cls.SLICE_INDEX_FIELD not in base_row and (
            not field_names or cls.SLICE_INDEX_FIELD in field_names
        )

class FieldValueAuthority:
    """Resolve declared row fields through the shared materialization extractor."""

    @staticmethod
    def value(item: MaterializationValue, field_name: str) -> MaterializationValue:
        field_values = tabular_row_from_payload(item, (field_name,))
        if field_name in field_values:
            return field_values[field_name]
        return SourceSegmentAuthority.resolve(item, field_name)

def _render_csv(data: MaterializationValue, options: CsvOptions) -> str:
    if isinstance(data, pd.DataFrame):
        return data.to_csv(index=False)

    if direct_rows := _direct_csv_mapping_rows(data, options):
        rows, fieldnames = direct_rows
        return _render_csv_rows(rows, fieldnames)
    if direct_object_rows := _direct_csv_object_rows(data, options):
        rows, fieldnames = direct_object_rows
        return _render_csv_object_rows(rows, fieldnames)

    rows = TabularRowsAuthority.build(data, options)
    if not rows and options.fields:
        return _render_csv_rows((), tuple(options.fields))
    if not rows:
        return pd.DataFrame(rows).to_csv(index=False)
    return _render_csv_rows(
        rows,
        CsvFieldnamesAuthority.fieldnames(rows, options.fields),
    )

def _direct_csv_mapping_rows(
    data: MaterializationValue,
    options: CsvOptions,
) -> tuple[Sequence[Mapping], tuple[str, ...]] | None:
    """Return existing mapping rows when generic extraction would be a no-op."""
    if options.row_unpacker is not None or options.row_field is not None:
        return None
    if not isinstance(data, (list, tuple)):
        return None
    if not data:
        if options.fields:
            return data, tuple(options.fields)
        return None
    first_row = data[0]
    if not isinstance(first_row, Mapping):
        return None

    fieldnames = CsvFieldnamesAuthority.fieldnames(data, options.fields)
    if (
        TabularRowsAuthority.SLICE_INDEX_FIELD in fieldnames
        and TabularRowsAuthority.SLICE_INDEX_FIELD not in first_row
    ):
        return None
    return data, fieldnames

def _direct_csv_object_rows(
    data: MaterializationValue,
    options: CsvOptions,
) -> tuple[Sequence, tuple[str, ...]] | None:
    """Return object rows when declared fields let us avoid dict materialization."""
    if (
        options.row_unpacker is not None
        or options.row_field is not None
        or not options.fields
        or not isinstance(data, (list, tuple))
    ):
        return None
    if not data:
        return data, tuple(options.fields)
    first_row = data[0]
    if isinstance(first_row, Mapping):
        return None
    if not CsvObjectRowAuthority.accepts(first_row):
        return None
    return data, tuple(options.fields)

def _render_csv_rows(
    rows: Sequence[Mapping],
    fieldnames: Sequence[str],
) -> str:
    output = io.StringIO()
    ordered_fieldnames = tuple(fieldnames)
    writer = csv.writer(output)
    writer.writerow(ordered_fieldnames)
    writer.writerows(
        tuple(row.get(fieldname) for fieldname in ordered_fieldnames)
        for row in rows
    )
    return output.getvalue()

def _render_csv_object_rows(
    rows: Sequence,
    fieldnames: Sequence[str],
) -> str:
    output = io.StringIO()
    ordered_fieldnames = tuple(fieldnames)
    writer = csv.writer(output)
    writer.writerow(ordered_fieldnames)
    writer.writerows(
        CsvObjectRowAuthority.values(row, ordered_fieldnames)
        for row in rows
    )
    return output.getvalue()

class CsvObjectRowAuthority:
    """Read declared fields from object rows without structural probing at write time."""

    @staticmethod
    def accepts(row: MaterializationValue) -> bool:
        return is_dataclass(row)

    @staticmethod
    def values(row: MaterializationValue, fieldnames: Sequence[str]) -> tuple:
        fields = tabular_row_from_payload(row, fieldnames)
        return tuple(
            CsvObjectRowAuthority.field_value(fields, fieldname)
            for fieldname in fieldnames
        )

    @staticmethod
    def field_value(fields: Mapping, fieldname: str) -> MaterializationValue:
        if fieldname in fields:
            return fields[fieldname]
        return None

class CsvFieldnamesAuthority:
    """Own CSV field order for direct and extracted row materialization."""

    @staticmethod
    def fieldnames(
        rows: Sequence[Mapping],
        declared_fields: Sequence[str] | None,
    ) -> tuple[str, ...]:
        if declared_fields is not None:
            return tuple(declared_fields)
        fieldnames: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for fieldname in row:
                if fieldname in seen:
                    continue
                seen.add(fieldname)
                fieldnames.append(fieldname)
        return tuple(fieldnames)

def _render_json(data: MaterializationValue, options: JsonOptions) -> str:
    payload: MaterializationValue
    if TabularRowsAuthority.requested(options):
        payload = TabularRowsAuthority.build(data, options)
    else:
        payload = data

    if options.wrap_list and isinstance(payload, list):
        payload = {"total_items": len(payload), "results": payload}

    return json.dumps(JsonPayloadAuthority.jsonable(payload), indent=options.indent)

class JsonPayloadAuthority:
    """Convert materialized payloads into JSON-native values before encoding."""

    @classmethod
    def jsonable(cls, value: MaterializationValue) -> MaterializationValue:
        return json_payload_value(value)

@singledispatch
def json_payload_value(value: MaterializationValue) -> MaterializationValue:
    if supports_measurement_row_mapping(value):
        return {
            key: JsonPayloadAuthority.jsonable(item)
            for key, item in measurement_row_mapping(value).items()
        }
    return value

@json_payload_value.register(dict)
def json_payload_mapping(value: dict) -> dict:
    return {key: JsonPayloadAuthority.jsonable(item) for key, item in value.items()}

@json_payload_value.register(list)
def json_payload_list(value: list) -> list:
    return [JsonPayloadAuthority.jsonable(item) for item in value]

@json_payload_value.register(tuple)
def json_payload_tuple(value: tuple) -> tuple:
    return tuple(JsonPayloadAuthority.jsonable(item) for item in value)

@json_payload_value.register(pd.DataFrame)
def json_payload_dataframe(value: pd.DataFrame) -> TabularRows:
    return TabularRows(TabularRow(dict(row)) for row in value.to_dict(orient="records"))

@json_payload_value.register(pd.Series)
def json_payload_series(value: pd.Series) -> dict:
    return value.to_dict()

@json_payload_value.register(np.generic)
def json_payload_numpy_scalar(value: np.generic) -> MaterializationValue:
    return value.item()

@json_payload_value.register(np.ndarray)
def json_payload_numpy_array(value: np.ndarray) -> list:
    return value.tolist()

class SingleFileWriterAuthority:
    """Create writers for formats that produce one file from one payload."""

    @staticmethod
    def writer(
        render: Callable,
        *,
        validate_payload: Callable | None = None,
    ) -> Callable:
        def write(data: MaterializationValue, options, ctx: MaterializationContext) -> list[Output]:
            materialization_input = MaterializationInput.from_value(data, options)
            payload = materialization_input.data
            if validate_payload is not None:
                validate_payload(payload, options)
            return [
                Output(
                    path=ctx.paths(options).primary_output_path(options),
                    content=render(payload, options),
                )
            ]

        return write

def register_single_file_writer(
    options_type: type,
    fmt: MaterializationFormat,
    *,
    render: Callable,
    validate_payload: Callable | None = None,
    primary_path: PrimaryPathSelector | None = None,
) -> None:
    writer_for(options_type, fmt, primary_path=primary_path)(
        SingleFileWriterAuthority.writer(render, validate_payload=validate_payload)
    )

register_single_file_writer(CsvOptions, MaterializationFormat.CSV, render=_render_csv)
register_single_file_writer(JsonOptions, MaterializationFormat.JSON, render=_render_json)

class TextPayloadAuthority:
    """Validate text materialization payloads."""

    @staticmethod
    def validate(payload: MaterializationValue, options: TextOptions) -> None:
        del options
        if not isinstance(payload, str):
            raise TypeError(
                f"TextOptions expects a str payload, got {payload.__class__.__name__}"
            )

register_single_file_writer(
    TextOptions,
    MaterializationFormat.TEXT,
    render=lambda payload, _options: payload,
    validate_payload=TextPayloadAuthority.validate,
)


_NAMED_PATH_BACKREFERENCE = re.compile(r"\\g<(?P<name>[^>]+)>")
_PATH_TEMPLATE_FORMATTER = string.Formatter()


def _normalized_relative_materialization_path(value: object) -> PurePosixPath:
    """Validate and normalize one platform-neutral relative output path."""
    if not isinstance(value, str):
        raise TypeError("Materialization relative path must be a string.")
    if not value:
        raise ValueError("Materialization relative path cannot be empty.")
    if PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute():
        raise ValueError(f"Materialization path must be relative: {value!r}.")
    normalized = PurePosixPath(value.replace("\\", "/"))
    if ".." in normalized.parts:
        raise ValueError(
            f"Materialization relative path cannot traverse parents: {value!r}."
        )
    if str(normalized) in {"", "."}:
        raise ValueError("Materialization relative path cannot be empty.")
    return normalized


def _relative_materialization_output_path(
    base_path: str,
    relative_path: PurePosixPath,
) -> str:
    return str(Path(base_path).parent.joinpath(*relative_path.parts))


def _image_template_candidate_path(template: str) -> PurePosixPath:
    candidate = _NAMED_PATH_BACKREFERENCE.sub(
        lambda match: match.group("name"),
        template,
    )
    candidate = _format_image_sequence_index(candidate, 1)
    return _normalized_relative_materialization_path(candidate)


def _image_template_uses_source_metadata(template: str | None) -> bool:
    """Return whether one image path template requires per-item source metadata."""

    return template is not None and _NAMED_PATH_BACKREFERENCE.search(template) is not None


def _image_template_uses_sequence_index(template: str | None) -> bool:
    """Return whether one image path template declares per-item numbering."""

    if template is None:
        return False
    return any(
        field_name == "index"
        for _literal, field_name, _format_spec, _conversion in (
            _PATH_TEMPLATE_FORMATTER.parse(template)
        )
        if field_name is not None
    )


def _format_image_sequence_index(template: str, sequence_index: int) -> str:
    """Expand the sole generic per-item field accepted by image templates."""

    field_names = tuple(
        field_name
        for _literal, field_name, _format_spec, _conversion in (
            _PATH_TEMPLATE_FORMATTER.parse(template)
        )
        if field_name is not None
    )
    unsupported = tuple(
        field_name for field_name in field_names if field_name != "index"
    )
    if unsupported:
        raise ValueError(
            "Image materialization path template contains unsupported format "
            f"fields: {unsupported!r}."
        )
    return template.format(index=sequence_index)


def _image_relative_output_path(
    data: MaterializationValue,
    options: ImageFileOptions,
    context: MaterializationContext,
    *,
    sequence_index: int,
) -> str:
    if options.relative_path_template is None:
        return context.paths(options).primary_output_path(options)

    metadata = image_payload_metadata(data)
    source_identity = metadata.source_provenance.scalar_source_identity
    component_metadata = source_identity.component_metadata

    def replace_backreference(match: re.Match[str]) -> str:
        if component_metadata is None:
            raise ValueError(
                "Image materialization path template requires source component "
                f"metadata for {match.group('name')!r}."
            )
        value = source_metadata_value(component_metadata, match.group("name"))
        if value is None:
            raise ValueError(
                "Image materialization path template references undeclared source "
                f"metadata {match.group('name')!r}; available fields are "
                f"{tuple(component_metadata)!r}."
            )
        return value

    expanded = _NAMED_PATH_BACKREFERENCE.sub(
        replace_backreference,
        options.relative_path_template,
    )
    expanded = _format_image_sequence_index(expanded, sequence_index)
    return _relative_materialization_output_path(
        context.base_path,
        _normalized_relative_materialization_path(expanded),
    )


class ImageFileCandidatePathAuthority:
    """Project image-file options to their compile-time backend-routing path."""

    @staticmethod
    def paths(options: FileOutputOptions, base_path: str) -> tuple[str, ...]:
        if not isinstance(options, ImageFileOptions):
            raise TypeError(
                "ImageFileCandidatePathAuthority requires ImageFileOptions."
            )
        if options.relative_path_template is None:
            candidate = PathHelper(base_path, options).primary_output_path(options)
        else:
            candidate = _relative_materialization_output_path(
                base_path,
                _image_template_candidate_path(options.relative_path_template),
            )
        ImageFileFormat.require_path(candidate)
        return (candidate,)


@writer_for(
    ImageFileOptions,
    MaterializationFormat.IMAGE_FILE,
    candidate_paths=ImageFileCandidatePathAuthority.paths,
)
def write_image_file(
    data: MaterializationValue,
    options: ImageFileOptions,
    context: MaterializationContext,
) -> list[Output]:
    """Write image payloads through the nominal image serialization format."""
    materialization_input = MaterializationInput.from_runtime_slice_projected_value(
        data,
        options,
    )
    metadata_projection = RuntimePlaneStackAxisMetadataProjection(
        context.variable_components,
        context.artifact_source_identity,
    )
    projected_items = tuple(
        replace(
            item,
            value=metadata_projection.metadata_for_item(item).attach_to(item.value),
        )
        for item in materialization_input.items
    )
    if (
        len(projected_items) > 1
        and not materialization_emits_variable_component_planes(options, data)
    ):
        projected_items = (
            MaterializationInputItem(
                value=data,
                source_description="materialization payload",
            ),
        )

    paths = context.paths(options)
    source_stem_authority = SourceStemAuthority.from_processing_context(
        context.context
    )
    preserves_planned_path = (
        options.filename_identity is MaterializedFilenameIdentity.ARTIFACT_NAME
        or paths.source_identity_base_path(options) is not None
    )
    outputs = tuple(
        (
            (
                _image_relative_output_path(
                    item.value,
                    options,
                    context,
                    sequence_index=sequence_index,
                )
                if options.relative_path_template is not None
                else paths.primary_output_path(options)
                if preserves_planned_path
                else str(
                    paths.parent
                    / (
                        source_stem_authority.required_source_stem(item.metadata)
                        + options.primary_output_suffix
                    )
                )
            ),
            item,
        )
        for sequence_index, item in enumerate(projected_items, start=1)
    )
    output_paths = tuple(path for path, _item in outputs)
    if len(set(output_paths)) != len(output_paths):
        raise ValueError(
            "Image materialization source identities produced duplicate paths: "
            f"{output_paths!r}."
        )
    return [
        Output.from_metadata(
            path=path,
            content=ImageFileFormat.require_path(path).prepare(
                item.value
            ),
            metadata=item.metadata,
        )
        for path, item in outputs
    ]


def _file_bundle_outputs(
    data: MaterializationValue,
    context: MaterializationContext,
) -> list[Output]:
    if type(data) is not dict:
        raise TypeError("File bundle payload must be a built-in dict.")

    outputs: list[Output] = []
    normalized_paths: set[PurePosixPath] = set()
    for relative_value, content in data.items():
        relative_path = _normalized_relative_materialization_path(relative_value)
        if relative_path in normalized_paths:
            raise ValueError(
                f"File bundle contains duplicate normalized path {relative_path!s}."
            )
        normalized_paths.add(relative_path)
        if isinstance(content, str):
            encoded = content.encode("utf-8")
        elif isinstance(content, bytes):
            encoded = content
        else:
            raise TypeError("File bundle values must be str or bytes.")
        outputs.append(
            Output(
                path=_relative_materialization_output_path(
                    context.base_path,
                    relative_path,
                ),
                content=encoded,
            )
        )
    return outputs


@writer_for(FileBundleOptions, MaterializationFormat.FILE_BUNDLE)
def write_file_bundle(
    data: MaterializationValue,
    options: FileBundleOptions,
    context: MaterializationContext,
) -> list[Output]:
    """Write a validated insertion-ordered bundle beneath the artifact directory."""
    del options
    return _file_bundle_outputs(data, context)

class ROIPrimaryPathAuthority:
    """Resolve the user-visible primary path for ROI materialization outputs."""

    @staticmethod
    def primary_path(outs: list[Output]) -> str:
        for out in outs:
            if out.path.endswith(PathSuffixes.ROI_ZIP):
                return out.path
        return PrimaryPathAuthority.first_output_path(outs)

class ROIMaterializationTargetPolicy(
    MostDerivedContextStrategyMixin[ROIMaterializationTargetRequest],
    ABC,
):
    """Registered target selection policy for ROI materialization."""

    @abstractmethod
    def targets(
        self,
        request: ROIMaterializationTargetRequest,
    ) -> ROIMaterializationTargets:
        """Return ROI archive targets for the request."""

@dataclass(frozen=True, slots=True)
class ROIPlaneSourceIdentitySet:
    """Source identities available for a runtime-slice-projected ROI plane set."""

    materialization_input: MaterializationInput

    @property
    def items(self) -> tuple[MaterializationInputItem, ...]:
        return self.materialization_input.items

    @property
    def identities(self) -> tuple:
        return tuple(
            item.metadata.source_provenance.scalar_source_identity.identity
            for item in self.items
        )

    @property
    def missing_indices(self) -> tuple[int, ...]:
        return tuple(
            index
            for index, identity in enumerate(self.identities)
            if identity == (None, None)
        )

    @property
    def has_addressable_identity(self) -> bool:
        return any(identity != (None, None) for identity in self.identities)

    @property
    def has_single_component_address(self) -> bool:
        if len(self.items) != 1:
            return False
        return (
            self.items[0].metadata.source_provenance
            .scalar_source_identity.component_metadata
            is not None
        )

    @property
    def requires_source_identity(self) -> bool:
        return self.materialization_input.source_plane_projection.requires_source_identity

    @property
    def requires_projected_targeting(self) -> bool:
        return len(self.items) > 1 and (
            self.has_addressable_identity
            or self.requires_source_identity
        )

    def validate_projected_targeting(self) -> None:
        if not self.missing_indices:
            return
        if self.has_addressable_identity or self.requires_source_identity:
            raise ValueError(
                "Projected ROI planes require per-plane source identity; planes "
                f"{self.missing_indices!r} have neither source_path nor "
                "source_component_metadata."
            )

@dataclass(frozen=True, slots=True)
class ROIMaterializationPathContext:
    """Path-selection facts derived from the projected ROI plane set."""

    source_stems: tuple[str, ...]
    metadata_items: tuple[ImagePayloadMetadata, ...]

    @property
    def reference_source_stem(self) -> str | None:
        return self.source_stems[0]

    @property
    def reference_metadata(self) -> ImagePayloadMetadata | None:
        return self.metadata_items[0]

class CombinedROIMaterializationTargetPolicy(
    AlwaysMatchesContextMixin[ROIMaterializationTargetRequest],
    ROIMaterializationTargetPolicy,
):
    """Single archive behavior for ROI payloads without addressable planes."""

    strategy_key = "combined"

    def targets(
        self,
        request: ROIMaterializationTargetRequest,
    ) -> ROIMaterializationTargets:
        return (
            ROIMaterializationTarget(
                archive=ROIMaterializationArchiveIdentity(
                    request.paths.primary_output_path(request.options)
                ),
                items=request.materialization_input.unprojected_items,
            ),
        )

class ComponentAddressedCombinedROIMaterializationTargetPolicy(
    CombinedROIMaterializationTargetPolicy
):
    """Single archive behavior that preserves stream component identity."""

    strategy_key = "component_addressed_combined"

    def matches(self, request: ROIMaterializationTargetRequest) -> bool:
        return (
            request.source_identity_set.has_single_component_address
        )

    def targets(
        self,
        request: ROIMaterializationTargetRequest,
    ) -> ROIMaterializationTargets:
        item = request.materialization_input.items[0]
        return (
            ROIMaterializationTarget(
                archive=ROIMaterializationArchiveIdentity.from_metadata(
                    path=request.paths.primary_output_path(request.options),
                    metadata=item.metadata,
                ),
                items=(item,),
            ),
        )

class ProjectedROIMaterializationTargetPolicy(
    CombinedROIMaterializationTargetPolicy
):
    """Use one archive per runtime-slice-projected ROI plane."""

    strategy_key = "projected"

    def matches(self, request: ROIMaterializationTargetRequest) -> bool:
        return (
            request.requires_projected_targeting
        )

    def targets(
        self,
        request: ROIMaterializationTargetRequest,
    ) -> ROIMaterializationTargets:
        items = request.materialization_input.items
        request.source_identity_set.validate_projected_targeting()
        path_context = ROIMaterializationPathContext(
            source_stems=tuple(
                request.source_stem_authority.required_source_stem(item.metadata)
                for item in items
            ),
            metadata_items=tuple(item.metadata for item in items),
        )
        return tuple(
            ROIMaterializationTarget(
                archive=ROIMaterializationArchiveIdentity.from_metadata(
                    path=request.path(
                        paths=request.paths,
                        options=request.options,
                        metadata=item.metadata,
                        reference_source_stem=path_context.reference_source_stem,
                        reference_metadata=path_context.reference_metadata,
                    ),
                    metadata=item.metadata,
                ),
                items=(item,),
            )
            for item in items
        )

class ROIMaterializationTargetCoalescer:
    """Coalesce repeated ROI targets without allowing path identity conflicts."""

    def coalesce(
        self,
        targets: ROIMaterializationTargets,
        *,
        common_source_identity: SourceImageIdentity | None = None,
    ) -> ROIMaterializationTargets:
        targets_by_path: dict[str, ROIMaterializationTarget] = {}

        for target in targets:
            path = target.archive.path
            if path in targets_by_path:
                existing = targets_by_path[path]
                if (
                    common_source_identity is not None
                    and not existing.archive.compatible_with(target.archive)
                ):
                    existing = existing.with_archive_source_identity(
                        common_source_identity
                    )
                    target = target.with_archive_source_identity(
                        common_source_identity
                    )
                targets_by_path[path] = existing.merge_with(target)
                continue
            targets_by_path[path] = target

        coalesced = tuple(targets_by_path.values())
        if len(coalesced) != len(targets):
            logger.info(
                "ROI materialization coalesced %d target(s) into %d archive(s)",
                len(targets),
                len(coalesced),
            )
        return coalesced

class ROIMaterializationTargetAuthority:
    """Resolve ROI archive targets through registered cardinality policies."""

    def __init__(self) -> None:
        self._coalescer = ROIMaterializationTargetCoalescer()

    def targets(
        self,
        request: ROIMaterializationTargetRequest,
    ) -> ROIMaterializationTargets:
        policy = ROIMaterializationTargetPolicy.for_context(
            request,
            error_subject="ROI materialization target",
        )
        targets = self._coalescer.coalesce(
            policy.targets(request),
            common_source_identity=request.artifact_source_identity,
        )
        self._log_targets(targets)
        return targets

    @staticmethod
    def _log_targets(targets: ROIMaterializationTargets) -> None:
        if len(targets) <= 1:
            return
        logger.info(
            "ROI materialization split into %d addressable archive(s): %s",
            len(targets),
            [
                {
                    "path": target.archive.path,
                    "component_metadata": target.archive.source_component_metadata,
                }
                for target in targets
            ],
        )

_ROI_MATERIALIZATION_TARGETS = ROIMaterializationTargetAuthority()


class ROIMaterializationPlaneMetadataAuthority:
    """Declare ROI-local plane metadata from the compiled output plane domain."""

    @staticmethod
    def metadata_for_target(
        target: ROIMaterializationTarget,
        variable_components: Sequence[VariableComponents],
        metadata: ImagePayloadMetadata,
    ) -> ImagePayloadMetadata:
        components = ComponentSet.coerce(variable_components).as_tuple()
        if not components:
            return metadata
        if len(target.items) != 1:
            if any(np.ndim(item.data) > 2 for item in target.items):
                raise ValueError(
                    "ROI output plane components require one unprojected stack "
                    "per materialization target."
                )
            return metadata

        item = target.items[0]
        local_plane_shape = tuple(int(size) for size in np.shape(item.data)[:-2])
        if not local_plane_shape:
            return metadata
        if len(local_plane_shape) != len(components):
            raise ValueError(
                "ROI output plane component rank does not match the materialized "
                f"label stack: {components!r} for {local_plane_shape!r}."
            )

        local_plane_count = 1
        for extent in local_plane_shape:
            local_plane_count *= extent
        source_plane_count = metadata.source_provenance.source_plane_count
        if source_plane_count not in (0, local_plane_count):
            raise ValueError(
                "ROI output source-plane provenance does not match its materialized "
                f"label stack: {source_plane_count} != {local_plane_count}."
            )
        if local_plane_count > 1 and source_plane_count == 0:
            raise ValueError(
                "Multi-plane ROI output requires exact source-plane provenance for "
                f"its declared components {components!r}."
            )

        return ImagePayloadMetadata.compose(
            (item.value,),
            mode=ImagePayloadMetadataCompositionMode.STACK,
            source_metadata=(metadata,),
        )


@writer_for(
    ROIOptions,
    MaterializationFormat.ROI_ZIP,
    primary_path=ROIPrimaryPathAuthority.primary_path,
    candidate_paths=ROICandidatePathAuthority.paths,
)
def _write_roi_zip(
    data: MaterializationValue,
    options: ROIOptions,
    ctx: MaterializationContext,
) -> list[Output]:
    from polystore.roi import extract_rois_from_labeled_mask

    materialization_input = MaterializationInput.from_runtime_slice_projected_value(
        data,
        options,
    )
    payload = materialization_input.data
    paths = ctx.paths(options)
    summary_path = paths.related_output_path(options, options.summary_suffix)

    if materialization_is_empty(payload):
        return [Output(path=summary_path, content="No segmentation masks generated (empty data)\n")]

    targets = _ROI_MATERIALIZATION_TARGETS.targets(
        ROIMaterializationTargetRequest(
            paths=paths,
            options=options,
            source_stem_authority=SourceStemAuthority.from_processing_context(
                ctx.context
            ),
            materialization_input=materialization_input,
            artifact_source_identity=ctx.artifact_source_identity,
            output_key=ctx.output_key,
            step_index=ctx.step_index,
        )
    )
    outs: list[Output] = []
    total_roi_count = 0
    roi_paths: list[str] = []
    source_domain_authority = ROIMaterializationSourceSpatialDomain()
    for target in targets:
        target_rois: list = []
        for item in target.items:
            source_domain = source_domain_authority.domain_for_item(item)
            rois = extract_rois_from_labeled_mask(
                item.data,
                min_area=options.min_area,
                extract_contours=options.extract_contours,
                spatial_origin_yx=source_domain.origin_yx,
                source_spatial_shape_yx=source_domain.source_shape_yx,
            )
            target_rois.extend(rois)

        total_roi_count += len(target_rois)
        if target_rois:
            roi_paths.append(target.archive.path)
            item_metadata = target.items[0].metadata
            item_metadata = item_metadata.with_source_provenance(
                item_metadata.source_provenance.with_common_scalar_identity_from_planes()
            )
            item_metadata = ROIMaterializationPlaneMetadataAuthority.metadata_for_target(
                target,
                ctx.variable_components,
                item_metadata,
            )
            source_identity = target.archive.source_identity
            if source_identity is not None:
                item_metadata = item_metadata.with_source_provenance(
                    SourceImageProvenance(
                        source_path=source_identity.path,
                        source_component_metadata=source_identity.component_metadata,
                    ).with_missing_from(item_metadata.source_provenance)
                )
            outs.append(
                Output(
                    path=target.archive.path,
                    content=target_rois,
                    metadata=item_metadata.replace_fields(
                        source_spatial_domain=(
                            source_domain_authority.domain_for_target(target)
                        ),
                    ),
                )
            )

    summary = (
        f"Segmentation ROIs: {total_roi_count} cells\n"
        "Spatial dimensions: 2D\n"
        f"Projected source planes: {len(materialization_input.items)}\n"
    )
    if roi_paths:
        summary += "ROI files:\n"
        for roi_path in roi_paths:
            summary += f"- {roi_path}\n"
    else:
        summary += "No ROIs extracted (all regions below min_area threshold)\n"
    outs.append(Output(path=summary_path, content=summary))
    return outs

class TiffStackSlicePayloadAuthority:
    """Own how TIFF stack materialization expands payloads into saved slices."""

    @staticmethod
    def slices(
        materialization_input: MaterializationInput,
    ) -> tuple[MaterializationValue, ...]:
        if materialization_input.sequence_type is not None:
            return tuple(item.data for item in materialization_input.items)
        if len(materialization_input.items) != 1:
            raise ValueError(
                "TIFF stack materialization requires exactly one payload or an "
                "explicit payload sequence."
            )
        item = materialization_input.items[0]
        if item.metadata.plane_axis is None:
            return (item.data,)
        data = np.asarray(item.data)
        if data.ndim < 3 or data.shape[0] <= 0:
            raise ValueError(
                "TIFF stack payload storage does not match its declared leading "
                f"plane axis: shape {data.shape!r}."
            )
        return tuple(data[index] for index in range(data.shape[0]))


@singledispatch
def materialization_emits_variable_component_planes(
    options: FileOutputOptions,
    data: MaterializationValue,
) -> bool:
    """Return whether one writer emits source-stack identity as separate outputs."""
    del options, data
    return False


@materialization_emits_variable_component_planes.register(TiffStackOptions)
def tiff_stack_emits_variable_component_planes(
    options: TiffStackOptions,
    data: MaterializationValue,
) -> bool:
    materialization_input = MaterializationInput.from_value(data, options)
    metadata_input = MaterializationInput.from_runtime_slice_projected_value(
        data,
        options,
    )
    slices = TiffStackSlicePayloadAuthority.slices(materialization_input)
    return len(slices) > 1 and len(metadata_input.items) == len(slices)


@materialization_emits_variable_component_planes.register(ImageFileOptions)
def image_file_emits_variable_component_planes(
    options: ImageFileOptions,
    data: MaterializationValue,
) -> bool:
    if not (
        options.filename_identity is MaterializedFilenameIdentity.SOURCE_IDENTITY
        or _image_template_uses_source_metadata(options.relative_path_template)
        or _image_template_uses_sequence_index(options.relative_path_template)
    ):
        return False
    return len(
        MaterializationInput.from_runtime_slice_projected_value(data, options).items
    ) > 1

@materialization_emits_variable_component_planes.register(ROIOptions)
def roi_emits_variable_component_planes(
    options: ROIOptions,
    data: MaterializationValue,
) -> bool:
    materialization_input = MaterializationInput.from_runtime_slice_projected_value(
        data,
        options,
    )
    return materialization_input.source_plane_projection.requires_source_identity


@singledispatch
def materialization_uses_filename_source_identity(
    options: FileOutputOptions,
    data: MaterializationValue,
) -> bool:
    """Return whether stream identity should follow the materialized filename."""
    del options, data
    return False


@materialization_uses_filename_source_identity.register(TiffStackOptions)
def tiff_stack_uses_filename_source_identity(
    options: TiffStackOptions,
    data: MaterializationValue,
) -> bool:
    return not materialization_emits_variable_component_planes(options, data)


@materialization_uses_filename_source_identity.register(ImageFileOptions)
def image_file_uses_filename_source_identity(
    options: ImageFileOptions,
    data: MaterializationValue,
) -> bool:
    del data
    return options.filename_identity is MaterializedFilenameIdentity.SOURCE_IDENTITY


@writer_for(
    TiffStackOptions,
    MaterializationFormat.TIFF_STACK,
    candidate_paths=TiffStackCandidatePathAuthority.paths,
)
def _write_tiff_stack(
    data: MaterializationValue,
    options: TiffStackOptions,
    ctx: MaterializationContext,
) -> list[Output]:
    materialization_input = MaterializationInput.from_value(data, options)
    metadata_input = MaterializationInput.from_runtime_slice_projected_value(
        data,
        options,
    )
    data = materialization_input.data
    paths = ctx.paths(options)

    if materialization_is_empty(data):
        summary_path = _tiff_stack_summary_path(paths, options)
        return [Output(path=summary_path, content=options.empty_summary)]

    slices = TiffStackSlicePayloadAuthority.slices(materialization_input)

    slice_metadata = _TIFF_STACK_SLICE_METADATA.metadata_for_slices(
        metadata_input,
        len(slices),
        variable_components=ctx.variable_components,
        artifact_source_identity=ctx.artifact_source_identity,
    )

    outs: list[Output] = []
    for i, arr in enumerate(slices):
        filename = _tiff_stack_slice_path(
            paths,
            options,
            index=i,
            slice_count=len(slices),
        )
        out_arr = TiffArrayAuthority.output_array(arr, options)
        outs.append(
            Output.from_metadata(
                path=filename,
                content=out_arr,
                metadata=slice_metadata[i],
            )
        )

    summary_path = _tiff_stack_summary_path(paths, options)
    first = None
    if slices:
        first = slices[0]
    source_path = paths.source_identity_base_path(options)
    if source_path is not None:
        base_filename_pattern = (
            source_path.name
            if len(slices) == 1
            else f"{paths.primary_output_basename(options)}"
            f"{options.slice_pattern}"
        )
    else:
        base_filename_pattern = f"{paths.name}{options.slice_pattern}"
    summary_content = (
        f"Images saved: {len(slices)} files\n"
        f"Base filename pattern: {base_filename_pattern}\n"
        f"Image dtype: {TiffArrayAuthority.dtype_name(first)}\n"
        f"Image shape: {TiffArrayAuthority.shape_text(first)}\n"
    )
    outs.append(Output(path=summary_path, content=summary_content))
    return outs

class TiffArrayAuthority:
    """Own array inspection and normalization for TIFF stack materialization."""

    UINT8_DTYPE: ClassVar[np.dtype] = np.dtype("uint8")

    @classmethod
    def output_array(
        cls,
        value: MaterializationValue,
        options: TiffStackOptions,
    ) -> MaterializationValue:
        if not options.normalize_uint8:
            return value
        array = np.asarray(value)
        if array.dtype == cls.UINT8_DTYPE:
            return value
        max_val = cls.max_value(array)
        if max_val <= 1.0:
            return (array * 255).astype(cls.UINT8_DTYPE)
        return array.astype(cls.UINT8_DTYPE)

    @staticmethod
    def max_value(array: np.ndarray) -> float:
        if array.size == 0:
            return 0.0
        return float(np.max(array))

    @staticmethod
    def dtype_name(value: MaterializationValue) -> str:
        if value is None:
            return "unknown"
        return str(np.asarray(value).dtype)

    @staticmethod
    def shape_text(value: MaterializationValue) -> str:
        if value is None:
            return "unknown"
        return str(np.shape(value))

@dataclass(frozen=True, slots=True)
class RuntimePlaneStackAxisMetadataProjection:
    """Project scalar source identity across variable component runtime planes."""

    variable_components: Sequence[VariableComponents]
    artifact_source_identity: SourceImageIdentity | None = None

    def metadata_for_item(
        self,
        item: MaterializationInputItem,
    ) -> ImagePayloadMetadata:
        metadata = self.metadata_with_artifact_source_identity(item.metadata)
        runtime_plane = item.runtime_plane
        if runtime_plane is None or not self.variable_components:
            return metadata
        if runtime_plane.carries_source_plane_identity:
            return metadata

        axes = self.ordered_axes()
        if len(axes) != len(runtime_plane.plane_indices):
            raise ValueError(
                "TIFF stack output uses variable components "
                f"{axes!r}, but runtime plane metadata has "
                f"{len(runtime_plane.plane_indices)} coordinate(s)."
            )

        component_metadata = metadata.source_component_metadata
        if component_metadata is None:
            raise ValueError(
                "TIFF stack output uses variable components "
                f"{axes!r}, but slice metadata has no scalar "
                "source_component_metadata."
            )

        projected_metadata = VariableComponentAxisProjection(
            frozenset(axes)
        ).project_component_metadata(
            component_metadata,
            runtime_plane.plane_indices,
        )

        return metadata.with_source_provenance(
            metadata.source_provenance.with_source_component_metadata(
                projected_metadata,
            )
        )

    def metadata_with_artifact_source_identity(
        self,
        metadata: ImagePayloadMetadata,
    ) -> ImagePayloadMetadata:
        if self.artifact_source_identity is None:
            return metadata
        artifact_provenance = SourceImageProvenance(
            source_path=self.artifact_source_identity.path,
            source_component_metadata=self.artifact_source_identity.component_metadata,
        )
        return metadata.with_source_provenance(
            metadata.source_provenance.with_missing_from(artifact_provenance)
        )

    def ordered_axes(self) -> tuple[str, ...]:
        return VariableComponentAxisProjection(
            frozenset(
                component.value
                for component in self.variable_components
                if component.value is not None
            )
        ).ordered_axes()


@dataclass(frozen=True, slots=True)
class RuntimePlaneStackAxesProjectionSelection:
    """Select variable component axes that still need runtime-plane projection."""

    variable_components: Sequence[VariableComponents]
    items: tuple[MaterializationInputItem, ...]
    artifact_source_identity: SourceImageIdentity | None = None

    def axes(self) -> frozenset[str]:
        ordered_axes = RuntimePlaneStackAxisMetadataProjection(
            self.variable_components
        ).ordered_axes()
        if not ordered_axes:
            return frozenset()
        runtime_rank = self.runtime_plane_rank()
        if runtime_rank == 0:
            return frozenset()
        if any(self.axis_varies(axis) for axis in ordered_axes):
            return frozenset()
        indexed_source_plane_axis = self.indexed_source_plane_axis(ordered_axes)
        if indexed_source_plane_axis is not None:
            return frozenset((indexed_source_plane_axis,))
        if runtime_rank != len(ordered_axes):
            raise ValueError(
                "TIFF stack output cannot map runtime plane metadata with "
                f"rank {runtime_rank} onto variable component axes "
                f"{ordered_axes!r}; slice metadata does not already vary across "
                "any variable component axis."
            )
        return frozenset(ordered_axes)

    def runtime_plane_rank(self) -> int:
        ranks = tuple(
            len(item.runtime_plane.plane_indices)
            for item in self.items
            if item.runtime_plane is not None
        )
        if not ranks:
            return 0
        first = ranks[0]
        if any(rank != first for rank in ranks):
            raise ValueError(
                "TIFF stack materialization received inconsistent runtime "
                f"plane metadata ranks: {ranks!r}."
            )
        return first

    def runtime_plane_shape(self) -> tuple[int, ...]:
        shapes = tuple(
            item.runtime_plane.plane_shape
            for item in self.items
            if item.runtime_plane is not None
        )
        if not shapes:
            return ()
        first = shapes[0]
        if any(shape != first for shape in shapes):
            raise ValueError(
                "TIFF stack materialization received inconsistent runtime "
                f"plane metadata shapes: {shapes!r}."
            )
        return first

    def runtime_planes_carry_source_identity(self) -> bool:
        states = tuple(
            item.runtime_plane.carries_source_plane_identity
            for item in self.items
            if item.runtime_plane is not None
        )
        if not states:
            return False
        if any(state != states[0] for state in states):
            raise ValueError(
                "TIFF stack materialization received mixed source-plane "
                f"runtime metadata states: {states!r}."
            )
        return states[0]

    def axis_varies(self, axis: str) -> bool:
        component = VariableComponentAxisProjection.component_for_axis(axis)
        values = tuple(
            source_component_metadata_value(metadata, component)
            for metadata in self.item_component_metadata()
        )
        return len(frozenset(values)) > 1

    def indexed_source_plane_axis(
        self,
        ordered_axes: tuple[str, ...],
    ) -> str | None:
        runtime_shape = self.runtime_plane_shape()
        if len(runtime_shape) != 1:
            return None
        source_plane_axis = SourcePlaneIndexedMetadata.projected_component().value
        if source_plane_axis not in ordered_axes:
            return None
        metadata_values = self.item_component_metadata()
        if not metadata_values:
            return None
        for metadata in metadata_values:
            indexed_metadata = SourcePlaneIndexedMetadata.from_metadata(
                metadata,
                expected_plane_count=runtime_shape[0],
            )
            if indexed_metadata is None:
                indexed_metadata = (
                    SourcePlaneIndexedMetadata.from_declared_runtime_stack_origin(
                        metadata,
                        source_plane_count=runtime_shape[0],
                    )
                )
            if indexed_metadata is None:
                return None
        return source_plane_axis

    def item_component_metadata(self) -> tuple[SourceComponentMetadata, ...]:
        metadata_values = tuple(
            self.effective_item_metadata(item).source_component_metadata
            for item in self.items
        )
        if any(metadata is None for metadata in metadata_values):
            return ()
        return tuple(
            metadata
            for metadata in metadata_values
            if metadata is not None
        )

    def effective_item_metadata(
        self,
        item: MaterializationInputItem,
    ) -> ImagePayloadMetadata:
        return RuntimePlaneStackAxisMetadataProjection(
            frozenset(),
            self.artifact_source_identity,
        ).metadata_with_artifact_source_identity(item.metadata)

@dataclass(frozen=True, slots=True)
class TiffStackSliceMetadataRequest:
    """Input facts needed to align metadata to emitted TIFF slices."""

    materialization_input: MaterializationInput
    slice_count: int
    variable_components: Sequence[VariableComponents] = field(default_factory=tuple)
    artifact_source_identity: SourceImageIdentity | None = None

    @property
    def input_item_count(self) -> int:
        return len(self.materialization_input.items)

    @property
    def has_empty_slice_set(self) -> bool:
        return self.slice_count <= 0

    @property
    def has_presliced_payloads(self) -> bool:
        return self.input_item_count == self.slice_count

    @property
    def has_single_payload(self) -> bool:
        return self.input_item_count == 1

    @property
    def has_single_presliced_payload(self) -> bool:
        return self.has_single_payload and self.has_presliced_payloads

class TiffStackSliceMetadataPolicy(
    MostDerivedContextStrategyMixin[TiffStackSliceMetadataRequest],
    ABC,
):
    """One cardinality policy for TIFF slice metadata projection."""

    @abstractmethod
    def metadata_for_slices(
        self,
        request: TiffStackSliceMetadataRequest,
    ) -> tuple[ImagePayloadMetadata, ...]:
        """Return one metadata record for each emitted TIFF slice."""

class UnknownTiffStackSliceMetadataPolicy(
    AlwaysMatchesContextMixin[TiffStackSliceMetadataRequest],
    TiffStackSliceMetadataPolicy,
):
    """Ambiguous cardinality gets empty metadata rather than invented identity."""

    strategy_key = "unknown"

    def metadata_for_slices(
        self,
        request: TiffStackSliceMetadataRequest,
    ) -> tuple[ImagePayloadMetadata, ...]:
        return tuple(ImagePayloadMetadata() for _index in range(request.slice_count))

class PreslicedTiffStackSliceMetadataPolicy(UnknownTiffStackSliceMetadataPolicy):
    """Input items already correspond one-to-one with emitted TIFF slices."""

    strategy_key = "presliced"

    def matches(self, request: TiffStackSliceMetadataRequest) -> bool:
        return request.has_presliced_payloads

    def metadata_for_slices(
        self,
        request: TiffStackSliceMetadataRequest,
    ) -> tuple[ImagePayloadMetadata, ...]:
        selected_variable_components = RuntimePlaneStackAxesProjectionSelection(
            request.variable_components,
            request.materialization_input.items,
            request.artifact_source_identity,
        ).axes()
        projector = RuntimePlaneStackAxisMetadataProjection(
            tuple(
                component
                for component in request.variable_components
                if component.value in selected_variable_components
            ),
            request.artifact_source_identity,
        )
        return tuple(
            projector.metadata_for_item(item)
            for item in request.materialization_input.items
        )

class SinglePayloadTiffStackSliceMetadataPolicy(UnknownTiffStackSliceMetadataPolicy):
    """One stacked payload is being split into emitted TIFF slices."""

    strategy_key = "single_payload"

    def matches(self, request: TiffStackSliceMetadataRequest) -> bool:
        return request.has_single_payload

    def metadata_for_slices(
        self,
        request: TiffStackSliceMetadataRequest,
    ) -> tuple[ImagePayloadMetadata, ...]:
        metadata = request.materialization_input.items[0].metadata
        return tuple(
            metadata.for_source_plane(index) for index in range(request.slice_count)
        )

class SinglePreslicedTiffStackSliceMetadataPolicy(
    PreslicedTiffStackSliceMetadataPolicy,
    SinglePayloadTiffStackSliceMetadataPolicy,
):
    """One input item already represents the one emitted TIFF slice."""

    strategy_key = "single_presliced"

    def matches(self, request: TiffStackSliceMetadataRequest) -> bool:
        return request.has_single_presliced_payload

class EmptyTiffStackSliceMetadataPolicy(
    PreslicedTiffStackSliceMetadataPolicy,
    SinglePayloadTiffStackSliceMetadataPolicy,
):
    """No emitted slices means no metadata records."""

    strategy_key = "empty"

    def matches(self, request: TiffStackSliceMetadataRequest) -> bool:
        return request.has_empty_slice_set

    def metadata_for_slices(
        self,
        request: TiffStackSliceMetadataRequest,
    ) -> tuple[ImagePayloadMetadata, ...]:
        return ()

class TiffStackSliceMetadataAuthority:
    """Owns metadata alignment for TIFF slices emitted from runtime payloads."""

    def metadata_for_slices(
        self,
        materialization_input: MaterializationInput,
        slice_count: int,
        *,
        variable_components: Sequence[VariableComponents] = (),
        artifact_source_identity: SourceImageIdentity | None = None,
    ) -> tuple[ImagePayloadMetadata, ...]:
        request = TiffStackSliceMetadataRequest(
            materialization_input=materialization_input,
            slice_count=slice_count,
            variable_components=tuple(variable_components),
            artifact_source_identity=artifact_source_identity,
        )
        policy = TiffStackSliceMetadataPolicy.for_context(
            request,
            error_subject="TIFF stack slice metadata",
        )
        return policy.metadata_for_slices(request)

_TIFF_STACK_SLICE_METADATA = TiffStackSliceMetadataAuthority()

@dataclass(init=False)
class MaterializationSpec(ArtifactMaterializationPayload):
    """Declarative materialization spec.

    The spec is a list of *writer options* objects. Writer dispatch is inferred
    from the option type.
    """

    outputs: tuple[FileOutputOptions, ...]
    allowed_backends: list[str] | None
    primary: int
    write_mode: WriteMode

    def __init__(
        self,
        *outputs: FileOutputOptions | Sequence[FileOutputOptions],
        allowed_backends: list[str] | None = None,
        primary: int = 0,
        write_mode: WriteMode = WriteMode.OVERWRITE,
    ):
        if len(outputs) == 1 and isinstance(outputs[0], (list, tuple)):
            outputs = tuple(outputs[0])

        if not outputs:
            raise ValueError("MaterializationSpec requires at least one output options object")

        for opt in outputs:
            if isinstance(opt, dict):
                raise TypeError("dict-based materialization options are not supported")
            option_type = opt.__class__
            if option_type not in _WRITERS_BY_OPTIONS:
                raise ValueError(
                    f"No writer registered for options type {option_type.__name__}. "
                    f"Registered: {[t.__name__ for t in _WRITERS_BY_OPTIONS.keys()]}"
                )

        if primary < 0 or primary >= len(outputs):
            raise IndexError("MaterializationSpec.primary out of range")
        if not isinstance(write_mode, WriteMode):
            raise TypeError(
                "MaterializationSpec.write_mode must be a WriteMode, "
                f"got {type(write_mode).__name__}."
            )

        self.outputs = tuple(outputs)
        self.allowed_backends = allowed_backends
        self.primary = primary
        self.write_mode = write_mode

    @classmethod
    def __objectstate_rebuild__(
        cls,
        *,
        outputs: tuple[FileOutputOptions, ...],
        allowed_backends: list[str] | None = None,
        primary: int = 0,
        write_mode: WriteMode = WriteMode.OVERWRITE,
    ) -> "MaterializationSpec":
        # Rebuild via the normal constructor to keep validation behavior.
        return cls(
            *outputs,
            allowed_backends=allowed_backends,
            primary=primary,
            write_mode=write_mode,
        )

    def participates_in_runtime_export_observation(self) -> bool:
        """Explicit materialization specs are externally observed exports."""
        return True

    def participates_in_persistent_materialization(self) -> bool:
        """Explicit materialization specs write to configured persistent targets."""
        return True

    def tabular_field_names(self) -> tuple[str, ...]:
        """Return declared tabular field names from the primary writer first."""
        return tabular_field_names_from_options(self.outputs, primary=self.primary)

    def candidate_paths(self, base_path: str) -> tuple[str, ...]:
        """Return all paths this materialization spec may emit for backend routing."""
        return tuple(
            path
            for output_options in self.outputs
            for path in _WRITERS_BY_OPTIONS[
                output_options.__class__
            ].candidate_paths(output_options, base_path)
        )

    def emits_variable_component_planes(self, data: MaterializationValue) -> bool:
        """Return whether any writer emits variable-component planes."""
        return any(
            materialization_emits_variable_component_planes(output_options, data)
            for output_options in self.outputs
        )

    def emitted_source_identities(
        self,
        data: MaterializationValue,
    ) -> tuple[SourceImageIdentity, ...]:
        """Return source identities for writers that emit projected planes."""
        for output_options in self.outputs:
            if not materialization_emits_variable_component_planes(
                output_options,
                data,
            ):
                continue
            materialization_input = (
                MaterializationInput.from_runtime_slice_projected_value(
                    data,
                    output_options,
                )
            )
            if len(materialization_input.items) == 1:
                provenance = materialization_input.items[0].metadata.source_provenance
                if provenance.source_plane_count > 1:
                    return tuple(
                        provenance.for_source_plane(index).scalar_source_identity
                        for index in range(provenance.source_plane_count)
                    )
            return tuple(
                item.metadata.source_provenance.scalar_source_identity
                for item in materialization_input.items
            )
        return ()

    def uses_filename_source_identity(self, data: MaterializationValue) -> bool:
        """Return whether any writer routes stream identity by materialized filename."""
        return any(
            materialization_uses_filename_source_identity(output_options, data)
            for output_options in self.outputs
        )

    def uses_source_identity_filename(self) -> bool:
        """Return whether this spec materializes files by source identity."""
        return any(
            output_options.filename_identity
            is MaterializedFilenameIdentity.SOURCE_IDENTITY
            for output_options in self.outputs
        )

def tabular_field_names_from_options(
    options: Sequence[FileOutputOptions],
    *,
    primary: int = 0,
) -> tuple[str, ...]:
    """Return declared tabular fields from writer options, primary first."""
    if not options:
        return ()
    output_tuple = tuple(options)
    ordered_options = (
        output_tuple[primary : primary + 1]
        + output_tuple[:primary]
        + output_tuple[primary + 1 :]
    )
    for output_options in ordered_options:
        if (
            isinstance(output_options, TabularExtractionOptions)
            and output_options.fields
        ):
            return tuple(output_options.fields)
    return ()

def tabular_field_names_from_materialization(materialization) -> tuple[str, ...]:
    """Return declared tabular fields from a materialization spec."""
    if not isinstance(materialization, MaterializationSpec):
        return ()
    return materialization.tabular_field_names()

class BackendSequenceAuthority:
    """Normalize materialization backend declarations."""

    @staticmethod
    def normalize(backends: Sequence[str] | str) -> list[str]:
        if isinstance(backends, str):
            return [backends]
        return list(backends)

class AllowedBackendsAuthority:
    """Validate materialization backends against the spec allow-list."""

    @staticmethod
    def validate(spec: MaterializationSpec, backends: list[str]) -> None:
        if not spec.allowed_backends:
            return
        invalid = [b for b in backends if b not in spec.allowed_backends]
        if invalid:
            raise ValueError(
                f"Backend(s) {invalid} not in allowed backends for this spec: "
                f"{spec.allowed_backends}"
            )


def _materialization_output_groups(
    spec: MaterializationSpec,
    data: MaterializationValue,
    context: MaterializationContext,
) -> tuple[tuple[WriterSpec, tuple[Output, ...]], ...]:
    """Build writer outputs once for persistence and exact output observation."""

    return tuple(
        (
            writer,
            tuple(
                output.with_source_identity_fallback(
                    context.artifact_source_identity
                ).with_variable_components(context.variable_components)
                for output in writer.write(data, options, context)
            ),
        )
        for options in spec.outputs
        for writer in (_WRITERS_BY_OPTIONS[options.__class__],)
    )


def materialization_outputs(
    spec: MaterializationSpec,
    data: MaterializationValue,
    path: str,
    filemanager: "FileManager",
    context: "ProcessingContext | None" = None,
    extra_inputs: dict | None = None,
    *,
    artifact_source_identity: SourceImageIdentity | None = None,
    variable_components: Sequence[VariableComponents] = (),
    source_paths: Sequence[str] = (),
    output_key: str | None = None,
    step_index: int | None = None,
) -> tuple[Output, ...]:
    """Derive every concrete output from a materialization contract without saving."""

    materialization_context = MaterializationContext(
        base_path=path,
        backends=[],
        backend_kwargs={},
        filemanager=filemanager,
        extra_inputs={} if extra_inputs is None else extra_inputs,
        context=context,
        artifact_source_identity=artifact_source_identity,
        variable_components=tuple(variable_components),
        write_mode=spec.write_mode,
        source_paths=tuple(str(p) for p in source_paths),
        output_key=output_key,
        step_index=step_index,
    )
    return tuple(
        output
        for _writer, outputs in _materialization_output_groups(
            spec,
            data,
            materialization_context,
        )
        for output in outputs
    )


def materialize(
    spec: MaterializationSpec,
    data: MaterializationValue,
    path: str,
    filemanager: "FileManager",
    backends: Sequence[str] | str,
    backend_kwargs: BackendKwargsInput = BACKEND_KWARGS_ABSENT,
    context: "ProcessingContext | None" = None,
    extra_inputs: dict | None = None,
    *,
    artifact_source_identity: SourceImageIdentity | None = None,
    variable_components: Sequence[VariableComponents] = (),
    source_paths: Sequence[str] = (),
    output_key: str | None = None,
    step_index: int | None = None,
) -> str:
    """Materialize data to one or more backends."""

    normalized_backends = BackendSequenceAuthority.normalize(backends)
    AllowedBackendsAuthority.validate(spec, normalized_backends)
    effective_backend_kwargs = BackendKwargsAuthority.normalize(backend_kwargs)
    if extra_inputs is None:
        effective_extra_inputs = {}
    else:
        effective_extra_inputs = extra_inputs

    ctx = MaterializationContext(
        base_path=path,
        backends=normalized_backends,
        backend_kwargs=effective_backend_kwargs,
        filemanager=filemanager,
        extra_inputs=effective_extra_inputs,
        context=context,
        artifact_source_identity=artifact_source_identity,
        variable_components=tuple(variable_components),
        write_mode=spec.write_mode,
        source_paths=tuple(str(p) for p in source_paths),
        output_key=output_key,
        step_index=step_index,
    )

    primary_path = ""

    for i, (writer, outs) in enumerate(
        _materialization_output_groups(spec, data, ctx)
    ):
        ctx.saver.save_all(outs)
        if i == spec.primary:
            primary_path = writer.primary_path(list(outs))

    return primary_path
