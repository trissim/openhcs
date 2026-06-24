"""Materialization core (writer-based, greenfield).

Key idea: the abstraction boundary is the output *format* (writers), not per-analysis handlers.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import operator
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence, Sized
from dataclasses import dataclass, field, is_dataclass
from functools import singledispatch
from pathlib import Path
from typing import ClassVar, TYPE_CHECKING, TypeAlias

import pandas as pd
import numpy as np
from metaclass_registry import AutoRegisterMeta
from polystore.streaming.viewer_transport import ViewerStreamBackendKwargs

from openhcs.core.artifacts import ArtifactMaterializationPayload
from openhcs.core.image_shapes import ColorImageShapeRole
from openhcs.processing.materialization.constants import MaterializationFormat, WriteMode
from openhcs.processing.materialization.options import (
    CsvOptions,
    FileOutputOptions,
    JsonOptions,
    ROIOptions,
    SourceOptions,
    TabularExtractionOptions,
    TextOptions,
    TiffStackOptions,
)
from openhcs.core.runtime_values import (
    ColumnarRows,
    ImagePayloadMetadata,
    image_payload_data,
    image_payload_metadata,
    runtime_array_operand,
)
from openhcs.core.source_image_provenance import (
    SourceComponentMetadata,
    SourceImageIdentity,
)
from openhcs.core.registry_strategies import (
    AlwaysMatchesContextMixin,
    MostDerivedContextStrategyMixin,
)
from openhcs.core.runtime_semantics import (
    measurement_row_mapping,
    supports_measurement_row_mapping,
)
from openhcs.core.runtime_slice_projection import (
    RuntimeProjectedPayloadItem,
    RuntimeProjectionSourceIdentityRequirement,
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

@dataclass(frozen=True)
class Output:
    path: str
    content: MaterializationValue
    source_identity: SourceImageIdentity | None = None

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
            source_identity=cls.source_identity_from_metadata(metadata),
        )

    @staticmethod
    def source_identity_from_metadata(
        metadata: ImagePayloadMetadata | None,
    ) -> SourceImageIdentity | None:
        if metadata is None:
            return None
        return metadata.source_provenance.scalar_source_identity

    @property
    def source_component_metadata(self) -> SourceComponentMetadata | None:
        if self.source_identity is None:
            return None
        return self.source_identity.component_metadata

    @property
    def viewer_stream_requires_source_metadata(self) -> bool:
        return not isinstance(self.content, str)

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
        return Path(metadata.source_provenance.scalar_source_identity.path).stem

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
        suffixes = Path(source_path).suffixes
        if not suffixes:
            return ()
        return ("".join(suffixes),)

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
            source_identity=Output.source_identity_from_metadata(metadata),
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
            return Path(source_path).stem
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

    @property
    def has_unprojected_stack_payload(self) -> bool:
        items = self.materialization_input.items
        return len(items) == 1 and len(np.shape(items[0].data)) >= 3

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
                )
                for source_payload in source_payloads
                for item in (
                    RuntimeProjectionSourceIdentityRequirement.OPTIONAL
                ).project_payload_items(
                    source_payload,
                    source_description="materialization payload",
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
        p = Path(path)
        name = p.name

        if name.endswith(PathSuffixes.ROI_ZIP):
            name = name[: -len(PathSuffixes.ROI_ZIP)]

        if options.strip_pkl and name.endswith(PathSuffixes.PKL):
            name = name[: -len(PathSuffixes.PKL)]
        if options.strip_roi_suffix and name.endswith(PathSuffixes.ROI):
            name = name[: -len(PathSuffixes.ROI)]

        return p.with_name(name)

    def with_suffix(self, suffix: str) -> str:
        return str(self.parent / f"{self.name}{suffix}")

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
        component_metadata = output.source_component_metadata
        if component_metadata is None:
            if not output.viewer_stream_requires_source_metadata:
                return self.values.to_kwargs()
            raise ValueError(
                "Viewer stream materialization requires output metadata with "
                "source_component_metadata."
            )
        return self.values.with_single_item_component_metadata(
            component_metadata,
        ).to_kwargs()

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

    def save(
        self,
        output: Output,
    ) -> None:
        for backend in self.backends:
            self._prepare_path(backend, output.path)
            if backend in self.backend_kwargs:
                kwargs_payload = self.backend_kwargs[backend]
            else:
                kwargs_payload = EMPTY_BACKEND_CALL_KWARGS
            kwargs = kwargs_payload.to_filemanager_kwargs(output)
            self.filemanager.save(output.content, output.path, backend, **kwargs)

    def _prepare_path(self, backend: str, path: str) -> None:
        backend_instance = self.filemanager._get_backend(backend)
        if not backend_instance.requires_filesystem_validation:
            return

        self.filemanager.ensure_directory(str(Path(path).parent), backend)

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
    write_mode: WriteMode = WriteMode.OVERWRITE

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
        _WRITERS_BY_OPTIONS[options_type] = WriterSpec(
            format=fmt,
            options_type=options_type,
            write=fn,
            primary_path=selected_primary_path,
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
                    path=ctx.paths(options).with_suffix(options.filename_suffix),
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
                    request.paths.with_suffix(request.options.roi_suffix)
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
        return request.source_identity_set.has_single_component_address

    def targets(
        self,
        request: ROIMaterializationTargetRequest,
    ) -> ROIMaterializationTargets:
        item = request.materialization_input.items[0]
        return (
            ROIMaterializationTarget(
                archive=ROIMaterializationArchiveIdentity.from_metadata(
                    path=request.paths.with_suffix(request.options.roi_suffix),
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
        return request.requires_projected_targeting

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

class UnprojectedStackROIMaterializationTargetPolicy(
    CombinedROIMaterializationTargetPolicy
):
    """Reject stacked ROI labels before they can collapse into one archive."""

    strategy_key = "unprojected_stack"

    def matches(self, request: ROIMaterializationTargetRequest) -> bool:
        return request.has_unprojected_stack_payload

    def targets(
        self,
        request: ROIMaterializationTargetRequest,
    ) -> ROIMaterializationTargets:
        shape = np.shape(request.materialization_input.items[0].data)
        raise ValueError(
            "ROI materialization received an unprojected label stack with shape "
            f"{shape!r}. Multi-plane ROI labels must be projected through "
            "RuntimeSliceProjection into addressable source planes before archive "
            "targeting."
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

@writer_for(
    ROIOptions,
    MaterializationFormat.ROI_ZIP,
    primary_path=ROIPrimaryPathAuthority.primary_path,
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
    summary_path = paths.with_suffix(options.summary_suffix)

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
        )
    )
    outs: list[Output] = []
    total_roi_count = 0
    roi_paths: list[str] = []
    for target in targets:
        target_rois: list = []
        for item in target.items:
            source_domain = item.metadata.source_spatial_domain.with_value_name(
                "ROI materialization source image",
            )
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
            outs.append(
                Output(
                    path=target.archive.path,
                    content=target_rois,
                    source_identity=target.archive.source_identity,
                )
            )

    summary = f"Segmentation ROIs: {total_roi_count} cells\nZ-planes: {len(materialization_input.items)}\n"
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
        data: MaterializationValue,
        options: TiffStackOptions,
    ) -> tuple[MaterializationValue, ...]:
        if isinstance(data, (list, tuple)):
            return tuple(data)

        shape = np.shape(data)
        if TiffStackSlicePayloadAuthority.emits_stack_planes(data, options, shape):
            return tuple(data[index] for index in range(shape[0]))  # type: ignore[index]
        return (data,)

    @staticmethod
    def emits_stack_planes(
        data: MaterializationValue,
        options: TiffStackOptions,
        shape: tuple[int, ...],
    ) -> bool:
        if len(shape) != 3:
            return False
        if not options.preserve_channels_last_color:
            return True
        return not ColorImageShapeRole.matches_slice(data)

@writer_for(TiffStackOptions, MaterializationFormat.TIFF_STACK)
def _write_tiff_stack(
    data: MaterializationValue,
    options: TiffStackOptions,
    ctx: MaterializationContext,
) -> list[Output]:
    materialization_input = MaterializationInput.from_value(data, options)
    data = materialization_input.data
    paths = ctx.paths(options)
    base_name = paths.name

    if materialization_is_empty(data):
        summary_path = paths.with_suffix(options.summary_suffix)
        return [Output(path=summary_path, content=options.empty_summary)]

    slices = TiffStackSlicePayloadAuthority.slices(data, options)

    slice_metadata = _TIFF_STACK_SLICE_METADATA.metadata_for_slices(
        materialization_input,
        len(slices),
    )

    outs: list[Output] = []
    for i, arr in enumerate(slices):
        filename = str(
            paths.parent / f"{base_name}{options.slice_pattern.format(index=i)}"
        )
        out_arr = TiffArrayAuthority.output_array(arr, options)
        outs.append(
            Output.from_metadata(
                path=filename,
                content=out_arr,
                metadata=slice_metadata[i],
            )
        )

    summary_path = paths.with_suffix(options.summary_suffix)
    first = None
    if slices:
        first = slices[0]
    summary_content = (
        f"Images saved: {len(slices)} files\n"
        f"Base filename pattern: {base_name}{options.slice_pattern}\n"
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
class TiffStackSliceMetadataRequest:
    """Input facts needed to align metadata to emitted TIFF slices."""

    materialization_input: MaterializationInput
    slice_count: int

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
        return tuple(item.metadata for item in request.materialization_input.items)

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
    ) -> tuple[ImagePayloadMetadata, ...]:
        request = TiffStackSliceMetadataRequest(
            materialization_input=materialization_input,
            slice_count=slice_count,
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

    def __init__(
        self,
        *outputs: FileOutputOptions | Sequence[FileOutputOptions],
        allowed_backends: list[str] | None = None,
        primary: int = 0,
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

        self.outputs = tuple(outputs)
        self.allowed_backends = allowed_backends
        self.primary = primary

    @classmethod
    def __objectstate_rebuild__(
        cls,
        *,
        outputs: tuple[FileOutputOptions, ...],
        allowed_backends: list[str] | None = None,
        primary: int = 0,
    ) -> "MaterializationSpec":
        # Rebuild via the normal constructor to keep validation behavior.
        return cls(*outputs, allowed_backends=allowed_backends, primary=primary)

    def tabular_field_names(self) -> tuple[str, ...]:
        """Return declared tabular field names from the primary writer first."""
        return tabular_field_names_from_options(self.outputs, primary=self.primary)

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
    write_mode: WriteMode = WriteMode.OVERWRITE,
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
        write_mode=write_mode,
    )

    primary_path = ""

    for i, opt in enumerate(spec.outputs):
        writer = _WRITERS_BY_OPTIONS[opt.__class__]
        outs = writer.write(data, opt, ctx)
        for out in outs:
            ctx.saver.save(out)
        if i == spec.primary:
            primary_path = writer.primary_path(outs)

    return primary_path
