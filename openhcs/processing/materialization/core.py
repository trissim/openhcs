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
from dataclasses import dataclass, is_dataclass
from functools import singledispatch
from pathlib import Path
from typing import ClassVar, TYPE_CHECKING

import pandas as pd
import numpy as np
from metaclass_registry import AutoRegisterMeta

from openhcs.core.artifacts import ArtifactMaterializationPayload
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
    ImagePayloadMetadata,
    SourceComponentMetadata,
    image_payload_data,
    image_payload_metadata,
    runtime_array_operand,
)
from openhcs.processing.materialization.utils import (
    discover_array_fields,
    expand_array_field,
    extract_fields,
)

if TYPE_CHECKING:
    from polystore.filemanager import FileManager
    from openhcs.core.context.processing_context import ProcessingContext

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
    | np.ndarray
    | pd.DataFrame
    | pd.Series
    | None
)
BackendKwargs = dict[str, dict]
TabularRow = dict
TabularRows = list[TabularRow]


@dataclass(frozen=True, slots=True)
class BackendKwargsAbsent:
    """Default backend kwargs variant for callers that do not provide overrides."""


BACKEND_KWARGS_ABSENT = BackendKwargsAbsent()
BackendKwargsInput = BackendKwargs | BackendKwargsAbsent
PrimaryPathSelector = Callable[[list["Output"]], str]


class PriorityRegisteredPolicy(ABC, metaclass=AutoRegisterMeta):
    """Registered policy ordered by an integer priority."""

    __registry_key__ = "priority"
    __skip_if_no_key__ = True
    priority: ClassVar[int | None] = None
    __registry__: ClassVar[dict[int, type["PriorityRegisteredPolicy"]]] = {}

    @classmethod
    def ordered_policies(cls) -> tuple["PriorityRegisteredPolicy", ...]:
        return tuple(
            policy_type()
            for _priority, policy_type in sorted(cls.__registry__.items())
        )


class AlwaysMatchesPolicy(ABC):
    """Fallback policy marker for authorities that require a terminal policy."""

    def matches(self, _request) -> bool:
        return True


@dataclass(frozen=True, slots=True)
class FirstMatchingPolicyAuthority:
    """Dispatch one request to the first matching registered policy."""

    policies: tuple
    family_name: str

    def matching_policy(self, request):
        for policy in self.policies:
            if policy.matches(request):
                return policy
        raise RuntimeError(f"{self.family_name} has no fallback policy.")


@dataclass(frozen=True)
class Output:
    path: str
    content: MaterializationValue
    metadata: ImagePayloadMetadata | None = None


class SourceSegmentAuthority:
    """Resolve one source selector segment against mappings or named fields."""

    @staticmethod
    def resolve(value: MaterializationValue, segment: str) -> MaterializationValue:
        if isinstance(value, dict):
            return value[segment]
        return operator.attrgetter(segment)(value)


def _resolve_source(value: MaterializationValue, source: str | None) -> MaterializationValue:
    if not source:
        return value

    cur = value
    for part in source.split("."):
        cur = SourceSegmentAuthority.resolve(cur, part)
    return cur


def _as_sequence(value: MaterializationValue) -> list:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


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


@materialization_is_empty.register(list)
@materialization_is_empty.register(tuple)
@materialization_is_empty.register(dict)
@materialization_is_empty.register(str)
@materialization_is_empty.register(bytes)
def sized_materialization_is_empty(value: Sized) -> bool:
    return len(value) == 0


@dataclass(frozen=True, slots=True)
class MaterializationInputItem:
    """One materialization payload item with semantic metadata preserved."""

    data: MaterializationValue
    metadata: ImagePayloadMetadata

    @classmethod
    def from_payload(cls, payload: MaterializationValue) -> "MaterializationInputItem":
        return cls(
            data=runtime_array_operand(image_payload_data(payload)),
            metadata=image_payload_metadata(payload),
        )


@dataclass(frozen=True, slots=True)
class ROIMaterializationTarget:
    """One ROI archive output with the label planes and metadata it owns."""

    roi_path: str
    items: tuple[MaterializationInputItem, ...]
    stream_metadata: ImagePayloadMetadata | None = None


ROIMaterializationTargets = tuple[ROIMaterializationTarget, ...]


@dataclass(frozen=True, slots=True)
class ROIMaterializationTargetRequest:
    """Input facts needed to derive ROI archive targets."""

    materialization_input: "MaterializationInput"
    paths: "PathHelper"
    options: ROIOptions


@dataclass(frozen=True, slots=True)
class ROIPathRequest:
    """Input facts needed to name one addressable ROI archive."""

    paths: "PathHelper"
    options: ROIOptions
    metadata: ImagePayloadMetadata
    reference_source_stem: str | None
    fallback_index: int


@dataclass(frozen=True, slots=True)
class MaterializationInput:
    """Normalized writer input with data and metadata from one authority."""

    items: tuple[MaterializationInputItem, ...]
    sequence_type: type | None = None

    @classmethod
    def from_value(
        cls,
        value: MaterializationValue,
        options: SourceOptions,
    ) -> "MaterializationInput":
        payload = _resolve_source(value, options.source)
        sequence_type = None
        if isinstance(payload, (list, tuple)):
            sequence_type = payload.__class__
        return cls(
            items=tuple(
                MaterializationInputItem.from_payload(item)
                for item in _as_sequence(payload)
            ),
            sequence_type=sequence_type,
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


class BackendSaver:
    """Centralized multi-backend saving."""

    def __init__(
        self,
        backends: list[str],
        filemanager: "FileManager",
        backend_kwargs: BackendKwargs,
        *,
        write_mode: WriteMode,
    ):
        self.backends = backends
        self.filemanager = filemanager
        self.backend_kwargs = backend_kwargs
        self.write_mode = write_mode

    def save(
        self,
        content: MaterializationValue,
        path: str,
        *,
        metadata: ImagePayloadMetadata | None = None,
    ) -> None:
        for backend in self.backends:
            self._prepare_path(backend, path)
            if backend in self.backend_kwargs:
                kwargs = self.backend_kwargs[backend]
            else:
                kwargs = {}
            if (
                metadata is not None
                and metadata.source_component_metadata is not None
                and "component_metadata" in kwargs
            ):
                kwargs = {
                    **kwargs,
                    "component_metadata": dict(metadata.source_component_metadata),
                }
            self.filemanager.save(content, path, backend, **kwargs)

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
    backend_kwargs: BackendKwargs
    filemanager: "FileManager"
    extra_inputs: dict
    context: "ProcessingContext | None" = None
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
    def normalize(backend_kwargs: BackendKwargsInput) -> BackendKwargs:
        if isinstance(backend_kwargs, BackendKwargsAbsent):
            return {}
        return backend_kwargs


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


def _wants_tabular(options: TabularExtractionOptions) -> bool:
    return bool(
        options.fields
        or options.row_field
        or options.row_unpacker
        or options.row_columns
    )


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
    return data.to_dict(orient="records")


@tabular_rows_from_payload.register(pd.Series)
def series_tabular_rows(
    data: pd.Series,
    options: TabularExtractionOptions,
) -> TabularRows:
    del options
    return [data.to_dict()]


class TabularRowsAuthority:
    """Build canonical tabular rows for CSV and JSON materialization."""

    SLICE_INDEX_FIELD: ClassVar[str] = "slice_index"

    @classmethod
    def build(
        cls,
        data: MaterializationValue,
        options: TabularExtractionOptions,
    ) -> TabularRows:
        direct_rows = tabular_rows_from_payload(data, options)
        if direct_rows is not None:
            return direct_rows

        rows: TabularRows = []
        for idx, item in enumerate(_as_sequence(data)):
            base_row = cls.base_row(item, idx, options)

            if options.row_unpacker:
                for exp_row in options.row_unpacker(item):
                    rows.append({**base_row, **exp_row})
                continue

            if options.row_field:
                array_data = FieldValueAuthority.value(item, options.row_field)
                rows.extend(expand_array_field(array_data, base_row, options.row_columns))
                continue

            if array_fields := discover_array_fields(item):
                primary_field = array_fields[0]
                array_data = FieldValueAuthority.value(item, primary_field)
                rows.extend(expand_array_field(array_data, base_row, {}))
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
        base_row = extract_fields(item, field_names)
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
        field_values = extract_fields(item, [field_name])
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
        fields = extract_fields(row, list(fieldnames))
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
    # Make common OpenHCS outputs JSON-friendly:
    # - dataclass -> dict
    # - list[dataclass] -> list[dict]
    # - list[dict] unchanged
    # If the options look tabular, use the canonical tabular builder.
    payload: MaterializationValue
    if _wants_tabular(options):
        payload = TabularRowsAuthority.build(data, options)
    else:
        seq = _as_sequence(data)
        if len(seq) == 1 and seq[0] is data:
            # single element (non-list input)
            payload = extract_fields(data, options.fields)
        else:
            payload = [extract_fields(item, options.fields) for item in seq]

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
    return value.to_dict(orient="records")


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


def _validate_text(payload: MaterializationValue, options: TextOptions) -> None:
    if not isinstance(payload, str):
        raise TypeError(
            f"TextOptions expects a str payload, got {payload.__class__.__name__}"
        )


register_single_file_writer(
    TextOptions,
    MaterializationFormat.TEXT,
    render=lambda payload, _options: payload,
    validate_payload=_validate_text,
)


def _roi_primary_path(outs: list[Output]) -> str:
    for out in outs:
        if out.path.endswith(PathSuffixes.ROI_ZIP):
            return out.path
    if outs:
        return outs[0].path
    return ""


class ROIMaterializationTargetPolicy(PriorityRegisteredPolicy):
    """Registered target selection policy for ROI materialization."""

    __registry__: ClassVar[dict[int, type["ROIMaterializationTargetPolicy"]]] = {}

    @abstractmethod
    def matches(self, request: ROIMaterializationTargetRequest) -> bool:
        """Return whether this policy owns the target request."""

    @abstractmethod
    def targets(
        self,
        request: ROIMaterializationTargetRequest,
    ) -> ROIMaterializationTargets:
        """Return ROI archive targets for the request."""


class ROIPathPolicy(PriorityRegisteredPolicy):
    """Registered path naming policy for addressable ROI archives."""

    __registry__: ClassVar[dict[int, type["ROIPathPolicy"]]] = {}

    @abstractmethod
    def matches(self, request: ROIPathRequest) -> bool:
        """Return whether this policy owns the path request."""

    @abstractmethod
    def path(self, request: ROIPathRequest) -> str:
        """Return the concrete ROI archive path."""


class SourceStemPrefixROIPathPolicy(ROIPathPolicy):
    """Replace the source stem prefix when the base path was built from it."""

    priority = 0

    def matches(self, request: ROIPathRequest) -> bool:
        source_stem = ROIPathAuthority.source_stem(request.metadata)
        reference_source_stem = request.reference_source_stem
        return (
            source_stem is not None
            and reference_source_stem is not None
            and request.paths.name.startswith(reference_source_stem)
        )

    def path(self, request: ROIPathRequest) -> str:
        source_stem = ROIPathAuthority.required_source_stem(request.metadata)
        reference_source_stem = ROIPathAuthority.required_reference_source_stem(
            request
        )
        suffix = request.paths.name[len(reference_source_stem):]
        return str(
            request.paths.parent
            / f"{source_stem}{suffix}{request.options.roi_suffix}"
        )


class SourceStemSuffixROIPathPolicy(ROIPathPolicy):
    """Append the source stem when the base path has unrelated naming."""

    priority = 1

    def matches(self, request: ROIPathRequest) -> bool:
        return ROIPathAuthority.source_stem(request.metadata) is not None

    def path(self, request: ROIPathRequest) -> str:
        source_stem = ROIPathAuthority.required_source_stem(request.metadata)
        return str(
            request.paths.parent
            / f"{request.paths.name}_{source_stem}{request.options.roi_suffix}"
        )


class PlaneIndexROIPathPolicy(AlwaysMatchesPolicy, ROIPathPolicy):
    """Fallback path for addressable planes that do not carry a source path."""

    priority = 2

    def path(self, request: ROIPathRequest) -> str:
        return str(
            request.paths.parent
            / (
                f"{request.paths.name}_plane_{request.fallback_index:03d}"
                f"{request.options.roi_suffix}"
            )
        )


class ROIPathAuthority:
    """Name addressable ROI archives from source metadata."""

    def __init__(self) -> None:
        self._authority = FirstMatchingPolicyAuthority(
            ROIPathPolicy.ordered_policies(),
            "ROIPathAuthority",
        )

    def path(self, request: ROIPathRequest) -> str:
        policy = self._authority.matching_policy(request)
        path = policy.path(request)
        if not path:
            raise RuntimeError(
                f"{policy.__class__.__name__} produced an empty ROI path."
            )
        return path

    @staticmethod
    def source_stem(metadata: ImagePayloadMetadata) -> str | None:
        if metadata.source_path:
            return Path(metadata.source_path).stem
        return None

    @classmethod
    def required_source_stem(cls, metadata: ImagePayloadMetadata) -> str:
        source_stem = cls.source_stem(metadata)
        if source_stem is None:
            raise ValueError("ROI source-stem path policy requires metadata.source_path.")
        return source_stem

    @staticmethod
    def required_reference_source_stem(request: ROIPathRequest) -> str:
        if request.reference_source_stem is None:
            raise ValueError("ROI source-stem prefix policy requires a reference stem.")
        return request.reference_source_stem


_ROI_PATH_AUTHORITY = ROIPathAuthority()


class StackedROIMaterializationTargetPolicy(ROIMaterializationTargetPolicy):
    """Split one stacked label payload when each leading plane has metadata."""

    priority = 0

    def matches(self, request: ROIMaterializationTargetRequest) -> bool:
        if len(request.materialization_input.items) != 1:
            return False
        return self._plane_count(request.materialization_input.items[0]) > 1

    def targets(
        self,
        request: ROIMaterializationTargetRequest,
    ) -> ROIMaterializationTargets:
        item = request.materialization_input.items[0]
        plane_count = self._plane_count(item)
        reference_metadata = item.metadata.for_channel(0)
        reference_source_stem = ROIPathAuthority.source_stem(reference_metadata)
        targets: list[ROIMaterializationTarget] = []
        for plane_index in range(plane_count):
            plane_metadata = item.metadata.for_channel(plane_index)
            self.require_addressable_plane_metadata(plane_metadata, plane_index)
            targets.append(
                ROIMaterializationTarget(
                    roi_path=_ROI_PATH_AUTHORITY.path(
                        ROIPathRequest(
                            paths=request.paths,
                            options=request.options,
                            metadata=plane_metadata,
                            reference_source_stem=reference_source_stem,
                            fallback_index=plane_index,
                        )
                    ),
                    items=(
                        MaterializationInputItem(
                            data=item.data[plane_index],
                            metadata=plane_metadata,
                        ),
                    ),
                    stream_metadata=plane_metadata,
                )
            )
        return tuple(targets)

    @staticmethod
    def _plane_count(item: MaterializationInputItem) -> int:
        shape = np.shape(item.data)
        if len(shape) < 3:
            return 0
        metadata_count = max(
            len(item.metadata.channel_source_paths),
            len(item.metadata.channel_source_component_metadata),
        )
        if metadata_count <= 1:
            return 0
        if int(shape[0]) != metadata_count:
            return 0
        return metadata_count

    @staticmethod
    def require_addressable_plane_metadata(
        metadata: ImagePayloadMetadata,
        plane_index: int,
    ) -> None:
        if (
            metadata.source_path is not None
            or metadata.source_component_metadata is not None
        ):
            return
        raise ValueError(
            "Addressable ROI label stacks require per-plane source identity; "
            f"plane {plane_index} has neither source_path nor "
            "source_component_metadata."
        )


class PreslicedROIMaterializationTargetPolicy(ROIMaterializationTargetPolicy):
    """Use one archive per input item when items already carry source paths."""

    priority = 1

    def matches(self, request: ROIMaterializationTargetRequest) -> bool:
        items = request.materialization_input.items
        return len(items) > 1 and all(item.metadata.source_path for item in items)

    def targets(
        self,
        request: ROIMaterializationTargetRequest,
    ) -> ROIMaterializationTargets:
        items = request.materialization_input.items
        reference_source_stem = ROIPathAuthority.source_stem(items[0].metadata)
        return tuple(
            ROIMaterializationTarget(
                roi_path=_ROI_PATH_AUTHORITY.path(
                    ROIPathRequest(
                        paths=request.paths,
                        options=request.options,
                        metadata=item.metadata,
                        reference_source_stem=reference_source_stem,
                        fallback_index=index,
                    )
                ),
                items=(item,),
                stream_metadata=item.metadata,
            )
            for index, item in enumerate(items)
        )


class CombinedROIMaterializationTargetPolicy(
    AlwaysMatchesPolicy,
    ROIMaterializationTargetPolicy,
):
    """Historical single archive behavior for unaddressed ROI payloads."""

    priority = 2

    def targets(
        self,
        request: ROIMaterializationTargetRequest,
    ) -> ROIMaterializationTargets:
        return (
            ROIMaterializationTarget(
                roi_path=request.paths.with_suffix(request.options.roi_suffix),
                items=request.materialization_input.items,
            ),
        )


class ROIMaterializationTargetAuthority:
    """Resolve ROI archive targets through registered cardinality policies."""

    def __init__(self) -> None:
        self._authority = FirstMatchingPolicyAuthority(
            ROIMaterializationTargetPolicy.ordered_policies(),
            "ROIMaterializationTargetAuthority",
        )

    def targets(
        self,
        request: ROIMaterializationTargetRequest,
    ) -> ROIMaterializationTargets:
        targets = self._authority.matching_policy(request).targets(request)
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
                    "path": target.roi_path,
                    "component_metadata": ROIMaterializationTargetAuthority._metadata_payload(
                        target
                    ),
                }
                for target in targets
            ],
        )

    @staticmethod
    def _metadata_payload(
        target: ROIMaterializationTarget,
    ) -> SourceComponentMetadata | None:
        metadata = target.stream_metadata
        if metadata is None or metadata.source_component_metadata is None:
            return None
        return dict(metadata.source_component_metadata)


_ROI_MATERIALIZATION_TARGETS = ROIMaterializationTargetAuthority()


@writer_for(ROIOptions, MaterializationFormat.ROI_ZIP, primary_path=_roi_primary_path)
def _write_roi_zip(
    data: MaterializationValue,
    options: ROIOptions,
    ctx: MaterializationContext,
) -> list[Output]:
    from polystore.roi import extract_rois_from_labeled_mask

    materialization_input = MaterializationInput.from_value(data, options)
    payload = materialization_input.data
    paths = ctx.paths(options)
    summary_path = paths.with_suffix(options.summary_suffix)

    if materialization_is_empty(payload):
        return [Output(path=summary_path, content="No segmentation masks generated (empty data)\n")]

    targets = _ROI_MATERIALIZATION_TARGETS.targets(
        ROIMaterializationTargetRequest(
            materialization_input=materialization_input,
            paths=paths,
            options=options,
        )
    )
    outs: list[Output] = []
    total_roi_count = 0
    roi_paths: list[str] = []
    for target in targets:
        target_rois: list = []
        for item in target.items:
            rois = extract_rois_from_labeled_mask(
                item.data,
                min_area=options.min_area,
                extract_contours=options.extract_contours,
                spatial_origin_yx=item.metadata.spatial_origin_yx,
                source_spatial_shape_yx=item.metadata.source_spatial_shape_yx,
            )
            target_rois.extend(rois)

        total_roi_count += len(target_rois)
        if target_rois:
            roi_paths.append(target.roi_path)
            outs.append(
                Output(
                    path=target.roi_path,
                    content=target_rois,
                    metadata=target.stream_metadata,
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

    if isinstance(data, (list, tuple)):
        slices = list(data)
    else:
        shape = np.shape(data)
        ndim = len(shape)
        if (
            ndim == 3
            and not (
                options.preserve_channels_last_color
                and _is_channels_last_color_image(data)
            )
        ):
            slices = [data[i] for i in range(shape[0])]  # type: ignore[index]
        else:
            slices = [data]

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
        outs.append(Output(path=filename, content=out_arr, metadata=slice_metadata[i]))

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


class TiffStackSliceMetadataPolicy(PriorityRegisteredPolicy):
    """One cardinality policy for TIFF slice metadata projection."""

    __registry__: ClassVar[dict[int, type["TiffStackSliceMetadataPolicy"]]] = {}

    @abstractmethod
    def matches(self, request: TiffStackSliceMetadataRequest) -> bool:
        """Return whether this policy owns the request cardinality."""

    @abstractmethod
    def metadata_for_slices(
        self,
        request: TiffStackSliceMetadataRequest,
    ) -> tuple[ImagePayloadMetadata, ...]:
        """Return one metadata record for each emitted TIFF slice."""


class EmptyTiffStackSliceMetadataPolicy(TiffStackSliceMetadataPolicy):
    """No emitted slices means no metadata records."""

    priority = 0

    def matches(self, request: TiffStackSliceMetadataRequest) -> bool:
        return request.slice_count <= 0

    def metadata_for_slices(
        self,
        request: TiffStackSliceMetadataRequest,
    ) -> tuple[ImagePayloadMetadata, ...]:
        return ()


class PreslicedTiffStackSliceMetadataPolicy(TiffStackSliceMetadataPolicy):
    """Input items already correspond one-to-one with emitted TIFF slices."""

    priority = 1

    def matches(self, request: TiffStackSliceMetadataRequest) -> bool:
        return len(request.materialization_input.items) == request.slice_count

    def metadata_for_slices(
        self,
        request: TiffStackSliceMetadataRequest,
    ) -> tuple[ImagePayloadMetadata, ...]:
        return tuple(item.metadata for item in request.materialization_input.items)


class SinglePayloadTiffStackSliceMetadataPolicy(TiffStackSliceMetadataPolicy):
    """One stacked payload is being split into emitted TIFF slices."""

    priority = 2

    def matches(self, request: TiffStackSliceMetadataRequest) -> bool:
        return len(request.materialization_input.items) == 1

    def metadata_for_slices(
        self,
        request: TiffStackSliceMetadataRequest,
    ) -> tuple[ImagePayloadMetadata, ...]:
        metadata = request.materialization_input.items[0].metadata
        return tuple(
            metadata.for_channel(index) for index in range(request.slice_count)
        )


class UnknownTiffStackSliceMetadataPolicy(
    AlwaysMatchesPolicy,
    TiffStackSliceMetadataPolicy,
):
    """Ambiguous cardinality gets empty metadata rather than invented identity."""

    priority = 3

    def metadata_for_slices(
        self,
        request: TiffStackSliceMetadataRequest,
    ) -> tuple[ImagePayloadMetadata, ...]:
        return tuple(ImagePayloadMetadata() for _index in range(request.slice_count))


class TiffStackSliceMetadataAuthority:
    """Owns metadata alignment for TIFF slices emitted from runtime payloads."""

    def __init__(self) -> None:
        self._authority = FirstMatchingPolicyAuthority(
            TiffStackSliceMetadataPolicy.ordered_policies(),
            "TiffStackSliceMetadataAuthority",
        )

    def metadata_for_slices(
        self,
        materialization_input: MaterializationInput,
        slice_count: int,
    ) -> tuple[ImagePayloadMetadata, ...]:
        request = TiffStackSliceMetadataRequest(
            materialization_input=materialization_input,
            slice_count=slice_count,
        )
        return self._authority.matching_policy(request).metadata_for_slices(request)


_TIFF_STACK_SLICE_METADATA = TiffStackSliceMetadataAuthority()


def _is_channels_last_color_image(data: MaterializationValue) -> bool:
    """Return whether a 3D array is one channel-last RGB/RGBA image."""
    shape = np.shape(data)
    if len(shape) != 3:
        return False
    return int(shape[-1]) in (3, 4)


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
        write_mode=write_mode,
    )

    primary_path = ""

    for i, opt in enumerate(spec.outputs):
        writer = _WRITERS_BY_OPTIONS[opt.__class__]
        outs = writer.write(data, opt, ctx)
        for out in outs:
            ctx.saver.save(out.content, out.path, metadata=out.metadata)
        if i == spec.primary:
            primary_path = writer.primary_path(outs)

    return primary_path
