"""Renderer-independent debug inspector view models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, fields, is_dataclass
from enum import Enum
import json
from typing import ClassVar, Mapping, cast

from metaclass_registry import AutoRegisterMeta

from openhcs.core.artifacts import ArtifactType
from openhcs.core.debug import DebugArtifactRef, DebugInvocationParameter, DebugSnapshot
from openhcs.core.runtime_stores import (
    RuntimeArtifactAddress,
    RuntimeValueStore,
    StoredRuntimeValue,
)


class DebugViewTableProjection(Enum):
    """Closed table projection family used by debug inspectors."""

    ARTIFACT_REFS = "artifact_refs"
    INVOCATION_PARAMETERS = "invocation_parameters"
    RUNTIME_VALUE_RECORDS = "runtime_value_records"


class DebugViewSectionKind(str, Enum):
    """Closed renderer-independent debug inspector section family."""

    SUMMARY = "summary"
    SOURCES = "sources"
    INPUT_ARTIFACTS = "input_artifacts"
    OUTPUT_ARTIFACTS = "output_artifacts"
    PREVIEW_ARTIFACTS = "preview_artifacts"
    INVOCATION_PARAMETERS = "invocation_parameters"
    RUNTIME_VALUES = "runtime_values"
    MEASUREMENTS = "measurements"
    RELATIONSHIPS = "relationships"
    TIMING = "timing"
    ERROR = "error"


class DebugViewSectionDeclarationBase(ABC, metaclass=AutoRegisterMeta):
    """Nominal declaration for one debug inspector section kind."""

    __registry_key__ = "kind"
    __skip_if_no_key__ = True
    __registry__: ClassVar[
        dict[DebugViewSectionKind, type["DebugViewSectionDeclarationBase"]]
    ] = {}

    kind: ClassVar[DebugViewSectionKind | None] = None
    title_words: ClassVar[tuple[str, ...]]

    @classmethod
    def require_kind(cls) -> DebugViewSectionKind:
        if cls.kind is None:
            raise TypeError(f"{cls.__name__} does not declare a section kind.")
        return cls.kind

    @classmethod
    def for_kind(
        cls,
        kind: DebugViewSectionKind,
    ) -> type["DebugViewSectionDeclarationBase"]:
        return cls.__registry__[kind]

    @classmethod
    def default_title(cls) -> str:
        return " ".join(cls.title_words)

    @classmethod
    def registered_sections(cls) -> tuple[type["DebugViewSectionDeclarationBase"], ...]:
        return tuple(
            cls.__registry__[kind]
            for kind in DebugViewSectionKind
            if kind in cls.__registry__
        )

    @classmethod
    def section_for_snapshot(cls, snapshot: DebugSnapshot) -> "DebugViewSection | None":
        table = cls.table_for_snapshot(snapshot)
        text = cls.text_for_snapshot(snapshot)
        if table is None and text is None:
            return None
        section = DebugViewSection(
            kind=cls.require_kind(),
            title=cls.default_title(),
            table=table,
            text=text,
        )
        if section.is_empty and not cls.show_empty_snapshot_section():
            return None
        return section

    @classmethod
    def table_for_snapshot(cls, snapshot: DebugSnapshot) -> "DebugViewTable | None":
        del snapshot
        return None

    @classmethod
    def text_for_snapshot(cls, snapshot: DebugSnapshot) -> str | None:
        del snapshot
        return None

    @classmethod
    def show_empty_snapshot_section(cls) -> bool:
        return False

    @classmethod
    @abstractmethod
    def empty_message(cls) -> str:
        """Return section-level empty text."""


class AvailableEmptySection:
    """Section empty-message strategy for absent available values."""

    empty_subject: ClassVar[str]
    empty_verb: ClassVar[str] = "are"

    @classmethod
    def empty_message(cls) -> str:
        return f"No {cls.empty_subject} {cls.empty_verb} available."


class ReportedEmptySection:
    """Section empty-message strategy for absent reported values."""

    empty_subject: ClassVar[str]

    @classmethod
    def empty_message(cls) -> str:
        return f"No {cls.empty_subject} was reported."


class SummaryDebugViewSection(AvailableEmptySection, DebugViewSectionDeclarationBase):
    kind = DebugViewSectionKind.SUMMARY
    title_words = ("Summary",)
    empty_subject = "summary values"

    @classmethod
    def text_for_snapshot(cls, snapshot: DebugSnapshot) -> str:
        values = (
            ("step", snapshot.step_name),
            ("callable", snapshot.callable_name or ""),
            ("axis", snapshot.axis_id or ""),
            ("cursor", snapshot.cursor.invocation_key or ""),
            (
                "timing_seconds",
                (
                    ""
                    if snapshot.timing_seconds is None
                    else f"{snapshot.timing_seconds:.6f}"
                ),
            ),
        )
        return "\n".join(f"{name}: {value}" for name, value in values)

    @classmethod
    def show_empty_snapshot_section(cls) -> bool:
        return True


class SourcesDebugViewSection(AvailableEmptySection, DebugViewSectionDeclarationBase):
    kind = DebugViewSectionKind.SOURCES
    title_words = ("Sources",)
    empty_subject = "source paths"

    @classmethod
    def text_for_snapshot(cls, snapshot: DebugSnapshot) -> str | None:
        if not snapshot.source_paths:
            return None
        return "\n".join(snapshot.source_paths)


class InputArtifactsDebugViewSection(
    AvailableEmptySection,
    DebugViewSectionDeclarationBase,
):
    kind = DebugViewSectionKind.INPUT_ARTIFACTS
    title_words = ("Input", "Artifacts")
    empty_subject = "input artifacts"

    @classmethod
    def table_for_snapshot(cls, snapshot: DebugSnapshot) -> "DebugViewTable | None":
        return artifact_refs_snapshot_table(snapshot.input_artifact_refs)


class OutputArtifactsDebugViewSection(
    AvailableEmptySection,
    DebugViewSectionDeclarationBase,
):
    kind = DebugViewSectionKind.OUTPUT_ARTIFACTS
    title_words = ("Output", "Artifacts")
    empty_subject = "output artifacts"

    @classmethod
    def table_for_snapshot(cls, snapshot: DebugSnapshot) -> "DebugViewTable | None":
        return artifact_refs_snapshot_table(snapshot.output_artifact_refs)


class PreviewArtifactsDebugViewSection(
    AvailableEmptySection,
    DebugViewSectionDeclarationBase,
):
    kind = DebugViewSectionKind.PREVIEW_ARTIFACTS
    title_words = ("Preview", "Artifacts")
    empty_subject = "preview artifacts"

    @classmethod
    def table_for_snapshot(cls, snapshot: DebugSnapshot) -> "DebugViewTable | None":
        return artifact_refs_snapshot_table(snapshot.preview_refs)


class InvocationParametersDebugViewSection(
    AvailableEmptySection,
    DebugViewSectionDeclarationBase,
):
    kind = DebugViewSectionKind.INVOCATION_PARAMETERS
    title_words = ("Invocation", "Parameters")
    empty_subject = "invocation parameters"

    @classmethod
    def table_for_snapshot(cls, snapshot: DebugSnapshot) -> "DebugViewTable | None":
        if not snapshot.invocation_parameters:
            return None
        return DebugViewTable.from_projection(
            DebugViewTableProjection.INVOCATION_PARAMETERS,
            snapshot.invocation_parameters,
        )


class RuntimeValuesDebugViewSection(
    AvailableEmptySection,
    DebugViewSectionDeclarationBase,
):
    kind = DebugViewSectionKind.RUNTIME_VALUES
    title_words = ("Runtime", "Values")
    empty_subject = "runtime values"


class MeasurementsDebugViewSection(
    AvailableEmptySection,
    DebugViewSectionDeclarationBase,
):
    kind = DebugViewSectionKind.MEASUREMENTS
    title_words = ("Measurements",)
    empty_subject = "measurements"

    @classmethod
    def table_for_snapshot(cls, snapshot: DebugSnapshot) -> "DebugViewTable | None":
        return artifact_refs_snapshot_table(snapshot.measurement_refs)


class RelationshipsDebugViewSection(
    AvailableEmptySection,
    DebugViewSectionDeclarationBase,
):
    kind = DebugViewSectionKind.RELATIONSHIPS
    title_words = ("Relationships",)
    empty_subject = "relationships"

    @classmethod
    def table_for_snapshot(cls, snapshot: DebugSnapshot) -> "DebugViewTable | None":
        return artifact_refs_snapshot_table(snapshot.relationship_refs)


class TimingDebugViewSection(AvailableEmptySection, DebugViewSectionDeclarationBase):
    kind = DebugViewSectionKind.TIMING
    title_words = ("Timing",)
    empty_subject = "timing value"
    empty_verb = "is"

    @classmethod
    def text_for_snapshot(cls, snapshot: DebugSnapshot) -> str | None:
        if snapshot.timing_seconds is None:
            return None
        return f"{snapshot.timing_seconds:.6f}s"


class ErrorDebugViewSection(ReportedEmptySection, DebugViewSectionDeclarationBase):
    kind = DebugViewSectionKind.ERROR
    title_words = ("Error",)
    empty_subject = "error"

    @classmethod
    def text_for_snapshot(cls, snapshot: DebugSnapshot) -> str | None:
        return snapshot.exception


class DebugViewTableProjectionDeclarationBase(ABC, metaclass=AutoRegisterMeta):
    """Nominal declaration for one debug table projection."""

    __registry_key__ = "projection"
    __skip_if_no_key__ = True
    __registry__: ClassVar[
        dict[
            DebugViewTableProjection,
            type["DebugViewTableProjectionDeclarationBase"],
        ]
    ] = {}

    projection: ClassVar[DebugViewTableProjection | None] = None
    value_type: ClassVar[type]
    record_type: ClassVar[type | None] = None
    empty_message: ClassVar[str]
    supports_artifact_actions: ClassVar[bool] = False

    @classmethod
    def require_projection(cls) -> DebugViewTableProjection:
        if cls.projection is None:
            raise TypeError(f"{cls.__name__} does not declare a table projection.")
        return cls.projection

    @classmethod
    def for_projection(
        cls,
        projection: DebugViewTableProjection,
    ) -> type["DebugViewTableProjectionDeclarationBase"]:
        return cls.__registry__[projection]

    @classmethod
    def table_for(cls, values: tuple[object, ...]) -> "DebugViewTable":
        records = tuple(cls.table_record(value) for value in values)
        columns = cls.table_columns()
        return DebugViewTable(
            columns=columns,
            rows=tuple(cls.table_row(record, columns) for record in records),
            projection=cls.require_projection(),
            empty_message=cls.empty_message(),
        )

    @classmethod
    def table_columns(cls) -> tuple[str, ...]:
        record_type = cls.require_record_type()
        return dataclass_record_columns(record_type)

    @classmethod
    def require_record_type(cls) -> type:
        return cls.value_type if cls.record_type is None else cls.record_type

    @classmethod
    def table_record(cls, value: object) -> object:
        return cls.require_value(value)

    @classmethod
    def table_row(cls, record: object, columns: tuple[str, ...]) -> tuple[str, ...]:
        return dataclass_record_cells(record, columns)

    @classmethod
    def require_value(cls, value: object) -> object:
        if not isinstance(value, cls.value_type):
            raise TypeError(
                f"{cls.__name__} requires {cls.value_type.__name__}, "
                f"got {type(value).__name__}."
            )
        return value

    @classmethod
    def empty_message(cls) -> str:
        return "No rows are available."


class AvailableEmptyTable:
    """Table empty-message strategy for absent available rows."""

    empty_subject: ClassVar[str]

    @classmethod
    def empty_message(cls) -> str:
        return f"No {cls.empty_subject} are available."


class ArtifactActionDebugTable:
    """Trait for debug tables whose rows identify viewable/exportable artifacts."""

    supports_artifact_actions: ClassVar[bool] = True


class ArtifactRefsDebugViewTable(
    ArtifactActionDebugTable,
    AvailableEmptyTable,
    DebugViewTableProjectionDeclarationBase,
):
    projection = DebugViewTableProjection.ARTIFACT_REFS
    value_type = DebugArtifactRef
    empty_subject = "artifact references"


class InvocationParametersDebugViewTable(
    AvailableEmptyTable,
    DebugViewTableProjectionDeclarationBase,
):
    projection = DebugViewTableProjection.INVOCATION_PARAMETERS
    value_type = DebugInvocationParameter
    empty_subject = "invocation parameters"


class RuntimeValueRecordsDebugViewTable(
    AvailableEmptyTable,
    DebugViewTableProjectionDeclarationBase,
):
    projection = DebugViewTableProjection.RUNTIME_VALUE_RECORDS
    value_type = StoredRuntimeValue
    record_type = RuntimeArtifactAddress
    empty_subject = "runtime values"

    @classmethod
    def table_record(cls, value: object) -> RuntimeArtifactAddress:
        runtime_value = cast(StoredRuntimeValue, cls.require_value(value))
        return RuntimeArtifactAddress.from_record(runtime_value)


@dataclass(frozen=True, slots=True)
class DebugViewTable:
    """Small table-like payload for debug inspectors."""

    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    projection: DebugViewTableProjection | None = None
    empty_message: str | None = None

    @classmethod
    def from_projection(
        cls,
        projection: DebugViewTableProjection,
        values: tuple[object, ...],
    ) -> "DebugViewTable":
        return DebugViewTableProjectionDeclarationBase.for_projection(
            projection
        ).table_for(values)

    @classmethod
    def from_dataclass_records(
        cls,
        *,
        record_type: type,
        records: tuple[object, ...],
        empty_message: str | None = None,
        projection: DebugViewTableProjection | None = None,
    ) -> "DebugViewTable":
        columns = dataclass_record_columns(record_type)
        return cls(
            columns=columns,
            rows=tuple(dataclass_record_cells(record, columns) for record in records),
            projection=projection,
            empty_message=empty_message,
        )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "columns": list(self.columns),
            "rows": [list(row) for row in self.rows],
            "projection": None if self.projection is None else self.projection.value,
            "empty_message": self.empty_message,
        }

    @classmethod
    def from_json_dict(cls, data: Mapping[str, object]) -> "DebugViewTable":
        projection_value = data["projection"]
        return cls(
            columns=tuple(str(column) for column in data["columns"]),
            rows=tuple(
                tuple(str(value) for value in row)
                for row in data["rows"]
            ),
            projection=(
                None
                if projection_value is None
                else DebugViewTableProjection(str(projection_value))
            ),
            empty_message=(
                None
                if data["empty_message"] is None
                else str(data["empty_message"])
            ),
        )


def _debug_table_cell_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, Enum):
        return str(value.value)
    if isinstance(value, type) and issubclass(value, ArtifactType):
        return value.require_value()
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    if isinstance(value, tuple):
        return ", ".join(_debug_table_cell_text(item) for item in value)
    if is_dataclass(value):
        return json.dumps(_debug_table_jsonable(value), sort_keys=True)
    if isinstance(value, Mapping):
        return json.dumps(_debug_table_jsonable(value), sort_keys=True)
    return str(value)


def _debug_table_jsonable(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, type) and issubclass(value, ArtifactType):
        return value.require_value()
    if isinstance(value, tuple):
        return [_debug_table_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_debug_table_jsonable(item) for item in value]
    if isinstance(value, Mapping):
        return {
            str(key): _debug_table_jsonable(item)
            for key, item in value.items()
        }
    if is_dataclass(value):
        return {
            field.name: _debug_table_jsonable(getattr(value, field.name))
            for field in fields(value)
        }
    return str(value)


def dataclass_record_columns(record_type: type) -> tuple[str, ...]:
    if not is_dataclass(record_type):
        raise TypeError(
            "dataclass_record_columns requires a dataclass record type, "
            f"got {record_type!r}."
        )
    return tuple(field.name for field in fields(record_type))


def dataclass_record_cells(record: object, columns: tuple[str, ...]) -> tuple[str, ...]:
    if not is_dataclass(record):
        raise TypeError(
            "dataclass_record_cells requires dataclass table records, "
            f"got {type(record).__name__}."
        )
    mapping = asdict(record)
    return tuple(_debug_table_cell_text(mapping[column]) for column in columns)


def artifact_refs_snapshot_table(
    refs: tuple[DebugArtifactRef, ...],
) -> DebugViewTable | None:
    if not refs:
        return None
    return DebugViewTable.from_projection(DebugViewTableProjection.ARTIFACT_REFS, refs)


@dataclass(frozen=True, slots=True)
class DebugViewSection:
    """One named debug view section."""

    kind: DebugViewSectionKind
    title: str
    table: DebugViewTable | None = None
    text: str | None = None

    @property
    def is_empty(self) -> bool:
        return (
            (self.table is None or not self.table.rows)
            and not self.text
        )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "title": self.title,
            "table": None if self.table is None else self.table.to_json_dict(),
            "text": self.text,
        }

    @classmethod
    def from_json_dict(cls, data: Mapping[str, object]) -> "DebugViewSection":
        table = data["table"]
        if table is not None and not isinstance(table, Mapping):
            raise TypeError("DebugViewSection.table must be a mapping or None.")
        return cls(
            kind=DebugViewSectionKind(str(data["kind"])),
            title=str(data["title"]),
            table=(
                None
                if table is None
                else DebugViewTable.from_json_dict(table)
            ),
            text=None if data["text"] is None else str(data["text"]),
        )


@dataclass(frozen=True, slots=True)
class DebugViewModel:
    """Renderer-independent debug inspector model."""

    title: str
    sections: tuple[DebugViewSection, ...]

    @classmethod
    def from_debug_snapshot(
        cls,
        snapshot: DebugSnapshot,
        *,
        title: str | None = None,
    ) -> "DebugViewModel":
        return cls(
            title=title or snapshot.callable_name or snapshot.step_name,
            sections=tuple(
                section
                for declaration in DebugViewSectionDeclarationBase.registered_sections()
                for section in (declaration.section_for_snapshot(snapshot),)
                if section is not None
            ),
        )

    @classmethod
    def from_runtime_value_store(
        cls,
        store: RuntimeValueStore,
        *,
        title: str = "Runtime Values",
    ) -> "DebugViewModel":
        if not isinstance(store, RuntimeValueStore):
            raise TypeError(
                "DebugViewModel.from_runtime_value_store requires RuntimeValueStore, "
                f"got {type(store).__name__}."
            )
        section_declaration = DebugViewSectionDeclarationBase.for_kind(
            DebugViewSectionKind.RUNTIME_VALUES
        )
        return cls(
            title=title,
            sections=(
                DebugViewSection(
                    kind=DebugViewSectionKind.RUNTIME_VALUES,
                    title=section_declaration.default_title(),
                    table=DebugViewTable.from_projection(
                        DebugViewTableProjection.RUNTIME_VALUE_RECORDS,
                        store.values(),
                    ),
                ),
            ),
        )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "title": self.title,
            "sections": [section.to_json_dict() for section in self.sections],
        }

    @classmethod
    def from_json_dict(cls, data: Mapping[str, object]) -> "DebugViewModel":
        return cls(
            title=str(data["title"]),
            sections=tuple(
                DebugViewSection.from_json_dict(section)
                for section in data["sections"]
            ),
        )


def is_debug_view_export(name: str, value: object) -> bool:
    return (
        isinstance(value, type)
        and value.__module__ == __name__
        and not name.startswith("_")
    )


__all__ = tuple(
    name
    for name, value in globals().items()
    if is_debug_view_export(name, value)
)
