"""CellProfiler database table and column projection rules."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from hashlib import md5
import re
from types import MappingProxyType
from typing import Any, ClassVar
from openhcs.core.runtime_measurements import MeasurementScope, MeasurementSubject, ObjectCoreMeasurementFeature
from openhcs.core.runtime_tabular_values import FieldSpec
from openhcs.interop.cellprofiler.source_metadata import (
    CellProfilerSourceMetadataField,
)


class CellProfilerExperimentProjectionField(Enum):
    """Pipeline-level fields CellProfiler declares for every database export."""

    PIPELINE = ("Pipeline_Pipeline", bytes, True)
    VERSION = ("CellProfiler_Version", str, True)
    RUN_TIMESTAMP = ("Run_Timestamp", str, True)
    MODIFICATION_TIMESTAMP = ("Modification_Timestamp", str, True)

    def __init__(
        self,
        field_name: str,
        dtype: type[object],
        volatile_value: bool,
    ):
        self.field_name = field_name
        self.dtype = dtype
        self.volatile_value = volatile_value

    @property
    def field_spec(self) -> FieldSpec:
        return FieldSpec(self.field_name, self.dtype, required=False)


class CellProfilerImageStructuralFieldFamily(Enum):
    """Exact CellProfiler image-table fields that are not measurements."""

    CHANNEL = "Channel"
    EXECUTION_TIME = "ExecutionTime"
    FILE_NAME = "FileName"
    FRAME = "Frame"
    GROUP = "Group"
    HEIGHT = "Height"
    IMAGE_QUALITY_SCALING = "ImageQuality_Scaling"
    IMAGE_SET = "ImageSet"
    MD5_DIGEST = "MD5Digest"
    MODULE_ERROR = "ModuleError"
    PATH_NAME = "PathName"
    SCALING = "Scaling"
    SERIES = "Series"
    URL = "URL"
    WIDTH = "Width"

    @property
    def field_name(self) -> str:
        return str(self.value)

    @property
    def field_prefix(self) -> str:
        return f"{self.field_name}_"

    def qualified_name(self, qualifier: str) -> str:
        return f"{self.field_prefix}{qualifier}"


class CellProfilerSourceImageProjectionField(Enum):
    """Per-channel source fields declared by CellProfiler database export."""

    FRAME = (CellProfilerImageStructuralFieldFamily.FRAME, int)
    HEIGHT = (CellProfilerImageStructuralFieldFamily.HEIGHT, int)
    MD5_DIGEST = (CellProfilerImageStructuralFieldFamily.MD5_DIGEST, str)
    SCALING = (CellProfilerImageStructuralFieldFamily.SCALING, float)
    SERIES = (CellProfilerImageStructuralFieldFamily.SERIES, int)
    URL = (CellProfilerImageStructuralFieldFamily.URL, str)
    WIDTH = (CellProfilerImageStructuralFieldFamily.WIDTH, int)

    def __init__(
        self,
        family: CellProfilerImageStructuralFieldFamily,
        dtype: type[object],
    ):
        self.family = family
        self.field_name = family.field_name
        self.dtype = dtype

    @property
    def field_spec(self) -> FieldSpec:
        return FieldSpec(self.field_name, self.dtype, required=False)


class CellProfilerObjectCoreMeasurementFeature(str, Enum):
    """CellProfiler names for generic core object-measurement features."""

    OBJECT_NUMBER = "Number_Object_Number"
    CENTER_X = "Location_Center_X"
    CENTER_Y = "Location_Center_Y"
    CENTER_Z = "Location_Center_Z"

    @property
    def core_feature(self) -> ObjectCoreMeasurementFeature:
        return ObjectCoreMeasurementFeature[self.name]


class CellProfilerImageAggregateStatistic(str, Enum):
    """CellProfiler names for object statistics projected onto image rows."""

    MEAN = "Mean"
    MEDIAN = "Median"
    STANDARD_DEVIATION = "StDev"

    @property
    def field_prefix(self) -> str:
        return f"{self.value}_"

    @classmethod
    def for_field_name(
        cls,
        field_name: str,
    ) -> "CellProfilerImageAggregateStatistic | None":
        """Return the declared aggregate statistic owning an external field."""

        return next(
            (
                statistic
                for statistic in cls
                if field_name.startswith(statistic.field_prefix)
            ),
            None,
        )


@dataclass(frozen=True, slots=True)
class CellProfilerProjectedTable:
    """One CP-local external table with exact raw field names."""

    table_name: str
    rows: tuple[Mapping[str, Any], ...]
    columns: tuple[FieldSpec, ...]
    subject: MeasurementSubject | None = None

    def __post_init__(self) -> None:
        if not self.table_name:
            raise ValueError("CellProfilerProjectedTable.table_name cannot be empty.")
        declared_columns = tuple(self.columns)
        columns = FieldSpec.merge_exact(
            (declared_columns,),
            context=f"CellProfiler projected table {self.table_name!r} columns",
        )
        if len(columns) != len(declared_columns):
            duplicate_names = tuple(
                dict.fromkeys(
                    field_spec.name
                    for field_spec in declared_columns
                    if sum(
                        candidate.name == field_spec.name
                        for candidate in declared_columns
                    )
                    > 1
                )
            )
            raise ValueError(
                f"CellProfiler projected table {self.table_name!r} declares "
                f"duplicate fields {duplicate_names!r}."
            )
        declared_names = frozenset(field_spec.name for field_spec in columns)
        rows = tuple(self.rows)
        for row in rows:
            invalid_keys = tuple(key for key in row if not isinstance(key, str))
            if invalid_keys:
                raise TypeError(
                    f"CellProfiler projected table {self.table_name!r} row keys "
                    f"must be field-name strings, got {invalid_keys!r}."
                )
            undeclared = tuple(name for name in row if name not in declared_names)
            if undeclared:
                raise ValueError(
                    f"CellProfiler projected table {self.table_name!r} rows "
                    f"contain undeclared fields {undeclared!r}."
                )
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "columns", columns)


@dataclass(frozen=True, slots=True)
class CellProfilerColumnNameMapping:
    """Deterministic CellProfiler mapping for one complete database schema."""

    maximum_length: int
    source_names: tuple[str, ...]
    _mapped_names: Mapping[str, str] = field(init=False, repr=False, compare=False)

    _VALID_NAME: ClassVar[re.Pattern[str]] = re.compile(r"^[0-9A-Za-z_$]+$")
    _REMOVAL_GROUPS: ClassVar[tuple[str, ...]] = (
        "aeiou",
        "bcdfghjklmnpqrstvwxyz",
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    )

    def __post_init__(self) -> None:
        if isinstance(self.maximum_length, bool) or not isinstance(
            self.maximum_length,
            int,
        ):
            raise TypeError(
                "CellProfilerColumnNameMapping.maximum_length must be an integer."
            )
        if not 10 <= self.maximum_length <= 64:
            raise ValueError(
                "CellProfilerColumnNameMapping.maximum_length must be between "
                "10 and 64."
            )
        source_names = tuple(dict.fromkeys(self.source_names))
        if any(not isinstance(name, str) or not name for name in source_names):
            raise ValueError(
                "CellProfilerColumnNameMapping.source_names requires non-empty "
                "strings."
            )
        object.__setattr__(self, "source_names", source_names)
        object.__setattr__(
            self,
            "_mapped_names",
            MappingProxyType(self._map_source_names(source_names)),
        )

    def render(self, source_name: str) -> str:
        """Return the mapped external name for a registered schema field."""

        try:
            return self._mapped_names[source_name]
        except KeyError as exc:
            raise ValueError(
                "CellProfiler database rendering encountered a field absent "
                f"from the complete schema mapping: {source_name!r}."
            ) from exc

    def source(self, external_name: str) -> str:
        """Return the unique schema field rendered as ``external_name``."""

        matches = tuple(
            source_name
            for source_name, rendered_name in self._mapped_names.items()
            if rendered_name == external_name
        )
        if len(matches) != 1:
            raise ValueError(
                "CellProfiler database projection cannot invert external field "
                f"{external_name!r}; matching schema fields={matches!r}."
            )
        return matches[0]

    def _map_source_names(self, source_names: Sequence[str]) -> dict[str, str]:
        mapped_names = {name: name for name in source_names}
        source_by_external = {name: name for name in source_names}
        problem_names = tuple(
            name
            for name in sorted(source_names)
            if len(name) > self.maximum_length
            or self._VALID_NAME.fullmatch(name) is None
        )
        for original_name in problem_names:
            source_name = source_by_external[original_name]
            external_name = original_name
            if self._VALID_NAME.fullmatch(external_name) is None:
                external_name = re.sub(r"[^0-9A-Za-z_$]", "_", external_name)
                if external_name in source_by_external:
                    suffix = 1
                    while f"{external_name}{suffix}" in source_by_external:
                        suffix += 1
                    external_name = f"{external_name}{suffix}"

            starting_name = external_name
            starting_positions = tuple(
                position for position in (external_name.find("_"), 0) if position != -1
            )
            for starting_position in starting_positions:
                characters_to_remove = len(external_name) - self.maximum_length
                removed = 0
                if characters_to_remove > 0:
                    for removal_group in self._REMOVAL_GROUPS:
                        for index in range(
                            len(external_name) - 1,
                            starting_position - 1,
                            -1,
                        ):
                            if external_name[index] not in removal_group:
                                continue
                            external_name = (
                                external_name[:index] + external_name[index + 1 :]
                            )
                            removed += 1
                            if removed == characters_to_remove:
                                break
                        if removed == characters_to_remove:
                            break

                random_numbers = None
                while external_name in source_by_external:
                    if random_numbers is None:
                        random_numbers = self._random_numbers(starting_name)
                    external_name = starting_name
                    while len(external_name) > self.maximum_length:
                        index = next(random_numbers) % len(external_name)
                        external_name = (
                            external_name[:index] + external_name[index + 1 :]
                        )

            source_by_external.pop(original_name)
            source_by_external[external_name] = source_name
            mapped_names[source_name] = external_name
        return mapped_names

    @staticmethod
    def _random_numbers(seed: str) -> Iterator[int]:
        digest_state = md5(usedforsecurity=False)
        digest_state.update(seed.encode())
        while True:
            digest = digest_state.digest()
            digest_state.update(digest)
            yield digest[0] + 256 * digest[1]


@dataclass(frozen=True, slots=True)
class CellProfilerDatabaseColumnDialect:
    """Construct and finally map CellProfiler/CPA database names."""

    table_prefix: str = ""
    column_name_mapping: CellProfilerColumnNameMapping | None = None

    @staticmethod
    def structural_field_prefixes() -> Iterator[str]:
        """Return exact raw prefixes for CellProfiler image-table structure."""

        return (
            family.field_prefix for family in CellProfilerImageStructuralFieldFamily
        )

    def with_column_names(
        self,
        source_names: Sequence[str],
        maximum_length: int,
    ) -> "CellProfilerDatabaseColumnDialect":
        """Bind this dialect to the complete raw schema name mapping."""

        return replace(
            self,
            column_name_mapping=CellProfilerColumnNameMapping(
                maximum_length=maximum_length,
                source_names=tuple(source_names),
            ),
        )

    def render_name(self, raw_name: str) -> str:
        """Apply only the final shortening/collision mapping to a raw CP name."""

        if self.column_name_mapping is None:
            return raw_name
        return self.column_name_mapping.render(raw_name)

    def source_name(self, external_name: str) -> str:
        """Invert final database-name rendering against the bound schema."""

        if self.column_name_mapping is None:
            return external_name
        return self.column_name_mapping.source(external_name)

    def ordered_measurement_fields(
        self,
        fields: Sequence[FieldSpec],
        primary_key: Sequence[FieldSpec] = (),
    ) -> tuple[FieldSpec, ...]:
        """Apply CellProfiler's image/object measurement field order."""

        declared = FieldSpec.merge_exact(
            (tuple(fields),),
            context="CellProfiler measurement fields",
        )
        primary = tuple(field for field in primary_key if field in declared)
        primary_names = frozenset(field.name for field in primary)
        return (
            *primary,
            *sorted(
                (field for field in declared if field.name not in primary_names),
                key=lambda field: self.render_name(field.name),
            ),
        )

    @staticmethod
    def ordered_fields(
        fields: Sequence[FieldSpec],
        primary_key: Sequence[FieldSpec] = (),
    ) -> tuple[FieldSpec, ...]:
        declared = FieldSpec.merge_exact(
            (tuple(fields),),
            context="CellProfiler projected fields",
        )
        primary = tuple(field for field in primary_key if field in declared)
        primary_names = frozenset(field.name for field in primary)
        return (
            *primary,
            *(field for field in declared if field.name not in primary_names),
        )

    @staticmethod
    def source_metadata_defaults() -> dict[str, object]:
        """Return CellProfiler's source-plane metadata default values."""

        return CellProfilerSourceMetadataField.static_defaults()

    def image_table_name(self) -> str:
        return f"{self.table_prefix}Per_Image"

    @classmethod
    def from_image_table(cls, table_name: str) -> "CellProfilerDatabaseColumnDialect":
        """Invert an exact CPA image-table name into its database dialect."""

        declared_table = cls._required(table_name, "image table name")
        unprefixed_table = cls().image_table_name()
        if not declared_table.endswith(unprefixed_table):
            raise ValueError(
                "CellProfiler CPA image table does not declare the canonical "
                f"{unprefixed_table!r} suffix: {table_name!r}."
            )
        dialect = cls(table_prefix=declared_table[: -len(unprefixed_table)])
        if dialect.image_table_name() != declared_table:
            raise ValueError(
                "CellProfiler CPA image table does not round-trip through the "
                f"database dialect: {table_name!r}."
            )
        return dialect

    def image_subject(
        self,
        table_name: str,
        image_id_field_name: str,
    ) -> MeasurementSubject:
        """Invert an exact CPA image-table declaration into its subject."""

        subject = MeasurementSubject(MeasurementScope.IMAGE, "Image")
        expected_table = self.image_table_name()
        expected_image_id = self.image_id_field().name
        if (
            table_name != expected_table
            or self.source_name(image_id_field_name) != expected_image_id
        ):
            raise ValueError(
                "CellProfiler CPA image declaration does not round-trip through "
                "the database dialect: "
                f"table={table_name!r}, image_id={image_id_field_name!r}, "
                f"expected_table={expected_table!r}, "
                f"expected_image_id={expected_image_id!r}."
            )
        return subject

    def object_table_name(self, object_name: str) -> str:
        self._require_name(object_name, "object_name")
        return f"{self.table_prefix}Per_{object_name}"

    def combined_object_table_name(self) -> str:
        """Return CellProfiler's single-table object export name."""

        return f"{self.table_prefix}Per_Object"

    def relationship_table_name(self, relationship_name: str) -> str:
        self._require_name(relationship_name, "relationship_name")
        return f"{self.table_prefix}{relationship_name}"

    @staticmethod
    def image_id_field() -> FieldSpec:
        return FieldSpec("ImageNumber", int, required=True)

    def object_id_field(
        self,
        subject: MeasurementSubject | None = None,
    ) -> FieldSpec:
        if subject is None:
            return FieldSpec("ObjectNumber", int, required=True)
        object_name = self._required_object_name(subject)
        return FieldSpec(
            f"{object_name}_{CellProfilerObjectCoreMeasurementFeature.OBJECT_NUMBER.value}",
            int,
            required=True,
        )

    def object_subject(
        self,
        table_name: str,
        object_id_field_name: str,
    ) -> MeasurementSubject | None:
        """Invert an exact CPA object-table declaration into its subject."""

        raw_object_id = self.source_name(object_id_field_name)
        combined_object_id = self.object_id_field().name
        if raw_object_id == combined_object_id:
            expected_table = self.combined_object_table_name()
            if table_name != expected_table:
                raise ValueError(
                    "CellProfiler CPA combined-object declaration does not "
                    "round-trip through the database dialect: "
                    f"table={table_name!r}, object_id={object_id_field_name!r}, "
                    f"expected_table={expected_table!r}, "
                    f"expected_object_id={combined_object_id!r}."
                )
            return None

        suffix = (
            f"_{CellProfilerObjectCoreMeasurementFeature.OBJECT_NUMBER.value}"
        )
        if not raw_object_id.endswith(suffix):
            raise ValueError(
                "CellProfiler object id field does not match the declared core "
                f"object-number feature: {object_id_field_name!r}."
            )
        object_name = self._required(
            raw_object_id[: -len(suffix)],
            "object_id object name",
        )
        subject = MeasurementSubject(
            MeasurementScope.OBJECT,
            object_name,
            id_field=CellProfilerObjectCoreMeasurementFeature.OBJECT_NUMBER.value,
        )
        expected_table = self.object_table_name(object_name)
        expected_object_id = self.object_id_field(subject).name
        if table_name != expected_table or raw_object_id != expected_object_id:
            raise ValueError(
                "CellProfiler CPA object declaration does not round-trip through "
                "the database dialect: "
                f"table={table_name!r}, object_id={object_id_field_name!r}, "
                f"expected_table={expected_table!r}, "
                f"expected_object_id={expected_object_id!r}."
            )
        return subject

    def measurement_field(
        self,
        subject: MeasurementSubject,
        field_spec: FieldSpec,
    ) -> FieldSpec:
        if subject.scope is MeasurementScope.IMAGE:
            raw_name = f"Image_{field_spec.name}"
        elif subject.scope is MeasurementScope.OBJECT:
            raw_name = f"{self._required_object_name(subject)}_{field_spec.name}"
        elif subject.scope is MeasurementScope.EXPERIMENT:
            raw_name = field_spec.name
        else:
            raise ValueError(
                "CellProfiler measurement fields require image, object, or "
                f"experiment scope, got {subject.scope.value!r}."
            )
        return replace(field_spec, name=raw_name)

    def source_measurement_field(
        self,
        subject: MeasurementSubject,
        external_field: FieldSpec,
    ) -> FieldSpec:
        """Invert one exact external measurement field for a known subject."""

        raw_name = self.source_name(external_field.name)
        if raw_name == self.image_id_field().name:
            return replace(external_field, name=raw_name)
        if (
            subject.scope is MeasurementScope.OBJECT
            and raw_name == self.object_id_field(subject).name
        ):
            return replace(
                external_field,
                name=CellProfilerObjectCoreMeasurementFeature.OBJECT_NUMBER.value,
            )
        if subject.scope is MeasurementScope.IMAGE:
            aggregate_statistic = CellProfilerImageAggregateStatistic.for_field_name(
                raw_name
            )
            if aggregate_statistic is not None:
                object_field = replace(
                    external_field,
                    name=raw_name[len(aggregate_statistic.field_prefix) :],
                )
                if not object_field.name:
                    raise ValueError(
                        f"CellProfiler aggregate field {external_field.name!r} has "
                        "no object feature name."
                    )
                if (
                    self.image_aggregate_field(
                        aggregate_statistic,
                        object_field,
                    ).name
                    != raw_name
                ):
                    raise ValueError(
                        "CellProfiler aggregate field does not round-trip through "
                        f"the database dialect: {external_field.name!r}."
                    )
                return replace(external_field, name=raw_name)

        if subject.scope is MeasurementScope.IMAGE:
            prefix = "Image_"
        elif subject.scope is MeasurementScope.OBJECT:
            prefix = f"{self._required_object_name(subject)}_"
        elif subject.scope is MeasurementScope.EXPERIMENT:
            prefix = ""
        else:
            raise ValueError(
                "CellProfiler measurement fields require image, object, or "
                f"experiment scope, got {subject.scope.value!r}."
            )
        if prefix and not raw_name.startswith(prefix):
            raise ValueError(
                f"CellProfiler field {external_field.name!r} is not owned by "
                f"measurement subject {subject.name!r}."
            )
        source_field = replace(external_field, name=raw_name[len(prefix) :])
        if not source_field.name:
            raise ValueError(
                f"CellProfiler field {external_field.name!r} has no feature name."
            )
        projected_name = self.measurement_field(subject, source_field).name
        if projected_name != raw_name:
            raise ValueError(
                "CellProfiler measurement field does not round-trip through the "
                f"database dialect: {external_field.name!r}."
            )
        return source_field

    @staticmethod
    def metadata_field(field_spec: FieldSpec) -> FieldSpec:
        return replace(field_spec, name=f"Image_Metadata_{field_spec.name}")

    @staticmethod
    def group_field(group_name: str) -> FieldSpec:
        required_name = CellProfilerDatabaseColumnDialect._required(
            group_name,
            "group_name",
        )
        return FieldSpec(
            "Image_"
            f"{CellProfilerImageStructuralFieldFamily.GROUP.qualified_name(required_name)}",
            int,
            required=False,
        )

    @staticmethod
    def source_image_path_field(source_image_name: str) -> FieldSpec:
        required_name = CellProfilerDatabaseColumnDialect._required(
            source_image_name,
            "source_image_name",
        )
        return FieldSpec(
            "Image_"
            f"{CellProfilerImageStructuralFieldFamily.PATH_NAME.qualified_name(required_name)}",
            str,
            required=False,
        )

    @staticmethod
    def source_image_file_field(source_image_name: str) -> FieldSpec:
        required_name = CellProfilerDatabaseColumnDialect._required(
            source_image_name,
            "source_image_name",
        )
        return FieldSpec(
            "Image_"
            f"{CellProfilerImageStructuralFieldFamily.FILE_NAME.qualified_name(required_name)}",
            str,
            required=False,
        )

    @staticmethod
    def source_image_feature_field(
        source_image_name: str,
        field_spec: FieldSpec,
    ) -> FieldSpec:
        required_name = CellProfilerDatabaseColumnDialect._required(
            source_image_name,
            "source_image_name",
        )
        return replace(field_spec, name=f"Image_{field_spec.name}_{required_name}")

    def object_location_field(
        self,
        subject: MeasurementSubject,
        axis_name: str,
    ) -> FieldSpec:
        object_name = self._required_object_name(subject)
        normalized_axis = axis_name.strip().upper()
        if normalized_axis not in {"X", "Y", "Z"}:
            raise ValueError(
                "CellProfiler object location axis must be X, Y, or Z; "
                f"got {axis_name!r}."
            )
        return FieldSpec(
            f"{object_name}_Location_Center_{normalized_axis}",
            float,
            required=False,
        )

    @staticmethod
    def thumbnail_field(source_image_name: str) -> FieldSpec:
        required_name = CellProfilerDatabaseColumnDialect._required(
            source_image_name,
            "source_image_name",
        )
        return FieldSpec(
            f"Image_Thumbnail_{required_name}",
            bytes,
            required=False,
        )

    @staticmethod
    def image_aggregate_field(
        statistic: CellProfilerImageAggregateStatistic,
        object_field: FieldSpec,
    ) -> FieldSpec:
        if not isinstance(statistic, CellProfilerImageAggregateStatistic):
            raise TypeError(
                "CellProfiler image aggregates require "
                "CellProfilerImageAggregateStatistic, got "
                f"{type(statistic).__name__}."
            )
        return FieldSpec(
            f"{statistic.field_prefix}{object_field.name}",
            float,
            required=False,
        )

    @staticmethod
    def _required_subject(
        subject: MeasurementSubject | None,
        field_name: str,
    ) -> MeasurementSubject:
        if subject is None:
            raise ValueError(f"CellProfiler field projection requires {field_name}.")
        return subject

    @classmethod
    def _required_object_name(cls, subject: MeasurementSubject | None) -> str:
        required_subject = cls._required_subject(subject, "subject")
        return cls._required(required_subject.object_name, "subject.object_name")

    @staticmethod
    def _required(value: str | None, field_name: str) -> str:
        if value is None or not value.strip():
            raise ValueError(f"CellProfiler field projection requires {field_name}.")
        return value.strip()

    @classmethod
    def _require_name(cls, value: str, field_name: str) -> None:
        cls._required(value, field_name)
