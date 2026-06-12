"""Typed CellProfiler ExportToDatabase setting semantics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

from metaclass_registry import AutoRegisterMeta

from .analyst_export import (
    CellProfilerDatabaseExportSettings,
    CellProfilerObjectTableMode,
)
from .parser import ModuleBlock
from .setting_names import (
    SettingNameFamily,
    optional_setting_value,
    required_setting_value,
    split_symbol_names,
)


EXPORT_TO_DATABASE_MODULE = "ExportToDatabase"
DATABASE_TYPE_SETTING = SettingNameFamily("Database type")
SQLITE_FILE_SETTING = SettingNameFamily("Name the SQLite database file")
EXPERIMENT_NAME_SETTING = SettingNameFamily("Experiment name")
WANT_TABLE_PREFIX_SETTING = SettingNameFamily("Add a prefix to table names?")
TABLE_PREFIX_SETTING = SettingNameFamily("Table prefix")
SAVE_CPA_PROPERTIES_SETTING = SettingNameFamily(
    "Create a CellProfiler Analyst properties file?"
)
OBJECTS_CHOICE_SETTING = SettingNameFamily(
    "Export measurements for all objects to the database?"
)
OBJECTS_LIST_SETTING = SettingNameFamily("Select the objects")
RELATIONSHIP_TABLE_SETTING = SettingNameFamily("Export object relationships?")
OBJECT_TABLE_MODE_SETTING = SettingNameFamily(
    "Create one table per object, a single object table or a single object view?"
)


@dataclass(frozen=True, slots=True)
class CellProfilerRequiredBooleanSetting:
    """Required CellProfiler binary setting parsed without default fallback."""

    module: ModuleBlock
    setting_name: SettingNameFamily

    @property
    def value(self) -> bool:
        raw_value = required_setting_value(self.module, self.setting_name)
        normalized = raw_value.strip().lower()
        if normalized in {"yes", "true", "1"}:
            return True
        if normalized in {"no", "false", "0"}:
            return False
        raise ValueError(
            f"ExportToDatabase setting {self.setting_name.canonical!r} requires "
            f"a boolean value, got {raw_value!r}."
        )


class ExportToDatabaseObjectTableModeLiteral(ABC, metaclass=AutoRegisterMeta):
    """Nominal parser for ExportToDatabase object-table mode literals."""

    __registry_key__ = "literal"
    __skip_if_no_key__ = True
    literal: ClassVar[str | None] = None

    @classmethod
    def parse(cls, value: str) -> CellProfilerObjectTableMode:
        normalized = value.strip().lower()
        parser_type = cls.__registry__.get(normalized)
        if parser_type is None:
            raise ValueError(
                f"Unsupported ExportToDatabase object table mode {value!r}."
            )
        return parser_type().mode()

    @abstractmethod
    def mode(self) -> CellProfilerObjectTableMode:
        """Return the OpenHCS object table mode."""


class ExportToDatabasePerObjectTableMode(ExportToDatabaseObjectTableModeLiteral):
    literal = "one table per object type"

    def mode(self) -> CellProfilerObjectTableMode:
        return CellProfilerObjectTableMode.PER_OBJECT


class ExportToDatabaseCombinedObjectTableMode(ExportToDatabaseObjectTableModeLiteral):
    literal = "single object table"

    def mode(self) -> CellProfilerObjectTableMode:
        return CellProfilerObjectTableMode.COMBINED


class ExportToDatabaseObjectViewMode(ExportToDatabaseObjectTableModeLiteral):
    literal = "single object view"

    def mode(self) -> CellProfilerObjectTableMode:
        return CellProfilerObjectTableMode.VIEW


class ExportToDatabaseObjectSelectionLiteral(ABC, metaclass=AutoRegisterMeta):
    """Nominal parser for ExportToDatabase object-selection literals."""

    __registry_key__ = "literal"
    __skip_if_no_key__ = True
    literal: ClassVar[str | None] = None

    @classmethod
    def parse(cls, module: ModuleBlock) -> tuple[str, ...] | None:
        raw_choice = required_setting_value(module, OBJECTS_CHOICE_SETTING)
        normalized = raw_choice.strip().lower()
        parser_type = cls.__registry__.get(normalized)
        if parser_type is None:
            raise ValueError(
                f"Unsupported ExportToDatabase object choice {raw_choice!r}."
            )
        return parser_type().selected_objects(module)

    @abstractmethod
    def selected_objects(self, module: ModuleBlock) -> tuple[str, ...] | None:
        """Return selected object names, or None when all objects are selected."""


class ExportToDatabaseAllObjectsSelection(ExportToDatabaseObjectSelectionLiteral):
    literal = "all"

    def selected_objects(self, module: ModuleBlock) -> tuple[str, ...] | None:
        return None


class ExportToDatabaseNoObjectsSelection(ExportToDatabaseObjectSelectionLiteral):
    literal = "none"

    def selected_objects(self, module: ModuleBlock) -> tuple[str, ...] | None:
        return ()


class ExportToDatabaseExplicitObjectsSelection(ExportToDatabaseObjectSelectionLiteral):
    literal = "select..."

    def selected_objects(self, module: ModuleBlock) -> tuple[str, ...] | None:
        value = optional_setting_value(module, OBJECTS_LIST_SETTING)
        if value is None:
            raise ValueError(
                "ExportToDatabase selected object export requires selected object "
                "names."
            )
        return split_symbol_names(value)


def export_to_database_settings(
    module: ModuleBlock,
) -> CellProfilerDatabaseExportSettings:
    """Return nominal CPA export settings from one ExportToDatabase module."""
    if module.name != EXPORT_TO_DATABASE_MODULE:
        raise ValueError(
            "export_to_database_settings requires an ExportToDatabase module, "
            f"got {module.name!r}."
        )
    database_type = required_setting_value(module, DATABASE_TYPE_SETTING)
    if database_type.strip().lower() != "sqlite":
        raise ValueError(
            "OpenHCS CPA export dry run only supports SQLite ExportToDatabase; "
            f"got {database_type!r}."
        )
    return CellProfilerDatabaseExportSettings(
        database_type="sqlite",
        sqlite_file=required_setting_value(module, SQLITE_FILE_SETTING),
        experiment_name=required_setting_value(module, EXPERIMENT_NAME_SETTING),
        table_prefix=_table_prefix(module),
        object_table_mode=_object_table_mode(module),
        selected_objects=ExportToDatabaseObjectSelectionLiteral.parse(module),
        wants_properties_file=CellProfilerRequiredBooleanSetting(
            module,
            SAVE_CPA_PROPERTIES_SETTING,
        ).value,
        wants_relationship_tables=CellProfilerRequiredBooleanSetting(
            module,
            RELATIONSHIP_TABLE_SETTING,
        ).value,
    )


def _table_prefix(module: ModuleBlock) -> str:
    if not CellProfilerRequiredBooleanSetting(
        module,
        WANT_TABLE_PREFIX_SETTING,
    ).value:
        return ""
    return required_setting_value(module, TABLE_PREFIX_SETTING)


def _object_table_mode(module: ModuleBlock) -> CellProfilerObjectTableMode:
    value = required_setting_value(module, OBJECT_TABLE_MODE_SETTING)
    return ExportToDatabaseObjectTableModeLiteral.parse(value)
