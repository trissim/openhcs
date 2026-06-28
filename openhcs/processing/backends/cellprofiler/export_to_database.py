"""CellProfiler database export module declaration."""

from __future__ import annotations

from typing import Any

from openhcs.interop.cellprofiler.setting_names import (
    SettingNameFamily,
    optional_setting_value,
    required_setting_value,
    split_symbol_names,
)
from openhcs.processing.backends.cellprofiler.module_classes import (
    InfrastructureCellProfilerModule,
)

class ExportToDatabaseModule(InfrastructureCellProfilerModule):
    module_name = 'ExportToDatabase'
    function_name = 'export_to_database'
    validated = True
    contract = 'unknown'
    confidence = 1.0
    database_type_setting = SettingNameFamily("Database type")
    sqlite_file_setting = SettingNameFamily("Name the SQLite database file")
    experiment_name_setting = SettingNameFamily("Experiment name")
    want_table_prefix_setting = SettingNameFamily("Add a prefix to table names?")
    table_prefix_setting = SettingNameFamily("Table prefix")
    save_cpa_properties_setting = SettingNameFamily(
        "Create a CellProfiler Analyst properties file?"
    )
    objects_choice_setting = SettingNameFamily(
        "Export measurements for all objects to the database?"
    )
    objects_list_setting = SettingNameFamily("Select the objects")
    relationship_table_setting = SettingNameFamily("Export object relationships?")
    object_table_mode_setting = SettingNameFamily(
        "Create one table per object, a single object table or a single object view?"
    )

    @classmethod
    def database_export_settings(
        cls,
        module: "ModuleBlock",
    ) -> "CellProfilerDatabaseExportSettings":
        from openhcs.interop.cellprofiler.analyst_export import (
            CellProfilerDatabaseExportSettings,
            CellProfilerObjectTableMode,
        )

        if CellProfilerModule.canonical_module_name(module.name) != cls.module_name:
            raise ValueError(
                "database_export_settings requires an ExportToDatabase module, "
                f"got {module.name!r}."
            )
        database_type = required_setting_value(module, cls.database_type_setting)
        if database_type.strip().lower() != "sqlite":
            raise ValueError(
                "OpenHCS CPA export dry run only supports SQLite ExportToDatabase; "
                f"got {database_type!r}."
            )
        return CellProfilerDatabaseExportSettings(
            database_type="sqlite",
            sqlite_file=required_setting_value(module, cls.sqlite_file_setting),
            experiment_name=required_setting_value(
                module,
                cls.experiment_name_setting,
            ),
            table_prefix=cls._table_prefix(module),
            object_table_mode=cls._object_table_mode(
                required_setting_value(module, cls.object_table_mode_setting),
                CellProfilerObjectTableMode,
            ),
            selected_objects=cls._selected_objects(module),
            wants_properties_file=cls._required_bool(
                module,
                cls.save_cpa_properties_setting,
            ),
            wants_relationship_tables=cls._required_bool(
                module,
                cls.relationship_table_setting,
            ),
        )

    @classmethod
    def _required_bool(
        cls,
        module: "ModuleBlock",
        setting_name: SettingNameFamily,
    ) -> bool:
        raw_value = required_setting_value(module, setting_name)
        normalized = raw_value.strip().lower()
        if normalized in {"yes", "true", "1"}:
            return True
        if normalized in {"no", "false", "0"}:
            return False
        raise ValueError(
            f"ExportToDatabase setting {setting_name.canonical!r} requires "
            f"a boolean value, got {raw_value!r}."
        )

    @classmethod
    def _table_prefix(cls, module: "ModuleBlock") -> str:
        if not cls._required_bool(module, cls.want_table_prefix_setting):
            return ""
        return required_setting_value(module, cls.table_prefix_setting)

    @staticmethod
    def _object_table_mode(
        value: str,
        object_table_mode_type: type[Any],
    ) -> Any:
        normalized = value.strip().lower()
        modes = {
            "one table per object type": object_table_mode_type.PER_OBJECT,
            "single object table": object_table_mode_type.COMBINED,
            "single object view": object_table_mode_type.VIEW,
        }
        try:
            return modes[normalized]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported ExportToDatabase object table mode {value!r}."
            ) from exc

    @classmethod
    def _selected_objects(cls, module: "ModuleBlock") -> tuple[str, ...] | None:
        raw_choice = required_setting_value(module, cls.objects_choice_setting)
        normalized = raw_choice.strip().lower()
        if normalized == "all":
            return None
        if normalized == "none":
            return ()
        if normalized == "select...":
            value = optional_setting_value(module, cls.objects_list_setting)
            if value is None:
                raise ValueError(
                    "ExportToDatabase selected object export requires selected "
                    "object names."
                )
            return split_symbol_names(value)
        raise ValueError(
            f"Unsupported ExportToDatabase object choice {raw_choice!r}."
        )
