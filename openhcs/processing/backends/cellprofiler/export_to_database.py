"""Executable CellProfiler-compatible database export."""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Literal

from openhcs.core.artifacts import (
    ArtifactInputPlan,
    ArtifactOutputPlan,
    ArtifactSpec,
    ArtifactSpecCollection,
    ImageArtifactType,
    MeasurementsArtifactType,
    RelationshipsArtifactType,
    SpecialArtifactType,
)
from openhcs.core.callable_contract import FunctionStepExecutionScope
from openhcs.core.context.processing_context import ProcessingContext
from openhcs.core.pipeline.function_contracts import (
    execution_scope,
    runtime_bound_parameters,
)
from openhcs.core.runtime_stores import RuntimeArtifactBatch
from openhcs.interop.cellprofiler.analyst_export import (
    CPAImageChannelSpec,
    CPAPropertiesRenderer,
    CPASQLiteRenderer,
    CellProfilerAnalystProjectionBuilder,
    CellProfilerDatabaseExportSettings,
    CellProfilerObjectTableMode,
)
from openhcs.interop.cellprofiler.cellprofiler_literals import (
    cellprofiler_setting_literal,
)
from openhcs.interop.cellprofiler.database_column_dialect import (
    CellProfilerDatabaseColumnDialect,
)
from openhcs.interop.cellprofiler.module_settings import (
    BoundModuleSettings,
)
from openhcs.interop.cellprofiler.module_artifact_declarations import (
    ArtifactExportModule,
)
from openhcs.interop.cellprofiler.parser import ModuleBlock, ModuleSetting
from openhcs.interop.cellprofiler.setting_names import (
    SettingNameFamily,
    normalized_symbol_name,
    optional_setting_value,
    required_setting_value,
    split_symbol_names,
)
from openhcs.interop.cellprofiler.settings_binder import (
    SettingToKeywordBinding,
    parse_cellprofiler_bool,
    parse_cellprofiler_int,
)
from openhcs.processing.materialization import FileBundleOptions, MaterializationSpec

if TYPE_CHECKING:
    from openhcs.core.function_patterns import (
        FunctionInvocationKey,
        NormalizedFunctionItem,
    )
    from openhcs.core.invocation_artifacts import ArtifactDeclarationStepContext


def _parse_sqlite_database_type(value: str) -> Literal["sqlite"]:
    if value.strip().casefold() != "sqlite":
        raise ValueError(f"OpenHCS ExportToDatabase supports SQLite, got {value!r}.")
    return "sqlite"


def _parse_object_table_mode(value: str) -> CellProfilerObjectTableMode:
    return CellProfilerObjectTableMode.from_cellprofiler(value)


def _parse_classification_type(value: str) -> Literal["object", "image"]:
    normalized = value.strip().casefold()
    if normalized == "object":
        return "object"
    if normalized == "image":
        return "image"
    raise ValueError(f"Unsupported ExportToDatabase classification type {value!r}.")


def _parse_overwrite_mode(
    value: str,
) -> Literal["never", "data_only", "data_and_schema"]:
    normalized = value.strip().casefold()
    if normalized == "never":
        return "never"
    if normalized == "data only":
        return "data_only"
    if normalized == "data and schema":
        return "data_and_schema"
    raise ValueError(f"Unsupported ExportToDatabase overwrite mode {value!r}.")


class ExportToDatabaseModule(ArtifactExportModule):
    """Nominal module owner for plate-scoped SQLite/CPA export semantics."""

    module_name = "ExportToDatabase"
    function_name = "export_to_database"
    validated = True
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
    include_all_images_setting = SettingNameFamily(
        "Include information for all images, using default values?"
    )
    aggregate_image_mean_setting = SettingNameFamily(
        "Calculate the per-image mean values of object measurements?"
    )
    aggregate_image_median_setting = SettingNameFamily(
        "Calculate the per-image median values of object measurements?"
    )
    aggregate_image_standard_deviation_setting = SettingNameFamily(
        "Calculate the per-image standard deviation values of object measurements?"
    )
    aggregate_well_mean_setting = SettingNameFamily(
        "Calculate the per-well mean values of object measurements?"
    )
    aggregate_well_median_setting = SettingNameFamily(
        "Calculate the per-well median values of object measurements?"
    )
    aggregate_well_standard_deviation_setting = SettingNameFamily(
        "Calculate the per-well standard deviation values of object measurements?"
    )
    maximum_column_name_length_setting = SettingNameFamily(
        "Maximum # of characters in a column name"
    )
    image_url_prepend_setting = SettingNameFamily(
        "Enter an image url prepend if you plan to access your files via http"
    )
    write_thumbnails_setting = SettingNameFamily(
        "Write image thumbnails directly to the database?"
    )
    thumbnail_images_setting = SettingNameFamily(
        "Select the images for which you want to save thumbnails"
    )
    auto_scale_thumbnails_setting = SettingNameFamily(
        "Auto-scale thumbnail pixel intensities?"
    )
    plate_type_setting = SettingNameFamily("Select the plate type")
    plate_metadata_setting = SettingNameFamily("Select the plate metadata")
    well_metadata_setting = SettingNameFamily("Select the well metadata")
    property_image_count_setting = SettingNameFamily("Properties image group count")
    property_image_setting = SettingNameFamily("Select an image to include")
    property_use_image_name_setting = SettingNameFamily(
        "Use the image name for the display?"
    )
    property_image_name_setting = SettingNameFamily("Image name")
    property_channel_color_setting = SettingNameFamily("Channel color")
    location_object_setting = SettingNameFamily(
        "Which objects should be used for locations?"
    )
    property_group_count_setting = SettingNameFamily("Properties group field count")
    wants_group_fields_setting = SettingNameFamily("Do you want to add group fields?")
    group_name_setting = SettingNameFamily("Enter the name of the group")
    group_columns_setting = SettingNameFamily(
        "Enter the per-image columns which define the group, separated by commas"
    )
    property_filter_count_setting = SettingNameFamily("Properties filter field count")
    wants_filter_fields_setting = SettingNameFamily("Do you want to add filter fields?")
    create_plate_filters_setting = SettingNameFamily(
        "Automatically create a filter for each plate?"
    )
    phenotype_class_table_setting = SettingNameFamily(
        "Enter a phenotype class table name if using the Classifier tool in CellProfiler Analyst"
    )
    overwrite_mode_setting = SettingNameFamily("Overwrite without warning?")
    access_images_via_url_setting = SettingNameFamily(
        "Access CellProfiler Analyst images via URL?"
    )
    classification_type_setting = SettingNameFamily("Select the classification type")
    workspace_measurement_count_setting = SettingNameFamily(
        "Workspace measurement count"
    )
    wants_workspace_file_setting = SettingNameFamily(
        "Create a CellProfiler Analyst workspace file?"
    )
    workspace_display_tool_setting = SettingNameFamily(
        "Select the measurement display tool"
    )
    workspace_x_type_setting = SettingNameFamily(
        "Type of measurement to plot on the X-axis"
    )
    workspace_object_name_setting = SettingNameFamily("Enter the object name")
    workspace_x_measurement_setting = SettingNameFamily("Select the X-axis measurement")
    workspace_x_index_setting = SettingNameFamily("Select the X-axis index")
    workspace_y_type_setting = SettingNameFamily(
        "Type of measurement to plot on the Y-axis"
    )
    workspace_y_measurement_setting = SettingNameFamily("Select the Y-axis measurement")
    workspace_y_index_setting = SettingNameFamily("Select the Y-axis index")

    setting_bindings = (
        SettingToKeywordBinding(
            database_type_setting,
            "database_type",
            _parse_sqlite_database_type,
        ),
        SettingToKeywordBinding(sqlite_file_setting, "sqlite_file"),
        SettingToKeywordBinding(experiment_name_setting, "experiment_name"),
        SettingToKeywordBinding(
            want_table_prefix_setting,
            "add_table_prefix",
            parse_cellprofiler_bool,
        ),
        SettingToKeywordBinding(table_prefix_setting, "table_prefix"),
        SettingToKeywordBinding(
            save_cpa_properties_setting,
            "wants_properties_file",
            parse_cellprofiler_bool,
        ),
        SettingToKeywordBinding(
            relationship_table_setting,
            "wants_relationship_tables",
            parse_cellprofiler_bool,
        ),
        SettingToKeywordBinding(
            object_table_mode_setting,
            "object_table_mode",
            _parse_object_table_mode,
        ),
        SettingToKeywordBinding(
            include_all_images_setting,
            "include_all_images",
            parse_cellprofiler_bool,
        ),
        SettingToKeywordBinding(
            aggregate_image_mean_setting,
            "calculate_per_image_mean",
            parse_cellprofiler_bool,
        ),
        SettingToKeywordBinding(
            aggregate_image_median_setting,
            "calculate_per_image_median",
            parse_cellprofiler_bool,
        ),
        SettingToKeywordBinding(
            aggregate_image_standard_deviation_setting,
            "calculate_per_image_standard_deviation",
            parse_cellprofiler_bool,
        ),
        SettingToKeywordBinding(
            aggregate_well_mean_setting,
            "calculate_per_well_mean",
            parse_cellprofiler_bool,
        ),
        SettingToKeywordBinding(
            aggregate_well_median_setting,
            "calculate_per_well_median",
            parse_cellprofiler_bool,
        ),
        SettingToKeywordBinding(
            aggregate_well_standard_deviation_setting,
            "calculate_per_well_standard_deviation",
            parse_cellprofiler_bool,
        ),
        SettingToKeywordBinding(
            maximum_column_name_length_setting,
            "maximum_column_name_length",
            parse_cellprofiler_int,
        ),
        SettingToKeywordBinding(
            image_url_prepend_setting,
            "image_url_prepend",
            str,
        ),
        SettingToKeywordBinding(
            write_thumbnails_setting,
            "write_image_thumbnails",
            parse_cellprofiler_bool,
        ),
        SettingToKeywordBinding(
            thumbnail_images_setting,
            "thumbnail_image_names",
            str,
        ),
        SettingToKeywordBinding(
            auto_scale_thumbnails_setting,
            "auto_scale_thumbnail_intensities",
            parse_cellprofiler_bool,
        ),
        SettingToKeywordBinding(plate_type_setting, "plate_type", str),
        SettingToKeywordBinding(plate_metadata_setting, "plate_metadata", str),
        SettingToKeywordBinding(well_metadata_setting, "well_metadata", str),
        SettingToKeywordBinding(
            wants_group_fields_setting,
            "wants_group_fields",
            parse_cellprofiler_bool,
        ),
        SettingToKeywordBinding(
            wants_filter_fields_setting,
            "wants_filter_fields",
            parse_cellprofiler_bool,
        ),
        SettingToKeywordBinding(
            create_plate_filters_setting,
            "create_plate_filters",
            parse_cellprofiler_bool,
        ),
        SettingToKeywordBinding(
            phenotype_class_table_setting,
            "phenotype_class_table",
            str,
        ),
        SettingToKeywordBinding(
            overwrite_mode_setting,
            "overwrite_mode",
            _parse_overwrite_mode,
        ),
        SettingToKeywordBinding(
            access_images_via_url_setting,
            "access_images_via_url",
            parse_cellprofiler_bool,
        ),
        SettingToKeywordBinding(
            classification_type_setting,
            "classification_type",
            _parse_classification_type,
        ),
        SettingToKeywordBinding(
            wants_workspace_file_setting,
            "wants_workspace_file",
            parse_cellprofiler_bool,
        ),
    )
    ignored_settings = (
        "Database name",
        "Output file location",
        "Database host",
        "Username",
        "Password",
    )

    @classmethod
    def uses_cellprofiler_runtime_adapter(cls) -> bool:
        """Database export is a generic plate callable, not a CP workspace call."""

        return False

    @classmethod
    def writes_thumbnails(cls, module: ModuleBlock) -> bool:
        """Return whether this declaration owns the thumbnail measurement port."""

        value = optional_setting_value(module, cls.write_thumbnails_setting)
        return value is not None and parse_cellprofiler_bool(value)

    @classmethod
    def property_image_channels(
        cls,
        module: ModuleBlock,
    ) -> tuple[CPAImageChannelSpec, ...]:
        """Parse explicitly configured CPA image groups in declaration order."""

        aliases = module.get_setting_values(cls.property_image_setting.canonical)
        display_flags = module.get_setting_values(
            cls.property_use_image_name_setting.canonical
        )
        image_names = module.get_setting_values(
            cls.property_image_name_setting.canonical
        )
        colors = module.get_setting_values(cls.property_channel_color_setting.canonical)
        count_value = optional_setting_value(module, cls.property_image_count_setting)
        declared_count = int(count_value) if count_value is not None else len(aliases)
        for setting_name, values in (
            (cls.property_image_setting.canonical, aliases),
            (cls.property_use_image_name_setting.canonical, display_flags),
            (cls.property_image_name_setting.canonical, image_names),
            (cls.property_channel_color_setting.canonical, colors),
        ):
            if values and len(values) != declared_count:
                raise ValueError(
                    f"ExportToDatabase declares {declared_count} CPA image groups but "
                    f"{setting_name!r} has {len(values)} rows."
                )

        channels: list[CPAImageChannelSpec] = []
        for index in range(declared_count):
            alias = normalized_symbol_name(aliases[index]) if aliases else None
            if alias is None:
                continue
            use_image_name = (
                parse_cellprofiler_bool(display_flags[index]) if display_flags else True
            )
            display_name = (
                image_names[index].strip()
                if use_image_name and image_names and image_names[index].strip()
                else alias
            )
            color = colors[index].strip().casefold() if colors else "none"
            channels.append(
                CPAImageChannelSpec(
                    alias=alias,
                    image_name=display_name,
                    channel_color=color,
                )
            )
        return tuple(channels)

    @classmethod
    def postprocess_bound_settings(
        cls,
        module: ModuleBlock,
        bound: BoundModuleSettings,
    ) -> BoundModuleSettings:
        """Bind compound object selection and repeated CPA image-group rows."""

        location_value = optional_setting_value(module, cls.location_object_setting)
        filter_count_value = optional_setting_value(
            module,
            cls.property_filter_count_setting,
        )
        filter_count = 0 if filter_count_value is None else int(filter_count_value)
        if filter_count:
            raise ValueError(
                "ExportToDatabase custom CPA filter fields are not represented "
                "by the public callable."
            )
        bound = bound.with_kwargs(
            {"selected_objects": cls._selected_objects(module)}
        ).with_consumed_settings(
            cls.objects_choice_setting,
            cls.objects_list_setting,
        )
        bound = bound.with_kwargs(
            {"image_channels": cls.property_image_channels(module)}
        ).with_consumed_settings(
            cls.property_image_count_setting,
            cls.property_image_setting,
            cls.property_use_image_name_setting,
            cls.property_image_name_setting,
            cls.property_channel_color_setting,
        )
        bound = bound.with_kwargs(
            {
                "location_object": (
                    None if location_value is None else location_value.strip() or None
                )
            }
        ).with_consumed_settings(
            cls.location_object_setting,
        )
        bound = bound.with_kwargs(
            {"group_fields": cls._group_fields(module)}
        ).with_consumed_settings(
            cls.property_group_count_setting,
            cls.group_name_setting,
            cls.group_columns_setting,
        )
        bound = bound.with_consumed_settings(
            cls.property_filter_count_setting,
        )
        return bound.with_kwargs(
            {"workspace_measurements": cls._workspace_measurements(module)}
        ).with_consumed_settings(
            cls.workspace_measurement_count_setting,
            cls.workspace_display_tool_setting,
            cls.workspace_x_type_setting,
            cls.workspace_object_name_setting,
            cls.workspace_x_measurement_setting,
            cls.workspace_x_index_setting,
            cls.workspace_y_type_setting,
            cls.workspace_y_measurement_setting,
            cls.workspace_y_index_setting,
        )

    @classmethod
    def finalize_module_blocks_for_invocation(
        cls,
        blocks, *,
        invocation: NormalizedFunctionItem,
        step_context: ArtifactDeclarationStepContext,
    ) -> tuple[ModuleBlock, ...]:
        """Reconstruct compound settings needed by exact public compilation."""

        blocks = super().finalize_module_blocks_for_invocation(
            blocks, invocation=invocation,
            step_context=step_context,
        )
        signature = inspect.signature(cls.require_callable())
        arguments = signature.bind_partial(**invocation.kwargs_dict)
        arguments.apply_defaults()
        selected_objects = arguments.arguments["selected_objects"]
        image_channels = arguments.arguments["image_channels"]
        group_fields = (
            arguments.arguments["group_fields"]
            if arguments.arguments["wants_group_fields"]
            else ()
        )
        compound_records = (
            *cls._selected_object_setting_records(selected_objects),
            *cls._property_image_setting_records(image_channels),
            *cls._group_setting_records(group_fields),
        )
        return tuple(cls._block_with_records(block, compound_records) for block in blocks)

    @classmethod
    def artifact_contract_inputs(
        cls,
        module: ModuleBlock,
        *,
        invocation_key: FunctionInvocationKey,
        step_context: ArtifactDeclarationStepContext,
    ) -> tuple[ArtifactSpec, ...]:
        """Select the exact artifacts exported to the database."""

        del invocation_key
        available = ArtifactSpecCollection(
            step_context.available_artifacts.unique(
                conflict_context="ExportToDatabase available artifact"
            )
        )
        include_all_images = cls._include_all_images(module)
        explicit_channels = cls.property_image_channels(module)
        thumbnail_image_names = cls._thumbnail_image_names(module)
        property_image_names = (
            tuple(
                spec.name
                for spec in available.of_artifact_type(ImageArtifactType)
                if not spec.source_context_sources()
            )
            if include_all_images
            else tuple(channel.alias for channel in explicit_channels)
        )
        selected_image_names = tuple(
            dict.fromkeys((*property_image_names, *thumbnail_image_names))
        )
        thumbnail_image_name_set = frozenset(thumbnail_image_names)
        available_image_names = available.name_set_of_artifact_type(ImageArtifactType)
        missing_images = tuple(
            name for name in selected_image_names if name not in available_image_names
        )
        if missing_images:
            raise ValueError(
                "ExportToDatabase CPA image groups reference unavailable image "
                f"artifacts: {missing_images!r}."
            )

        selected_image_name_set = frozenset(selected_image_names)
        inputs: list[ArtifactSpec] = []
        for spec in available.specs:
            if spec.artifact_type in {
                MeasurementsArtifactType,
                RelationshipsArtifactType,
            }:
                inputs.append(spec.for_plan_type(ArtifactInputPlan))
            elif (
                spec.artifact_type is ImageArtifactType
                and spec.name in selected_image_name_set
            ):
                if spec.plan_type not in (ArtifactInputPlan, ArtifactOutputPlan):
                    raise TypeError(
                        "ExportToDatabase selected image artifacts must use an "
                        f"input or output plan, got {spec.plan_type.__name__}."
                    )
                inputs.append(
                    replace(
                        spec.for_plan_type(ArtifactInputPlan),
                        required=(
                            spec.plan_type is ArtifactOutputPlan
                            and spec.name in thumbnail_image_name_set
                        ),
                    )
                )
        return tuple(inputs)

    @classmethod
    def artifact_contract_outputs(
        cls,
        module: ModuleBlock,
        *,
        invocation_key: FunctionInvocationKey,
        step_context: ArtifactDeclarationStepContext,
        artifact_inputs: ArtifactSpecCollection,
    ) -> tuple[ArtifactSpec, ...]:
        """Declare the terminal database file bundle."""

        return (
            ArtifactSpec.output(
                cls.canonical_output_artifact_name(
                    artifact_type=SpecialArtifactType,
                    output_position=0,
                    block_position=0,
                    step_context=step_context,
                ),
                SpecialArtifactType,
                materialization=MaterializationSpec(FileBundleOptions()),
            ),
        )

    @classmethod
    def _required_bool(
        cls,
        module: ModuleBlock,
        setting_name: SettingNameFamily,
    ) -> bool:
        return parse_cellprofiler_bool(required_setting_value(module, setting_name))

    @classmethod
    def _table_prefix(cls, module: ModuleBlock) -> str:
        if not cls._required_bool(module, cls.want_table_prefix_setting):
            return ""
        return required_setting_value(module, cls.table_prefix_setting)

    @classmethod
    def _selected_objects(cls, module: ModuleBlock) -> tuple[str, ...] | None:
        raw_choice = required_setting_value(module, cls.objects_choice_setting)
        normalized = raw_choice.strip().casefold()
        if normalized == "all":
            return None
        if normalized == "none":
            return ()
        if normalized == "select...":
            value = optional_setting_value(module, cls.objects_list_setting)
            if value is None:
                raise ValueError(
                    "ExportToDatabase selected-object export requires object names."
                )
            return split_symbol_names(value)
        raise ValueError(f"Unsupported ExportToDatabase object choice {raw_choice!r}.")

    @classmethod
    def _include_all_images(cls, module: ModuleBlock) -> bool:
        value = optional_setting_value(module, cls.include_all_images_setting)
        return True if value is None else parse_cellprofiler_bool(value)

    @classmethod
    def _thumbnail_image_names(cls, module: ModuleBlock) -> tuple[str, ...]:
        if not cls.writes_thumbnails(module):
            return ()
        selected = optional_setting_value(module, cls.thumbnail_images_setting)
        return () if selected is None else split_symbol_names(selected)

    @classmethod
    def _group_fields(cls, module: ModuleBlock) -> tuple[tuple[str, str], ...]:
        declared_count = int(
            required_setting_value(module, cls.property_group_count_setting)
        )
        names = module.get_setting_values(cls.group_name_setting.canonical)
        columns = module.get_setting_values(cls.group_columns_setting.canonical)
        cls._require_record_count(
            declared_count,
            cls.group_name_setting,
            names,
        )
        cls._require_record_count(
            declared_count,
            cls.group_columns_setting,
            columns,
        )
        if not cls._required_bool(module, cls.wants_group_fields_setting):
            return ()
        if any(
            not name.strip() or not statement.strip()
            for name, statement in zip(
                names,
                columns,
                strict=True,
            )
        ):
            raise ValueError(
                "ExportToDatabase enabled CPA group fields require a name and "
                "per-image column declaration."
            )
        return tuple(
            (name.strip(), statement.strip())
            for name, statement in zip(names, columns, strict=True)
        )

    @classmethod
    def _workspace_measurements(
        cls,
        module: ModuleBlock,
    ) -> tuple[tuple[str, str, str, str, str, str, str, str, str], ...]:
        declared_count = int(
            required_setting_value(module, cls.workspace_measurement_count_setting)
        )
        display_tools = module.get_setting_values(
            cls.workspace_display_tool_setting.canonical
        )
        x_types = module.get_setting_values(cls.workspace_x_type_setting.canonical)
        object_names = module.get_setting_values(
            cls.workspace_object_name_setting.canonical
        )
        x_measurements = module.get_setting_values(
            cls.workspace_x_measurement_setting.canonical
        )
        x_indices = module.get_setting_values(cls.workspace_x_index_setting.canonical)
        y_types = module.get_setting_values(cls.workspace_y_type_setting.canonical)
        y_measurements = module.get_setting_values(
            cls.workspace_y_measurement_setting.canonical
        )
        y_indices = module.get_setting_values(cls.workspace_y_index_setting.canonical)
        for setting_name, values in (
            (cls.workspace_display_tool_setting, display_tools),
            (cls.workspace_x_type_setting, x_types),
            (cls.workspace_x_measurement_setting, x_measurements),
            (cls.workspace_x_index_setting, x_indices),
            (cls.workspace_y_type_setting, y_types),
            (cls.workspace_y_measurement_setting, y_measurements),
            (cls.workspace_y_index_setting, y_indices),
        ):
            cls._require_record_count(declared_count, setting_name, values)
        cls._require_record_count(
            declared_count * 2,
            cls.workspace_object_name_setting,
            object_names,
        )
        if not cls._required_bool(module, cls.wants_workspace_file_setting):
            return ()
        return tuple(
            (
                display_tools[index].strip(),
                x_types[index].strip(),
                object_names[index * 2].strip(),
                x_measurements[index].strip(),
                x_indices[index].strip(),
                y_types[index].strip(),
                object_names[index * 2 + 1].strip(),
                y_measurements[index].strip(),
                y_indices[index].strip(),
            )
            for index in range(declared_count)
        )

    @classmethod
    def _require_record_count(
        cls,
        declared_count: int,
        setting_name: SettingNameFamily,
        values: tuple[str, ...],
    ) -> None:
        if len(values) != declared_count:
            raise ValueError(
                f"ExportToDatabase declares {declared_count} rows for "
                f"{setting_name.canonical!r}, got {len(values)}."
            )

    @classmethod
    def _selected_object_setting_records(
        cls,
        selected_objects: object,
    ) -> tuple[ModuleSetting, ...]:
        if selected_objects is None:
            return (ModuleSetting(cls.objects_choice_setting.canonical, "All"),)
        if not isinstance(selected_objects, (tuple, list)):
            raise TypeError("selected_objects must be a tuple, list, or None.")
        names = tuple(str(name).strip() for name in selected_objects)
        if any(not name for name in names):
            raise ValueError("selected_objects cannot contain empty names.")
        if not names:
            return (ModuleSetting(cls.objects_choice_setting.canonical, "None"),)
        return (
            ModuleSetting(cls.objects_choice_setting.canonical, "Select..."),
            ModuleSetting(cls.objects_list_setting.canonical, ",".join(names)),
        )

    @classmethod
    def _property_image_setting_records(
        cls,
        image_channels: object,
    ) -> tuple[ModuleSetting, ...]:
        if not isinstance(image_channels, (tuple, list)):
            raise TypeError("image_channels must be a tuple or list.")
        channels = tuple(image_channels)
        for channel in channels:
            if not isinstance(channel, CPAImageChannelSpec):
                raise TypeError(
                    "image_channels must contain CPAImageChannelSpec values."
                )
        return (
            ModuleSetting(
                cls.property_image_count_setting.canonical,
                str(len(channels)),
            ),
            *(
                record
                for channel in channels
                for record in (
                    ModuleSetting(cls.property_image_setting.canonical, channel.alias),
                    ModuleSetting(
                        cls.property_use_image_name_setting.canonical,
                        cellprofiler_setting_literal(True),
                    ),
                    ModuleSetting(
                        cls.property_image_name_setting.canonical,
                        channel.image_name,
                    ),
                    ModuleSetting(
                        cls.property_channel_color_setting.canonical,
                        channel.channel_color,
                    ),
                )
            ),
        )

    @classmethod
    def _group_setting_records(
        cls,
        group_fields: object,
    ) -> tuple[ModuleSetting, ...]:
        if not isinstance(group_fields, (tuple, list)):
            raise TypeError("group_fields must be a tuple or list.")
        fields = tuple(group_fields)
        normalized: list[tuple[str, str]] = []
        for field in fields:
            if not isinstance(field, (tuple, list)) or len(field) != 2:
                raise TypeError(
                    "group_fields must contain two-item name/column sequences."
                )
            name, columns = (str(value).strip() for value in field)
            if not name or not columns:
                raise ValueError("group_fields names and columns cannot be empty.")
            normalized.append((name, columns))
        return (
            ModuleSetting(
                cls.property_group_count_setting.canonical,
                str(len(normalized)),
            ),
            *(
                record
                for name, columns in normalized
                for record in (
                    ModuleSetting(cls.group_name_setting.canonical, name),
                    ModuleSetting(cls.group_columns_setting.canonical, columns),
                )
            ),
        )

    @staticmethod
    def _block_with_records(
        block: ModuleBlock,
        additional_records: Sequence[ModuleSetting],
    ) -> ModuleBlock:
        records = [*block.iter_settings(), *additional_records]
        return replace(
            block,
            setting_records=records,
        )


@execution_scope(FunctionStepExecutionScope.PLATE)
@runtime_bound_parameters(RuntimeArtifactBatch)
def export_to_database(
    *,
    artifact_batch: RuntimeArtifactBatch,
    context: ProcessingContext,
    database_type: Literal["sqlite"] = "sqlite",
    sqlite_file: str = "DefaultDB.db",
    experiment_name: str = "MyExpt",
    add_table_prefix: bool = False,
    table_prefix: str = "",
    object_table_mode: CellProfilerObjectTableMode = CellProfilerObjectTableMode.PER_OBJECT,
    selected_objects: tuple[str, ...] | None = None,
    wants_properties_file: bool = True,
    wants_relationship_tables: bool = False,
    include_all_images: bool = True,
    image_channels: tuple[CPAImageChannelSpec, ...] = (),
    location_object: str | None = None,
    calculate_per_image_mean: bool = False,
    calculate_per_image_median: bool = False,
    calculate_per_image_standard_deviation: bool = False,
    calculate_per_well_mean: bool = False,
    calculate_per_well_median: bool = False,
    calculate_per_well_standard_deviation: bool = False,
    maximum_column_name_length: int = 64,
    image_url_prepend: str = "",
    write_image_thumbnails: bool = False,
    thumbnail_image_names: str = "",
    auto_scale_thumbnail_intensities: bool = True,
    plate_type: str = "None",
    plate_metadata: str = "Plate",
    well_metadata: str = "Well",
    wants_group_fields: bool = False,
    group_fields: tuple[tuple[str, str], ...] = (),
    wants_filter_fields: bool = False,
    create_plate_filters: bool = False,
    phenotype_class_table: str = "",
    overwrite_mode: Literal["never", "data_only", "data_and_schema"] = "never",
    access_images_via_url: bool = False,
    classification_type: Literal["object", "image"] = "object",
    wants_workspace_file: bool = False,
    workspace_measurements: tuple[
        tuple[str, str, str, str, str, str, str, str, str], ...
    ] = (),
) -> dict[str, bytes | str]:
    """Render exact contract-selected plate artifacts as SQLite and CPA files.

    Args:
        selected_objects: Object measurement subjects to export; use ``None``
            for all subjects or an empty tuple for no object subjects.
        image_channels: CellProfiler Analyst image-channel declarations to
            write when ``include_all_images`` is disabled.
        location_object: Object measurement subject whose center coordinates
            populate CellProfiler Analyst location columns; use ``None`` when
            object locations are not needed.
        group_fields: CellProfiler Analyst group rows as name and
            comma-separated per-image-column pairs, used when
            ``wants_group_fields`` is enabled.
        workspace_measurements: CellProfiler Analyst workspace display rows
            containing the tool, X-axis, and Y-axis measurement settings, used
            when ``wants_workspace_file`` is enabled.
    """

    settings = CellProfilerDatabaseExportSettings(
        database_type=database_type,
        sqlite_file=sqlite_file,
        experiment_name=experiment_name,
        table_prefix=table_prefix if add_table_prefix else "",
        object_table_mode=object_table_mode,
        selected_objects=selected_objects,
        wants_properties_file=wants_properties_file,
        wants_relationship_tables=wants_relationship_tables,
        maximum_column_name_length=maximum_column_name_length,
        location_object=location_object,
        plate_type=plate_type,
        plate_metadata=plate_metadata,
        well_metadata=well_metadata,
        image_url_prepend=image_url_prepend if access_images_via_url else "",
        group_fields=group_fields if wants_group_fields else (),
        classification_type=classification_type,
        phenotype_class_table=phenotype_class_table,
        calculate_per_image_mean=calculate_per_image_mean,
        calculate_per_image_median=calculate_per_image_median,
        calculate_per_image_standard_deviation=(calculate_per_image_standard_deviation),
        write_image_thumbnails=write_image_thumbnails,
        thumbnail_image_names=(
            split_symbol_names(thumbnail_image_names) if write_image_thumbnails else ()
        ),
        auto_scale_thumbnail_intensities=auto_scale_thumbnail_intensities,
    )
    resolved_channels = (
        CPAImageChannelSpec.defaults_for_artifacts(
            artifact_batch.specs_of_type(ImageArtifactType),
            source_binding_plan=artifact_batch.source_binding_plan,
        )
        if include_all_images
        else tuple(image_channels)
    )
    projection = CellProfilerAnalystProjectionBuilder(
        source_binding_plan=artifact_batch.source_binding_plan,
        context=context,
    ).build(
        artifact_batch,
        settings,
        resolved_channels,
    )
    dialect = projection.database_dialect(
        CellProfilerDatabaseColumnDialect(settings.table_prefix),
        settings,
    )
    bundle: dict[str, bytes | str] = {
        settings.sqlite_file: CPASQLiteRenderer(dialect).render(projection, settings)
    }
    for properties_file in CPAPropertiesRenderer(dialect).render(
        settings,
        resolved_channels,
        projection,
    ):
        if properties_file.file_name in bundle:
            raise ValueError(
                f"ExportToDatabase emits duplicate file {properties_file.file_name!r}."
            )
        bundle[properties_file.file_name] = properties_file.text
    return bundle
