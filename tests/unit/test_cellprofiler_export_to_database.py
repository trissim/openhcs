from contextlib import contextmanager
import inspect
from pathlib import Path
import sqlite3
from collections.abc import Iterator
from queue import SimpleQueue

import numpy as np
import pytest
import tifffile
from objectstate import config_context
from pycodify import Assignment, generate_python_source

import openhcs.serialization.pycodify_formatters  # noqa: F401
from objectstate.lazy_factory import ensure_global_config_context
from openhcs.constants.input_source import InputSource
from openhcs.core.artifacts import (
    ArtifactInputPlan,
    ObjectLabelsArtifactType,
    ArtifactOutputPlan,
    ArtifactSpec,
    ArtifactSpecCollection,
    ImageArtifactType,
    MeasurementsArtifactType,
    RelationshipsArtifactType,
)
from openhcs.core.callable_contract import (
    CallableContract,
    FunctionStepExecutionScope,
)
from openhcs.core.config import GlobalPipelineConfig, PipelineConfig, ProcessingConfig
from openhcs.core.compiled_step_plan import CompiledStepPlan
from openhcs.core.context.processing_context import ProcessingContext
from openhcs.core.input_workspace import InputWorkspacePreparationRequest
from openhcs.core.function_patterns import (
    FunctionInvocationKey,
    normalize_function_pattern,
)
from openhcs.core.function_step_transport import FunctionStepTransportAuthority
from openhcs.core.invocation_artifacts import ArtifactDeclarationStepContext
from openhcs.core.pipeline.artifact_planning import artifact_producers_for_outputs
from openhcs.core.measurement_row_materialization import (
    MeasurementSparseColumnarRows,
)
from openhcs.core.orchestrator.compiled_plate_execution import (
    validate_plate_scoped_contexts,
)
from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
from openhcs.core.progress import set_progress_queue
from openhcs.core.runtime_stores import RuntimeArtifactBatch
from openhcs.core.runtime_stores import RuntimeValueStore
from openhcs.core.runtime_artifact_values import RuntimeValue
from openhcs.core.steps.function_step import FunctionStep
from openhcs.core.runtime_measurements import MeasurementTable
from openhcs.core.runtime_relationships import (
    DirectedObjectRelationshipPayload,
    ObjectRelationshipDeclaration,
    ObjectRelationship,
)
from openhcs.core.runtime_measurements import MeasurementScope, MeasurementSubject
from openhcs.core.runtime_tabular_values import FieldSpec
from openhcs.core.source_image_provenance import SourceImageProvenancePlanes
from openhcs.core.source_bindings import (
    CompiledSourceBindingPlan,
    NamedSourceBinding,
    SourceProjectionRole,
    StepSourceBindingsConfig,
)
from openhcs.core.source_matching import SourceImageSetIdentityPolicy
from openhcs.interop.cellprofiler.analyst_export import (
    CPASQLiteRenderer,
    CellProfilerAnalystProjection,
    CellProfilerAnalystProjectionBuilder,
    CellProfilerDatabaseExportSettings,
    CellProfilerObjectTableMode,
)
from openhcs.interop.cellprofiler.database_column_dialect import (
    CellProfilerProjectedTable,
    CellProfilerColumnNameMapping,
)
from openhcs.interop.cellprofiler.module_declarations import CellProfilerModule
from openhcs.interop.cellprofiler.parser import CPPipeParser, ModuleBlock, ModuleSetting
from openhcs.interop.cellprofiler.pipeline_import import import_cellprofiler_pipeline
from openhcs.interop.cellprofiler.plate_workspace import (
    prepare_cellprofiler_input_workspace,
)
from openhcs.interop.cellprofiler.setting_names import (
    optional_setting_value,
    split_symbol_names,
)
from openhcs.interop.cellprofiler.settings_binder import SettingsBinder
from openhcs.processing.backends.cellprofiler.export_to_database import (
    ExportToDatabaseModule,
)
from openhcs.processing.backends.cellprofiler.relationships import RelateObjectsModule

OFFICIAL30_NATIVE_REFS = Path(__file__).parents[2] / "benchmark" / "native_refs"
OFFICIAL30_DATABASE_PIPELINES = (
    OFFICIAL30_NATIVE_REFS
    / "official30_scoped_rows"
    / "CellProfiler_tutorials_cp_tutorial_advanced_segmentation_final_wells_include_first1"
    / "native_cellprofiler_headless"
    / "BBBC022_Analysis_Final.cppipe",
    OFFICIAL30_NATIVE_REFS
    / "official30_scoped_rows"
    / "CellProfiler_tutorials_cp_tutorial_quality_control_wells_include_first1"
    / "native_cellprofiler_headless"
    / "BBBC022_QC.cppipe",
    OFFICIAL30_NATIVE_REFS
    / "official30_scoped_rows"
    / "CellProfiler_tutorials_cp_tutorial_translocation_final_wells_include_first1"
    / "native_cellprofiler_headless"
    / "Translocation_final.cppipe",
)

DATABASE_SEMANTIC_KWARGS = {
    "sqlite_file": "analysis.db",
    "experiment_name": "Experiment",
    "add_table_prefix": True,
    "table_prefix": "QC_",
    "object_table_mode": CellProfilerObjectTableMode.COMBINED,
    "image_url_prepend": "https://images.example/",
    "plate_type": "384",
    "plate_metadata": "Batch",
    "well_metadata": "Position",
    "wants_group_fields": True,
    "phenotype_class_table": "Phenotypes",
    "access_images_via_url": True,
    "classification_type": "image",
    "location_object": "Cells",
    "group_fields": (
        (
            "PerWell",
            "ImageNumber, Image_Metadata_Batch, Image_Metadata_Position",
        ),
    ),
}
REMOVED_EXPORT_FALSE_OPTIONS = {
    "calculate_per_well_mean",
    "calculate_per_well_median",
    "calculate_per_well_standard_deviation",
    "wants_filter_fields",
    "create_plate_filters",
    "overwrite_mode",
    "wants_workspace_file",
    "workspace_measurements",
}


def _projection_builder() -> CellProfilerAnalystProjectionBuilder:
    return CellProfilerAnalystProjectionBuilder(
        source_binding_plan=CompiledSourceBindingPlan.empty()
    )


def _export_context() -> ProcessingContext:
    context = ProcessingContext(
        step_plans={
            0: CompiledStepPlan(
                step_index=0,
                step_name="ExportToDatabase",
                step_type="FunctionStep",
                axis_id="A01",
                source_binding_plan=CompiledSourceBindingPlan.empty(),
            )
        }
    )
    context.plate_path = Path("/")
    return context


def test_cellprofiler_column_name_mapping_matches_native_global_grammar() -> None:
    names = (
        "ImageNumber",
        "ObjectNumber",
        "Cells_Mean_Mitochondria_Neighbors_FirstClosestObjectNumber_Expanded",
        "Cells_Mean_Mitochondria_Neighbors_SecondClosestObjectNumber_Expanded",
        "Bad Name",
        "Bad-Name",
    )

    mapping = CellProfilerColumnNameMapping(64, names)

    assert tuple(mapping.render(name) for name in names) == (
        "ImageNumber",
        "ObjectNumber",
        "Cells_Mean_Mitochondria_Neighbors_FirstClosestObjectNumbr_Expndd",
        "Cells_Mean_Mitochondria_Neighbors_SecondClosestObjectNmbr_Expndd",
        "Bad_Name",
        "Bad_Name1",
    )


def test_registered_plate_scoped_modules_resolve_previous_step_input() -> None:
    plate_declarations = tuple(
        (module_type, callable_contract)
        for module_type in CellProfilerModule.__registry__.values()
        for function_name in module_type.declared_function_names()
        for callable_contract in (
            CallableContract.from_callable(module_type.require_callable(function_name)),
        )
        if callable_contract.execution_scope is FunctionStepExecutionScope.PLATE
    )

    assert plate_declarations
    for module_type, callable_contract in plate_declarations:
        resolved = module_type.processing_config(
            callable_contract=callable_contract,
            inherited=ProcessingConfig(input_source=InputSource.PIPELINE_START),
        )

        assert resolved.input_source is InputSource.PREVIOUS_STEP


@pytest.mark.parametrize(
    "write_thumbnails",
    (False, True),
)
def test_database_plate_contract_preserves_exact_mixed_input_origins(
    write_thumbnails: bool,
) -> None:
    source_image = ArtifactSpec.input("SourceImage", ImageArtifactType)
    measurements = ArtifactSpec.output(
        "ImageMeasurements",
        MeasurementsArtifactType,
    )
    relationships = ArtifactSpec.output(
        "ObjectRelationships",
        RelationshipsArtifactType,
    )
    module = ModuleBlock(
        name=str(ExportToDatabaseModule.module_name),
        module_num=1,
        setting_records=(
            ModuleSetting(
                ExportToDatabaseModule.include_all_images_setting.canonical,
                "Yes",
            ),
            ModuleSetting(
                ExportToDatabaseModule.write_thumbnails_setting.canonical,
                "Yes" if write_thumbnails else "No",
            ),
            ModuleSetting(
                ExportToDatabaseModule.thumbnail_images_setting.canonical,
                source_image.name,
            ),
        ),
    )
    source_bindings = StepSourceBindingsConfig(
        enabled=True,
        bindings=(NamedSourceBinding(alias=source_image.name),),
    )
    invocation_key = FunctionInvocationKey(
        ExportToDatabaseModule.require_callable().__name__,
        "default",
        0,
    )

    contract = ExportToDatabaseModule.callable_contract(
        module=module,
        invocation_key=invocation_key,
        step_context=ArtifactDeclarationStepContext(
            step_name=module.name,
            step_index=0,
            source_bindings=source_bindings,
            available_artifacts=ArtifactSpecCollection(
                (source_image, measurements, relationships)
            ),
            main_flow_artifacts=ArtifactSpecCollection(()),
            available_artifact_producers=artifact_producers_for_outputs(
                (measurements, relationships),
                groups=(None,),
                invocation_keys=(
                    FunctionInvocationKey(
                        "fixture_producer",
                        invocation_key.group_key,
                        0,
                    ),
                ),
            ),
        ),
    )

    assert source_bindings.declares_artifact_ref(source_image.ref())
    assert contract.artifact_inputs.specs == (
        measurements.for_plan_type(ArtifactInputPlan),
        relationships.for_plan_type(ArtifactInputPlan),
        ArtifactSpec.input(
            source_image.name,
            source_image.artifact_type,
            required=False,
        ),
    )


def test_database_plate_contract_requires_produced_thumbnail_payload() -> None:
    derived_image = ArtifactSpec.output("DerivedImage", ImageArtifactType)
    module = ModuleBlock(
        name=str(ExportToDatabaseModule.module_name),
        module_num=2,
        setting_records=(
            ModuleSetting(
                ExportToDatabaseModule.include_all_images_setting.canonical,
                "No",
            ),
            ModuleSetting(
                ExportToDatabaseModule.write_thumbnails_setting.canonical,
                "Yes",
            ),
            ModuleSetting(
                ExportToDatabaseModule.thumbnail_images_setting.canonical,
                derived_image.name,
            ),
        ),
    )
    invocation_key = FunctionInvocationKey(
        ExportToDatabaseModule.require_callable().__name__,
        "default",
        0,
    )

    contract = ExportToDatabaseModule.callable_contract(
        module=module,
        invocation_key=invocation_key,
        step_context=ArtifactDeclarationStepContext(
            step_name=module.name,
            step_index=1,
            source_bindings=StepSourceBindingsConfig(),
            available_artifacts=ArtifactSpecCollection((derived_image,)),
            main_flow_artifacts=ArtifactSpecCollection(()),
            available_artifact_producers=artifact_producers_for_outputs(
                (derived_image,),
                groups=(None,),
                invocation_keys=(
                    FunctionInvocationKey(
                        "fixture_producer",
                        invocation_key.group_key,
                        0,
                    ),
                ),
            ),
        ),
    )

    assert contract.artifact_inputs.specs == (
        ArtifactSpec.input(
            derived_image.name,
            derived_image.artifact_type,
            required=True,
        ),
    )


def test_cppipe_import_codegen_and_transport_preserve_database_semantics(
    tmp_path: Path,
) -> None:
    cppipe_path = tmp_path / "database.cppipe"
    cppipe_path.write_text(_database_cppipe_text(), encoding="utf-8")

    steps, pipeline_config = import_cellprofiler_pipeline(cppipe_path)

    assert len(steps) == 1
    assert isinstance(pipeline_config, PipelineConfig)
    (invocation,) = tuple(normalize_function_pattern(steps[0].func).iter_items())
    imported_kwargs = invocation.kwargs_dict
    assert imported_kwargs == DATABASE_SEMANTIC_KWARGS

    (block,), _consumed = ExportToDatabaseModule.module_blocks_for_invocation(
        invocation=invocation,
        step_context=ArtifactDeclarationStepContext(
            step_name="ExportToDatabase",
            step_index=0,
        ),
    )
    assert block.get_setting_values(
        ExportToDatabaseModule.group_name_setting.canonical
    ) == ("PerWell",)
    assert (
        block.get_setting_values(ExportToDatabaseModule.group_columns_setting.canonical)
        == DATABASE_SEMANTIC_KWARGS["group_fields"][0][1:]
    )
    assert block.get_setting_values(
        ExportToDatabaseModule.aggregate_well_mean_setting.canonical
    ) == ("No",)
    assert block.get_setting_values(
        ExportToDatabaseModule.overwrite_mode_setting.canonical
    ) == ("Never",)
    assert block.get_setting_values(
        ExportToDatabaseModule.wants_workspace_file_setting.canonical
    ) == ("No",)

    pipeline_source = FunctionStepTransportAuthority.source_from_pipeline(steps)
    config_source = generate_python_source(
        Assignment("config", pipeline_config),
        clean_mode=True,
    )
    namespace: dict[str, object] = {}
    exec(compile(config_source, "<config>", "exec"), namespace)
    exec(compile(pipeline_source, "<pipeline>", "exec"), namespace)
    restored_steps = FunctionStepTransportAuthority.pipeline_steps_from_namespace(
        namespace
    )
    assert all(isinstance(step, FunctionStep) for step in restored_steps)
    (restored_invocation,) = tuple(
        normalize_function_pattern(restored_steps[0].func).iter_items()
    )

    assert isinstance(namespace["config"], PipelineConfig)
    assert restored_invocation.kwargs_dict == imported_kwargs


def test_database_export_public_contract_omits_false_options() -> None:
    parameter_names = set(
        inspect.signature(ExportToDatabaseModule.require_callable()).parameters
    )

    assert REMOVED_EXPORT_FALSE_OPTIONS.isdisjoint(parameter_names)


def test_database_import_rejects_enabled_unsupported_well_aggregation(
    tmp_path: Path,
) -> None:
    cppipe_path = tmp_path / "unsupported-per-well.cppipe"
    cppipe_path.write_text(
        _database_cppipe_text().replace(
            "    Experiment name:Experiment",
            "    Experiment name:Experiment\n"
            "    Calculate the per-well mean values of object measurements?:Yes",
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="does not support enabled.*per-well mean",
    ):
        import_cellprofiler_pipeline(cppipe_path)


@pytest.mark.parametrize("pipeline_path", OFFICIAL30_DATABASE_PIPELINES)
def test_official30_database_pipelines_import_and_round_trip_public_steps(
    pipeline_path: Path,
) -> None:
    steps, pipeline_config = import_cellprofiler_pipeline(pipeline_path)

    assert isinstance(pipeline_config, PipelineConfig)
    assert all(isinstance(step, FunctionStep) for step in steps)
    assert [step.name for step in steps].count("ExportToDatabase") == 1
    with config_context(pipeline_config):
        export_step = next(step for step in steps if step.name == "ExportToDatabase")
        assert export_step.processing_config.input_source is InputSource.PREVIOUS_STEP
        assert export_step.source_bindings.enabled is True
        assert export_step.source_bindings.binding_declarations

    source = FunctionStepTransportAuthority.source_from_pipeline(steps)
    namespace: dict[str, object] = {}
    exec(compile(source, str(pipeline_path.with_suffix(".py")), "exec"), namespace)
    restored_steps = FunctionStepTransportAuthority.pipeline_steps_from_namespace(
        namespace
    )

    assert all(isinstance(step, FunctionStep) for step in restored_steps)
    assert [step.name for step in restored_steps] == [step.name for step in steps]


def test_quality_control_plate_invocation_is_context_independent(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "20585"
    source_root.mkdir()
    for well in ("A01", "B01"):
        for channel in range(1, 6):
            tifffile.imwrite(
                source_root / f"IXMtest_{well}_s1_w{channel}.tif",
                np.full((4, 4), channel, dtype=np.uint16),
            )

    pipeline_path = OFFICIAL30_DATABASE_PIPELINES[1]
    prepared = prepare_cellprofiler_input_workspace(
        InputWorkspacePreparationRequest(
            selected_path=source_root,
            selected_pipeline_path=pipeline_path,
            workspace_root=tmp_path / "workspace",
        )
    )
    assert prepared.pipeline_import_error is None
    assert prepared.pipeline_steps is not None
    assert prepared.pipeline_config is not None

    ensure_global_config_context(GlobalPipelineConfig, GlobalPipelineConfig())
    orchestrator = PipelineOrchestrator(
        prepared.execution_plate_path,
        pipeline_config=prepared.pipeline_config,
    )
    set_progress_queue(SimpleQueue())
    try:
        orchestrator.initialize()
        compilation = orchestrator.compile_pipelines(
            pipeline_definition=prepared.pipeline_steps,
            well_filter=["A01", "B01"],
            is_zmq_execution=True,
        )
    finally:
        set_progress_queue(None)

    contexts = compilation.runtime_contexts
    (plate_step_index,) = validate_plate_scoped_contexts(contexts)
    invocations = tuple(
        context.step_plans[
            plate_step_index
        ].compiled_function_pattern.default_group.invocations[0]
        for context in contexts.values()
    )
    assert invocations[0] == invocations[1]

    image_specs = tuple(
        spec
        for spec in invocations[0].contract.artifact_inputs
        if spec.artifact_type is ImageArtifactType
    )
    assert tuple(spec.name for spec in image_specs) == (
        "OrigER",
        "OrigHoechst",
        "OrigMito",
        "OrigPh_golgi",
        "OrigSyto",
    )
    assert all(not spec.required for spec in image_specs)
    image_refs = frozenset(spec.ref() for spec in image_specs)
    image_edges = tuple(
        edge
        for edge in invocations[0].artifact_input_edges
        if edge.spec.ref() in image_refs
    )
    assert len(image_edges) == len(image_specs)
    assert all(edge.storage_plan is None for edge in image_edges)
    assert all(edge.projection is None for edge in image_edges)

    (output_plan,) = invocations[0].artifact_output_plans
    assert Path(output_plan.path).name == f"{output_plan.name}_step{plate_step_index}.pkl"
    input_storage_paths = tuple(
        tuple(
            edge.storage_plan.path
            for edge in invocation.artifact_input_edges
            if edge.storage_plan is not None
        )
        for invocation in invocations
    )
    assert input_storage_paths[0] != input_storage_paths[1]


def test_invocation_block_enrichment_preserves_parser_provenance(
    tmp_path: Path,
) -> None:
    cppipe_path = tmp_path / "database.cppipe"
    source_block = ModuleBlock(
        name="ExportToDatabase",
        module_num=3,
        metadata={"variable_revision_number": "27"},
        cppipe_path=cppipe_path,
    )
    setting = ModuleSetting(
        ExportToDatabaseModule.objects_choice_setting.canonical,
        "All",
    )

    enriched_block = ExportToDatabaseModule._block_with_records(
        source_block,
        (setting,),
    )

    assert enriched_block is not source_block
    assert enriched_block.cppipe_path == cppipe_path
    assert enriched_block.metadata == source_block.metadata
    assert enriched_block.get_setting_values(
        ExportToDatabaseModule.objects_choice_setting.canonical
    ) == ("All",)


def test_transported_database_semantics_reach_cpa_properties(tmp_path: Path) -> None:
    cppipe_path = tmp_path / "database.cppipe"
    cppipe_path.write_text(_database_cppipe_text(), encoding="utf-8")
    steps, _pipeline_config = import_cellprofiler_pipeline(cppipe_path)
    source = FunctionStepTransportAuthority.source_from_pipeline(steps)
    namespace: dict[str, object] = {}
    exec(compile(source, "<pipeline>", "exec"), namespace)
    (restored_step,) = FunctionStepTransportAuthority.pipeline_steps_from_namespace(
        namespace
    )
    (invocation,) = tuple(normalize_function_pattern(restored_step.func).iter_items())
    artifact_batch = RuntimeArtifactBatch(
        input_specs=(),
        records_by_axis={"A01": ()},
        source_image_set_identity_policy=SourceImageSetIdentityPolicy(),
    )

    bundle = invocation.contract.resolve_runtime_callable()(
        artifact_batch=artifact_batch,
        context=_export_context(),
        **invocation.kwargs_dict,
    )

    properties = _property_values(str(bundle["analysis_QC.properties"]))
    assert properties["plate_type"] == "384"
    assert properties["plate_id"] == ""
    assert properties["well_id"] == ""
    assert properties["image_url_prepend"] == "https://images.example/"
    assert properties["classification_type"] == "image"
    assert properties["class_table"] == "QC_Phenotypes"
    assert properties["group_SQL_PerWell"] == (
        "SELECT ImageNumber, Image_Metadata_Batch, Image_Metadata_Position "
        "FROM QC_Per_Image"
    )


@pytest.mark.parametrize("pipeline_path", OFFICIAL30_DATABASE_PIPELINES)
def test_official30_database_modules_bind_and_build_exact_contracts(
    pipeline_path: Path,
) -> None:
    if not pipeline_path.exists():
        pytest.skip(f"official30 pipeline cache is absent: {pipeline_path}")

    export_modules = tuple(
        module
        for module in CPPipeParser().parse(pipeline_path)
        if module.name == ExportToDatabaseModule.module_name
    )
    assert len(export_modules) == 1
    (module,) = export_modules

    bound = ExportToDatabaseModule.bind_settings(module, binder=SettingsBinder())
    assert bound.unmapped_kwargs == {}
    assert REMOVED_EXPORT_FALSE_OPTIONS.isdisjoint(bound.kwargs)
    assert len(bound.setting_coverage) == len(module.iter_settings())

    thumbnail_value = optional_setting_value(
        module,
        ExportToDatabaseModule.thumbnail_images_setting,
    )
    thumbnail_names = (
        split_symbol_names(thumbnail_value) if thumbnail_value is not None else ()
    )
    source_names = thumbnail_names if thumbnail_names else ("SourceImage",)
    available = ArtifactSpecCollection(
        (
            ArtifactSpec.output("ImageMeasurements", MeasurementsArtifactType),
            *(ArtifactSpec.input(name, ImageArtifactType) for name in source_names),
            ArtifactSpec.output("GeneratedImage", ImageArtifactType),
            ArtifactSpec.output("ObjectRelationships", RelationshipsArtifactType),
        )
    )
    invocation_key = FunctionInvocationKey("export_to_database", "default", 0)
    contract = ExportToDatabaseModule.callable_contract(
        module=module,
        invocation_key=invocation_key,
        step_context=ArtifactDeclarationStepContext(
            step_name="ExportToDatabase",
            step_index=module.module_num - 1,
            source_bindings=StepSourceBindingsConfig(
                enabled=True,
                bindings=tuple(
                    NamedSourceBinding(
                        alias=name,
                        projection_role=SourceProjectionRole.SOURCE_ARTIFACT,
                    )
                    for name in source_names
                ),
            ),
            available_artifacts=available,
            main_flow_artifacts=ArtifactSpecCollection(()),
            available_artifact_producers=artifact_producers_for_outputs(
                tuple(
                    spec for spec in available if spec.plan_type is ArtifactOutputPlan
                ),
                groups=(None,),
                invocation_keys=(
                    FunctionInvocationKey(
                        "fixture_producer",
                        invocation_key.group_key,
                        0,
                    ),
                ),
            ),
        ),
    )

    assert contract.artifact_inputs.names() == (
        "ImageMeasurements",
        *source_names,
        "GeneratedImage",
        "ObjectRelationships",
    )


def test_sqlite_schema_retains_zero_row_declared_measurement_fields() -> None:
    store = RuntimeValueStore()
    image_spec = _record_measurement_table(
        store,
        MeasurementTable(
            name="ImageMeasurements",
            rows=MeasurementSparseColumnarRows.from_rows(
                (),
                fields=(
                    FieldSpec("slice_index", int),
                    FieldSpec("Count_Nuclei", int, required=False),
                ),
            ),
            subject=MeasurementSubject(MeasurementScope.IMAGE, "Image"),
        ),
    )
    object_spec = _record_measurement_table(
        store,
        MeasurementTable(
            name="NucleiMeasurements",
            rows=MeasurementSparseColumnarRows.from_rows(
                (),
                fields=(
                    FieldSpec("slice_index", int),
                    FieldSpec("object_label", int),
                    FieldSpec("AreaShape_Area", float, required=False),
                ),
            ),
            subject=MeasurementSubject(
                MeasurementScope.OBJECT,
                "Nuclei",
                "object_label",
            ),
        ),
    )
    experiment_spec = _record_measurement_table(
        store,
        MeasurementTable(
            name="ExperimentMeasurements",
            rows=MeasurementSparseColumnarRows.from_rows(
                (
                    {
                        "Pipeline_Pipeline": b"CellProfiler Pipeline",
                        "CellProfiler_Version": "4.2.8",
                        "Threshold_Std": 3.0,
                        "Threshold_Mean": 1.0,
                        "Threshold_Median": 2.0,
                    },
                ),
                fields=(
                    FieldSpec("Pipeline_Pipeline", bytes, required=False),
                    FieldSpec("CellProfiler_Version", str, required=False),
                    FieldSpec("Threshold_Std", float, required=False),
                    FieldSpec("Threshold_Mean", float, required=False),
                    FieldSpec("Threshold_Median", float, required=False),
                ),
            ),
            subject=MeasurementSubject(MeasurementScope.EXPERIMENT),
        ),
    )
    batch = RuntimeArtifactBatch(
        input_specs=(image_spec, object_spec, experiment_spec),
        records_by_axis={"A01": store.values()},
        source_image_set_identity_policy=SourceImageSetIdentityPolicy(),
    )
    settings = _cpa_settings(object_table_mode=CellProfilerObjectTableMode.COMBINED)
    projection = _projection_builder().build(batch, settings, ())

    with _sqlite_connection(CPASQLiteRenderer().render(projection, settings)) as db:
        assert _table_info(db, "CPA_Per_Image") == (
            ("ImageNumber", "INTEGER", 0, 1),
            ("Image_Count_Nuclei", "INTEGER", 0, 0),
            ("Image_Group_Index", "INTEGER", 0, 0),
            ("Image_Group_Length", "INTEGER", 0, 0),
            ("Image_Group_Number", "INTEGER", 0, 0),
        )
        assert _table_info(db, "CPA_Per_Object") == (
            ("ImageNumber", "INTEGER", 0, 1),
            ("ObjectNumber", "INTEGER", 0, 2),
            ("Nuclei_Number_Object_Number", "INTEGER", 0, 0),
            ("Nuclei_AreaShape_Area", "float", 0, 0),
        )
        assert _table_info(db, "CPA_Per_Experiment") == (
            ("Pipeline_Pipeline", "longblob", 0, 0),
            ("CellProfiler_Version", "TEXT", 0, 0),
            ("Run_Timestamp", "TEXT", 0, 0),
            ("Modification_Timestamp", "TEXT", 0, 0),
            ("Threshold_Std", "float", 0, 0),
            ("Threshold_Mean", "float", 0, 0),
            ("Threshold_Median", "float", 0, 0),
        )
        assert _table_info(db, "Experiment") == (
            ("experiment_id", "INTEGER", 0, 1),
            ("name", "TEXT", 0, 0),
        )
        assert _table_info(db, "Experiment_Properties") == (
            ("experiment_id", "INTEGER", 1, 1),
            ("object_name", "TEXT", 1, 2),
            ("field", "TEXT", 1, 3),
            ("value", "longtext", 0, 0),
        )
        assert db.execute(
            'SELECT Pipeline_Pipeline, CellProfiler_Version FROM "CPA_Per_Experiment"'
        ).fetchall() == [(b"CellProfiler Pipeline", "4.2.8")]
        property_fields = {
            row[0]
            for row in db.execute(
                'SELECT field FROM "Experiment_Properties" WHERE object_name = "Object"'
            )
        }
        assert {
            "db_type",
            "image_table",
            "object_table",
            "image_id",
            "object_id",
            "image_path_cols",
            "image_file_cols",
            "object_name",
            "classifier_ignore_columns",
            "classification_type",
            "check_tables",
            "force_bioformats",
            "use_legacy_fetcher",
            "process_3D",
        } <= property_fields


def test_combined_object_schema_orders_subjects_and_identifiers_exactly() -> None:
    object_tables = tuple(
        CellProfilerProjectedTable(
            table_name=f"CPA_Per_{object_name}",
            rows=(),
            columns=(
                FieldSpec("ImageNumber", int),
                FieldSpec(
                    f"{object_name}_Number_Object_Number",
                    int,
                ),
                *(
                    FieldSpec(
                        f"{object_name}_{feature_name}",
                        float,
                        required=False,
                    )
                    for feature_name in ("Feature_Z", "Feature_A")
                ),
            ),
            subject=MeasurementSubject(MeasurementScope.OBJECT, object_name),
        )
        for object_name in ("Nuclei", "Cells", "Cytoplasm")
    )
    projection = CellProfilerAnalystProjection(
        image_table=CellProfilerProjectedTable(
            "CPA_Per_Image",
            (),
            (),
            MeasurementSubject(MeasurementScope.IMAGE, "Image"),
        ),
        object_tables=object_tables,
        relationship_tables=(),
        experiment_table=CellProfilerProjectedTable(
            "CPA_Per_Experiment",
            (),
            (),
            MeasurementSubject(MeasurementScope.EXPERIMENT, "Experiment"),
        ),
    )
    settings = _cpa_settings(object_table_mode=CellProfilerObjectTableMode.COMBINED)

    with _sqlite_connection(CPASQLiteRenderer().render(projection, settings)) as db:
        assert tuple(
            row[1] for row in db.execute('PRAGMA table_info("CPA_Per_Object")')
        ) == (
            "ImageNumber",
            "ObjectNumber",
            "Cells_Number_Object_Number",
            "Cells_Feature_A",
            "Cells_Feature_Z",
            "Cytoplasm_Number_Object_Number",
            "Cytoplasm_Feature_A",
            "Cytoplasm_Feature_Z",
            "Nuclei_Number_Object_Number",
            "Nuclei_Feature_A",
            "Nuclei_Feature_Z",
        )


def test_projection_keeps_mixed_image_and_object_subjects_separate() -> None:
    store = RuntimeValueStore()
    mixed_spec = _record_measurement_table(
        store,
        MeasurementTable(
            name="MixedMeasurements",
            rows=MeasurementSparseColumnarRows.from_rows(
                (
                    {
                        "slice_index": 0,
                        "feature_name": "Count_Nuclei",
                        "result_value": 2,
                    },
                    {
                        "slice_index": 0,
                        "object_name": "Nuclei",
                        "object_label": 1,
                        "feature_name": "AreaShape_Area",
                        "result_value": 24.0,
                    },
                ),
                fields=(
                    FieldSpec("slice_index", int),
                    FieldSpec("object_name", str, required=False),
                    FieldSpec("object_label", int, required=False),
                    FieldSpec("feature_name", str),
                    FieldSpec("result_value", float, required=False),
                ),
            ),
            subject=MeasurementSubject(MeasurementScope.IMAGE, "Image"),
        ),
    )
    batch = RuntimeArtifactBatch(
        input_specs=(mixed_spec,),
        records_by_axis={"A01": store.values()},
        source_image_set_identity_policy=SourceImageSetIdentityPolicy(),
    )

    projection = _projection_builder().build(
        batch,
        _cpa_settings(),
        (),
    )

    assert _field_rows(projection.image_table)[0] == {
        "ImageNumber": 1,
        "Image_Count_Nuclei": 2,
        "Image_Group_Index": 1,
        "Image_Group_Length": 1,
        "Image_Group_Number": 1,
    }
    assert len(projection.object_tables) == 1
    assert projection.object_tables[0].subject == MeasurementSubject(
        MeasurementScope.OBJECT,
        "Nuclei",
    )
    assert _field_rows(projection.object_tables[0]) == (
        {
            "ImageNumber": 1,
            "Nuclei_Number_Object_Number": 1,
            "Nuclei_AreaShape_Area": 24.0,
        },
    )


def test_long_projection_uses_nominal_field_for_matching_row_feature() -> None:
    store = RuntimeValueStore()
    measurement_spec = _record_measurement_table(
        store,
        MeasurementTable(
            name="IdentifySecondaryObjects_2_measurements",
            rows=MeasurementSparseColumnarRows.from_rows(
                (
                    {
                        "slice_index": 0,
                        "object_label": 1,
                        "feature_name": "Parent_Nuclei",
                        "result_value": 1,
                        "Parent_Nuclei": 1,
                    },
                ),
                fields=(
                    FieldSpec("slice_index", int),
                    FieldSpec("object_label", int),
                    FieldSpec("feature_name", str),
                    FieldSpec("result_value", int, required=False),
                    FieldSpec("Parent_Nuclei", int, required=True),
                ),
            ),
            subject=MeasurementSubject(
                MeasurementScope.OBJECT,
                "Cells",
                "object_label",
            ),
        ),
    )
    batch = RuntimeArtifactBatch(
        input_specs=(measurement_spec,),
        records_by_axis={"A01": store.values()},
        source_image_set_identity_policy=SourceImageSetIdentityPolicy(),
    )

    projection = _projection_builder().build(
        batch,
        _cpa_settings(),
        (),
    )

    assert projection.object_tables[0].columns == (
        FieldSpec("ImageNumber", int),
        FieldSpec("Cells_Number_Object_Number", int),
        FieldSpec("Cells_Parent_Nuclei", int),
    )
    with _sqlite_connection(
        CPASQLiteRenderer().render(projection, _cpa_settings())
    ) as db:
        assert _table_info(db, "CPA_Per_Cells") == (
            ("ImageNumber", "INTEGER", 0, 1),
            ("Cells_Number_Object_Number", "INTEGER", 0, 2),
            ("Cells_Parent_Nuclei", "INTEGER", 0, 0),
        )


def test_relate_objects_database_schema_preserves_native_integer_distance_type() -> None:
    store = RuntimeValueStore()
    measurement_spec = _record_measurement_table(
        store,
        MeasurementTable(
            name="RelateObjects_measurements",
            rows=MeasurementSparseColumnarRows.from_rows(
                (
                    {
                        "slice_index": 0,
                        "object_label": 1,
                        "Distance_Centroid_Cells": 2.5,
                        "Distance_Minimum_Cells": 1.5,
                    },
                ),
                fields=(
                    FieldSpec("slice_index", int),
                    FieldSpec("object_label", int),
                    FieldSpec("Distance_Centroid_Cells", float),
                    FieldSpec("Distance_Minimum_Cells", float),
                ),
            ),
            subject=MeasurementSubject(
                MeasurementScope.OBJECT,
                "Mitochondria",
                "object_label",
            ),
            measurement_feature_owner=RelateObjectsModule,
        ),
    )
    batch = RuntimeArtifactBatch(
        input_specs=(measurement_spec,),
        records_by_axis={"A01": store.values()},
        source_image_set_identity_policy=SourceImageSetIdentityPolicy(),
    )
    settings = _cpa_settings()
    projection = _projection_builder().build(batch, settings, ())

    with _sqlite_connection(CPASQLiteRenderer().render(projection, settings)) as db:
        columns = dict(
            (name, declared_type)
            for _index, name, declared_type, *_rest in db.execute(
                'PRAGMA table_info("CPA_Per_Mitochondria")'
            )
        )
        assert columns["Mitochondria_Distance_Centroid_Cells"] == "INTEGER"
        assert columns["Mitochondria_Distance_Minimum_Cells"] == "INTEGER"


def test_literal_none_location_is_preserved_for_combined_cpa_export(
    tmp_path: Path,
) -> None:
    cppipe_path = tmp_path / "database.cppipe"
    cppipe_path.write_text(
        _database_cppipe_text().replace(
            "Which objects should be used for locations?:Cells",
            "Which objects should be used for locations?:None",
        ),
        encoding="utf-8",
    )
    steps, _pipeline_config = import_cellprofiler_pipeline(cppipe_path)
    (invocation,) = tuple(normalize_function_pattern(steps[0].func).iter_items())
    assert invocation.kwargs_dict["location_object"] == "None"

    bundle = invocation.contract.resolve_runtime_callable()(
        artifact_batch=RuntimeArtifactBatch(
            input_specs=(),
            records_by_axis={"A01": ()},
            source_image_set_identity_policy=SourceImageSetIdentityPolicy(),
        ),
        context=_export_context(),
        **invocation.kwargs_dict,
    )
    properties = _property_values(str(bundle["analysis_QC.properties"]))
    assert properties["cell_x_loc"] == "None_Location_Center_X"
    assert properties["cell_y_loc"] == "None_Location_Center_Y"
    assert properties["cell_z_loc"] == "None_Location_Center_Z"


def test_sqlite_renders_canonical_relationship_schema_indexes_and_view() -> None:
    store = RuntimeValueStore()
    relationship = ObjectRelationship(
        name="Cells_Nuclei_relationships",
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            component_metadata=({"well": "A01", "site": "1"},)
        ),
        declaration=ObjectRelationshipDeclaration(
            source=ArtifactSpec.output("Cells", ObjectLabelsArtifactType).ref(),
            target=ArtifactSpec.output("Nuclei", ObjectLabelsArtifactType).ref(),
            relationship_type="Parent",
            source_role="parent",
            target_role="child",
            source_id_field="parent_id",
            target_id_field="child_id",
            producer_module_number=7,
            source_runtime_slice_offset=0,
            target_runtime_slice_offset=0,
        ),
        payload=DirectedObjectRelationshipPayload(
            source_ids=(10, 10),
            target_ids=(1, 2),
            slice_indices=(0, 0),
            slice_count=None,
        ),
    )
    relationship_spec = _record_relationship(store, relationship)
    batch = RuntimeArtifactBatch(
        input_specs=(relationship_spec,),
        records_by_axis={"A01": store.values()},
        source_image_set_identity_policy=SourceImageSetIdentityPolicy(),
    )
    settings = _cpa_settings(wants_relationship_tables=True)
    projection = _projection_builder().build(batch, settings, ())

    with _sqlite_connection(CPASQLiteRenderer().render(projection, settings)) as db:
        assert db.execute(
            "SELECT relationship_type_id, module_number, relationship, "
            'object_name1, object_name2 FROM "CPA_Per_RelationshipTypes"'
        ).fetchall() == [(1, 7, "Parent", "Cells", "Nuclei")]
        assert db.execute(
            'SELECT * FROM "CPA_Per_Relationships" ORDER BY object_number2'
        ).fetchall() == [
            (1, 1, 10, 1, 1),
            (1, 1, 10, 1, 2),
        ]
        assert db.execute(
            'SELECT * FROM "CPA_Per_RelationshipsView" ORDER BY object_number2'
        ).fetchall() == [
            (7, "Parent", "Cells", "Nuclei", 1, 10, 1, 1),
            (7, "Parent", "Cells", "Nuclei", 1, 10, 1, 2),
        ]
        indexes = {
            row[1] for row in db.execute('PRAGMA index_list("CPA_Per_Relationships")')
        }
        assert {"CPA_IRelationships1", "CPA_IRelationships2"} <= indexes
        assert (
            db.execute('PRAGMA foreign_key_list("CPA_Per_Relationships")').fetchall()[
                0
            ][2]
            == "CPA_Per_RelationshipTypes"
        )


def test_object_view_uses_keyed_inner_joins_instead_of_union() -> None:
    projection = CellProfilerAnalystProjection(
        image_table=CellProfilerProjectedTable(
            "CPA_Per_Image",
            ({"ImageNumber": 1},),
            (FieldSpec("ImageNumber", int),),
            MeasurementSubject(MeasurementScope.IMAGE, "Image"),
        ),
        object_tables=(
            CellProfilerProjectedTable(
                "CPA_Per_Cells",
                (
                    {
                        "ImageNumber": 1,
                        "Cells_Number_Object_Number": 1,
                        "Cells_Area": 10.0,
                    },
                    {
                        "ImageNumber": 1,
                        "Cells_Number_Object_Number": 2,
                        "Cells_Area": 20.0,
                    },
                ),
                (
                    FieldSpec("ImageNumber", int),
                    FieldSpec("Cells_Number_Object_Number", int),
                    FieldSpec("Cells_Area", float),
                ),
                MeasurementSubject(MeasurementScope.OBJECT, "Cells"),
            ),
            CellProfilerProjectedTable(
                "CPA_Per_Nuclei",
                (
                    {
                        "ImageNumber": 1,
                        "Nuclei_Number_Object_Number": 1,
                        "Nuclei_Intensity": 0.5,
                    },
                    {
                        "ImageNumber": 1,
                        "Nuclei_Number_Object_Number": 3,
                        "Nuclei_Intensity": 0.8,
                    },
                ),
                (
                    FieldSpec("ImageNumber", int),
                    FieldSpec("Nuclei_Number_Object_Number", int),
                    FieldSpec("Nuclei_Intensity", float),
                ),
                MeasurementSubject(MeasurementScope.OBJECT, "Nuclei"),
            ),
        ),
        relationship_tables=(),
        experiment_table=CellProfilerProjectedTable(
            "CPA_Per_Experiment",
            (),
            (),
            MeasurementSubject(MeasurementScope.EXPERIMENT, "Experiment"),
        ),
    )
    settings = _cpa_settings(object_table_mode=CellProfilerObjectTableMode.VIEW)

    with _sqlite_connection(CPASQLiteRenderer().render(projection, settings)) as db:
        view_sql = db.execute(
            "SELECT sql FROM sqlite_master WHERE type='view' AND name='CPA_Per_Object'"
        ).fetchone()[0]
        assert "INNER JOIN" in view_sql
        assert "UNION" not in view_sql
        assert db.execute(
            'SELECT ImageNumber, ObjectNumber FROM "CPA_Per_Object"'
        ).fetchall() == [(1, 1)]


def _database_cppipe_text() -> str:
    return "\n".join(
        (
            "CellProfiler Pipeline: http://www.cellprofiler.org",
            "Version:3",
            "ExportToDatabase:[module_num:1|enabled:True]",
            "    Database type:SQLite",
            "    Name the SQLite database file:analysis.db",
            "    Experiment name:Experiment",
            "    Add a prefix to table names?:Yes",
            "    Table prefix:QC_",
            "    Create a CellProfiler Analyst properties file?:Yes",
            "    Export measurements for all objects to the database?:All",
            "    Export object relationships?:No",
            "    Create one table per object, a single object table or a single object view?:Single object table",
            "    Include information for all images, using default values?:Yes",
            "    Select the plate type:384",
            "    Select the plate metadata:Batch",
            "    Select the well metadata:Position",
            "    Properties image group count:0",
            "    Which objects should be used for locations?:Cells",
            "    Properties group field count:1",
            "    Do you want to add group fields?:Yes",
            "    Enter the name of the group:PerWell",
            "    Enter the per-image columns which define the group, separated by commas:ImageNumber, Image_Metadata_Batch, Image_Metadata_Position",
            "    Properties filter field count:0",
            "    Do you want to add filter fields?:No",
            "    Automatically create a filter for each plate?:No",
            "    Enter a phenotype class table name if using the Classifier tool in CellProfiler Analyst:Phenotypes",
            "    Access CellProfiler Analyst images via URL?:Yes",
            "    Enter an image url prepend if you plan to access your files via http:https://images.example/",
            "    Select the classification type:Image",
            "    Workspace measurement count:0",
            "    Create a CellProfiler Analyst workspace file?:No",
            "",
        )
    )


def _property_values(text: str) -> dict[str, str]:
    return {
        key.strip(): value.strip()
        for line in text.splitlines()
        if "=" in line and not line.lstrip().startswith("#")
        for key, value in (line.split("=", 1),)
    }


def _field_rows(
    table: CellProfilerProjectedTable,
) -> tuple[dict[str, object], ...]:
    return tuple(dict(row) for row in table.rows)


def _cpa_settings(**overrides: object) -> CellProfilerDatabaseExportSettings:
    values: dict[str, object] = {
        "sqlite_file": "analysis.db",
        "experiment_name": "Experiment",
        "table_prefix": "CPA_",
        "object_table_mode": CellProfilerObjectTableMode.PER_OBJECT,
        "selected_objects": None,
        "wants_properties_file": True,
        "wants_relationship_tables": False,
    }
    values.update(overrides)
    return CellProfilerDatabaseExportSettings(**values)


def _record_measurement_table(
    store: RuntimeValueStore,
    table: MeasurementTable,
) -> ArtifactSpec:
    table = table.replace_fields(
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            component_metadata=({"well": "A01", "site": "1"},)
        )
    )
    output_plan = ArtifactOutputPlan(
        name=table.name,
        path=f"/memory/{table.name}.pkl",
        artifact_type=MeasurementsArtifactType,
    )
    store.record(
        RuntimeValue.normalize(output_plan, table, axis_id="A01"),
        path=output_plan.path,
        backend="memory",
    )
    return ArtifactSpec.input(table.name, MeasurementsArtifactType)


def _record_relationship(
    store: RuntimeValueStore,
    relationship: ObjectRelationship,
) -> ArtifactSpec:
    output_plan = ArtifactOutputPlan(
        name=relationship.name,
        path=f"/memory/{relationship.name}.pkl",
        artifact_type=RelationshipsArtifactType,
    )
    store.record(
        RuntimeValue.normalize(output_plan, relationship, axis_id="A01"),
        path=output_plan.path,
        backend="memory",
    )
    return ArtifactSpec.input(relationship.name, RelationshipsArtifactType)


@contextmanager
def _sqlite_connection(payload: bytes) -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(":memory:")
    try:
        connection.deserialize(payload)
        yield connection
    finally:
        connection.close()


def _table_info(
    connection: sqlite3.Connection,
    table_name: str,
) -> tuple[tuple[str, str, int, int], ...]:
    escaped_name = table_name.replace('"', '""')
    return tuple(
        (str(row[1]), str(row[2]), int(row[3]), int(row[5]))
        for row in connection.execute(f'PRAGMA table_info("{escaped_name}")')
    )
