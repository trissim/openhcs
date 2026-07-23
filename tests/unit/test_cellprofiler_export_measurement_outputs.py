from __future__ import annotations

from base64 import b64decode
import inspect
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from openhcs.core.artifacts import (
    ArtifactInputPlan,
    ArtifactOutputPlan,
    ArtifactSidecarRole,
    ArtifactSpec,
    ArtifactSpecCollection,
    ArtifactSpecRelation,
    ImageArtifactType,
    MeasurementsArtifactType,
    SpecialArtifactType,
)
from openhcs.core.compiled_step_plan import CompiledStepPlan
from openhcs.core.context.processing_context import ProcessingContext
from openhcs.core.function_patterns import (
    FunctionInvocationKey,
    compile_function_pattern,
    normalize_function_pattern,
)
from openhcs.core.invocation_artifacts import ArtifactDeclarationStepContext
from openhcs.core.pipeline.artifact_planning import artifact_producers_for_outputs
from openhcs.core.runtime_artifact_values import RuntimeValue
from openhcs.core.runtime_image_values import ImagePayloadMetadata
from openhcs.core.runtime_measurements import MeasurementTable
from openhcs.core.runtime_measurements import (
    MeasurementRowAxisField,
    MeasurementScope,
)
from openhcs.core.runtime_stores import RuntimeArtifactBatch, RuntimeValueStore
from openhcs.core.source_bindings import CompiledSourceBindingPlan
from openhcs.core.source_matching import (
    SourceImageSetIdentityPolicy,
    with_original_source_metadata,
)
from openhcs.core.steps.function_runtime import FunctionOutputContextStrategy
from openhcs.interop.cellprofiler.parser import ModuleBlock, ModuleSetting
from openhcs.interop.cellprofiler.setting_names import optional_setting_value
from openhcs.processing.backends.cellprofiler.export_to_database import (
    ExportToDatabaseModule,
    export_to_database,
)
from openhcs.processing.backends.cellprofiler.save_images import (
    SaveImagesFilenameMethod,
    SaveImagesModule,
    SaveImagesRecordedMeasurementSourceRelation,
    save_images,
    save_images_with_measurements,
)


def _module(name: str, **settings: str) -> ModuleBlock:
    records = [
        ModuleSetting(setting_name, value) for setting_name, value in settings.items()
    ]
    return ModuleBlock(
        name=name,
        module_num=1,
        setting_records=records,
    )


def _context(
    *,
    step_name: str,
    main_flow_image_names: tuple[str, ...] = (),
    runtime_image_names: tuple[str, ...] = (),
) -> ArtifactDeclarationStepContext:
    overlapping_names = frozenset(main_flow_image_names) & frozenset(
        runtime_image_names
    )
    if overlapping_names:
        raise ValueError(
            f"Fixture images cannot have multiple input owners: {overlapping_names!r}."
        )
    main_flow_images = tuple(
        ArtifactSpec.output(image_name, ImageArtifactType)
        for image_name in main_flow_image_names
    )
    runtime_images = tuple(
        ArtifactSpec.output(image_name, ImageArtifactType)
        for image_name in runtime_image_names
    )
    images = (*main_flow_images, *runtime_images)
    return ArtifactDeclarationStepContext(
        step_name=step_name,
        step_index=0,
        available_artifacts=ArtifactSpecCollection(images),
        main_flow_artifacts=ArtifactSpecCollection(
            image.for_plan_type(ArtifactInputPlan) for image in main_flow_images
        ),
        available_artifact_producers=artifact_producers_for_outputs(
            images,
            groups=(None,),
            invocation_keys=(FunctionInvocationKey("fixture_producer", "default", 0),),
        ),
    )


def _contract(
    module_type,
    module: ModuleBlock,
    function_name: str,
    context: ArtifactDeclarationStepContext,
):
    return module_type.callable_contract(
        module=module,
        invocation_key=FunctionInvocationKey(function_name, "default", 0),
        step_context=context,
    )


def _invocation_contract(module_type, invocation, context):
    blocks, consumed_names = module_type.module_blocks_for_invocation(
        invocation=invocation,
        step_context=context,
    )
    (numbered_blocks,), _next_module_num = module_type.number_step_invocation_blocks(
        (blocks,),
        first_module_num=1,
    )
    return module_type.invocation_callable_contract(
        invocation=invocation,
        numbered_module_blocks=numbered_blocks,
        consumed_kwarg_names=consumed_names,
        step_context=context,
    )


def _runtime_image_batch(*image_names: str) -> RuntimeArtifactBatch:
    store = RuntimeValueStore()
    input_specs: list[ArtifactSpec] = []
    for index, image_name in enumerate(image_names, start=1):
        source_path = f"/input/{image_name}.png"
        component_metadata = {"Well": "A01", "Site": "1"}
        output_plan = ArtifactOutputPlan(
            name=image_name,
            path=f"/memory/{image_name}.pkl",
            artifact_type=ImageArtifactType,
        )
        payload = ImagePayloadMetadata(
            source_path=source_path,
            source_component_metadata=with_original_source_metadata(
                component_metadata,
                component_metadata,
                path=source_path,
            ),
        ).payload_with(np.full((12, 10), index / 3, dtype=np.float32), None)
        store.record(
            RuntimeValue.normalize(output_plan, payload, axis_id="A01"),
            path=output_plan.path,
            backend="memory",
        )
        input_specs.append(ArtifactSpec.input(image_name, ImageArtifactType))
    return RuntimeArtifactBatch(
        input_specs=tuple(input_specs),
        records_by_axis={"A01": store.values()},
        source_image_set_identity_policy=SourceImageSetIdentityPolicy(),
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
                compiled_function_pattern=compile_function_pattern(
                    export_to_database,
                    {},
                    {},
                ),
            )
        },
        filemanager=SimpleNamespace(exists=lambda *_args: False),
    )
    context.plate_path = Path("/")
    context.microscope_handler = SimpleNamespace(
        metadata_handler=SimpleNamespace(
            source_workspace_metadata_document=lambda _plate_path: None
        )
    )
    return context


def test_export_to_database_thumbnail_setting_preserves_one_terminal_output() -> None:
    inactive = _contract(
        ExportToDatabaseModule,
        _module(
            "ExportToDatabase",
            **{
                "Write image thumbnails directly to the database?": "No",
                "Select the images for which you want to save thumbnails": "DNA, RNA",
            },
        ),
        export_to_database.__name__,
        _context(
            step_name=str(ExportToDatabaseModule.module_name),
            runtime_image_names=("DNA", "RNA"),
        ),
    )
    active = _contract(
        ExportToDatabaseModule,
        _module(
            "ExportToDatabase",
            **{
                "Write image thumbnails directly to the database?": "Yes",
                "Select the images for which you want to save thumbnails": "DNA, RNA",
            },
        ),
        export_to_database.__name__,
        _context(
            step_name=str(ExportToDatabaseModule.module_name),
            runtime_image_names=("DNA", "RNA"),
        ),
    )

    assert tuple(spec.artifact_type for spec in inactive.artifact_outputs) == (
        SpecialArtifactType,
    )
    assert tuple(spec.artifact_type for spec in active.artifact_outputs) == (
        SpecialArtifactType,
    )


def test_export_to_database_writes_selected_thumbnails_into_sqlite() -> None:
    bundle = export_to_database(
        artifact_batch=_runtime_image_batch("DNA", "RNA"),
        context=_export_context(),
        sqlite_file="analysis.sqlite",
        wants_properties_file=False,
        write_image_thumbnails=True,
        thumbnail_image_names="DNA, RNA",
    )

    assert tuple(bundle) == ("analysis.sqlite",)
    connection = sqlite3.connect(":memory:")
    try:
        connection.deserialize(bundle["analysis.sqlite"])
        ((dna, rna),) = tuple(
            connection.execute(
                'SELECT "Image_Thumbnail_DNA", "Image_Thumbnail_RNA" FROM "Per_Image"'
            )
        )
        assert b64decode(dna).startswith(b"\x89PNG\r\n\x1a\n")
        assert b64decode(rna).startswith(b"\x89PNG\r\n\x1a\n")
    finally:
        connection.close()


def test_save_images_file_measurement_output_and_rows_are_conditional() -> None:
    common_settings = {
        "Select the image to save": "DNA",
        "Select method for constructing file names": "Single name",
        "Enter single file name": "SavedDNA",
        "Saved file format": "png",
        "Output file location": "Default Output Folder sub-folder|exports",
    }
    inactive = _contract(
        SaveImagesModule,
        _module(
            "SaveImages",
            **common_settings,
            **{"Record the file and path information to the saved image?": "No"},
        ),
        save_images.__name__,
        _context(
            step_name=str(SaveImagesModule.module_name),
            main_flow_image_names=("DNA",),
        ),
    )
    active = _contract(
        SaveImagesModule,
        _module(
            "SaveImages",
            **common_settings,
            **{"Record the file and path information to the saved image?": "Yes"},
        ),
        save_images_with_measurements.__name__,
        _context(
            step_name=str(SaveImagesModule.module_name),
            main_flow_image_names=("DNA",),
        ),
    )

    assert tuple(spec.artifact_type for spec in inactive.artifact_outputs) == (
        ImageArtifactType,
    )
    assert tuple(spec.artifact_type for spec in active.artifact_outputs) == (
        ImageArtifactType,
        MeasurementsArtifactType,
    )
    measurement = active.artifact_outputs[1]
    assert any(
        isinstance(relation, SaveImagesRecordedMeasurementSourceRelation)
        and relation.source.name == "DNA"
        for relation in measurement.relations
    )
    assert ArtifactSpecRelation(source=active.artifact_outputs[0].ref()) in (
        measurement.relations
    )

    image = np.ones((4, 5), dtype=np.float32)
    returned_main, saved = save_images(image, image_to_save=image)
    assert returned_main is image
    assert np.asarray(saved).shape == image.shape

    returned_main, saved, rows = save_images_with_measurements(
        image,
        image_to_save=image,
        saved_image_name="DNA",
        filename_method=SaveImagesFilenameMethod.SINGLE_NAME,
        single_file_name="SavedDNA",
        file_format="png",
        output_location="exports",
        slice_index=3,
    )
    assert returned_main is image
    assert np.asarray(saved).shape == image.shape
    row_mappings = tuple(rows.iter_row_mappings())
    assert tuple(
        row[MeasurementRowAxisField.FEATURE_NAME.value] for row in row_mappings
    ) == (
        "FileName_DNA",
        "PathName_DNA",
        "URL_DNA",
    )
    assert {row[MeasurementRowAxisField.SLICE_INDEX.value] for row in row_mappings} == {
        3
    }

    measurement_plan = ArtifactOutputPlan(
        name=measurement.name,
        path="/memory/save_images_measurements.pkl",
        artifact_type=MeasurementsArtifactType,
        relations=measurement.relations,
    )
    source = ImagePayloadMetadata(
        source_path="/input/DNA.png",
        source_image_names=("DNA",),
    ).payload_with(image, None)
    contextualized = FunctionOutputContextStrategy.for_output_plan(
        measurement_plan
    ).contextualize(source, rows, measurement_plan, None)

    assert isinstance(contextualized, MeasurementTable)
    assert contextualized.name == measurement.name
    assert contextualized.subject.scope is MeasurementScope.IMAGE
    assert contextualized.subject.name == "DNA"
    assert contextualized.source_image_name == "DNA"
    RuntimeValue.normalize(measurement_plan, contextualized, axis_id="A01")


def test_public_callables_reconstruct_active_export_contracts() -> None:
    assert "record_file_and_path" not in inspect.signature(save_images).parameters
    assert (
        "record_file_and_path"
        not in inspect.signature(save_images_with_measurements).parameters
    )

    export_primary_invocation = next(
        normalize_function_pattern(
            (
                export_to_database,
                {
                    "include_all_images": False,
                    "wants_properties_file": False,
                },
            )
        ).iter_items()
    )
    export_primary_contract, _ = _invocation_contract(
        ExportToDatabaseModule,
        export_primary_invocation,
        _context(
            step_name=str(ExportToDatabaseModule.module_name),
            runtime_image_names=("DNA", "RNA"),
        ),
    )
    export_invocation = next(
        normalize_function_pattern(
            (
                export_to_database,
                {
                    "include_all_images": False,
                    "write_image_thumbnails": True,
                    "thumbnail_image_names": "DNA, RNA",
                    "wants_properties_file": False,
                },
            )
        ).iter_items()
    )
    export_contract, export_consumed = _invocation_contract(
        ExportToDatabaseModule,
        export_invocation,
        _context(
            step_name=str(ExportToDatabaseModule.module_name),
            runtime_image_names=("DNA", "RNA"),
        ),
    )
    export_blocks, _ = ExportToDatabaseModule.module_blocks_for_invocation(
        invocation=export_invocation,
        step_context=_context(
            step_name=str(ExportToDatabaseModule.module_name),
            runtime_image_names=("DNA", "RNA"),
        ),
    )

    assert export_consumed == ()
    assert tuple(
        spec.artifact_type for spec in export_primary_contract.artifact_outputs
    ) == (SpecialArtifactType,)
    assert (
        optional_setting_value(
            export_blocks[0], ExportToDatabaseModule.write_thumbnails_setting
        )
        == "Yes"
    )
    assert tuple(spec.artifact_type for spec in export_contract.artifact_outputs) == (
        SpecialArtifactType,
    )

    save_primary_invocation = next(
        normalize_function_pattern(
            (
                save_images,
                {
                    "select_the_image_to_save": "DNA",
                    "filename_method": SaveImagesFilenameMethod.SINGLE_NAME,
                    "single_file_name": "SavedDNA",
                    "file_format": "png",
                },
            )
        ).iter_items()
    )
    save_primary_contract, save_primary_consumed = _invocation_contract(
        SaveImagesModule,
        save_primary_invocation,
        _context(
            step_name=str(SaveImagesModule.module_name),
            main_flow_image_names=("DNA",),
        ),
    )
    save_invocation = next(
        normalize_function_pattern(
            (
                save_images_with_measurements,
                {
                    "select_the_image_to_save": "DNA",
                    "saved_image_name": "DNA",
                    "filename_method": SaveImagesFilenameMethod.SINGLE_NAME,
                    "single_file_name": "SavedDNA",
                    "file_format": "png",
                },
            )
        ).iter_items()
    )
    save_contract, save_consumed = _invocation_contract(
        SaveImagesModule,
        save_invocation,
        _context(
            step_name=str(SaveImagesModule.module_name),
            main_flow_image_names=("DNA",),
        ),
    )
    save_blocks, _ = SaveImagesModule.module_blocks_for_invocation(
        invocation=save_invocation,
        step_context=_context(
            step_name=str(SaveImagesModule.module_name),
            main_flow_image_names=("DNA",),
        ),
    )

    assert save_consumed == ("select_the_image_to_save",)
    assert save_primary_consumed == ("select_the_image_to_save",)
    assert tuple(
        spec.artifact_type for spec in save_primary_contract.artifact_outputs
    ) == (ImageArtifactType,)
    assert tuple(
        spec.sidecar_role for spec in save_primary_contract.artifact_outputs
    ) == (ArtifactSidecarRole.MATERIALIZED_IMAGE_COPY,)
    assert save_primary_contract.canonical_return_output_specs.names() == ()
    assert save_primary_contract.trailing_return_output_specs.names() == (
        save_primary_contract.artifact_outputs[0].name,
    )
    assert (
        optional_setting_value(save_blocks[0], SaveImagesModule.record_file_setting)
        == "Yes"
    )
    assert tuple(spec.artifact_type for spec in save_contract.artifact_outputs) == (
        ImageArtifactType,
        MeasurementsArtifactType,
    )
    assert tuple(spec.sidecar_role for spec in save_contract.artifact_outputs) == (
        ArtifactSidecarRole.MATERIALIZED_IMAGE_COPY,
        None,
    )
    assert save_contract.canonical_return_output_specs.names() == ()
    assert save_contract.trailing_return_output_specs.names() == (
        save_contract.artifact_outputs[0].name,
        save_contract.artifact_outputs[1].name,
    )
