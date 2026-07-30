from __future__ import annotations

from dataclasses import replace
import inspect
from typing import get_args, get_type_hints

import numpy as np
import pytest

from openhcs.core.aligned_image_payload import AlignedImageStack
from openhcs.core.callable_contract import CallableContract
from openhcs.core.artifacts import (
    ArtifactInputPlan,
    ArtifactOutputPlan,
    ArtifactSidecarRole,
    ArtifactSidecarSourceRelation,
    ArtifactSpec,
    ArtifactSpecCollection,
    GroupLineageSourceRelation,
    ImageArtifactType,
    InputGroupLineageSourceRelation,
    InputStackBroadcastSourceRelation,
    SourceStackLineageSourceRelation,
    MeasurementsArtifactType,
    ObjectLabelsArtifactType,
    ObjectLineageArtifactType,
    RelationshipsArtifactType,
    SpatialGridArtifactType,
)
from openhcs.core.function_patterns import (
    FunctionInvocationKey,
    normalize_function_pattern,
)
from openhcs.core.config import LazyDtypeConfig
from openhcs.core.invocation_artifacts import ArtifactDeclarationStepContext
from openhcs.core.runtime_relationships import DirectedObjectRelationshipPayload
from openhcs.core.pipeline.function_contracts import (
    annotation_produces_runtime_type,
    runtime_bound_parameter_names_from_callable,
)
from openhcs.core.pipeline.artifact_planning import artifact_producers_for_outputs
from openhcs.core.runtime_tabular_values import ColumnarRows
from openhcs.interop.cellprofiler.module_declarations import (
    CellProfilerModule,
)
from openhcs.interop.cellprofiler.module_artifact_declarations import (
    MeasurementArtifactOutputModule,
)
from openhcs.interop.cellprofiler.parser import ModuleBlock, ModuleSetting
from openhcs.processing.backends.cellprofiler.colocalization import (
    measure_colocalization_objects,
)
from openhcs.processing.backends.cellprofiler.crop import (
    CropModule,
    crop,
)
from openhcs.processing.backends.cellprofiler.flagging import (
    FlagResult,
    flag_image_intensity,
    flag_image_result,
)
from openhcs.processing.backends.cellprofiler.grid import (
    IdentifyObjectsInGridModule,
    identify_objects_in_grid,
)
from openhcs.processing.backends.cellprofiler.intensity import (
    MeasureImageIntensityModule,
    MeasureObjectIntensityModule,
    measure_image_intensity,
    measure_image_intensity_objects,
    measure_object_intensity,
)
from openhcs.processing.backends.cellprofiler.morphology import (
    CombineobjectsModule,
    combineobjects,
)
from openhcs.processing.backends.cellprofiler.neighbors import (
    MeasureObjectNeighborsModule,
    NeighborMeasurements,
    NeighborRetainedImageRequest,
    measure_object_neighbors,
)
from openhcs.processing.backends.cellprofiler.primary_objects import (
    IdentifyPrimaryObjectsModule,
    identify_primary_objects,
)
from openhcs.processing.backends.cellprofiler.secondary import (
    IdentifySecondaryObjectsModule,
    IdentifyTertiaryObjectsModule,
    identify_secondary_objects,
    identify_tertiary_objects,
)
from openhcs.processing.backends.cellprofiler.shape import (
    MeasureObjectSizeShapeModule,
)
from openhcs.processing.backends.cellprofiler.tracking import (
    track_objects,
)
from openhcs.processing.backends.cellprofiler.watershed import (
    WatershedModule,
    watershed_cellprofiler4,
)
from python_introspect import parameter_exclusions


def test_flag_image_backend_owns_threshold_evaluation() -> None:
    result = flag_image_result(
        flag_name="Saturation",
        flag_category="QC",
        measurement_name="intensity_mean",
        measurement_value=0.75,
        check_minimum=True,
        minimum_value=0.1,
        check_maximum=True,
        maximum_value=0.5,
    )

    assert result == FlagResult(
        slice_index=0,
        flag_name="QC_Saturation",
        flag_value=1,
        measurement_name="intensity_mean",
        measurement_value=0.75,
        min_threshold=0.1,
        max_threshold=0.5,
        pass_fail="Fail",
    )


def test_flag_image_backend_callable_preserves_image_and_result() -> None:
    image = np.array([[0.2, 0.8]], dtype=np.float32)

    output_image, results = flag_image_intensity(
        image,
        use_mean=False,
        maximum_value=0.4,
    )

    np.testing.assert_array_equal(output_image, image)
    assert results.column_values("measurement_name") == ("intensity_median",)
    assert results.column_values("measurement_value") == pytest.approx((0.5,))
    assert results.column_values("pass_fail") == ("Fail",)


def _contract(
    module_type,
    settings: tuple[tuple[str, str], ...],
    *,
    available: tuple[ArtifactSpec, ...],
    main_flow: tuple[ArtifactSpec, ...] = (),
):
    module = ModuleBlock(
        name=str(module_type.module_name),
        module_num=1,
        setting_records=[ModuleSetting(name, value) for name, value in settings],
    )
    invocation_key = FunctionInvocationKey(
        str(module_type.function_name),
        "default",
        0,
    )
    return module_type.callable_contract(
        module=module,
        invocation_key=invocation_key,
        step_context=ArtifactDeclarationStepContext(
            step_index=0,
            available_artifacts=ArtifactSpecCollection(available),
            main_flow_artifacts=ArtifactSpecCollection(
                spec.for_plan_type(ArtifactInputPlan) for spec in main_flow
            ),
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


def _abi_contract(
    *,
    inputs: tuple[ArtifactSpec, ...] = (),
    outputs: tuple[ArtifactSpec, ...] = (),
) -> CallableContract:
    context_source = ArtifactSpec.input("SyntheticSource", ImageArtifactType)
    contextual_outputs = tuple(
        (
            output.with_group_scope_relation(
                GroupLineageSourceRelation(source=context_source.ref())
            )
            if output.artifact_type.carries_source_image_context
            else output
        )
        for output in outputs
    )
    raw_contract = CallableContract.from_callable(lambda image: image)
    return replace(
        raw_contract,
        module_name="Synthetic",
        metadata=replace(
            raw_contract.metadata,
            artifact_inputs=(context_source, *inputs),
            artifact_outputs=contextual_outputs,
        ),
    )


def test_crop_previous_mask_declares_primary_image_stack_broadcast() -> None:
    primary = ArtifactSpec.output("OrigGreen", ImageArtifactType)
    previous_crop = ArtifactSpec.output("CropBlue", ImageArtifactType)
    previous_mask = ArtifactSpec.output(
        "opaque-mask-identity",
        ImageArtifactType,
        sidecar_role=ArtifactSidecarRole.CROP_MASK,
        relations=(ArtifactSidecarSourceRelation(source=previous_crop.ref()),),
    )
    contract = _contract(
        CropModule,
        (
            ("Select the input image", "OrigGreen"),
            ("Name the output image", "CropGreen"),
            ("Select the cropping shape", "Previous cropping"),
            ("Select the image with a cropping mask", "CropBlue"),
        ),
        available=(primary, previous_crop, previous_mask),
        main_flow=(primary,),
    )

    mask_input = contract.artifact_inputs.require_by_name_and_artifact_type(
        "opaque-mask-identity",
        ImageArtifactType,
    )

    assert (
        mask_input.parameter_name
        == CropModule.previous_image_binding.require_runtime_parameter_name()
    )
    assert mask_input.relations == (
        InputStackBroadcastSourceRelation(
            source=ArtifactSpec.input("OrigGreen", ImageArtifactType).ref()
        ),
    )
    source_ref = mask_input.relations[0].source
    assert mask_input.group_scope_sources() == ()
    assert mask_input.source_context_sources() == (source_ref,)
    assert mask_input.stack_broadcast_sources() == (source_ref,)


def test_crop_special_mask_does_not_consume_redirected_main_flow() -> None:
    primary = ArtifactSpec.output("Worms", ImageArtifactType)
    mask = ArtifactSpec.output("ErodedWellEdge", ImageArtifactType)
    contract = _contract(
        CropModule,
        (
            ("Select the input image", "Worms"),
            ("Name the output image", "WormsCropped"),
            ("Select the cropping shape", "Image"),
            ("Select the masking image", "ErodedWellEdge"),
        ),
        available=(primary, mask),
        main_flow=(mask,),
    )

    assert contract.artifact_inputs.names() == (
        "Worms",
        "ErodedWellEdge",
    )


def test_crop_outputs_preserve_primary_image_stack_scope() -> None:
    primary = ArtifactSpec.output("OrigGreen", ImageArtifactType)
    contract = _contract(
        CropModule,
        (
            ("Select the input image", "OrigGreen"),
            ("Name the output image", "CropGreen"),
            ("Select the cropping shape", "Rectangle"),
        ),
        available=(primary,),
        main_flow=(primary,),
    )

    expected_relation = SourceStackLineageSourceRelation(
        source=ArtifactSpec.input("OrigGreen", ImageArtifactType).ref()
    )
    image_outputs = contract.artifact_outputs.for_artifact_type(ImageArtifactType)

    primary_output, sidecar_output = image_outputs.specs
    assert primary_output.relations == (expected_relation,)
    assert sidecar_output.relations == (
        expected_relation,
        ArtifactSidecarSourceRelation(source=primary_output.ref()),
    )
    assert tuple(output.group_scope_sources() for output in image_outputs) == (
        (expected_relation.source,),
        (expected_relation.source,),
    )
    assert tuple(output.source_context_sources() for output in image_outputs) == (
        (expected_relation.source,),
        (expected_relation.source,),
    )


def test_object_outputs_preserve_declared_input_stack_scope() -> None:
    primary = ArtifactSpec.output("DNA", ImageArtifactType)
    contract = _contract(
        IdentifyPrimaryObjectsModule,
        (
            ("Select the input image", "DNA"),
            ("Name the primary objects to be identified", "Nuclei"),
        ),
        available=(primary,),
        main_flow=(primary,),
    )

    object_output = contract.artifact_outputs.require_by_name_and_artifact_type(
        "Nuclei",
        ObjectLabelsArtifactType,
    )

    assert object_output.relations == (
        SourceStackLineageSourceRelation(
            source=ArtifactSpec.input("DNA", ImageArtifactType).ref()
        ),
    )


def test_secondary_objects_project_primary_labels_to_image_invocation() -> None:
    image = ArtifactSpec.output("CropGreen", ImageArtifactType)
    labels = ArtifactSpec.output("Nuclei", ObjectLabelsArtifactType)
    contract = _contract(
        IdentifySecondaryObjectsModule,
        (
            ("Select the input image", "CropGreen"),
            ("Select the input objects", "Nuclei"),
            ("Name the objects to be identified", "Cells"),
        ),
        available=(image, labels),
    )

    object_input = contract.artifact_inputs.require_by_name_and_artifact_type(
        "Nuclei",
        ObjectLabelsArtifactType,
    )

    assert object_input.relations == (
        InputGroupLineageSourceRelation(
            source=ArtifactSpec.input("CropGreen", ImageArtifactType).ref()
        ),
    )


def test_tertiary_objects_project_smaller_labels_to_larger_invocation() -> None:
    larger = ArtifactSpec.output("Cells", ObjectLabelsArtifactType)
    smaller = ArtifactSpec.output("Nuclei", ObjectLabelsArtifactType)
    contract = _contract(
        IdentifyTertiaryObjectsModule,
        (
            ("Select the larger identified objects", "Cells"),
            ("Select the smaller identified objects", "Nuclei"),
            ("Name the tertiary objects to be identified", "Cytoplasm"),
        ),
        available=(larger, smaller),
    )

    smaller_input = contract.artifact_inputs.require_by_name_and_artifact_type(
        "Nuclei",
        ObjectLabelsArtifactType,
    )

    assert smaller_input.relations == (
        InputGroupLineageSourceRelation(
            source=ArtifactSpec.input("Cells", ObjectLabelsArtifactType).ref()
        ),
    )


def test_object_measurement_inputs_do_not_invent_self_lineage() -> None:
    objects = tuple(
        ArtifactSpec.output(name, ObjectLabelsArtifactType)
        for name in ("Cells", "Nuclei", "Cytoplasm")
    )
    contract = _contract(
        MeasureObjectSizeShapeModule,
        tuple(("Select object sets to measure", spec.name) for spec in objects),
        available=objects,
    )

    object_inputs = contract.artifact_inputs.for_artifact_type(
        ObjectLabelsArtifactType
    )

    assert tuple(spec.name for spec in object_inputs.specs) == (
        "Cells",
        "Nuclei",
        "Cytoplasm",
    )
    assert tuple(spec.relations for spec in object_inputs.specs) == ((), (), ())


def test_measurement_inputs_do_not_invent_image_or_object_self_lineage() -> None:
    image = ArtifactSpec.output("CropBlue", ImageArtifactType)
    labels = ArtifactSpec.output("Nuclei", ObjectLabelsArtifactType)
    contract = _contract(
        MeasureObjectIntensityModule,
        (
            ("Select images to measure", "CropBlue"),
            ("Select objects to measure", "Nuclei"),
        ),
        available=(image, labels),
    )

    inputs = contract.artifact_inputs

    assert tuple(spec.name for spec in inputs.specs) == ("CropBlue", "Nuclei")
    assert tuple(spec.relations for spec in inputs.specs) == ((), ())


def test_optional_object_special_input_does_not_consume_primary_image() -> None:
    image = ArtifactSpec.input("CropBlue", ImageArtifactType)
    objects = ArtifactSpec.input("Nuclei", ObjectLabelsArtifactType)

    assert MeasureImageIntensityModule.primary_image_inputs(
        measure_image_intensity,
        (image,),
    ) == (image,)
    assert MeasureImageIntensityModule.primary_image_inputs(
        measure_image_intensity_objects,
        (image, objects),
    ) == (image,)


def test_callable_abi_exposes_no_fixed_object_input_candidate_for_wrong_cardinality() -> None:
    invocation = next(normalize_function_pattern(combineobjects).iter_items())
    only_one = ArtifactSpec.output("OnlyOne", ObjectLabelsArtifactType)
    blocks, consumed = CombineobjectsModule.module_blocks_for_invocation(
        invocation=invocation,
        step_context=ArtifactDeclarationStepContext(
            step_index=0,
            available_artifacts=ArtifactSpecCollection((only_one,)),
            available_artifact_producers=artifact_producers_for_outputs(
                (only_one,),
                groups=(None,),
                invocation_keys=(
                    FunctionInvocationKey("identify_primary_objects", "default", 0),
                ),
            ),
        ),
    )

    assert blocks == ()
    assert consumed == ()


def test_callable_abi_rejects_special_output_return_count_drift() -> None:
    def invalid(image: np.ndarray) -> np.ndarray:
        return image

    with pytest.raises(ValueError, match="does not declare the canonical return"):
        CellProfilerModule.validate_callable_artifact_abi(
            invalid,
            _abi_contract(
                outputs=(
                    ArtifactSpec.output("Image", ImageArtifactType),
                    ArtifactSpec.output("Measurements", MeasurementsArtifactType),
                ),
            ),
        )


def test_callable_abi_rejects_raw_object_label_return_before_runtime() -> None:
    runtime_calls: list[str] = []

    def invalid(image: np.ndarray) -> np.ndarray:
        runtime_calls.append("invalid")
        return image

    with pytest.raises(TypeError, match="must explicitly return ObjectLabelValue"):
        CellProfilerModule.validate_callable_artifact_abi(
            invalid,
            _abi_contract(
                outputs=(
                    ArtifactSpec.output("Objects", ObjectLabelsArtifactType),
                ),
            ),
        )

    assert runtime_calls == []


def test_callable_abi_rejects_raw_measurement_rows_before_runtime() -> None:
    runtime_calls: list[str] = []

    def invalid(image: np.ndarray) -> tuple[np.ndarray, list[dict[str, object]]]:
        runtime_calls.append("invalid")
        return image, []

    with pytest.raises(TypeError, match="must explicitly return ColumnarRows"):
        CellProfilerModule.validate_callable_artifact_abi(
            invalid,
            _abi_contract(
                outputs=(
                    ArtifactSpec.output("Measurements", MeasurementsArtifactType),
                ),
            ),
        )

    assert runtime_calls == []


def test_registered_measurement_modules_publish_typed_columnar_rows() -> None:
    def contains_columnar_rows(annotation: object) -> bool:
        return annotation_produces_runtime_type(
            annotation,
            ColumnarRows,
        ) or any(
            member is not Ellipsis and contains_columnar_rows(member)
            for member in get_args(annotation)
        )

    violations = []
    for module_type in set(CellProfilerModule.__registry__.values()):
        if not issubclass(module_type, MeasurementArtifactOutputModule):
            continue
        typed_functions = []
        for function_name in (
            module_type.function_name,
            *module_type.function_variants,
        ):
            if function_name is None:
                continue
            func = module_type.require_callable(str(function_name))
            return_annotation = get_type_hints(func)["return"]
            if contains_columnar_rows(return_annotation):
                typed_functions.append(function_name)
        if not typed_functions:
            violations.append(module_type.module_name)

    assert violations == []


def test_fixed_return_slots_follow_nominal_contract_output_order() -> None:
    dna = ArtifactSpec.output("DNA", ImageArtifactType)
    nuclei = ArtifactSpec.output("Nuclei", ObjectLabelsArtifactType)
    cells = ArtifactSpec.output("Cells", ObjectLabelsArtifactType)
    grid = ArtifactSpec.output("Grid", SpatialGridArtifactType)

    contracts_and_functions = (
        (
            _contract(
                CropModule,
                (
                    ("Select the input image", "DNA"),
                    ("Name the output image", "CropDNA"),
                    ("Select the cropping shape", "Rectangle"),
                ),
                available=(dna,),
                main_flow=(dna,),
            ),
            crop,
            (ImageArtifactType, MeasurementsArtifactType),
        ),
        (
            _contract(
                MeasureImageIntensityModule,
                (("Select images to measure", "DNA"),),
                available=(dna,),
                main_flow=(dna,),
            ),
            measure_image_intensity,
            (MeasurementsArtifactType,),
        ),
        (
            _contract(
                MeasureObjectIntensityModule,
                (
                    ("Select images to measure", "DNA"),
                    ("Select objects to measure", "Nuclei"),
                ),
                available=(dna, nuclei),
                main_flow=(dna,),
            ),
            measure_object_intensity,
            (MeasurementsArtifactType,),
        ),
        (
            _contract(
                IdentifyPrimaryObjectsModule,
                (
                    ("Select the input image", "DNA"),
                    ("Name the primary objects to be identified", "Nuclei"),
                ),
                available=(dna,),
                main_flow=(dna,),
            ),
            identify_primary_objects,
            (MeasurementsArtifactType, ObjectLabelsArtifactType),
        ),
        (
            _contract(
                WatershedModule,
                (
                    ("Select the input image", "DNA"),
                    ("Name the output object", "WatershedObjects"),
                ),
                available=(dna,),
                main_flow=(dna,),
            ),
            watershed_cellprofiler4,
            (MeasurementsArtifactType, ObjectLabelsArtifactType),
        ),
        (
            _contract(
                IdentifySecondaryObjectsModule,
                (
                    ("Select the input image", "DNA"),
                    ("Select the input objects", "Nuclei"),
                    ("Name the objects to be identified", "Cells"),
                ),
                available=(dna, nuclei),
                main_flow=(dna,),
            ),
            identify_secondary_objects,
            (
                MeasurementsArtifactType,
                ObjectLineageArtifactType,
                ObjectLabelsArtifactType,
            ),
        ),
        (
            _contract(
                IdentifyTertiaryObjectsModule,
                (
                    ("Select the larger identified objects", "Cells"),
                    ("Select the smaller identified objects", "Nuclei"),
                    ("Name the tertiary objects to be identified", "Cytoplasm"),
                ),
                available=(cells, nuclei),
            ),
            identify_tertiary_objects,
            (
                ObjectLineageArtifactType,
                ObjectLineageArtifactType,
                MeasurementsArtifactType,
                ObjectLabelsArtifactType,
            ),
        ),
        (
            _contract(
                IdentifyObjectsInGridModule,
                (
                    ("Select the defined grid", "Grid"),
                    (
                        "Select object shapes and locations",
                        "Natural Shape and Location",
                    ),
                    ("Select the guiding objects", "Nuclei"),
                    ("Name the objects to be identified", "GridObjects"),
                ),
                available=(grid, nuclei),
            ),
            identify_objects_in_grid,
            (MeasurementsArtifactType, ObjectLabelsArtifactType),
        ),
        (
            _contract(
                MeasureObjectNeighborsModule,
                (
                    ("Select objects to measure", "Nuclei"),
                    ("Select neighboring objects to measure", "Cells"),
                    (
                        "Retain the image of objects colored by numbers of neighbors?",
                        "Yes",
                    ),
                    (
                        "Retain the image of objects colored by percent of touching pixels?",
                        "Yes",
                    ),
                    ("Name the output image", "NeighborCount"),
                    ("Name the output image", "PercentTouching"),
                ),
                available=(nuclei, cells),
            ),
            measure_object_neighbors,
            (
                RelationshipsArtifactType,
                MeasurementsArtifactType,
            ),
        ),
    )

    for contract, func, artifact_types in contracts_and_functions:
        assert tuple(
            spec.artifact_type for spec in contract.trailing_return_output_specs
        ) == artifact_types, func.__name__
        assert len(get_args(get_type_hints(func)["return"])) == (
            len(contract.trailing_return_output_specs) + 1
        )


def test_contract_order_owns_guided_grid_and_tertiary_input_mapping() -> None:
    grid = ArtifactSpec.output("Grid", SpatialGridArtifactType)
    nuclei = ArtifactSpec.output("Nuclei", ObjectLabelsArtifactType)
    cells = ArtifactSpec.output("Cells", ObjectLabelsArtifactType)

    grid_contract = _contract(
        IdentifyObjectsInGridModule,
        (
            ("Select the defined grid", "Grid"),
            (
                "Select object shapes and locations",
                "Natural Shape and Location",
            ),
            ("Select the guiding objects", "Nuclei"),
            ("Name the objects to be identified", "GridObjects"),
        ),
        available=(grid, nuclei),
    )
    assert tuple(spec.name for spec in grid_contract.artifact_inputs) == (
        "Grid",
        "Nuclei",
    )

    tertiary_contract = _contract(
        IdentifyTertiaryObjectsModule,
        (
            ("Select the larger identified objects", "Cells"),
            ("Select the smaller identified objects", "Nuclei"),
            ("Name the tertiary objects to be identified", "Cytoplasm"),
        ),
        available=(cells, nuclei),
    )
    assert tuple(spec.name for spec in tertiary_contract.artifact_inputs) == (
        "Cells",
        "Nuclei",
    )


def test_neighbors_pack_multiple_main_images_before_fixed_measurement_slot() -> None:
    labels = np.array([[0, 1], [2, 0]], dtype=np.int32)
    request = NeighborRetainedImageRequest(
        labels=labels,
        retain_neighbor_count_image=True,
        neighbor_count_colormap="viridis",
        retain_percent_touching_image=True,
        percent_touching_colormap="magma",
    )

    relationship = DirectedObjectRelationshipPayload(
        source_ids=(1,),
        target_ids=(2,),
    )
    measurement = NeighborMeasurements(
        slice_index=0,
        object_id=1,
        scale=1,
        number_of_neighbors=1,
        percent_touching=50.0,
        first_closest_object_number=2,
        first_closest_distance=1.0,
        second_closest_object_number=0,
        second_closest_distance=0.0,
        angle_between_neighbors=0.0,
    )
    main_output, actual_relationship, measurements = request.output(
        np.zeros_like(labels, dtype=np.float32),
        relationship,
        [measurement],
        neighbor_count_image=labels.astype(np.float32),
        percent_touching_image=labels.astype(np.float32),
    )

    assert isinstance(main_output, AlignedImageStack)
    assert tuple(output.shape for output in main_output.slices) == (
        (2, 2, 3),
        (2, 2, 3),
    )
    assert actual_relationship is relationship
    assert measurements.rows == (measurement,)


def test_track_objects_state_is_local() -> None:
    parameters = inspect.signature(track_objects).parameters

    assert "_tracking_state" not in parameters
    assert "_tracking_state" not in parameter_exclusions(track_objects)


def test_object_colocalization_rank_provider_is_runtime_bound() -> None:
    parameters = inspect.signature(measure_colocalization_objects).parameters
    parameter = parameters["rank_provider"]

    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert runtime_bound_parameter_names_from_callable(
        measure_colocalization_objects
    ) == ("rank_provider", "threshold_mask_outputs")
    assert CallableContract.from_callable(
        measure_colocalization_objects
    ).config_bound_parameter_names == ("dtype_config",)
    assert parameters["dtype_config"].annotation is LazyDtypeConfig
    assert "rank_provider" in parameter_exclusions(measure_colocalization_objects)
