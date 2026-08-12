"""Nominal compile and runtime contracts for MeasureObjectNeighbors."""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import get_type_hints

import numpy as np

from openhcs.core.artifacts import (
    ArtifactOutputPlan,
    ArtifactSpec,
    ArtifactSpecCollection,
    ObjectLabelsArtifactType,
    RelationshipsArtifactType,
)
from openhcs.core.compiled_step_plan import CompiledStepPlan
from openhcs.core.config import GlobalPipelineConfig, PipelineConfig
from openhcs.core.context.processing_context import ProcessingContext
from openhcs.core.function_patterns import (
    DEFAULT_GROUP_KEY,
    FunctionInvocationKey,
    compile_function_pattern,
    normalize_function_pattern,
)
from openhcs.core.function_step_transport import FunctionStepTransportAuthority
from openhcs.core.invocation_artifacts import ArtifactDeclarationStepContext
from openhcs.core.pipeline.compilation_session import CompilationSession
from openhcs.core.pipeline.artifact_planning import artifact_producers_for_outputs
from openhcs.core.pipeline.function_contracts import artifact_outputs
from openhcs.core.memory.decorators import numpy
from openhcs.core.pipeline.step_snapshot import StepSnapshot
from openhcs.core.runtime_object_labels import (
    ObjectLabelSet,
    ObjectLabelPayload,
    ObjectLabelVariantData,
)
from openhcs.core.runtime_relationships import ObjectRelationshipDeclaration
from openhcs.core.runtime_image_values import image_payload_data
from openhcs.core.source_bindings import (
    NamedSourceBinding,
    SourceProjectionRole,
    StepSourceBindingsConfig,
)
from openhcs.core.steps.function_step import FunctionStep
from openhcs.interop.cellprofiler.compile_time_contracts import (
    CellProfilerInvocationContractProviderFactory,
)
from openhcs.interop.cellprofiler.parser import ModuleBlock, ModuleSetting
from openhcs.interop.cellprofiler.runtime.invocation import CellProfilerImageRequest
from openhcs.interop.cellprofiler.settings_binder import SettingsBinder
from openhcs.processing.backends.cellprofiler.neighbors import (
    DistanceMethod,
    MeasureObjectNeighborsModule,
    measure_object_neighbors,
)
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract


def _neighbor_step() -> FunctionStep:
    source_bindings = StepSourceBindingsConfig(
        enabled=True,
        bindings=(
            NamedSourceBinding(
                alias="Cells",
                artifact_kind=ObjectLabelsArtifactType,
                projection_role=SourceProjectionRole.SOURCE_ARTIFACT,
            ),
        ),
    )
    return FunctionStep(
        func=(
            measure_object_neighbors,
            {
                (
                    MeasureObjectNeighborsModule.measured_objects_binding.require_parameter_name()
                ): "Cells",
                "distance_method": DistanceMethod.EXPAND,
                "neighbor_distance": 5,
            },
        ),
        name="MeasureObjectNeighbors",
        source_bindings=source_bindings,
    )


def _reconstructed_neighbor_step() -> FunctionStep:
    source = FunctionStepTransportAuthority.source_from_pipeline([_neighbor_step()])
    namespace: dict[str, object] = {}
    exec(compile(source, "<neighbors-pipeline>", "exec"), namespace)
    return FunctionStepTransportAuthority.pipeline_steps_from_namespace(namespace)[0]


def test_relationship_module_number_is_derived_after_public_transport() -> None:
    step = _neighbor_step()
    source = FunctionStepTransportAuthority.source_from_pipeline([step])
    assert "invocation_source_positions" not in source
    assert "FunctionInvocationSourcePosition" not in source
    namespace: dict[str, object] = {}
    exec(compile(source, "<neighbors-module-order>", "exec"), namespace)
    (restored_step,) = FunctionStepTransportAuthority.pipeline_steps_from_namespace(
        namespace
    )
    snapshot = StepSnapshot(
        index=0,
        scope_id="test::neighbors-module-order",
        step=restored_step,
    )
    session = CompilationSession.from_context(
        context=ProcessingContext(
            step_plans={
                0: CompiledStepPlan(
                    step_index=0,
                    step_name=restored_step.name,
                    step_type=type(restored_step).__name__,
                    axis_id="A01",
                )
            },
            axis_id="A01",
        ),
        steps=[restored_step],
        orchestrator=SimpleNamespace(pipeline_config=PipelineConfig()),
        global_config=GlobalPipelineConfig(),
        step_state_map={0: object()},
        snapshots=(snapshot,),
    )

    provider = CellProfilerInvocationContractProviderFactory.provider_for_session(
        session
    )

    assert provider is not None
    (plan,) = tuple(provider.plans.values())
    contract = plan.contract
    ((relationship, declaration),) = contract.artifact_outputs.relation_refs(
        ObjectRelationshipDeclaration
    )
    assert relationship.artifact_type is RelationshipsArtifactType
    assert declaration.producer_module_number == 1


def _compiled_neighbor_invocation():
    step = _reconstructed_neighbor_step()
    snapshot = StepSnapshot(index=0, scope_id="test::neighbors", step=step)
    context = ProcessingContext(
        step_plans={
            0: CompiledStepPlan(
                step_index=0,
                step_name=step.name,
                step_type=type(step).__name__,
                axis_id="A01",
            )
        },
        axis_id="A01",
    )
    session = CompilationSession.from_context(
        context=context,
        steps=[step],
        orchestrator=SimpleNamespace(pipeline_config=PipelineConfig()),
        global_config=GlobalPipelineConfig(),
        step_state_map={0: object()},
        snapshots=(snapshot,),
    )
    provider = CellProfilerInvocationContractProviderFactory.provider_for_session(
        session
    )
    assert provider is not None
    step_context = ArtifactDeclarationStepContext(
        step_index=0,
        source_bindings=step.source_bindings,
    )
    plan = provider(
        next(normalize_function_pattern(step.func).iter_items()), step_context
    )
    assert plan is not None
    output_plans = {
        spec.ref(): ArtifactOutputPlan(
            spec.name,
            f"/tmp/{spec.name}",
            artifact_type=spec.artifact_type,
            sidecar_role=spec.sidecar_role,
        )
        for spec in plan.contract.artifact_outputs
    }
    compiled = compile_function_pattern(
        step.func,
        {},
        output_plans,
        invocation_contract_provider=provider,
        step_context=step_context,
    )
    return next(compiled.iter_invocations())


def test_module_binding_preserves_exact_typed_distance_contract() -> None:
    parameters = inspect.signature(measure_object_neighbors).parameters
    annotations = get_type_hints(measure_object_neighbors)
    assert annotations["distance_method"] is DistanceMethod
    assert parameters["distance_method"].default is inspect.Parameter.empty
    assert parameters["neighbor_distance"].default is inspect.Parameter.empty

    records = [
        ModuleSetting("Method to determine neighbors", "Expand until adjacent"),
        ModuleSetting("Neighbor distance", "5"),
    ]
    module = ModuleBlock(
        name="MeasureObjectNeighbors",
        module_num=1,
        setting_records=records,
    )

    bound = MeasureObjectNeighborsModule.bind_settings(module, binder=SettingsBinder())
    assert bound.kwargs["distance_method"] is DistanceMethod.EXPAND
    assert bound.kwargs["neighbor_distance"] == 5


def test_neighbor_invocation_image_uses_declared_object_label_domain() -> None:
    labels = ObjectLabelSet(
        name="Cells",
        variant_data=ObjectLabelVariantData(
            labels=np.asarray([[0, 1, 1], [0, 0, 2]], dtype=np.int32)
        ),
    )
    ambient_rgb = np.zeros((2, 3, 3), dtype=np.float32)

    projected = MeasureObjectNeighborsModule.project_invocation_image_request(
        image_request=CellProfilerImageRequest(
            source_image_name="InvertedRedOutlines",
            source_aliases=("InvertedRedOutlines",),
            image_count=1,
            payload=ambient_rgb,
        ),
        runtime_kwargs={"labels": labels, "neighbor_labels": labels},
    )

    assert np.shape(image_payload_data(projected.payload)) == (2, 3)
    assert projected.source_image_name is None
    assert projected.source_aliases == ()
    assert projected.plane_projection == labels.declared_plane_projection()


def test_function_step_only_reconstruction_compiles_neighbor_distance_contract() -> (
    None
):
    invocation = _compiled_neighbor_invocation()
    contract = invocation.contract

    assert invocation.kwargs_dict == {
        "distance_method": DistanceMethod.EXPAND,
        "neighbor_distance": 5,
    }
    assert (
        MeasureObjectNeighborsModule.for_callable_contract(contract)
        is MeasureObjectNeighborsModule
    )
    assert [spec.name for spec in contract.artifact_inputs] == ["Cells", "Cells"]


def test_neighbor_relationship_uses_numbered_module_block_identity() -> None:
    cells = ArtifactSpec.input("Cells", ObjectLabelsArtifactType)
    module = ModuleBlock(
        name="MeasureObjectNeighbors",
        module_num=37,
        setting_records=[
            ModuleSetting("Select objects to measure", "Cells"),
            ModuleSetting("Select neighboring objects to measure", "Cells"),
            ModuleSetting(
                "Retain the image of objects colored by numbers of neighbors?", "No"
            ),
            ModuleSetting(
                "Retain the image of objects colored by percent of touching pixels?",
                "No",
            ),
        ],
    )

    contract = MeasureObjectNeighborsModule.callable_contract(
        module=module,
        invocation_key=FunctionInvocationKey(
            "measure_object_neighbors",
            DEFAULT_GROUP_KEY,
            0,
        ),
        step_context=ArtifactDeclarationStepContext(
            step_index=0,
            available_artifacts=ArtifactSpecCollection((cells,)),
            available_artifact_producers=artifact_producers_for_outputs(
                (cells,),
                groups=(None,),
                invocation_keys=(
                    FunctionInvocationKey(
                        "fixture_producer",
                        DEFAULT_GROUP_KEY,
                        0,
                    ),
                ),
            ),
        ),
    )

    ((relationship, declaration),) = contract.artifact_outputs.relation_refs(
        ObjectRelationshipDeclaration
    )
    assert relationship.artifact_type is RelationshipsArtifactType
    assert declaration.producer_module_number == 37


def test_compiler_numbers_neighbor_invocation_equivalence_only_within_each_step(
    monkeypatch,
) -> None:
    from openhcs.core.runtime_adapters import RuntimeAdapterSpec
    from openhcs.interop.cellprofiler.runtime.adapter import (
        CellProfilerRuntimeAdapter,
    )

    monkeypatch.setattr(
        CellProfilerRuntimeAdapter,
        "runtime_adapter_spec",
        classmethod(
            lambda cls: RuntimeAdapterSpec(
                parameter_name=cls.require_parameter_name(),
                factory=lambda _request: None,
                manages_artifact_inputs=True,
            )
        ),
    )
    common_kwargs = {
        "distance_method": DistanceMethod.EXPAND,
        "neighbor_distance": 5,
    }
    cells = ArtifactSpec.output("Cells", ObjectLabelsArtifactType)

    @artifact_outputs(cells)
    @numpy(contract=ProcessingContract.PURE_3D)
    def fixture_producer(image):
        return image

    steps = (
        FunctionStep(
            func={
                group_key: fixture_producer
                for group_key in ("first", "equivalent", "different", "default")
            },
            name="FixtureProducer",
        ),
        FunctionStep(
            func={
                "first": (measure_object_neighbors, common_kwargs),
                "equivalent": (measure_object_neighbors, common_kwargs),
                "different": (
                    measure_object_neighbors,
                    {
                        "distance_method": DistanceMethod.EXPAND,
                        "neighbor_distance": 7,
                    },
                ),
            },
            name="GroupedNeighbors",
        ),
        FunctionStep(
            func=(measure_object_neighbors, common_kwargs),
            name="LaterNeighbors",
        ),
    )
    snapshots = tuple(
        StepSnapshot(
            index=index,
            scope_id=f"test::numbered-neighbors::{index}",
            step=step,
        )
        for index, step in enumerate(steps)
    )
    session = CompilationSession.from_context(
        context=ProcessingContext(
            step_plans={
                index: CompiledStepPlan(
                    step_index=index,
                    step_name=step.name,
                    step_type=type(step).__name__,
                    axis_id="A01",
                )
                for index, step in enumerate(steps)
            },
            axis_id="A01",
        ),
        steps=steps,
        orchestrator=SimpleNamespace(pipeline_config=PipelineConfig()),
        global_config=GlobalPipelineConfig(),
        step_state_map={index: object() for index in range(len(steps))},
        snapshots=snapshots,
    )

    provider = CellProfilerInvocationContractProviderFactory.provider_for_session(
        session
    )

    assert provider is not None
    module_numbers = {
        (step_index, invocation_key.group_key): declaration.producer_module_number
        for (step_index, invocation_key), plan in provider.plans.items()
        for _relationship, declaration in plan.contract.artifact_outputs.relation_refs(
            ObjectRelationshipDeclaration
        )
    }
    assert module_numbers == {
        (1, "first"): 1,
        (1, "equivalent"): 2,
        (1, "different"): 3,
        (2, "default"): 4,
    }


def test_public_numbering_reconstructs_advanced_repeated_and_distinct_modules() -> (
    None
):
    from openhcs.core.artifacts import MeasurementsArtifactType
    from openhcs.processing.backends.cellprofiler.relationships import (
        relate_objects_with_saved_children,
    )
    from openhcs.processing.backends.cellprofiler.shape import measure_object_size_shape

    object_names = ("Cells", "Cytoplasm", "Mitochondria", "Nuclei", "Nucleoli")
    object_specs = ArtifactSpecCollection(
        ArtifactSpec.output(name, ObjectLabelsArtifactType) for name in object_names
    )
    size_shape_pattern = {
        "4": [
            (
                measure_object_size_shape,
                {
                    "calculate_advanced": False,
                    "select_object_sets_to_measure": "Cells",
                },
            ),
            (
                measure_object_size_shape,
                {
                    "calculate_advanced": False,
                    "select_object_sets_to_measure": "Cytoplasm",
                },
            ),
        ],
        "5": (
            measure_object_size_shape,
            {
                "calculate_advanced": False,
                "select_object_sets_to_measure": "Mitochondria",
            },
        ),
        "1": (
            measure_object_size_shape,
            {
                "calculate_advanced": False,
                "select_object_sets_to_measure": "Nuclei",
            },
        ),
        "3": (
            measure_object_size_shape,
            {
                "calculate_advanced": False,
                "select_object_sets_to_measure": "Nucleoli",
            },
        ),
    }
    neighbor_pattern = {
        group_key: (
            measure_object_neighbors,
            {
                "distance_method": DistanceMethod.EXPAND,
                "neighbor_distance": 5,
                "select_objects_to_measure": object_name,
                "select_neighboring_objects_to_measure": object_name,
            },
        )
        for group_key, object_name in (
            ("4", "Cells"),
            ("1", "Nuclei"),
            ("5", "Mitochondria"),
        )
    }
    relate_pattern = {
        "3": (
            relate_objects_with_saved_children,
            {
                "calculate_per_parent_means": True,
                "save_children_with_parents": True,
                "name_the_output_object": "NucleoliChildObjects",
                "select_the_parent_objects": "Nuclei",
                "select_the_child_objects": "Nucleoli",
            },
        ),
        "5": (
            relate_objects_with_saved_children,
            {
                "calculate_per_parent_means": True,
                "save_children_with_parents": True,
                "name_the_output_object": "MitochondriaChildObjects",
                "select_the_parent_objects": "Cells",
                "select_the_child_objects": "Mitochondria",
            },
        ),
    }

    @artifact_outputs(*object_specs)
    @numpy(contract=ProcessingContract.PURE_3D)
    def fixture_producer(image):
        return image

    steps = (
        FunctionStep(func=fixture_producer, name="FixtureProducer"),
        FunctionStep(func=size_shape_pattern, name="MeasureObjectSizeShape"),
        FunctionStep(func=neighbor_pattern, name="MeasureObjectNeighbors"),
        FunctionStep(func=relate_pattern, name="RelateObjects"),
    )
    snapshots = tuple(
        StepSnapshot(
            index=index,
            scope_id=f"test::advanced-module-numbering::{index}",
            step=step,
        )
        for index, step in enumerate(steps)
    )
    session = CompilationSession.from_context(
        context=ProcessingContext(
            step_plans={
                index: CompiledStepPlan(
                    step_index=index,
                    step_name=step.name,
                    step_type=type(step).__name__,
                    axis_id="A01",
                )
                for index, step in enumerate(steps)
            },
            axis_id="A01",
        ),
        steps=steps,
        orchestrator=SimpleNamespace(pipeline_config=PipelineConfig()),
        global_config=GlobalPipelineConfig(),
        step_state_map={index: object() for index in range(len(steps))},
        snapshots=snapshots,
    )
    provider = CellProfilerInvocationContractProviderFactory.provider_for_session(
        session
    )

    assert provider is not None
    size_shape_measurement_names = {
        spec.name
        for (step_index, _invocation_key), plan in provider.plans.items()
        if step_index == 1
        for spec in plan.contract.artifact_outputs.for_artifact_type(
            MeasurementsArtifactType
        )
    }
    relationship_numbers = {
        (step_index, invocation_key.group_key, invocation_key.position): {
            declaration.producer_module_number
            for _relationship, declaration in plan.contract.artifact_outputs.relation_refs(
                ObjectRelationshipDeclaration
            )
        }
        for (step_index, invocation_key), plan in provider.plans.items()
        if step_index in (2, 3)
    }

    assert size_shape_measurement_names == {"MeasureObjectSizeShape_1_measurements"}
    assert relationship_numbers == {
        (2, "4", 0): {2},
        (2, "1", 0): {3},
        (2, "5", 0): {4},
        (3, "3", 0): {5},
        (3, "5", 0): {6},
    }


def test_compiled_neighbor_distance_contract_drives_runtime_rows() -> None:
    invocation = _compiled_neighbor_invocation()
    labels = np.zeros((7, 8), dtype=np.int32)
    labels[2:5, 1:3] = 1
    labels[2:5, 5:7] = 2
    payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=labels),
    )

    _image, _relationship, rows = measure_object_neighbors.__wrapped__(
        np.zeros(labels.shape, dtype=np.float32),
        payload,
        **invocation.kwargs_dict,
    )
    projected = MeasureObjectNeighborsModule.MeasurementRows.for_request(
        MeasureObjectNeighborsModule,
        SimpleNamespace(output_value=rows, call_kwargs=invocation.kwargs_dict),
    ).rows()

    assert len(projected) == 2
    assert {
        "Neighbors_NumberOfNeighbors_Expanded",
        "Neighbors_PercentTouching_Expanded",
    } <= {field.name for field in projected.fields}


def test_neighbor_projection_delegates_scale_qualification() -> None:
    labels = np.zeros((7, 8), dtype=np.int32)
    labels[2:5, 1:3] = 1
    labels[2:5, 5:7] = 2
    payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=labels),
    )
    _image, _relationship, rows = measure_object_neighbors.__wrapped__(
        np.zeros(labels.shape, dtype=np.float32),
        payload,
        distance_method=DistanceMethod.EXPAND,
        neighbor_distance=5,
    )
    calls: list[tuple[str, tuple[object, ...]]] = []

    class RecordingFeatureOwner:
        @classmethod
        def measurement_feature_name(
            cls,
            field_name: str,
            *qualified_parts: object,
        ) -> str:
            del cls
            calls.append((field_name, qualified_parts))
            return "_".join(
                ("Owned", field_name, *(str(part) for part in qualified_parts))
            )

    projected = MeasureObjectNeighborsModule.MeasurementRows(
        rows,
        module_type=RecordingFeatureOwner,
        measurement_scale="Expanded",
    ).rows()

    assert calls
    assert {qualified_parts for _field_name, qualified_parts in calls} == {
        ("Expanded",)
    }
    assert {
        "Owned_number_of_neighbors_Expanded",
        "Owned_percent_touching_Expanded",
    } <= {field.name for field in projected.fields}
