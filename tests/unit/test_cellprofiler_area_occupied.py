import inspect
from dataclasses import replace

import numpy as np
import pytest

from openhcs.core.artifacts import (
    ArtifactInputPlan,
    ArtifactSpec,
    ArtifactSpecCollection,
    ImageArtifactType,
    MeasurementsArtifactType,
    ObjectLabelsArtifactType,
)
from openhcs.core.callable_contract import CallableContract
from openhcs.core.component_group_scope import (
    ComponentGroupScope,
    RuntimeExecutionAxisScope,
)
from openhcs.core.function_patterns import (
    FunctionInvocationKey,
    normalize_function_pattern,
)
from openhcs.core.invocation_artifacts import ArtifactDeclarationStepContext
from openhcs.core.pipeline.artifact_planning import artifact_producers_for_outputs
from openhcs.core.runtime_stores import RuntimeValueStore
from openhcs.interop.cellprofiler.parser import CPPipeParser
from openhcs.processing.backends.cellprofiler.area_occupied import (
    AreaOccupiedRow,
    AreaOccupiedRowsRuntimeParameter,
    MeasureImageAreaOccupiedBinaryModule,
    OperandChoice,
    measure_image_area_occupied,
)
from openhcs.interop.cellprofiler.runtime.artifact_binding import RuntimeInputBindingRequest
from tests.unit.cellprofiler_runtime_test_support import (
    cellprofiler_runtime_adapter_for_test,
    cellprofiler_runtime_input_edge_for_test,
)


def _binding_request(
    *,
    object_inputs: tuple[ArtifactSpec, ...] = (),
    primary_image_inputs: tuple[ArtifactSpec, ...] = (),
    kwargs: dict[str, object] | None = None,
) -> RuntimeInputBindingRequest:
    contract = CallableContract.from_callable(measure_image_area_occupied)
    contract = replace(
        contract,
        metadata=replace(
            contract.metadata,
            artifact_inputs=(*primary_image_inputs, *object_inputs),
        ),
    )
    edges = tuple(
        cellprofiler_runtime_input_edge_for_test(
            ArtifactInputPlan(
                spec.name,
                f"/memory/{spec.name}",
                artifact_type=spec.artifact_type,
            ),
            spec=spec,
            input_index=input_index,
            invocation_scope=ComponentGroupScope.ungrouped(),
            producer_selection_scope=ComponentGroupScope.ungrouped(),
            component_scopes=(),
            consumer_variable_components=(),
        )
        for input_index, spec in enumerate((*primary_image_inputs, *object_inputs))
    )
    return RuntimeInputBindingRequest(
        adapter=cellprofiler_runtime_adapter_for_test(
            runtime_value_store=RuntimeValueStore(),
            callable_contract=contract,
            artifact_inputs={edge.key: edge for edge in edges},
            axis_scope=RuntimeExecutionAxisScope.from_raw(
                "A01",
                component=None,
                value=None,
            ),
        ),
        kwargs={} if kwargs is None else kwargs,
        current_image=np.zeros((2, 2), dtype=np.float32),
        selected_object_inputs=object_inputs,
    )


def test_runtime_binding_uses_authoritative_area_occupied_rows() -> None:
    binary = ArtifactSpec.input("Mask", ImageArtifactType)
    objects = ArtifactSpec.input("Nuclei", ObjectLabelsArtifactType)
    request = _binding_request(
        object_inputs=(objects,),
        primary_image_inputs=(binary,),
        kwargs={
            MeasureImageAreaOccupiedBinaryModule.operand_choices_binding.require_parameter_name(): (
                OperandChoice.BINARY_IMAGE,
                OperandChoice.OBJECTS,
            )
        },
    )

    rows = MeasureImageAreaOccupiedBinaryModule._runtime_rows(request)

    assert rows == (
        AreaOccupiedRow(OperandChoice.BINARY_IMAGE, "Mask"),
        AreaOccupiedRow(OperandChoice.OBJECTS, "Nuclei"),
    )


def test_runtime_binding_reads_default_operand_from_callable_contract() -> None:
    binary = ArtifactSpec.input("Mask", ImageArtifactType)

    rows = MeasureImageAreaOccupiedBinaryModule._runtime_rows(
        _binding_request(primary_image_inputs=(binary,))
    )

    assert rows == (AreaOccupiedRow(OperandChoice.BINARY_IMAGE, "Mask"),)


def test_runtime_binding_preserves_repeated_source_occurrences() -> None:
    binary = ArtifactSpec.input("Mask", ImageArtifactType)

    rows = MeasureImageAreaOccupiedBinaryModule._runtime_rows(
        _binding_request(
            primary_image_inputs=(binary, binary),
            kwargs={
                MeasureImageAreaOccupiedBinaryModule.operand_choices_binding.require_parameter_name(): (
                    OperandChoice.BINARY_IMAGE,
                    OperandChoice.BINARY_IMAGE,
                )
            },
        )
    )

    assert rows == (
        AreaOccupiedRow(OperandChoice.BINARY_IMAGE, "Mask"),
        AreaOccupiedRow(OperandChoice.BINARY_IMAGE, "Mask"),
    )


def test_runtime_row_identity_is_not_public_behavior() -> None:
    contract = CallableContract.from_callable(measure_image_area_occupied)

    assert contract.runtime_bound_parameter_types[0] is AreaOccupiedRowsRuntimeParameter
    signature = inspect.signature(contract.resolve_canonical_raw_callable())
    assert "input_names" not in signature.parameters
    assert "retained_image_names" not in signature.parameters
    with pytest.raises(TypeError, match="runtime-owned parameter 'area_occupied_rows'"):
        contract.validate_public_kwargs(
            {
                "area_occupied_rows": (
                    AreaOccupiedRow(OperandChoice.BINARY_IMAGE, "Mask"),
                )
            }
        )


def test_runtime_rows_must_match_public_operand_behavior() -> None:
    with pytest.raises(ValueError, match="do not match configured operand choices"):
        measure_image_area_occupied.__wrapped__(
            np.zeros((2, 2), dtype=np.float32),
            operand_choices=(OperandChoice.OBJECTS,),
            area_occupied_rows=(
                AreaOccupiedRow(OperandChoice.BINARY_IMAGE, "Mask"),
            ),
        )


def test_parsed_cppipe_area_occupied_declares_measurements_only(tmp_path) -> None:
    cppipe_path = tmp_path / "area-occupied.cppipe"
    cppipe_path.write_text(
        """CellProfiler Pipeline: https://cellprofiler.org
MeasureImageAreaOccupied:[module_num:3|enabled:True]
    Measure the area occupied by:Objects
    Select binary images to measure:
    Select object sets to measure:Nuclei, Cells
""",
        encoding="utf-8",
    )
    (module,) = CPPipeParser(cppipe_path).parse()
    invocation = next(
        normalize_function_pattern(measure_image_area_occupied).iter_items()
    )
    available = ArtifactSpecCollection(
        (
            ArtifactSpec.output("Nuclei", ObjectLabelsArtifactType),
            ArtifactSpec.output("Cells", ObjectLabelsArtifactType),
        )
    )
    context = ArtifactDeclarationStepContext(
        step_name="MeasureImageAreaOccupied",
        step_index=2,
        available_artifact_producers=artifact_producers_for_outputs(
            available.specs,
            groups=(None,),
            invocation_keys=(
                FunctionInvocationKey("fixture_producer", invocation.key.group_key, 0),
            ),
        ),
        available_artifacts=available,
        main_flow_artifacts=ArtifactSpecCollection(()),
    )

    contract = MeasureImageAreaOccupiedBinaryModule.callable_contract(
        module=module,
        invocation_key=invocation.key,
        step_context=context,
    )

    assert MeasureImageAreaOccupiedBinaryModule.measurement_rows(module) == (
        AreaOccupiedRow(OperandChoice.OBJECTS, "Nuclei"),
        AreaOccupiedRow(OperandChoice.OBJECTS, "Cells"),
    )
    assert contract.artifact_outputs.names_of_artifact_type(ImageArtifactType) == ()
    assert len(
        contract.artifact_outputs.names_of_artifact_type(MeasurementsArtifactType)
    ) == 1
    MeasureImageAreaOccupiedBinaryModule.validate_callable_artifact_abi(
        measure_image_area_occupied,
        contract,
    )
