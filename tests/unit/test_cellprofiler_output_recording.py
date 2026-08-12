from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from openhcs.constants.constants import VariableComponents
from openhcs.core.aligned_image_payload import ImagePayloadExecutionMode
from openhcs.core.artifacts import (
    ArtifactOutputPlan,
    ArtifactSpec,
    ArtifactSpecRelation,
    ImageArtifactType,
    ObjectLabelsArtifactType,
    ObjectLineageArtifactType,
    SourceStackLineageSourceRelation,
    SpatialGridArtifactType,
)
from openhcs.core.callable_contract import CallableContract, CallableMetadata
from openhcs.core.runtime_adapters import RuntimeFunctionInvocationRequest
from openhcs.core.runtime_image_values import (
    ImagePayloadMetadata,
    image_payload_data,
    image_payload_metadata,
)
from openhcs.core.runtime_plane_projection import (
    RuntimePlaneAxis,
    RuntimePlaneAxisValueProjection,
)
from openhcs.core.runtime_relationships import (
    DirectedObjectRelationshipPayload,
    ObjectRelationshipDeclaration,
)
from openhcs.core.runtime_slice_alignment import RuntimeSliceAlignedValues
from openhcs.interop.cellprofiler.module_declarations import CellProfilerModule
from openhcs.interop.cellprofiler.runtime.adapter import CellProfilerRuntimeAdapter
from openhcs.interop.cellprofiler.runtime.invocation import CellProfilerImageRequest
from openhcs.interop.cellprofiler.runtime.output_record_request import (
    CellProfilerOutputRecordRequest,
)
from openhcs.interop.cellprofiler.runtime.output_recording import (
    CellProfilerOutputRecorder,
    ImageOutputRecorder,
    RelationshipsOutputRecorder,
)


def _contract(*, outputs: tuple[ArtifactSpec, ...]) -> CallableContract:
    return CallableContract(
        func=lambda image: image,
        function_name="recording_probe",
        module_name=__name__,
        metadata=CallableMetadata(
            artifact_outputs=outputs,
            runtime_adapter=CellProfilerRuntimeAdapter.runtime_adapter_spec(),
        ),
    )


def _image_request(image: np.ndarray) -> CellProfilerImageRequest:
    return CellProfilerImageRequest(
        image_count=1,
        payload=ImagePayloadMetadata().payload_with(image),
        source_image_name=None,
        source_aliases=(),
        execution_mode=ImagePayloadExecutionMode.NATURAL,
    )


def test_output_recording_uses_artifact_dependency_order() -> None:
    source = ArtifactSpec.output("source", SpatialGridArtifactType)
    dependent = ArtifactSpec.output(
        "dependent",
        SpatialGridArtifactType,
        relations=(ArtifactSpecRelation(source.ref()),),
    )
    source_plan = ArtifactOutputPlan(
        name=source.name,
        path="source.pkl",
        artifact_type=source.artifact_type,
    )
    dependent_plan = ArtifactOutputPlan(
        name=dependent.name,
        path="dependent.pkl",
        artifact_type=dependent.artifact_type,
        relations=dependent.relations,
    )
    adapter = Mock(spec=CellProfilerRuntimeAdapter)
    adapter.request = SimpleNamespace(
        artifact_outputs={dependent_plan.ref(): dependent_plan, source_plan.ref(): source_plan}
    )
    image = np.zeros((2, 2), dtype=np.float32)
    invocation = RuntimeFunctionInvocationRequest(
        image_count=1,
        image=image,
        kwargs={},
        source_image_name=None,
        execution_mode=ImagePayloadExecutionMode.NATURAL,
    )

    CellProfilerOutputRecorder.record_module_outputs(
        callable_contract=_contract(outputs=(dependent, source)),
        active_input_edges=(),
        adapter=adapter,
        returned_values={dependent.ref(): "dependent", source.ref(): "source"},
        matched_outputs=(
            (dependent_plan, dependent, "dependent"),
            (source_plan, source, "source"),
        ),
        invocation=invocation,
        image_request=_image_request(image),
        current_image=image,
    )

    assert tuple(call.args[0] for call in adapter.add_spatial_grid.call_args_list) == (
        "source",
        "dependent",
    )


def test_relationship_recording_uses_exact_artifact_relation() -> None:
    parent = ArtifactSpec.input("Parent", ObjectLabelsArtifactType)
    child = ArtifactSpec.output("Child", ObjectLabelsArtifactType)
    declaration = ObjectRelationshipDeclaration.parent_child(
        source=parent.ref(),
        target=child.ref(),
        producer_module_number=1,
    )
    relationship = ArtifactSpec.output(
        "Parent_Child_relationships",
        ObjectLineageArtifactType,
        relations=(declaration,),
    )
    output_plan = ArtifactOutputPlan(
        name=relationship.name,
        path="relationship.pkl",
        artifact_type=relationship.artifact_type,
        relations=relationship.relations,
    )
    adapter = Mock(spec=CellProfilerRuntimeAdapter)
    adapter.request = SimpleNamespace(artifact_outputs={output_plan.ref(): output_plan})
    source = _image_request(np.zeros((2, 2), dtype=np.float32))
    request = CellProfilerOutputRecordRequest(
        callable_contract=_contract(outputs=(child, relationship)),
        active_input_edges=(),
        adapter=adapter,
        spec=relationship,
        output_plan=output_plan,
        output_value=DirectedObjectRelationshipPayload(
            source_ids=(1,),
            target_ids=(2,),
        ),
        source=source,
        call_kwargs={},
        current_image=source.payload,
    )

    RelationshipsOutputRecorder().record(request)

    recorded = adapter.add_relationship.call_args.args[0]
    assert recorded.declaration == declaration


@pytest.mark.parametrize(
    ("plane_projection", "output_shape", "expected_plane_axis"),
    (
        (None, (4, 5, 3), None),
        (
            RuntimePlaneAxisValueProjection.preserve(
                axis=RuntimePlaneAxis.RUNTIME_SLICE,
                axis_size=1,
            ),
            (1, 4, 5, 3),
            RuntimePlaneAxis.RUNTIME_SLICE,
        ),
    ),
)
def test_image_output_recording_uses_exact_invocation_projection_for_rgb(
    monkeypatch: pytest.MonkeyPatch,
    plane_projection: RuntimePlaneAxisValueProjection | None,
    output_shape: tuple[int, ...],
    expected_plane_axis: RuntimePlaneAxis | None,
) -> None:
    measured_objects = ArtifactSpec.input("Cells", ObjectLabelsArtifactType)
    output = ArtifactSpec.output(
        "ColorNeighbors",
        ImageArtifactType,
        relations=(
            SourceStackLineageSourceRelation(source=measured_objects.ref()),
        ),
    )
    output_plan = ArtifactOutputPlan(
        name=output.name,
        path="ColorNeighbors.tif",
        artifact_type=output.artifact_type,
        relations=output.relations,
        variable_components=(VariableComponents.SITE,),
    )
    source_slice = ImagePayloadMetadata(
        source_image_names=(measured_objects.name,),
    ).payload_with(np.zeros((4, 5), dtype=np.float32))
    source_payload = RuntimeSliceAlignedValues((source_slice,))
    output_value = np.zeros(output_shape, dtype=np.float32)
    adapter = Mock(spec=CellProfilerRuntimeAdapter)

    class RecordingProbeModule:
        @classmethod
        def output_value(cls, request: CellProfilerOutputRecordRequest) -> object:
            return request.output_value

        @classmethod
        def source_payload(cls, request: CellProfilerOutputRecordRequest) -> object:
            return request.source.payload

    monkeypatch.setattr(
        CellProfilerModule,
        "require_callable_contract_owner",
        classmethod(lambda cls, contract: RecordingProbeModule),
    )

    ImageOutputRecorder().record(
        CellProfilerOutputRecordRequest(
            callable_contract=_contract(outputs=(output,)),
            active_input_edges=(),
            adapter=adapter,
            spec=output,
            output_plan=output_plan,
            output_value=output_value,
            source=CellProfilerImageRequest(
                image_count=1,
                payload=source_payload,
                source_image_name=measured_objects.name,
                source_aliases=(),
                execution_mode=ImagePayloadExecutionMode.NATURAL,
                plane_projection=plane_projection,
            ),
            call_kwargs={},
            current_image=source_slice,
        )
    )

    recorded = adapter.add_image.call_args.args[1]
    assert image_payload_data(recorded).shape == output_shape
    assert image_payload_metadata(recorded).plane_axis is expected_plane_axis


def test_output_recording_carries_exact_invocation_plane_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = ArtifactSpec.output("grid", SpatialGridArtifactType)
    output_plan = ArtifactOutputPlan(
        name=output.name,
        path="grid.pkl",
        artifact_type=output.artifact_type,
    )
    adapter = Mock(spec=CellProfilerRuntimeAdapter)
    adapter.request = SimpleNamespace(artifact_outputs={output_plan.ref(): output_plan})
    image = np.zeros((2, 2), dtype=np.float32)
    stale_projection = RuntimePlaneAxisValueProjection.preserve(
        axis=RuntimePlaneAxis.RUNTIME_SLICE,
        axis_size=1,
    )
    invocation_projection = RuntimePlaneAxisValueProjection.preserve(
        axis=RuntimePlaneAxis.RUNTIME_SLICE,
        axis_size=2,
    )
    recorder = Mock()
    monkeypatch.setattr(
        CellProfilerOutputRecorder,
        "for_artifact_type",
        classmethod(lambda cls, artifact_type: recorder),
    )

    CellProfilerOutputRecorder.record_module_outputs(
        callable_contract=_contract(outputs=(output,)),
        active_input_edges=(),
        adapter=adapter,
        returned_values={output.ref(): "grid"},
        matched_outputs=((output_plan, output, "grid"),),
        invocation=RuntimeFunctionInvocationRequest(
            image_count=2,
            image=image,
            kwargs={},
            source_image_name=None,
            execution_mode=ImagePayloadExecutionMode.FULL_STACK,
            plane_projection=invocation_projection,
        ),
        image_request=CellProfilerImageRequest(
            image_count=1,
            payload=ImagePayloadMetadata().payload_with(image),
            source_image_name=None,
            source_aliases=(),
            execution_mode=ImagePayloadExecutionMode.NATURAL,
            plane_projection=stale_projection,
        ),
        current_image=image,
    )

    request = recorder.record.call_args.args[0]
    assert request.source.plane_projection is invocation_projection
