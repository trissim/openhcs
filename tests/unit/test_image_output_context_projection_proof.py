import numpy as np
import pytest

from openhcs.constants.constants import AllComponents
from openhcs.core.artifacts import (
    ArtifactOutputPlan,
    ArtifactSpec,
    GroupLineageSourceRelation,
    ImageArtifactType,
)
from openhcs.core.runtime_image_values import (
    ImagePayloadMetadata,
    image_payload_data,
    image_payload_metadata,
    with_image_payload_data,
)
from openhcs.core.runtime_plane_projection import (
    RuntimePlaneAxis,
    RuntimePlaneAxisValueProjection,
    RuntimePlaneProjection,
)
from openhcs.core.source_image_provenance import SourceImageProvenancePlanes
from openhcs.core.steps.function_runtime import (
    FunctionOutputContextStrategy,
    ImageFunctionOutputContextStrategy,
)


def _track_ownership_proofs(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[object, ...]]:
    proof_calls: list[tuple[object, ...]] = []
    original = ImageFunctionOutputContextStrategy.output_owns_source_context

    def record_proof(*args: object) -> bool:
        proof_calls.append(args)
        return original(*args)

    monkeypatch.setattr(
        ImageFunctionOutputContextStrategy,
        "output_owns_source_context",
        staticmethod(record_proof),
    )
    return proof_calls


def _scalar_rgb_output():
    source_spec = ArtifactSpec.input("ColorNeighbors", ImageArtifactType)
    source = ImagePayloadMetadata(
        source_path="/input/A01_s001_w1_z001_t001.tif",
        source_component_metadata={
            "well": "A01",
            "site": "1",
            "channel": "1",
            "z_index": "1",
            "timepoint": "1",
        },
        source_image_names=(source_spec.name,),
        source_channel_axis=2,
    ).payload_with(np.ones((4, 5, 3), dtype=np.float32), None)
    output = with_image_payload_data(
        source,
        np.ones((4, 5, 3), dtype=np.uint8),
    )
    output_plan = ArtifactOutputPlan(
        name="SavedColorNeighbors",
        path="/memory/SavedColorNeighbors.png",
        artifact_type=ImageArtifactType,
        variable_components=(AllComponents.SITE,),
        relations=(GroupLineageSourceRelation(source_spec.ref()),),
    )
    return source, output, output_plan


def test_projected_variable_stack_proves_source_ownership_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_spec = ArtifactSpec.input("Volume", ImageArtifactType)
    source = ImagePayloadMetadata(
        source_image_names=(source_spec.name,),
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=(
                "/input/A01_s001_w1_z001_t001.tif",
                "/input/A01_s001_w1_z002_t001.tif",
            ),
            component_metadata=(
                {"well": "A01", "z_index": "1"},
                {"well": "A01", "z_index": "2"},
            ),
        ),
    ).payload_with(np.ones((2, 4, 5), dtype=np.float32), None)
    output = image_payload_metadata(source).replace_fields(
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    ).payload_with(
        np.ones((2, 4, 5), dtype=np.uint16),
        None,
    )
    output_plan = ArtifactOutputPlan(
        name="SavedVolume",
        path="/memory/SavedVolume.tif",
        artifact_type=ImageArtifactType,
        variable_components=(AllComponents.Z_INDEX,),
        relations=(GroupLineageSourceRelation(source_spec.ref()),),
    )
    proof_calls = _track_ownership_proofs(monkeypatch)

    result = FunctionOutputContextStrategy.for_output_plan(
        output_plan,
    ).contextualize_from_projector(
        source,
        output,
        output_plan,
        RuntimePlaneProjection.stack(2),
    )

    assert len(proof_calls) == 1
    projection = proof_calls[0][3]
    assert isinstance(projection, RuntimePlaneAxisValueProjection)
    assert projection.axis is RuntimePlaneAxis.RUNTIME_SLICE
    assert projection.axis_size == 2
    np.testing.assert_array_equal(
        image_payload_data(result), image_payload_data(output)
    )
    assert image_payload_metadata(result).plane_axis is RuntimePlaneAxis.RUNTIME_SLICE


def test_derived_2d_image_does_not_inherit_input_runtime_plane_axis() -> None:
    source_spec = ArtifactSpec.input("Volume", ImageArtifactType)
    source = ImagePayloadMetadata(
        source_image_names=(source_spec.name,),
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=(
                "/input/A01_s001_w1_z001_t001.tif",
                "/input/A01_s001_w1_z002_t001.tif",
            ),
            component_metadata=(
                {"well": "A01", "z_index": "1"},
                {"well": "A01", "z_index": "2"},
            ),
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    ).payload_with(np.ones((2, 4, 5), dtype=np.float32), None)
    output_plan = ArtifactOutputPlan(
        name="Projection",
        path="/memory/Projection.tif",
        artifact_type=ImageArtifactType,
        variable_components=(AllComponents.Z_INDEX,),
        relations=(GroupLineageSourceRelation(source_spec.ref()),),
    )

    result = FunctionOutputContextStrategy.for_output_plan(
        output_plan,
    ).contextualize_from_projector(
        source,
        image_payload_metadata(source)
        .collapse_leading_plane_axis()
        .payload_with(np.ones((4, 5), dtype=np.float32), None),
        output_plan,
        RuntimePlaneProjection.stack(2),
    )

    assert image_payload_data(result).shape == (4, 5)
    assert image_payload_metadata(result).plane_axis is None
    assert image_payload_metadata(result).source_image_provenance_planes.count == 2


def test_bare_full_stack_image_inherits_source_runtime_plane_axis() -> None:
    source_spec = ArtifactSpec.input("Volume", ImageArtifactType)
    source = ImagePayloadMetadata(
        source_image_names=(source_spec.name,),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    ).payload_with(np.ones((2, 4, 5), dtype=np.float32), None)
    output_plan = ArtifactOutputPlan(
        name="FilteredVolume",
        path="/memory/FilteredVolume.tif",
        artifact_type=ImageArtifactType,
        variable_components=(AllComponents.Z_INDEX,),
        relations=(GroupLineageSourceRelation(source_spec.ref()),),
    )

    result = FunctionOutputContextStrategy.for_output_plan(
        output_plan,
    ).contextualize_from_projector(
        source,
        np.ones((2, 4, 5), dtype=np.float32),
        output_plan,
        RuntimePlaneProjection.stack(2),
    )

    assert image_payload_data(result).shape == (2, 4, 5)
    assert image_payload_metadata(result).plane_axis is RuntimePlaneAxis.RUNTIME_SLICE


def test_derived_image_preserves_explicit_output_runtime_plane_axis() -> None:
    source_spec = ArtifactSpec.input("Volume", ImageArtifactType)
    source = ImagePayloadMetadata(
        source_image_names=(source_spec.name,),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    ).payload_with(np.ones((2, 4, 5), dtype=np.float32), None)
    output = ImagePayloadMetadata(
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    ).payload_with(np.ones((2, 4, 5), dtype=np.float32), None)
    output_plan = ArtifactOutputPlan(
        name="FilteredVolume",
        path="/memory/FilteredVolume.tif",
        artifact_type=ImageArtifactType,
        variable_components=(AllComponents.Z_INDEX,),
        relations=(GroupLineageSourceRelation(source_spec.ref()),),
    )

    result = FunctionOutputContextStrategy.for_output_plan(
        output_plan,
    ).contextualize_from_projector(
        source,
        output,
        output_plan,
        RuntimePlaneProjection.stack(2),
    )

    assert image_payload_data(result).shape == (2, 4, 5)
    assert image_payload_metadata(result).plane_axis is RuntimePlaneAxis.RUNTIME_SLICE


def test_projected_crop_preserves_complete_spatial_domain_after_plane_composition() -> (
    None
):
    source_spec = ArtifactSpec.input("OrigBlue", ImageArtifactType)
    source_metadata = ImagePayloadMetadata(
        source_image_names=(source_spec.name,),
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=(
                "/input/A01_s001_w1.tif",
                "/input/A01_s002_w1.tif",
            ),
            component_metadata=(
                {"well": "A01", "site": "1", "channel": "1"},
                {"well": "A01", "site": "2", "channel": "1"},
            ),
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    )
    source = source_metadata.payload_with(
        np.ones((2, 10, 12), dtype=np.float32),
        None,
    )
    cropped_planes = tuple(
        source_metadata.for_source_plane(index)
        .without_leading_plane_axis()
        .with_spatial_crop(
            input_shape_yx=(10, 12),
            output_shape_yx=(4, 5),
            offset_yx=(3, 2),
        )
        .payload_with(np.ones((4, 5), dtype=np.float32), None)
        for index in range(2)
    )
    output = ImagePayloadMetadata.compose(cropped_planes).payload_with(
        np.ones((2, 4, 5), dtype=np.float32),
        None,
    )
    output_plan = ArtifactOutputPlan(
        name="CropBlue",
        path="/memory/CropBlue.pkl",
        artifact_type=ImageArtifactType,
        variable_components=(AllComponents.SITE,),
        relations=(GroupLineageSourceRelation(source_spec.ref()),),
    )

    result = FunctionOutputContextStrategy.for_output_plan(
        output_plan,
    ).contextualize_from_projector(
        source,
        output,
        output_plan,
        RuntimePlaneProjection.stack(2),
    )

    assert result is output
    assert image_payload_metadata(result).source_spatial_domain == (
        image_payload_metadata(cropped_planes[0]).source_spatial_domain
    )


def test_projected_scalar_rgb_proves_complete_identity_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, output, output_plan = _scalar_rgb_output()
    proof_calls = _track_ownership_proofs(monkeypatch)

    result = FunctionOutputContextStrategy.for_output_plan(
        output_plan,
    ).contextualize_from_projector(
        source,
        output,
        output_plan,
        RuntimePlaneProjection.stack(1),
    )

    assert len(proof_calls) == 1
    projection = proof_calls[0][3]
    assert isinstance(projection, RuntimePlaneAxisValueProjection)
    assert projection.axis is RuntimePlaneAxis.RUNTIME_SLICE
    assert projection.axis_size == 1
    assert result is output
    assert image_payload_metadata(result).plane_axis is None
    assert image_payload_metadata(result).normalized_source_channel_axis(result) == 2


def test_owned_variable_output_without_projector_returns_after_one_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, output, output_plan = _scalar_rgb_output()
    proof_calls = _track_ownership_proofs(monkeypatch)

    result = FunctionOutputContextStrategy.for_output_plan(
        output_plan,
    ).contextualize_from_projector(
        source,
        output,
        output_plan,
        None,
    )

    assert len(proof_calls) == 1
    assert proof_calls[0][3] is None
    assert result is output


def test_unowned_variable_output_without_projector_preserves_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_spec = ArtifactSpec.input("Volume", ImageArtifactType)
    output_plan = ArtifactOutputPlan(
        name="SavedVolume",
        path="/memory/SavedVolume.tif",
        artifact_type=ImageArtifactType,
        variable_components=(AllComponents.Z_INDEX,),
        relations=(GroupLineageSourceRelation(source_spec.ref()),),
    )
    proof_calls = _track_ownership_proofs(monkeypatch)

    with pytest.raises(
        ValueError,
        match="runtime invocation supplies no plane projector",
    ):
        FunctionOutputContextStrategy.for_output_plan(
            output_plan,
        ).contextualize_from_projector(
            np.ones((2, 4, 5), dtype=np.float32),
            np.ones((2, 4, 5), dtype=np.uint16),
            output_plan,
            None,
        )

    assert len(proof_calls) == 1
    assert proof_calls[0][3] is None
