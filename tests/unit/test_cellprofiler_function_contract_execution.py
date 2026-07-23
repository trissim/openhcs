"""Focused tests for the CellProfiler mode/processing-contract boundary."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from openhcs.core.aligned_image_payload import (
    AlignedImageStack,
    ImageOutputBundle,
    ImagePayloadExecutionMode,
    compose_aligned_image_payload,
)
from openhcs.core.artifacts import (
    ArtifactSpec,
    ImageArtifactType,
    MeasurementsArtifactType,
)
from openhcs.core.callable_contract import CallableContract, CallableMetadata
from openhcs.core.runtime_plane_projection import RuntimePlaneAxis, RuntimePlaneAxisValueProjection
from openhcs.core.runtime_slice_alignment import RuntimeSliceAlignedValues
from openhcs.core.runtime_slice_projection import RuntimeSliceProjectionDeclarationError
from openhcs.core.runtime_image_values import (
    ImagePayloadMetadata,
    image_payload_data,
)
from openhcs.core.source_spatial_domain import SourceSpatialDomain
from openhcs.core.source_image_provenance import SourceImageProvenancePlanes
from openhcs.interop.cellprofiler.runtime.function_contract_execution import (
    CellProfilerFunctionContractExecutor,
)
from openhcs.processing.backends.cellprofiler.alignment import AlignShiftMeasurement
from openhcs.processing.backends.cellprofiler.morphology import morphologicalskeleton
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract


def _compiled_contract(
    func: Callable[..., object],
    processing_contract: ProcessingContract,
    *,
    artifact_inputs: tuple[ArtifactSpec, ...] = (),
    artifact_outputs: tuple[ArtifactSpec, ...] = (),
) -> CallableContract:
    return CallableContract(
        func=func,
        function_name="dispatch_probe",
        module_name="DispatchProbeModule",
        metadata=CallableMetadata(
            processing_contract=processing_contract,
            artifact_inputs=artifact_inputs,
            artifact_outputs=artifact_outputs,
        ),
    )


def test_morphological_skeleton_executes_one_planar_runtime_image() -> None:
    callable_contract = CallableContract.from_callable(morphologicalskeleton)
    raw_callable = callable_contract.resolve_canonical_raw_callable()
    image = np.zeros((7, 7), dtype=np.float32)
    image[2:5, 3] = 1.0

    result = CellProfilerFunctionContractExecutor().execute(
        callable_contract,
        raw_callable,
        image,
        {},
        execution_mode=ImagePayloadExecutionMode.NATURAL,
    )

    assert callable_contract.processing_contract is ProcessingContract.PURE_2D
    assert image_payload_data(result).shape == image.shape


@pytest.mark.parametrize("processing_contract", tuple(ProcessingContract))
@pytest.mark.parametrize(
    ("image_mode", "expected_call_count"),
    (
        (ImagePayloadExecutionMode.NATURAL, 1),
        (ImagePayloadExecutionMode.FULL_STACK, 1),
        (ImagePayloadExecutionMode.ALIGNED_MULTI_IMAGE_STACK, 2),
    ),
)
def test_executor_owns_the_closed_mode_processing_contract_matrix(
    processing_contract: ProcessingContract,
    image_mode: ImagePayloadExecutionMode,
    expected_call_count: int,
) -> None:
    calls: list[tuple[int, ...]] = []

    def dispatch_probe(image: np.ndarray) -> np.ndarray:
        calls.append(image.shape)
        if processing_contract is ProcessingContract.VOLUMETRIC_TO_SLICE:
            return image[0]
        return image

    callable_contract = _compiled_contract(dispatch_probe, processing_contract)
    image = (
        AlignedImageStack(
            slices=(
                np.zeros((2, 3), dtype=np.float32),
                np.ones((2, 3), dtype=np.float32),
            )
        )
        if image_mode is ImagePayloadExecutionMode.ALIGNED_MULTI_IMAGE_STACK
        else (
            ImagePayloadMetadata(
                    plane_axis=RuntimePlaneAxis.RUNTIME_SLICE
                ).payload_with(np.zeros((2, 2, 3), dtype=np.float32), None)
            if processing_contract is ProcessingContract.VOLUMETRIC_TO_SLICE
            else np.zeros((2, 3), dtype=np.float32)
        )
    )

    if (
        image_mode is ImagePayloadExecutionMode.ALIGNED_MULTI_IMAGE_STACK
        and processing_contract is ProcessingContract.PURE_3D
    ):
        with pytest.raises(
            ValueError,
            match=(
                "DispatchProbeModule.*dispatch_probe.*"
                "PURE_3D.*RuntimePlaneAxis.SOURCE_BINDING"
            ),
        ):
            CellProfilerFunctionContractExecutor().execute(
                callable_contract,
                dispatch_probe,
                image,
                {},
                execution_mode=image_mode,
                plane_projection=RuntimePlaneAxisValueProjection.preserve(
                    axis=RuntimePlaneAxis.RUNTIME_SLICE,
                    axis_size=2,
                ),
            )
        assert calls == []
        return

    CellProfilerFunctionContractExecutor().execute(
        callable_contract,
        dispatch_probe,
        image,
        {},
        execution_mode=image_mode,
        plane_projection=(
            RuntimePlaneAxisValueProjection.preserve(
                axis=RuntimePlaneAxis.RUNTIME_SLICE,
                axis_size=2,
            )
            if image_mode is ImagePayloadExecutionMode.ALIGNED_MULTI_IMAGE_STACK
            else None
        ),
    )

    assert len(calls) == expected_call_count


def test_aligned_pure_2d_consumes_unique_declared_source_binding_plane() -> None:
    calls: list[tuple[int, ...]] = []

    def dispatch_probe(image: np.ndarray) -> np.ndarray:
        calls.append(image.shape)
        return image

    image_spec = ArtifactSpec.input("DNA", ImageArtifactType)
    callable_contract = _compiled_contract(
        dispatch_probe,
        ProcessingContract.PURE_2D,
        artifact_inputs=(image_spec,),
    )
    source_stack = ImagePayloadMetadata(
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=("/tmp/site-1.tif", "/tmp/site-2.tif"),
                component_metadata=(({"site": "1"}), ({"site": "2"})),
            ),
        ).payload_with(np.stack(
            (
                np.zeros((2, 3), dtype=np.float32),
                np.ones((2, 3), dtype=np.float32),
            )
        ), None)
    composition = compose_aligned_image_payload(
        "DispatchProbeModule image inputs ('DNA',)",
        (source_stack,),
    )

    CellProfilerFunctionContractExecutor().execute(
        callable_contract,
        dispatch_probe,
        composition.payload,
        {},
        execution_mode=composition.execution_mode,
        plane_projection=RuntimePlaneAxisValueProjection.preserve(
            axis=RuntimePlaneAxis.RUNTIME_SLICE,
            axis_size=2,
        ),
    )

    assert calls == [(2, 3), (2, 3)]


def test_declared_unaligned_input_joins_runtime_slices_before_pure_2d_execution() -> (
    None
):
    calls: list[tuple[int, ...]] = []

    def dispatch_probe(image: object) -> np.ndarray:
        data = np.asarray(image_payload_data(image))
        calls.append(data.shape)
        return data[0] - data[1]

    spatial_domain = SourceSpatialDomain(source_shape_yx=(2, 3))
    source_stack = ImagePayloadMetadata(
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
            source_spatial_domain=spatial_domain,
            source_image_names=("Raw", "Raw"),
        ).payload_with(np.stack(
            (
                np.full((2, 3), 11, dtype=np.float32),
                np.full((2, 3), 22, dtype=np.float32),
            )
        ), None)
    illumination = ImagePayloadMetadata(
            source_spatial_domain=spatial_domain,
            source_image_names=("Illum",),
        ).payload_with(np.full((2, 3), 3, dtype=np.float32), None)

    with pytest.raises(ValueError, match="explicit runtime-slice owner"):
        compose_aligned_image_payload(
            "DispatchProbeModule image inputs ('Raw', 'Illum')",
            (source_stack, illumination),
        )

    composition = compose_aligned_image_payload(
        "DispatchProbeModule image inputs ('Raw', 'Illum')",
        (source_stack, illumination),
        stack_broadcast_source_indices=(None, 0),
    )
    output_spec = ArtifactSpec.output("Corrected", ImageArtifactType)
    callable_contract = _compiled_contract(
        dispatch_probe,
        ProcessingContract.PURE_2D,
        artifact_outputs=(output_spec,),
    )
    projection = RuntimePlaneAxisValueProjection.preserve(
        axis=RuntimePlaneAxis.RUNTIME_SLICE,
        axis_size=2,
        source_aliases=("Raw", "Illum"),
    )

    result = CellProfilerFunctionContractExecutor().execute(
        callable_contract,
        dispatch_probe,
        composition.payload,
        {},
        execution_mode=composition.execution_mode,
        plane_projection=projection,
    )

    assert calls == [(2, 2, 3), (2, 2, 3)]
    np.testing.assert_array_equal(
        image_payload_data(result),
        np.stack(
            (
                np.full((2, 3), 8, dtype=np.float32),
                np.full((2, 3), 19, dtype=np.float32),
            )
        ),
    )


@pytest.mark.parametrize(
    "processing_contract",
    (ProcessingContract.PURE_2D, ProcessingContract.PURE_3D),
)
def test_aligned_execution_preserves_each_slice_inner_source_binding_axis(
    processing_contract: ProcessingContract,
) -> None:
    calls: list[tuple[int, ...]] = []

    def dispatch_probe(image: np.ndarray) -> np.ndarray:
        calls.append(image.shape)
        return image[0]

    pair = ImagePayloadMetadata(
            plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
            source_image_names=("Orig", "Illum"),
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=("/tmp/orig.tif", "/tmp/illum.pkl"),
            ),
        ).payload_with(np.zeros((2, 2, 3), dtype=np.float32), None)
    callable_contract = _compiled_contract(
        dispatch_probe,
        processing_contract,
    )

    CellProfilerFunctionContractExecutor().execute(
        callable_contract,
        dispatch_probe,
        AlignedImageStack(slices=(pair, pair)),
        {},
        execution_mode=ImagePayloadExecutionMode.ALIGNED_MULTI_IMAGE_STACK,
        plane_projection=RuntimePlaneAxisValueProjection.preserve(
            axis=RuntimePlaneAxis.RUNTIME_SLICE,
            axis_size=2,
        ),
    )

    assert calls == [(2, 2, 3), (2, 2, 3)]


def test_aligned_stack_type_mismatch_fails_before_raw_invocation() -> None:
    calls = 0

    def dispatch_probe(image: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        return image

    callable_contract = _compiled_contract(
        dispatch_probe,
        ProcessingContract.PURE_2D,
    )

    with pytest.raises(
        TypeError,
        match="DispatchProbeModule.*dispatch_probe.*AlignedImageStack.*ndarray",
    ):
        CellProfilerFunctionContractExecutor().execute(
            callable_contract,
            dispatch_probe,
            np.zeros((2, 3), dtype=np.float32),
            {},
            execution_mode=ImagePayloadExecutionMode.ALIGNED_MULTI_IMAGE_STACK,
            plane_projection=RuntimePlaneAxisValueProjection.preserve(
                axis=RuntimePlaneAxis.RUNTIME_SLICE,
                axis_size=2,
            ),
        )

    assert calls == 0


def test_aligned_stack_requires_exact_compiled_runtime_slice_projection() -> None:
    calls = 0

    def dispatch_probe(image: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        return image

    callable_contract = _compiled_contract(
        dispatch_probe,
        ProcessingContract.PURE_2D,
    )
    image = AlignedImageStack(
        slices=(
            np.zeros((2, 3), dtype=np.float32),
            np.ones((2, 3), dtype=np.float32),
        )
    )

    with pytest.raises(
        ValueError,
        match="compiled runtime-slice projection",
    ):
        CellProfilerFunctionContractExecutor().execute(
            callable_contract,
            dispatch_probe,
            image,
            {},
            execution_mode=ImagePayloadExecutionMode.ALIGNED_MULTI_IMAGE_STACK,
        )

    with pytest.raises(
        ValueError,
        match="cardinality.*2 != 3",
    ):
        CellProfilerFunctionContractExecutor().execute(
            callable_contract,
            dispatch_probe,
            image,
            {},
            execution_mode=ImagePayloadExecutionMode.ALIGNED_MULTI_IMAGE_STACK,
            plane_projection=RuntimePlaneAxisValueProjection.preserve(
                axis=RuntimePlaneAxis.RUNTIME_SLICE,
                axis_size=3,
            ),
        )

    assert calls == 0


def test_aligned_stack_projects_runtime_plane_projection_kwarg_nominally() -> None:
    received: list[RuntimePlaneAxisValueProjection] = []

    def dispatch_probe(
        image: np.ndarray,
        *,
        runtime_plane_projection: RuntimePlaneAxisValueProjection,
    ) -> np.ndarray:
        received.append(runtime_plane_projection)
        return image

    callable_contract = _compiled_contract(
        dispatch_probe,
        ProcessingContract.PURE_2D,
    )
    preserved = RuntimePlaneAxisValueProjection.preserve(
        axis=RuntimePlaneAxis.RUNTIME_SLICE,
        axis_size=2,
    )

    CellProfilerFunctionContractExecutor().execute(
        callable_contract,
        dispatch_probe,
        AlignedImageStack(
            slices=(
                np.zeros((2, 3), dtype=np.float32),
                np.ones((2, 3), dtype=np.float32),
            )
        ),
        {"runtime_plane_projection": preserved},
        execution_mode=ImagePayloadExecutionMode.ALIGNED_MULTI_IMAGE_STACK,
        plane_projection=preserved,
    )

    assert received == [preserved.selected_plane(0), preserved.selected_plane(1)]


@pytest.mark.parametrize(
    "execution_mode",
    (ImagePayloadExecutionMode.NATURAL, ImagePayloadExecutionMode.FULL_STACK),
)
def test_declared_output_axis_contextualizes_multiple_canonical_outputs(
    execution_mode: ImagePayloadExecutionMode,
) -> None:
    trailing = (
        AlignShiftMeasurement(
            slice_index=0,
            output_index=0,
            x_shift=1.0,
            y_shift=2.0,
        ),
    )

    def dispatch_probe(
        image: np.ndarray,
    ) -> tuple[object, tuple[AlignShiftMeasurement, ...]]:
        outputs = np.stack((np.asarray(image) + 1, np.asarray(image) + 2))
        return (
            ImagePayloadMetadata(
                plane_axis=RuntimePlaneAxis.RUNTIME_SLICE
            ).payload_with(outputs, None),
            trailing,
        )

    callable_contract = _compiled_contract(
        dispatch_probe,
        ProcessingContract.PURE_3D,
        artifact_outputs=(
            ArtifactSpec.output("First", ImageArtifactType),
            ArtifactSpec.output("Second", ImageArtifactType),
            ArtifactSpec.output("Measurements", MeasurementsArtifactType),
        ),
    )
    source = np.arange(6, dtype=np.float32).reshape((2, 3))

    result = CellProfilerFunctionContractExecutor().execute(
        callable_contract,
        dispatch_probe,
        source,
        {},
        execution_mode=execution_mode,
        plane_projection=RuntimePlaneAxisValueProjection.preserve(
            axis=RuntimePlaneAxis.RUNTIME_SLICE,
            axis_size=2,
        ),
    )

    assert isinstance(result, tuple)
    outputs, returned_trailing = result
    assert isinstance(outputs, ImageOutputBundle)
    assert tuple(context.output_key for context in outputs.slice_contexts) == (
        "First",
        "Second",
    )
    np.testing.assert_array_equal(image_payload_data(outputs.slices[0]), source + 1)
    np.testing.assert_array_equal(image_payload_data(outputs.slices[1]), source + 2)
    assert returned_trailing == trailing


def test_multi_canonical_output_rejects_undeclared_array_axis() -> None:
    def dispatch_probe(image: np.ndarray) -> np.ndarray:
        return np.stack((np.asarray(image) + 1, np.asarray(image) + 2))

    callable_contract = _compiled_contract(
        dispatch_probe,
        ProcessingContract.PURE_3D,
        artifact_outputs=(
            ArtifactSpec.output("First", ImageArtifactType),
            ArtifactSpec.output("Second", ImageArtifactType),
        ),
    )

    with pytest.raises(
        RuntimeSliceProjectionDeclarationError,
        match="returned payload does not declare the compiled 'runtime_slice' plane axis",
    ):
        CellProfilerFunctionContractExecutor().execute(
            callable_contract,
            dispatch_probe,
            np.zeros((2, 3), dtype=np.float32),
            {},
            execution_mode=ImagePayloadExecutionMode.FULL_STACK,
            plane_projection=RuntimePlaneAxisValueProjection.preserve(
                axis=RuntimePlaneAxis.RUNTIME_SLICE,
                axis_size=2,
            ),
        )


def test_multi_canonical_output_requires_exact_projection_cardinality() -> None:
    def dispatch_probe(image: np.ndarray) -> object:
        outputs = np.stack(
            tuple(np.asarray(image) + output_index for output_index in range(3))
        )
        return ImagePayloadMetadata(
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE
        ).payload_with(outputs, None)

    callable_contract = _compiled_contract(
        dispatch_probe,
        ProcessingContract.PURE_3D,
        artifact_outputs=(
            ArtifactSpec.output("First", ImageArtifactType),
            ArtifactSpec.output("Second", ImageArtifactType),
        ),
    )

    with pytest.raises(
        ValueError,
        match="declares 2 canonical outputs.*projection declares 3 value",
    ):
        CellProfilerFunctionContractExecutor().execute(
            callable_contract,
            dispatch_probe,
            np.zeros((2, 3), dtype=np.float32),
            {},
            execution_mode=ImagePayloadExecutionMode.FULL_STACK,
            plane_projection=RuntimePlaneAxisValueProjection.preserve(
                axis=RuntimePlaneAxis.RUNTIME_SLICE,
                axis_size=3,
            ),
        )


@pytest.mark.parametrize("runtime_slice_count", (1, 2))
def test_aligned_stack_transposes_multiple_canonical_outputs_across_runtime_slices(
    runtime_slice_count: int,
) -> None:
    def dispatch_probe(
        image: np.ndarray,
    ) -> tuple[AlignedImageStack, tuple[AlignShiftMeasurement, ...]]:
        return (
            AlignedImageStack(
                tuple(np.asarray(image) + output_index for output_index in range(3))
            ),
            (
                AlignShiftMeasurement(
                    slice_index=0,
                    output_index=0,
                    x_shift=float(np.asarray(image)[0, 0]),
                    y_shift=0.0,
                ),
            ),
        )

    output_specs = (
        *(
            ArtifactSpec.output(f"Aligned{index}", ImageArtifactType)
            for index in range(3)
        ),
        ArtifactSpec.output("Measurements", MeasurementsArtifactType),
    )
    callable_contract = _compiled_contract(
        dispatch_probe,
        ProcessingContract.PURE_2D,
        artifact_outputs=output_specs,
    )
    runtime_slices = tuple(
        np.full((2, 3), slice_index * 10, dtype=np.float32)
        for slice_index in range(runtime_slice_count)
    )

    result = CellProfilerFunctionContractExecutor().execute(
        callable_contract,
        dispatch_probe,
        AlignedImageStack(runtime_slices),
        {},
        execution_mode=ImagePayloadExecutionMode.ALIGNED_MULTI_IMAGE_STACK,
        plane_projection=RuntimePlaneAxisValueProjection.preserve(
            axis=RuntimePlaneAxis.RUNTIME_SLICE,
            axis_size=runtime_slice_count,
        ),
    )

    assert isinstance(result, tuple)
    aligned_outputs = result[0]
    assert isinstance(aligned_outputs, AlignedImageStack)
    assert len(aligned_outputs.slices) == 3
    for output_index, output in enumerate(aligned_outputs.slices):
        expected = tuple(value + output_index for value in runtime_slices)
        np.testing.assert_array_equal(image_payload_data(output), np.stack(expected))


def test_aligned_stack_preserves_one_scalar_output_per_declared_surface() -> None:
    def dispatch_probe(image: np.ndarray) -> np.ndarray:
        return np.asarray(image) + 1

    output_specs = tuple(
        ArtifactSpec.output(f"Corrected{index}", ImageArtifactType)
        for index in range(2)
    )
    callable_contract = _compiled_contract(
        dispatch_probe,
        ProcessingContract.PURE_2D,
        artifact_outputs=output_specs,
    )
    input_surfaces = (
        np.zeros((2, 3), dtype=np.float32),
        np.full((2, 3), 10, dtype=np.float32),
    )

    result = CellProfilerFunctionContractExecutor().execute(
        callable_contract,
        dispatch_probe,
        AlignedImageStack(input_surfaces),
        {},
        execution_mode=ImagePayloadExecutionMode.ALIGNED_MULTI_IMAGE_STACK,
        plane_projection=RuntimePlaneAxisValueProjection.preserve(
            axis=RuntimePlaneAxis.RUNTIME_SLICE,
            axis_size=2,
        ),
    )

    assert isinstance(result, AlignedImageStack)
    assert len(result.slices) == 2
    for output, source in zip(result.slices, input_surfaces, strict=True):
        np.testing.assert_array_equal(image_payload_data(output), source + 1)


def test_aligned_stack_aggregates_one_canonical_output_for_one_runtime_slice() -> None:
    def dispatch_probe(image: np.ndarray) -> np.ndarray:
        return np.asarray(image)[0] + np.asarray(image)[1]

    callable_contract = _compiled_contract(
        dispatch_probe,
        ProcessingContract.PURE_2D,
        artifact_outputs=(ArtifactSpec.output("Combined", ImageArtifactType),),
    )
    first = np.full((2, 3), 2, dtype=np.float32)
    second = np.full((2, 3), 3, dtype=np.float32)

    result = CellProfilerFunctionContractExecutor().execute(
        callable_contract,
        dispatch_probe,
        AlignedImageStack((np.stack((first, second)),)),
        {},
        execution_mode=ImagePayloadExecutionMode.ALIGNED_MULTI_IMAGE_STACK,
        plane_projection=RuntimePlaneAxisValueProjection.preserve(
            axis=RuntimePlaneAxis.RUNTIME_SLICE,
            axis_size=1,
        ),
    )

    assert not isinstance(result, AlignedImageStack)
    np.testing.assert_array_equal(
        image_payload_data(result),
        np.full((1, 2, 3), 5, dtype=np.float32),
    )


def test_aligned_stack_unwraps_one_declared_surface_after_slice_transpose() -> None:
    def dispatch_probe(image: np.ndarray) -> AlignedImageStack:
        return AlignedImageStack((np.asarray(image) + 1,))

    callable_contract = _compiled_contract(
        dispatch_probe,
        ProcessingContract.PURE_2D,
        artifact_outputs=(ArtifactSpec.output("Corrected", ImageArtifactType),),
    )
    runtime_slices = (
        np.zeros((2, 3), dtype=np.float32),
        np.full((2, 3), 10, dtype=np.float32),
    )

    result = CellProfilerFunctionContractExecutor().execute(
        callable_contract,
        dispatch_probe,
        AlignedImageStack(runtime_slices),
        {},
        execution_mode=ImagePayloadExecutionMode.ALIGNED_MULTI_IMAGE_STACK,
        plane_projection=RuntimePlaneAxisValueProjection.preserve(
            axis=RuntimePlaneAxis.RUNTIME_SLICE,
            axis_size=2,
        ),
    )

    assert not isinstance(result, AlignedImageStack)
    np.testing.assert_array_equal(
        image_payload_data(result),
        np.stack(tuple(value + 1 for value in runtime_slices)),
    )


def test_aligned_stack_rejects_canonical_output_surface_count_mismatch() -> None:
    def dispatch_probe(image: np.ndarray) -> AlignedImageStack:
        return AlignedImageStack((np.asarray(image), np.asarray(image)))

    output_specs = tuple(
        ArtifactSpec.output(f"Aligned{index}", ImageArtifactType)
        for index in range(3)
    )
    callable_contract = _compiled_contract(
        dispatch_probe,
        ProcessingContract.PURE_2D,
        artifact_outputs=output_specs,
    )

    with pytest.raises(
        ValueError,
        match="dispatch_probe produced 2 aligned main-flow value.*3 declared output",
    ):
        CellProfilerFunctionContractExecutor().execute(
            callable_contract,
            dispatch_probe,
            AlignedImageStack((np.zeros((2, 3), dtype=np.float32),)),
            {},
            execution_mode=ImagePayloadExecutionMode.ALIGNED_MULTI_IMAGE_STACK,
            plane_projection=RuntimePlaneAxisValueProjection.preserve(
                axis=RuntimePlaneAxis.RUNTIME_SLICE,
                axis_size=1,
            ),
        )


def test_raw_callable_mismatch_fails_before_invocation() -> None:
    calls = 0

    def compiled_probe(image: np.ndarray) -> np.ndarray:
        return image

    def substituted_probe(image: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        return image

    callable_contract = _compiled_contract(
        compiled_probe,
        ProcessingContract.PURE_2D,
    )

    with pytest.raises(
        ValueError,
        match="DispatchProbeModule.*dispatch_probe",
    ):
        CellProfilerFunctionContractExecutor().execute(
            callable_contract,
            substituted_probe,
            np.zeros((2, 3), dtype=np.float32),
            {},
            execution_mode=ImagePayloadExecutionMode.NATURAL,
        )

    assert calls == 0


def test_full_stack_pure_3d_rejects_slice_aligned_kwargs_before_invocation() -> None:
    calls = 0

    def dispatch_probe(image: np.ndarray, *, labels: object) -> np.ndarray:
        nonlocal calls
        calls += 1
        del labels
        return image

    callable_contract = _compiled_contract(
        dispatch_probe,
        ProcessingContract.PURE_3D,
    )

    with pytest.raises(
        ValueError,
        match=(
            "DispatchProbeModule.*dispatch_probe.*PURE_3D.*"
            "runtime-slice-aligned kwargs.*labels"
        ),
    ):
        CellProfilerFunctionContractExecutor().execute(
            callable_contract,
            dispatch_probe,
            np.zeros((2, 3), dtype=np.float32),
            {"labels": RuntimeSliceAlignedValues(slices=(1, 2))},
            execution_mode=ImagePayloadExecutionMode.FULL_STACK,
        )

    assert calls == 0
