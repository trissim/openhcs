from dataclasses import dataclass

import numpy as np

from openhcs.core.aligned_image_payload import (
    AlignedImageStack,
    ImagePayloadExecutionMode,
    compose_aligned_image_payload,
    payload_slice_count,
    payload_slices_for_alignment,
)
from benchmark.cellprofiler_compat.module_contract import CellProfilerModuleContract
from benchmark.cellprofiler_compat.module_execution import (
    CellProfilerFunctionContractMetadata,
    CellProfilerFunctionContractExecutor,
    CellProfilerModuleExecutor,
    _coerce_invocation_kwargs,
    _measurement_image_for_labels,
    _measurement_labels,
    _measurement_labels_for_image,
    _measurement_table_rows,
    _object_only_reference_image,
)
from benchmark.cellprofiler_library.functions.colortogray import color_to_gray
from benchmark.cellprofiler_library.functions.filterobjects import (
    FilterMethod,
    FilterMode,
    filter_objects,
)
from benchmark.cellprofiler_library.functions.identifyprimaryobjects import (
    ExcessObjectHandling,
    FillHolesOption,
    UnclumpMethod,
    identify_primary_objects,
)
from openhcs.core.artifacts import ArtifactKind, ArtifactSpec
from openhcs.core.config import DtypeConfig
from openhcs.core.runtime_values import MeasurementTable
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract


@dataclass(frozen=True, slots=True)
class _FakeRuntimeImage:
    data: np.ndarray
    source_image_name: str | None = None


class _FakeCellProfilerRuntime:
    def __init__(self, images: dict[str, _FakeRuntimeImage]) -> None:
        self.images = images
        self.measurements: list[tuple[str, list[object], dict[str, object]]] = []
        self.objects: list[tuple[str, np.ndarray, dict[str, object]]] = []

    def require_resolvable_source_aliases(self, aliases: tuple[str, ...]) -> None:
        missing = tuple(alias for alias in aliases if alias not in self.images)
        if missing:
            raise AssertionError(f"Unexpected missing image aliases: {missing!r}")

    def resolve_source_image(self, alias: str, fallback_image: object) -> np.ndarray:
        del fallback_image
        return self.images[alias].data

    def get_image(self, name: str) -> _FakeRuntimeImage:
        return self.images[name]

    def add_measurements(
        self,
        name: str,
        rows: object,
        **kwargs: object,
    ) -> None:
        self.measurements.append((name, _measurement_table_rows(rows), kwargs))

    def add_objects(
        self,
        name: str,
        labels: object,
        **kwargs: object,
    ) -> None:
        self.objects.append((name, labels, kwargs))

    def add_image(
        self,
        name: str,
        data: object,
        **kwargs: object,
    ) -> None:
        del kwargs
        self.images[name] = _FakeRuntimeImage(data)


def test_coerce_invocation_kwargs_uses_function_enum_annotations() -> None:
    coerced = _coerce_invocation_kwargs(
        identify_primary_objects,
        {
            "unclump_method": "Shape",
            "fill_holes": "After both thresholding and declumping",
            "limit_erase": "Continue",
        },
    )

    assert coerced["unclump_method"] is UnclumpMethod.SHAPE
    assert coerced["fill_holes"] is FillHolesOption.AFTER_BOTH
    assert coerced["limit_erase"] is ExcessObjectHandling.CONTINUE


def test_cellprofiler_contract_executor_applies_pure_2d_after_input_resolution():
    calls = []

    def add_one(image: np.ndarray) -> np.ndarray:
        calls.append(image.shape)
        return image + 1

    add_one.__processing_contract__ = ProcessingContract.PURE_2D
    stack = np.zeros((2, 4, 5), dtype=np.uint16)

    result = CellProfilerFunctionContractExecutor().execute(add_one, stack, {})

    assert calls == [(4, 5), (4, 5)]
    assert result.shape == stack.shape
    np.testing.assert_array_equal(result, np.ones_like(stack))


def test_cellprofiler_contract_executor_stacks_color_slice_outputs():
    calls = []

    def colorize(image: np.ndarray) -> np.ndarray:
        calls.append(image.shape)
        return np.stack((image, image, image), axis=-1)

    colorize.__processing_contract__ = ProcessingContract.PURE_2D
    stack = np.zeros((2, 4, 5), dtype=np.float32)

    result = CellProfilerFunctionContractExecutor().execute(colorize, stack, {})

    assert calls == [(4, 5), (4, 5)]
    assert result.shape == (2, 4, 5, 3)


def test_color_to_gray_combines_openhcs_color_stack() -> None:
    image = np.zeros((2, 4, 5, 3), dtype=np.float32)
    image[..., 0] = 2.0
    image[..., 1] = 4.0
    image[..., 2] = 6.0

    result = color_to_gray(
        image,
        mode="combine",
        image_type="rgb",
        channel_indices=(0, 1, 2),
        contributions=(1.0, 1.0, 2.0),
        dtype_config=DtypeConfig(),
    )

    assert result.shape == (2, 4, 5)
    np.testing.assert_array_equal(result, np.full((2, 4, 5), 4.5, dtype=np.float32))


def test_color_to_gray_splits_openhcs_color_slice_by_selected_channels() -> None:
    image = np.zeros((4, 5, 3), dtype=np.float32)
    image[..., 0] = 1.0
    image[..., 1] = 2.0
    image[..., 2] = 3.0

    red, blue = color_to_gray(
        image,
        mode="split",
        image_type="rgb",
        channel_indices=(0, 2),
        dtype_config=DtypeConfig(),
    )

    assert red.shape == (4, 5)
    assert blue.shape == (4, 5)
    np.testing.assert_array_equal(red, np.ones((4, 5), dtype=np.float32))
    np.testing.assert_array_equal(blue, np.full((4, 5), 3.0, dtype=np.float32))


def test_aligned_payload_treats_hwc_color_as_one_slice() -> None:
    color_slice = np.zeros((4, 5, 3), dtype=np.float32)

    slices = payload_slices_for_alignment(color_slice)

    assert len(slices) == 1
    assert slices[0] is color_slice
    assert payload_slice_count(color_slice) == 1


def test_module_executor_rewraps_single_image_output_for_openhcs_main_flow() -> None:
    def to_gray(image: np.ndarray) -> np.ndarray:
        return image[..., 0]

    color_slice = np.zeros((4, 5, 3), dtype=np.float32)
    color_stack = color_slice[np.newaxis, ...]
    runtime = _FakeCellProfilerRuntime(
        {"OrigColor": _FakeRuntimeImage(color_slice, source_image_name="OrigColor")}
    )
    executor = CellProfilerModuleExecutor(
        CellProfilerModuleContract(
            module_name="ColorToGray",
            inputs=(ArtifactSpec("OrigColor", ArtifactKind.IMAGE),),
            outputs=(ArtifactSpec("OrigGray", ArtifactKind.IMAGE),),
        )
    )

    result = executor.run(to_gray, color_stack, cellprofiler_runtime=runtime)

    assert result.shape == (1, 4, 5)
    assert runtime.images["OrigGray"].data.shape == (4, 5)


def test_cellprofiler_contract_executor_slices_aligned_runtime_kwargs():
    calls = []

    def keep_labels(image: np.ndarray, *, labels: np.ndarray):
        calls.append((image.shape, labels.shape))
        return image, labels

    keep_labels.__processing_contract__ = ProcessingContract.PURE_2D
    stack = np.zeros((2, 4, 5), dtype=np.uint16)
    labels = np.ones_like(stack, dtype=np.int32)

    result_image, result_labels = CellProfilerFunctionContractExecutor().execute(
        keep_labels,
        stack,
        {"labels": labels},
    )

    assert calls == [((4, 5), (4, 5)), ((4, 5), (4, 5))]
    assert result_image.shape == stack.shape
    assert result_labels.shape == labels.shape


def test_cellprofiler_contract_executor_broadcasts_2d_image_to_stacked_kwargs():
    calls = []

    def increment_labels(image: np.ndarray, *, labels: np.ndarray):
        calls.append((image.shape, labels.shape))
        return labels + 1

    increment_labels.__processing_contract__ = ProcessingContract.PURE_2D
    image = np.zeros((4, 5), dtype=np.uint16)
    labels = np.stack(
        (
            np.ones((4, 5), dtype=np.int32),
            np.full((4, 5), 2, dtype=np.int32),
        )
    )

    result = CellProfilerFunctionContractExecutor().execute(
        increment_labels,
        image,
        {"labels": labels},
    )

    assert calls == [((4, 5), (4, 5)), ((4, 5), (4, 5))]
    assert result.shape == labels.shape
    np.testing.assert_array_equal(result, labels + 1)


def test_cellprofiler_contract_executor_preserves_multi_image_stack_payload():
    calls = []

    def keep_stack(image: np.ndarray) -> np.ndarray:
        calls.append(image.shape)
        return image

    keep_stack.__processing_contract__ = ProcessingContract.PURE_2D
    stack = np.zeros((3, 4, 5), dtype=np.uint16)

    result = CellProfilerFunctionContractExecutor().execute(
        keep_stack,
        stack,
        {},
        force_full_stack=True,
    )

    assert calls == [(3, 4, 5)]
    assert result.shape == stack.shape


def test_object_only_reference_image_reduces_color_stacks_to_one_intensity_plane():
    color_stack = np.zeros((2, 4, 5, 3), dtype=np.float32)
    color_stack[0, :, :, 1] = 7

    reference = _object_only_reference_image(color_stack)

    assert reference.shape == (4, 5)
    np.testing.assert_array_equal(reference, color_stack[0, :, :, 0])


def test_compose_image_payload_aligns_multislice_inputs_with_broadcast():
    raw_stack = np.stack(
        (
            np.full((4, 5), 11, dtype=np.float32),
            np.full((4, 5), 22, dtype=np.float32),
        )
    )
    illumination = np.full((4, 5), 3, dtype=np.float32)

    composition = compose_aligned_image_payload(
        "CorrectIlluminationApply",
        (raw_stack, illumination),
    )

    assert composition.execution_mode is ImagePayloadExecutionMode.ALIGNED_MULTI_IMAGE_STACK
    assert isinstance(composition.payload, AlignedImageStack)
    assert len(composition.payload.slices) == 2
    for slice_index, composed_slice in enumerate(composition.payload.slices):
        assert composed_slice.shape == (2, 4, 5)
        np.testing.assert_array_equal(composed_slice[0], raw_stack[slice_index])
        np.testing.assert_array_equal(composed_slice[1], illumination)


def test_cellprofiler_contract_executor_applies_aligned_multi_image_stack():
    calls = []

    def subtract_illumination(image: np.ndarray) -> np.ndarray:
        calls.append(image.shape)
        return (image[0] - image[1])[np.newaxis, ...]

    aligned_stack = AlignedImageStack(
        slices=(
            np.stack(
                (
                    np.full((4, 5), 11, dtype=np.float32),
                    np.full((4, 5), 3, dtype=np.float32),
                )
            ),
            np.stack(
                (
                    np.full((4, 5), 22, dtype=np.float32),
                    np.full((4, 5), 3, dtype=np.float32),
                )
            ),
        )
    )

    result = CellProfilerFunctionContractExecutor().execute(
        subtract_illumination,
        aligned_stack,
        {},
        execution_mode=ImagePayloadExecutionMode.ALIGNED_MULTI_IMAGE_STACK,
    )

    assert calls == [(2, 4, 5), (2, 4, 5)]
    assert result.shape == (2, 4, 5)
    np.testing.assert_array_equal(result[0], np.full((4, 5), 8, dtype=np.float32))
    np.testing.assert_array_equal(result[1], np.full((4, 5), 19, dtype=np.float32))


def test_aligned_multi_image_stack_slices_runtime_array_kwargs() -> None:
    calls = []

    def keep_labels(image: np.ndarray, *, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        calls.append((image.shape, labels.shape))
        return image[0], labels

    aligned_stack = AlignedImageStack(
        slices=(
            np.stack(
                (
                    np.full((4, 5), 11, dtype=np.float32),
                    np.full((4, 5), 3, dtype=np.float32),
                )
            ),
            np.stack(
                (
                    np.full((4, 5), 22, dtype=np.float32),
                    np.full((4, 5), 7, dtype=np.float32),
                )
            ),
        )
    )
    labels = np.stack(
        (
            np.full((4, 5), 1, dtype=np.int32),
            np.full((4, 5), 2, dtype=np.int32),
        )
    )

    result_image, result_labels = CellProfilerFunctionContractExecutor().execute(
        keep_labels,
        aligned_stack,
        {"labels": labels},
        execution_mode=ImagePayloadExecutionMode.ALIGNED_MULTI_IMAGE_STACK,
    )

    assert calls == [((2, 4, 5), (4, 5)), ((2, 4, 5), (4, 5))]
    assert result_image.shape == (2, 4, 5)
    assert result_labels.shape == labels.shape
    np.testing.assert_array_equal(result_labels, labels)


def test_module_executor_runs_image_measurements_per_declared_image() -> None:
    calls = []

    def measure_image(image: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        calls.append(float(image[0, 0]))
        return image, {"mean": float(np.mean(image))}

    measure_image.__processing_contract__ = ProcessingContract.PURE_2D
    fallback = np.zeros((4, 5), dtype=np.float32)
    runtime = _FakeCellProfilerRuntime(
        {
            "OrigBlue": _FakeRuntimeImage(
                np.ones((4, 5), dtype=np.float32),
                source_image_name="OrigBlue",
            ),
            "OrigGreen": _FakeRuntimeImage(
                np.full((4, 5), 2, dtype=np.float32),
                source_image_name="OrigGreen",
            ),
        }
    )
    executor = CellProfilerModuleExecutor(
        CellProfilerModuleContract(
            module_name="MeasureImageQuality",
            inputs=(
                ArtifactSpec("OrigBlue", ArtifactKind.IMAGE),
                ArtifactSpec("OrigGreen", ArtifactKind.IMAGE),
            ),
            outputs=(ArtifactSpec("ImageQuality", ArtifactKind.MEASUREMENTS),),
        )
    )

    result = executor.run(
        measure_image,
        fallback,
        cellprofiler_runtime=runtime,
    )

    assert result is fallback
    assert calls == [1.0, 2.0]
    assert runtime.measurements == [
        (
            "ImageQuality",
            [{"mean": 1.0}, {"mean": 2.0}],
            {"source_image_name": None},
        )
    ]


def test_module_executor_preserves_composed_image_measurements() -> None:
    calls = []

    def measure_pair(
        image: np.ndarray,
        channel_1: int = 0,
        channel_2: int = 1,
    ) -> tuple[np.ndarray, dict[str, float]]:
        calls.append(image.shape)
        return image[channel_1], {
            "delta": float(np.mean(image[channel_2] - image[channel_1]))
        }

    fallback = np.zeros((4, 5), dtype=np.float32)
    runtime = _FakeCellProfilerRuntime(
        {
            "OrigBlue": _FakeRuntimeImage(np.ones((4, 5), dtype=np.float32)),
            "OrigGreen": _FakeRuntimeImage(np.full((4, 5), 3, dtype=np.float32)),
        }
    )
    executor = CellProfilerModuleExecutor(
        CellProfilerModuleContract(
            module_name="MeasureColocalization",
            inputs=(
                ArtifactSpec("OrigBlue", ArtifactKind.IMAGE),
                ArtifactSpec("OrigGreen", ArtifactKind.IMAGE),
            ),
            outputs=(ArtifactSpec("Colocalization", ArtifactKind.MEASUREMENTS),),
        )
    )

    result = executor.run(
        measure_pair,
        fallback,
        cellprofiler_runtime=runtime,
    )

    assert result is fallback
    assert calls == [(2, 4, 5)]
    assert runtime.measurements == [
        (
            "Colocalization",
            [{"delta": 2.0}],
            {"object_name": None, "source_image_name": None},
        )
    ]


def test_module_executor_records_multiple_declared_object_outputs() -> None:
    labels_without_overlap = np.ones((4, 5), dtype=np.int32)
    labels_with_overlap = np.full((4, 5), 2, dtype=np.int32)

    def untangle_like(
        image: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, float], np.ndarray, np.ndarray]:
        return image, {"worm_count": 1.0}, labels_with_overlap, labels_without_overlap

    untangle_like.__processing_contract__ = ProcessingContract.PURE_2D
    fallback = np.zeros((4, 5), dtype=np.float32)
    runtime = _FakeCellProfilerRuntime(
        {
            "WormBinary": _FakeRuntimeImage(
                fallback,
                source_image_name="WormBinary",
            ),
        }
    )
    executor = CellProfilerModuleExecutor(
        CellProfilerModuleContract(
            module_name="UntangleWorms",
            inputs=(ArtifactSpec("WormBinary", ArtifactKind.IMAGE),),
            outputs=(
                ArtifactSpec("UntangleWorms_3_measurements", ArtifactKind.MEASUREMENTS),
                ArtifactSpec("OverlappingWorms", ArtifactKind.OBJECT_LABELS),
                ArtifactSpec("NonOverlappingWorms", ArtifactKind.OBJECT_LABELS),
            ),
        )
    )

    result = executor.run(
        untangle_like,
        fallback,
        cellprofiler_runtime=runtime,
    )

    assert result is fallback
    assert runtime.measurements == [
        (
            "UntangleWorms_3_measurements",
            [{"worm_count": 1.0}],
            {"object_name": None, "source_image_name": "WormBinary"},
        )
    ]
    assert [name for name, _labels, _kwargs in runtime.objects] == [
        "OverlappingWorms",
        "NonOverlappingWorms",
    ]
    np.testing.assert_array_equal(runtime.objects[0][1], labels_with_overlap)
    np.testing.assert_array_equal(runtime.objects[1][1], labels_without_overlap)


def test_cellprofiler_contract_executor_infers_unknown_absorbed_contract():
    def two_dimensional_only(image: np.ndarray, **kwargs) -> np.ndarray:
        if image.ndim != 2:
            raise RuntimeError("2D only")
        return image

    two_dimensional_only.__cellprofiler_declared_contract__ = "unknown"

    assert (
        CellProfilerFunctionContractMetadata.from_callable(
            two_dimensional_only
        ).resolve(two_dimensional_only)
        is ProcessingContract.PURE_2D
    )


def test_measurement_image_for_labels_reduces_stack_to_reference_slice() -> None:
    image = np.arange(2 * 4 * 5, dtype=np.uint16).reshape(2, 4, 5)
    labels = np.ones((4, 5), dtype=np.int32)

    measurement_image = _measurement_image_for_labels(image, labels)

    assert measurement_image.shape == labels.shape
    np.testing.assert_array_equal(measurement_image, image[0])


def test_measurement_labels_collapse_singleton_label_stack() -> None:
    labels = np.ones((1, 4, 5), dtype=np.int32)

    measurement_labels = _measurement_labels(labels)

    assert measurement_labels.shape == (4, 5)
    np.testing.assert_array_equal(measurement_labels, labels[0])


def test_measurement_labels_align_to_single_channel_image_stack() -> None:
    image = np.ones((1, 4, 5), dtype=np.float32)
    labels = np.arange(2 * 4 * 5, dtype=np.int32).reshape(2, 4, 5)

    measurement_labels = _measurement_labels_for_image(image, labels)

    assert measurement_labels.shape == (4, 5)
    np.testing.assert_array_equal(measurement_labels, labels[0])


def test_object_only_reference_image_uses_one_stack_plane() -> None:
    image = np.arange(3 * 4 * 5, dtype=np.uint16).reshape(3, 4, 5)

    reference_image = _object_only_reference_image(image)

    assert reference_image.shape == (4, 5)
    np.testing.assert_array_equal(reference_image, image[0])


def test_measurement_table_rows_wrap_scalar_measurement() -> None:
    row = {"mean_intensity": 1.5}

    measurement_rows = _measurement_table_rows(row)

    assert measurement_rows == [row]


def test_filterobjects_relabels_additional_object_inputs_by_primary_retention() -> None:
    image = np.zeros((6, 6), dtype=np.float32)
    primary = np.zeros((6, 6), dtype=np.int32)
    primary[0:2, 0:2] = 1
    primary[2:5, 2:5] = 2
    cells = np.zeros_like(primary)
    cells[0:2, 0:2] = 10
    cells[2:5, 2:5] = 11

    result = filter_objects(
        image,
        mode=FilterMode.BORDER,
        object_labels=(primary, cells),
        additional_object_count=1,
        outline_object_indices=(0, 1),
        dtype_config=DtypeConfig(),
    )

    (
        _output_image,
        stats,
        filtered_primary,
        filtered_cells,
        primary_outline,
        cells_outline,
    ) = result

    assert stats.objects_pre_filter == 2
    assert stats.objects_post_filter == 1
    assert filtered_primary.max() == 1
    assert filtered_primary[3, 3] == 1
    assert filtered_cells.max() == 1
    assert filtered_cells[3, 3] == 1
    assert filtered_cells[0, 0] == 0
    assert primary_outline.max() == 1
    assert cells_outline.max() == 1


def test_filterobjects_uses_named_measurement_feature_rules() -> None:
    image = np.zeros((5, 5), dtype=np.float32)
    primary = np.zeros((5, 5), dtype=np.int32)
    primary[1:3, 1:3] = 1
    primary[3:5, 3:5] = 2
    measurement_rows = [
        {"object_label": 1, "lower_quartile_intensity": 0.1},
        {"object_label": 2, "lower_quartile_intensity": 0.8},
    ]

    result = filter_objects(
        image,
        mode=FilterMode.MEASUREMENTS,
        filter_method=FilterMethod.LIMITS,
        object_labels=(primary,),
        measurement_features=("Intensity_LowerQuartileIntensity_DNA",),
        measurement_min_values=(0.2,),
        measurement_max_values=(None,),
        measurement_use_minimum=(True,),
        measurement_use_maximum=(False,),
        measurement_tables=(
            MeasurementTable(name="NucleiMeasurements", rows=measurement_rows),
        ),
        dtype_config=DtypeConfig(),
    )

    _output_image, stats, filtered_primary = result

    assert stats.objects_pre_filter == 2
    assert stats.objects_post_filter == 1
    assert filtered_primary[1, 1] == 0
    assert filtered_primary[3, 3] == 1
