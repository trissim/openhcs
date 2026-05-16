from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
import skimage.measure
import skimage.morphology

from openhcs.core.aligned_image_payload import (
    AlignedImageStack,
    ImagePayloadSliceProjector,
    ImagePayloadExecutionMode,
    compose_aligned_image_payload,
    compose_one_image_bundle,
    aligned_image_stack_kwargs,
    payload_slice_count,
    payload_slices_for_alignment,
)
from openhcs.core.runtime_slice_projection import RuntimeSliceProjection
from openhcs.core.runtime_artifact_queries import (
    columnar_row_values,
    measurement_row_mapping,
    measurement_table_for_slice,
    measurement_values_for_label_slices,
)
from openhcs.core.runtime_semantics import MeasurementRowAxisField
from openhcs.constants.constants import MemoryType
from openhcs.interop.cellprofiler.runtime.invocation import (
    CellProfilerMeasurementImage,
    CellProfilerSliceAlignedValues,
)
from openhcs.interop.cellprofiler.runtime import (
    CellProfilerGridCycleScope,
    CellProfilerInvocationOptions,
)
from openhcs.interop.cellprofiler.runtime.module_execution import (
    AlignMeasurementFeature,
    CalculateMathInputPolicy,
    CellProfilerFunctionContractExecutor,
    CellProfilerInvocationExecutionModePolicy,
    CellProfilerMeasurementFieldSchema,
    CellProfilerMeasurementOutputProjection,
    CellProfilerProjectedMeasurementRow,
    CellProfilerMeasurementRecordBuilder,
    ClassifyObjectsMeasurementFeatureTemplate,
    CombineObjectsInputPolicy,
    MeasurementLabelSourceAlignmentStrategy,
    CellProfilerMeasurementImageDomain,
    CellProfilerModuleExecutor,
    CellProfilerObjectMeasurementExecutionDomainPolicy,
    CellProfilerObjectMeasurementRowPolicy,
    CompactMeasuredObjectMeasurementRowPolicy,
    DefaultObjectMeasurementRowPolicy,
    ObjectInputBindingRequest,
    ObjectLocationMeasurementRows,
    RelationshipMeasurementRows,
    RelationshipMeasurementFeatureTemplate,
    CellProfilerOutputValueResolution,
    CellProfilerOutputRecordRequest,
    CellProfilerOutputRecorder,
    CellProfilerPure2DOutputAggregator,
    CallableInvocationKwargSpec,
    CellProfilerGlobalImageNumberProjection,
    _measurement_image_for_labels,
    _measurement_labels,
    _measurement_labels_for_measurement_image,
    _measurement_table_rows,
    _object_only_reference_image,
    _output_values_by_kind,
    _processing_contract_for_callable,
    _unstack_cellprofiler_image_slices,
)
from openhcs.processing.backends.cellprofiler.structuring_elements import StructuringElement
from benchmark.cellprofiler_library.functions.colortogray import color_to_gray
from benchmark.cellprofiler_library.functions.crop import crop
from benchmark.cellprofiler_library.functions.correctilluminationapply import (
    correct_illumination_apply,
)
from benchmark.cellprofiler_library.functions.align import AlignShiftMeasurement
from benchmark.cellprofiler_library.functions.filterobjects import (
    FilterMethod,
    FilterMode,
    PerObjectAssignment,
    filter_objects,
)
from benchmark.cellprofiler_library.functions.enhanceorsuppressfeatures import (
    EnhanceMethod,
    NeuriteMethod,
    SpeckleAccuracy,
    enhance_or_suppress_features,
)
from benchmark.cellprofiler_library.functions.expandorshrinkobjects import (
    expand_or_shrink_objects,
)
from benchmark.cellprofiler_library.functions.classifyobjects import (
    ClassificationResult,
)
from benchmark.cellprofiler_library.functions.threshold import ThresholdResult
from benchmark.cellprofiler_library.functions.relateobjects import (
    RelateObjectsResult,
    RelationshipMeasurements,
)
from benchmark.cellprofiler_library.functions.identifyprimaryobjects import (
    ExcessObjectHandling,
    FillHolesOption,
    UnclumpMethod,
    identify_primary_objects,
)
from benchmark.cellprofiler_library.functions.definegrid import define_grid_automatic
from benchmark.cellprofiler_library.functions.identifyobjectsingrid import (
    identify_objects_in_grid,
    identify_objects_in_grid_with_guides,
)
from openhcs.processing.backends.cellprofiler.grid import (
    GridDefinition,
    GridShapeRequest,
    NaturalGridShapeStrategy,
)
from benchmark.cellprofiler_library.functions.measureobjectsizeshape import (
    measure_object_size_shape,
)
from benchmark.cellprofiler_semantics.crop import CropShape, RemovalMethod
from benchmark.cellprofiler_library.functions.maskobjects import mask_objects
from benchmark.cellprofiler_library.functions import identifysecondaryobjects as iso
from benchmark.cellprofiler_library.functions import identifytertiaryobjects as ito
from benchmark.cellprofiler_library.functions.identifysecondaryobjects import (
    DistanceMaskedSegmentationStrategy,
    PropagationSegmentationStrategy,
    SecondarySegmentationRequest,
)
from openhcs.processing.backends.cellprofiler.secondary import (
    _filter_labels,
    _secondary_seed_labels,
)
from benchmark.cellprofiler_library.functions.tile import tile
from benchmark.cellprofiler_library.functions.watershed import watershed
from openhcs.core.artifacts import ArtifactKind, ArtifactSidecarRole, ArtifactSpec
from openhcs.core.callable_contract import attach_callable_contract_metadata
from openhcs.core.config import DtypeConfig
from openhcs.core.module_artifact_contract import ModuleArtifactContract
from openhcs.core.pipeline.function_contracts import special_inputs
from openhcs.processing.backends.lib_registry.unified_registry import Pure2DSliceResultBatch
from openhcs.core.runtime_invocation import RuntimeSliceAlignedValues
from openhcs.core.runtime_semantics import (
    MeasurementObjectRowIdentity,
    ObjectLocationMeasurementFeature,
    ObjectLabelDomainScope,
    ObjectLabelRepresentation,
    ParentChildRelationshipPayload,
    RelationshipSemantics,
    RuntimePlaneAxis,
    RuntimePlaneAxisProjector,
    SpatialGridOrdering,
    object_label_parent_child_payload,
    object_shape_measurement_all_field_names,
    object_shape_measurement_field_names,
)
from openhcs.core.runtime_values import (
    ColumnarRows,
    FieldSpec,
    ImagePayloadMetadata,
    ImageMetadataPayload,
    MaskedImagePayload,
    MeasurementTable,
    ObjectLabelPayload,
    ObjectLabelSet,
    ObjectRelationship,
    SparseIJVLabelRows,
    SpatialGrid,
    image_payload_data,
    image_payload_mask,
    image_payload_metadata,
    image_payload_with_context,
    object_label_dense_array,
)
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.processing.backends.cellprofiler.relationships import (
    ObjectRelationshipBackendStrategy,
)
from openhcs.processing.materialization import csv_materializer


@dataclass(frozen=True, slots=True)
class _FakeRuntimeImage:
    data: np.ndarray
    source_image_name: str | None = None


@dataclass(frozen=True, slots=True)
class _SyntheticObjectMeasurement:
    object_label: int
    value: float


def test_projected_measurement_rows_support_mapping_and_attribute_access() -> None:
    fields, rows = CellProfilerMeasurementOutputProjection(
        fields=(FieldSpec("AreaShape_Area"),),
        rows=({"AreaShape_Area": 7.0, "ObjectName": "Worms"},),
    ).apply()

    (row,) = rows

    assert isinstance(row, CellProfilerProjectedMeasurementRow)
    assert tuple(field.name for field in fields) == ("area_shape_area",)
    assert row["area_shape_area"] == 7.0
    assert row.area_shape_area == 7.0
    assert row.get("object_name") == "Worms"


def test_source_qualified_image_rows_use_current_image_number_not_slice_index() -> None:
    adapter = SimpleNamespace(
        cellprofiler_axis_image_number_start=lambda: 7,
        cellprofiler_image_number_for_source_paths=lambda _paths: None,
    )
    source_payload = image_payload_with_context(
        np.zeros((2, 2), dtype=np.float32),
        metadata=ImagePayloadMetadata(source_path="/plate/A01_s007_w1.tif"),
    )
    rows, _mappings = CellProfilerGlobalImageNumberProjection(
        adapter=adapter,
        rows=(
            {
                "slice_index": 0,
                "source_image_name": "Objects1",
                "area_occupied": 10,
            },
            {
                "slice_index": 1,
                "source_image_name": "Objects2",
                "area_occupied": 20,
            },
        ),
        source_image_name=None,
        source_image_payload=source_payload,
        object_name=None,
        need_row_mappings=True,
    ).apply()

    assert [row["image_number"] for row in rows] == [7, 7]


@dataclass(frozen=True, slots=True)
class _SyntheticAxisObjectMeasurement:
    image_number: int
    object_label: int
    value: float


@dataclass(frozen=True, slots=True)
class _ColumnarMeasurementRows(ColumnarRows):
    columns: dict[str, tuple[object, ...]]


def _synthetic_object_measurement_function(
    image: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, list[_SyntheticObjectMeasurement]]:
    return image, []


def _synthetic_axis_object_measurement_function(
    image: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, list[_SyntheticAxisObjectMeasurement]]:
    return image, []


def complete_object_measurement_rows(
    rows,
    *,
    label_payload,
    func,
    object_identity=MeasurementObjectRowIdentity.LABEL_ID,
    row_policy=None,
):
    if row_policy is None:
        row_policy = (
            CompactMeasuredObjectMeasurementRowPolicy()
            if object_identity is MeasurementObjectRowIdentity.ROW_ORDINAL
            else DefaultObjectMeasurementRowPolicy()
        )
    return row_policy.complete_rows(rows, label_payload=label_payload, func=func)


def _recorded_measurements_for_assertion(measurements):
    normalized = []
    for name, rows, kwargs in measurements:
        comparable_kwargs = dict(kwargs)
        if "fields" in comparable_kwargs:
            field_names = tuple(
                field.name for field in comparable_kwargs["fields"]
            )
            if field_names and "slice_index" not in field_names:
                field_names = ("slice_index", *field_names)
            comparable_kwargs["fields"] = field_names
        normalized_rows = []
        for row in rows:
            normalized_row = dict(row)
            normalized_row.setdefault("slice_index", 0)
            normalized_rows.append(normalized_row)
        normalized.append((name, normalized_rows, comparable_kwargs))
    return normalized


class _FakeCellProfilerRuntime:
    def __init__(
        self,
        images: dict[str, _FakeRuntimeImage],
        objects: dict[str, ObjectLabelSet] | None = None,
        measurement_tables: dict[str, tuple[MeasurementTable, ...]] | None = None,
        image_number_start: int = 1,
        ordered_pipeline_image_paths: tuple[str, ...] = (),
    ) -> None:
        self.images = images
        self.runtime_objects = objects or {}
        self.runtime_measurement_tables = measurement_tables or {}
        self.image_number_start = image_number_start
        self.ordered_pipeline_image_paths = ordered_pipeline_image_paths
        self.measurements: list[tuple[str, list[object], dict[str, object]]] = []
        self.objects: list[tuple[str, np.ndarray, dict[str, object]]] = []
        self.spatial_grids: dict[str, SpatialGrid] = {}
        self.relationships: list[tuple[str, dict[str, object]]] = []

    def require_resolvable_source_aliases(self, aliases: tuple[str, ...]) -> None:
        missing = tuple(alias for alias in aliases if alias not in self.images)
        if missing:
            raise AssertionError(f"Unexpected missing image aliases: {missing!r}")

    def cellprofiler_ordered_pipeline_image_paths(self) -> tuple[str, ...]:
        return self.ordered_pipeline_image_paths

    def cellprofiler_source_order_path(self, path: str) -> str:
        return path

    def cellprofiler_image_number_for_source_paths(
        self,
        source_paths: tuple[str, ...],
    ) -> int | None:
        if not source_paths:
            return None
        first_source_path = self.cellprofiler_source_order_path(source_paths[0])
        try:
            return self.ordered_pipeline_image_paths.index(first_source_path) + 1
        except ValueError:
            return None

    def cellprofiler_axis_image_number_start(self) -> int:
        return self.image_number_start

    def resolve_source_image(self, alias: str, current_image: object) -> np.ndarray:
        del current_image
        return self.images[alias].data

    def get_image(self, name: str) -> _FakeRuntimeImage:
        return self.images[name]

    def add_image(
        self,
        name: str,
        data: object,
        **kwargs: object,
    ) -> None:
        self.images[name] = _FakeRuntimeImage(
            data,
            source_image_name=(
                str(kwargs["source_image_name"])
                if kwargs.get("source_image_name") is not None
                else None
            ),
        )

    def get_objects(
        self,
        name: str,
        *,
        current_image: object | None = None,
    ) -> ObjectLabelSet:
        del current_image
        return self.runtime_objects[name]

    def measurement_tables_for_object(self, name: str) -> tuple[object, ...]:
        return self.runtime_measurement_tables.get(name, ())

    def measurement_tables_for_object_feature(
        self,
        object_name: str,
        feature_name: str,
        *,
        match_group: bool = True,
    ) -> tuple[MeasurementTable, ...]:
        del match_group
        return tuple(
            table
            for table in self.runtime_measurement_tables.get(object_name, ())
            if any(feature_name in measurement_row_mapping(row) for row in table.rows)
        )

    def measurement_values_for_label_slices(
        self,
        object_name: str,
        feature_name: str,
        labels: object,
        *,
        group_key: str | None = None,
        image_number: int | None = None,
    ) -> tuple[object, ...]:
        del group_key, image_number
        return measurement_values_for_label_slices(
            self.runtime_measurement_tables.get(object_name, ()),
            feature_name,
            labels,
            object_name=object_name,
        )

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

    def add_spatial_grid(
        self,
        name: str,
        grid: SpatialGrid | RuntimeSliceAlignedValues,
    ) -> None:
        if isinstance(grid, RuntimeSliceAlignedValues):
            self.spatial_grids[name] = RuntimeSliceAlignedValues(
                slices=tuple(value.with_name(name) for value in grid.slices)
            )
            return
        self.spatial_grids[name] = grid.with_name(name)

    def get_spatial_grid(self, name: str) -> SpatialGrid | RuntimeSliceAlignedValues:
        return self.spatial_grids[name]

    def add_relationship(self, name: str, **kwargs: object) -> None:
        self.relationships.append((name, kwargs))


class _CalculateMathObjectOperandAdapter:
    def __init__(self, labels: np.ndarray) -> None:
        self.labels = labels
        self.feature_requests: list[tuple[str, str, object]] = []

    def get_objects(
        self,
        name: str,
        *,
        current_image: object | None = None,
    ) -> ObjectLabelSet:
        del current_image
        return ObjectLabelSet(name=name, labels=self.labels)

    def resolve_source_objects(
        self,
        name: str,
        current_image: object,
    ) -> ObjectLabelSet:
        del current_image
        return self.get_objects(name)

    def measurement_values_for_label_slices(
        self,
        object_name: str,
        feature_name: str,
        labels: object,
        *,
        group_key: str | None = None,
        image_number: int | None = None,
    ) -> tuple[np.ndarray, ...]:
        del group_key, image_number
        self.feature_requests.append((object_name, feature_name, labels))
        return (
            np.asarray([1.0, 2.0], dtype=float),
            np.asarray([3.0, 4.0], dtype=float),
        )

    def measurement_values_for_object_feature(
        self,
        object_name: str,
        feature_name: str,
        *,
        group_key: str | None = None,
    ) -> np.ndarray:
        raise AssertionError(
            "CalculateMath object operands must use the label-slice domain "
            f"for {object_name}:{feature_name}, not the collapsed object feature path."
        )


def test_calculate_math_object_operands_preserve_label_slice_domain() -> None:
    labels = np.asarray(
        [
            [[0, 1], [2, 0]],
            [[0, 1], [0, 2]],
        ],
        dtype=np.int32,
    )
    adapter = _CalculateMathObjectOperandAdapter(labels)
    request = ObjectInputBindingRequest(
        module_name="CalculateMath",
        object_inputs=(ArtifactSpec("Nuclei", ArtifactKind.OBJECT_LABELS),),
        adapter=adapter,
        kwargs={
            "operand1_feature": "Intensity_MeanIntensity_DNA",
            "operand1_object_name": "Nuclei",
        },
        current_image=np.zeros((2, 2), dtype=np.float32),
        external_object_names=frozenset(),
    )

    value = CalculateMathInputPolicy().operand_value(
        request,
        feature_kwarg="operand1_feature",
        object_kwarg="operand1_object_name",
    )

    assert isinstance(value, CellProfilerSliceAlignedValues)
    assert value.slice_count == 2
    np.testing.assert_array_equal(value.value_for_slice(0), [1.0, 2.0])
    np.testing.assert_array_equal(value.value_for_slice(1), [3.0, 4.0])
    assert adapter.feature_requests == [
        ("Nuclei", "Intensity_MeanIntensity_DNA", labels)
    ]


class _CombineObjectsAdapter:
    def __init__(self, payloads):
        self.payloads = payloads

    def get_objects(self, name, *, current_image):
        del current_image
        return self.payloads[name]


def test_combine_objects_broadcasts_2d_labels_to_runtime_slice_domain() -> None:
    stacked_labels = np.asarray(
        [
            [[0, 1], [0, 0]],
            [[0, 0], [2, 0]],
        ],
        dtype=np.int32,
    )
    plane_labels = np.asarray([[0, 3], [0, 0]], dtype=np.int32)
    adapter = _CombineObjectsAdapter(
        {
            "Primary": ObjectLabelSet(
                name="Primary",
                labels=ObjectLabelPayload(labels=stacked_labels),
            ),
            "Secondary": ObjectLabelSet(
                name="Secondary",
                labels=ObjectLabelPayload(labels=plane_labels),
            ),
        }
    )
    request = ObjectInputBindingRequest(
        module_name="CombineObjects",
        object_inputs=(
            ArtifactSpec("Primary", ArtifactKind.OBJECT_LABELS),
            ArtifactSpec("Secondary", ArtifactKind.OBJECT_LABELS),
        ),
        adapter=adapter,
        kwargs={},
        current_image=np.zeros((2, 2), dtype=np.float32),
        external_object_names=frozenset(),
    )

    primary, secondary = CombineObjectsInputPolicy().label_pair_payload(request)

    assert primary.shape == (2, 2, 2)
    assert secondary.shape == (2, 2, 2)
    np.testing.assert_array_equal(primary, stacked_labels)
    np.testing.assert_array_equal(secondary[0], plane_labels)
    np.testing.assert_array_equal(secondary[1], plane_labels)


def test_special_inputs_bind_from_declared_role_order_not_runtime_dedup_order() -> None:
    memb_final = np.asarray([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    cell_seeds = np.asarray([[0, 3], [0, 0]], dtype=np.int32)
    intensity_metadata = ImagePayloadMetadata(
        intensity_scale=255,
        source_dtype="uint8",
    )
    adapter = _FakeCellProfilerRuntime(
        {
            "MembFinal": _FakeRuntimeImage(memb_final),
            "cellSeeds": _FakeRuntimeImage(
                ImageMetadataPayload(cell_seeds, intensity_metadata),
            ),
        }
    )
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="Watershed",
            inputs=(
                ArtifactSpec("MembFinal", ArtifactKind.IMAGE),
                ArtifactSpec("cellSeeds", ArtifactKind.IMAGE),
                ArtifactSpec("MembFinal", ArtifactKind.IMAGE),
            ),
            runtime_artifact_inputs=(
                ArtifactSpec("MembFinal", ArtifactKind.IMAGE),
                ArtifactSpec("cellSeeds", ArtifactKind.IMAGE),
            ),
            outputs=(ArtifactSpec("Cells", ArtifactKind.OBJECT_LABELS),),
        )
    )

    kwargs = executor._runtime_input_kwargs(
        watershed,
        adapter,
        memb_final,
        {"watershed_method": "markers"},
    )

    np.testing.assert_array_equal(kwargs["markers"], cell_seeds)
    assert kwargs["markers"].dtype == np.int32
    np.testing.assert_array_equal(kwargs["mask"], memb_final)

    def add_image(
        self,
        name: str,
        data: object,
        **kwargs: object,
    ) -> None:
        del kwargs
        self.images[name] = _FakeRuntimeImage(data)


def test_coerce_invocation_kwargs_uses_function_enum_annotations() -> None:
    coerced = CallableInvocationKwargSpec.from_callable(
        identify_primary_objects
    ).coerce_kwargs(
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


def test_cellprofiler_contract_executor_flattens_volume_stacks_for_pure_2d():
    calls = []

    def add_labels(image: np.ndarray, *, labels: np.ndarray) -> np.ndarray:
        calls.append((image.shape, labels.shape, int(labels[0, 0])))
        return image + labels

    add_labels.__processing_contract__ = ProcessingContract.PURE_2D
    image = np.zeros((2, 3, 4, 5), dtype=np.float32)
    labels = np.arange(6, dtype=np.int32).reshape((2, 3, 1, 1))
    labels = np.broadcast_to(labels, image.shape)

    result = CellProfilerFunctionContractExecutor().execute(
        add_labels,
        image,
        {"labels": labels},
    )

    assert calls == [((4, 5), (4, 5), index) for index in range(6)]
    assert result.shape == (6, 4, 5)
    np.testing.assert_array_equal(result, labels.reshape((6, 4, 5)))


def test_cellprofiler_contract_executor_slices_high_rank_labels_by_runtime_axis():
    calls = []

    def keep_labels(image: np.ndarray, *, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        calls.append((image.shape, labels.shape, int(labels[0, 0, 0])))
        return image, labels

    keep_labels.__processing_contract__ = ProcessingContract.PURE_2D
    image = np.zeros((4, 5), dtype=np.float32)
    labels = np.arange(3, dtype=np.int32).reshape((3, 1, 1, 1))
    labels = np.broadcast_to(labels, (3, 2, 4, 5))

    result_image, result_labels = CellProfilerFunctionContractExecutor().execute(
        keep_labels,
        image,
        {"labels": labels},
    )

    assert calls == [((4, 5), (2, 4, 5), index) for index in range(3)]
    assert result_image.shape == (3, 4, 5)
    assert result_labels.shape == (3, 2, 4, 5)
    np.testing.assert_array_equal(result_labels, labels)


def test_runtime_slice_projection_counts_high_rank_kwargs_by_first_axis_for_2d_image():
    labels = np.zeros((3, 2, 4, 5), dtype=np.int32)

    assert RuntimeSliceProjection.first_axis_slice_count_from_values((labels,)) == 3


def test_cellprofiler_contract_executor_projects_flat_grouped_label_kwargs():
    calls = []

    def keep_labels(image: np.ndarray, *, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        calls.append((image.shape, labels.shape, int(labels[0, 0])))
        return image, labels

    keep_labels.__processing_contract__ = ProcessingContract.PURE_2D
    image = np.zeros((3, 4, 5), dtype=np.float32)
    labels = np.zeros((2, 3, 4, 5), dtype=np.int32)
    labels[0, :, 1:3, 1:3] = 1
    labels[1, :, 2:4, 2:4] = 2
    flattened_labels = labels.reshape((-1, *labels.shape[-2:]))

    result_image, result_labels = CellProfilerFunctionContractExecutor().execute(
        keep_labels,
        image,
        {"labels": flattened_labels},
    )

    expected_labels = np.max(labels, axis=0)
    assert calls == [((4, 5), (4, 5), 0) for _ in range(3)]
    assert result_image.shape == image.shape
    np.testing.assert_array_equal(result_labels, expected_labels)


def test_cellprofiler_contract_executor_stacks_singleton_plane_outputs():
    def add_singleton_plane(image: np.ndarray) -> np.ndarray:
        return image[np.newaxis, ...] + 1

    add_singleton_plane.__processing_contract__ = ProcessingContract.PURE_2D
    stack = np.zeros((2, 4, 5), dtype=np.uint16)

    result = CellProfilerFunctionContractExecutor().execute(
        add_singleton_plane,
        stack,
        {},
    )

    assert result.shape == stack.shape
    np.testing.assert_array_equal(result, np.ones_like(stack))


def test_cellprofiler_contract_executor_stacks_singleton_color_outputs():
    def add_singleton_color_plane(image: np.ndarray) -> np.ndarray:
        rgb = np.repeat(image[..., np.newaxis], 3, axis=-1)
        return rgb[np.newaxis, ...] + 1

    add_singleton_color_plane.__processing_contract__ = ProcessingContract.PURE_2D
    stack = np.zeros((2, 4, 5), dtype=np.uint16)

    result = CellProfilerFunctionContractExecutor().execute(
        add_singleton_color_plane,
        stack,
        {},
    )

    assert result.shape == (2, 4, 5, 3)
    np.testing.assert_array_equal(result, np.ones((2, 4, 5, 3), dtype=np.uint16))


def test_complete_object_measurement_rows_uses_declared_label_domain() -> None:
    payload = ObjectLabelPayload(
        labels=np.zeros((4, 4), dtype=np.int32),
        declared_object_count=3,
    )

    rows = complete_object_measurement_rows(
        [],
        label_payload=payload,
        func=_synthetic_object_measurement_function,
    )

    assert [row["object_label"] for row in rows] == [1, 2, 3]
    assert all(np.isnan(row["value"]) for row in rows)


def test_complete_object_measurement_rows_handles_empty_rows_with_axis_fields() -> None:
    payload = ObjectLabelPayload(
        labels=np.zeros((4, 4), dtype=np.int32),
        declared_object_count=2,
    )

    rows = complete_object_measurement_rows(
        [],
        label_payload=payload,
        func=_synthetic_axis_object_measurement_function,
    )

    assert [row["object_label"] for row in rows] == [1, 2]
    assert all(np.isnan(row["image_number"]) for row in rows)
    assert all(np.isnan(row["value"]) for row in rows)


def test_complete_object_measurement_rows_preserves_sliced_object_label_set_domain() -> None:
    labels = np.zeros((2, 4, 4), dtype=np.int32)
    labels[0, 1, 1] = 1
    labels[1, 2, 2] = 1
    payload = ObjectLabelSet(
        name="GridObjects",
        labels=labels,
        declared_object_count=3,
        source_image_name="BF_image",
    )

    rows = complete_object_measurement_rows(
        [
            {"slice_index": 0, "object_label": 1, "value": 10.0},
            {"slice_index": 1, "object_label": 1, "value": 20.0},
        ],
        label_payload=payload,
        func=_synthetic_object_measurement_function,
    )

    by_key = {
        (row["slice_index"], row["object_label"]): row["value"]
        for row in rows
    }
    assert tuple(by_key) == (
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 1),
        (1, 2),
        (1, 3),
    )
    assert by_key[(0, 1)] == 10.0
    assert by_key[(1, 1)] == 20.0
    assert np.isnan(by_key[(0, 2)])
    assert np.isnan(by_key[(0, 3)])
    assert np.isnan(by_key[(1, 2)])
    assert np.isnan(by_key[(1, 3)])


def test_global_image_number_projection_ignores_missing_axis_values() -> None:
    rows = [
        {"slice_index": 0, "object_label": 1, "value": 1.0},
        {"slice_index": np.nan, "object_label": 2, "value": np.nan},
    ]

    projected, projected_mappings = CellProfilerGlobalImageNumberProjection(
        adapter=_FakeCellProfilerRuntime({}),
        rows=rows,
        source_image_name=None,
        object_name=None,
        source_image_payload=None,
        need_row_mappings=True,
    ).apply()

    assert projected is projected_mappings
    assert projected[0]["image_number"] == 1
    assert "image_number" not in projected[1]


def test_global_image_number_projection_applies_to_columnar_rows() -> None:
    rows = _ColumnarMeasurementRows(
        {
            MeasurementRowAxisField.IMAGE_NUMBER.value: (1, 2, np.nan),
            "object_label": (1, 2, 3),
            "value": (10.0, 20.0, 30.0),
        }
    )

    projected, projected_mappings = CellProfilerGlobalImageNumberProjection(
        adapter=_FakeCellProfilerRuntime(
            {},
            image_number_start=23,
            ordered_pipeline_image_paths=("well-a",),
        ),
        rows=rows,
        source_image_name=None,
        object_name=None,
        source_image_payload=None,
        need_row_mappings=True,
    ).apply()

    assert projected is projected_mappings
    assert tuple(
        columnar_row_values(projected, MeasurementRowAxisField.IMAGE_NUMBER.value)
    )[:2] == (23, 24)
    assert np.isnan(
        columnar_row_values(projected, MeasurementRowAxisField.IMAGE_NUMBER.value)[2]
    )


def test_global_image_number_projection_uses_source_payload_for_columnar_rows() -> None:
    rows = _ColumnarMeasurementRows(
        {
            MeasurementRowAxisField.IMAGE_NUMBER.value: (1,),
            "object_label": (1,),
            "value": (10.0,),
        }
    )
    source_payload = image_payload_with_context(
        np.zeros((1, 1), dtype=np.float32),
        metadata=ImagePayloadMetadata(source_path="well-h12-w1.tif"),
    )

    projected, _projected_mappings = CellProfilerGlobalImageNumberProjection(
        adapter=_FakeCellProfilerRuntime(
            {},
            image_number_start=1,
            ordered_pipeline_image_paths=(
                "well-a01-w1.tif",
                "well-h12-w1.tif",
            ),
        ),
        rows=rows,
        source_image_name="rawGFP",
        object_name=None,
        source_image_payload=source_payload,
        need_row_mappings=True,
    ).apply()

    assert tuple(
        columnar_row_values(projected, MeasurementRowAxisField.IMAGE_NUMBER.value)
    ) == (2,)


def test_measure_object_intensity_zero_fills_missing_positive_extent() -> None:
    payload = ObjectLabelPayload(
        labels=np.asarray([[1, 0, 3]], dtype=np.int32),
        declared_object_count=5,
    )
    row_policy = CellProfilerObjectMeasurementRowPolicy.for_module(
        "MeasureObjectIntensity"
    )

    rows = complete_object_measurement_rows(
        [{"object_label": 1, "value": 7.0}],
        label_payload=payload,
        func=_synthetic_object_measurement_function,
        object_identity=row_policy.object_identity(),
        row_policy=row_policy,
    )
    by_label = {row["object_label"]: row for row in rows}

    assert by_label[2]["value"] == 0.0
    assert by_label[3]["value"] == 0.0
    assert np.isnan(by_label[4]["value"])
    assert np.isnan(by_label[5]["value"])


def test_complete_object_measurement_rows_uses_slice_local_label_domain() -> None:
    labels = np.zeros((2, 3, 5), dtype=np.int32)
    labels[0, 0, 0] = 1
    labels[0, 0, 2] = 3
    labels[1, 0, 0] = 1
    labels[1, 0, 1] = 2
    payload = ObjectLabelPayload(labels=labels)
    row_policy = CellProfilerObjectMeasurementRowPolicy.for_module(
        "MeasureObjectIntensity"
    )

    rows = complete_object_measurement_rows(
        [
            {"slice_index": 0, "object_label": 1, "value": 10.0},
            {"slice_index": 0, "object_label": 3, "value": 30.0},
            {"slice_index": 1, "object_label": 1, "value": 100.0},
            {"slice_index": 1, "object_label": 2, "value": 200.0},
        ],
        label_payload=payload,
        func=_synthetic_object_measurement_function,
        object_identity=row_policy.object_identity(),
        row_policy=row_policy,
    )

    values_by_key = {
        (row["slice_index"], row["object_label"]): row["value"]
        for row in rows
    }
    assert values_by_key == {
        (0, 1): 10.0,
        (0, 3): 30.0,
        (1, 1): 100.0,
        (1, 2): 200.0,
    }


def test_complete_object_measurement_rows_orders_sparse_label_domain() -> None:
    payload = ObjectLabelPayload(
        labels=np.asarray([[1, 0, 3]], dtype=np.int32),
        declared_object_count=5,
    )

    rows = complete_object_measurement_rows(
        [
            {"object_label": 3, "value": 30.0},
            {"object_label": 1, "value": 10.0},
        ],
        label_payload=payload,
        func=_synthetic_object_measurement_function,
    )

    assert [row["object_label"] for row in rows] == [1, 2, 3, 4, 5]
    assert rows[0]["value"] == 10.0
    assert np.isnan(rows[1]["value"])
    assert rows[2]["value"] == 30.0


def test_complete_object_measurement_rows_preserves_measurement_axes() -> None:
    payload = ObjectLabelPayload(
        labels=np.zeros((4, 4), dtype=np.int32),
        declared_object_count=2,
    )

    rows = complete_object_measurement_rows(
        [
            {
                "object_label": 1,
                "scale": 3,
                "direction": 0,
                "gray_levels": 256,
                "angular_second_moment": 0.25,
            },
            {
                "object_label": 1,
                "scale": 3,
                "direction": 1,
                "gray_levels": 256,
                "angular_second_moment": 0.5,
            },
        ],
        label_payload=payload,
        func=_synthetic_object_measurement_function,
    )

    assert {
        (row["object_label"], row["scale"], row["direction"], row["gray_levels"])
        for row in rows
    } == {
        (1, 3, 0, 256),
        (1, 3, 1, 256),
        (2, 3, 0, 256),
        (2, 3, 1, 256),
    }
    missing_rows = [row for row in rows if row["object_label"] == 2]
    assert all(np.isnan(row["angular_second_moment"]) for row in missing_rows)


def test_complete_object_measurement_rows_supports_compact_row_identity() -> None:
    payload = ObjectLabelPayload(
        labels=np.zeros((4, 4), dtype=np.int32),
        declared_object_ids=(10, 20, 30, 40, 50),
    )

    rows = complete_object_measurement_rows(
        [
            {"object_label": 10, "Area": 10.0},
            {"object_label": 20},
            {"object_label": 25, "Area": np.nan},
            {"object_label": 30, "Area": 30.0},
            {"object_label": 50, "Area": 50.0},
        ],
        label_payload=payload,
        func=_synthetic_object_measurement_function,
        object_identity=MeasurementObjectRowIdentity.ROW_ORDINAL,
    )

    assert [row["object_label"] for row in rows] == [1, 2, 3, 4, 5]
    assert [row["Area"] for row in rows[:3]] == [10.0, 30.0, 50.0]
    assert "Area" not in rows[3]
    assert np.isnan(rows[4]["Area"])


def test_measure_texture_compact_rows_preserve_declared_padding_domain() -> None:
    payload = ObjectLabelPayload(
        labels=np.zeros((4, 4), dtype=np.int32),
        declared_object_count=5,
    )
    row_policy = CellProfilerObjectMeasurementRowPolicy.for_module("MeasureTexture")

    rows = row_policy.complete_rows(
        [
            {"object_label": 10, "angular_second_moment": 0.1},
            {"object_label": 30, "angular_second_moment": 0.3},
            {"object_label": 50, "angular_second_moment": 0.5},
        ],
        label_payload=payload,
        func=_synthetic_object_measurement_function,
    )

    assert [row["object_label"] for row in rows] == [1, 2, 3, 4, 5]
    assert [row["angular_second_moment"] for row in rows[:3]] == [0.1, 0.3, 0.5]
    assert all(np.isnan(row["angular_second_moment"]) for row in rows[3:])


def test_measure_object_size_shape_compact_rows_preserve_emitted_padding() -> None:
    payload = ObjectLabelPayload(
        labels=np.zeros((4, 4), dtype=np.int32),
        declared_object_ids=(10, 20, 30, 40, 50),
    )
    row_policy = CellProfilerObjectMeasurementRowPolicy.for_module(
        "MeasureObjectSizeShape"
    )

    rows = complete_object_measurement_rows(
        [
            {"object_label": 10, "Area": 10.0},
            {"object_label": 20},
            {"object_label": 25, "Area": np.nan},
            {"object_label": 30, "Area": 30.0},
            {"object_label": 50, "Area": 50.0},
        ],
        label_payload=payload,
        func=_synthetic_object_measurement_function,
        object_identity=row_policy.object_identity(),
        row_policy=row_policy,
    )

    assert [row["object_label"] for row in rows] == [1, 2, 3, 4, 5]
    assert [row["Area"] for row in rows[:3]] == [10.0, 30.0, 50.0]
    assert "Area" not in rows[3]
    assert np.isnan(rows[4]["Area"])


def test_measure_object_size_shape_uses_compact_row_identity_policy() -> None:
    assert (
        CellProfilerObjectMeasurementRowPolicy.for_module(
            "MeasureObjectSizeShape"
        ).object_identity()
        is MeasurementObjectRowIdentity.ROW_ORDINAL
    )
    assert (
        CellProfilerObjectMeasurementRowPolicy.for_module(
            "MeasureObjectIntensity"
        ).object_identity()
        is MeasurementObjectRowIdentity.LABEL_ID
    )


def test_per_image_measurements_use_registered_record_builder() -> None:
    def threshold_like(image):
        return image, ThresholdResult(
            slice_index=0,
            final_threshold=0.3,
            original_threshold=0.2,
            guide_threshold=0.0,
            sigma=1.0,
            weighted_variance=0.4,
            sum_of_entropies=0.5,
        )

    threshold_like.__processing_contract__ = ProcessingContract.PURE_2D
    runtime = _FakeCellProfilerRuntime(
        {"phase": _FakeRuntimeImage(np.ones((4, 4), dtype=np.float32))}
    )
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="Threshold",
            inputs=(ArtifactSpec("phase", ArtifactKind.IMAGE),),
            runtime_artifact_inputs=(),
            outputs=(ArtifactSpec("Threshold_5_measurements", ArtifactKind.MEASUREMENTS),),
            declared_outputs=(
                ArtifactSpec("phaseThresh", ArtifactKind.IMAGE),
                ArtifactSpec("Threshold_5_measurements", ArtifactKind.MEASUREMENTS),
            ),
        )
    )

    result = executor.run(
        threshold_like,
        np.ones((4, 4), dtype=np.float32),
        cellprofiler_runtime=runtime,
    )

    assert result.shape == (4, 4)
    assert len(runtime.measurements) == 1
    _name, rows, kwargs = runtime.measurements[0]
    assert kwargs["source_image_name"] == "phase"
    assert {row["feature_name"]: row["result_value"] for row in rows} == {
        "FinalThreshold_phaseThresh": 0.3,
        "OrigThreshold_phaseThresh": 0.2,
        "WeightedVariance_phaseThresh": 0.4,
        "SumOfEntropies_phaseThresh": 0.5,
    }


def test_per_object_measurement_reuses_2d_labels_for_each_image_stack_slice() -> None:
    @dataclass(frozen=True)
    class SliceObjectMeasurement:
        slice_index: int
        object_label: int
        value: float

    calls: list[tuple[tuple[int, ...], tuple[int, ...], float]] = []

    def measure(image: np.ndarray, *, labels: np.ndarray):
        calls.append((image.shape, labels.shape, float(image[0, 0])))
        return image, [
            SliceObjectMeasurement(slice_index=0, object_label=1, value=float(image[0, 0])),
            SliceObjectMeasurement(slice_index=0, object_label=2, value=float(image[0, 1])),
        ]

    measure.__processing_contract__ = ProcessingContract.PURE_2D
    image_stack = np.stack(
        (
            np.asarray([[10.0, 11.0], [0.0, 0.0]], dtype=np.float32),
            np.asarray([[20.0, 21.0], [0.0, 0.0]], dtype=np.float32),
        )
    )
    labels = np.asarray([[1, 2], [0, 0]], dtype=np.int32)
    runtime = _FakeCellProfilerRuntime(
        {"Intensity": _FakeRuntimeImage(image_stack)},
        {
            "Objects": ObjectLabelSet(
                name="Objects",
                labels=labels,
            )
        },
    )
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="MeasureObjectIntensity",
            inputs=(
                ArtifactSpec("Intensity", ArtifactKind.IMAGE),
                ArtifactSpec("Objects", ArtifactKind.OBJECT_LABELS),
            ),
            runtime_artifact_inputs=(
                ArtifactSpec("Intensity", ArtifactKind.IMAGE),
                ArtifactSpec("Objects", ArtifactKind.OBJECT_LABELS),
            ),
            outputs=(ArtifactSpec("ObjectIntensity", ArtifactKind.MEASUREMENTS),),
        )
    )

    result = executor.run(measure, image_stack, cellprofiler_runtime=runtime)

    assert result is image_stack
    assert calls == [((2, 2), (2, 2), 10.0), ((2, 2), (2, 2), 20.0)]
    assert len(runtime.measurements) == 1
    _name, rows, kwargs = runtime.measurements[0]
    assert kwargs["object_name"] == "Objects"
    assert {
        (row["slice_index"], row["object_label"], row["value"])
        for row in rows
    } == {
        (0, 1, 10.0),
        (0, 2, 11.0),
        (1, 1, 20.0),
        (1, 2, 21.0),
    }


def test_per_object_measurement_records_declared_empty_measurement_table() -> None:
    def measure(image: np.ndarray, *, labels: np.ndarray):
        return image, []

    measure.__processing_contract__ = ProcessingContract.PURE_2D
    measure.__special_outputs__ = (
        (
            "ObjectSizeShape",
            csv_materializer(fields=("object_label", "area")),
        ),
    )
    image = np.ones((3, 3), dtype=np.float32)
    runtime = _FakeCellProfilerRuntime(
        {"Intensity": _FakeRuntimeImage(image)},
        {
            "Objects": ObjectLabelSet(
                name="Objects",
                labels=np.zeros(image.shape, dtype=np.int32),
            )
        },
    )
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="MeasureObjectSizeShape",
            inputs=(
                ArtifactSpec("Intensity", ArtifactKind.IMAGE),
                ArtifactSpec("Objects", ArtifactKind.OBJECT_LABELS),
            ),
            runtime_artifact_inputs=(
                ArtifactSpec("Intensity", ArtifactKind.IMAGE),
                ArtifactSpec("Objects", ArtifactKind.OBJECT_LABELS),
            ),
            outputs=(
                ArtifactSpec(
                    "ObjectSizeShape",
                    ArtifactKind.MEASUREMENTS,
                ),
            ),
        )
    )

    result = executor.run(measure, image, cellprofiler_runtime=runtime)

    assert result is image
    assert len(runtime.measurements) == 1
    name, rows, kwargs = runtime.measurements[0]
    assert name == "ObjectSizeShape"
    assert rows == []
    assert kwargs["object_name"] == "Objects"
    assert tuple(field.name for field in kwargs["fields"]) == (
        "object_label",
        "area",
    )


def test_measurement_record_fields_prefers_artifact_materialization_schema() -> None:
    spec = ArtifactSpec(
        name="measurements",
        kind=ArtifactKind.MEASUREMENTS,
        materialization=csv_materializer(fields=["object_label", "area"]),
    )

    fields = CellProfilerMeasurementFieldSchema.for_record(
        spec, [], measure_object_size_shape
    )

    assert tuple(field.name for field in fields) == ("object_label", "area")


def test_measure_object_size_shape_declares_schema_on_special_output() -> None:
    spec = ArtifactSpec(name="measurements", kind=ArtifactKind.MEASUREMENTS)

    fields = CellProfilerMeasurementFieldSchema.for_record(
        spec, [], measure_object_size_shape
    )

    assert tuple(field.name for field in fields) == object_shape_measurement_all_field_names()


def test_measure_object_size_shape_outputs_basic_measurement_rows() -> None:
    image = np.ones((7, 7), dtype=np.float32)
    labels = np.zeros(image.shape, dtype=np.int32)
    labels[1:4, 1:4] = 1

    _image, rows = measure_object_size_shape(
        image,
        labels,
        calculate_advanced=False,
        calculate_zernikes=False,
        dtype_config=DtypeConfig(),
    )

    assert len(rows) == 1
    assert rows[0]["object_label"] == 1
    assert rows[0]["Area"] == 9.0
    assert rows[0]["Center_X"] == 2.0
    assert rows[0]["Center_Y"] == 2.0


def test_measure_object_size_shape_exports_skimage_perimeter() -> None:
    image = np.ones((9, 9), dtype=np.float32)
    labels = np.zeros(image.shape, dtype=np.int32)
    y, x = np.ogrid[-2:3, -2:3]
    labels[2:7, 2:7][x * x + y * y <= 4] = 1

    _image, rows = measure_object_size_shape(
        image,
        labels,
        calculate_advanced=False,
        calculate_zernikes=False,
        dtype_config=DtypeConfig(),
    )

    expected_perimeter = skimage.measure.perimeter(labels == 1, neighborhood=4)
    assert abs(rows[0]["Perimeter"] - expected_perimeter) < 1e-12


def test_measure_object_size_shape_form_factor_uses_exported_perimeter() -> None:
    image = np.ones((9, 9), dtype=np.float32)
    labels = np.zeros(image.shape, dtype=np.int32)
    y, x = np.ogrid[-2:3, -2:3]
    labels[2:7, 2:7][x * x + y * y <= 4] = 1

    _image, rows = measure_object_size_shape(
        image,
        labels,
        calculate_advanced=False,
        calculate_zernikes=False,
        dtype_config=DtypeConfig(),
    )

    expected_form_factor = (
        4.0
        * np.pi
        * float(rows[0]["Area"])
        / float(rows[0]["Perimeter"]) ** 2
    )
    assert abs(rows[0]["FormFactor"] - expected_form_factor) < 1e-12
    assert abs(rows[0]["Compactness"] - (1.0 / expected_form_factor)) < 1e-12


def test_measure_object_size_shape_orientation_uses_cellprofiler_diagonal_tie() -> None:
    image = np.ones((26, 26), dtype=np.float32)
    labels = np.zeros(image.shape, dtype=np.int32)
    mask = np.array(
        [
            [0, 0, 1, 1, 0],
            [0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
        ],
        dtype=bool,
    )
    labels[10:15, 10:15][mask] = 1

    _image, rows = measure_object_size_shape(
        image,
        labels,
        calculate_advanced=True,
        calculate_zernikes=False,
        dtype_config=DtypeConfig(),
    )

    assert rows[0]["Orientation"] == 45.0


def test_measure_object_size_shape_zernikes_use_declared_row_ordinal_domain() -> None:
    image = np.ones((12, 12), dtype=np.float32)
    labels = np.zeros(image.shape, dtype=np.int32)
    labels[1:4, 1:4] = 1
    labels[6:10, 6:10] = 3

    _image, rows = measure_object_size_shape(
        image,
        ObjectLabelSet(name="Nuclei", labels=labels, declared_object_count=3),
        calculate_advanced=False,
        calculate_zernikes=True,
        dtype_config=DtypeConfig(),
    )

    assert [row["object_label"] for row in rows] == [1, 2, 3]
    assert np.isfinite(rows[0]["Zernike_0_0"])
    assert np.isnan(rows[1]["Area"])
    assert np.isnan(rows[1]["Center_X"])
    assert np.isnan(rows[1]["Center_Y"])
    assert np.isfinite(rows[1]["Zernike_0_0"])
    assert np.isnan(rows[2]["Zernike_0_0"])
    assert rows[2]["Area"] == 16.0


def test_measure_object_size_shape_uses_present_domain_for_undeclared_dense_labels() -> None:
    image = np.ones((12, 12), dtype=np.float32)
    labels = np.zeros(image.shape, dtype=np.int32)
    labels[1:4, 1:4] = 1
    labels[6:10, 6:10] = 1000

    _image, rows = measure_object_size_shape(
        image,
        labels,
        calculate_advanced=False,
        calculate_zernikes=False,
        dtype_config=DtypeConfig(),
    )

    assert [row["object_label"] for row in rows] == [1, 1000]
    assert rows[1]["Center_X"] == 7.5
    assert rows[1]["Center_Y"] == 7.5


def test_filterobjects_uses_upstream_form_factor_table_when_available() -> None:
    image = np.ones((9, 9), dtype=np.float32)
    labels = np.zeros(image.shape, dtype=np.int32)
    y, x = np.ogrid[-2:3, -2:3]
    labels[2:7, 2:7][x * x + y * y <= 4] = 1
    exported_perimeter = skimage.measure.perimeter(labels == 1, neighborhood=4)
    exported_form_factor = 4.0 * np.pi * float(np.count_nonzero(labels)) / exported_perimeter**2

    result = filter_objects(
        image,
        mode=FilterMode.MEASUREMENTS,
        filter_method=FilterMethod.LIMITS,
        object_labels=(labels,),
        measurement_features=("AreaShape_FormFactor",),
        measurement_min_values=(0.2,),
        measurement_max_values=(1.0,),
        measurement_use_minimum=(True,),
        measurement_use_maximum=(True,),
        measurement_tables=(
            MeasurementTable(
                name="Shape",
                object_name="Objects",
                rows=(
                    {
                        "object_label": 1,
                        "FormFactor": exported_form_factor,
                    },
                ),
            ),
        ),
        dtype_config=DtypeConfig(),
    )

    assert exported_form_factor > 1.0
    _output_image, stats, filtered_labels = result[:3]
    assert stats.objects_post_filter == 0
    assert object_label_dense_array(filtered_labels).max() == 0


def test_filterobjects_derives_form_factor_when_measurement_table_is_absent() -> None:
    image = np.ones((9, 9), dtype=np.float32)
    labels = np.zeros(image.shape, dtype=np.int32)
    y, x = np.ogrid[-2:3, -2:3]
    labels[2:7, 2:7][x * x + y * y <= 4] = 1

    result = filter_objects(
        image,
        mode=FilterMode.MEASUREMENTS,
        filter_method=FilterMethod.LIMITS,
        object_labels=(labels,),
        measurement_features=("AreaShape_FormFactor",),
        measurement_min_values=(0.2,),
        measurement_max_values=(1.0,),
        measurement_use_minimum=(True,),
        measurement_use_maximum=(True,),
        dtype_config=DtypeConfig(),
    )

    _output_image, stats, filtered_labels = result[:3]
    assert stats.objects_post_filter == 1
    assert object_label_dense_array(filtered_labels).max() == 1


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


def test_color_to_gray_splits_channel_last_non_rgb_slice() -> None:
    image = np.zeros((4, 5, 2), dtype=np.float32)
    image[..., 0] = 7.0
    image[..., 1] = 11.0

    (first_channel,) = color_to_gray(
        image,
        mode="split",
        image_type="rgb",
        channel_indices=(0,),
        dtype_config=DtypeConfig(),
    )

    assert first_channel.shape == (4, 5)
    np.testing.assert_array_equal(first_channel, np.full((4, 5), 7.0, dtype=np.float32))


def test_color_to_gray_preserves_masked_image_payload() -> None:
    image = np.zeros((3, 4, 3), dtype=np.float32)
    image[..., 0] = 0.75
    mask = np.array(
        (
            (True, False, True, True),
            (True, True, False, True),
            (False, True, True, True),
        )
    )

    (red,) = color_to_gray(
        MaskedImagePayload(data=image, mask=mask),
        mode="split",
        image_type="rgb",
        channel_indices=(0,),
        dtype_config=DtypeConfig(),
    )

    assert isinstance(red, MaskedImagePayload)
    np.testing.assert_array_equal(red.data, image[..., 0])
    np.testing.assert_array_equal(red.mask, mask)


def test_aligned_payload_treats_hwc_color_as_one_slice() -> None:
    color_slice = np.zeros((4, 5, 3), dtype=np.float32)

    slices = payload_slices_for_alignment(color_slice)

    assert len(slices) == 1
    assert slices[0] is color_slice
    assert payload_slice_count(color_slice) == 1


def test_aligned_payload_slices_masked_image_stacks() -> None:
    stack = np.zeros((2, 4, 5), dtype=np.float32)
    mask = np.array(
        (
            np.ones((4, 5), dtype=bool),
            np.zeros((4, 5), dtype=bool),
        )
    )

    slices = payload_slices_for_alignment(MaskedImagePayload(data=stack, mask=mask))

    assert len(slices) == 2
    assert all(isinstance(slice_payload, MaskedImagePayload) for slice_payload in slices)
    np.testing.assert_array_equal(slices[0].mask, mask[0])
    np.testing.assert_array_equal(slices[1].mask, mask[1])


def test_aligned_payload_slices_masked_volume_channel_stacks() -> None:
    stack = np.zeros((2, 3, 4, 5), dtype=np.float32)
    mask = np.zeros_like(stack, dtype=bool)
    mask[0] = True

    slices = payload_slices_for_alignment(MaskedImagePayload(data=stack, mask=mask))

    assert len(slices) == 2
    assert all(isinstance(slice_payload, MaskedImagePayload) for slice_payload in slices)
    np.testing.assert_array_equal(slices[0].mask, mask[0])
    np.testing.assert_array_equal(slices[1].mask, mask[1])


def test_aligned_payload_slices_preserve_image_metadata() -> None:
    stack = np.zeros((2, 4, 5), dtype=np.float32)
    payload = ImageMetadataPayload(
        data=stack,
        metadata=ImagePayloadMetadata(
            channel_intensity_scales=(65535.0, 255.0),
            channel_source_dtypes=("uint16", "uint8"),
        ),
    )

    slices = payload_slices_for_alignment(payload)

    assert len(slices) == 2
    assert slices[0].metadata.intensity_scale == 65535.0
    assert slices[0].metadata.source_dtype == "uint16"
    assert slices[1].metadata.intensity_scale == 255.0
    assert slices[1].metadata.source_dtype == "uint8"


def test_compose_one_image_bundle_stacks_per_image_masks_for_volume_bundle() -> None:
    first = MaskedImagePayload(
        data=np.zeros((3, 4, 5), dtype=np.float32),
        mask=np.ones((3, 4, 5), dtype=bool),
    )
    second = MaskedImagePayload(
        data=np.ones((3, 4, 5), dtype=np.float32),
        mask=np.zeros((3, 4, 5), dtype=bool),
    )

    bundle = compose_one_image_bundle((first, second))

    assert isinstance(bundle, MaskedImagePayload)
    assert bundle.data.shape == (2, 3, 4, 5)
    assert bundle.mask.shape == bundle.data.shape
    np.testing.assert_array_equal(bundle.mask[0], first.mask)
    np.testing.assert_array_equal(bundle.mask[1], second.mask)


def test_cellprofiler_auxiliary_payload_stack_preserves_metadata() -> None:
    first = image_payload_with_context(
        np.zeros((1, 4, 5), dtype=np.float32),
        metadata=ImagePayloadMetadata(intensity_scale=65535.0, source_dtype="uint16"),
    )
    second = image_payload_with_context(
        np.ones((1, 4, 5), dtype=np.float32),
        metadata=ImagePayloadMetadata(intensity_scale=255.0, source_dtype="uint8"),
    )

    stacked = CellProfilerPure2DOutputAggregator.aggregate(
        [first, second],
        "numpy",
    )

    assert isinstance(stacked, ImageMetadataPayload)
    assert image_payload_data(stacked).shape == (2, 4, 5)
    assert image_payload_metadata(stacked).for_channel(0).intensity_scale == 65535.0
    assert image_payload_metadata(stacked).for_channel(1).source_dtype == "uint8"


def test_module_executor_rewraps_single_image_output_for_openhcs_main_flow() -> None:
    def to_gray(image: np.ndarray) -> np.ndarray:
        return image[..., 0]

    to_gray.__processing_contract__ = ProcessingContract.PURE_2D
    color_slice = np.zeros((4, 5, 3), dtype=np.float32)
    color_stack = color_slice[np.newaxis, ...]
    runtime = _FakeCellProfilerRuntime(
        {"OrigColor": _FakeRuntimeImage(color_slice, source_image_name="OrigColor")}
    )
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="ColorToGray",
            inputs=(ArtifactSpec("OrigColor", ArtifactKind.IMAGE),),
            outputs=(ArtifactSpec("OrigGray", ArtifactKind.IMAGE),),
        )
    )

    result = executor.run(to_gray, color_stack, cellprofiler_runtime=runtime)

    assert result.shape == (1, 4, 5)
    assert runtime.images["OrigGray"].data.shape == (4, 5)


def test_module_executor_replaces_main_flow_for_declared_image_fan_in_output() -> None:
    def add_channels(image: np.ndarray) -> np.ndarray:
        return image.sum(axis=0)

    add_channels.__processing_contract__ = ProcessingContract.PURE_3D
    channel_stack = np.stack(
        (
            np.full((2, 5, 6), 1.0, dtype=np.float32),
            np.full((2, 5, 6), 2.0, dtype=np.float32),
            np.full((2, 5, 6), 3.0, dtype=np.float32),
        )
    )
    runtime = _FakeCellProfilerRuntime(
        {"Orig": _FakeRuntimeImage(channel_stack, source_image_name="Orig")}
    )
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="ImageMath",
            inputs=(ArtifactSpec("Orig", ArtifactKind.IMAGE),),
            outputs=(ArtifactSpec("Combined", ArtifactKind.IMAGE),),
        )
    )

    result = executor.run(add_channels, channel_stack, cellprofiler_runtime=runtime)

    assert result.shape == (1, *channel_stack.shape[1:])
    np.testing.assert_allclose(result[0], 6.0)
    assert runtime.images["Combined"].data.shape == channel_stack.shape[1:]


def test_module_executor_preserves_duplicate_image_roles_for_illumination_apply():
    illumination = np.full((4, 5), 2.0, dtype=np.float32)
    runtime = _FakeCellProfilerRuntime(
        {"IllumGreen": _FakeRuntimeImage(illumination, source_image_name="IllumGreen")}
    )
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="CorrectIlluminationApply",
            inputs=(
                ArtifactSpec("IllumGreen", ArtifactKind.IMAGE),
                ArtifactSpec("IllumGreen", ArtifactKind.IMAGE),
            ),
            runtime_artifact_inputs=(ArtifactSpec("IllumGreen", ArtifactKind.IMAGE),),
            outputs=(ArtifactSpec("CorrGreen", ArtifactKind.IMAGE),),
        )
    )

    result = executor.run(
        correct_illumination_apply,
        illumination,
        cellprofiler_runtime=runtime,
        method="divide",
        dtype_config=DtypeConfig(),
    )

    np.testing.assert_allclose(image_payload_data(result), np.ones((1, 4, 5)))
    np.testing.assert_allclose(
        image_payload_data(runtime.images["CorrGreen"].data),
        np.ones((4, 5)),
    )


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


def test_cellprofiler_contract_executor_aggregates_object_label_payload_auxiliary():
    def keep_payload(image: np.ndarray):
        labels = np.full(image.shape, int(image[0, 0]) + 1, dtype=np.int32)
        return (
            image,
            ObjectLabelPayload(
                labels=labels,
                unedited_labels=labels + 10,
                small_removed_labels=labels + 20,
            ),
        )

    keep_payload.__processing_contract__ = ProcessingContract.PURE_2D
    stack = np.stack(
        (
            np.zeros((4, 5), dtype=np.uint16),
            np.ones((4, 5), dtype=np.uint16),
        )
    )

    result_image, result_payload = CellProfilerFunctionContractExecutor().execute(
        keep_payload,
        stack,
        {},
    )

    assert result_image.shape == stack.shape
    assert isinstance(result_payload, ObjectLabelPayload)
    assert result_payload.labels.shape == stack.shape
    np.testing.assert_array_equal(result_payload.labels[0], np.full((4, 5), 1))
    np.testing.assert_array_equal(result_payload.labels[1], np.full((4, 5), 2))
    np.testing.assert_array_equal(
        result_payload.unedited_labels,
        result_payload.labels + 10,
    )
    np.testing.assert_array_equal(
        result_payload.small_removed_labels,
        result_payload.labels + 20,
    )


def test_cellprofiler_contract_executor_aggregates_volume_label_auxiliary():
    def keep_volume_labels(image: np.ndarray):
        labels = (image > 0).astype(np.int32)
        return image, labels

    keep_volume_labels.__processing_contract__ = ProcessingContract.PURE_2D
    stack = np.stack(
        (
            np.ones((3, 4, 5), dtype=np.float32),
            np.full((3, 4, 5), 2.0, dtype=np.float32),
        )
    )

    result_image, result_labels = CellProfilerFunctionContractExecutor().execute(
        keep_volume_labels,
        stack,
        {},
    )

    assert result_image.shape == (6, 4, 5)
    assert isinstance(result_labels, np.ndarray)
    assert result_labels.shape == (6, 4, 5)


def test_cellprofiler_contract_executor_preserves_single_slice_dataclass_auxiliary():
    @dataclass(frozen=True)
    class SliceStats:
        slice_index: int
        threshold_used: float

    def segment(image: np.ndarray, *, slice_index: int = 0, slice_count: int = 1):
        assert slice_count == 1
        return image, SliceStats(slice_index=slice_index, threshold_used=0.25)

    segment.__processing_contract__ = ProcessingContract.PURE_2D
    image = np.ones((4, 5), dtype=np.float32)

    result_image, result_stats = CellProfilerFunctionContractExecutor().execute(
        segment,
        image,
        {},
    )

    np.testing.assert_array_equal(result_image, image)
    assert result_stats == SliceStats(slice_index=0, threshold_used=0.25)


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


def test_mask_objects_uses_object_labels_as_primary_execution_domain() -> None:
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="MaskObjects",
            inputs=(
                ArtifactSpec("Nuclei", ArtifactKind.OBJECT_LABELS),
                ArtifactSpec("Cells", ArtifactKind.OBJECT_LABELS),
                ArtifactSpec("CarrierImage", ArtifactKind.IMAGE),
            ),
            runtime_artifact_inputs=(
                ArtifactSpec("Nuclei", ArtifactKind.OBJECT_LABELS),
                ArtifactSpec("Cells", ArtifactKind.OBJECT_LABELS),
                ArtifactSpec("CarrierImage", ArtifactKind.IMAGE),
            ),
            outputs=(),
        )
    )

    assert executor.primary_image_inputs(mask_objects) == ()


def test_module_executor_slices_aligned_object_labels_for_pure_2d_module():
    calls = []

    def crop_like(image: np.ndarray, *, cropping_labels: np.ndarray) -> np.ndarray:
        calls.append((image.shape, cropping_labels.shape, int(cropping_labels[0, 0])))
        return image + cropping_labels

    crop_like.__processing_contract__ = ProcessingContract.PURE_2D
    image_stack = np.stack(
        (
            np.full((4, 5), 10, dtype=np.float32),
            np.full((4, 5), 20, dtype=np.float32),
        )
    )
    label_stack = np.stack(
        (
            np.full((4, 5), 1, dtype=np.int32),
            np.full((4, 5), 2, dtype=np.int32),
        )
    )
    runtime = _FakeCellProfilerRuntime(
        {"InvBlue": _FakeRuntimeImage(image_stack)},
        {
            "NonOverlappingWorms": ObjectLabelSet(
                name="NonOverlappingWorms",
                labels=label_stack,
            )
        },
    )
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="Crop",
            inputs=(
                ArtifactSpec("InvBlue", ArtifactKind.IMAGE),
                ArtifactSpec("NonOverlappingWorms", ArtifactKind.OBJECT_LABELS),
            ),
            runtime_artifact_inputs=(
                ArtifactSpec("InvBlue", ArtifactKind.IMAGE),
                ArtifactSpec("NonOverlappingWorms", ArtifactKind.OBJECT_LABELS),
            ),
            outputs=(ArtifactSpec("CropBlue", ArtifactKind.IMAGE),),
        )
    )

    result = executor.run(crop_like, image_stack, cellprofiler_runtime=runtime)

    assert calls == [((4, 5), (4, 5), 1), ((4, 5), (4, 5), 2)]
    assert result.shape == image_stack.shape
    np.testing.assert_array_equal(result, image_stack + label_stack)


def test_module_executor_binds_crop_previous_mask_when_image_output_is_pruned() -> None:
    image = np.arange(4 * 5, dtype=np.float32).reshape(4, 5)
    previous_mask = np.zeros((4, 5), dtype=bool)
    previous_mask[1:3, 1:4] = True
    runtime = _FakeCellProfilerRuntime(
        {
            "OrigRed": _FakeRuntimeImage(image, source_image_name="OrigRed"),
            "CropBlue__crop_mask": _FakeRuntimeImage(
                previous_mask.astype(np.float32),
                source_image_name="OrigBlue",
            ),
        }
    )
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="Crop",
            inputs=(
                ArtifactSpec("OrigRed", ArtifactKind.IMAGE),
                ArtifactSpec("CropBlue__crop_mask", ArtifactKind.IMAGE),
            ),
            runtime_artifact_inputs=(
                ArtifactSpec("CropBlue__crop_mask", ArtifactKind.IMAGE),
            ),
            outputs=(ArtifactSpec("Crop_7_measurements", ArtifactKind.MEASUREMENTS),),
            declared_outputs=(
                ArtifactSpec("CropRed", ArtifactKind.IMAGE),
                ArtifactSpec(
                    "CropRed__crop_mask",
                    ArtifactKind.IMAGE,
                    sidecar_role=ArtifactSidecarRole.CROP_MASK,
                ),
                ArtifactSpec("Crop_7_measurements", ArtifactKind.MEASUREMENTS),
            ),
        )
    )

    result = executor.run(
        crop,
        image,
        cellprofiler_runtime=runtime,
        crop_shape=CropShape.CROPPING,
        removal_method=RemovalMethod.EDGES,
        dtype_config=DtypeConfig(),
    )

    assert result is image
    assert len(runtime.measurements) == 1
    name, rows, kwargs = runtime.measurements[0]
    assert name == "Crop_7_measurements"
    assert kwargs["source_image_name"] == "CropRed"
    assert len(rows) == 1
    assert rows[0]["area_retained"] == 6


def test_output_value_matching_skips_pruned_intermediate_artifacts() -> None:
    image = np.zeros((3, 4), dtype=np.float32)
    stats = {"objects_pre_filter": 3, "objects_post_filter": 2}
    labels = np.ones((3, 4), dtype=np.int32)
    relationship = ParentChildRelationshipPayload(
        parent_ids=np.asarray([1, 2], dtype=np.int32),
        child_ids=np.asarray([1, 2], dtype=np.int32),
        slice_indices=np.asarray([0, 0], dtype=np.int32),
        slice_count=1,
    )

    values = _output_values_by_kind(
        (
            ArtifactSpec("FilterStats", ArtifactKind.MEASUREMENTS),
            ArtifactSpec("FilteredObjects", ArtifactKind.OBJECT_LABELS),
        ),
        image,
        (stats, labels, relationship),
        func=filter_objects,
    )

    assert values["FilterStats"] is stats
    assert values["FilteredObjects"] is labels

    relationship_values = _output_values_by_kind(
        (
            ArtifactSpec("FilterStats", ArtifactKind.MEASUREMENTS),
            ArtifactSpec("FilteredRelationships", ArtifactKind.RELATIONSHIPS),
        ),
        image,
        (stats, labels, relationship),
        func=filter_objects,
    )

    assert relationship_values["FilterStats"] is stats
    assert relationship_values["FilteredRelationships"] is relationship


def test_output_value_matching_uses_declared_sidecar_specs_for_pruned_outputs() -> None:
    cropped_image = np.ones((3, 4), dtype=np.float32)
    crop_mask = np.ones((10, 12), dtype=bool)
    measurements = [{"area_retained": 12}]

    values = _output_values_by_kind(
        (
            ArtifactSpec("CropGreen", ArtifactKind.IMAGE),
            ArtifactSpec("Crop_6_measurements", ArtifactKind.MEASUREMENTS),
        ),
        cropped_image,
        (crop_mask, measurements),
        declared_output_specs=(
            ArtifactSpec("CropGreen", ArtifactKind.IMAGE),
            ArtifactSpec(
                "CropGreen__crop_mask",
                ArtifactKind.IMAGE,
                sidecar_role=ArtifactSidecarRole.CROP_MASK,
            ),
            ArtifactSpec("Crop_6_measurements", ArtifactKind.MEASUREMENTS),
        ),
    )

    assert values["CropGreen"] is cropped_image
    assert values["Crop_6_measurements"] is measurements


def test_pure_2d_batch_lowers_nominal_runtime_output_bundles() -> None:
    first_output = np.ones((2, 2), dtype=np.float32)
    second_output = np.full((2, 2), 2, dtype=np.float32)
    first_relationship = ParentChildRelationshipPayload(parent_ids=(1,), child_ids=(1,))
    second_relationship = ParentChildRelationshipPayload(parent_ids=(2,), child_ids=(2,))
    first_measurements = RelationshipMeasurements(
        slice_index=0,
        parent_object_count=1,
        child_object_count=1,
        children_with_parents_count=1,
        mean_children_per_parent=1.0,
        mean_centroid_distance=0.0,
        mean_minimum_distance=0.0,
    )
    second_measurements = RelationshipMeasurements(
        slice_index=1,
        parent_object_count=1,
        child_object_count=1,
        children_with_parents_count=1,
        mean_children_per_parent=1.0,
        mean_centroid_distance=0.0,
        mean_minimum_distance=0.0,
    )

    batch = Pure2DSliceResultBatch.from_results(
        (
            RelateObjectsResult(first_output, first_relationship, first_measurements),
            RelateObjectsResult(second_output, second_relationship, second_measurements),
        )
    )

    assert batch.main_outputs == [first_output, second_output]
    assert batch.auxiliary_groups == (
        [first_relationship, second_relationship],
        [first_measurements, second_measurements],
    )


def test_output_value_resolution_preserves_pruned_context_outputs() -> None:
    cropped_image = np.zeros((3, 4), dtype=np.float32)
    crop_mask = np.ones((5, 6), dtype=bool)
    measurements = {"area_retained": 12}

    resolution = CellProfilerOutputValueResolution.from_returned_values(
        (ArtifactSpec("Crop_7_measurements", ArtifactKind.MEASUREMENTS),),
        declared_specs=(
            ArtifactSpec("CropRed", ArtifactKind.IMAGE),
            ArtifactSpec(
                "CropRed__crop_mask",
                ArtifactKind.IMAGE,
                sidecar_role=ArtifactSidecarRole.CROP_MASK,
            ),
            ArtifactSpec("Crop_7_measurements", ArtifactKind.MEASUREMENTS),
        ),
        main_output=cropped_image,
        artifact_values=(crop_mask, measurements),
        func=crop,
    )

    assert resolution.recorded_values == {"Crop_7_measurements": measurements}
    assert resolution.context_values["CropRed"] is cropped_image
    assert resolution.context_values["CropRed__crop_mask"] is crop_mask
    assert resolution.context_values["Crop_7_measurements"] is measurements


def test_output_value_resolution_preserves_pruned_object_label_context() -> None:
    image = np.zeros((3, 4), dtype=np.float32)
    stats = {"objects_pre_filter": 3, "objects_post_filter": 1}
    labels = ObjectLabelPayload(
        labels=np.ones((3, 4), dtype=np.int32),
        declared_object_count=1,
    )
    relationship = ParentChildRelationshipPayload(parent_ids=(1,), child_ids=(1,))

    resolution = CellProfilerOutputValueResolution.from_returned_values(
        (
            ArtifactSpec("FilterStats", ArtifactKind.MEASUREMENTS),
            ArtifactSpec(
                "Objects1_ColocalizedObjects_relationships",
                ArtifactKind.RELATIONSHIPS,
            ),
        ),
        declared_specs=(
            ArtifactSpec("FilterStats", ArtifactKind.MEASUREMENTS),
            ArtifactSpec("ColocalizedObjects", ArtifactKind.OBJECT_LABELS),
            ArtifactSpec(
                "Objects1_ColocalizedObjects_relationships",
                ArtifactKind.RELATIONSHIPS,
            ),
        ),
        main_output=image,
        artifact_values=(stats, labels, relationship),
        func=filter_objects,
    )

    assert "ColocalizedObjects" not in resolution.recorded_values
    assert resolution.context_values["FilterStats"] is stats
    assert resolution.context_values["ColocalizedObjects"] is labels
    assert (
        resolution.context_values["Objects1_ColocalizedObjects_relationships"]
        is relationship
    )


def test_relationship_recorder_resolves_pruned_child_endpoint_from_artifact_name() -> None:
    runtime = _FakeCellProfilerRuntime(
        {"Carrier": _FakeRuntimeImage(np.zeros((3, 4), dtype=np.float32))},
    )
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="FilterObjects",
            inputs=(ArtifactSpec("Objects1", ArtifactKind.OBJECT_LABELS),),
            runtime_artifact_inputs=(
                ArtifactSpec("Objects1", ArtifactKind.OBJECT_LABELS),
            ),
            outputs=(
                ArtifactSpec(
                    "Objects1_ColocalizedObjects_relationships",
                    ArtifactKind.RELATIONSHIPS,
                ),
            ),
        )
    )
    payload = ParentChildRelationshipPayload(parent_ids=(1,), child_ids=(1,))

    executor._record_outputs(
        lambda image: image,
        runtime,
        np.zeros((3, 4), dtype=np.float32),
        (payload,),
        source_image_name=None,
    )

    name, kwargs = runtime.relationships[0]
    assert name == "Objects1_ColocalizedObjects_relationships"
    assert kwargs["parent_object_name"] == "Objects1"
    assert kwargs["child_object_name"] == "ColocalizedObjects"


def test_cellprofiler_contract_executor_broadcasts_2d_labels_to_image_stack():
    calls = []

    def add_label_values(image: np.ndarray, *, labels: np.ndarray):
        calls.append((image.shape, labels.shape, int(image[0, 0])))
        return image + labels

    add_label_values.__processing_contract__ = ProcessingContract.PURE_2D
    image = np.stack(
        (
            np.full((4, 5), 10, dtype=np.uint16),
            np.full((4, 5), 20, dtype=np.uint16),
        )
    )
    labels = np.ones((4, 5), dtype=np.uint16)

    result = CellProfilerFunctionContractExecutor().execute(
        add_label_values,
        image,
        {"labels": labels},
    )

    assert calls == [((4, 5), (4, 5), 10), ((4, 5), (4, 5), 20)]
    assert result.shape == image.shape
    np.testing.assert_array_equal(result, image + labels[np.newaxis, ...])


def test_secondary_seed_labels_remap_accepted_labels_and_preserve_edge_constraints():
    final_labels = np.array(
        [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.int32,
    )
    unedited_labels = np.array(
        [
            [4, 0, 0, 0],
            [0, 7, 7, 0],
            [0, 3, 3, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.int32,
    )

    labels_in = _secondary_seed_labels(final_labels, unedited_labels)

    expected = np.array(
        [
            [2, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.int32,
    )
    np.testing.assert_array_equal(labels_in, expected)


def test_filter_labels_maps_unedited_secondary_labels_to_accepted_primary_labels():
    primary_labels = np.array(
        [
            [1, 1, 0, 0],
            [0, 0, 0, 2],
        ],
        dtype=np.int32,
    )
    secondary_labels = np.array(
        [
            [7, 7, 7, 0],
            [8, 8, 0, 9],
        ],
        dtype=np.int32,
    )

    filtered = _filter_labels(secondary_labels, primary_labels)

    expected = np.array(
        [
            [1, 1, 1, 0],
            [0, 0, 0, 2],
        ],
        dtype=np.int32,
    )
    np.testing.assert_array_equal(filtered, expected)


def test_distance_b_limits_expansion_from_accepted_primary_labels(monkeypatch):
    def fake_propagate(
        self: DistanceMaskedSegmentationStrategy,
        request: SecondarySegmentationRequest,
        *,
        regularization: float,
        max_distance: float | None = None,
    ) -> np.ndarray:
        del self, request, regularization, max_distance
        return np.array([[1, 0, 0, 1, 4]], dtype=np.int32)

    monkeypatch.setattr(
        DistanceMaskedSegmentationStrategy,
        "propagate_labels",
        fake_propagate,
    )
    final_labels = np.array([[1, 0, 0, 0, 0]], dtype=np.int32)
    unedited_labels = np.array([[1, 0, 0, 0, 4]], dtype=np.int32)

    segmented = DistanceMaskedSegmentationStrategy().segment(
        SecondarySegmentationRequest(
            image=np.zeros((1, 5), dtype=np.float32),
            labels=final_labels,
            unedited_labels=unedited_labels,
            thresholded=np.ones((1, 5), dtype=bool),
            distance_to_dilate=2,
            regularization_factor=0.05,
            watershed_backend_provider=None,
        )
    )

    expected = np.array([[1, 0, 0, 1, 0]], dtype=np.int32)
    np.testing.assert_array_equal(segmented, expected)


def test_secondary_propagation_uses_threshold_mask_without_seed_or(monkeypatch):
    captured: dict[str, np.ndarray] = {}

    def fake_propagate(
        self: PropagationSegmentationStrategy,
        request: SecondarySegmentationRequest,
        *,
        regularization: float,
        max_distance: float | None = None,
    ) -> np.ndarray:
        del self, regularization, max_distance
        captured["mask"] = request.thresholded
        return request.unedited_labels.copy()

    monkeypatch.setattr(
        PropagationSegmentationStrategy,
        "propagate_labels",
        fake_propagate,
    )
    labels = np.array([[1, 0], [0, 0]], dtype=np.int32)
    thresholded = np.array([[False, False], [False, True]])

    PropagationSegmentationStrategy().segment(
        SecondarySegmentationRequest(
            image=np.zeros((2, 2), dtype=np.float32),
            labels=labels,
            unedited_labels=labels,
            thresholded=thresholded,
            distance_to_dilate=10,
            regularization_factor=0.05,
            watershed_backend_provider=None,
        )
    )

    np.testing.assert_array_equal(captured["mask"], thresholded)


def test_identify_secondary_objects_collapses_singleton_image_label_and_mask_planes():
    image = np.zeros((1, 5, 5), dtype=np.float32)
    image[0, 1:4, 1:4] = 1.0
    labels = np.zeros((1, 5, 5), dtype=np.int32)
    labels[0, 2, 2] = 1
    payload = image_payload_with_context(image, mask=np.ones_like(image, dtype=bool))

    _image, _stats, _relationships, secondary = iso.identify_secondary_objects(
        payload,
        labels,
        method=iso.SecondaryMethod.DISTANCE_B,
        distance_to_dilate=1,
        dtype_config=DtypeConfig(),
    )

    secondary_labels = object_label_dense_array(secondary)

    assert secondary_labels.shape == (5, 5)
    assert secondary_labels.max() == 1


def test_secondary_propagation_aligns_label_and_mask_planes_to_image_geometry():
    image = np.zeros((6, 7), dtype=np.float32)
    labels = np.zeros((5, 5), dtype=np.int32)
    labels[2, 2] = 1
    mask = np.ones((5, 5), dtype=bool)

    propagated = PropagationSegmentationStrategy().propagate_labels(
        SecondarySegmentationRequest(
            image=image,
            labels=labels,
            unedited_labels=labels,
            thresholded=mask,
            distance_to_dilate=10,
            regularization_factor=1.0,
            watershed_backend_provider=None,
        ),
        regularization=1.0,
    )

    assert propagated.shape == image.shape


def test_parent_child_relationship_aligns_cropped_object_label_payload_to_source_domain():
    parent = ObjectLabelPayload(
        labels=np.array([[1, 0], [0, 0]], dtype=np.int32),
        spatial_origin_yx=(2, 3),
        source_spatial_shape_yx=(6, 7),
    )
    child = np.zeros((6, 7), dtype=np.int32)
    child[2, 3] = 4

    relationship = object_label_parent_child_payload(parent, child)

    assert relationship.parent_ids == (1,)
    assert relationship.child_ids == (4,)


def test_pure_2d_object_label_payload_aggregation_preserves_source_domain():
    payload = ObjectLabelPayload(
        labels=np.array([[1, 0], [0, 0]], dtype=np.int32),
        spatial_origin_yx=(2, 3),
        source_spatial_shape_yx=(6, 7),
    )

    aggregated = CellProfilerPure2DOutputAggregator.aggregate(
        [payload],
        MemoryType.NUMPY.value,
    )

    assert isinstance(aggregated, ObjectLabelPayload)
    assert aggregated.spatial_origin_yx == (2, 3)
    assert aggregated.source_spatial_shape_yx == (6, 7)


def test_pure_2d_object_label_payload_aggregation_expands_varying_crop_domains():
    first = ObjectLabelPayload(
        labels=np.array([[1]], dtype=np.int32),
        declared_object_count=3,
        spatial_origin_yx=(1, 2),
        source_spatial_shape_yx=(4, 5),
    )
    second = ObjectLabelPayload(
        labels=np.array([[2]], dtype=np.int32),
        declared_object_count=5,
        spatial_origin_yx=(2, 3),
        source_spatial_shape_yx=(4, 5),
    )

    aggregated = CellProfilerPure2DOutputAggregator.aggregate(
        [first, second],
        MemoryType.NUMPY.value,
    )

    assert isinstance(aggregated, ObjectLabelPayload)
    assert aggregated.labels.shape == (2, 4, 5)
    assert aggregated.declared_object_id_domains == (
        (1, 2, 3),
        (1, 2, 3, 4, 5),
    )
    assert aggregated.domain_scope is ObjectLabelDomainScope.PLANE
    assert aggregated.spatial_origin_yx is None
    assert aggregated.source_spatial_shape_yx == (4, 5)
    assert aggregated.labels[0, 1, 2] == 1
    assert aggregated.labels[1, 2, 3] == 2


def test_pure_2d_object_label_payload_aggregation_derives_missing_plane_domains():
    first = ObjectLabelPayload(
        labels=np.array([[1, 0], [0, 0]], dtype=np.int32),
        declared_object_ids=(1, 2),
        domain_scope=ObjectLabelDomainScope.PLANE,
    )
    second = ObjectLabelPayload(
        labels=np.array([[3, 0], [0, 0]], dtype=np.int32),
        domain_scope=ObjectLabelDomainScope.PLANE,
    )

    aggregated = CellProfilerPure2DOutputAggregator.aggregate(
        [first, second],
        MemoryType.NUMPY.value,
    )

    assert isinstance(aggregated, ObjectLabelPayload)
    assert aggregated.declared_object_id_domains == ((1, 2), (3,))
    assert aggregated.domain_scope is ObjectLabelDomainScope.PLANE


def test_pure_2d_object_label_payload_slice_projects_plane_domain() -> None:
    payload = ObjectLabelPayload(
        labels=np.array(
            [
                [[1, 0], [0, 2]],
                [[1, 0], [3, 4]],
            ],
            dtype=np.int32,
        ),
        declared_object_id_domains=((1, 2), (1, 2, 3, 4)),
        domain_scope=ObjectLabelDomainScope.PLANE,
    )

    sliced = RuntimeSliceProjection.value_for_slice(payload, slice_index=1, slice_count=2)

    assert isinstance(sliced, ObjectLabelPayload)
    assert sliced.labels.shape == (2, 2)
    assert sliced.declared_object_ids == (1, 2, 3, 4)
    assert sliced.declared_object_id_domains == ()
    assert sliced.domain_scope is ObjectLabelDomainScope.PLANE


def test_pure_2d_object_label_set_slice_projects_plane_domain() -> None:
    label_set = ObjectLabelSet(
        name="Cells",
        labels=np.array(
            [
                [[1, 0], [0, 2]],
                [[1, 0], [3, 4]],
            ],
            dtype=np.int32,
        ),
        declared_object_id_domains=((1, 2), (1, 2, 3, 4)),
        domain_scope=ObjectLabelDomainScope.PLANE,
    )

    sliced = RuntimeSliceProjection.value_for_slice(label_set, slice_index=0, slice_count=2)

    assert isinstance(sliced, ObjectLabelSet)
    assert sliced.labels.shape == (2, 2)
    assert sliced.declared_object_ids == (1, 2)
    assert sliced.declared_object_id_domains == ()
    assert sliced.domain_scope is ObjectLabelDomainScope.PLANE


def test_pure_2d_object_label_payload_slice_projects_grouped_plane_domains() -> None:
    payload = ObjectLabelPayload(
        labels=np.zeros((2, 3, 4, 5), dtype=np.int32),
        declared_object_id_domains=((1,), (2,), (3,), (4,), (5,), (6,)),
        domain_scope=ObjectLabelDomainScope.PLANE,
    )

    sliced = RuntimeSliceProjection.value_for_slice(payload, slice_index=1, slice_count=2)

    assert isinstance(sliced, ObjectLabelPayload)
    assert sliced.labels.shape == (3, 4, 5)
    assert sliced.declared_object_ids == ()
    assert sliced.declared_object_id_domains == ((4,), (5,), (6,))
    assert sliced.domain_scope is ObjectLabelDomainScope.PLANE


def test_runtime_slice_count_allows_grouped_object_label_planes() -> None:
    parent = ObjectLabelPayload(
        labels=np.zeros((2, 3, 4, 5), dtype=np.int32),
        declared_object_id_domains=((1,), (2,), (3,), (4,), (5,), (6,)),
        domain_scope=ObjectLabelDomainScope.PLANE,
    )
    child = ObjectLabelPayload(
        labels=np.zeros((6, 4, 5), dtype=np.int32),
        declared_object_id_domains=((1,), (2,), (3,), (4,), (5,), (6,)),
        domain_scope=ObjectLabelDomainScope.PLANE,
    )

    assert RuntimeSliceProjection.slice_count_from_values((parent, child)) == 2


def test_runtime_slice_count_treats_sequence_kwargs_as_operands() -> None:
    first = ObjectLabelPayload(
        labels=np.zeros((2, 4, 5), dtype=np.int32),
        declared_object_id_domains=((1,), (2,)),
        domain_scope=ObjectLabelDomainScope.PLANE,
    )
    second = ObjectLabelPayload(
        labels=np.zeros((2, 4, 5), dtype=np.int32),
        declared_object_id_domains=((3,), (4,)),
        domain_scope=ObjectLabelDomainScope.PLANE,
    )

    assert RuntimeSliceProjection.slice_count_from_values(((first, second),)) == 4
    assert (
        RuntimeSliceProjection.slice_count_from_kwargs(
            {"object_labels": (first, second)},
            sequence_kwargs=frozenset({"object_labels"}),
        )
        == 2
    )


def test_runtime_slice_projection_offsets_repeated_scalar_measurement_tables() -> None:
    first = MeasurementTable(
        name="MeasureObjectIntensity_7_measurements",
        rows=[{"slice_index": 0, "object_label": 1, "std_intensity": 0.1}],
        object_name="Tile_of_grid",
        object_id_field="object_label",
        source_image_name="DF_image",
    )
    second = MeasurementTable(
        name="MeasureObjectIntensity_7_measurements",
        rows=[{"slice_index": 0, "object_label": 1, "std_intensity": 0.2}],
        object_name="Tile_of_grid",
        object_id_field="object_label",
        source_image_name="DF_image",
    )

    tables = (first, second)

    assert RuntimeSliceProjection.slice_count_from_values((tables,)) == 2
    sliced = RuntimeSliceProjection.kwargs_for_slice(
        {"measurement_tables": tables},
        slice_index=1,
        slice_count=2,
    )["measurement_tables"]
    assert tuple(len(table.rows) for table in sliced) == (0, 1)
    assert list(sliced[1].rows)[0]["std_intensity"] == 0.2


def test_identify_tertiary_batch_aligns_cropped_primary_labels_to_secondary_domain():
    from openhcs.core.runtime_batch_contracts import RuntimePure2DSliceBatchRequest

    primary = ObjectLabelPayload(
        labels=np.array([[1, 0], [0, 0]], dtype=np.int32),
        spatial_origin_yx=(2, 3),
        source_spatial_shape_yx=(6, 7),
    )
    secondary = np.zeros((6, 7), dtype=np.int32)
    secondary[2, 3] = 5
    secondary[2, 4] = 5

    results = ito._identify_tertiary_objects_batch(
        RuntimePure2DSliceBatchRequest(
            func=ito.identify_tertiary_objects,
            slices_2d=(np.zeros((6, 7), dtype=np.float32),),
            kwargs={
                "primary_labels": primary,
                "secondary_labels": secondary,
                "shrink_primary": False,
            },
            execute_slice=(
                lambda func, image, kwargs, _slice_index, _slice_count: func(
                    image,
                    **kwargs,
                )
            ),
        )
    )

    tertiary = results[0][-1]
    assert tertiary.shape == secondary.shape
    assert tertiary[2, 3] == 0
    assert tertiary[2, 4] == 5


def test_identify_tertiary_single_slice_aligns_payload_domains_before_dense_extraction():
    primary = ObjectLabelPayload(
        labels=np.array([[1, 0], [0, 0]], dtype=np.int32),
        spatial_origin_yx=(2, 3),
        source_spatial_shape_yx=(6, 7),
    )
    secondary = np.zeros((6, 7), dtype=np.int32)
    secondary[2, 3] = 5
    secondary[2, 4] = 5

    _, _, _, _, tertiary = ito.identify_tertiary_objects.__wrapped__(
        np.zeros((6, 7), dtype=np.float32),
        primary_labels=primary,
        secondary_labels=secondary,
        shrink_primary=False,
    )

    assert tertiary.shape == secondary.shape
    assert tertiary[2, 3] == 0
    assert tertiary[2, 4] == 5


def test_enhance_or_suppress_features_matches_white_tophat_reference():
    image = np.zeros((15, 15), dtype=np.float32)
    image[4, 4] = 1.0
    image[8, 9] = 0.75

    result = enhance_or_suppress_features(
        image,
        radius=4,
        speckle_accuracy=SpeckleAccuracy.SLOW,
        dtype_config=DtypeConfig(),
    )

    expected = skimage.morphology.white_tophat(
        image,
        footprint=skimage.morphology.disk(4),
    ).astype(np.float32)
    np.testing.assert_allclose(image_payload_data(result), expected)


def test_enhance_or_suppress_features_fast_speckles_uses_cellprofiler_disk():
    from scipy import ndimage as ndi

    image = np.zeros((17, 17), dtype=np.float32)
    image[8, 8] = 1.0
    image[8, 13] = 0.5
    footprint = skimage.morphology.disk(5)

    result = enhance_or_suppress_features(
        image,
        radius=5,
        speckle_accuracy=SpeckleAccuracy.FAST,
        dtype_config=DtypeConfig(),
    )

    expected = image - ndi.maximum_filter(
        ndi.minimum_filter(image, footprint=footprint),
        footprint=footprint,
    )
    np.testing.assert_allclose(image_payload_data(result), expected.astype(np.float32))


def test_enhance_or_suppress_features_tubeness_matches_hessian_reference():
    from scipy import ndimage as ndi

    image = np.zeros((21, 21), dtype=np.float32)
    image[5:16, 10] = 1.0
    image[10, 5:16] = 0.5
    smoothing_value = 2.0

    result = enhance_or_suppress_features(
        image,
        enhance_method=EnhanceMethod.NEURITES,
        neurite_method=NeuriteMethod.TUBENESS,
        smoothing_value=smoothing_value,
        dtype_config=DtypeConfig(),
    )

    smoothed = ndi.gaussian_filter(image, smoothing_value)
    hessian = np.zeros((*smoothed.shape, 2, 2), dtype=np.float64)
    hessian[1:-1, :, 0, 0] = (
        smoothed[:-2, :]
        - (2 * smoothed[1:-1, :])
        + smoothed[2:, :]
    )
    hessian[1:-1, 1:-1, 0, 1] = (
        smoothed[2:, 2:]
        + smoothed[:-2, :-2]
        - smoothed[2:, :-2]
        - smoothed[:-2, 2:]
    ) / 4
    hessian[:, 1:-1, 1, 1] = (
        smoothed[:, :-2]
        - (2 * smoothed[:, 1:-1])
        + smoothed[:, 2:]
    )
    a = hessian[:, :, 0, 0]
    b = hessian[:, :, 0, 1]
    c = hessian[:, :, 1, 1]
    linear = -(a + c)
    constant = a * c - b * b
    discriminant = np.maximum(linear * linear - 4 * constant, 0)
    roots = np.empty((*smoothed.shape, 2), dtype=np.float64)
    sqrt_discriminant = np.sqrt(discriminant)
    roots[:, :, 0] = (-linear + sqrt_discriminant) / 2
    roots[:, :, 1] = (-linear - sqrt_discriminant) / 2
    swap = np.abs(roots[:, :, 1]) > np.abs(roots[:, :, 0])
    roots[swap] = roots[swap, ::-1]
    expected = (
        -roots[..., 0]
        * (roots[..., 0] < 0)
        * (smoothing_value ** 2)
    ).astype(np.float32)
    np.testing.assert_allclose(image_payload_data(result), expected, rtol=1e-6, atol=1e-7)


def test_cellprofiler_module_executor_normalizes_integer_image_inputs() -> None:
    source_image = "DNA"
    raw = np.full((4, 5), 255, dtype=np.uint8)
    runtime = _FakeCellProfilerRuntime(
        {source_image: _FakeRuntimeImage(raw, source_image_name=source_image)}
    )
    seen: list[np.ndarray] = []

    def capture(image: np.ndarray) -> np.ndarray:
        seen.append(image)
        return image

    capture.__processing_contract__ = ProcessingContract.PURE_2D
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="Opening",
            inputs=(ArtifactSpec(source_image, ArtifactKind.IMAGE),),
            outputs=(ArtifactSpec("Normalized", ArtifactKind.IMAGE),),
        )
    )

    result = executor.run(capture, raw, cellprofiler_runtime=runtime)

    assert seen[0].dtype == np.float32
    np.testing.assert_array_equal(seen[0], np.ones_like(raw, dtype=np.float32))
    assert result.dtype == np.float32
    np.testing.assert_array_equal(
        runtime.images["Normalized"].data,
        np.ones_like(raw, dtype=np.float32),
    )


def test_cellprofiler_module_executor_uses_payload_intensity_scale() -> None:
    source_image = "DNA"
    raw = np.array([[0, 4095]], dtype=np.uint16)
    payload = image_payload_with_context(
        raw,
        metadata=ImagePayloadMetadata(intensity_scale=4095.0, source_dtype="uint16"),
    )
    runtime = _FakeCellProfilerRuntime(
        {source_image: _FakeRuntimeImage(payload, source_image_name=source_image)}
    )
    seen: list[object] = []

    def capture(image: object) -> object:
        seen.append(image)
        return image

    capture.__processing_contract__ = ProcessingContract.PURE_2D
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="Opening",
            inputs=(ArtifactSpec(source_image, ArtifactKind.IMAGE),),
            outputs=(ArtifactSpec("Normalized", ArtifactKind.IMAGE),),
        )
    )

    result = executor.run(capture, raw, cellprofiler_runtime=runtime)

    np.testing.assert_allclose(image_payload_data(seen[0]), [[0.0, 1.0]])
    assert image_payload_metadata(seen[0]).intensity_scale == 4095.0
    np.testing.assert_allclose(
        image_payload_data(runtime.images["Normalized"].data),
        [[0.0, 1.0]],
    )
    np.testing.assert_allclose(image_payload_data(result), [[0.0, 1.0]])


def test_cellprofiler_contract_executor_slices_plane_sequence_kwargs():
    calls = []

    def keep_labels(image: np.ndarray, *, labels: np.ndarray):
        calls.append((image.shape, labels.shape, int(labels[0, 0])))
        return labels

    keep_labels.__processing_contract__ = ProcessingContract.PURE_2D
    image = np.zeros((4, 5), dtype=np.uint16)
    labels = (
        np.full((4, 5), 1, dtype=np.int32),
        np.full((4, 5), 2, dtype=np.int32),
    )

    result = CellProfilerFunctionContractExecutor().execute(
        keep_labels,
        image,
        {"labels": labels},
    )

    assert calls == [((4, 5), (4, 5), 1), ((4, 5), (4, 5), 2)]
    assert result.shape == (2, 4, 5)
    np.testing.assert_array_equal(result, np.asarray(labels))


def test_cellprofiler_contract_executor_slices_array_convertible_kwargs():
    class ArrayConvertible:
        def __init__(self, data: np.ndarray) -> None:
            self.shape = data.shape
            self._data = data

        def __array__(self) -> np.ndarray:
            return self._data

    calls = []

    def keep_labels(image: np.ndarray, *, labels: np.ndarray):
        calls.append((image.shape, labels.shape, int(labels[0, 0])))
        return labels

    keep_labels.__processing_contract__ = ProcessingContract.PURE_2D
    image = np.zeros((4, 5), dtype=np.uint16)
    labels = np.stack(
        (
            np.full((4, 5), 1, dtype=np.int32),
            np.full((4, 5), 2, dtype=np.int32),
        )
    )

    result = CellProfilerFunctionContractExecutor().execute(
        keep_labels,
        image,
        {"labels": ArrayConvertible(labels)},
    )

    assert calls == [((4, 5), (4, 5), 1), ((4, 5), (4, 5), 2)]
    assert result.shape == labels.shape
    np.testing.assert_array_equal(result, labels)


def test_cellprofiler_contract_executor_slices_nested_sequence_kwargs():
    calls = []

    def keep_labels(image: np.ndarray, *, labels: np.ndarray):
        calls.append((image.shape, labels.shape, int(labels[0, 0])))
        return labels

    keep_labels.__processing_contract__ = ProcessingContract.PURE_2D
    image = np.zeros((2, 2), dtype=np.uint16)
    labels = [
        [[1, 1], [1, 1]],
        [[2, 2], [2, 2]],
    ]

    result = CellProfilerFunctionContractExecutor().execute(
        keep_labels,
        image,
        {"labels": labels},
    )

    assert calls == [((2, 2), (2, 2), 1), ((2, 2), (2, 2), 2)]
    np.testing.assert_array_equal(result, np.asarray(labels))


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


def test_correct_illumination_all_scope_module_executor_uses_full_stack():
    calls = []

    def calculate_illumination(image: np.ndarray, *, calculation_scope: str):
        calls.append((image.shape, calculation_scope))
        return image.mean(axis=0).astype(np.float32), []

    calculate_illumination.__processing_contract__ = ProcessingContract.PURE_2D
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="CorrectIlluminationCalculate",
            inputs=(ArtifactSpec("OrigGreen", ArtifactKind.IMAGE),),
            outputs=(ArtifactSpec("IllumGreen", ArtifactKind.IMAGE),),
        )
    )
    stack = np.stack(
        (
            np.full((4, 5), 0.25, dtype=np.float32),
            np.full((4, 5), 0.75, dtype=np.float32),
        )
    )
    runtime = _FakeCellProfilerRuntime({"OrigGreen": _FakeRuntimeImage(stack)})

    result = executor.run(
        calculate_illumination,
        stack,
        cellprofiler_runtime=runtime,
        calculation_scope="all_first_cycle",
    )

    assert calls == [((2, 4, 5), "all_first_cycle")]
    np.testing.assert_array_equal(result, stack)
    np.testing.assert_array_equal(
        runtime.images["IllumGreen"].data,
        np.full((4, 5), 0.5, dtype=np.float32),
    )


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


def test_compose_image_payload_collapses_pairwise_slice_grids_before_alignment():
    first = np.zeros((2, 2, 4, 5), dtype=np.float32)
    second = np.zeros((2, 2, 4, 5), dtype=np.float32)
    first[0, 0] = 11
    first[0, 1] = 99
    first[1, 0] = 98
    first[1, 1] = 22
    second[0, 0] = 3
    second[0, 1] = 97
    second[1, 0] = 96
    second[1, 1] = 7

    composition = compose_aligned_image_payload(
        "MeasureColocalization",
        (first, second),
    )

    assert composition.execution_mode is ImagePayloadExecutionMode.ALIGNED_MULTI_IMAGE_STACK
    assert isinstance(composition.payload, AlignedImageStack)
    assert len(composition.payload.slices) == 2
    np.testing.assert_array_equal(
        composition.payload.slices[0],
        np.stack((np.full((4, 5), 11, dtype=np.float32), np.full((4, 5), 3, dtype=np.float32))),
    )
    np.testing.assert_array_equal(
        composition.payload.slices[1],
        np.stack((np.full((4, 5), 22, dtype=np.float32), np.full((4, 5), 7, dtype=np.float32))),
    )


def test_compose_image_bundle_promotes_grayscale_into_color_bundle():
    color = np.zeros((4, 5, 3), dtype=np.float32)
    color[:, :, 0] = 1
    grayscale = np.full((4, 5), 7, dtype=np.float32)

    bundle = compose_one_image_bundle((color, grayscale))

    assert bundle.shape == (2, 4, 5, 3)
    np.testing.assert_array_equal(bundle[0], color)
    np.testing.assert_array_equal(bundle[1, :, :, 0], grayscale)
    np.testing.assert_array_equal(bundle[1, :, :, 1], grayscale)
    np.testing.assert_array_equal(bundle[1, :, :, 2], grayscale)


def test_compose_image_bundle_collapses_singleton_grayscale_plane_stacks():
    singleton = np.full((1, 4, 5), 3, dtype=np.float32)
    plane = np.full((4, 5), 7, dtype=np.float32)

    bundle = compose_one_image_bundle((singleton, plane))

    assert bundle.shape == (2, 4, 5)
    np.testing.assert_array_equal(bundle[0], singleton[0])
    np.testing.assert_array_equal(bundle[1], plane)


def test_compose_image_bundle_intersects_masks() -> None:
    image_a = np.ones((4, 5), dtype=np.float32)
    image_b = np.full((4, 5), 2, dtype=np.float32)
    mask_a = np.array(
        (
            (True, False, True, True, True),
            (True, True, True, True, True),
            (False, True, True, True, True),
            (True, True, True, False, True),
        )
    )
    mask_b = np.array(
        (
            (True, True, True, False, True),
            (True, True, False, True, True),
            (True, True, True, True, True),
            (True, False, True, True, True),
        )
    )

    bundle = compose_one_image_bundle(
        (
            MaskedImagePayload(data=image_a, mask=mask_a),
            MaskedImagePayload(data=image_b, mask=mask_b),
        )
    )

    assert isinstance(bundle, MaskedImagePayload)
    assert bundle.data.shape == (2, 4, 5)
    np.testing.assert_array_equal(bundle.mask, mask_a & mask_b)


def test_compose_image_bundle_aligns_cropped_payload_to_source_domain() -> None:
    full = ImageMetadataPayload(
        data=np.full((4, 5), 2, dtype=np.float32),
        metadata=ImagePayloadMetadata(
            spatial_origin_yx=(0, 0),
            source_spatial_shape_yx=(4, 5),
        ),
    )
    cropped = ImageMetadataPayload(
        data=np.full((2, 2), 7, dtype=np.float32),
        metadata=ImagePayloadMetadata(
            spatial_origin_yx=(1, 2),
            source_spatial_shape_yx=(4, 5),
        ),
    )

    bundle = compose_one_image_bundle((full, cropped))

    assert isinstance(bundle, ImageMetadataPayload)
    assert bundle.metadata.spatial_origin_yx == (0, 0)
    assert bundle.metadata.source_spatial_shape_yx == (4, 5)
    assert bundle.data.shape == (2, 4, 5)
    np.testing.assert_array_equal(bundle.data[0], full.data)
    expected_cropped = np.zeros((4, 5), dtype=np.float32)
    expected_cropped[1:3, 2:4] = 7
    np.testing.assert_array_equal(bundle.data[1], expected_cropped)


def test_tile_preserves_color_stack_output_shape():
    image = np.zeros((2, 3, 4, 3), dtype=np.float32)
    image[0, :, :, 0] = 1
    image[1, :, :, 1] = 2

    output = tile(image, rows=1, columns=2, dtype_config=DtypeConfig())

    assert output.shape == (1, 3, 8, 3)
    np.testing.assert_array_equal(output[0, :, :4, 0], np.ones((3, 4)))
    np.testing.assert_array_equal(output[0, :, 4:, 1], np.full((3, 4), 2))


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
        ModuleArtifactContract(
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
    assert _recorded_measurements_for_assertion(runtime.measurements) == [
        (
            "ImageQuality",
            [
                {"mean": 1.0, "source_image_name": "OrigBlue", "slice_index": 0},
                {"mean": 2.0, "source_image_name": "OrigGreen", "slice_index": 0},
            ],
            {
                "source_image_name": None,
                "fields": ("slice_index", "mean", "source_image_name"),
            },
        )
    ]


def test_module_executor_runs_object_distribution_measurements_per_declared_image() -> None:
    calls = []

    def measure_distribution(
        image: np.ndarray,
        labels: np.ndarray,
    ) -> tuple[np.ndarray, list[dict[str, float | int]]]:
        calls.append((float(image[0, 0]), labels.copy()))
        return image, [
            {
                "object_label": 1,
                "mean": float(np.mean(image[labels > 0])),
            }
            ]

    measure_distribution.__processing_contract__ = ProcessingContract.PURE_2D
    labels = np.zeros((4, 5), dtype=np.int32)
    labels[1:3, 1:3] = 1
    fallback = np.zeros((4, 5), dtype=np.float32)
    runtime = _FakeCellProfilerRuntime(
        {
            "OrigBlue": _FakeRuntimeImage(np.ones((4, 5), dtype=np.float32)),
            "OrigGreen": _FakeRuntimeImage(np.full((4, 5), 2, dtype=np.float32)),
        },
        objects={
            "Cells": ObjectLabelSet(
                name="Cells",
                labels=labels,
            )
        },
    )
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="MeasureObjectIntensityDistribution",
            inputs=(
                ArtifactSpec("OrigBlue", ArtifactKind.IMAGE),
                ArtifactSpec("OrigGreen", ArtifactKind.IMAGE),
                ArtifactSpec("Cells", ArtifactKind.OBJECT_LABELS),
            ),
            runtime_artifact_inputs=(ArtifactSpec("Cells", ArtifactKind.OBJECT_LABELS),),
            outputs=(ArtifactSpec("MID", ArtifactKind.MEASUREMENTS),),
        )
    )

    result = executor.run(
        measure_distribution,
        fallback,
        cellprofiler_runtime=runtime,
    )

    assert result is fallback
    assert [call[0] for call in calls] == [1.0, 2.0]
    for _image_value, bound_labels in calls:
        np.testing.assert_array_equal(bound_labels, labels)
    assert _recorded_measurements_for_assertion(runtime.measurements) == [
        (
            "MID",
            [
                {
                    "object_label": 1,
                    "mean": 1.0,
                    "object_name": "Cells",
                    "source_image_name": "OrigBlue",
                    "slice_index": 0,
                },
                {
                    "object_label": 1,
                    "mean": 2.0,
                    "object_name": "Cells",
                    "source_image_name": "OrigGreen",
                    "slice_index": 0,
                },
            ],
            {
                "object_name": "Cells",
                "source_image_name": None,
                "fields": (
                    "slice_index",
                    "object_label",
                    "mean",
                    "object_name",
                    "source_image_name",
                ),
            },
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
        ModuleArtifactContract(
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
    assert _recorded_measurements_for_assertion(runtime.measurements) == [
        (
            "Colocalization",
            [{"delta": 2.0, "slice_index": 0}],
            {
                "object_name": None,
                "source_image_name": "OrigBlue__OrigGreen",
                "fields": ("slice_index", "delta"),
            },
        )
    ]


def test_colocalization_object_row_policy_projects_source_pair_features() -> None:
    policy = CellProfilerObjectMeasurementRowPolicy.for_module("MeasureColocalization")
    measurement_image = CellProfilerMeasurementImage(
        source_image_name="DNA__ER__RNA",
        source_image_names=("DNA", "ER", "RNA"),
        payload=np.zeros((3, 4, 5), dtype=np.float32),
    )

    invocations = policy.invocations(measurement_image, {"do_manders": True})

    assert [invocation.kwargs["channel_1"] for invocation in invocations] == [0, 0, 1]
    assert [invocation.kwargs["channel_2"] for invocation in invocations] == [1, 2, 2]
    assert [invocation.source_pair.first_name for invocation in invocations] == [
        "DNA",
        "DNA",
        "ER",
    ]
    projected = policy.project_rows(
        [
            {
                "slice_index": 0,
                "object_label": 1,
                "correlation": 0.5,
                "manders_m1": 0.7,
                "manders_m2": 0.8,
                "costes_threshold_1": 42.0,
            }
        ],
        invocations[0],
    )

    assert projected == [
        {
            "slice_index": 0,
            "object_label": 1,
            "Correlation_Correlation_DNA_ER": 0.5,
            "Correlation_Manders_DNA_ER": 0.7,
            "Correlation_Manders_ER_DNA": 0.8,
        }
    ]
    assert policy.table_source_image_name((measurement_image,), "DNA__ER__RNA") is None


def test_colocalization_record_builder_projects_image_source_pair_features() -> None:
    def measure_colocalization(image: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        return image, {"correlation": 0.5}

    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="MeasureColocalization",
            inputs=(
                ArtifactSpec("DNA", ArtifactKind.IMAGE),
                ArtifactSpec("ER", ArtifactKind.IMAGE),
            ),
            runtime_artifact_inputs=(
                ArtifactSpec("DNA", ArtifactKind.IMAGE),
                ArtifactSpec("ER", ArtifactKind.IMAGE),
            ),
            outputs=(ArtifactSpec("Coloc", ArtifactKind.MEASUREMENTS),),
        )
    )

    record = CellProfilerMeasurementRecordBuilder.for_module(
        "MeasureColocalization"
    ).build(
        CellProfilerOutputRecordRequest(
            executor=executor,
            adapter=None,
            spec=ArtifactSpec("Coloc", ArtifactKind.MEASUREMENTS),
            value={"slice_index": 0, "correlation": 0.5, "manders_m1": 0.7},
            output_values={"Coloc": {"correlation": 0.5}},
            source_image_name="DNA__ER",
            func=measure_colocalization,
            source_image_names=("DNA", "ER"),
        )
    )

    assert record.rows == [
        {
            "slice_index": 0,
            "Correlation_Correlation_DNA_ER": 0.5,
            "Correlation_Manders_DNA_ER": 0.7,
        }
    ]
    assert record.source_image_name is None


def test_measure_object_neighbors_records_object_topology_without_image_source() -> None:
    def measure_neighbors(image: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        return image, {"number_of_neighbors": 1.0}

    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="MeasureObjectNeighbors",
            inputs=(ArtifactSpec("Nuclei", ArtifactKind.OBJECT_LABELS),),
            runtime_artifact_inputs=(ArtifactSpec("Nuclei", ArtifactKind.OBJECT_LABELS),),
            outputs=(ArtifactSpec("Neighbors", ArtifactKind.MEASUREMENTS),),
        )
    )

    record = CellProfilerMeasurementRecordBuilder.for_module(
        "MeasureObjectNeighbors"
    ).build(
        CellProfilerOutputRecordRequest(
            executor=executor,
            adapter=None,
            spec=ArtifactSpec("Neighbors", ArtifactKind.MEASUREMENTS),
            value={"number_of_neighbors": 1.0},
            output_values={"Neighbors": {"number_of_neighbors": 1.0}},
            source_image_name="OrigBlue",
            func=measure_neighbors,
        )
    )

    assert record.object_name == "Nuclei"
    assert record.source_image_name is None


def test_track_objects_record_builder_uses_nominal_image_table_ownership() -> None:
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="TrackObjects",
            inputs=(ArtifactSpec("Embryos", ArtifactKind.OBJECT_LABELS),),
            runtime_artifact_inputs=(ArtifactSpec("Embryos", ArtifactKind.OBJECT_LABELS),),
            outputs=(ArtifactSpec("Tracking", ArtifactKind.MEASUREMENTS),),
        )
    )

    record = CellProfilerMeasurementRecordBuilder.for_module("TrackObjects").build(
        CellProfilerOutputRecordRequest(
            executor=executor,
            adapter=None,
            spec=ArtifactSpec("Tracking", ArtifactKind.MEASUREMENTS),
            value=[
                {
                    "image_number": 1,
                    "object_label": 1,
                    "feature_name": "TrackObjects_Label_50",
                    "measurement_value": 1,
                },
                {
                    "image_number": 1,
                    "feature_name": "TrackObjects_NewObjectCount_Embryos_50",
                    "measurement_value": 1,
                },
            ],
            output_values={},
            source_image_name="OrigColor",
            func=_synthetic_object_measurement_function,
        )
    )

    object_row, image_row = record.rows
    assert object_row["object_name"] == "Embryos"
    assert "source_image_name" not in object_row
    assert image_row["source_image_name"] == "image"
    assert "object_name" not in image_row


def test_object_label_output_recorder_uses_output_label_domain() -> None:
    input_labels = np.zeros((5, 5), dtype=np.int32)
    input_labels[1, 1] = 1
    input_labels[2, 2] = 4
    input_payload = ObjectLabelPayload(
        labels=input_labels,
        declared_object_count=9,
    )
    output_labels = input_labels.copy()
    output_payload = ObjectLabelPayload(
        labels=output_labels,
        declared_object_ids=tuple(range(1, 5)),
    )
    runtime = _FakeCellProfilerRuntime(
        {},
        objects={
            "InputObjects": ObjectLabelSet(
                name="InputObjects",
                labels=input_payload,
            )
        },
    )
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="ExpandOrShrinkObjects",
            inputs=(ArtifactSpec("InputObjects", ArtifactKind.OBJECT_LABELS),),
            runtime_artifact_inputs=(
                ArtifactSpec("InputObjects", ArtifactKind.OBJECT_LABELS),
            ),
            outputs=(ArtifactSpec("ExpandedObjects", ArtifactKind.OBJECT_LABELS),),
        )
    )

    CellProfilerOutputRecorder.for_kind(ArtifactKind.OBJECT_LABELS).record(
        CellProfilerOutputRecordRequest(
            executor=executor,
            adapter=runtime,
            spec=ArtifactSpec("ExpandedObjects", ArtifactKind.OBJECT_LABELS),
            value=output_payload,
            output_values={"ExpandedObjects": output_payload},
            source_image_name=None,
            func=lambda image: image,
        )
    )

    _name, recorded_payload, _kwargs = runtime.objects[0]
    assert isinstance(recorded_payload, ObjectLabelPayload)
    assert recorded_payload.declared_object_count is None
    assert recorded_payload.declared_object_ids == tuple(range(1, 5))
    np.testing.assert_array_equal(recorded_payload.labels, output_labels)


def test_expand_or_shrink_executor_declares_output_label_extent() -> None:
    input_labels = np.zeros((7, 7), dtype=np.int32)
    input_labels[3, 3] = 4
    input_payload = ObjectLabelPayload(
        labels=input_labels,
        declared_object_count=9,
    )
    runtime = _FakeCellProfilerRuntime(
        {},
        objects={
            "InputObjects": ObjectLabelSet(
                name="InputObjects",
                labels=input_payload,
            )
        },
    )
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="ExpandOrShrinkObjects",
            inputs=(ArtifactSpec("InputObjects", ArtifactKind.OBJECT_LABELS),),
            runtime_artifact_inputs=(
                ArtifactSpec("InputObjects", ArtifactKind.OBJECT_LABELS),
            ),
            outputs=(ArtifactSpec("ExpandedObjects", ArtifactKind.OBJECT_LABELS),),
        )
    )

    executor.run(
        expand_or_shrink_objects,
        np.zeros_like(input_labels, dtype=np.float32),
        cellprofiler_runtime=runtime,
        mode="expand_defined_pixels",
        iterations=1,
        dtype_config=DtypeConfig(),
    )

    _name, recorded_payload, _kwargs = runtime.objects[0]
    assert isinstance(recorded_payload, ObjectLabelPayload)
    assert recorded_payload.declared_object_count == 4
    assert recorded_payload.declared_object_ids == ()
    assert int(np.max(recorded_payload.labels)) == 4


def test_align_measurement_builder_records_output_scoped_shifts() -> None:
    def align_function(image: np.ndarray) -> tuple[
        np.ndarray,
        np.ndarray,
        tuple[AlignShiftMeasurement, AlignShiftMeasurement],
    ]:
        return (
            image[0],
            image[1],
            (
                AlignShiftMeasurement(0, 0, 0.0, 0.0),
                AlignShiftMeasurement(0, 1, -1.0, 1.0),
            ),
        )

    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="Align",
            inputs=(
                ArtifactSpec("Stain1Raw", ArtifactKind.IMAGE),
                ArtifactSpec("Stain2Raw", ArtifactKind.IMAGE),
            ),
            outputs=(
                ArtifactSpec("Stain1", ArtifactKind.IMAGE),
                ArtifactSpec("Stain2", ArtifactKind.IMAGE),
                ArtifactSpec("AlignMeasurements", ArtifactKind.MEASUREMENTS),
            ),
        )
    )

    record = CellProfilerMeasurementRecordBuilder.for_module("Align").build(
        CellProfilerOutputRecordRequest(
            executor=executor,
            adapter=None,
            spec=ArtifactSpec("AlignMeasurements", ArtifactKind.MEASUREMENTS),
            value=(
                AlignShiftMeasurement(0, 0, 0.0, 0.0),
                AlignShiftMeasurement(0, 1, -1.0, 1.0),
            ),
            output_values={},
            source_image_name="Stain1Raw__Stain2Raw",
            func=align_function,
        )
    )

    assert record.object_name is None
    assert record.source_image_name is None
    assert record.rows == [
        {
            "slice_index": 0,
            "source_image_name": "Stain1",
            "feature_name": AlignMeasurementFeature.X_SHIFT.value,
            "result_value": 0.0,
        },
        {
            "slice_index": 0,
            "source_image_name": "Stain1",
            "feature_name": AlignMeasurementFeature.Y_SHIFT.value,
            "result_value": 0.0,
        },
        {
            "slice_index": 0,
            "source_image_name": "Stain2",
            "feature_name": AlignMeasurementFeature.X_SHIFT.value,
            "result_value": -1.0,
        },
        {
            "slice_index": 0,
            "source_image_name": "Stain2",
            "feature_name": AlignMeasurementFeature.Y_SHIFT.value,
            "result_value": 1.0,
        },
    ]


def test_align_measurement_builder_records_additional_output_shifts() -> None:
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="Align",
            inputs=(
                ArtifactSpec("Template", ArtifactKind.IMAGE),
                ArtifactSpec("Red", ArtifactKind.IMAGE),
                ArtifactSpec("Combined", ArtifactKind.IMAGE),
            ),
            outputs=(
                ArtifactSpec("AlignedTemplate", ArtifactKind.IMAGE),
                ArtifactSpec("AlignedRed", ArtifactKind.IMAGE),
                ArtifactSpec("AlignedCombined", ArtifactKind.IMAGE),
                ArtifactSpec("AlignMeasurements", ArtifactKind.MEASUREMENTS),
            ),
        )
    )

    record = CellProfilerMeasurementRecordBuilder.for_module("Align").build(
        CellProfilerOutputRecordRequest(
            executor=executor,
            adapter=None,
            spec=ArtifactSpec("AlignMeasurements", ArtifactKind.MEASUREMENTS),
            value=(
                AlignShiftMeasurement(0, 0, 0.0, 0.0),
                AlignShiftMeasurement(0, 1, -2.0, 1.0),
                AlignShiftMeasurement(0, 2, -2.0, 1.0),
            ),
            output_values={},
            source_image_name=None,
            func=lambda image: image,
        )
    )

    assert record.rows[-2:] == [
        {
            "slice_index": 0,
            "source_image_name": "AlignedCombined",
            "feature_name": AlignMeasurementFeature.X_SHIFT.value,
            "result_value": -2.0,
        },
        {
            "slice_index": 0,
            "source_image_name": "AlignedCombined",
            "feature_name": AlignMeasurementFeature.Y_SHIFT.value,
            "result_value": 1.0,
        },
    ]


def test_align_measurement_builder_uses_declared_outputs_when_images_pruned() -> None:
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="Align",
            inputs=(
                ArtifactSpec("Plate", ArtifactKind.IMAGE),
                ArtifactSpec("Well", ArtifactKind.IMAGE),
            ),
            outputs=(ArtifactSpec("AlignMeasurements", ArtifactKind.MEASUREMENTS),),
            declared_outputs=(
                ArtifactSpec("AlignedPlate", ArtifactKind.IMAGE),
                ArtifactSpec("AlignedWell", ArtifactKind.IMAGE),
                ArtifactSpec("AlignMeasurements", ArtifactKind.MEASUREMENTS),
            ),
        )
    )

    record = CellProfilerMeasurementRecordBuilder.for_module("Align").build(
        CellProfilerOutputRecordRequest(
            executor=executor,
            adapter=None,
            spec=ArtifactSpec("AlignMeasurements", ArtifactKind.MEASUREMENTS),
            value=(
                AlignShiftMeasurement(0, 0, 0.0, 0.0),
                AlignShiftMeasurement(0, 1, -3.0, 2.0),
            ),
            output_values={},
            source_image_name=None,
            func=lambda image: image,
        )
    )

    assert record.rows[-2:] == [
        {
            "slice_index": 0,
            "source_image_name": "AlignedWell",
            "feature_name": AlignMeasurementFeature.X_SHIFT.value,
            "result_value": -3.0,
        },
        {
            "slice_index": 0,
            "source_image_name": "AlignedWell",
            "feature_name": AlignMeasurementFeature.Y_SHIFT.value,
            "result_value": 2.0,
        },
    ]


def test_measure_object_neighbors_binds_small_removed_label_variant() -> None:
    calls = []

    def measure_neighbors(
        image: np.ndarray,
        labels: np.ndarray,
        small_removed_labels: np.ndarray | None = None,
        neighbor_labels: np.ndarray | None = None,
        small_removed_neighbor_labels: np.ndarray | None = None,
        neighbors_are_same_objects: bool = False,
    ) -> tuple[np.ndarray, list[object]]:
        calls.append(
            (
                labels.copy(),
                None if small_removed_labels is None else small_removed_labels.copy(),
                neighbor_labels,
                small_removed_neighbor_labels,
                neighbors_are_same_objects,
            )
        )
        return image, []

    measure_neighbors.__processing_contract__ = ProcessingContract.FLEXIBLE
    final_labels = np.zeros((4, 4), dtype=np.int32)
    final_labels[1, 1] = 1
    small_removed = final_labels.copy()
    small_removed[1, 2] = 2
    fallback = np.zeros((4, 4), dtype=np.float32)
    runtime = _FakeCellProfilerRuntime(
        {},
        objects={
            "Nuclei": ObjectLabelSet(
                name="Nuclei",
                labels=final_labels,
                small_removed_labels=small_removed,
            )
        },
    )
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="MeasureObjectNeighbors",
            inputs=(ArtifactSpec("Nuclei", ArtifactKind.OBJECT_LABELS),),
            runtime_artifact_inputs=(ArtifactSpec("Nuclei", ArtifactKind.OBJECT_LABELS),),
            outputs=(ArtifactSpec("Neighbors", ArtifactKind.MEASUREMENTS),),
        )
    )

    result = executor.run(
        measure_neighbors,
        fallback,
        cellprofiler_runtime=runtime,
    )

    assert result is fallback
    bound_labels, bound_small_removed, bound_neighbor, bound_small_neighbor, same = calls[0]
    np.testing.assert_array_equal(bound_labels, final_labels)
    np.testing.assert_array_equal(bound_small_removed, small_removed)
    assert bound_neighbor is None
    assert bound_small_neighbor is None
    assert same is True


def test_classification_rows_include_unclassified_objects() -> None:
    def classify_like(image: np.ndarray) -> tuple[np.ndarray, ClassificationResult]:
        return image, ClassificationResult(
            slice_index=0,
            total_objects=3,
            bin_counts='{"Small": 1, "Large": 1}',
            bin_percentages='{"Small": 33.3333333333, "Large": 33.3333333333}',
            object_classes='{"1": "Small", "3": "Large"}',
        )

    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="ClassifyObjectsSingleMeasurement",
            inputs=(ArtifactSpec("Nuclei", ArtifactKind.OBJECT_LABELS),),
            runtime_artifact_inputs=(ArtifactSpec("Nuclei", ArtifactKind.OBJECT_LABELS),),
            outputs=(ArtifactSpec("ClassifyObjects", ArtifactKind.MEASUREMENTS),),
        )
    )

    rows = CellProfilerMeasurementRecordBuilder.for_module(
        "ClassifyObjectsSingleMeasurement"
    ).build(
        CellProfilerOutputRecordRequest(
            executor=executor,
            adapter=None,
            spec=ArtifactSpec("ClassifyObjects", ArtifactKind.MEASUREMENTS),
            value=classify_like(np.zeros((2, 2), dtype=np.float32))[1],
            output_values={},
            source_image_name=None,
            func=classify_like,
        )
    ).rows

    object_rows = [
        row for row in rows
        if row.get("object_name") == "Nuclei"
    ]
    assert len(object_rows) == 6
    assert {
        (row["object_label"], row["feature_name"], row["result_value"])
        for row in object_rows
    } == {
        (
            1,
            ClassifyObjectsMeasurementFeatureTemplate.OBJECT_CLASS.feature_name(
                bin_name="Small"
            ),
            1,
        ),
        (
            1,
            ClassifyObjectsMeasurementFeatureTemplate.OBJECT_CLASS.feature_name(
                bin_name="Large"
            ),
            0,
        ),
        (
            2,
            ClassifyObjectsMeasurementFeatureTemplate.OBJECT_CLASS.feature_name(
                bin_name="Small"
            ),
            0,
        ),
        (
            2,
            ClassifyObjectsMeasurementFeatureTemplate.OBJECT_CLASS.feature_name(
                bin_name="Large"
            ),
            0,
        ),
        (
            3,
            ClassifyObjectsMeasurementFeatureTemplate.OBJECT_CLASS.feature_name(
                bin_name="Small"
            ),
            0,
        ),
        (
            3,
            ClassifyObjectsMeasurementFeatureTemplate.OBJECT_CLASS.feature_name(
                bin_name="Large"
            ),
            1,
        ),
    }


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
        ModuleArtifactContract(
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
    assert _recorded_measurements_for_assertion(runtime.measurements) == [
        (
            "UntangleWorms_3_measurements",
            [{"worm_count": 1.0, "slice_index": 0}],
            {
                "object_name": None,
                "source_image_name": "WormBinary",
                "fields": ("slice_index", "worm_count"),
            },
        )
    ]
    assert [name for name, _labels, _kwargs in runtime.objects] == [
        "OverlappingWorms",
        "NonOverlappingWorms",
    ]
    np.testing.assert_array_equal(runtime.objects[0][1], labels_with_overlap)
    np.testing.assert_array_equal(runtime.objects[1][1], labels_without_overlap)


def test_default_measurement_builder_preserves_row_declared_object_scope() -> None:
    image = np.zeros((4, 5), dtype=np.float32)

    def object_rows(image: np.ndarray) -> tuple[np.ndarray, list[dict[str, object]]]:
        return image, [
            {
                "object_name": "Worms",
                "object_number": 1,
                "worm_length": 10.0,
            }
        ]

    object_rows.__processing_contract__ = ProcessingContract.PURE_2D
    runtime = _FakeCellProfilerRuntime(
        {
            "WormBinary": _FakeRuntimeImage(
                image,
                source_image_name="WormBinary",
            ),
        }
    )
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="UntangleWorms",
            inputs=(ArtifactSpec("WormBinary", ArtifactKind.IMAGE),),
            outputs=(
                ArtifactSpec("UntangleWorms_3_measurements", ArtifactKind.MEASUREMENTS),
            ),
        )
    )

    executor.run(
        object_rows,
        image,
        cellprofiler_runtime=runtime,
    )

    assert _recorded_measurements_for_assertion(runtime.measurements) == [
        (
            "UntangleWorms_3_measurements",
            [
                {
                    "object_name": "Worms",
                    "object_number": 1,
                    "worm_length": 10.0,
                    "slice_index": 0,
                }
            ],
            {
                "object_name": None,
                "source_image_name": None,
                "fields": (
                    "slice_index",
                    "object_name",
                    "object_number",
                    "worm_length",
                ),
            },
        )
    ]


def test_module_executor_routes_spatial_grid_artifacts() -> None:
    image = np.zeros((20, 20), dtype=np.float32)
    grid = SpatialGrid(
        name="grid_info",
        rows=2,
        columns=2,
        x_spacing=8.0,
        y_spacing=8.0,
        x_origin=4.0,
        y_origin=4.0,
    )
    runtime = _FakeCellProfilerRuntime({"DNA": _FakeRuntimeImage(image)})
    define_executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="DefineGrid",
            inputs=(ArtifactSpec("DNA", ArtifactKind.IMAGE),),
            outputs=(ArtifactSpec("Grid", ArtifactKind.SPATIAL_GRID),),
        )
    )
    identify_executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="IdentifyObjectsInGrid",
            inputs=(ArtifactSpec("Grid", ArtifactKind.SPATIAL_GRID),),
            runtime_artifact_inputs=(
                ArtifactSpec("Grid", ArtifactKind.SPATIAL_GRID),
            ),
            outputs=(ArtifactSpec("GridObjects", ArtifactKind.OBJECT_LABELS),),
        )
    )

    def define_grid_like(image: np.ndarray) -> tuple[np.ndarray, SpatialGrid]:
        return image, grid

    define_grid_like.__processing_contract__ = ProcessingContract.PURE_2D

    @special_inputs("grid")
    def identify_grid_like(
        image: np.ndarray,
        grid: SpatialGrid,
    ) -> tuple[np.ndarray, np.ndarray]:
        labels = np.full(image.shape, grid.rows * grid.columns, dtype=np.int32)
        return image, labels

    identify_grid_like.__processing_contract__ = ProcessingContract.PURE_2D

    define_executor.run(define_grid_like, image, cellprofiler_runtime=runtime)
    identify_executor.run(identify_grid_like, image, cellprofiler_runtime=runtime)

    assert runtime.spatial_grids["Grid"].rows == 2
    assert runtime.spatial_grids["Grid"].columns == 2
    assert [name for name, _labels, _kwargs in runtime.objects] == ["GridObjects"]
    np.testing.assert_array_equal(runtime.objects[0][1], np.full(image.shape, 4))


def test_define_grid_manual_once_scope_executes_once_for_stacked_image() -> None:
    image = np.zeros((3, 20, 20), dtype=np.float32)
    grid = SpatialGrid(
        name="grid_info",
        rows=2,
        columns=2,
        x_spacing=8.0,
        y_spacing=8.0,
        x_origin=4.0,
        y_origin=4.0,
    )
    runtime = _FakeCellProfilerRuntime({"DNA": _FakeRuntimeImage(image)})
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="DefineGridManual",
            inputs=(ArtifactSpec("DNA", ArtifactKind.IMAGE),),
            outputs=(ArtifactSpec("Grid", ArtifactKind.SPATIAL_GRID),),
        )
    )
    calls = 0

    def define_grid_like(image: np.ndarray) -> tuple[np.ndarray, SpatialGrid]:
        nonlocal calls
        calls += 1
        return image, grid

    attach_callable_contract_metadata(
        define_grid_like,
        declared_processing_contract="pure_2d",
    )

    executor.run(
        define_grid_like,
        image,
        cellprofiler_runtime=runtime,
        invocation_options=CellProfilerInvocationOptions(
            grid_cycle_scope=CellProfilerGridCycleScope.ONCE
        ),
    )

    assert calls == 1
    assert runtime.spatial_grids["Grid"].rows == 2
    assert runtime.spatial_grids["Grid"].columns == 2


def test_define_grid_manual_each_cycle_scope_emits_slice_aligned_grids() -> None:
    image = np.stack(
        [
            np.full((20, 20), 1, dtype=np.float32),
            np.full((20, 20), 2, dtype=np.float32),
            np.full((20, 20), 3, dtype=np.float32),
        ],
        axis=0,
    )
    runtime = _FakeCellProfilerRuntime({"DNA": _FakeRuntimeImage(image)})
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="DefineGridManual",
            inputs=(ArtifactSpec("DNA", ArtifactKind.IMAGE),),
            outputs=(ArtifactSpec("Grid", ArtifactKind.SPATIAL_GRID),),
        )
    )
    calls = 0

    def define_grid_like(image: np.ndarray) -> tuple[np.ndarray, SpatialGrid]:
        nonlocal calls
        calls += 1
        return image, SpatialGrid(
            name="grid_info",
            rows=2,
            columns=2,
            x_spacing=8.0,
            y_spacing=8.0,
            x_origin=float(image[0, 0]),
            y_origin=4.0,
        )

    attach_callable_contract_metadata(
        define_grid_like,
        declared_processing_contract="pure_2d",
    )

    executor.run(
        define_grid_like,
        image,
        cellprofiler_runtime=runtime,
        invocation_options=CellProfilerInvocationOptions(
            grid_cycle_scope=CellProfilerGridCycleScope.EACH_CYCLE
        ),
    )

    assert calls == 3
    grids = runtime.spatial_grids["Grid"]
    assert isinstance(grids, RuntimeSliceAlignedValues)
    assert [grid.x_origin for grid in grids.slices] == [1.0, 2.0, 3.0]


def test_grid_only_module_uses_single_carrier_plane_for_stacked_image() -> None:
    image = np.zeros((3, 20, 20), dtype=np.float32)
    grid = SpatialGrid(
        name="Grid",
        rows=2,
        columns=2,
        x_spacing=8.0,
        y_spacing=8.0,
        x_origin=4.0,
        y_origin=4.0,
    )
    runtime = _FakeCellProfilerRuntime({"DNA": _FakeRuntimeImage(image)})
    runtime.spatial_grids["Grid"] = grid
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="IdentifyObjectsInGrid",
            inputs=(ArtifactSpec("Grid", ArtifactKind.SPATIAL_GRID),),
            runtime_artifact_inputs=(
                ArtifactSpec("Grid", ArtifactKind.SPATIAL_GRID),
            ),
            outputs=(ArtifactSpec("GridObjects", ArtifactKind.OBJECT_LABELS),),
        )
    )

    @special_inputs("grid")
    def identify_grid_like(
        image: np.ndarray,
        grid: SpatialGrid,
    ) -> tuple[np.ndarray, np.ndarray]:
        return image, np.full(image.shape, grid.rows * grid.columns, dtype=np.int32)

    attach_callable_contract_metadata(
        identify_grid_like,
        declared_processing_contract="pure_2d",
    )

    executor.run(identify_grid_like, image, cellprofiler_runtime=runtime)

    assert len(runtime.objects) == 1
    assert runtime.objects[0][1].shape == (20, 20)


def test_grid_input_module_slices_runtime_aligned_grid_for_2d_carrier() -> None:
    image = np.zeros((20, 20), dtype=np.float32)
    runtime = _FakeCellProfilerRuntime({"DNA": _FakeRuntimeImage(image)})
    runtime.spatial_grids["Grid"] = RuntimeSliceAlignedValues(
        slices=(
            SpatialGrid(
                name="Grid",
                rows=2,
                columns=2,
                x_spacing=8.0,
                y_spacing=8.0,
                x_origin=1.0,
                y_origin=4.0,
            ),
            SpatialGrid(
                name="Grid",
                rows=2,
                columns=2,
                x_spacing=8.0,
                y_spacing=8.0,
                x_origin=2.0,
                y_origin=4.0,
            ),
        )
    )
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="IdentifyObjectsInGrid",
            inputs=(ArtifactSpec("Grid", ArtifactKind.SPATIAL_GRID),),
            runtime_artifact_inputs=(
                ArtifactSpec("Grid", ArtifactKind.SPATIAL_GRID),
            ),
            outputs=(ArtifactSpec("GridObjects", ArtifactKind.OBJECT_LABELS),),
        )
    )
    seen_origins: list[float] = []

    @special_inputs("grid")
    def identify_grid_like(
        image: np.ndarray,
        grid: SpatialGrid,
    ) -> tuple[np.ndarray, np.ndarray]:
        seen_origins.append(grid.x_origin)
        return image, np.full(image.shape, grid.x_origin, dtype=np.int32)

    attach_callable_contract_metadata(
        identify_grid_like,
        declared_processing_contract="pure_2d",
    )

    executor.run(identify_grid_like, image, cellprofiler_runtime=runtime)

    assert seen_origins == [1.0, 2.0]
    assert runtime.objects[0][1].shape == (2, 20, 20)


def test_grid_input_module_unwraps_singleton_runtime_aligned_grid_for_2d_carrier() -> None:
    image = np.zeros((20, 20), dtype=np.float32)
    runtime = _FakeCellProfilerRuntime({"DNA": _FakeRuntimeImage(image)})
    runtime.spatial_grids["Grid"] = RuntimeSliceAlignedValues(
        slices=(
            SpatialGrid(
                name="Grid",
                rows=2,
                columns=2,
                x_spacing=8.0,
                y_spacing=8.0,
                x_origin=1.0,
                y_origin=4.0,
            ),
        )
    )
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="IdentifyObjectsInGrid",
            inputs=(ArtifactSpec("Grid", ArtifactKind.SPATIAL_GRID),),
            runtime_artifact_inputs=(
                ArtifactSpec("Grid", ArtifactKind.SPATIAL_GRID),
            ),
            outputs=(ArtifactSpec("GridObjects", ArtifactKind.OBJECT_LABELS),),
        )
    )
    seen_origins: list[float] = []

    @special_inputs("grid")
    def identify_grid_like(
        image: np.ndarray,
        grid: SpatialGrid,
    ) -> tuple[np.ndarray, np.ndarray]:
        seen_origins.append(grid.x_origin)
        return image, np.full(image.shape, grid.x_origin, dtype=np.int32)

    attach_callable_contract_metadata(
        identify_grid_like,
        declared_processing_contract="pure_2d",
    )

    executor.run(identify_grid_like, image, cellprofiler_runtime=runtime)

    assert seen_origins == [1.0]
    assert runtime.objects[0][1].shape == (1, 20, 20)


def test_flexible_object_module_slices_tuple_label_stack() -> None:
    image = np.zeros((3, 6, 6), dtype=np.float32)
    labels = np.zeros((3, 6, 6), dtype=np.int32)
    labels[:, 1:4, 1:4] = np.arange(1, 4, dtype=np.int32)[:, None, None]
    runtime = _FakeCellProfilerRuntime(
        {"Carrier": _FakeRuntimeImage(image)},
        objects={
            "Cells": ObjectLabelSet(
                name="Cells",
                labels=labels,
            )
        },
    )
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="FilterObjects",
            inputs=(ArtifactSpec("Cells", ArtifactKind.OBJECT_LABELS),),
            runtime_artifact_inputs=(ArtifactSpec("Cells", ArtifactKind.OBJECT_LABELS),),
            outputs=(ArtifactSpec("FilteredCells", ArtifactKind.OBJECT_LABELS),),
        )
    )
    calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def filter_like(
        image: np.ndarray,
        object_labels: tuple[np.ndarray, ...] = (),
    ) -> tuple[np.ndarray, np.ndarray]:
        calls.append((tuple(image.shape), tuple(object_labels[0].shape)))
        return image, object_labels[0]

    attach_callable_contract_metadata(
        filter_like,
        declared_processing_contract="flexible",
    )

    executor.run(filter_like, image, cellprofiler_runtime=runtime)

    assert calls == [((6, 6), (6, 6)), ((6, 6), (6, 6)), ((6, 6), (6, 6))]
    assert runtime.objects[0][1].shape == (3, 6, 6)


def test_object_only_reference_image_collapses_payload_stack() -> None:
    payload = image_payload_with_context(
        np.zeros((4, 6, 6), dtype=np.float32),
        metadata=ImagePayloadMetadata(source_dtype="float32"),
    )

    reference = _object_only_reference_image(payload)

    assert reference.shape == (6, 6)


def test_object_only_reference_image_collapses_aligned_stack() -> None:
    payload = AlignedImageStack(
        (
            image_payload_with_context(
                np.zeros((6, 6), dtype=np.float32),
                metadata=ImagePayloadMetadata(source_dtype="float32"),
            ),
            image_payload_with_context(
                np.ones((6, 6), dtype=np.float32),
                metadata=ImagePayloadMetadata(source_dtype="float32"),
            ),
        )
    )

    reference = _object_only_reference_image(payload)

    assert reference.shape == (6, 6)
    assert np.all(reference == 0)


def test_flexible_object_module_slices_measurement_tables_with_label_stack() -> None:
    image = np.zeros((2, 6, 6), dtype=np.float32)
    labels = np.zeros((2, 6, 6), dtype=np.int32)
    labels[0, 1:3, 1:3] = 1
    labels[1, 3:5, 3:5] = 1
    measurements = MeasurementTable(
        name="CellShape",
        object_name="Cells",
        object_id_field="object_label",
        rows=[
            {"slice_index": 0, "object_label": 1, "Area": 4.0},
            {"slice_index": 1, "object_label": 1, "Area": 9.0},
        ],
    )
    relationship_measurements = MeasurementTable(
        name="RelationshipFacts",
        object_name="Cells",
        object_id_field="object_label",
        rows=[
            {"slice_index": 999, "object_label": 1, "Children_Count": 1},
        ],
    )
    runtime = _FakeCellProfilerRuntime(
        {"Carrier": _FakeRuntimeImage(image)},
        objects={
            "Cells": ObjectLabelSet(
                name="Cells",
                labels=labels,
            )
        },
        measurement_tables={"Cells": (measurements, relationship_measurements)},
    )
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="FilterObjects",
            inputs=(ArtifactSpec("Cells", ArtifactKind.OBJECT_LABELS),),
            runtime_artifact_inputs=(ArtifactSpec("Cells", ArtifactKind.OBJECT_LABELS),),
            outputs=(ArtifactSpec("FilteredCells", ArtifactKind.OBJECT_LABELS),),
        )
    )
    seen_areas: list[tuple[float, ...]] = []

    def filter_like(
        image: np.ndarray,
        object_labels: tuple[np.ndarray, ...] = (),
        measurement_tables: tuple[MeasurementTable, ...] = (),
    ) -> tuple[np.ndarray, np.ndarray]:
        del object_labels
        seen_areas.append(
            tuple(
                float(row["Area"])
                for table in measurement_tables
                for row in table.rows
            )
        )
        return image, np.zeros(image.shape, dtype=np.int32)

    attach_callable_contract_metadata(
        filter_like,
        declared_processing_contract="flexible",
    )

    executor.run(filter_like, image, cellprofiler_runtime=runtime)

    assert seen_areas == [(4.0,), (9.0,)]


def test_filterobjects_binds_selection_measurement_values_to_label_slices() -> None:
    image = np.zeros((2, 6, 6), dtype=np.float32)
    children = np.zeros((2, 6, 6), dtype=np.int32)
    children[:, 0:2, 0:2] = 1
    children[:, 3:5, 3:5] = 2
    parents = np.ones_like(children)
    measurements = MeasurementTable(
        name="ChildMeasurements",
        object_name="Cells",
        object_id_field="object_label",
        rows=[
            {"slice_index": 0, "object_label": 1, "AreaShape_Area": 10.0},
            {"slice_index": 0, "object_label": 2, "AreaShape_Area": 20.0},
            {"slice_index": 1, "object_label": 1, "AreaShape_Area": 30.0},
            {"slice_index": 1, "object_label": 2, "AreaShape_Area": 5.0},
        ],
    )
    runtime = _FakeCellProfilerRuntime(
        {"Carrier": _FakeRuntimeImage(image)},
        objects={
            "Cells": ObjectLabelSet(name="Cells", labels=children),
            "Tiles": ObjectLabelSet(name="Tiles", labels=parents),
        },
        measurement_tables={"Cells": (measurements,)},
    )
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="FilterObjects",
            inputs=(
                ArtifactSpec("Cells", ArtifactKind.OBJECT_LABELS),
                ArtifactSpec("Tiles", ArtifactKind.OBJECT_LABELS),
            ),
            runtime_artifact_inputs=(
                ArtifactSpec("Cells", ArtifactKind.OBJECT_LABELS),
                ArtifactSpec("Tiles", ArtifactKind.OBJECT_LABELS),
            ),
            outputs=(
                ArtifactSpec("FilterObjects_measurements", ArtifactKind.MEASUREMENTS),
                ArtifactSpec("FilteredCells", ArtifactKind.OBJECT_LABELS),
                ArtifactSpec(
                    "Cells_FilteredCells_relationships",
                    ArtifactKind.RELATIONSHIPS,
                ),
            ),
        )
    )

    executor.run(
        filter_objects,
        image,
        cellprofiler_runtime=runtime,
        mode=FilterMode.MEASUREMENTS,
        filter_method=FilterMethod.MAXIMAL_PER_OBJECT,
        measurement_features=("AreaShape_Area",),
        enclosing_object_name="Tiles",
        per_object_assignment=PerObjectAssignment.BOTH_PARENTS,
    )

    filtered = next(value for name, value, _kwargs in runtime.objects if name == "FilteredCells")
    assert filtered.shape == (2, 6, 6)
    assert filtered[0, 0, 0] == 0
    assert filtered[0, 3, 3] == 1
    assert filtered[1, 0, 0] == 1
    assert filtered[1, 3, 3] == 0


def test_flexible_object_module_slices_measurement_tables_with_2d_labels() -> None:
    image = np.zeros((6, 6), dtype=np.float32)
    labels = np.zeros((6, 6), dtype=np.int32)
    labels[1:3, 1:3] = 1
    measurements = MeasurementTable(
        name="TileIntensity",
        object_name="Tiles",
        object_id_field="object_label",
        rows=[
            {"slice_index": 0, "object_label": 1, "StdIntensity": 4.0},
            {"slice_index": 1, "object_label": 1, "StdIntensity": 9.0},
        ],
    )
    runtime = _FakeCellProfilerRuntime(
        {"Carrier": _FakeRuntimeImage(image)},
        objects={
            "Tiles": ObjectLabelSet(
                name="Tiles",
                labels=labels,
            )
        },
        measurement_tables={"Tiles": (measurements,)},
    )
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="FilterObjects",
            inputs=(ArtifactSpec("Tiles", ArtifactKind.OBJECT_LABELS),),
            runtime_artifact_inputs=(ArtifactSpec("Tiles", ArtifactKind.OBJECT_LABELS),),
            outputs=(ArtifactSpec("FilteredTiles", ArtifactKind.OBJECT_LABELS),),
        )
    )
    seen_areas: list[tuple[float, tuple[int, ...]]] = []

    def filter_like(
        image: np.ndarray,
        object_labels: tuple[np.ndarray, ...] = (),
        measurement_tables: tuple[MeasurementTable, ...] = (),
    ) -> tuple[np.ndarray, np.ndarray]:
        area = float(measurement_tables[0].rows[0]["StdIntensity"])
        seen_areas.append((area, tuple(object_labels[0].shape)))
        return image, object_labels[0] * int(area)

    attach_callable_contract_metadata(
        filter_like,
        declared_processing_contract="flexible",
    )

    executor.run(filter_like, image, cellprofiler_runtime=runtime)

    assert seen_areas == [(4.0, (6, 6)), (9.0, (6, 6))]
    assert runtime.objects[0][1].shape == (2, 6, 6)
    np.testing.assert_array_equal(runtime.objects[0][1][0], labels * 4)
    np.testing.assert_array_equal(runtime.objects[0][1][1], labels * 9)


def test_artifact_measurement_table_does_not_drive_object_only_slicing() -> None:
    image = np.zeros((6, 6), dtype=np.float32)
    labels = np.zeros((6, 6), dtype=np.int32)
    labels[1:3, 1:3] = 1
    relationship_facts = MeasurementTable(
        name="RelationshipFacts",
        rows=[
            {"slice_index": index, "object_name": "Cells", "object_label": index}
            for index in range(4)
        ],
    )
    runtime = _FakeCellProfilerRuntime(
        {"Carrier": _FakeRuntimeImage(image)},
        objects={
            "Cells": ObjectLabelSet(
                name="Cells",
                labels=labels,
            )
        },
        measurement_tables={"Cells": (relationship_facts,)},
    )
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="FilterObjects",
            inputs=(ArtifactSpec("Cells", ArtifactKind.OBJECT_LABELS),),
            runtime_artifact_inputs=(ArtifactSpec("Cells", ArtifactKind.OBJECT_LABELS),),
            outputs=(ArtifactSpec("FilteredCells", ArtifactKind.OBJECT_LABELS),),
        )
    )
    calls: list[tuple[int, ...]] = []

    def filter_like(
        image: np.ndarray,
        object_labels: tuple[np.ndarray, ...] = (),
        measurement_tables: tuple[MeasurementTable, ...] = (),
    ) -> tuple[np.ndarray, np.ndarray]:
        del measurement_tables
        calls.append(tuple(object_labels[0].shape))
        return image, object_labels[0]

    attach_callable_contract_metadata(
        filter_like,
        declared_processing_contract="flexible",
    )

    executor.run(filter_like, image, cellprofiler_runtime=runtime)

    assert calls == [(6, 6)]
    assert runtime.objects[0][1].shape == (6, 6)


def test_flexible_object_module_aggregates_sliced_relationship_payloads() -> None:
    image = np.zeros((3, 6, 6), dtype=np.float32)
    labels = np.zeros((3, 6, 6), dtype=np.int32)
    labels[:, 1:4, 1:4] = np.arange(1, 4, dtype=np.int32)[:, None, None]
    runtime = _FakeCellProfilerRuntime(
        {"Carrier": _FakeRuntimeImage(image)},
        objects={
            "Cells": ObjectLabelSet(
                name="Cells",
                labels=labels,
            )
        },
    )
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="FilterObjects",
            inputs=(ArtifactSpec("Cells", ArtifactKind.OBJECT_LABELS),),
            runtime_artifact_inputs=(ArtifactSpec("Cells", ArtifactKind.OBJECT_LABELS),),
            outputs=(
                ArtifactSpec("FilteredCells", ArtifactKind.OBJECT_LABELS),
                ArtifactSpec(
                    "Cells_FilteredCells_relationships",
                    ArtifactKind.RELATIONSHIPS,
                ),
            ),
        )
    )

    def filter_like(
        image: np.ndarray,
        object_labels: tuple[np.ndarray, ...] = (),
    ) -> tuple[np.ndarray, np.ndarray, ParentChildRelationshipPayload]:
        label_id = int(np.max(object_labels[0]))
        return (
            image,
            object_labels[0],
            ParentChildRelationshipPayload(
                parent_ids=(label_id,),
                child_ids=(label_id,),
            ),
        )

    attach_callable_contract_metadata(
        filter_like,
        declared_processing_contract="flexible",
    )

    executor.run(filter_like, image, cellprofiler_runtime=runtime)

    assert runtime.relationships == [
        (
            "Cells_FilteredCells_relationships",
            {
                "parent_object_name": "Cells",
                "child_object_name": "FilteredCells",
                "parent_ids": (1, 2, 3),
                "child_ids": (1, 2, 3),
                "slice_indices": (0, 1, 2),
                "slice_count": 3,
            },
        )
    ]


def test_relationship_measurements_preserve_pure_2d_slice_indices() -> None:
    parent_labels = np.zeros((2, 5, 5), dtype=np.int32)
    child_labels = np.zeros((2, 5, 5), dtype=np.int32)
    parent_labels[0, 1:3, 1:3] = 1
    child_labels[0, 1:3, 1:3] = 1
    parent_labels[1, 2:4, 2:4] = 2
    child_labels[1, 2:4, 2:4] = 2
    runtime = _FakeCellProfilerRuntime(
        {"Carrier": _FakeRuntimeImage(np.zeros((2, 5, 5), dtype=np.float32))},
        objects={
            "Parents": ObjectLabelSet(name="Parents", labels=parent_labels),
            "Children": ObjectLabelSet(name="Children", labels=child_labels),
        },
    )
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="RelateObjects",
            inputs=(
                ArtifactSpec("Parents", ArtifactKind.OBJECT_LABELS),
                ArtifactSpec("Children", ArtifactKind.OBJECT_LABELS),
            ),
            runtime_artifact_inputs=(
                ArtifactSpec("Parents", ArtifactKind.OBJECT_LABELS),
                ArtifactSpec("Children", ArtifactKind.OBJECT_LABELS),
            ),
            outputs=(
                ArtifactSpec(
                    "Parents_Children_relationships",
                    ArtifactKind.RELATIONSHIPS,
                ),
            ),
        )
    )
    payload = ParentChildRelationshipPayload(
        parent_ids=(1, 2),
        child_ids=(1, 2),
        slice_indices=(0, 1),
        slice_count=2,
    )
    request = CellProfilerOutputRecordRequest(
        executor=executor,
        adapter=runtime,
        spec=executor.outputs[0],
        value=payload,
        output_values={executor.outputs[0].name: payload},
        source_image_name=None,
        func=lambda image: image,
    )

    rows = RelationshipMeasurementRows.for_request(request).rows()
    slice_indices = {
        int(row["slice_index"])
        for row in rows
        if "slice_index" in row
    }

    assert slice_indices == {0, 1}
    assert all(int(row["slice_index"]) in {0, 1} for row in rows)


def test_parent_child_relationship_payload_slices_with_pure_2d_kwargs() -> None:
    payload = ParentChildRelationshipPayload(
        parent_ids=(1, 2, 3, 4),
        child_ids=(10, 20, 30, 40),
        slice_indices=(0, 1, 0, 1),
        slice_count=2,
    )

    sliced = RuntimeSliceProjection.value_for_slice(payload, slice_index=1, slice_count=2)

    assert sliced == ParentChildRelationshipPayload(
        parent_ids=(2, 4),
        child_ids=(20, 40),
        slice_count=1,
    )


def test_object_relationship_slices_with_pure_2d_kwargs() -> None:
    semantics = RelationshipSemantics.parent_child("Parents", "Children")
    relationship = ObjectRelationship(
        name="Parents_Children_relationships",
        source=semantics.source,
        target=semantics.target,
        source_ids=(1, 2, 3, 4),
        target_ids=(10, 20, 30, 40),
        relationship_type=semantics.relationship_type,
        slice_indices=(0, 1, 0, 1),
        slice_count=2,
    )

    sliced = RuntimeSliceProjection.value_for_slice(relationship, slice_index=1, slice_count=2)

    assert isinstance(sliced, ObjectRelationship)
    assert sliced.source_ids == (2, 4)
    assert sliced.target_ids == (20, 40)
    assert sliced.slice_count == 1


def test_relationship_measurements_broadcast_singleton_label_counts() -> None:
    parent_labels = np.zeros((2, 5, 5), dtype=np.int32)
    child_labels = np.zeros((1, 5, 5), dtype=np.int32)
    parent_labels[:, 1:3, 1:3] = 1
    child_labels[0, 2:4, 2:4] = 1
    runtime = _FakeCellProfilerRuntime(
        {"Carrier": _FakeRuntimeImage(np.zeros((2, 5, 5), dtype=np.float32))},
        objects={
            "Parents": ObjectLabelSet(name="Parents", labels=parent_labels),
            "Children": ObjectLabelSet(name="Children", labels=child_labels),
        },
    )
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="RelateObjects",
            inputs=(
                ArtifactSpec("Parents", ArtifactKind.OBJECT_LABELS),
                ArtifactSpec("Children", ArtifactKind.OBJECT_LABELS),
            ),
            runtime_artifact_inputs=(
                ArtifactSpec("Parents", ArtifactKind.OBJECT_LABELS),
                ArtifactSpec("Children", ArtifactKind.OBJECT_LABELS),
            ),
            outputs=(
                ArtifactSpec(
                    "Parents_Children_relationships",
                    ArtifactKind.RELATIONSHIPS,
                ),
            ),
        )
    )
    payload = ParentChildRelationshipPayload(
        parent_ids=(1, 1),
        child_ids=(1, 1),
        slice_indices=(0, 1),
        slice_count=2,
    )
    request = CellProfilerOutputRecordRequest(
        executor=executor,
        adapter=runtime,
        spec=executor.outputs[0],
        value=payload,
        output_values={executor.outputs[0].name: payload},
        source_image_name=None,
        func=lambda image: image,
    )

    rows = RelationshipMeasurementRows.for_request(request).rows()

    parent_rows = [
        row
        for row in rows
        if row.get("object_name") == "Children"
        and RelationshipMeasurementFeatureTemplate.PARENT.feature_name(
            parent_object_name="Parents"
        )
        in row
    ]
    assert {(row["slice_index"], row["object_label"]) for row in parent_rows} == {
        (0, 1),
        (1, 1),
    }


def test_relateobjects_relationship_rows_project_distances_nominally() -> None:
    parent_labels = np.zeros((6, 6), dtype=np.int32)
    child_labels = np.zeros((6, 6), dtype=np.int32)
    parent_labels[1:5, 1:5] = 1
    child_labels[2:4, 2:4] = 1
    runtime = _FakeCellProfilerRuntime(
        {"Carrier": _FakeRuntimeImage(np.zeros((6, 6), dtype=np.float32))},
        objects={
            "Parents": ObjectLabelSet(name="Parents", labels=parent_labels),
            "Children": ObjectLabelSet(name="Children", labels=child_labels),
        },
    )
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="RelateObjects",
            inputs=(
                ArtifactSpec("Parents", ArtifactKind.OBJECT_LABELS),
                ArtifactSpec("Children", ArtifactKind.OBJECT_LABELS),
            ),
            runtime_artifact_inputs=(
                ArtifactSpec("Parents", ArtifactKind.OBJECT_LABELS),
                ArtifactSpec("Children", ArtifactKind.OBJECT_LABELS),
            ),
            outputs=(
                ArtifactSpec(
                    "Parents_Children_relationships",
                    ArtifactKind.RELATIONSHIPS,
                ),
            ),
        )
    )
    payload = ParentChildRelationshipPayload(parent_ids=(1,), child_ids=(1,))
    request = CellProfilerOutputRecordRequest(
        executor=executor,
        adapter=runtime,
        spec=executor.outputs[0],
        value=RelationshipMeasurements(
            slice_index=0,
            parent_object_count=1,
            child_object_count=1,
            children_with_parents_count=1,
            mean_children_per_parent=1.0,
            mean_centroid_distance=0.0,
            mean_minimum_distance=1.0,
        ),
        output_values={executor.outputs[0].name: payload},
        source_image_name=None,
        func=lambda image: image,
    )

    rows = RelationshipMeasurementRows.for_request(request).rows()

    distance_rows = [
        row
        for row in rows
        if row.get("object_name") == "Children"
        and "Distance_Centroid_Parents" in row
    ]
    assert len(distance_rows) == 1
    assert distance_rows[0]["Distance_Centroid_Parents"] == pytest.approx(0.0)
    assert distance_rows[0]["Distance_Minimum_Parents"] == pytest.approx(
        np.sqrt(2.5)
    )


def test_relateobjects_relationship_rows_project_distances_from_slice_measurements() -> None:
    parent_labels = np.zeros((2, 6, 6), dtype=np.int32)
    child_labels = np.zeros((2, 6, 6), dtype=np.int32)
    parent_labels[:, 1:5, 1:5] = 1
    child_labels[:, 2:4, 2:4] = 1
    runtime = _FakeCellProfilerRuntime(
        {"Carrier": _FakeRuntimeImage(np.zeros((2, 6, 6), dtype=np.float32))},
        objects={
            "Parents": ObjectLabelSet(name="Parents", labels=parent_labels),
            "Children": ObjectLabelSet(name="Children", labels=child_labels),
        },
    )
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="RelateObjects",
            inputs=(
                ArtifactSpec("Parents", ArtifactKind.OBJECT_LABELS),
                ArtifactSpec("Children", ArtifactKind.OBJECT_LABELS),
            ),
            runtime_artifact_inputs=(
                ArtifactSpec("Parents", ArtifactKind.OBJECT_LABELS),
                ArtifactSpec("Children", ArtifactKind.OBJECT_LABELS),
            ),
            outputs=(
                ArtifactSpec(
                    "Parents_Children_relationships",
                    ArtifactKind.RELATIONSHIPS,
                ),
            ),
        )
    )
    payload = ParentChildRelationshipPayload(
        parent_ids=(1, 1),
        child_ids=(1, 1),
        slice_indices=(0, 1),
        slice_count=2,
    )
    request = CellProfilerOutputRecordRequest(
        executor=executor,
        adapter=runtime,
        spec=executor.outputs[0],
        value=[
            RelationshipMeasurements(
                slice_index=0,
                parent_object_count=1,
                child_object_count=1,
                children_with_parents_count=1,
                mean_children_per_parent=1.0,
                mean_centroid_distance=0.0,
                mean_minimum_distance=1.0,
            ),
            RelationshipMeasurements(
                slice_index=1,
                parent_object_count=1,
                child_object_count=1,
                children_with_parents_count=1,
                mean_children_per_parent=1.0,
                mean_centroid_distance=0.0,
                mean_minimum_distance=1.0,
            ),
        ],
        output_values={executor.outputs[0].name: payload},
        source_image_name=None,
        func=lambda image: image,
    )

    rows = RelationshipMeasurementRows.for_request(request).rows()

    distance_rows = [
        row
        for row in rows
        if row.get("object_name") == "Children"
        and "Distance_Centroid_Parents" in row
    ]
    assert {(row["slice_index"], row["object_label"]) for row in distance_rows} == {
        (0, 1),
        (1, 1),
    }


def test_object_relationship_backend_uses_sparse_ijv_contract_nominally() -> None:
    parent_dense = np.zeros((8, 8), dtype=np.int32)
    child_dense = np.zeros((8, 8), dtype=np.int32)
    parent_sparse = ObjectLabelSet(
        name="Parents",
        labels=SparseIJVLabelRows(
            np.asarray(
                (
                    (1, 1, 1),
                    (2, 2, 2),
                ),
                dtype=np.int32,
            )
        ),
        representation=ObjectLabelRepresentation.SPARSE_IJV,
    )
    child_sparse = ObjectLabelSet(
        name="Children",
        labels=SparseIJVLabelRows(
            np.asarray(
                (
                    (1, 1, 7),
                    (2, 2, 8),
                    (7, 7, 9),
                ),
                dtype=np.int32,
            )
        ),
        representation=ObjectLabelRepresentation.SPARSE_IJV,
    )
    parent_dense[1, 1] = 1
    parent_dense[2, 2] = 2
    backend = ObjectRelationshipBackendStrategy.for_memory_type()

    dense_payload = backend.parent_child_payload_from_labels(
        np.zeros((8, 8), dtype=np.int32),
        child_dense,
    )
    sparse_payload = backend.parent_child_payload_from_labels(parent_sparse, child_sparse)
    mixed_payload = backend.parent_child_payload_from_labels(parent_dense, child_sparse)

    assert dense_payload == ParentChildRelationshipPayload(
        parent_ids=(),
        child_ids=(),
    )
    assert sparse_payload == ParentChildRelationshipPayload(
        parent_ids=(1, 2),
        child_ids=(7, 8),
    )
    assert backend.parents_of_from_payload(sparse_payload, 9)[6:9].tolist() == [
        1,
        2,
        0,
    ]
    assert mixed_payload == sparse_payload


def test_define_grid_automatic_uses_integer_lowest_spot_origin() -> None:
    image = np.zeros((20, 20), dtype=np.float32)
    labels = np.zeros((20, 20), dtype=np.int32)
    labels[2:6, 3:7] = 1
    labels[12:16, 13:17] = 2

    _image, grid = define_grid_automatic.__wrapped__(
        image,
        labels,
        grid_rows=2,
        grid_columns=2,
    )

    assert grid.x_location_of_lowest_x_spot == 4.0
    assert grid.y_location_of_lowest_y_spot == 3.0
    assert grid.x_spacing == 10.0
    assert grid.y_spacing == 10.0


def test_identify_objects_in_grid_respects_row_primary_ordering() -> None:
    image = np.zeros((6, 9), dtype=np.float32)
    grid = SpatialGrid(
        name="grid_info",
        rows=2,
        columns=3,
        x_spacing=3.0,
        y_spacing=3.0,
        x_origin=1.0,
        y_origin=1.0,
        ordering=SpatialGridOrdering.BY_ROWS,
    )

    _image, _stats, payload = identify_objects_in_grid(
        image,
        grid=grid,
        dtype_config=DtypeConfig(),
    )

    labels = np.asarray(payload.labels)
    assert labels[1, 1] == 1
    assert labels[4, 1] == 2
    assert labels[1, 4] == 3
    assert labels[4, 4] == 4
    assert labels[1, 7] == 5
    assert labels[4, 7] == 6


def test_identify_objects_in_grid_respects_column_primary_ordering() -> None:
    image = np.zeros((6, 9), dtype=np.float32)
    grid = SpatialGrid(
        name="grid_info",
        rows=2,
        columns=3,
        x_spacing=3.0,
        y_spacing=3.0,
        x_origin=1.0,
        y_origin=1.0,
        ordering=SpatialGridOrdering.BY_COLUMNS,
    )

    _image, _stats, payload = identify_objects_in_grid(
        image,
        grid=grid,
        dtype_config=DtypeConfig(),
    )

    labels = np.asarray(payload.labels)
    assert labels[1, 1] == 1
    assert labels[1, 4] == 2
    assert labels[1, 7] == 3
    assert labels[4, 1] == 4
    assert labels[4, 4] == 5
    assert labels[4, 7] == 6


def test_identify_objects_in_grid_fill_boundaries_match_floor_bins() -> None:
    grid = GridDefinition.from_runtime(
        image_shape=(11, 14),
        grid=None,
        grid_rows=3,
        grid_columns=4,
        x_spacing=3.25,
        y_spacing=2.75,
        x_origin=1.2,
        y_origin=1.6,
        ordering=SpatialGridOrdering.BY_ROWS,
    )

    labels = grid.filled_labels()
    row_origin = int(grid.y_location_of_lowest_y_spot - grid.y_spacing / 2)
    col_origin = int(grid.x_location_of_lowest_x_spot - grid.x_spacing / 2)
    expected = np.zeros(labels.shape, dtype=np.int32)
    rows, cols = np.indices(labels.shape)
    row_bins = np.floor((rows - row_origin) / grid.y_spacing).astype(int)
    col_bins = np.floor((cols - col_origin) / grid.x_spacing).astype(int)
    mask = (
        (row_bins >= 0)
        & (row_bins < grid.rows)
        & (col_bins >= 0)
        & (col_bins < grid.columns)
    )
    expected[mask] = grid.spot_table[row_bins[mask], col_bins[mask]]

    np.testing.assert_array_equal(labels, expected)


def test_identify_objects_in_grid_natural_shape_preserves_guide_shape() -> None:
    image = np.zeros((5, 10), dtype=np.float32)
    guide_labels = np.zeros((5, 10), dtype=np.int32)
    guide_labels[2, 1:6] = 1
    grid = SpatialGrid(
        name="grid_info",
        rows=1,
        columns=2,
        x_spacing=5.0,
        y_spacing=5.0,
        x_origin=2.0,
        y_origin=2.0,
        ordering=SpatialGridOrdering.BY_ROWS,
    )

    _image, _stats, payload = identify_objects_in_grid_with_guides(
        image,
        guide_labels,
        grid=grid,
        shape_choice="natural_shape_and_location",
        dtype_config=DtypeConfig(),
    )

    labels = np.asarray(payload.labels)
    np.testing.assert_array_equal(labels[2, 1:6], np.asarray([1, 1, 1, 1, 1]))
    assert labels[2, 7] == 0
    assert labels[0, 0] == 0


def test_identify_objects_in_grid_natural_shape_preserves_accepted_grid_ids() -> None:
    image = np.zeros((5, 15), dtype=np.float32)
    guide_labels = np.zeros((5, 15), dtype=np.int32)
    guide_labels[2, 1:3] = 10
    guide_labels[2, 11:13] = 20
    grid = SpatialGrid(
        name="grid_info",
        rows=1,
        columns=3,
        x_spacing=5.0,
        y_spacing=5.0,
        x_origin=2.0,
        y_origin=2.0,
        ordering=SpatialGridOrdering.BY_ROWS,
    )

    _image, _stats, payload = identify_objects_in_grid_with_guides(
        image,
        guide_labels,
        grid=grid,
        shape_choice="natural_shape_and_location",
        dtype_config=DtypeConfig(),
    )

    labels = np.asarray(payload.labels)
    assert set(np.unique(labels)) == {0, 1, 3}
    np.testing.assert_array_equal(labels[2, 1:3], np.asarray([1, 1]))
    assert labels[2, 7] == 0
    np.testing.assert_array_equal(labels[2, 11:13], np.asarray([3, 3]))
    assert payload.declared_object_count == 3


def test_identify_objects_in_grid_natural_shape_uses_filtered_guides() -> None:
    guide_labels = np.zeros((5, 10), dtype=np.int32)
    guide_labels[2, 1:4] = 7
    filtered_guides = np.zeros_like(guide_labels)
    grid = GridDefinition.from_runtime(
        image_shape=guide_labels.shape,
        grid=None,
        grid_rows=1,
        grid_columns=2,
        x_spacing=5.0,
        y_spacing=5.0,
        x_origin=2.0,
        y_origin=2.0,
        ordering=SpatialGridOrdering.BY_ROWS,
    )

    labels = NaturalGridShapeStrategy().labels(
        GridShapeRequest(
            grid=grid,
            guiding_labels=guide_labels,
            filtered_guides=filtered_guides,
        )
    )
    assert not np.any(labels)

    filtered_guides[2, 1:4] = 1
    labels = NaturalGridShapeStrategy().labels(
        GridShapeRequest(
            grid=grid,
            guiding_labels=guide_labels,
            filtered_guides=filtered_guides,
        )
    )

    np.testing.assert_array_equal(labels[2, 1:4], np.asarray([1, 1, 1]))
    assert labels[2, 6] == 0


def test_object_location_measurements_preserve_declared_empty_grid_cells() -> None:
    labels = np.zeros((2, 5, 5), dtype=np.int32)
    labels[0, 1:3, 1:3] = 1
    labels[1, 3:5, 3:5] = 2
    payload = ObjectLabelPayload(labels=labels, declared_object_count=3)

    rows = ObjectLocationMeasurementRows(
        payload,
        object_name="GridObjects",
    ).rows()

    by_key = {
        (
            row["slice_index"],
            row["object_label"],
            row["feature_name"],
        ): row["result_value"]
        for row in rows
    }
    assert by_key[(0, 1, ObjectLocationMeasurementFeature.CENTER_X.value)] == 1.5
    assert by_key[(0, 1, ObjectLocationMeasurementFeature.CENTER_Y.value)] == 1.5
    assert np.isnan(by_key[(0, 2, ObjectLocationMeasurementFeature.CENTER_X.value)])
    assert np.isnan(by_key[(0, 3, ObjectLocationMeasurementFeature.CENTER_Y.value)])
    assert np.isnan(by_key[(1, 1, ObjectLocationMeasurementFeature.CENTER_X.value)])
    assert by_key[(1, 2, ObjectLocationMeasurementFeature.CENTER_X.value)] == 3.5
    assert by_key[(1, 2, ObjectLocationMeasurementFeature.CENTER_Y.value)] == 3.5
    assert np.isnan(by_key[(1, 3, ObjectLocationMeasurementFeature.CENTER_Y.value)])


def test_sparse_object_label_aggregation_preserves_declared_domain() -> None:
    first = ObjectLabelSet(
        name="GridObjects",
        labels=SparseIJVLabelRows.from_yx_label(
            np.asarray([[0, 0, 1], [1, 1, 3]], dtype=np.int32)
        ),
        representation=ObjectLabelRepresentation.SPARSE_IJV,
        declared_object_count=4,
    )
    second = ObjectLabelSet(
        name="GridObjects",
        labels=SparseIJVLabelRows.from_yx_label(
            np.asarray([[0, 0, 2], [1, 1, 4]], dtype=np.int32)
        ),
        representation=ObjectLabelRepresentation.SPARSE_IJV,
        declared_object_count=4,
    )

    aggregated = CellProfilerPure2DOutputAggregator.aggregate(
        [first, second],
        MemoryType.NUMPY.value,
    )

    assert isinstance(aggregated, ObjectLabelSet)
    assert aggregated.declared_object_count == 4
    assert isinstance(aggregated.labels, SparseIJVLabelRows)


def test_measurement_table_collection_slice_count_accepts_sharded_offsets() -> None:
    first = MeasurementTable(
        name="ObjectMeasurements",
        rows=[{"slice_index": 0, "object_label": 1, "area": 11.0}],
        object_name="Objects",
    )
    second = MeasurementTable(
        name="ObjectMeasurements",
        rows=[{"slice_index": 1, "object_label": 1, "area": 13.0}],
        object_name="Objects",
    )

    assert RuntimeSliceProjection.slice_count_from_values((first, second)) == 2


def test_measurement_table_slice_count_accepts_columnar_rows() -> None:
    table = MeasurementTable(
        name="ObjectMeasurements",
        rows=_ColumnarMeasurementRows(
            {
                "slice_index": (0, 0, 1),
                "object_label": (1, 2, 1),
                "area": (11.0, 12.0, 13.0),
            }
        ),
        object_name="Objects",
    )

    assert RuntimeSliceProjection.measurement_table_slice_count(table) == 2


def test_measurement_table_for_slice_preserves_columnar_rows() -> None:
    table = MeasurementTable(
        name="ObjectMeasurements",
        rows=_ColumnarMeasurementRows(
            {
                "slice_index": (0, 0, 1),
                "feature_name": ("Area", "MeanIntensity", "Area"),
                "result_value": (11.0, 12.0, 13.0),
            }
        ),
        fields=(
            FieldSpec("slice_index"),
            FieldSpec("feature_name"),
            FieldSpec("result_value"),
        ),
        object_name="Objects",
    )

    sliced = measurement_table_for_slice(table, 1)

    assert isinstance(sliced.rows, ColumnarRows)
    assert tuple(sliced.rows.columns["slice_index"]) == (1,)
    assert tuple(sliced.rows.columns["result_value"]) == (13.0,)


def test_measurement_table_for_slice_normalizes_mixed_sequence_rows() -> None:
    table = MeasurementTable(
        name="ObjectMeasurements",
        rows=[
            {"slice_index": 1, "feature_name": "Area", "result_value": 13.0},
            {"slice_index": 1, "MeanIntensity": 17.0},
        ],
        object_name="Objects",
    )

    sliced = measurement_table_for_slice(table, 1)

    assert sliced.fields == ()
    assert all("feature_name" in row for row in sliced.rows)
    assert all("result_value" in row for row in sliced.rows)


def test_unstack_cellprofiler_image_slices_collapses_pairwise_slice_grid() -> None:
    pairwise = np.zeros((2, 2, 5, 6), dtype=np.float32)
    pairwise[0, 0] = 1.0
    pairwise[0, 1] = 2.0
    pairwise[1, 0] = 3.0
    pairwise[1, 1] = 4.0

    slices = _unstack_cellprofiler_image_slices(pairwise, MemoryType.NUMPY.value)

    assert len(slices) == 2
    np.testing.assert_array_equal(image_payload_data(slices[0]), np.full((5, 6), 1.0))
    np.testing.assert_array_equal(image_payload_data(slices[1]), np.full((5, 6), 4.0))


def test_unstack_cellprofiler_image_slices_projects_singleton_volume_stack_mask() -> None:
    data = np.arange(1 * 3 * 4 * 5, dtype=np.float32).reshape(1, 3, 4, 5)
    mask = np.ones((1, 3, 4, 5), dtype=bool)
    mask[0, 1] = False
    payload = image_payload_with_context(data, mask=mask)

    slices = _unstack_cellprofiler_image_slices(payload, MemoryType.NUMPY.value)

    assert len(slices) == 3
    np.testing.assert_array_equal(image_payload_data(slices[1]), data[0, 1])
    np.testing.assert_array_equal(image_payload_data(slices[2]), data[0, 2])
    np.testing.assert_array_equal(image_payload_mask(slices[1]), mask[0, 1])
    np.testing.assert_array_equal(image_payload_mask(slices[2]), mask[0, 2])


def test_unstack_cellprofiler_image_slices_projects_high_rank_plane_mask() -> None:
    data = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5)
    mask = np.ones((2, 3, 4, 5), dtype=bool)
    mask[1, 2] = False
    payload = image_payload_with_context(data, mask=mask)

    slices = _unstack_cellprofiler_image_slices(payload, MemoryType.NUMPY.value)

    assert len(slices) == 6
    np.testing.assert_array_equal(image_payload_data(slices[5]), data[1, 2])
    np.testing.assert_array_equal(image_payload_mask(slices[5]), mask[1, 2])


def test_unstack_cellprofiler_image_slices_projects_source_axis_mask_to_plane() -> None:
    data_slice = np.arange(4 * 5, dtype=np.float32).reshape(4, 5)
    mask = np.ones((2, 4, 5), dtype=bool)
    mask[1] = False

    payload = ImagePayloadSliceProjector(
        mask=mask,
        metadata=ImagePayloadMetadata(),
    ).payload_for_slice(data_slice, 2)

    np.testing.assert_array_equal(
        image_payload_mask(payload),
        np.zeros((4, 5), dtype=bool),
    )


def test_cellprofiler_contract_executor_rejects_uncoerced_unknown_absorbed_contract():
    def two_dimensional_only(image: np.ndarray, **kwargs) -> np.ndarray:
        if image.ndim != 2:
            raise RuntimeError("2D only")
        return image

    attach_callable_contract_metadata(
        two_dimensional_only,
        declared_processing_contract="unknown",
    )

    with pytest.raises(TypeError, match="no nominal __processing_contract__"):
        _processing_contract_for_callable(two_dimensional_only)


def test_measurement_image_for_labels_preserves_source_stack_for_2d_labels() -> None:
    image = np.arange(2 * 4 * 5, dtype=np.uint16).reshape(2, 4, 5)
    labels = np.ones((4, 5), dtype=np.int32)

    measurement_image = _measurement_image_for_labels(image, labels)

    assert measurement_image is image


def test_measurement_image_for_labels_preserves_object_domain_stack_for_2d_labels() -> None:
    image = np.arange(2 * 4 * 5, dtype=np.uint16).reshape(2, 4, 5)
    labels = np.ones((4, 5), dtype=np.int32)

    measurement_image = _measurement_image_for_labels(
        image,
        labels,
        reference_domain=CellProfilerMeasurementImageDomain.OBJECT_LABELS,
    )

    assert measurement_image is image


def test_measurement_image_for_labels_uses_object_domain_reference_shape() -> None:
    image = np.ones((10, 12), dtype=np.float32)
    labels = np.ones((4, 5), dtype=np.int32)

    measurement_image = _measurement_image_for_labels(
        image,
        labels,
        reference_domain=CellProfilerMeasurementImageDomain.OBJECT_LABELS,
    )

    assert measurement_image.shape == labels.shape
    assert measurement_image.dtype == image.dtype
    np.testing.assert_array_equal(measurement_image, np.zeros_like(labels, dtype=image.dtype))


def test_measurement_image_for_labels_keeps_source_domain_shape_mismatch() -> None:
    image = np.ones((10, 12), dtype=np.float32)
    labels = np.ones((4, 5), dtype=np.int32)

    measurement_image = _measurement_image_for_labels(image, labels)

    assert measurement_image is image


def test_measurement_labels_collapse_singleton_label_stack() -> None:
    labels = np.ones((1, 4, 5), dtype=np.int32)

    measurement_labels = _measurement_labels(labels)

    assert measurement_labels.shape == (4, 5)
    np.testing.assert_array_equal(measurement_labels, labels[0])


def test_measurement_labels_preserve_stack_for_object_domain_alignment() -> None:
    image = np.ones((1, 4, 5), dtype=np.float32)
    labels = np.arange(2 * 4 * 5, dtype=np.int32).reshape(2, 4, 5)

    measurement_labels = MeasurementLabelSourceAlignmentStrategy.align(image, labels)

    assert measurement_labels.shape == labels.shape
    np.testing.assert_array_equal(measurement_labels, labels)


def test_measurement_label_alignment_preserves_runtime_slice_payload_for_aligned_stack() -> None:
    first_image = image_payload_with_context(
        np.ones((2, 2), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            spatial_origin_yx=(0, 0),
            source_spatial_shape_yx=(4, 4),
        ),
    )
    second_image = image_payload_with_context(
        np.ones((2, 2), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            spatial_origin_yx=(2, 0),
            source_spatial_shape_yx=(4, 4),
        ),
    )
    labels = np.stack(
        (
            np.full((4, 4), 1, dtype=np.int32),
            np.full((4, 4), 2, dtype=np.int32),
        )
    )
    label_payload = ObjectLabelPayload(
        labels=labels,
        domain_scope=ObjectLabelDomainScope.PLANE,
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    )

    measurement_labels = MeasurementLabelSourceAlignmentStrategy.align(
        AlignedImageStack((first_image, second_image)),
        labels,
        label_payload=label_payload,
    )

    assert measurement_labels.shape == labels.shape
    np.testing.assert_array_equal(measurement_labels, labels)


def test_aligned_stack_kwargs_projects_runtime_slice_labels_to_reference_slice_domain() -> None:
    reference_slice = image_payload_with_context(
        np.ones((2, 2), dtype=np.float32),
        metadata=ImagePayloadMetadata(
            spatial_origin_yx=(2, 0),
            source_spatial_shape_yx=(4, 4),
        ),
    )
    labels = np.stack(
        (
            np.full((4, 4), 1, dtype=np.int32),
            np.full((4, 4), 2, dtype=np.int32),
        )
    )
    label_payload = ObjectLabelPayload(
        labels=labels,
        domain_scope=ObjectLabelDomainScope.PLANE,
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        source_spatial_shape_yx=(4, 4),
    )

    resolved = aligned_image_stack_kwargs(
        {"labels": label_payload},
        slice_index=1,
        slice_count=2,
        reference_payload=reference_slice,
    )

    np.testing.assert_array_equal(
        resolved["labels"],
        np.full((2, 2), 2, dtype=np.int32),
    )


def test_aligned_stack_kwargs_projects_runtime_slice_labels_without_source_metadata() -> None:
    labels = np.stack(
        (
            np.full((4, 4), 1, dtype=np.int32),
            np.full((4, 4), 2, dtype=np.int32),
        )
    )
    label_payload = ObjectLabelPayload(
        labels=labels,
        domain_scope=ObjectLabelDomainScope.PLANE,
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    )

    resolved = aligned_image_stack_kwargs(
        {"labels": label_payload},
        slice_index=1,
        slice_count=2,
        reference_payload=np.ones((4, 4), dtype=np.float32),
    )

    assert isinstance(resolved["labels"], ObjectLabelPayload)
    np.testing.assert_array_equal(
        resolved["labels"].labels,
        np.full((4, 4), 2, dtype=np.int32),
    )
    assert resolved["labels"].domain_scope is ObjectLabelDomainScope.PLANE
    assert resolved["labels"].plane_axis is RuntimePlaneAxis.RUNTIME_SLICE


def test_measurement_labels_collapse_channel_broadcast_label_stack() -> None:
    image = np.ones((2, 4, 5), dtype=np.float32)
    label_plane = np.arange(4 * 5, dtype=np.int32).reshape(4, 5)
    labels = np.stack((label_plane, label_plane))

    measurement_labels = MeasurementLabelSourceAlignmentStrategy.align(image, labels)

    assert measurement_labels.shape == label_plane.shape
    np.testing.assert_array_equal(measurement_labels, label_plane)


def test_measurement_labels_select_source_named_plane_from_composed_image() -> None:
    image = np.ones((2, 4, 5), dtype=np.float32)
    dna_labels = np.full((4, 5), 3, dtype=np.int32)
    gfp_labels = np.full((4, 5), 7, dtype=np.int32)
    labels = ObjectLabelSet(
        name="Nuclei",
        labels=np.stack((dna_labels, gfp_labels)),
        source_image_name="rawDNA",
        domain_scope=ObjectLabelDomainScope.PLANE,
        plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
    )
    measurement_image = CellProfilerMeasurementImage(
        source_image_name="rawDNA__rawGFP",
        source_image_names=("rawDNA", "rawGFP"),
        payload=image,
    )

    measurement_labels = _measurement_labels_for_measurement_image(
        measurement_image,
        labels,
    )

    assert measurement_labels.shape == dna_labels.shape
    np.testing.assert_array_equal(measurement_labels, dna_labels)


def test_measurement_labels_select_current_source_binding_plane() -> None:
    class Adapter(RuntimePlaneAxisProjector):
        def runtime_slice_plane_index(self) -> int | None:
            return None

        def source_binding_axis_plane_index(
            self,
            source_aliases: tuple[str, ...],
        ) -> int | None:
            return {"rawDNA": 2}.get(source_aliases[0])

    image = np.ones((2, 4, 5), dtype=np.float32)
    label_planes = np.stack(
        (
            np.full((4, 5), 1, dtype=np.int32),
            np.full((4, 5), 2, dtype=np.int32),
            np.full((4, 5), 3, dtype=np.int32),
        )
    )
    labels = ObjectLabelSet(
        name="Nuclei",
        labels=label_planes,
        source_image_name="rawDNA",
        domain_scope=ObjectLabelDomainScope.PLANE,
        plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
    )
    measurement_image = CellProfilerMeasurementImage(
        source_image_name="rawDNA__rawGFP",
        source_image_names=("rawDNA", "rawGFP"),
        payload=image,
    )

    measurement_labels = _measurement_labels_for_measurement_image(
        measurement_image,
        labels,
        adapter=Adapter(),
    )

    assert measurement_labels.shape == (4, 5)
    np.testing.assert_array_equal(measurement_labels, label_planes[2])


def test_measurement_labels_select_measurement_source_binding_over_label_origin() -> None:
    class Adapter(RuntimePlaneAxisProjector):
        def runtime_slice_plane_index(self) -> int | None:
            return None

        def source_binding_axis_plane_index(
            self,
            source_aliases: tuple[str, ...],
        ) -> int | None:
            return {"rawDNA": 0, "rawGFP": 1}.get(source_aliases[0])

    image = np.ones((2, 4, 5), dtype=np.float32)
    label_planes = np.stack(
        (
            np.full((4, 5), 11, dtype=np.int32),
            np.full((4, 5), 22, dtype=np.int32),
        )
    )
    labels = ObjectLabelSet(
        name="ExpandedCells",
        labels=label_planes,
        source_image_name="rawDNA",
        domain_scope=ObjectLabelDomainScope.PLANE,
        plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
    )
    measurement_image = CellProfilerMeasurementImage(
        source_image_name="rawGFP",
        source_image_names=("rawGFP",),
        payload=image,
    )

    measurement_labels = _measurement_labels_for_measurement_image(
        measurement_image,
        labels,
        adapter=Adapter(),
    )

    assert measurement_labels.shape == (4, 5)
    np.testing.assert_array_equal(measurement_labels, label_planes[1])


def test_measurement_labels_do_not_slice_site_stack_for_single_source_binding() -> None:
    class Adapter(RuntimePlaneAxisProjector):
        def runtime_slice_plane_index(self) -> int | None:
            return None

        def source_binding_axis_plane_index(
            self,
            source_aliases: tuple[str, ...],
        ) -> int | None:
            return {"rawGFP": 0}.get(source_aliases[0])

    image = np.ones((3, 4, 5), dtype=np.float32)
    labels = np.stack(
        (
            np.full((4, 5), 1, dtype=np.int32),
            np.full((4, 5), 2, dtype=np.int32),
            np.full((4, 5), 3, dtype=np.int32),
        )
    )
    measurement_image = CellProfilerMeasurementImage(
        source_image_name="rawGFP",
        source_image_names=("rawGFP",),
        payload=image,
    )

    measurement_labels = _measurement_labels_for_measurement_image(
        measurement_image,
        labels,
        adapter=Adapter(),
    )

    assert measurement_labels.shape == labels.shape
    np.testing.assert_array_equal(measurement_labels, labels)


def test_measurement_labels_do_not_project_unowned_label_stack_by_source_alias() -> None:
    class Adapter(RuntimePlaneAxisProjector):
        def runtime_slice_plane_index(self) -> int | None:
            return None

        def source_binding_axis_plane_index(
            self,
            source_aliases: tuple[str, ...],
        ) -> int | None:
            return {"rawGFP": 0}.get(source_aliases[0])

    image = np.ones((3, 4, 5), dtype=np.float32)
    label_planes = np.stack(
        (
            np.full((4, 5), 1, dtype=np.int32),
            np.full((4, 5), 2, dtype=np.int32),
            np.full((4, 5), 3, dtype=np.int32),
        )
    )
    labels = ObjectLabelSet(
        name="Cells",
        labels=label_planes,
        domain_scope=ObjectLabelDomainScope.PLANE,
    )
    measurement_image = CellProfilerMeasurementImage(
        source_image_name="rawGFP",
        source_image_names=("rawGFP",),
        payload=image,
    )

    measurement_labels = _measurement_labels_for_measurement_image(
        measurement_image,
        labels,
        adapter=Adapter(),
    )

    assert measurement_labels.shape == label_planes.shape
    np.testing.assert_array_equal(measurement_labels, label_planes)


def test_measurement_labels_do_not_project_runtime_slice_label_stack_by_source_alias() -> None:
    class Adapter(RuntimePlaneAxisProjector):
        def runtime_slice_plane_index(self) -> int | None:
            return 1

        def source_binding_axis_plane_index(
            self,
            source_aliases: tuple[str, ...],
        ) -> int | None:
            return {"BF_image": 1}.get(source_aliases[0])

    image = np.ones((2, 4, 5), dtype=np.float32)
    label_planes = np.stack(
        (
            np.full((4, 5), 10, dtype=np.int32),
            np.full((4, 5), 20, dtype=np.int32),
        )
    )
    labels = ObjectLabelSet(
        name="Cells",
        labels=label_planes,
        source_image_name="BF_image",
        domain_scope=ObjectLabelDomainScope.PLANE,
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    )
    measurement_image = CellProfilerMeasurementImage(
        source_image_name="BF_image",
        source_image_names=("BF_image",),
        payload=image,
    )

    measurement_labels = _measurement_labels_for_measurement_image(
        measurement_image,
        labels,
        adapter=Adapter(),
    )

    assert measurement_labels.shape == label_planes.shape
    np.testing.assert_array_equal(measurement_labels, label_planes)


def test_measurement_labels_do_not_project_runtime_slice_stack_for_aligned_measurement_image() -> None:
    class Adapter(RuntimePlaneAxisProjector):
        def runtime_slice_plane_index(self) -> int | None:
            return 1

        def source_binding_axis_plane_index(
            self,
            source_aliases: tuple[str, ...],
        ) -> int | None:
            return {"CropBlue": 0, "CropGreen": 1}.get(source_aliases[0])

    label_planes = np.stack(
        (
            np.full((4, 5), 10, dtype=np.int32),
            np.full((4, 5), 20, dtype=np.int32),
            np.full((4, 5), 30, dtype=np.int32),
        )
    )
    labels = ObjectLabelSet(
        name="Nuclei",
        labels=label_planes,
        source_image_name="CropBlue",
        domain_scope=ObjectLabelDomainScope.PLANE,
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    )
    measurement_image = CellProfilerMeasurementImage(
        source_image_name="CropBlue__CropGreen",
        source_image_names=("CropBlue", "CropGreen"),
        payload=AlignedImageStack(
            (
                np.ones((2, 4, 5), dtype=np.float32),
                np.ones((2, 4, 5), dtype=np.float32),
                np.ones((2, 4, 5), dtype=np.float32),
            )
        ),
    )

    measurement_labels = _measurement_labels_for_measurement_image(
        measurement_image,
        labels,
        adapter=Adapter(),
    )

    assert measurement_labels.shape == label_planes.shape
    np.testing.assert_array_equal(measurement_labels, label_planes)


def test_measurement_labels_project_runtime_slice_stack_for_single_plane_source() -> None:
    class Adapter(RuntimePlaneAxisProjector):
        def runtime_slice_plane_index(self) -> int | None:
            return 1

        def source_binding_axis_plane_index(
            self,
            source_aliases: tuple[str, ...],
        ) -> int | None:
            return {"BF_image": 1}.get(source_aliases[0])

    image = np.ones((4, 5), dtype=np.float32)
    label_planes = np.stack(
        (
            np.full((4, 5), 10, dtype=np.int32),
            np.full((4, 5), 20, dtype=np.int32),
        )
    )
    labels = ObjectLabelSet(
        name="Cells",
        labels=label_planes,
        source_image_name="BF_image",
        domain_scope=ObjectLabelDomainScope.PLANE,
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    )
    measurement_image = CellProfilerMeasurementImage(
        source_image_name="BF_image",
        source_image_names=("BF_image",),
        payload=image,
    )

    measurement_labels = _measurement_labels_for_measurement_image(
        measurement_image,
        labels,
        adapter=Adapter(),
    )

    assert measurement_labels.shape == label_planes[1].shape
    np.testing.assert_array_equal(measurement_labels, label_planes[1])


def test_measurement_labels_preserve_runtime_slice_stack_for_object_domain() -> None:
    class Adapter(RuntimePlaneAxisProjector):
        def runtime_slice_plane_index(self) -> int | None:
            return 1

        def source_binding_axis_plane_index(
            self,
            source_aliases: tuple[str, ...],
        ) -> int | None:
            return None

    label_planes = np.stack(
        (
            np.full((4, 5), 10, dtype=np.int32),
            np.full((4, 5), 20, dtype=np.int32),
        )
    )
    labels = ObjectLabelSet(
        name="Cells",
        labels=label_planes,
        domain_scope=ObjectLabelDomainScope.PLANE,
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    )
    measurement_image = CellProfilerMeasurementImage(
        source_image_name=None,
        source_image_names=(),
        payload=np.ones((4, 5), dtype=np.float32),
        reference_domain=CellProfilerMeasurementImageDomain.OBJECT_LABELS,
    )

    measurement_labels = _measurement_labels_for_measurement_image(
        measurement_image,
        labels,
        adapter=Adapter(),
    )

    assert measurement_labels.shape == label_planes.shape
    np.testing.assert_array_equal(measurement_labels, label_planes)


def test_measurement_labels_preserve_payload_scoped_volume_for_source_binding() -> None:
    class Adapter(RuntimePlaneAxisProjector):
        def runtime_slice_plane_index(self) -> int | None:
            return None

        def source_binding_axis_plane_index(
            self,
            source_aliases: tuple[str, ...],
        ) -> int | None:
            return {"rawDNA": 0}.get(source_aliases[0])

    image = np.ones((3, 4, 5), dtype=np.float32)
    label_volume = np.stack(
        (
            np.full((4, 5), 1, dtype=np.int32),
            np.full((4, 5), 2, dtype=np.int32),
            np.full((4, 5), 3, dtype=np.int32),
        )
    )
    labels = ObjectLabelSet(
        name="Cells",
        labels=label_volume,
        source_image_name="MembFinal",
        domain_scope=ObjectLabelDomainScope.PAYLOAD,
    )
    measurement_image = CellProfilerMeasurementImage(
        source_image_name="rawDNA",
        source_image_names=("rawDNA",),
        payload=image,
    )

    measurement_labels = _measurement_labels_for_measurement_image(
        measurement_image,
        labels,
        adapter=Adapter(),
    )

    assert measurement_labels.shape == label_volume.shape
    np.testing.assert_array_equal(measurement_labels, label_volume)


def test_object_label_payload_preserves_source_metadata_for_measurements() -> None:
    image = np.ones((4, 5), dtype=np.float32)
    labels = ObjectLabelSet(
        name="Nuclei",
        labels=np.ones((4, 5), dtype=np.int32),
        source_image_name="rawDNA",
    )
    runtime = _FakeCellProfilerRuntime(
        {"rawDNA": _FakeRuntimeImage(image)},
        {"Nuclei": labels},
    )
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="MeasureObjectIntensity",
            inputs=(
                ArtifactSpec("rawDNA", ArtifactKind.IMAGE),
                ArtifactSpec("Nuclei", ArtifactKind.OBJECT_LABELS),
            ),
            runtime_artifact_inputs=(ArtifactSpec("Nuclei", ArtifactKind.OBJECT_LABELS),),
            outputs=(ArtifactSpec("Measurements", ArtifactKind.MEASUREMENTS),),
        )
    )

    payload = executor._object_label_payload(
        ArtifactSpec("Nuclei", ArtifactKind.OBJECT_LABELS),
        runtime,
        image,
    )

    assert payload is labels
    assert payload.source_image_name == "rawDNA"


def test_measurement_image_for_labels_uses_object_label_source_spatial_crop() -> None:
    image = image_payload_with_context(
        np.arange(25, dtype=np.float32).reshape(5, 5),
        metadata=ImagePayloadMetadata(
            spatial_origin_yx=(0, 0),
            source_spatial_shape_yx=(5, 5),
        ),
    )
    label_payload = ObjectLabelPayload(
        labels=np.ones((2, 2), dtype=np.int32),
        spatial_origin_yx=(1, 1),
        source_spatial_shape_yx=(5, 5),
    )

    aligned = _measurement_image_for_labels(
        image,
        label_payload.labels,
        label_payload=label_payload,
        reference_domain=CellProfilerMeasurementImageDomain.OBJECT_LABELS,
    )

    np.testing.assert_array_equal(
        np.asarray(aligned),
        np.asarray([[6.0, 7.0], [11.0, 12.0]], dtype=np.float32),
    )


def test_image_request_source_name_uses_primary_images_not_object_inputs() -> None:
    image = np.ones((4, 5), dtype=np.float32)
    runtime = _FakeCellProfilerRuntime(
        {"rawGFP": _FakeRuntimeImage(image)},
        {
            "Nuclei": ObjectLabelSet(
                name="Nuclei",
                labels=np.ones((4, 5), dtype=np.int32),
                source_image_name="rawDNA",
            )
        },
    )
    executor = CellProfilerModuleExecutor(
        ModuleArtifactContract(
            module_name="IdentifySecondaryObjects",
            inputs=(
                ArtifactSpec("Nuclei", ArtifactKind.OBJECT_LABELS),
                ArtifactSpec("rawGFP", ArtifactKind.IMAGE),
            ),
            runtime_artifact_inputs=(
                ArtifactSpec("Nuclei", ArtifactKind.OBJECT_LABELS),
            ),
            outputs=(ArtifactSpec("Cells", ArtifactKind.OBJECT_LABELS),),
        )
    )

    image_request = executor._image_request(lambda image: image, image, runtime)

    assert image_request.source_image_name == "rawGFP"


def test_object_only_reference_image_uses_one_stack_plane() -> None:
    image = np.arange(3 * 4 * 5, dtype=np.uint16).reshape(3, 4, 5)

    reference_image = _object_only_reference_image(image)

    assert reference_image.shape == (4, 5)
    np.testing.assert_array_equal(reference_image, image[0])


def test_object_only_reference_image_collapses_high_rank_carrier_to_plane() -> None:
    image = np.arange(2 * 3 * 4 * 5, dtype=np.uint16).reshape(2, 3, 4, 5)

    reference_image = _object_only_reference_image(image)

    assert reference_image.shape == (4, 5)
    np.testing.assert_array_equal(reference_image, image[0, 0])


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
        *_relationship_and_outline_outputs,
    ) = result
    primary_outline, cells_outline = _relationship_and_outline_outputs[-2:]

    assert stats.objects_pre_filter == 2
    assert stats.objects_post_filter == 1
    assert object_label_dense_array(filtered_primary).max() == 1
    assert filtered_primary[3, 3] == 1
    assert object_label_dense_array(filtered_cells).max() == 1
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

    _output_image, stats, filtered_primary = result[:3]

    assert stats.objects_pre_filter == 2
    assert stats.objects_post_filter == 1
    assert filtered_primary[1, 1] == 0
    assert filtered_primary[3, 3] == 1


def test_filterobjects_feature_rules_use_bound_measurement_values() -> None:
    image = np.zeros((5, 5), dtype=np.float32)
    primary = np.zeros((5, 5), dtype=np.int32)
    primary[1:3, 1:3] = 1
    primary[3:5, 3:5] = 2

    result = filter_objects(
        image,
        mode=FilterMode.MEASUREMENTS,
        filter_method=FilterMethod.LIMITS,
        object_labels=(primary,),
        measurement_values=np.array([0.1, 0.8], dtype=np.float64),
        measurement_features=("Intensity_LowerQuartileIntensity_DNA",),
        measurement_min_values=(0.2,),
        measurement_max_values=(None,),
        measurement_use_minimum=(True,),
        measurement_use_maximum=(False,),
        measurement_tables=(
            MeasurementTable(
                name="UnrelatedMeasurements",
                rows=({"object_label": 1, "AreaShape_Area": 4.0},),
            ),
        ),
        dtype_config=DtypeConfig(),
    )

    _output_image, stats, filtered_primary = result[:3]

    assert stats.objects_pre_filter == 2
    assert stats.objects_post_filter == 1
    assert filtered_primary[1, 1] == 0
    assert filtered_primary[3, 3] == 1


def test_filterobjects_binds_measurements_to_sparse_object_label_ids() -> None:
    image = np.zeros((6, 6), dtype=np.float32)
    primary = np.zeros((6, 6), dtype=np.int32)
    primary[1:3, 1:3] = 3
    primary[3:5, 3:5] = 5
    measurement_rows = [
        {"object_label": 3, "lower_quartile_intensity": 0.1},
        {"object_label": 5, "lower_quartile_intensity": 0.8},
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

    _output_image, stats, filtered_primary = result[:3]

    assert stats.objects_pre_filter == 2
    assert stats.objects_post_filter == 1
    assert filtered_primary[1, 1] == 0
    assert filtered_primary[3, 3] == 1


def test_filterobjects_keeps_maximal_child_per_enclosing_object() -> None:
    image = np.zeros((6, 6), dtype=np.float32)
    children = np.zeros((6, 6), dtype=np.int32)
    children[0:2, 0:2] = 1
    children[0:2, 3:5] = 2
    children[3:5, 0:2] = 3
    children[3:5, 3:5] = 4
    parents = np.zeros_like(children)
    parents[0:2, :] = 1
    parents[3:5, :] = 2

    result = filter_objects(
        image,
        mode=FilterMode.MEASUREMENTS,
        filter_method=FilterMethod.MAXIMAL_PER_OBJECT,
        object_labels=(children,),
        enclosing_object_labels=parents,
        per_object_assignment=PerObjectAssignment.BOTH_PARENTS,
        measurement_features=("AreaShape_Area",),
        measurement_tables=(
            MeasurementTable(
                name="ChildMeasurements",
                rows=[
                    {"object_label": 1, "AreaShape_Area": 10.0},
                    {"object_label": 2, "AreaShape_Area": 20.0},
                    {"object_label": 3, "AreaShape_Area": 40.0},
                    {"object_label": 4, "AreaShape_Area": 30.0},
                ],
            ),
        ),
        dtype_config=DtypeConfig(),
    )

    _output_image, stats, filtered_children = result[:3]

    assert stats.objects_pre_filter == 4
    assert stats.objects_post_filter == 2
    assert filtered_children[0, 0] == 0
    assert filtered_children[0, 3] == 1
    assert filtered_children[3, 0] == 2
    assert filtered_children[3, 3] == 0


def test_filterobjects_filters_by_children_count_relationship() -> None:
    image = np.zeros((6, 6), dtype=np.float32)
    nuclei = np.zeros((6, 6), dtype=np.int32)
    nuclei[0:2, 0:2] = 1
    nuclei[2:4, 2:4] = 2
    nuclei[4:6, 4:6] = 3
    semantics = RelationshipSemantics.parent_child("Nuclei", "PH3")
    relationship = ObjectRelationship(
        name="Nuclei_PH3_relationships",
        source=semantics.source,
        target=semantics.target,
        source_ids=(1, 3),
        target_ids=(1, 2),
        relationship_type=semantics.relationship_type,
    )

    result = filter_objects(
        image,
        mode=FilterMode.MEASUREMENTS,
        filter_method=FilterMethod.LIMITS,
        object_labels=(nuclei,),
        measurement_features=("Children_PH3_Count",),
        measurement_min_values=(1.0,),
        measurement_max_values=(1.0,),
        measurement_use_minimum=(True,),
        measurement_use_maximum=(False,),
        parent_child_relationships=(relationship,),
        dtype_config=DtypeConfig(),
    )

    _output_image, stats, filtered_nuclei = result[:3]

    assert stats.objects_pre_filter == 3
    assert stats.objects_post_filter == 2
    assert filtered_nuclei[0, 0] == 1
    assert filtered_nuclei[2, 2] == 0
    assert filtered_nuclei[4, 4] == 2


def test_filterobjects_both_parents_tie_uses_cellprofiler_pixel_order() -> None:
    image = np.zeros((6, 6), dtype=np.float32)
    children = np.zeros((6, 6), dtype=np.int32)
    children[0:2, 0:2] = 2
    children[4:6, 4:6] = 1
    parents = np.ones_like(children)

    result = filter_objects(
        image,
        mode=FilterMode.MEASUREMENTS,
        filter_method=FilterMethod.MAXIMAL_PER_OBJECT,
        object_labels=(children,),
        enclosing_object_labels=parents,
        per_object_assignment=PerObjectAssignment.BOTH_PARENTS,
        measurement_features=("AreaShape_Area",),
        measurement_tables=(
            MeasurementTable(
                name="ChildMeasurements",
                rows=[
                    {"object_label": 1, "AreaShape_Area": 10.0},
                    {"object_label": 2, "AreaShape_Area": 10.0},
                ],
            ),
        ),
        dtype_config=DtypeConfig(),
    )

    _output_image, stats, filtered_children = result[:3]

    assert stats.objects_post_filter == 1
    assert filtered_children[0, 0] == 0
    assert filtered_children[4, 4] == 1


def test_filterobjects_both_parents_minimal_tie_uses_cellprofiler_pixel_order() -> None:
    image = np.zeros((6, 6), dtype=np.float32)
    children = np.zeros((6, 6), dtype=np.int32)
    children[0:2, 0:2] = 2
    children[4:6, 4:6] = 1
    parents = np.ones_like(children)

    result = filter_objects(
        image,
        mode=FilterMode.MEASUREMENTS,
        filter_method=FilterMethod.MINIMAL_PER_OBJECT,
        object_labels=(children,),
        enclosing_object_labels=parents,
        per_object_assignment=PerObjectAssignment.BOTH_PARENTS,
        measurement_features=("AreaShape_Area",),
        measurement_tables=(
            MeasurementTable(
                name="ChildMeasurements",
                rows=[
                    {"object_label": 1, "AreaShape_Area": 10.0},
                    {"object_label": 2, "AreaShape_Area": 10.0},
                ],
            ),
        ),
        dtype_config=DtypeConfig(),
    )

    _output_image, stats, filtered_children = result[:3]

    assert stats.objects_post_filter == 1
    assert filtered_children[0, 0] == 1
    assert filtered_children[4, 4] == 0


def test_filterobjects_both_parents_keeps_single_child_for_sparse_parent() -> None:
    image = np.zeros((6, 6), dtype=np.float32)
    children = np.zeros((6, 6), dtype=np.int32)
    children[0:2, 0:2] = 1
    children[4:6, 4:6] = 2
    parents = np.zeros_like(children)
    parents[0:2, 0:2] = 1
    parents[4:6, 4:6] = 3

    result = filter_objects(
        image,
        mode=FilterMode.MEASUREMENTS,
        filter_method=FilterMethod.MAXIMAL_PER_OBJECT,
        object_labels=(children,),
        enclosing_object_labels=parents,
        per_object_assignment=PerObjectAssignment.BOTH_PARENTS,
        measurement_features=("AreaShape_Area",),
        measurement_tables=(
            MeasurementTable(
                name="ChildMeasurements",
                rows=[
                    {"object_label": 1, "AreaShape_Area": 10.0},
                    {"object_label": 2, "AreaShape_Area": 40.0},
                ],
            ),
        ),
        dtype_config=DtypeConfig(),
    )

    _output_image, stats, filtered_children = result[:3]

    assert stats.objects_post_filter == 2
    assert filtered_children[0, 0] == 1
    assert filtered_children[4, 4] == 2


def test_filterobjects_both_parents_uses_all_pixel_overlaps() -> None:
    image = np.zeros((5, 6), dtype=np.float32)
    children = np.zeros((5, 6), dtype=np.int32)
    children[1:3, 0:2] = 1
    children[1:3, 2:5] = 2
    parents = np.zeros_like(children)
    parents[:, 0:3] = 1
    parents[:, 3:6] = 2

    result = filter_objects(
        image,
        mode=FilterMode.MEASUREMENTS,
        filter_method=FilterMethod.MAXIMAL_PER_OBJECT,
        object_labels=(children,),
        enclosing_object_labels=parents,
        parent_child_relationship=ParentChildRelationshipPayload(
            parent_ids=(1, 2),
            child_ids=(1, 2),
        ),
        per_object_assignment=PerObjectAssignment.BOTH_PARENTS,
        measurement_features=("AreaShape_Area",),
        measurement_tables=(
            MeasurementTable(
                name="ChildMeasurements",
                rows=[
                    {"object_label": 1, "AreaShape_Area": 10.0},
                    {"object_label": 2, "AreaShape_Area": 20.0},
                ],
            ),
        ),
        dtype_config=DtypeConfig(),
    )

    _output_image, stats, filtered_children = result[:3]

    assert stats.objects_post_filter == 1
    assert filtered_children[1, 0] == 0
    assert filtered_children[1, 3] == 1


def test_filterobjects_most_overlap_can_use_relationship_payload() -> None:
    image = np.zeros((5, 6), dtype=np.float32)
    children = np.zeros((5, 6), dtype=np.int32)
    children[1:3, 0:2] = 1
    children[1:3, 2:5] = 2
    parents = np.zeros_like(children)
    parents[:, 0:3] = 1
    parents[:, 3:6] = 2

    result = filter_objects(
        image,
        mode=FilterMode.MEASUREMENTS,
        filter_method=FilterMethod.MAXIMAL_PER_OBJECT,
        object_labels=(children,),
        enclosing_object_labels=parents,
        parent_child_relationship=ParentChildRelationshipPayload(
            parent_ids=(1, 2),
            child_ids=(1, 2),
        ),
        per_object_assignment=PerObjectAssignment.PARENT_WITH_MOST_OVERLAP,
        measurement_features=("AreaShape_Area",),
        measurement_tables=(
            MeasurementTable(
                name="ChildMeasurements",
                rows=[
                    {"object_label": 1, "AreaShape_Area": 10.0},
                    {"object_label": 2, "AreaShape_Area": 20.0},
                ],
            ),
        ),
        dtype_config=DtypeConfig(),
    )

    _output_image, stats, filtered_children = result[:3]

    assert stats.objects_post_filter == 2
    assert filtered_children[1, 0] == 1
    assert filtered_children[1, 3] == 2


def test_filterobjects_aligns_enclosing_label_stack_to_child_plane() -> None:
    image = np.zeros((6, 6), dtype=np.float32)
    children = np.zeros((6, 6), dtype=np.int32)
    children[0:2, 0:2] = 1
    children[0:2, 3:5] = 2
    children[3:5, 0:2] = 3
    children[3:5, 3:5] = 4
    parents = np.zeros_like(children)
    parents[0:2, :] = 1
    parents[3:5, :] = 2

    result = filter_objects(
        image,
        mode=FilterMode.MEASUREMENTS,
        filter_method=FilterMethod.MAXIMAL_PER_OBJECT,
        object_labels=(children,),
        enclosing_object_labels=np.stack((parents, parents)),
        per_object_assignment=PerObjectAssignment.BOTH_PARENTS,
        measurement_features=("AreaShape_Area",),
        measurement_tables=(
            MeasurementTable(
                name="ChildMeasurements",
                rows=[
                    {"object_label": 1, "AreaShape_Area": 10.0},
                    {"object_label": 2, "AreaShape_Area": 20.0},
                    {"object_label": 3, "AreaShape_Area": 40.0},
                    {"object_label": 4, "AreaShape_Area": 30.0},
                ],
            ),
        ),
        dtype_config=DtypeConfig(),
    )

    _output_image, stats, filtered_children = result[:3]

    assert stats.objects_post_filter == 2
    assert filtered_children[0, 3] == 1
    assert filtered_children[3, 0] == 2


def test_structuring_element_execution_policy_uses_full_stack_for_3d_footprint() -> None:
    policy = CellProfilerInvocationExecutionModePolicy.for_module("Closing")
    image = np.zeros((3, 5, 5), dtype=np.float32)

    mode = policy.execution_mode(
        default=ImagePayloadExecutionMode.NATURAL,
        image=image,
        kwargs={
            "structuring_element": StructuringElement.BALL,
            "size": 1,
        },
    )

    assert mode is ImagePayloadExecutionMode.FULL_STACK


def test_structuring_element_execution_policy_keeps_planewise_for_2d_footprint() -> None:
    policy = CellProfilerInvocationExecutionModePolicy.for_module("Closing")
    image = np.zeros((3, 5, 5), dtype=np.float32)

    mode = policy.execution_mode(
        default=ImagePayloadExecutionMode.NATURAL,
        image=image,
        kwargs={
            "structuring_element": StructuringElement.DISK,
            "size": 1,
        },
    )

    assert mode is ImagePayloadExecutionMode.NATURAL


def test_structuring_element_execution_policy_uses_object_label_kwargs_rank() -> None:
    policy = CellProfilerInvocationExecutionModePolicy.for_module("ErodeObjects")
    image = np.zeros((5, 5), dtype=np.float32)
    labels = ObjectLabelSet(
        name="Nuclei",
        labels=np.zeros((3, 5, 5), dtype=np.int32),
    )

    mode = policy.execution_mode(
        default=ImagePayloadExecutionMode.NATURAL,
        image=image,
        kwargs={
            "labels": labels,
            "structuring_element": StructuringElement.BALL,
            "size": 1,
        },
    )

    assert mode is ImagePayloadExecutionMode.FULL_STACK


def test_object_measurement_execution_policy_uses_full_stack_for_3d_labels() -> None:
    policy = CellProfilerObjectMeasurementExecutionDomainPolicy.for_module(
        "MeasureObjectSizeShape"
    )
    labels = ObjectLabelSet(
        name="Nuclei",
        labels=np.zeros((3, 5, 5), dtype=np.int32),
    )

    mode = policy.execution_mode(
        measure_object_size_shape,
        labels,
        ImagePayloadExecutionMode.NATURAL,
    )

    assert mode is ImagePayloadExecutionMode.FULL_STACK


def test_object_measurement_execution_policy_keeps_runtime_slice_stack_natural() -> None:
    policy = CellProfilerObjectMeasurementExecutionDomainPolicy.for_module(
        "MeasureObjectSizeShape"
    )
    labels = ObjectLabelSet(
        name="Nuclei",
        labels=np.zeros((3, 5, 5), dtype=np.int32),
        domain_scope=ObjectLabelDomainScope.PLANE,
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    )

    mode = policy.execution_mode(
        measure_object_size_shape,
        labels,
        ImagePayloadExecutionMode.NATURAL,
        runtime_slice_count=1,
    )

    assert mode is ImagePayloadExecutionMode.NATURAL


def test_object_measurement_execution_policy_keeps_2d_labels_natural() -> None:
    policy = CellProfilerObjectMeasurementExecutionDomainPolicy.for_module(
        "MeasureObjectSizeShape"
    )
    labels = ObjectLabelSet(
        name="Nuclei",
        labels=np.zeros((5, 5), dtype=np.int32),
    )

    mode = policy.execution_mode(
        measure_object_size_shape,
        labels,
        ImagePayloadExecutionMode.NATURAL,
    )

    assert mode is ImagePayloadExecutionMode.NATURAL
