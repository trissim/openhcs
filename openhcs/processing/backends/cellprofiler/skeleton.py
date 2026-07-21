"""CellProfiler-compatible skeleton measurement backends."""

from __future__ import annotations
from collections.abc import Callable
from typing import Annotated, TYPE_CHECKING
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.interop.cellprofiler.module_declarations import (
    CellProfilerModule,
)
from openhcs.interop.cellprofiler.module_artifact_declarations import (
    ImageMeasurementInputModule,
    MeasurementArtifactOutputModule,
    ObjectArtifactInputModule,
    PlaneRuntimeArtifactModule,
    SourceQualifiedWideMeasurementRowsModule,
)
from dataclasses import dataclass
import numpy as np
import scipy.ndimage
from skimage.morphology import remove_small_holes, skeletonize
from openhcs.core.aligned_image_payload import (
    AlignedImageSliceContext,
    pack_aligned_image_outputs,
)
from openhcs.core.artifacts import ObjectLabelsArtifactType, ImageArtifactType
from openhcs.core.memory.decorators import numpy as numpy_backend
from openhcs.core.measurement_row_materialization import (
    DataclassMeasurementColumnarRows,
)
from openhcs.core.runtime_measurements import (
    MeasurementRowAxisField,
    RuntimeMeasurementFeature,
)
from openhcs.core.pipeline.function_contracts import special_inputs
from openhcs.core.public_api import public_names_from_objects
from openhcs.core.runtime_object_labels import (
    ObjectLabelValue,
    object_label_dense_array,
)
from openhcs.core.runtime_array_values import RuntimeArrayData
from openhcs.core.runtime_image_values import (
    image_payload_metadata,
    with_image_payload_data,
)
from openhcs.interop.cellprofiler.parser import ModuleSetting
from openhcs.interop.cellprofiler.setting_names import (
    optional_setting_value,
    required_setting_value,
)
from openhcs.interop.cellprofiler.settings_binder import (
    SettingToKeywordBinding,
    parse_cellprofiler_bool,
    parse_cellprofiler_int,
)
from openhcs.interop.cellprofiler.runtime.measurement_recording import (
    MeasurementFeatureRecord,
)

SeedObjectLabelsInput = Annotated[
    ObjectLabelValue,
    (
        "Seed-object labels that assign skeleton pixels, branches, and endpoint "
        "measurements to their originating objects."
    ),
]

if TYPE_CHECKING:
    from openhcs.interop.cellprofiler.parser import ModuleBlock

EIGHT_NEIGHBOR_KERNEL = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)


@dataclass(frozen=True, slots=True)
class SkeletonMeasurement(MeasurementFeatureRecord):
    """Measurements from skeleton analysis."""

    slice_index: Annotated[int, MeasurementRowAxisField.SLICE_INDEX]
    branches: int
    endpoints: int


@dataclass(frozen=True, slots=True)
class ObjectSkeletonMeasurement:
    """Measurements for skeleton branching structures per seed object."""

    slice_index: int
    object_label: int
    number_trunks: int
    number_non_trunk_branches: int
    number_branch_ends: int
    total_skeleton_length: float


@dataclass(frozen=True, slots=True)
class ObjectSkeletonSliceResult:
    """Measurements and the exact CellProfiler branchpoint visualization."""

    measurements: tuple[ObjectSkeletonMeasurement, ...]
    branchpoint_image: np.ndarray


@dataclass(frozen=True, slots=True)
class SkeletonNeighborhood:
    """Neighbor-count semantics for 2-D and 3-D skeleton measurements."""

    image: np.ndarray

    @property
    def binary(self) -> np.ndarray:
        return (self.image > 0).astype(np.uint8)

    def neighbor_counts(self) -> np.ndarray:
        binary = self.binary
        padding = np.pad(binary, 1, mode="constant", constant_values=0)
        mask = padding > 0
        response = (
            3**binary.ndim
            * scipy.ndimage.uniform_filter(padding.astype(np.float64), size=3)
            - 1
        )
        interior = tuple((slice(1, -1) for _ in range(binary.ndim)))
        return (response * mask)[interior].astype(np.uint16)

    def measurement(self, *, slice_index: int = 0) -> SkeletonMeasurement:
        neighbors = self.neighbor_counts()
        return SkeletonMeasurement(
            slice_index=slice_index,
            branches=int(np.count_nonzero(neighbors > 2)),
            endpoints=int(np.count_nonzero(neighbors == 1)),
        )


@dataclass(frozen=True, slots=True)
class DiskStructuringElement:
    """Disk footprint used by CellProfiler object-skeleton measurements."""

    radius: float

    def footprint(self) -> np.ndarray:
        radius = int(self.radius + 0.5)
        y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
        return (x * x + y * y <= self.radius * self.radius).astype(np.uint8)


@dataclass(frozen=True, slots=True)
class SkeletonLabelPropagation:
    """Propagate seed labels onto the skeleton support."""

    labels: np.ndarray
    mask: np.ndarray

    def propagate(self) -> tuple[np.ndarray, np.ndarray]:
        distance = scipy.ndimage.distance_transform_edt(self.labels == 0)
        propagated = self.labels.copy()
        max_distance = int(np.max(distance[self.mask])) + 1 if np.any(self.mask) else 0
        for _ in range(max_distance):
            dilated = scipy.ndimage.grey_dilation(propagated, size=3)
            propagated = np.where((propagated == 0) & self.mask, dilated, propagated)
        return (propagated, distance)


@dataclass(frozen=True, slots=True)
class ObjectSkeletonSliceMeasurement:
    """Seed-relative skeleton measurements for one 2-D plane."""

    skeleton: np.ndarray
    seed_labels: np.ndarray
    slice_index: int
    fill_small_holes: bool
    maximum_hole_size: int

    def analyze(self) -> ObjectSkeletonSliceResult:
        labels = self.seed_labels.astype(np.int32)
        label_count = int(np.max(labels))
        label_range = np.arange(1, label_count + 1, dtype=np.int32)
        disk = DiskStructuringElement(1.5).footprint()
        dilated_labels = scipy.ndimage.grey_dilation(labels, footprint=disk)
        seed_mask = dilated_labels > 0
        combined_skeleton = (self.skeleton > 0) | seed_mask
        closed_labels = scipy.ndimage.grey_erosion(dilated_labels, footprint=disk)
        combined_skeleton = combined_skeleton & ~(closed_labels > 0)
        if self.fill_small_holes:
            combined_skeleton = remove_small_holes(
                combined_skeleton, area_threshold=self.maximum_hole_size
            )
        combined_skeleton = skeletonize(combined_skeleton)
        outside_skeleton = combined_skeleton & (dilated_labels == 0)
        propagated_labels, distance_map = SkeletonLabelPropagation(
            labels=dilated_labels, mask=combined_skeleton
        ).propagate()
        combined_skeleton = combined_skeleton & (propagated_labels > 0)
        branch_points = SkeletonConvolutionFeatures(combined_skeleton).branchpoints()
        end_points = SkeletonConvolutionFeatures(combined_skeleton).endpoints()
        branching_counts = SkeletonConvolutionFeatures(
            combined_skeleton
        ).branching_counts()
        dilated_skeleton = scipy.ndimage.binary_dilation(
            outside_skeleton, structure=np.ones((3, 3))
        )
        branching_counts[~dilated_skeleton] = 0
        nearby_labels = propagated_labels.copy()
        nearby_labels[distance_map > 1.5] = 0
        outside_labels = propagated_labels.copy()
        outside_labels[nearby_labels > 0] = 0
        trunk_counts = np.array(
            [
                int(np.sum(branching_counts[nearby_labels == label]))
                for label in label_range
            ],
            dtype=np.int32,
        )
        branch_counts = np.array(
            [
                int(np.sum(branch_points[outside_labels == label]))
                for label in label_range
            ],
            dtype=np.int32,
        )
        end_counts = np.array(
            [int(np.sum(end_points[outside_labels == label])) for label in label_range],
            dtype=np.int32,
        )
        total_distance = SkeletonLengthByLabel(
            labels=propagated_labels * outside_skeleton.astype(np.int32),
            label_range=label_range,
        ).lengths()
        measurements = tuple(
            ObjectSkeletonMeasurement(
                slice_index=self.slice_index,
                object_label=int(label),
                number_trunks=int(trunk_counts[index]),
                number_non_trunk_branches=int(branch_counts[index]),
                number_branch_ends=int(end_counts[index]),
                total_skeleton_length=(
                    float(total_distance[index]) if index < len(total_distance) else 0.0
                ),
            )
            for index, label in enumerate(label_range)
        )
        trunk_mask = (branching_counts > 0) & (nearby_labels != 0)
        branch_mask = branch_points & (outside_labels != 0)
        end_mask = end_points & (outside_labels != 0)
        branchpoint_image = np.zeros((*self.skeleton.shape, 3), dtype=float)
        branchpoint_image[outside_skeleton, :] = 1
        branchpoint_image[trunk_mask | branch_mask | end_mask, :] = 0
        branchpoint_image[trunk_mask, 0] = 1
        branchpoint_image[branch_mask, 1] = 1
        branchpoint_image[end_mask, 2] = 1
        branchpoint_image[dilated_labels != 0, :] *= 0.875
        branchpoint_image[dilated_labels != 0, :] += 0.1
        return ObjectSkeletonSliceResult(measurements, branchpoint_image)

    def measurements(self) -> list[ObjectSkeletonMeasurement]:
        return list(self.analyze().measurements)


@dataclass(frozen=True, slots=True)
class SkeletonConvolutionFeatures:
    """2-D skeleton branch and endpoint features from CP neighbor semantics."""

    skeleton: np.ndarray

    def neighbor_counts(self) -> np.ndarray:
        return scipy.ndimage.convolve(
            self.skeleton.astype(np.uint8),
            EIGHT_NEIGHBOR_KERNEL,
            mode="constant",
            cval=0,
        )

    def branchpoints(self) -> np.ndarray:
        return (self.skeleton > 0) & (self.neighbor_counts() > 2)

    def endpoints(self) -> np.ndarray:
        return (self.skeleton > 0) & (self.neighbor_counts() == 1)

    def branching_counts(self) -> np.ndarray:
        counts = np.clip(self.neighbor_counts() - 2, 0, 2)
        counts[~self.skeleton] = 0
        return counts


@dataclass(frozen=True, slots=True)
class SkeletonLengthByLabel:
    """Skeleton length aggregation over propagated seed labels."""

    labels: np.ndarray
    label_range: np.ndarray

    def lengths(self) -> np.ndarray:
        if len(self.label_range) == 0:
            return np.zeros(0)
        lengths = scipy.ndimage.sum(self.labels > 0, self.labels, self.label_range)
        return np.atleast_1d(lengths).astype(float)


@numpy_backend(contract=ProcessingContract.PURE_2D)
def measure_image_skeleton(
    image: np.ndarray,
) -> tuple[np.ndarray, DataclassMeasurementColumnarRows]:
    """Measure branches and endpoints in a 2-D skeletonized image."""
    return (
        image,
        DataclassMeasurementColumnarRows(
            (SkeletonNeighborhood(image).measurement(),),
            row_type=SkeletonMeasurement,
        ),
    )


@numpy_backend(contract=ProcessingContract.PURE_3D)
def measure_image_skeleton_3d(
    image: np.ndarray,
) -> tuple[np.ndarray, DataclassMeasurementColumnarRows]:
    """Measure branches and endpoints in a 3-D skeletonized image."""
    return (
        image,
        DataclassMeasurementColumnarRows(
            (SkeletonNeighborhood(image).measurement(),),
            row_type=SkeletonMeasurement,
        ),
    )


@numpy_backend(contract=ProcessingContract.PURE_2D)
@special_inputs("seed_labels")
def measure_object_skeleton(
    image: np.ndarray,
    seed_labels: SeedObjectLabelsInput,
    fill_small_holes: bool = True,
    maximum_hole_size: int = 10,
) -> tuple[RuntimeArrayData, DataclassMeasurementColumnarRows]:
    """Measure branching structures in skeletonized images relative to seed objects."""

    result = _object_skeleton_slice_analysis(
        image,
        seed_labels,
        fill_small_holes=fill_small_holes,
        maximum_hole_size=maximum_hole_size,
    )
    return (
        image,
        DataclassMeasurementColumnarRows(
            result.measurements,
            row_type=ObjectSkeletonMeasurement,
        ),
    )


@numpy_backend(contract=ProcessingContract.PURE_2D)
@special_inputs("seed_labels")
def measure_object_skeleton_with_branchpoint_image(
    image: np.ndarray,
    seed_labels: SeedObjectLabelsInput,
    fill_small_holes: bool = True,
    maximum_hole_size: int = 10,
    branchpoint_image_name: str = "BranchpointImage",
) -> tuple[RuntimeArrayData, DataclassMeasurementColumnarRows]:
    """Measure object skeletons and retain the named branchpoint RGB image."""

    if not branchpoint_image_name.strip():
        raise ValueError(
            "MeasureObjectSkeleton branchpoint image name cannot be blank."
        )
    result = _object_skeleton_slice_analysis(
        image,
        seed_labels,
        fill_small_holes=fill_small_holes,
        maximum_hole_size=maximum_hole_size,
    )
    branchpoint_payload = with_image_payload_data(
        image,
        result.branchpoint_image,
        metadata=image_payload_metadata(image).replace_fields(source_channel_axis=-1),
    )
    branchpoint_image = pack_aligned_image_outputs(
        (branchpoint_payload,),
        slice_contexts=(
            AlignedImageSliceContext.main_flow(
                branchpoint_image_name,
                artifact_kind=ImageArtifactType.value,
            ),
        ),
    )
    return (
        branchpoint_image,
        DataclassMeasurementColumnarRows(
            result.measurements,
            row_type=ObjectSkeletonMeasurement,
        ),
    )


def _object_skeleton_slice_analysis(
    image: np.ndarray,
    seed_labels: ObjectLabelValue,
    *,
    fill_small_holes: bool,
    maximum_hole_size: int,
) -> ObjectSkeletonSliceResult:
    if not isinstance(seed_labels, ObjectLabelValue):
        raise TypeError(
            "MeasureObjectSkeleton requires a runtime-projected ObjectLabelValue."
        )
    image_plane = np.asarray(image)
    label_plane = object_label_dense_array(seed_labels, dtype=np.int32)
    if image_plane.ndim != 2 or label_plane.ndim != 2:
        raise ValueError(
            "MeasureObjectSkeleton requires runtime-projected 2-D image and label planes."
        )
    return ObjectSkeletonSliceMeasurement(
        skeleton=image_plane,
        seed_labels=label_plane,
        slice_index=0,
        fill_small_holes=fill_small_holes,
        maximum_hole_size=maximum_hole_size,
    ).analyze()


class MeasureImageSkeletonModule(
    SourceQualifiedWideMeasurementRowsModule,
    ImageMeasurementInputModule,
):
    module_name = "MeasureImageSkeleton"
    function_name = "measure_image_skeleton"
    function_variants = ("measure_image_skeleton_3d",)
    validated = True
    confidence = 1.0
    measurement_category_prefixes = (("skeleton",),)

    class MeasurementFeature(RuntimeMeasurementFeature):
        """Image-skeleton feature families emitted by CellProfiler."""

        BRANCHES = "Branches"
        ENDPOINTS = "Endpoints"


class MeasureObjectSkeletonModule(
    PlaneRuntimeArtifactModule,
    ObjectArtifactInputModule,
    MeasurementArtifactOutputModule,
    CellProfilerModule,
):
    module_name = "MeasureObjectSkeleton"
    function_name = "measure_object_skeleton"
    function_variants = ("measure_object_skeleton_with_branchpoint_image",)
    validated = True
    confidence = 1.0

    seed_objects_setting = "Select the seed objects"
    skeleton_image_setting = "Select the skeletonized image"
    retain_branchpoint_image_setting = "Retain the branchpoint image?"
    branchpoint_image_setting = "Name the branchpoint image"
    fill_small_holes_setting = "Fill small holes?"
    maximum_hole_size_setting = "Maximum hole size"
    export_graph_setting = "Export the skeleton graph relationships?"
    intensity_image_setting = "Intensity image"
    graph_directory_setting = "File output directory"
    vertex_file_setting = "Vertex file name"
    edge_file_setting = "Edge file name"
    branchpoint_image_binding = SettingToKeywordBinding.output(
        branchpoint_image_setting,
        ImageArtifactType,
        "branchpoint_image_name",
    )
    setting_bindings = (
        SettingToKeywordBinding.input(skeleton_image_setting, ImageArtifactType),
        SettingToKeywordBinding.input(
            seed_objects_setting,
            ObjectLabelsArtifactType,
            runtime_parameter_name="seed_labels",
        ),
        branchpoint_image_binding,
        SettingToKeywordBinding(
            fill_small_holes_setting,
            "fill_small_holes",
            parse_cellprofiler_bool,
        ),
        SettingToKeywordBinding(
            maximum_hole_size_setting,
            "maximum_hole_size",
            parse_cellprofiler_int,
        ),
    )
    ignored_settings = (
        retain_branchpoint_image_setting,
        export_graph_setting,
        intensity_image_setting,
        graph_directory_setting,
        vertex_file_setting,
        edge_file_setting,
    )

    @classmethod
    def retain_branchpoint_image(cls, module: "ModuleBlock") -> bool:
        return parse_cellprofiler_bool(
            required_setting_value(module, cls.retain_branchpoint_image_setting)
        )

    @classmethod
    def active_artifact_bindings(cls, module=None, *, invocation_key=None):
        bindings = super().active_artifact_bindings(
            module,
            invocation_key=invocation_key,
        )
        if module is None:
            return bindings
        retain_setting = optional_setting_value(
            module,
            cls.retain_branchpoint_image_setting,
        )
        if retain_setting is None:
            if invocation_key is None:
                required_setting_value(
                    module,
                    cls.retain_branchpoint_image_setting,
                )
            retain_branchpoint_image = (
                invocation_key.function_name == cls.function_variants[0]
            )
        else:
            retain_branchpoint_image = parse_cellprofiler_bool(retain_setting)
        return tuple(
            binding
            for binding in bindings
            if retain_branchpoint_image
            or binding is not cls.branchpoint_image_binding
        )

    @classmethod
    def main_flow_output_specs(cls, main_flow_candidates):
        """Record the retained image without replacing the skeleton main flow."""

        del cls, main_flow_candidates
        return ()

    @classmethod
    def bind_settings(cls, module, *, binder):
        graph_setting = optional_setting_value(module, cls.export_graph_setting)
        if graph_setting is not None and parse_cellprofiler_bool(graph_setting):
            raise NotImplementedError(
                "MeasureObjectSkeleton graph export is not supported by the "
                "absorbed callable."
            )
        return super().bind_settings(module, binder=binder)

    @classmethod
    def resolve_function(
        cls,
        module: "ModuleBlock",
        *,
        contract,
        source_bindings,
    ) -> Callable[..., object]:
        """Select the retained-image ABI from the declared retain setting."""

        del contract, source_bindings
        return cls.require_callable(
            cls.function_variants[0]
            if cls.retain_branchpoint_image(module)
            else cls.function_name
        )

    @classmethod
    def _derived_identity_setting_records(
        cls,
        *,
        invocation,
        block_position,
        existing_records,
        step_context,
    ):
        """Reconstruct the retain condition from the explicit callable ABI."""

        setting_key = cls.normalize_setting_name(cls.retain_branchpoint_image_setting)
        own_records = (
            ()
            if setting_key in cls._normalized_record_setting_names(existing_records)
            else (
                ModuleSetting(
                    cls.retain_branchpoint_image_setting,
                    (
                        "Yes"
                        if invocation.contract.function_name == cls.function_variants[0]
                        else "No"
                    ),
                ),
            )
        )
        return (
            *own_records,
            *super()._derived_identity_setting_records(
                invocation=invocation,
                block_position=block_position,
                existing_records=(*existing_records, *own_records),
                step_context=step_context,
            ),
        )


__all__ = public_names_from_objects(
    DiskStructuringElement,
    "EIGHT_NEIGHBOR_KERNEL",
    ObjectSkeletonMeasurement,
    ObjectSkeletonSliceMeasurement,
    SkeletonConvolutionFeatures,
    SkeletonLabelPropagation,
    SkeletonLengthByLabel,
    SkeletonMeasurement,
    SkeletonNeighborhood,
    measure_image_skeleton,
    measure_image_skeleton_3d,
    measure_object_skeleton,
    measure_object_skeleton_with_branchpoint_image,
)
