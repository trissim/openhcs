"""Secondary-object backend strategies for CellProfiler-compatible processing."""

from __future__ import annotations
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Annotated, Callable, ClassVar, Tuple, TypeAlias

import numpy as np
from metaclass_registry import AutoRegisterMeta
from numba import njit
from python_introspect import set_signature_analysis_target

from openhcs.constants.constants import MemoryType
from openhcs.core.artifacts import (
    ArtifactSpec,
    ArtifactSpecCollection,
    ImageArtifactType,
    InputGroupLineageSourceRelation,
    ObjectLabelsArtifactType,
    SourceStackLineageSourceRelation,
)
from openhcs.core.callable_contract import processing_prepare
from openhcs.core.memory import numpy
from openhcs.core.measurement_row_materialization import (
    DataclassMeasurementColumnarRows,
    MeasurementProjectedColumnarRows,
    MeasurementSparseColumnarRows,
)
from openhcs.core.pipeline.function_contracts import (
    special_inputs,
    )
from openhcs.core.public_api import public_names_from_objects
from openhcs.core.registry_strategies import (
    EnumKeyedStrategyMixin,
    RegisteredLeafClassSpec,
)
from openhcs.core.runtime_image_values import (
    image_payload_data,
    image_payload_mask,
    image_payload_metadata,
)
from openhcs.core.runtime_object_labels import (
    ObjectLabelPayload,
    ObjectLabelValue,
    ObjectLabelVariantData,
    object_label_dense_array,
    object_label_value_with_dense_labels,
)
from openhcs.core.runtime_object_label_building import (
    SourceImageObjectLabelBuildRequest,
)
from openhcs.core.runtime_measurements import (
    MeasurementRowAxisField,
)
from openhcs.core.runtime_object_label_domains import (
    ObjectLabelDomain,
    PresentObjectLabelIdsDomainDeclaration,
)
from openhcs.core.runtime_object_labels import (
    ObjectLabelRepresentation,
)
from openhcs.core.runtime_relationships import (
    DirectedObjectRelationshipPayload,
    object_label_parent_child_payload,
)
from openhcs.core.source_spatial_domain import SourceSpatialDomainAdapter
from openhcs.core.runtime_tabular_values import ColumnarRows
from openhcs.interop.cellprofiler.module_declarations import (
    CellProfilerModule,
)
from openhcs.interop.cellprofiler.module_artifact_declarations import (
    MeasurementArtifactOutputModule,
    ObjectArtifactInputModule,
    ObjectArtifactOutputModule,
    ParentChildLineageArtifactOutputModule,
)
from openhcs.interop.cellprofiler.runtime.measurement_recording import (
    NoObjectNameMeasurementRecordMixin,
)
from openhcs.interop.cellprofiler.database_column_dialect import (
    CellProfilerObjectCoreMeasurementFeature,
)
from openhcs.interop.cellprofiler.runtime.object_input_policies import (
    PairedPrimarySecondaryObjectInputPolicy,
    PrimaryObjectLabelInputPolicy,
)
from openhcs.interop.cellprofiler.runtime.primary_image_input_policies import (
    ObjectLabelDrivenPrimaryImageInputPolicy,
)
from openhcs.interop.cellprofiler.setting_names import (
    required_setting_value,
)
from openhcs.interop.cellprofiler.settings_binder import (
    SettingToKeywordBinding,
    coerce_cellprofiler_enum,
    parse_cellprofiler_bool,
)
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract

if TYPE_CHECKING:
    from openhcs.core.function_patterns import FunctionInvocationKey
    from openhcs.core.invocation_artifacts import ArtifactDeclarationStepContext
    from openhcs.interop.cellprofiler.parser import ModuleBlock


class IdentifyTertiaryObjectsModule(
    ObjectLabelDrivenPrimaryImageInputPolicy,
    PairedPrimarySecondaryObjectInputPolicy,
    NoObjectNameMeasurementRecordMixin,
    ObjectArtifactInputModule,
    ObjectArtifactOutputModule,
    ParentChildLineageArtifactOutputModule,
    MeasurementArtifactOutputModule,
    CellProfilerModule,
):
    module_name = "IdentifyTertiaryObjects"
    function_name = "identify_tertiary_objects"
    validated = True
    confidence = 1.0

    @classmethod
    def measurement_record_rows(cls, request) -> ColumnarRows:
        """Emit tertiary object locations in CellProfiler's X/Y scope only."""
        feature_field = MeasurementRowAxisField.FEATURE_NAME.value
        center_z = CellProfilerObjectCoreMeasurementFeature.CENTER_Z.value
        rows = super().measurement_record_rows(request)
        retained_indices = (
            tuple(
                row_index
                for row_index, feature_name in enumerate(
                    rows.column_values(feature_field)
                )
                if feature_name != center_z
            )
            if any(field_spec.name == feature_field for field_spec in rows.fields)
            else None
        )
        return MeasurementProjectedColumnarRows.from_columnar_rows(
            rows,
            row_indices=retained_indices,
            declared_object_measurement_domain_covered=(
                rows.covers_declared_object_measurement_domain
            ),
            object_row_identity=rows.object_row_identity,
        )

    @staticmethod
    def object_inputs(
        artifact_inputs: ArtifactSpecCollection,
    ) -> tuple[ArtifactSpec, ArtifactSpec]:
        """Return larger and smaller object inputs in declared role order."""

        object_inputs = artifact_inputs.for_artifact_type(
            ObjectLabelsArtifactType
        ).specs
        if len(object_inputs) != 2:
            raise ValueError(
                "IdentifyTertiaryObjects requires two ordered object inputs, got "
                f"{object_inputs!r}."
            )
        return object_inputs

    @classmethod
    def finalize_artifact_contract_inputs(
        cls,
        module: "ModuleBlock",
        *,
        invocation_key: "FunctionInvocationKey",
        step_context: "ArtifactDeclarationStepContext",
        artifact_inputs: ArtifactSpecCollection,
    ) -> tuple[ArtifactSpec, ...]:
        inputs = ArtifactSpecCollection(
            super().finalize_artifact_contract_inputs(
                module,
                invocation_key=invocation_key,
                step_context=step_context,
                artifact_inputs=artifact_inputs,
            )
        )
        larger_input, smaller_input = cls.object_inputs(inputs)
        projected_smaller_input = smaller_input.with_group_scope_relation(
            InputGroupLineageSourceRelation(source=larger_input.ref())
        )
        return tuple(
            projected_smaller_input if spec is smaller_input else spec
            for spec in inputs.specs
        )

    larger_objects_setting = "Select the larger identified objects"
    smaller_objects_setting = "Select the smaller identified objects"
    output_objects_setting = "Name the tertiary objects to be identified"
    shrink_primary_setting = "Shrink smaller object prior to subtraction?"
    larger_objects_binding = SettingToKeywordBinding.input(
        larger_objects_setting,
        ObjectLabelsArtifactType,
        runtime_parameter_name="secondary_labels",
    )
    smaller_objects_binding = SettingToKeywordBinding.input(
        smaller_objects_setting,
        ObjectLabelsArtifactType,
        runtime_parameter_name="primary_labels",
    )
    setting_bindings = (
        larger_objects_binding,
        smaller_objects_binding,
        SettingToKeywordBinding.output(
            output_objects_setting, ObjectLabelsArtifactType
        ),
        SettingToKeywordBinding(
            shrink_primary_setting,
            "shrink_primary",
            parse_cellprofiler_bool,
        ),
    )

    @classmethod
    def artifact_contract_outputs(
        cls,
        module,
        *,
        invocation_key,
        step_context,
        artifact_inputs: ArtifactSpecCollection,
    ):
        larger_input, smaller_input = cls.object_inputs(artifact_inputs)
        output_name = required_setting_value(module, cls.output_objects_setting)
        output = ArtifactSpec.output(
            output_name,
            ObjectLabelsArtifactType,
            relations=(SourceStackLineageSourceRelation(source=larger_input.ref()),),
        )
        outputs = [
            cls.parent_child_relationship_output_artifact(
                module,
                step_context=step_context,
                parent=larger_input,
                child=output,
                lineage_source=larger_input,
            ),
            cls.parent_child_relationship_output_artifact(
                module,
                step_context=step_context,
                parent=smaller_input,
                child=output,
                lineage_source=larger_input,
            ),
            cls.measurement_output_artifact(
                module,
                invocation_key=invocation_key,
                step_context=step_context,
                artifact_inputs=artifact_inputs,
            ),
            output,
        ]
        return tuple(outputs)


from openhcs.processing.backends.cellprofiler._backend import (
    DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    BackendProviderInput,
    CellProfilerBackendAuthority,
    CellProfilerBackendProvider,
    CellProfilerBackendStrategyMixin,
)
from openhcs.processing.backends.cellprofiler.distance_propagation_numba import (
    _distance_to_positive_labels_numba,
    _nearest_label_expansion_numba,
    _propagate_labels_and_distances_numba,
    _propagate_labels_and_distances_zero_image_numba,
)
from openhcs.processing.backends.cellprofiler.enum_attributes import (
    CellProfilerEnumAttributeMixin,
)
from openhcs.processing.backends.cellprofiler.granularity import (
    CellProfilerRuntimeProfiler,
)
from openhcs.processing.backends.cellprofiler.image_geometry import (
    CellProfilerPlaneGeometry,
)
from openhcs.processing.backends.cellprofiler.morphology import (
    MorphologyBackendStrategy,
)
from openhcs.processing.backends.cellprofiler.outlines import (
    ObjectOutlineBackendStrategy,
)
from openhcs.processing.backends.cellprofiler.thresholding import (
    CellProfilerAveragingMethod,
    CellProfilerOtsuMethod,
    CellProfilerThresholdAssignment,
    CellProfilerThresholdMethod,
    CellProfilerThresholdRequest,
    CellProfilerThresholdResult,
    CellProfilerThresholdScope,
    CellProfilerThresholdSettings,
    CellProfilerVarianceMethod,
    OutputObjectThresholdMeasurementRecordRowsMixin,
    ThresholdMeasurementFeatureRecord,
    ThresholdPrimitiveBackendStrategy,
    ThresholdSettingsModule,
    normalize_cellprofiler_image,
    threshold_primitives,
    unit_interval_scale_for_threshold_selection,
)
from openhcs.processing.backends.cellprofiler.watershed import (
    cellprofiler_legacy_watershed,
)

logger = logging.getLogger(__name__)
runtime_profiler = CellProfilerRuntimeProfiler(logger)
ClassNamespaceValue: TypeAlias = (
    str
    | bool
    | type
    | staticmethod
    | classmethod
    | property
    | Callable[..., float | str | None]
    | Enum
    | None
)
ObjectLabelSource: TypeAlias = ObjectLabelValue


class SecondaryMethod(CellProfilerEnumAttributeMixin, Enum):
    """CellProfiler IdentifySecondaryObjects segmentation method."""

    __cellprofiler_attribute_names__ = ("requires_threshold",)
    PROPAGATION = ("propagation", True)
    WATERSHED_GRADIENT = ("watershed_gradient", True)
    WATERSHED_IMAGE = ("watershed_image", True)
    DISTANCE_N = ("distance_n", False)
    DISTANCE_B = ("distance_b", True)


@dataclass(frozen=True, slots=True)
class LabelPropagationResult:
    """Regularized propagation labels and cumulative distances."""

    labels: np.ndarray
    distances: np.ndarray


@dataclass(frozen=True, slots=True)
class SecondaryInputNormalization:
    """Normalize secondary-object image and primary-label inputs."""

    image: np.ndarray
    primary_labels: ObjectLabelValue

    def object_label_request(self) -> SourceImageObjectLabelBuildRequest:
        if not isinstance(self.primary_labels, ObjectLabelValue):
            raise TypeError(
                "IdentifySecondaryObjects requires a runtime-projected ObjectLabelValue."
            )
        image = np.asarray(self.image)
        labels = object_label_dense_array(self.primary_labels, dtype=np.int32)
        unedited_labels = np.asarray(
            self.primary_labels.variant_data.labels_for_variant("unedited"),
            dtype=np.int32,
        )
        if image.ndim != 2 or labels.ndim != 2 or unedited_labels.ndim != 2:
            raise ValueError(
                "IdentifySecondaryObjects requires runtime-projected 2-D image and label planes."
            )
        return SourceImageObjectLabelBuildRequest(
            image=image,
            labels=labels,
            unedited_labels=_secondary_seed_labels(labels, unedited_labels),
        )


@dataclass(frozen=True)
class SecondarySegmentationRequest:
    """Inputs needed by nominal secondary-object segmentation strategies."""

    inputs: SourceImageObjectLabelBuildRequest
    thresholded: np.ndarray
    distance_to_dilate: int
    regularization_factor: float
    watershed_backend_provider: CellProfilerBackendProvider | None
    distance_backend_provider: BackendProviderInput = (
        DEFAULT_CELLPROFILER_BACKEND_SELECTION
    )
    propagation_backend_provider: BackendProviderInput = (
        DEFAULT_CELLPROFILER_BACKEND_SELECTION
    )

    @property
    def has_primary_objects(self) -> bool:
        return self.seed_labels.max() > 0

    @property
    def object_mask(self) -> np.ndarray:
        return self.thresholded | (self.seed_labels > 0)

    @property
    def seed_labels(self) -> np.ndarray:
        if self.inputs.unedited_labels is None:
            raise ValueError("Secondary segmentation requires unedited primary labels.")
        return np.asarray(self.inputs.unedited_labels, dtype=np.int32)

    @property
    def label_plane(self) -> np.ndarray:
        return np.asarray(self.inputs.labels, dtype=np.int32)

    @property
    def image_plane(self) -> np.ndarray:
        return np.asarray(self.inputs.image)


class SecondarySegmentationStrategy(
    EnumKeyedStrategyMixin[SecondaryMethod],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Segmentation strategy for one closed secondary-object method."""

    __enum_member_attr__ = "method"
    method: ClassVar[SecondaryMethod | None] = None
    default_propagation_backend_provider: ClassVar[BackendProviderInput] = (
        DEFAULT_CELLPROFILER_BACKEND_SELECTION
    )

    def segment(self, request: SecondarySegmentationRequest) -> np.ndarray:
        if not request.has_primary_objects:
            return np.zeros_like(request.label_plane)
        return self._segment_non_empty(request)

    def propagate_labels(
        self,
        request: SecondarySegmentationRequest,
        *,
        regularization: float,
        max_distance: float | None = None,
    ) -> np.ndarray:
        """Propagate secondary labels through the configured backend."""
        geometry = CellProfilerPlaneGeometry.from_image_plane(request.image_plane)
        labels = geometry.label_plane(request.seed_labels)
        mask = geometry.binary_mask(request.thresholded)
        return SecondaryPropagationBackendStrategy.for_memory_type(
            backend_provider=self.propagation_backend_provider(request)
        ).propagate(
            request.image_plane, labels, mask, regularization, max_distance=max_distance
        )

    def propagation_backend_provider(
        self, request: SecondarySegmentationRequest
    ) -> BackendProviderInput:
        """Return this method's backend provider when the caller requests default."""
        if (
            request.propagation_backend_provider is None
            or request.propagation_backend_provider
            is DEFAULT_CELLPROFILER_BACKEND_SELECTION
        ):
            return self.default_propagation_backend_provider
        return request.propagation_backend_provider

    def watershed_secondary_labels(
        self, request: SecondarySegmentationRequest, watershed_image: np.ndarray
    ) -> np.ndarray:
        """Build secondary labels from watershed markers and object mask."""
        return cellprofiler_legacy_watershed(
            watershed_image,
            markers=request.seed_labels,
            mask=request.object_mask,
            connectivity=np.ones((3, 3), bool),
            backend_provider=request.watershed_backend_provider,
        )

    @abstractmethod
    def _segment_non_empty(self, request: SecondarySegmentationRequest) -> np.ndarray:
        """Segment secondary objects when primary labels are present."""


class DistanceOnlySegmentationStrategy(SecondarySegmentationStrategy):
    method = SecondaryMethod.DISTANCE_N

    def _segment_non_empty(self, request: SecondarySegmentationRequest) -> np.ndarray:
        return SecondaryDistanceTransformBackendStrategy.for_memory_type(
            backend_provider=request.distance_backend_provider
        ).nearest_label_expansion(
            request.seed_labels, float(request.distance_to_dilate)
        )


class DistanceMaskedSegmentationStrategy(SecondarySegmentationStrategy):
    method = SecondaryMethod.DISTANCE_B
    default_propagation_backend_provider = CellProfilerBackendProvider.NUMBA

    def _segment_non_empty(self, request: SecondarySegmentationRequest) -> np.ndarray:
        labels_out = self.propagate_labels(
            request, regularization=1.0, max_distance=float(request.distance_to_dilate)
        )
        labels = request.label_plane
        labels_out[labels > 0] = labels[labels > 0]
        accepted_labels = np.unique(labels[labels > 0])
        if accepted_labels.size:
            labels_out[~np.isin(labels_out, accepted_labels)] = 0
        return labels_out


class PropagationSegmentationStrategy(SecondarySegmentationStrategy):
    method = SecondaryMethod.PROPAGATION
    default_propagation_backend_provider = CellProfilerBackendProvider.NUMBA

    def _segment_non_empty(self, request: SecondarySegmentationRequest) -> np.ndarray:
        return self.propagate_labels(
            request, regularization=request.regularization_factor
        )


class GradientWatershedSegmentationStrategy(SecondarySegmentationStrategy):
    method = SecondaryMethod.WATERSHED_GRADIENT

    def _segment_non_empty(self, request: SecondarySegmentationRequest) -> np.ndarray:
        from scipy.ndimage import sobel

        sobel_image = np.abs(sobel(request.image_plane, axis=0)) + np.abs(
            sobel(request.image_plane, axis=1)
        )
        return self.watershed_secondary_labels(request, sobel_image)


class ImageWatershedSegmentationStrategy(SecondarySegmentationStrategy):
    method = SecondaryMethod.WATERSHED_IMAGE

    def _segment_non_empty(self, request: SecondarySegmentationRequest) -> np.ndarray:
        return self.watershed_secondary_labels(request, 1.0 - request.image_plane)


class SecondaryDistanceTransformBackendStrategy(
    CellProfilerBackendStrategyMixin, ABC, metaclass=AutoRegisterMeta
):
    """Distance transform operations used by secondary segmentation."""

    __registry_key__ = "backend_key"
    __skip_if_no_key__ = True

    @abstractmethod
    def distance_to_foreground(self, labels: np.ndarray) -> np.ndarray:
        """Return Euclidean distance to the nearest positive label."""

    @abstractmethod
    def nearest_label_expansion(
        self, labels: np.ndarray, max_distance: float
    ) -> np.ndarray:
        """Expand labels to pixels within ``max_distance`` of a seed."""


class NumpySecondaryDistanceTransformBackendStrategy(
    SecondaryDistanceTransformBackendStrategy
):
    """Reference NumPy/SciPy secondary distance-transform backend."""

    backend_key = CellProfilerBackendAuthority.backend_key(MemoryType.NUMPY)
    memory_type = MemoryType.NUMPY
    is_default_backend = True

    def distance_to_foreground(self, labels: np.ndarray) -> np.ndarray:
        from scipy.ndimage import distance_transform_edt

        return distance_transform_edt(np.asarray(labels) == 0)

    def nearest_label_expansion(
        self, labels: np.ndarray, max_distance: float
    ) -> np.ndarray:
        from scipy.ndimage import distance_transform_edt

        label_array = np.asarray(labels, dtype=np.int32)
        distances, indices = distance_transform_edt(
            label_array == 0, return_indices=True
        )
        output = np.zeros_like(label_array)
        mask = distances <= float(max_distance)
        output[mask] = label_array[indices[0][mask], indices[1][mask]]
        return output


class NumbaSecondaryDistanceTransformBackendStrategy(
    SecondaryDistanceTransformBackendStrategy
):
    """Numba-accelerated exact 2-D Euclidean distance-transform backend."""

    backend_key = CellProfilerBackendAuthority.backend_key(
        MemoryType.NUMPY, CellProfilerBackendProvider.NUMBA
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NUMBA
    is_default_backend = False

    def prepare_backend(self) -> None:
        labels = np.array([[0, 1, 0], [0, 0, 2], [0, 0, 0]], dtype=np.int32)
        self.distance_to_foreground(labels)
        self.nearest_label_expansion(labels, 2.0)

    def distance_to_foreground(self, labels: np.ndarray) -> np.ndarray:
        label_array = np.asarray(labels, dtype=np.int32)
        if label_array.ndim != 2:
            raise NotImplementedError(
                "Numba secondary distance backend currently supports 2-D labels."
            )
        return _distance_to_positive_labels_numba(np.ascontiguousarray(label_array))

    def nearest_label_expansion(
        self, labels: np.ndarray, max_distance: float
    ) -> np.ndarray:
        label_array = np.asarray(labels, dtype=np.int32)
        if label_array.ndim != 2:
            raise NotImplementedError(
                "Numba secondary distance backend currently supports 2-D labels."
            )
        return _nearest_label_expansion_numba(
            np.ascontiguousarray(label_array), float(max_distance)
        )


class SecondaryPropagationBackendStrategy(
    CellProfilerBackendStrategyMixin, ABC, metaclass=AutoRegisterMeta
):
    """Regularized label propagation backend for secondary segmentation."""

    __registry_key__ = "backend_key"
    __skip_if_no_key__ = True

    @abstractmethod
    def propagate_result(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        mask: np.ndarray,
        regularization: float,
        *,
        max_distance: float | None = None,
    ) -> LabelPropagationResult:
        """Propagate seed labels through a mask and retain cumulative distances."""

    def propagate(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        mask: np.ndarray,
        regularization: float,
        *,
        max_distance: float | None = None,
    ) -> np.ndarray:
        """Propagate seed labels through a mask."""
        result = self.propagate_result(
            image, labels, mask, regularization, max_distance=max_distance
        )
        propagated = np.asarray(result.labels, dtype=np.int32)
        if max_distance is None:
            return propagated
        filtered = propagated.copy()
        source_labels = np.asarray(labels, dtype=np.int32)
        filtered[
            np.asarray(result.distances, dtype=np.float64) > float(max_distance)
        ] = 0
        filtered[source_labels > 0] = source_labels[source_labels > 0]
        return filtered

    def propagate_zero_image_result(
        self,
        labels: np.ndarray,
        mask: np.ndarray,
        regularization: float,
        *,
        max_distance: float | None = None,
    ) -> LabelPropagationResult:
        """Propagate seed labels when the propagation image has uniform intensity."""
        label_array = np.asarray(labels, dtype=np.int32)
        return self.propagate_result(
            np.zeros(label_array.shape, dtype=np.float64),
            label_array,
            mask,
            regularization,
            max_distance=max_distance,
        )


class NumbaSecondaryPropagationBackendStrategy(SecondaryPropagationBackendStrategy):
    """Numba implementation of regularized secondary-label propagation."""

    backend_key = CellProfilerBackendAuthority.backend_key(
        MemoryType.NUMPY, CellProfilerBackendProvider.NUMBA
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NUMBA
    is_default_backend = False

    def prepare_backend(self) -> None:
        image = np.arange(9, dtype=np.float64).reshape((3, 3))
        labels = np.array([[1, 0, 0], [0, 0, 2], [0, 0, 0]], dtype=np.int32)
        mask = np.ones(labels.shape, dtype=np.bool_)
        self.propagate_result(image, labels, mask, 0.1)
        self.propagate_result(image, labels, mask, 1.0, max_distance=2.0)
        self.propagate_zero_image_result(labels, mask, 1.0)

    def propagate_result(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        mask: np.ndarray,
        regularization: float,
        *,
        max_distance: float | None = None,
    ) -> LabelPropagationResult:
        image_array = np.asarray(image, dtype=np.float64)
        label_array = np.asarray(labels, dtype=np.int32)
        mask_array = np.asarray(mask, dtype=np.bool_)
        if image_array.ndim != 2 or label_array.ndim != 2 or mask_array.ndim != 2:
            raise NotImplementedError(
                "Numba secondary propagation backend currently supports 2-D arrays."
            )
        if (
            image_array.shape != label_array.shape
            or image_array.shape != mask_array.shape
        ):
            raise ValueError("image, labels, and mask must have the same shape.")
        if np.max(label_array) == 0:
            return LabelPropagationResult(
                labels=label_array.copy(),
                distances=np.zeros(label_array.shape, dtype=np.float64),
            )
        propagated, distances = _propagate_labels_and_distances_numba(
            np.ascontiguousarray(image_array),
            np.ascontiguousarray(label_array),
            np.ascontiguousarray(mask_array),
            float(regularization),
            -1.0 if max_distance is None else float(max_distance),
        )
        return LabelPropagationResult(labels=propagated, distances=distances)

    def propagate_zero_image_result(
        self,
        labels: np.ndarray,
        mask: np.ndarray,
        regularization: float,
        *,
        max_distance: float | None = None,
    ) -> LabelPropagationResult:
        label_array = np.asarray(labels, dtype=np.int32)
        mask_array = np.asarray(mask, dtype=np.bool_)
        if label_array.ndim != 2 or mask_array.ndim != 2:
            raise NotImplementedError(
                "Numba secondary propagation backend currently supports 2-D arrays."
            )
        if label_array.shape != mask_array.shape:
            raise ValueError("labels and mask must have the same shape.")
        if np.max(label_array) == 0:
            return LabelPropagationResult(
                labels=label_array.copy(),
                distances=np.zeros(label_array.shape, dtype=np.float64),
            )
        propagated, distances = _propagate_labels_and_distances_zero_image_numba(
            np.ascontiguousarray(label_array),
            np.ascontiguousarray(mask_array),
            float(regularization),
            -1.0 if max_distance is None else float(max_distance),
        )
        return LabelPropagationResult(labels=propagated, distances=distances)


class NativeSecondaryPropagationBackendStrategy(SecondaryPropagationBackendStrategy):
    """CellProfiler-native label propagation via centrosome."""

    backend_key = CellProfilerBackendAuthority.backend_key(
        MemoryType.NUMPY, CellProfilerBackendProvider.NATIVE
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NATIVE
    is_default_backend = True

    def propagate_result(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        mask: np.ndarray,
        regularization: float,
        *,
        max_distance: float | None = None,
    ) -> LabelPropagationResult:
        import centrosome.propagate

        propagated, distances = centrosome.propagate.propagate(
            np.asarray(image, dtype=np.float64),
            np.asarray(labels, dtype=np.int32),
            np.asarray(mask, dtype=np.bool_),
            float(regularization),
        )
        if max_distance is not None:
            propagated = np.asarray(propagated, dtype=np.int32).copy()
            source_labels = np.asarray(labels, dtype=np.int32)
            propagated[
                np.asarray(distances, dtype=np.float64) > float(max_distance)
            ] = 0
            propagated[source_labels > 0] = source_labels[source_labels > 0]
        return LabelPropagationResult(
            labels=np.asarray(propagated, dtype=np.int32),
            distances=np.asarray(distances, dtype=np.float64),
        )


def secondary_propagation_backend(
    *, backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION
) -> SecondaryPropagationBackendStrategy:
    """Return the selected secondary propagation backend."""
    return SecondaryPropagationBackendStrategy.for_memory_type(
        MemoryType.NUMPY, backend_provider=backend_provider
    )


class ThresholdMethod(Enum):
    OTSU = "otsu"
    LI = "li"
    MINIMUM = "minimum"
    TRIANGLE = "triangle"


@dataclass
class SecondaryObjectStats(ThresholdMeasurementFeatureRecord):
    slice_index: Annotated[int, MeasurementRowAxisField.SLICE_INDEX]
    method: SecondaryMethod
    object_count: int
    mean_area: float
    median_area: float
    total_area: int
    area_coverage_percent: float
    final_threshold: float
    original_threshold: float = 0.0
    weighted_variance: float = 0.0
    sum_of_entropies: float = 0.0

    @classmethod
    def from_labels(
        cls,
        labels: np.ndarray,
        *,
        image_shape: tuple[int, int],
        threshold: CellProfilerThresholdResult,
        method: SecondaryMethod,
    ) -> "SecondaryObjectStats":
        areas = np.bincount(np.asarray(labels).ravel())[1:]
        positive_areas = areas[areas > 0]
        object_count = int(positive_areas.size)
        if object_count == 0:
            mean_area = 0.0
            median_area = 0.0
            total_area = 0
        else:
            mean_area = float(np.mean(positive_areas))
            median_area = float(np.median(positive_areas))
            total_area = int(np.sum(positive_areas))
        height, width = image_shape
        pixel_count = height * width
        if pixel_count:
            area_coverage = 100.0 * total_area / pixel_count
        else:
            area_coverage = 0.0
        return cls(
            slice_index=0,
            method=method,
            object_count=object_count,
            mean_area=mean_area,
            median_area=median_area,
            total_area=total_area,
            area_coverage_percent=area_coverage,
            final_threshold=float(threshold.final_threshold),
            original_threshold=float(threshold.original_threshold),
            weighted_variance=float(threshold.weighted_variance),
            sum_of_entropies=float(threshold.sum_of_entropies),
        )


@dataclass(frozen=True)
class SecondaryObjectLabels:
    """CellProfiler object-container label variants for secondary objects."""

    segmented: np.ndarray
    unedited_segmented: np.ndarray
    small_removed_segmented: np.ndarray

    @classmethod
    def from_raw_labels(
        cls,
        labels: np.ndarray,
        *,
        fill_holes: bool,
        discard_edge_objects: bool,
        primary_labels: np.ndarray,
        morphology: MorphologyBackendStrategy,
    ) -> "SecondaryObjectLabels":
        small_removed = labels
        if fill_holes and small_removed.max() > 0:
            small_removed = morphology.fill_labeled_holes(
                small_removed,
                mask=small_removed == 0,
            )
        segmented = _filter_labels(small_removed, primary_labels)
        if discard_edge_objects and segmented.max() > 0:
            segmented = _discard_edge_objects(segmented, morphology)
        segmented = segmented.astype(np.int32, copy=False)
        small_removed = small_removed.astype(np.int32, copy=False)
        return cls(
            segmented=segmented,
            unedited_segmented=small_removed,
            small_removed_segmented=small_removed,
        )

    @property
    def object_count(self) -> int:
        if not self.segmented.size:
            return 0
        return int(np.max(self.segmented))


class ThresholdCalculator(
    EnumKeyedStrategyMixin[ThresholdMethod],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Threshold strategy for one closed CellProfiler threshold method."""

    __enum_member_attr__ = "method"
    method: ClassVar[ThresholdMethod | None] = None
    primitive: ClassVar[
        Callable[[ThresholdPrimitiveBackendStrategy, np.ndarray], float]
    ]

    def calculate(self, image: np.ndarray) -> float:
        """Calculate a threshold value for a normalized intensity image."""
        return type(self).primitive(threshold_primitives(), image)


@dataclass(frozen=True)
class ThresholdCalculatorDeclaration(RegisteredLeafClassSpec):
    """Typed declaration for metadata-only secondary threshold calculators."""

    method: ThresholdMethod
    primitive: Callable[[ThresholdPrimitiveBackendStrategy, np.ndarray], float]
    base_type: ClassVar[type[ThresholdCalculator]] = ThresholdCalculator

    @property
    def class_name(self) -> str:
        method_name = "".join((part.title() for part in self.method.name.split("_")))
        return f"{method_name}ThresholdCalculator"

    def class_attributes(self) -> Mapping[str, ClassNamespaceValue]:
        primitive = self.primitive

        def concrete_primitive(
            backend: ThresholdPrimitiveBackendStrategy, image: np.ndarray
        ) -> float:
            return primitive(backend, image)

        return {
            "method": self.method,
            "primitive": staticmethod(concrete_primitive),
        }


THRESHOLD_CALCULATOR_DECLARATIONS = (
    ThresholdCalculatorDeclaration(
        ThresholdMethod.OTSU, ThresholdPrimitiveBackendStrategy.otsu_threshold
    ),
    ThresholdCalculatorDeclaration(
        ThresholdMethod.LI, ThresholdPrimitiveBackendStrategy.li_threshold
    ),
    ThresholdCalculatorDeclaration(
        ThresholdMethod.MINIMUM, ThresholdPrimitiveBackendStrategy.minimum_threshold
    ),
    ThresholdCalculatorDeclaration(
        ThresholdMethod.TRIANGLE, ThresholdPrimitiveBackendStrategy.triangle_threshold
    ),
)
for threshold_calculator_declaration in THRESHOLD_CALCULATOR_DECLARATIONS:
    threshold_calculator_declaration.declare_in(globals())


def _secondary_seed_labels(
    final_labels: np.ndarray, unedited_labels: np.ndarray
) -> np.ndarray:
    """Match CellProfiler's secondary-object seed contract.

    CellProfiler seeds secondary segmentation from unedited primary labels, but
    removes rejected non-edge labels. Edge-touching rejected labels remain so
    they can constrain propagated secondary boundaries without becoming accepted
    parent objects.
    """
    labels_in = np.asarray(unedited_labels, dtype=np.int32).copy()
    if labels_in.size == 0 or labels_in.max() <= 0:
        return labels_in
    final = np.asarray(final_labels, dtype=np.int32)
    if final.shape != labels_in.shape:
        raise ValueError(
            "Secondary seed labels must share one explicitly aligned spatial "
            f"domain; got final={final.shape!r}, unedited={labels_in.shape!r}."
        )
    edge_labels = np.unique(
        np.concatenate(
            (labels_in[0, :], labels_in[-1, :], labels_in[:, 0], labels_in[:, -1])
        )
    )
    is_touching_lookup = np.zeros(int(labels_in.max()) + 1, dtype=bool)
    is_touching_lookup[edge_labels.astype(int)] = True
    return _secondary_seed_label_filter_numba(
        np.ascontiguousarray(labels_in, dtype=np.int32),
        np.ascontiguousarray(final, dtype=np.int32),
        is_touching_lookup,
    )


@njit(cache=True)
def _secondary_seed_label_filter_numba(
    unedited_labels: np.ndarray, final_labels: np.ndarray, is_touching_edge: np.ndarray
) -> np.ndarray:
    output = unedited_labels.copy()
    flat_unedited = unedited_labels.ravel()
    flat_final = final_labels.ravel()
    output_flat = output.ravel()
    for index in range(flat_unedited.size):
        unedited_label = int(flat_unedited[index])
        if (
            unedited_label > 0
            and not is_touching_edge[unedited_label]
            and int(flat_final[index]) == 0
        ):
            output_flat[index] = 0
    return output


def _filter_labels(labels_out: np.ndarray, primary_labels: np.ndarray) -> np.ndarray:
    """Keep secondary labels associated with accepted primary labels."""
    max_out = int(np.max(labels_out))
    if max_out <= 0:
        return labels_out.copy()
    if primary_labels.shape != labels_out.shape:
        raise ValueError(
            "Secondary and primary labels must share one explicitly aligned "
            f"spatial domain; got secondary={labels_out.shape!r}, "
            f"primary={primary_labels.shape!r}."
        )
    return _filter_labels_numba(
        np.ascontiguousarray(labels_out, dtype=np.int32),
        np.ascontiguousarray(primary_labels, dtype=np.int32),
        max_out,
    )


@njit(cache=True)
def _filter_labels_numba(
    labels_out: np.ndarray, aligned_primary: np.ndarray, max_out: int
) -> np.ndarray:
    lookup = np.zeros(max_out + 1, dtype=np.int32)
    labels_flat = labels_out.ravel()
    primary_flat = aligned_primary.ravel()
    for index in range(labels_flat.size):
        label = int(labels_flat[index])
        if label <= 0:
            continue
        primary_label = int(primary_flat[index])
        if primary_label > lookup[label]:
            lookup[label] = primary_label
    lookup[0] = 0
    filtered = np.empty(labels_out.shape, dtype=np.int32)
    filtered_flat = filtered.ravel()
    for index in range(labels_flat.size):
        filtered_flat[index] = lookup[int(labels_flat[index])]
    return filtered


def _discard_edge_objects(
    labels: np.ndarray, morphology: MorphologyBackendStrategy
) -> np.ndarray:
    edge_labels = np.unique(
        np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])
    )
    labels_out = labels.copy()
    for edge_label in edge_labels:
        if edge_label > 0:
            labels_out[labels_out == edge_label] = 0
    if labels_out.max() == 0:
        return labels_out
    relabeled, _count = morphology.connected_components(labels_out > 0, connectivity=2)
    return relabeled.astype(np.int32, copy=False)


IdentifySecondaryObjectsOutput: TypeAlias = Tuple[
    np.ndarray,
    ColumnarRows,
    DirectedObjectRelationshipPayload,
    ObjectLabelValue,
]
IdentifySecondaryObjectsReplacementPrimaryOutput: TypeAlias = Tuple[
    np.ndarray,
    ColumnarRows,
    DirectedObjectRelationshipPayload,
    DirectedObjectRelationshipPayload,
    DirectedObjectRelationshipPayload,
    ObjectLabelValue,
    ObjectLabelValue,
]


class IdentifySecondaryObjectsReplacementPrimarySourceRelation(
    SourceStackLineageSourceRelation
):
    """Mark the replacement-primary output and its exact source object set."""

    relation_key: ClassVar[str] = (
        "identify_secondary_objects_replacement_primary_source"
    )
    target_artifact_type = ObjectLabelsArtifactType

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.source.artifact_type is not ObjectLabelsArtifactType:
            raise ValueError(
                "IdentifySecondaryObjects replacement-primary output requires an "
                f"object-label source, got {self.source.artifact_type.value}:"
                f"{self.source.name}."
            )


@dataclass(frozen=True, slots=True)
class IdentifySecondaryObjectsExecution:
    """Typed result shared by the two exact public output ABIs."""

    image: np.ndarray
    measurement_rows: ColumnarRows
    primary_secondary_relationship: DirectedObjectRelationshipPayload
    secondary_output: ObjectLabelValue


def _replacement_primary_output_from_relationship(
    image: np.ndarray,
    primary_labels: ObjectLabelValue,
    primary_secondary_relationship: DirectedObjectRelationshipPayload,
) -> ObjectLabelValue:
    """Return retained primary labels in the secondary object's ID domain."""

    primary_array = object_label_dense_array(primary_labels, dtype=np.int32)
    max_primary_id = int(np.max(primary_array)) if primary_array.size else 0
    secondary_id_by_primary_id = np.zeros(max_primary_id + 1, dtype=np.int32)
    for primary_id, secondary_id in zip(
        primary_secondary_relationship.source_ids,
        primary_secondary_relationship.target_ids,
        strict=True,
    ):
        if primary_id <= 0 or primary_id > max_primary_id or secondary_id <= 0:
            raise ValueError(
                "IdentifySecondaryObjects replacement-primary relationship contains "
                f"an invalid ID pair {(primary_id, secondary_id)!r}."
            )
        previous_secondary_id = int(secondary_id_by_primary_id[primary_id])
        if previous_secondary_id not in (0, secondary_id):
            raise ValueError(
                "IdentifySecondaryObjects replacement-primary relationship maps "
                f"primary object {primary_id} to multiple secondary objects."
            )
        secondary_id_by_primary_id[primary_id] = secondary_id

    replacement_labels = secondary_id_by_primary_id[primary_array]
    variants = primary_labels.variant_data
    declared_ids = tuple(dict.fromkeys(primary_secondary_relationship.target_ids))
    return SourceImageObjectLabelBuildRequest(
        image=image,
        labels=replacement_labels,
        unedited_labels=variants.unedited_labels,
        small_removed_labels=variants.small_removed_labels,
        declared_object_ids=declared_ids,
        parent_image_source_voxel_spacing=(
            primary_labels.parent_image_source_voxel_spacing
        ),
    ).payload()


def _execute_identify_secondary_objects(
    image: np.ndarray,
    primary_labels: ObjectLabelValue,
    method: SecondaryMethod = SecondaryMethod.PROPAGATION,
    threshold_scope: CellProfilerThresholdScope = CellProfilerThresholdScope.GLOBAL,
    threshold_method: CellProfilerThresholdMethod = CellProfilerThresholdMethod.OTSU,
    threshold_smoothing_scale: float = 0.0,
    threshold_correction_factor: float = 1.0,
    threshold_min: float = 0.0,
    threshold_max: float = 1.0,
    manual_threshold: float = 0.0,
    otsu_class_count: CellProfilerOtsuMethod = CellProfilerOtsuMethod.TWO_CLASS,
    assign_middle_to_foreground: CellProfilerThresholdAssignment = CellProfilerThresholdAssignment.FOREGROUND,
    log_transform: bool = False,
    adaptive_window_size: int = 10,
    lower_outlier_fraction: float = 0.05,
    upper_outlier_fraction: float = 0.05,
    averaging_method: CellProfilerAveragingMethod = CellProfilerAveragingMethod.MEAN,
    variance_method: CellProfilerVarianceMethod = CellProfilerVarianceMethod.STANDARD_DEVIATION,
    number_of_deviations: float = 2.0,
    distance_to_dilate: int = 10,
    regularization_factor: float = 0.05,
    fill_holes: bool = True,
    discard_edge_objects: bool = False,
    watershed_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    morphology_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    distance_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    propagation_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
) -> IdentifySecondaryObjectsExecution:
    """
    Identify secondary objects using primary objects as seeds.

    Args:
        primary_labels: Primary objects used as seeds for secondary segmentation.
        method: Segmentation method used to expand the primary objects.
        threshold_method: Method used to calculate the foreground threshold.
        threshold_correction_factor: Multiplier applied to the calculated threshold.
        threshold_min: Lowest threshold value permitted after calculation.
        threshold_max: Highest threshold value permitted after calculation.
        distance_to_dilate: Expansion distance in pixels for distance-based segmentation.
        regularization_factor: Propagation balance between image gradients and distance.
        fill_holes: Whether holes inside segmented secondary objects are filled.
        discard_edge_objects: Whether objects touching the image border are discarded.
    Returns:
        The shared typed segmentation, measurement, relationship, and label result.
    """
    profile_total_started_at = time.perf_counter()
    phase_started_at = time.perf_counter()
    morphology = MorphologyBackendStrategy.for_callable(
        identify_secondary_objects, backend_provider=morphology_backend_provider
    )
    input_mask = image_payload_mask(image)
    if input_mask is not None:
        input_mask = np.asarray(input_mask, dtype=bool)
        if input_mask.ndim != 2:
            raise ValueError(
                "IdentifySecondaryObjects requires a runtime-projected 2-D image mask."
            )
    raw_image_data = image_payload_data(image)
    proven_unit_interval_scale = unit_interval_scale_for_threshold_selection(
        np.asarray(raw_image_data), image_payload_metadata(image)
    )
    inputs = SecondaryInputNormalization(
        raw_image_data, primary_labels
    ).object_label_request()
    img = normalize_cellprofiler_image(np.asarray(inputs.image))
    runtime_profiler.log(
        "iso_prepare_inputs",
        time.perf_counter() - phase_started_at,
        function="identify_secondary_objects",
        method=method.value,
    )
    phase_started_at = time.perf_counter()
    threshold = CellProfilerThresholdRequest(
        image=img,
        image_mask=input_mask,
        settings=CellProfilerThresholdSettings(
            use_advanced_settings=True,
            threshold_scope=threshold_scope,
            threshold_method=threshold_method,
            threshold_smoothing_scale=threshold_smoothing_scale,
            threshold_correction_factor=threshold_correction_factor,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            manual_threshold=manual_threshold,
            otsu_class_count=otsu_class_count,
            assign_middle_to_foreground=assign_middle_to_foreground,
            log_transform=log_transform,
            adaptive_window_size=adaptive_window_size,
            lower_outlier_fraction=lower_outlier_fraction,
            upper_outlier_fraction=upper_outlier_fraction,
            averaging_method=averaging_method,
            variance_method=variance_method,
            number_of_deviations=number_of_deviations,
        ),
        proven_unit_interval_scale=proven_unit_interval_scale,
        enabled=method.requires_threshold,
    ).calculate()
    runtime_profiler.log(
        "iso_threshold",
        time.perf_counter() - phase_started_at,
        function="identify_secondary_objects",
        method=method.value,
    )
    phase_started_at = time.perf_counter()
    raw_labels = SecondarySegmentationStrategy.for_enum_member(method).segment(
        SecondarySegmentationRequest(
            inputs=SourceImageObjectLabelBuildRequest(
                image=img, labels=inputs.labels, unedited_labels=inputs.unedited_labels
            ),
            thresholded=threshold.mask,
            distance_to_dilate=distance_to_dilate,
            regularization_factor=regularization_factor,
            watershed_backend_provider=watershed_backend_provider,
            distance_backend_provider=distance_backend_provider,
            propagation_backend_provider=propagation_backend_provider,
        )
    )
    runtime_profiler.log(
        "iso_segment",
        time.perf_counter() - phase_started_at,
        function="identify_secondary_objects",
        method=method.value,
    )
    phase_started_at = time.perf_counter()
    object_labels = SecondaryObjectLabels.from_raw_labels(
        raw_labels,
        fill_holes=fill_holes,
        discard_edge_objects=discard_edge_objects,
        primary_labels=inputs.labels,
        morphology=morphology,
    )
    runtime_profiler.log(
        "iso_label_variants",
        time.perf_counter() - phase_started_at,
        function="identify_secondary_objects",
        method=method.value,
    )
    phase_started_at = time.perf_counter()
    stats = SecondaryObjectStats.from_labels(
        object_labels.segmented,
        image_shape=img.shape,
        threshold=threshold,
        method=method,
    )
    runtime_profiler.log(
        "iso_stats",
        time.perf_counter() - phase_started_at,
        function="identify_secondary_objects",
        method=method.value,
    )
    phase_started_at = time.perf_counter()
    secondary_output = SourceImageObjectLabelBuildRequest(
        image=image,
        labels=object_labels.segmented,
        unedited_labels=object_labels.unedited_segmented,
        small_removed_labels=object_labels.small_removed_segmented,
        declared_object_count=object_labels.object_count,
    ).payload()
    primary_secondary_relationship = object_label_parent_child_payload(
        primary_labels,
        secondary_output,
    )
    runtime_profiler.log(
        "iso_relationships",
        time.perf_counter() - phase_started_at,
        function="identify_secondary_objects",
        method=method.value,
    )
    runtime_profiler.log(
        "iso_total",
        time.perf_counter() - profile_total_started_at,
        function="identify_secondary_objects",
        method=method.value,
    )
    measurement_rows = (
        threshold.measurement_rows()
        if method.requires_threshold
        else MeasurementSparseColumnarRows.from_rows((), fields=())
    )
    return IdentifySecondaryObjectsExecution(
        image=img.astype(np.float32),
        measurement_rows=measurement_rows,
        primary_secondary_relationship=primary_secondary_relationship,
        secondary_output=secondary_output,
    )


@numpy(contract=ProcessingContract.PURE_2D)
@special_inputs("primary_labels")
def identify_secondary_objects(
    image: np.ndarray,
    primary_labels: ObjectLabelValue,
    method: SecondaryMethod = SecondaryMethod.PROPAGATION,
    threshold_scope: CellProfilerThresholdScope = CellProfilerThresholdScope.GLOBAL,
    threshold_method: CellProfilerThresholdMethod = CellProfilerThresholdMethod.OTSU,
    threshold_smoothing_scale: float = 0.0,
    threshold_correction_factor: float = 1.0,
    threshold_min: float = 0.0,
    threshold_max: float = 1.0,
    manual_threshold: float = 0.0,
    otsu_class_count: CellProfilerOtsuMethod = CellProfilerOtsuMethod.TWO_CLASS,
    assign_middle_to_foreground: CellProfilerThresholdAssignment = CellProfilerThresholdAssignment.FOREGROUND,
    log_transform: bool = False,
    adaptive_window_size: int = 10,
    lower_outlier_fraction: float = 0.05,
    upper_outlier_fraction: float = 0.05,
    averaging_method: CellProfilerAveragingMethod = CellProfilerAveragingMethod.MEAN,
    variance_method: CellProfilerVarianceMethod = CellProfilerVarianceMethod.STANDARD_DEVIATION,
    number_of_deviations: float = 2.0,
    distance_to_dilate: int = 10,
    regularization_factor: float = 0.05,
    fill_holes: bool = True,
    discard_edge_objects: bool = False,
    watershed_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    morphology_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    distance_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    propagation_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
) -> IdentifySecondaryObjectsOutput:
    """Identify secondary objects without a replacement-primary output."""

    execution = _execute_identify_secondary_objects(
        image,
        primary_labels,
        method=method,
        threshold_scope=threshold_scope,
        threshold_method=threshold_method,
        threshold_smoothing_scale=threshold_smoothing_scale,
        threshold_correction_factor=threshold_correction_factor,
        threshold_min=threshold_min,
        threshold_max=threshold_max,
        manual_threshold=manual_threshold,
        otsu_class_count=otsu_class_count,
        assign_middle_to_foreground=assign_middle_to_foreground,
        log_transform=log_transform,
        adaptive_window_size=adaptive_window_size,
        lower_outlier_fraction=lower_outlier_fraction,
        upper_outlier_fraction=upper_outlier_fraction,
        averaging_method=averaging_method,
        variance_method=variance_method,
        number_of_deviations=number_of_deviations,
        distance_to_dilate=distance_to_dilate,
        regularization_factor=regularization_factor,
        fill_holes=fill_holes,
        discard_edge_objects=discard_edge_objects,
        watershed_backend_provider=watershed_backend_provider,
        morphology_backend_provider=morphology_backend_provider,
        distance_backend_provider=distance_backend_provider,
        propagation_backend_provider=propagation_backend_provider,
    )
    return (
        execution.image,
        execution.measurement_rows,
        execution.primary_secondary_relationship,
        execution.secondary_output,
    )


@numpy(contract=ProcessingContract.PURE_2D)
@special_inputs("primary_labels")
def identify_secondary_objects_with_replacement_primary(
    image: np.ndarray,
    primary_labels: ObjectLabelValue,
    method: SecondaryMethod = SecondaryMethod.PROPAGATION,
    threshold_scope: CellProfilerThresholdScope = CellProfilerThresholdScope.GLOBAL,
    threshold_method: CellProfilerThresholdMethod = CellProfilerThresholdMethod.OTSU,
    threshold_smoothing_scale: float = 0.0,
    threshold_correction_factor: float = 1.0,
    threshold_min: float = 0.0,
    threshold_max: float = 1.0,
    manual_threshold: float = 0.0,
    otsu_class_count: CellProfilerOtsuMethod = CellProfilerOtsuMethod.TWO_CLASS,
    assign_middle_to_foreground: CellProfilerThresholdAssignment = CellProfilerThresholdAssignment.FOREGROUND,
    log_transform: bool = False,
    adaptive_window_size: int = 10,
    lower_outlier_fraction: float = 0.05,
    upper_outlier_fraction: float = 0.05,
    averaging_method: CellProfilerAveragingMethod = CellProfilerAveragingMethod.MEAN,
    variance_method: CellProfilerVarianceMethod = CellProfilerVarianceMethod.STANDARD_DEVIATION,
    number_of_deviations: float = 2.0,
    distance_to_dilate: int = 10,
    regularization_factor: float = 0.05,
    fill_holes: bool = True,
    discard_edge_objects: bool = True,
    watershed_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    morphology_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    distance_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    propagation_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
) -> IdentifySecondaryObjectsReplacementPrimaryOutput:
    """Execute the contract variant that emits replacement-primary artifacts."""

    execution = _execute_identify_secondary_objects(
        image,
        primary_labels,
        method=method,
        threshold_scope=threshold_scope,
        threshold_method=threshold_method,
        threshold_smoothing_scale=threshold_smoothing_scale,
        threshold_correction_factor=threshold_correction_factor,
        threshold_min=threshold_min,
        threshold_max=threshold_max,
        manual_threshold=manual_threshold,
        otsu_class_count=otsu_class_count,
        assign_middle_to_foreground=assign_middle_to_foreground,
        log_transform=log_transform,
        adaptive_window_size=adaptive_window_size,
        lower_outlier_fraction=lower_outlier_fraction,
        upper_outlier_fraction=upper_outlier_fraction,
        averaging_method=averaging_method,
        variance_method=variance_method,
        number_of_deviations=number_of_deviations,
        distance_to_dilate=distance_to_dilate,
        regularization_factor=regularization_factor,
        fill_holes=fill_holes,
        discard_edge_objects=discard_edge_objects,
        watershed_backend_provider=watershed_backend_provider,
        morphology_backend_provider=morphology_backend_provider,
        distance_backend_provider=distance_backend_provider,
        propagation_backend_provider=propagation_backend_provider,
    )
    replacement_primary_output = _replacement_primary_output_from_relationship(
        image,
        primary_labels,
        execution.primary_secondary_relationship,
    )
    primary_replacement_relationship = object_label_parent_child_payload(
        primary_labels,
        replacement_primary_output,
    )
    replacement_secondary_relationship = object_label_parent_child_payload(
        replacement_primary_output,
        execution.secondary_output,
    )
    return (
        execution.image,
        execution.measurement_rows,
        execution.primary_secondary_relationship,
        primary_replacement_relationship,
        replacement_secondary_relationship,
        execution.secondary_output,
        replacement_primary_output,
    )


set_signature_analysis_target(
    identify_secondary_objects,
    _execute_identify_secondary_objects,
)
set_signature_analysis_target(
    identify_secondary_objects_with_replacement_primary,
    _execute_identify_secondary_objects,
)


class IdentifySecondaryObjectsModule(
    PrimaryObjectLabelInputPolicy,
    OutputObjectThresholdMeasurementRecordRowsMixin,
    NoObjectNameMeasurementRecordMixin,
    ObjectArtifactInputModule,
    ObjectArtifactOutputModule,
    ParentChildLineageArtifactOutputModule,
    MeasurementArtifactOutputModule,
    ThresholdSettingsModule,
):
    module_name = "IdentifySecondaryObjects"
    function_name = "identify_secondary_objects"
    function_variants = ("identify_secondary_objects_with_replacement_primary",)
    validated = True
    confidence = 1.0

    input_image_setting = "Select the input image"
    input_objects_setting = "Select the input objects"
    output_objects_setting = "Name the objects to be identified"
    method_setting = "Select the method to identify the secondary objects"
    distance_to_dilate_setting = (
        "Number of pixels by which to expand the primary objects"
    )
    regularization_factor_setting = "Regularization factor"
    fill_holes_setting = "Fill holes in identified objects?"
    discard_edge_objects_setting = (
        "Discard secondary objects touching the border of the image?"
    )
    discard_associated_primary_objects_setting = (
        "Discard the associated primary objects?"
    )
    replacement_primary_objects_setting = "Name the new primary objects"
    secondary_output_binding = SettingToKeywordBinding.output(
        output_objects_setting, ObjectLabelsArtifactType
    )
    replacement_primary_output_binding = SettingToKeywordBinding.output(
        replacement_primary_objects_setting, ObjectLabelsArtifactType
    )
    setting_bindings = (
        SettingToKeywordBinding.input(input_image_setting, ImageArtifactType),
        SettingToKeywordBinding.input(
            input_objects_setting,
            ObjectLabelsArtifactType,
            runtime_parameter_name="primary_labels",
        ),
        secondary_output_binding,
        replacement_primary_output_binding,
        SettingToKeywordBinding(method_setting, "method"),
        SettingToKeywordBinding(
            distance_to_dilate_setting,
            "distance_to_dilate",
        ),
        SettingToKeywordBinding(
            regularization_factor_setting,
            "regularization_factor",
        ),
        SettingToKeywordBinding(fill_holes_setting, "fill_holes"),
        SettingToKeywordBinding(
            discard_edge_objects_setting,
            "discard_edge_objects",
        ),
    )
    ignored_settings = (discard_associated_primary_objects_setting,)

    @classmethod
    def replacement_primary_output_active(cls, module: "ModuleBlock") -> bool:
        """Return the one declaration-owned replacement-primary condition."""

        discard_edge_value = cls.setting_value(
            module,
            cls.discard_edge_objects_setting,
        )
        discard_primary_value = cls.setting_value(
            module,
            cls.discard_associated_primary_objects_setting,
        )
        return (
            discard_edge_value is not None
            and discard_primary_value is not None
            and parse_cellprofiler_bool(discard_edge_value)
            and parse_cellprofiler_bool(discard_primary_value)
        )

    @classmethod
    def active_artifact_bindings(cls, module=None, *, invocation_key=None):
        """Expose replacement primaries only for the active module topology."""

        bindings = super().active_artifact_bindings(
            module,
            invocation_key=invocation_key,
        )
        if module is None or cls.replacement_primary_output_active(module):
            return bindings
        return tuple(
            binding
            for binding in bindings
            if binding is not cls.replacement_primary_output_binding
        )

    @classmethod
    def resolve_function(
        cls,
        module: "ModuleBlock",
        *,
        contract,
        source_bindings,
    ) -> Callable[..., object]:
        """Select the callable from the module-owned replacement-primary setting."""

        del contract, source_bindings
        if cls.replacement_primary_output_active(module):
            return cls.require_callable(cls.function_variants[0])
        return cls.require_callable()

    @classmethod
    def _derived_identity_setting_records(
        cls,
        *,
        invocation,
        block_position,
        existing_records,
        step_context,
    ):
        """Reconstruct replacement-primary topology from the explicit callable."""

        from openhcs.interop.cellprofiler.parser import ModuleSetting

        setting_key = cls.normalize_setting_name(
            cls.discard_associated_primary_objects_setting
        )
        own_records = (
            (
                ModuleSetting(
                    cls.discard_associated_primary_objects_setting,
                    "Yes",
                ),
            )
            if invocation.contract.function_name == cls.function_variants[0]
            and setting_key
            not in cls._normalized_record_setting_names(existing_records)
            else ()
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

    @staticmethod
    def segmentation_inputs(
        artifact_inputs: ArtifactSpecCollection,
    ) -> tuple[ArtifactSpec, ArtifactSpec]:
        """Return primary-object and image inputs from their nominal kinds."""

        object_inputs = artifact_inputs.for_artifact_type(
            ObjectLabelsArtifactType
        ).specs
        image_inputs = artifact_inputs.for_artifact_type(ImageArtifactType).specs
        if len(object_inputs) != 1 or len(image_inputs) != 1:
            raise ValueError(
                "IdentifySecondaryObjects requires one object and one image input, "
                f"got objects={object_inputs!r}, images={image_inputs!r}."
            )
        return object_inputs[0], image_inputs[0]

    @classmethod
    def finalize_artifact_contract_inputs(
        cls,
        module: "ModuleBlock",
        *,
        invocation_key: "FunctionInvocationKey",
        step_context: "ArtifactDeclarationStepContext",
        artifact_inputs: ArtifactSpecCollection,
    ) -> tuple[ArtifactSpec, ...]:
        inputs = ArtifactSpecCollection(
            super().finalize_artifact_contract_inputs(
                module,
                invocation_key=invocation_key,
                step_context=step_context,
                artifact_inputs=artifact_inputs,
            )
        )
        object_input, image_input = cls.segmentation_inputs(inputs)
        projected_object_input = object_input.with_group_scope_relation(
            InputGroupLineageSourceRelation(source=image_input.ref())
        )
        return tuple(
            projected_object_input if spec.ref() == object_input.ref() else spec
            for spec in inputs.specs
        )

    @classmethod
    def artifact_contract_outputs(
        cls,
        module,
        *,
        invocation_key,
        step_context,
        artifact_inputs: ArtifactSpecCollection,
    ):
        object_input, image_input = cls.segmentation_inputs(artifact_inputs)
        replacement_primary_active = cls.replacement_primary_output_active(module)
        output_name = required_setting_value(module, cls.output_objects_setting)
        output_objects = ArtifactSpec.output(
            output_name,
            ObjectLabelsArtifactType,
            relations=(SourceStackLineageSourceRelation(source=image_input.ref()),),
        )
        replacement_primary_objects = (
            ArtifactSpec.output(
                required_setting_value(
                    module,
                    cls.replacement_primary_objects_setting,
                ),
                ObjectLabelsArtifactType,
                relations=(
                    IdentifySecondaryObjectsReplacementPrimarySourceRelation(
                        source=object_input.ref()
                    ),
                ),
            )
            if replacement_primary_active
            else None
        )
        relationship_outputs = [
            cls.parent_child_relationship_output_artifact(
                module,
                step_context=step_context,
                parent=object_input,
                child=output_objects,
                lineage_source=image_input,
            )
        ]
        if replacement_primary_objects is not None:
            relationship_outputs.extend(
                (
                    cls.parent_child_relationship_output_artifact(
                        module,
                        step_context=step_context,
                        parent=object_input,
                        child=replacement_primary_objects,
                        lineage_source=image_input,
                    ),
                    cls.parent_child_relationship_output_artifact(
                        module,
                        step_context=step_context,
                        parent=replacement_primary_objects,
                        child=output_objects,
                        lineage_source=image_input,
                    ),
                )
            )
        outputs = [
            cls.measurement_output_artifact(
                module,
                invocation_key=invocation_key,
                step_context=step_context,
                artifact_inputs=artifact_inputs,
            ),
            *relationship_outputs,
            output_objects,
        ]
        if replacement_primary_objects is not None:
            outputs.append(replacement_primary_objects)
        return tuple(outputs)


@processing_prepare(identify_secondary_objects)
def _prepare_identify_secondary_objects() -> None:
    """Compile secondary-object threshold, distance, and propagation kernels."""
    image = np.zeros((64, 64), dtype=np.float32)
    yy, xx = np.ogrid[:64, :64]
    image[(yy - 24) ** 2 + (xx - 24) ** 2 <= 18 * 18] = 0.7
    image[(yy - 40) ** 2 + (xx - 40) ** 2 <= 14 * 14] = 0.5
    labels = np.zeros((64, 64), dtype=np.int32)
    labels[20:28, 20:28] = 1
    labels[36:44, 36:44] = 2
    primary_labels = SourceImageObjectLabelBuildRequest(
        image=image,
        labels=labels,
        declared_object_ids=(1, 2),
    ).payload()
    identify_secondary_objects.__wrapped__(
        image,
        primary_labels,
        method=SecondaryMethod.PROPAGATION,
        threshold_method=CellProfilerThresholdMethod.OTSU,
        threshold_smoothing_scale=1.3488,
        regularization_factor=0.05,
    )
    identify_secondary_objects.__wrapped__(
        image,
        primary_labels,
        method=SecondaryMethod.DISTANCE_B,
        threshold_method=CellProfilerThresholdMethod.OTSU,
        threshold_smoothing_scale=1.3488,
        distance_to_dilate=8,
    )


@dataclass
class TertiaryObjectStats:
    """Runtime measurements emitted by IdentifyTertiaryObjects."""

    slice_index: Annotated[int, MeasurementRowAxisField.SLICE_INDEX]
    object_count: int
    mean_area: float
    primary_parent_count: int
    secondary_parent_count: int


@dataclass(frozen=True, slots=True)
class TertiaryObjectLabelOutput:
    """Typed tertiary labels preserving the secondary object-label domain."""

    source: ObjectLabelSource
    labels: np.ndarray

    def value(self) -> ObjectLabelSource:
        return object_label_value_with_dense_labels(
            self.source,
            self.labels,
            domain_declaration=PresentObjectLabelIdsDomainDeclaration(),
            representation=ObjectLabelRepresentation.DENSE_LABELS,
        )


@dataclass(frozen=True, slots=True)
class TertiaryObjectMeasurement:
    """Nominal measurement summary for one tertiary label plane."""

    labels: np.ndarray

    @property
    def positive_label_count(self) -> int:
        return int(np.count_nonzero(np.bincount(np.asarray(self.labels).ravel())[1:]))

    @property
    def positive_label_mean_area(self) -> tuple[int, float]:
        areas = np.bincount(np.asarray(self.labels).ravel())[1:]
        positive_areas = areas[areas > 0]
        if positive_areas.size == 0:
            return (0, 0.0)
        return (int(positive_areas.size), float(np.mean(positive_areas)))


@dataclass(frozen=True, slots=True)
class TertiaryPrimaryMaskPlane:
    """Primary-object label plane as the mask authority for tertiary segmentation."""

    plane: np.ndarray

    def retained_secondary_mask(
        self, *, shrink_primary: bool, outline_backend_provider: BackendProviderInput
    ) -> np.ndarray:
        if not shrink_primary:
            return self.plane == 0
        primary_outline = ObjectOutlineBackendStrategy.for_memory_type(
            backend_provider=outline_backend_provider
        ).outline(self.plane)
        return np.logical_or(self.plane == 0, primary_outline > 0)

    @property
    def object_count(self) -> int:
        return TertiaryObjectMeasurement(self.plane).positive_label_count


@dataclass(frozen=True, slots=True)
class TertiarySecondaryRegionPlane:
    """Secondary-object label plane that supplies tertiary region identities."""

    plane: np.ndarray

    def initial_tertiary_labels(self) -> np.ndarray:
        return self.plane.copy()

    @property
    def object_count(self) -> int:
        return TertiaryObjectMeasurement(self.plane).positive_label_count


@dataclass(frozen=True, slots=True)
class TertiaryObjectInputs:
    """Dense aligned label inputs and source payloads for tertiary segmentation."""

    secondary_source: ObjectLabelSource
    primary_plane: TertiaryPrimaryMaskPlane
    region_labels: TertiarySecondaryRegionPlane

    @classmethod
    def from_labels(
        cls, primary_labels: ObjectLabelValue, secondary_labels: ObjectLabelValue
    ) -> "TertiaryObjectInputs":
        if not isinstance(primary_labels, ObjectLabelValue) or not isinstance(
            secondary_labels, ObjectLabelValue
        ):
            raise TypeError(
                "IdentifyTertiaryObjects requires runtime-projected ObjectLabelValue inputs."
            )
        aligned_values, _adapters = SourceSpatialDomainAdapter.aligned_values(
            (primary_labels, secondary_labels)
        )
        primary_label_plane, secondary_label_plane = aligned_values
        primary_label_plane = np.asarray(primary_label_plane, dtype=np.int32)
        secondary_label_plane = np.asarray(secondary_label_plane, dtype=np.int32)
        if primary_label_plane.ndim != 2 or secondary_label_plane.ndim != 2:
            raise ValueError(
                "IdentifyTertiaryObjects requires runtime-projected 2-D label planes."
            )
        if primary_label_plane.shape != secondary_label_plane.shape:
            raise ValueError(
                f"Primary and secondary label shapes must match. Got {primary_label_plane.shape} vs {secondary_label_plane.shape}"
            )
        return cls(
            secondary_source=secondary_labels,
            primary_plane=TertiaryPrimaryMaskPlane(primary_label_plane),
            region_labels=TertiarySecondaryRegionPlane(secondary_label_plane),
        )

    def restore_secondary_labels(self, labels: np.ndarray) -> np.ndarray:
        """Return derived labels in the secondary object's native XY domain."""
        adapter = SourceSpatialDomainAdapter.for_value(self.secondary_source)
        if adapter is None:
            raise TypeError(
                "IdentifyTertiaryObjects requires a registered source-spatial "
                "adapter for its secondary object labels."
            )
        return np.asarray(
            adapter.extract_source_array(
                labels,
                spatial_axes_yx=adapter.spatial_axes_yx,
            )
        )


class TertiaryObjectSegmentation:
    """Load-bearing IdentifyTertiaryObjects segmentation policy."""

    def segment(
        self,
        inputs: TertiaryObjectInputs,
        *,
        shrink_primary: bool,
        outline_backend_provider: BackendProviderInput,
    ) -> np.ndarray:
        tertiary_labels = inputs.region_labels.initial_tertiary_labels()
        primary_mask = inputs.primary_plane.retained_secondary_mask(
            shrink_primary=shrink_primary,
            outline_backend_provider=outline_backend_provider,
        )
        tertiary_labels[~primary_mask] = 0
        return tertiary_labels

    def stats(
        self,
        inputs: TertiaryObjectInputs,
        tertiary_labels: np.ndarray,
        *,
        slice_index: int = 0,
    ) -> TertiaryObjectStats:
        object_count, mean_area = TertiaryObjectMeasurement(
            tertiary_labels
        ).positive_label_mean_area
        return TertiaryObjectStats(
            slice_index=slice_index,
            object_count=object_count,
            mean_area=float(mean_area),
            primary_parent_count=inputs.primary_plane.object_count,
            secondary_parent_count=inputs.region_labels.object_count,
        )


@numpy(contract=ProcessingContract.PURE_2D)
@special_inputs("secondary_labels", "primary_labels")
def identify_tertiary_objects(
    image: np.ndarray,
    secondary_labels: ObjectLabelValue,
    primary_labels: ObjectLabelValue,
    shrink_primary: bool = True,
    outline_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
) -> Tuple[
    np.ndarray,
    DirectedObjectRelationshipPayload,
    DirectedObjectRelationshipPayload,
    DataclassMeasurementColumnarRows,
    ObjectLabelValue,
]:
    """Identify tertiary objects by subtracting primary labels from secondary labels.

    Args:
        secondary_labels: Outer object labels from which primary regions are
            subtracted to form tertiary objects.
        primary_labels: Inner object labels removed from each corresponding
            secondary object.
    """
    inputs = TertiaryObjectInputs.from_labels(primary_labels, secondary_labels)
    segmentation = TertiaryObjectSegmentation()
    tertiary_labels = segmentation.segment(
        inputs,
        shrink_primary=shrink_primary,
        outline_backend_provider=outline_backend_provider,
    )
    output_labels = inputs.restore_secondary_labels(tertiary_labels)
    tertiary_output = TertiaryObjectLabelOutput(
        inputs.secondary_source, output_labels
    ).value()
    return (
        image,
        object_label_parent_child_payload(inputs.region_labels.plane, tertiary_labels),
        object_label_parent_child_payload(
            inputs.primary_plane.plane,
            tertiary_labels,
            child_region_labels=inputs.region_labels.plane,
        ),
        DataclassMeasurementColumnarRows(
            (segmentation.stats(inputs, tertiary_labels),),
            row_type=TertiaryObjectStats,
        ),
        tertiary_output,
    )


@processing_prepare(identify_tertiary_objects)
def _prepare_identify_tertiary_objects() -> None:
    """Compile tertiary-object kernels before benchmarked execution."""
    image = np.zeros((32, 32), dtype=np.float32)
    primary = np.zeros((32, 32), dtype=np.int32)
    secondary = np.zeros((32, 32), dtype=np.int32)
    primary[10:20, 10:20] = 1
    secondary[6:24, 6:24] = 1
    identify_tertiary_objects.__wrapped__(
        image,
        ObjectLabelPayload(
            variant_data=ObjectLabelVariantData(labels=secondary),
            domain=ObjectLabelDomain(declared_object_ids=(1,)),
        ),
        ObjectLabelPayload(
            variant_data=ObjectLabelVariantData(labels=primary),
            domain=ObjectLabelDomain(declared_object_ids=(1,)),
        ),
        shrink_primary=True,
    )


__all__ = public_names_from_objects(
    DistanceMaskedSegmentationStrategy,
    DistanceOnlySegmentationStrategy,
    GradientWatershedSegmentationStrategy,
    ImageWatershedSegmentationStrategy,
    IdentifySecondaryObjectsModule,
    LabelPropagationResult,
    NumbaSecondaryDistanceTransformBackendStrategy,
    NumbaSecondaryPropagationBackendStrategy,
    NumpySecondaryDistanceTransformBackendStrategy,
    PropagationSegmentationStrategy,
    SecondaryDistanceTransformBackendStrategy,
    SecondaryMethod,
    SecondaryObjectLabels,
    SecondaryObjectStats,
    SecondaryPropagationBackendStrategy,
    SecondarySegmentationRequest,
    SecondarySegmentationStrategy,
    ThresholdCalculator,
    ThresholdCalculatorDeclaration,
    ThresholdMethod,
    TertiaryObjectInputs,
    TertiaryObjectLabelOutput,
    TertiaryObjectMeasurement,
    TertiaryObjectSegmentation,
    TertiaryObjectStats,
    identify_secondary_objects,
    identify_secondary_objects_with_replacement_primary,
    identify_tertiary_objects,
    secondary_propagation_backend,
    extra_names=("THRESHOLD_CALCULATOR_DECLARATIONS",),
)
