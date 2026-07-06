"""Thin CellProfiler-style view over OpenHCS runtime artifacts."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field, replace
from enum import Enum
from functools import lru_cache
import logging
import time
from types import MappingProxyType
from typing import Any, cast

import numpy as np

from openhcs.constants.constants import (
    AllComponents,
    Backend,
    FileFormat,
    VariableComponents,
    get_multiprocessing_axis,
)
from openhcs.core.artifacts import (
    ArtifactInputPlan,
    ArtifactOutputPlan,
    ArtifactType,
    ImageArtifactType,
    ObjectLabelsArtifactType,
    MeasurementsArtifactType,
    RelationshipsArtifactType,
    SpatialGridArtifactType,
)
from openhcs.core.image_shapes import is_color_image_slice
from openhcs.core.image_stack_layout import ImageStackLayout
from openhcs.core.memory import detect_memory_type, stack_slices
from openhcs.core.aligned_image_payload import payload_slices_for_alignment
from openhcs.core.pipeline_image_schema import SOURCE_IMAGE_TYPE_METADATA_FIELD
from openhcs.core.source_image_semantics import SourceImagePayloadSemantics
from openhcs.core.source_bindings import (
    CompiledSourceBindingPlan,
    MetadataSelector,
    NamedSourceBinding,
    SourceBindingMatchDimension,
    SourceBindingMatchField,
    SourceBindingMatchMethod,
    SourceBindingMatchPlan,
    SourceBindingRuntimeContext,
    SourceBindingOrigin,
    SourceRuntimePathLookup,
)
from openhcs.core.source_load_plan import SourceLoadPlan
from openhcs.core.source_metadata import SourceMetadataRoleView, SourceVoxelSpacing
from openhcs.core.source_schema_workspace import source_schema_auxiliary_payload
from openhcs.core.source_matching import (
    SourceAxisMetadataScope,
    is_image_path,
    merge_source_metadata,
    metadata_from_rules,
    source_filters_match,
    SourceImageSetIdentity,
    SourceImageSetIdentityPolicy,
    source_component_metadata_values,
    semantic_source_metadata_value,
    source_metadata_component,
    source_metadata_values_equal,
    source_metadata_value,
)
from openhcs.core.source_path_identity import (
    source_path_identity_key,
    source_paths_equal,
)
from openhcs.core.steps.function_output_identity import FunctionOutputIdentityCache
from openhcs.core.runtime_stores import (
    RuntimeArtifactGroupTarget,
    RuntimeArtifactLocation,
    RuntimeArtifactLocationTarget,
    RuntimeArtifactQuery,
    RuntimeValueStore,
    StoredRuntimeValue,
    replace_runtime_artifact_payload,
)
from openhcs.core.runtime_invocation import RuntimeSliceAlignedValues
from openhcs.core.runtime_slice_projection import (
    RuntimeSliceProjection,
    RuntimeProjectionAxis,
)
from openhcs.core.runtime_profile import RuntimeProfileLogger
from openhcs.core.runtime_artifact_queries import (
    MeasurementTableUnion,
    RuntimeArtifactQueryContext,
    runtime_measurement_tables,
    runtime_relationship,
    runtime_spatial_grid,
)
from openhcs.core.runtime_adapters import RuntimeExecutionAxisScope
from openhcs.core.measurement_feature_queries import MeasurementTableObjectFeatureSemantics
from openhcs.core.runtime_semantics import (
    ObjectLabelDomain,
    ObjectLabelDomainScope,
    RelationshipSemantics,
    RuntimePlaneAxis,
    RuntimePlaneProjection,
    RuntimePlaneAxisProjector,
)
from openhcs.core.runtime_values import (
    ColumnarRows,
    FieldSpec,
    ImagePayloadMetadata,
    ImagePayloadMetadataCompositionRequest,
    ImagePayloadMetadataInput,
    ImagePayloadSourceMetadataContext,
    MeasurementTable,
    NamedImage,
    NativeRuntimeValue,
    ObjectLabelData,
    ObjectLabelPure2DSliceAggregator,
    ObjectLabelRepresentation,
    ObjectLabelSet,
    ObjectLabelValue,
    ObjectLabelValueConstructionContext,
    ObjectRelationship,
    RuntimeArrayPayload,
    RuntimeImagePayloadContext,
    RuntimeImageSourceIdentityCompleteness,
    RuntimeValue,
    SourceAlignedObjectLabelProvenanceRequest,
    SourceImageObjectLabelBuildRequest,
    SourceImagePlaneAxisPolicy,
    SourceImagePlaneAxisRequest,
    SparseIJVLabelRows,
    SpatialGrid,
    image_payload_data,
    image_payload_mask,
    image_payload_metadata,
    image_payload_slice_context,
    normalize_artifact_value,
    object_label_dense_array,
)
from openhcs.core.source_image_provenance import (
    SourceComponentMetadata,
    SourceImageIdentity,
    SourceImageProvenancePlanes,
)
from openhcs.interop.cellprofiler.runtime.source_identity import (
    CellProfilerCurrentImage,
    CellProfilerParsedMetadataValue,
    CurrentSourcePayloadPlaneSelection,
    CurrentSourcePayloadPlaneSelectionAuthority,
    CurrentSourcePayloadPlaneSelectionRequest,
    MutableParsedSourceMetadata,
    ParsedSourceCandidateABC,
    ParsedSourceCandidatePathIdentity,
    ParsedSourceMetadata,
    RuntimeRecordSourceImageSetSelector,
    RuntimeSourceIdentityAdapterABC,
    SourceBindingRuntimePathIdentities,
    SourceImageSetIdentityCompatibility,
    SourcePayloadPlaneIdentitySequence,
    SourcePlaneIdentitySequence,
    SourceScopedPayload,
    _SOURCE_PLANE_IDENTITY_POLICY,
)
from openhcs.interop.cellprofiler.runtime.payload_types import (
    CellProfilerFilePayload,
    CellProfilerMeasurementVector,
    ImagePayloadMaskValue,
    ImagePayloadValue,
    MeasurementRowsInput,
    RuntimeArtifactNormalizationInput,
    RuntimeArtifactPayloadValue,
)
from openhcs.interop.cellprofiler.runtime.adapter_protocols import (
    CellProfilerFileManager,
    CellProfilerFileManagerOption,
    CellProfilerFilenameParser,
    CellProfilerMicroscopeHandler,
    CellProfilerProcessingContext,
    RequireProcessingContextBoundaryPolicy,
)
from openhcs.interop.cellprofiler.runtime.projection import (
    CurrentSourceImagePayloadProjection,
    CurrentSourceObjectLabelPayloadProjection,
    CurrentSourcePayloadPlaneSelector,
    RuntimePlaneCurrentImageContext,
    RuntimePlaneImagePayloadProjection,
    RuntimePlaneProjectionContext,
)
from openhcs.interop.cellprofiler.runtime.projection_requirements import (
    CellProfilerRuntimePlaneProjectionCapability,
    CurrentSourceImagePayloadProjectionCapability,
    RuntimeArtifactImageInputProjectionCapability,
    projection_capabilities_include,
)
from openhcs.interop.cellprofiler.runtime.adapter_scope import (
    CellProfilerRuntimeScope,
    CurrentSourceIdentityCacheScope,
    RuntimeGroupMatchScope,
)
from openhcs.interop.cellprofiler.runtime.adapter_profile import (
    AdapterProfileFieldValue,
    AdapterProfileLog,
    NativeRecordProfileContext,
    SourceCandidateProfileEvent,
)
from openhcs.interop.cellprofiler.runtime.object_label_measurements import (
    object_label_measurement_values_cache,
)
from openhcs.interop.cellprofiler.runtime.object_measurement_tables import (
    CellProfilerMeasurementCacheValue,
    MeasurementTableCacheMutation,
    MeasurementTableCacheMutationPolicy,
    MeasurementTableSelection,
    object_measurement_table_cache,
    object_measurement_table_index_cache,
)
from openhcs.interop.cellprofiler.runtime.runtime_artifact_records import (
    CurrentSourceRuntimeInputGroupResolution,
    RuntimeArtifactRecordResolver,
)
from openhcs.interop.cellprofiler.runtime.runtime_artifact_cache_invalidation import (
    FullRuntimeArtifactCacheInvalidationPolicy,
    ImageRuntimeArtifactCacheInvalidationPolicy,
    MeasurementRuntimeArtifactCacheInvalidationPolicy,
    ObjectLabelRuntimeArtifactCacheInvalidationPolicy,
    RelationshipRuntimeArtifactCacheInvalidationPolicy,
    RuntimeArtifactCacheInvalidationPolicy,
)
from openhcs.interop.cellprofiler.runtime.runtime_value_authorities import (
    MatlabPayloadEntryName,
    RuntimeRecordStackAuthority,
    SpatialGridGroupValues,
    SpatialGridInput,
    SpatialGridValueAuthority,
)
from openhcs.interop.cellprofiler.runtime.source_candidates import (
    CELLPROFILER_SOURCE_ORDER_PROCESS_CACHE,
    CellProfilerImageNumberResolver,
    SourceBindingPlaneCandidateContext,
    SourceCandidateRuntimeCache,
    cellprofiler_source_order_identity,
    source_candidate_process_cache_key,
)
from openhcs.interop.cellprofiler.runtime.source_binding_runtime import (
    SourceBindingAxisResolutionAuthority,
    SourceBindingAxisPlaneResolution,
    SourceBindingResolutionRequest,
    SourceBindingResolver,
)

logger = logging.getLogger(__name__)
AdapterObjectLabelInput = ObjectLabelValue | ObjectLabelData
RelationshipIdVector = np.ndarray | Sequence[int]

@dataclass(frozen=True, slots=True)
class SourceIdentitySetCardinality:
    """Closed cardinality authority for current source identity sets."""

    identities: frozenset[SourceImageSetIdentity]

    @property
    def has_single_identity(self) -> bool:
        return len(self.identities) in {1}

@dataclass(frozen=True, slots=True)
class DeclaredOutputResolution:
    """Resolution of a compiled output declaration for one artifact name."""

    plan: ArtifactOutputPlan | None = None

    @property
    def is_declared(self) -> bool:
        return self.plan is not None

@dataclass(slots=True)
class CellProfilerRuntimeAdapter(RuntimeSourceIdentityAdapterABC, RuntimePlaneAxisProjector):
    """CellProfiler-like API backed by typed OpenHCS runtime state.

    The adapter deliberately has no object/image/measurement dictionaries of its
    own. Writes require compiled output plans and a filemanager so the
    RuntimeValueStore record and VFS payload stay aligned with the normal
    FunctionStep runtime boundary.
    """

    @classmethod
    def require_parameter_name(cls) -> str:
        """Return the callable ABI name for CellProfiler adapter injection."""
        return "cellprofiler_runtime"

    runtime_value_store: RuntimeValueStore
    axis_scope: RuntimeExecutionAxisScope
    artifact_inputs: Mapping[str, ArtifactInputPlan] = field(default_factory=dict)
    artifact_outputs: Mapping[str, ArtifactOutputPlan] = field(default_factory=dict)
    source_binding_plan: CompiledSourceBindingPlan = field(
        default_factory=CompiledSourceBindingPlan.empty
    )
    source_binding_context: SourceBindingRuntimeContext = field(
        default_factory=SourceBindingRuntimeContext.empty
    )
    group_key: str | None = None
    processing_context: CellProfilerProcessingContext | None = None
    filename_parser: CellProfilerFilenameParser | None = None
    filemanager: CellProfilerFileManager | None = None
    output_identity_cache: FunctionOutputIdentityCache = field(
        default_factory=FunctionOutputIdentityCache
    )
    backend: str = Backend.MEMORY.value
    plane_projection: RuntimePlaneProjection = field(
        default_factory=RuntimePlaneProjection.stack
    )
    variable_components: tuple[VariableComponents, ...] = ()
    source_load_plan: SourceLoadPlan = field(default_factory=SourceLoadPlan)
    _source_paths_by_image_name_cache: dict[
        tuple[int, str],
        tuple[str, ...],
    ] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _image_cache: dict[tuple[str | None, str], NamedImage] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _current_image_cache: dict[tuple[int, str | None, str, int], NamedImage] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _object_cache: dict[tuple[str | None, str], ObjectLabelSet] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _current_object_cache: dict[
        tuple[int, str | None, str, int],
        ObjectLabelSet,
    ] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _measurement_cache: dict[Hashable, CellProfilerMeasurementCacheValue] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _artifact_availability_cache: dict[tuple[Hashable, ...], bool] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _source_candidate_cache: dict[
        tuple[str, ...],
        tuple["ParsedSourceCandidate", ...],
    ] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _source_candidate_process_key_cache: dict[
        tuple[str, ...],
        tuple[Hashable, ...],
    ] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    def __post_init__(self) -> None:
        if not isinstance(self.runtime_value_store, RuntimeValueStore):
            raise TypeError(
                "CellProfilerRuntimeAdapter.runtime_value_store must be "
                f"RuntimeValueStore, got {type(self.runtime_value_store).__name__}."
            )

        if not self.backend:
            raise ValueError("CellProfilerRuntimeAdapter.backend cannot be empty.")
        if not isinstance(self.source_binding_plan, CompiledSourceBindingPlan):
            raise TypeError(
                "CellProfilerRuntimeAdapter.source_binding_plan must be "
                "CompiledSourceBindingPlan, got "
                f"{type(self.source_binding_plan).__name__}."
            )
        if not isinstance(self.source_binding_context, SourceBindingRuntimeContext):
            raise TypeError(
                "CellProfilerRuntimeAdapter.source_binding_context must be "
                "SourceBindingRuntimeContext, got "
                f"{type(self.source_binding_context).__name__}."
            )
        if not isinstance(self.plane_projection, RuntimePlaneProjection):
            raise TypeError(
                "CellProfilerRuntimeAdapter.plane_projection must be "
                "RuntimePlaneProjection, got "
                f"{type(self.plane_projection).__name__}."
            )
        if not isinstance(self.axis_scope, RuntimeExecutionAxisScope):
            raise TypeError(
                "CellProfilerRuntimeAdapter.axis_scope must be "
                "RuntimeExecutionAxisScope, got "
                f"{type(self.axis_scope).__name__}."
            )
        if not isinstance(self.source_load_plan, SourceLoadPlan):
            raise TypeError(
                "CellProfilerRuntimeAdapter.source_load_plan must be "
                "SourceLoadPlan, got "
                f"{type(self.source_load_plan).__name__}."
            )

        outputs = dict(self.artifact_outputs)
        for name, plan in outputs.items():
            if not isinstance(plan, ArtifactOutputPlan):
                raise TypeError(
                    f"artifact_outputs['{name}'] must be ArtifactOutputPlan, "
                    f"got {type(plan).__name__}."
                )
            if name != plan.name:
                raise ValueError(
                    f"artifact_outputs key '{name}' does not match plan name "
                    f"'{plan.name}'."
                )
        self.artifact_outputs = MappingProxyType(outputs)
        if self.group_key is not None:
            self.group_key = str(self.group_key)

    def cellprofiler_source_order_path(self, path: str) -> str:
        """Return the source path identity used for CellProfiler image ordering."""
        source_paths = self.source_binding_context.step_input_source_paths
        mapped = source_paths.get(path)
        if mapped is None:
            mapped = path
        return source_path_identity_key(mapped)

    def cellprofiler_ordered_pipeline_image_paths(self) -> tuple[str, ...]:
        """Return loadable pipeline input paths in CellProfiler image order."""
        context = self.source_binding_context
        cache_key = (
            "ordered_pipeline_image_paths",
            context.pipeline_input_files_identity,
            tuple(sorted(context.step_input_source_paths.items())),
        )
        cache = CELLPROFILER_SOURCE_ORDER_PROCESS_CACHE
        cached = cache.cached_value(cache_key)
        if cached is not None:
            return tuple(cached)
        ordered = tuple(
            dict.fromkeys(
                self.cellprofiler_source_order_path(path)
                for path in context.pipeline_input_files_identity
                if is_image_path(path)
            )
        )
        return tuple(cache.store_value(cache_key, ordered))

    def source_axis_metadata_scope(self) -> "SourceAxisMetadataScope":
        """Return the execution-local source metadata scope for this axis."""
        return self.axis_scope.source_axis_metadata_scope()

    def source_binding_runtime_path_identities(
        self,
    ) -> "SourceBindingRuntimePathIdentities":
        """Return source-binding path identities for the current runtime scope."""
        source_context = self.source_binding_context
        step_input_source_paths = source_context.step_input_source_paths
        current_step_input_files = source_context.current_step_input_files
        return SourceBindingRuntimePathIdentities(
            current_step_input=ParsedSourceCandidatePathIdentity.from_paths(
                tuple(current_step_input_files)
                + tuple(
                    step_input_source_paths[path]
                    for path in current_step_input_files
                    if path in step_input_source_paths
                )
            ),
            virtual_step_input=ParsedSourceCandidatePathIdentity.from_paths(
                tuple(step_input_source_paths.keys())
                + tuple(step_input_source_paths.values())
            ),
        )

    def cellprofiler_current_step_source_paths(self) -> tuple[str, ...]:
        """Return source-path identities represented by this step invocation."""
        source_context = self.source_binding_context
        step_input_source_paths = source_context.step_input_source_paths
        return tuple(
            dict.fromkeys(
                path
                for current_path in source_context.current_step_input_files
                for path in (
                    current_path,
                    step_input_source_paths.get(current_path),
                )
                if path is not None
            )
        )

    def cellprofiler_axis_image_number_start(self) -> int:
        """Return CP's 1-based image number for this runtime axis."""
        return self._cellprofiler_axis_image_number_start(
            use_current_step_source_paths=True,
        )

    def _cellprofiler_axis_image_number_start(
        self,
        *,
        use_current_step_source_paths: bool,
    ) -> int:
        """Return CP's axis ImageNumber, optionally using current step provenance."""
        if self.processing_context is None:
            cache_key = (
                "axis_image_number_start",
                self.axis_scope.cache_key,
                "no_processing_context",
            )
            cache = CELLPROFILER_SOURCE_ORDER_PROCESS_CACHE
            cached = cache.cached_value(cache_key)
            if cached is not None:
                return int(cached)
            return int(cache.store_value(cache_key, 1))

        current_step_source_paths = (
            self.cellprofiler_current_step_source_paths()
            if use_current_step_source_paths
            else ()
        )
        axis_scope = self.source_axis_metadata_scope()
        if not current_step_source_paths and not axis_scope.has_component:
            return 1

        parser_identity = RequireProcessingContextBoundaryPolicy(
            self
        ).context.microscope_handler.parser.semantic_identity()
        cache_key = (
            "axis_image_number_start",
            self.axis_scope.cache_key,
            current_step_source_paths,
            self.source_binding_context.pipeline_input_files_identity,
            tuple(sorted(self.source_binding_context.step_input_source_paths.items())),
            self.source_binding_plan.metadata_rules,
            parser_identity,
        )
        cache = CELLPROFILER_SOURCE_ORDER_PROCESS_CACHE
        cached = cache.cached_value(cache_key)
        if cached is not None:
            return int(cached)

        resolver = CellProfilerImageNumberResolver.for_adapter(self)
        image_number = resolver.image_number_start_for_paths(current_step_source_paths)
        if image_number is not None:
            return int(cache.store_value(cache_key, image_number))

        image_number = resolver.image_number_start_for_axis_scope(axis_scope)
        return int(cache.store_value(cache_key, image_number))

    def cellprofiler_image_number_start_for_source_paths(
        self,
        source_paths: tuple[str, ...],
    ) -> int:
        """Return the first CP ImageNumber represented by a source-path group."""
        image_number = self.cellprofiler_image_number_for_source_paths(source_paths)
        if image_number is not None:
            return image_number
        return self._cellprofiler_axis_image_number_start(
            use_current_step_source_paths=False,
        )

    def cellprofiler_image_number_for_source_paths(
        self,
        source_paths: tuple[str, ...],
    ) -> int | None:
        """Return the CP ImageNumber for the first resolvable source path."""
        if not source_paths:
            return None
        source_order_paths = tuple(
            self.cellprofiler_source_order_path(source_path)
            for source_path in source_paths
        )
        parser_identity: Hashable | None = None
        if self.can_resolve_source_candidates:
            parser_identity = RequireProcessingContextBoundaryPolicy(
                self
            ).context.microscope_handler.parser.semantic_identity()
        axis_component_values = self.source_axis_metadata_scope().component_values
        cache_key = (
            "image_number_for_source_paths",
            self.source_binding_context.pipeline_input_files_identity,
            self.source_binding_plan.metadata_rules,
            parser_identity,
            axis_component_values,
            cellprofiler_source_order_identity(self),
            source_order_paths,
        )
        cache = CELLPROFILER_SOURCE_ORDER_PROCESS_CACHE
        cached = cache.cached_value(cache_key)
        if cached is not None:
            return int(cached)
        image_number = CellProfilerImageNumberResolver.for_adapter(
            self
        ).image_number_start_for_paths(source_paths)
        if image_number is not None:
            cache.store_value(cache_key, int(image_number))
        return image_number

    def cellprofiler_image_number_for_payload(
        self,
        payload: ImagePayloadMetadataInput,
    ) -> int | None:
        """Return the CellProfiler ImageNumber for a payload carrying source paths."""
        return CellProfilerImageNumberResolver.for_adapter(self).image_number_for_paths(
            image_payload_metadata(payload).source_image_paths
        )

    def cellprofiler_source_path_for_image_number(
        self,
        image_number: int,
    ) -> str | None:
        """Return a representative source path for one CellProfiler ImageNumber."""
        return CellProfilerImageNumberResolver.for_adapter(self).source_path_for_image_number(
            image_number
        )

    def cellprofiler_source_paths_for_image_name(
        self,
        image_name: str | None,
    ) -> tuple[str, ...]:
        """Return source paths carried by a runtime image with this source name."""
        if image_name is None:
            return ()
        if self.has_source_binding(image_name, ImageArtifactType):
            candidate_context = SourceBindingPlaneCandidateContext.from_adapter(
                self,
                image_name,
                axis_scope=self.source_axis_metadata_scope().image_set_identity_scope(),
            )
            if candidate_context is not None:
                match = candidate_context.match()
                if match is not None:
                    source_paths = tuple(
                        dict.fromkeys(
                            path
                            for candidate_index in match.index_set.indexes
                            for path in (
                                match.universe.plane_candidates.candidate_at(
                                    candidate_index
                                ).paths
                            )
                        )
                    )
                    if source_paths:
                        return source_paths
        cache_key = (self.runtime_value_store.revision, image_name)
        cached = self._source_paths_by_image_name_cache.get(cache_key)
        if cached is not None:
            return cached
        direct_records = self.runtime_value_store.find(
            name=image_name,
            artifact_type=ImageArtifactType,
            axis_id=self.axis_scope.axis_id,
        )
        lineage_records = self.runtime_value_store.find(
            artifact_type=ImageArtifactType,
            axis_id=self.axis_scope.axis_id,
        )
        for record in (*direct_records, *lineage_records):
            if (
                record.key.name != image_name
                and record.value.schema.source_image_name != image_name
            ):
                continue
            source_paths = image_payload_metadata(record.value.data).source_image_paths
            if source_paths:
                self._source_paths_by_image_name_cache[cache_key] = source_paths
                return source_paths
        self._source_paths_by_image_name_cache[cache_key] = ()
        return ()

    def source_image_payload_for_name(
        self,
        image_name: str,
        current_image: ImagePayloadValue | None,
    ) -> ImagePayloadValue | None:
        """Resolve an image name through source bindings or runtime image records."""
        if current_image is not None and self.has_source_binding(
            image_name,
            ImageArtifactType,
        ):
            return cast(
                ImagePayloadValue,
                self.resolve_source_image(image_name, current_image),
            )
        if not self.has_runtime_artifact(name=image_name, kind=ImageArtifactType):
            return None
        return cast(ImagePayloadValue, self.get_image(image_name).data)

    def invalidate_runtime_query_caches_for_artifact_type(
        self,
        artifact_type: type[ArtifactType],
    ) -> None:
        """Invalidate adapter caches whose semantic domain can change for ``artifact_type``."""
        RuntimeArtifactCacheInvalidationPolicy.for_artifact_type(
            artifact_type
        ).invalidate(self)

    def require_artifact_available(self, *, name: str, kind: ArtifactType) -> None:
        """Fail loudly unless an artifact is declared, bound, or resolvable."""
        cache_key = (
            "artifact_available",
            self.runtime_value_store.revision,
            self.group_key,
            name,
            kind,
        )
        if self._artifact_availability_cache.get(cache_key):
            return
        if self._declared_output_resolution(name, kind).is_declared:
            self._artifact_availability_cache[cache_key] = True
            return
        if self.has_source_binding(name, kind):
            self._artifact_availability_cache[cache_key] = True
            return
        RuntimeArtifactRecordResolver(
            adapter=self,
            name=name,
            artifact_type=kind,
            group_key=None,
            current_image=None,
        ).resolve()
        self._artifact_availability_cache[cache_key] = True

    def has_runtime_artifact(self, *, name: str, kind: ArtifactType) -> bool:
        """Return whether this execution scope contains a runtime artifact."""
        return bool(
            RuntimeGroupMatchScope(group_key=None)
            .runtime_scope(self)
            .artifact_query_context()
            .find(name=name, artifact_type=kind)
        )

    def requires_declared_source_image_domain(self, image_name: str) -> bool:
        """Return whether object labels must inherit source metadata from an image."""
        input_plan = self.artifact_inputs.get(image_name)
        if input_plan is not None:
            return input_plan.artifact_type is ImageArtifactType
        return False

    def resolve_source_image(
        self,
        alias: str,
        current_image: ImagePayloadValue,
    ) -> ImagePayloadValue:
        request = self._source_resolution_request(
            alias,
            ImageArtifactType,
            current_image,
        )
        image = SourceBindingResolver.for_origin(request.binding.origin).resolve_image(
            request
        )
        if image is current_image:
            return image
        source_metadata = image_payload_metadata(image)
        metadata = replace(
            source_metadata,
            source_image_names=source_metadata.source_image_names or (alias,),
        )
        return self.image_payload_for_current_runtime_plane(
            cast(
                ImagePayloadValue,
                metadata.payload_with(
                    image_payload_data(image),
                    mask=image_payload_mask(image),
                ),
            ),
            current_image=current_image,
        )

    def image_payload_for_current_runtime_plane(
        self,
        payload: ImagePayloadValue,
        *,
        current_image: ImagePayloadValue | None,
        projection_capabilities: frozenset[
            type[CellProfilerRuntimePlaneProjectionCapability]
        ]
        | None = None,
    ) -> ImagePayloadValue:
        """Return image payload data projected to this invocation's runtime plane."""
        capabilities = (
            projection_capabilities
            if projection_capabilities is not None
            else frozenset((RuntimeArtifactImageInputProjectionCapability,))
        )
        projected: ImagePayloadValue = payload
        if (
            projection_capabilities_include(
                capabilities,
                CurrentSourceImagePayloadProjectionCapability,
            )
            and current_image is not None
        ):
            projected = CurrentSourceImagePayloadProjection(
                self,
                current_image,
            ).project(projected)
        if not projection_capabilities_include(
            capabilities,
            RuntimeArtifactImageInputProjectionCapability,
        ):
            return projected
        return cast(
            ImagePayloadValue,
            RuntimePlaneImagePayloadProjection(
                RuntimePlaneProjectionContext(
                    adapter=self,
                    current_image_context=RuntimePlaneCurrentImageContext(
                        current_image
                    ),
                ),
            ).project(projected),
        )

    def runtime_slice_plane_index(self) -> int | None:
        """Return the current axis-local runtime-slice plane index."""
        return self.plane_projection.runtime_slice_plane_index()

    def runtime_slice_axis_size(self) -> int | None:
        """Return the current runtime-slice axis size when known."""
        return self.plane_projection.runtime_slice_axis_size()

    def source_binding_axis_plane_index(
        self,
        source_aliases: tuple[str, ...],
    ) -> int | None:
        """Return the current axis-local source-binding plane index."""
        return SourceBindingAxisResolutionAuthority.plane_index(
            self,
            tuple(source_aliases),
        )

    def source_binding_axis_plane_resolution(
        self,
        source_aliases: tuple[str, ...],
    ) -> SourceBindingAxisPlaneResolution:
        """Return the source-binding plane resolution for aliases."""
        return SourceBindingAxisResolutionAuthority.plane_resolution(
            self,
            tuple(source_aliases),
        )

    def source_binding_axis_size(
        self,
        source_aliases: tuple[str, ...],
    ) -> int | None:
        """Return the current source-binding axis size."""
        return SourceBindingAxisResolutionAuthority.axis_size(
            self,
            tuple(source_aliases),
        )

    def source_binding_plane_index(self, alias: str) -> int | None:
        """Return the current axis-local plane index for a source alias."""
        return SourceBindingAxisResolutionAuthority.plane_index(
            self,
            (alias,),
        )

    def source_binding_plane_candidate_context(
        self,
        alias: str,
    ) -> "SourceBindingPlaneCandidateContext | None":
        return SourceBindingPlaneCandidateContext.from_adapter(
            self,
            alias,
        )

    def resolve_source_objects(
        self,
        alias: str,
        current_image: CellProfilerCurrentImage,
    ) -> ObjectLabelSet:
        request = self._source_resolution_request(
            alias,
            ObjectLabelsArtifactType,
            current_image,
        )
        labels = SourceBindingResolver.for_origin(request.binding.origin).resolve_image(
            request
        )
        label_metadata = image_payload_metadata(labels)
        source_component_metadata = label_metadata.source_component_metadata
        if source_component_metadata is None:
            source_component_metadata = {}
        label_source_metadata = {
            **dict(source_component_metadata),
            SOURCE_IMAGE_TYPE_METADATA_FIELD: "Objects",
        }
        labels = SourceImagePayloadSemantics.from_source_metadata(
            label_source_metadata,
            label_metadata.source_path,
            self.source_binding_context.pipeline_input_backend,
            self.filemanager,
        ).apply(labels)
        return SourceImageObjectLabelBuildRequest(
            image=labels,
            labels=image_payload_data(labels),
        ).label_set(
            name=alias,
            source_image_name=alias,
        )

    def _source_resolution_request(
        self,
        alias: str,
        kind: ArtifactType,
        current_image: CellProfilerCurrentImage,
    ) -> "SourceBindingResolutionRequest":
        return SourceBindingResolutionRequest(
            alias=alias,
            binding=self._require_source_binding(alias, kind),
            adapter=self,
            current_image=current_image,
        )

    def require_resolvable_source_aliases(
        self,
        aliases: tuple[str, ...],
    ) -> None:
        for alias in aliases:
            self._require_source_binding(alias, ImageArtifactType)

    def has_source_binding(
        self,
        alias: str,
        kind: ArtifactType | None = None,
    ) -> bool:
        binding = self.source_binding_plan.binding_for_alias(alias)
        return binding is not None and (
            kind is None or binding.artifact_kind is kind
        )

    def _require_source_binding(
        self,
        alias: str,
        kind: ArtifactType,
    ) -> NamedSourceBinding:
        binding = self.source_binding_plan.binding_for_alias(alias)
        if binding is None:
            raise RuntimeError(
                f"Missing compiled source binding for CellProfiler "
                f"{kind.value} alias '{alias}' on axis '{self.axis_scope.axis_id}'."
            )
        if binding.artifact_kind is not kind:
            raise RuntimeError(
                f"CellProfiler source binding '{alias}' is declared as "
                f"{binding.artifact_kind.value}, not {kind.value}."
            )
        return binding

    def add_image(
        self,
        name: str,
        data: ImagePayloadValue,
        *,
        dimensions: tuple[str, ...] = (),
        source_image_name: str | None = None,
    ) -> StoredRuntimeValue:
        return self._record_native_value(
            name,
            ImageArtifactType,
            NamedImage(
                name=name,
                data=data,
                dimensions=dimensions,
                source_image_name=source_image_name,
            ),
        )

    def get_image(
        self,
        name: str,
        *,
        group_key: str | None = None,
        current_image: CellProfilerCurrentImage | None = None,
    ) -> NamedImage:
        resolved_group_key = self.group_key if group_key is None else group_key
        if current_image is None:
            cache_key = (resolved_group_key, name)
            cached = self._image_cache.get(cache_key)
            if cached is not None:
                return cached
        else:
            current_cache_key = (
                self.runtime_value_store.revision,
                resolved_group_key,
                name,
                id(current_image),
            )
            cached = self._current_image_cache.get(current_cache_key)
            if cached is not None:
                return cached
        records = RuntimeArtifactRecordResolver(
            adapter=self,
            name=name,
            artifact_type=ImageArtifactType,
            group_key=group_key,
            current_image=current_image,
        ).resolve()
        record = records[-1]
        data = (
            RuntimeRecordStackAuthority.stack_image_records(records)
            if len(records) > 1
            else record.value.data
        )
        data = self.image_payload_for_current_runtime_plane(
            data,
            current_image=current_image,
            projection_capabilities=frozenset(
                (
                    RuntimeArtifactImageInputProjectionCapability,
                    CurrentSourceImagePayloadProjectionCapability,
                )
            )
            if current_image is not None
            else None,
        )
        schema = record.value.schema
        image = NamedImage(
            name=name,
            data=data,
            dimensions=schema.dimensions,
            source_image_name=schema.source_image_name,
        )
        if current_image is None:
            cache_key = (resolved_group_key, name)
            self._image_cache[cache_key] = image
        else:
            self._current_image_cache[current_cache_key] = image
        return image

    def add_objects(
        self,
        name: str,
        labels: AdapterObjectLabelInput,
        *,
        source_image_name: str | None = None,
        source_image_names: tuple[str, ...] = (),
        source_image_payload: ImagePayloadMetadataInput | None = None,
        dimensions: tuple[str, ...] = (),
        representation: ObjectLabelRepresentation = (
            ObjectLabelRepresentation.DENSE_LABELS
        ),
    ) -> StoredRuntimeValue:
        construct_started_at = time.perf_counter()
        if isinstance(labels, ObjectLabelValue):
            provenance_source_names = labels.source_image_names
            if not provenance_source_names:
                provenance_source_names = source_image_names
            source_provenance = labels.source_provenance.with_source_image_names(
                provenance_source_names
            )
            resolved_source_image_name = source_image_name
            if resolved_source_image_name is None:
                resolved_source_image_name = labels.source_image_name
            resolved_dimensions = dimensions
            if not resolved_dimensions:
                resolved_dimensions = labels.dimensions
            normalized_labels = RuntimeRecordStackAuthority.normalize_dense_object_label_payload(
                labels.labels
            )
            normalized_unedited_labels = (
                RuntimeRecordStackAuthority.normalize_dense_object_label_payload(
                    labels.unedited_labels
                )
            )
            normalized_small_removed_labels = (
                RuntimeRecordStackAuthority.normalize_dense_object_label_payload(
                    labels.small_removed_labels
                )
            )
            object_labels = ObjectLabelValueConstructionContext.from_value(
                labels,
                source_provenance=source_provenance,
            ).label_set(
                name=name,
                labels=cast(
                    ObjectLabelData,
                    normalized_labels,
                ),
                unedited_labels=cast(
                    ObjectLabelData | None,
                    normalized_unedited_labels,
                ),
                small_removed_labels=cast(
                    ObjectLabelData | None,
                    normalized_small_removed_labels,
                ),
                source_image_name=resolved_source_image_name,
                dimensions=resolved_dimensions,
                representation=labels.representation,
            )
        else:
            construction_context = ObjectLabelValueConstructionContext(
                domain=ObjectLabelDomain(),
            )
            source_provenance = construction_context.source_provenance
            source_spatial_domain = construction_context.source_spatial_domain
            if source_image_name is not None:
                requires_source_coordinate = self.requires_declared_source_image_domain(
                    source_image_name
                )
                if requires_source_coordinate or self.has_runtime_artifact(
                    name=source_image_name,
                    kind=ImageArtifactType,
                ):
                    source_image = self.get_image(source_image_name)
                    metadata = image_payload_metadata(source_image.data)
                    source_provenance = metadata.source_provenance
                    source_spatial_domain = (
                        metadata.object_label_source_spatial_domain()
                    )
                    if (
                        requires_source_coordinate
                        and not RuntimeImageSourceIdentityCompleteness(
                            source_image.data
                        ).complete()
                    ):
                        raise RuntimeError(
                            "Object labels produced from declared source image "
                            f"{source_image_name!r} require source coordinate "
                            "metadata. The source image artifact did not carry "
                            "source_path, source_component_metadata, or "
                            "source_image_provenance_planes."
                        )
            declared_source_image_names = source_image_names
            if not declared_source_image_names and source_image_name is not None:
                declared_source_image_names = (source_image_name,)
            source_provenance = source_provenance.with_source_image_names(
                declared_source_image_names
            )
            object_labels = ObjectLabelValueConstructionContext(
                domain=ObjectLabelDomain(),
                source_provenance=source_provenance,
                source_spatial_domain=source_spatial_domain,
            ).label_set(
                name=name,
                labels=cast(
                    ObjectLabelData,
                    RuntimeRecordStackAuthority.normalize_dense_object_label_payload(
                        labels
                    ),
                ),
                source_image_name=source_image_name,
                dimensions=dimensions,
                representation=representation,
            )
        if source_image_payload is not None:
            object_labels = object_labels.with_source_image_context(source_image_payload)
            SourceAlignedObjectLabelProvenanceRequest(
                image=source_image_payload,
                labels=object_labels,
                label_name=name,
            ).validate()
        AdapterProfileLog.object_label_artifact(
            "adapter_construct_object_labels",
            time.perf_counter() - construct_started_at,
            artifact_name=name,
            payload_type=type(labels).__name__,
            labels=object_labels,
        )
        return self._record_native_value(
            name,
            ObjectLabelsArtifactType,
            object_labels,
        )

    def add_source_image_objects(
        self,
        name: str,
        labels: ObjectLabelData,
        *,
        source_image_payload: ImagePayloadMetadataInput,
        parent_image_source_voxel_spacing: SourceVoxelSpacing | None = None,
        domain_scope: ObjectLabelDomainScope | None = None,
        source_image_name: str | None = None,
        source_image_names: tuple[str, ...] = (),
        dimensions: tuple[str, ...] = (),
        representation: ObjectLabelRepresentation = (
            ObjectLabelRepresentation.DENSE_LABELS
        ),
    ) -> StoredRuntimeValue:
        construct_started_at = time.perf_counter()
        normalized_labels = RuntimeRecordStackAuthority.normalize_dense_object_label_payload(
            labels
        )
        build_request = SourceImageObjectLabelBuildRequest(
            image=source_image_payload,
            labels=cast(ObjectLabelData, normalized_labels),
            domain_scope=domain_scope,
            parent_image_source_voxel_spacing=parent_image_source_voxel_spacing,
        )
        construction_context = build_request.construction_context()
        provenance_source_names = (
            construction_context.source_provenance.source_image_names
        )
        if not provenance_source_names:
            provenance_source_names = source_image_names
        object_labels = replace(
            construction_context,
            source_provenance=(
                construction_context.source_provenance.with_source_image_names(
                    provenance_source_names
                )
            ),
        ).label_set(
            name=name,
            labels=cast(ObjectLabelData, normalized_labels),
            source_image_name=source_image_name,
            dimensions=dimensions,
            representation=representation,
        )
        object_labels = object_labels.with_source_image_context(source_image_payload)
        SourceAlignedObjectLabelProvenanceRequest(
            image=source_image_payload,
            labels=object_labels,
            label_name=name,
        ).validate()
        AdapterProfileLog.object_label_artifact(
            "adapter_construct_object_labels",
            time.perf_counter() - construct_started_at,
            artifact_name=name,
            payload_type=type(labels).__name__,
            labels=object_labels,
        )
        return self._record_native_value(
            name,
            ObjectLabelsArtifactType,
            object_labels,
        )

    def get_objects(
        self,
        name: str,
        *,
        group_key: str | None = None,
        current_image: CellProfilerCurrentImage | None = None,
    ) -> ObjectLabelSet:
        resolved_group_key = self.runtime_input_group_key(
            name=name,
            artifact_type=ObjectLabelsArtifactType,
            group_key=group_key,
            current_image=current_image,
        )
        cache_key = (resolved_group_key, name)
        if current_image is None:
            cached = self._object_cache.get(cache_key)
            if cached is not None:
                return cached
        else:
            current_cache_key = (
                self.runtime_value_store.revision,
                resolved_group_key,
                name,
                id(current_image),
            )
            cached = self._current_object_cache.get(current_cache_key)
            if cached is not None:
                return cached
        records = RuntimeArtifactRecordResolver(
            adapter=self,
            name=name,
            artifact_type=ObjectLabelsArtifactType,
            group_key=group_key,
            current_image=current_image,
        ).resolve()
        objects = (
            RuntimeRecordStackAuthority.stack_object_label_records(records)
            if len(records) > 1
            else ObjectLabelSet.from_runtime_value(records[0].value)
        )
        if current_image is not None:
            objects = CurrentSourceObjectLabelPayloadProjection(
                self,
                current_image,
            ).project(objects)
        if current_image is None:
            self._object_cache[cache_key] = objects
        else:
            self._current_object_cache[current_cache_key] = objects
        return objects

    def get_objects_across_groups(self, name: str) -> ObjectLabelSet:
        """Return object labels stacked across all producer groups for this axis."""
        records = RuntimeGroupMatchScope(
            group_key=None,
            match_group=False,
        ).runtime_scope(self).artifact_query_context().find(
            name=name,
            artifact_type=ObjectLabelsArtifactType,
        )
        if not records:
            raise RuntimeError(
                f"Missing RuntimeValueStore object-label records for {name!r} "
                f"on axis {self.axis_scope.axis_id!r}."
            )
        if len(records) == 1:
            return ObjectLabelSet.from_runtime_value(records[0].value)
        return RuntimeRecordStackAuthority.stack_object_label_records(records)

    def runtime_input_group_key(
        self,
        *,
        name: str,
        artifact_type: ArtifactType,
        group_key: str | None = None,
        current_image: CellProfilerCurrentImage | None = None,
    ) -> str | None:
        """Return the artifact input group for this adapter/runtime context."""
        requested_group_key = self.group_key if group_key is None else group_key
        input_plan = self.artifact_inputs.get(name)
        if input_plan is None or input_plan.artifact_type is not artifact_type:
            return requested_group_key
        selected = RuntimeGroupMatchScope(group_key=None).runtime_scope(
            self
        ).artifact_group_key(
            input_plan,
            requested_group_key=requested_group_key,
        )
        if selected is not None or current_image is None:
            return selected
        input_group_keys = input_plan.group_keys
        if input_group_keys is None:
            group_keys = set()
        else:
            group_keys = {str(group_key) for group_key in input_group_keys}
        if not group_keys:
            return selected
        current_source_scope = RuntimeRecordSourceImageSetSelector(self, current_image)
        current_source_cardinality = SourceIdentitySetCardinality(
            current_source_scope.current_source_identities()
        )
        step_input_files = self.source_binding_context.step_input_files
        if current_source_cardinality.has_single_identity and step_input_files:
            for candidate in self.source_candidates(
                step_input_files
            ):
                if not self.current_image_matches_source_candidate(current_image, candidate):
                    continue
                for value in candidate.metadata.values():
                    normalized = str(value)
                    if normalized in group_keys:
                        return normalized
        current_step_group = CurrentSourceRuntimeInputGroupResolution(
            adapter=self,
            group_keys=frozenset(group_keys),
            current_image=current_image,
        )
        current_step_group_key = current_step_group.resolve()
        if current_step_group_key is not None:
            return current_step_group_key
        return selected

    def runtime_input_group_key_from_current_sources(
        self,
        group_keys: set[str],
    ) -> str | None:
        """Infer the active input group from this invocation's source files."""
        current_files = self.source_binding_context.current_step_input_files
        if not current_files:
            return None
        candidates = self.source_candidates(current_files)
        universe_files = (
            self.source_binding_context.pipeline_input_files
            or self.source_binding_context.step_input_files
            or current_files
        )
        universe_candidates = self.source_candidates(universe_files)
        field_values = self.current_source_group_field_values(
            candidates,
            universe_candidates,
            group_keys,
        )
        matched_groups = tuple(
            value
            for values in field_values.values()
            if len(values) == 1
            for value in values
        )
        logger.debug(
            "Resolved current source group candidates from fields %s for groups %s",
            field_values,
            sorted(group_keys),
        )
        if len(matched_groups) == 1:
            return matched_groups[0]
        return None

    def runtime_group_key_from_source_payload(
        self,
        group_keys: frozenset[str],
        source_payload: SourceScopedPayload,
        *,
        group_component: AllComponents | None = None,
    ) -> str | None:
        """Infer a runtime group from an artifact payload's source identity."""
        metadata = image_payload_metadata(source_payload)
        metadata_group_key = self.runtime_group_key_from_payload_metadata(
            group_keys,
            metadata,
            group_component=group_component,
        )
        if metadata_group_key is not None:
            return metadata_group_key
        source_paths = metadata.source_image_paths
        if not source_paths:
            return None
        candidates = self.source_candidates(source_paths)
        if not candidates:
            return None
        universe_files = (
            self.source_binding_context.pipeline_input_files
            or self.source_binding_context.step_input_files
            or source_paths
        )
        field_values = self.current_source_group_field_values(
            candidates,
            self.source_candidates(universe_files),
            set(group_keys),
        )
        matched_groups = tuple(
            value
            for values in field_values.values()
            if len(values) == 1
            for value in values
        )
        if len(matched_groups) == 1:
            return matched_groups[0]
        return None

    @staticmethod
    def runtime_group_key_from_payload_metadata(
        group_keys: frozenset[str],
        metadata: ImagePayloadMetadata,
        *,
        group_component: AllComponents | None,
    ) -> str | None:
        """Return a group key carried by source-component payload metadata."""
        if group_component is None:
            return None
        metadata_values = (
            (metadata.source_component_metadata,)
            + metadata.source_image_provenance_planes.component_metadata
        )
        original_group_key = CellProfilerRuntimeAdapter.runtime_group_key_from_values(
            group_keys,
            (
                value
                for source_metadata in metadata_values
                if source_metadata is not None
                for field_name, value in SourceMetadataRoleView(
                    source_metadata
                ).original_items()
                if source_metadata_component(field_name) is group_component
            ),
        )
        if original_group_key is not None:
            return original_group_key
        return CellProfilerRuntimeAdapter.runtime_group_key_from_values(
            group_keys,
            (
                value
                for source_metadata in metadata_values
                if source_metadata is not None
                for value in source_component_metadata_values(
                    source_metadata,
                    group_component,
                )
            ),
        )

    @staticmethod
    def runtime_group_key_from_values(
        group_keys: frozenset[str],
        values: Iterable[str],
    ) -> str | None:
        """Return one planned group key matching source metadata values."""
        matched_group_keys = tuple(
            dict.fromkeys(
                group_key
                for value in values
                for group_key in group_keys
                if source_metadata_values_equal(value, group_key)
            )
        )
        if len(matched_group_keys) == 1:
            return matched_group_keys[0]
        return None

    @staticmethod
    def current_source_group_field_values(
        candidates: tuple["ParsedSourceCandidate", ...],
        universe_candidates: tuple["ParsedSourceCandidate", ...],
        group_keys: set[str],
    ) -> Mapping[str, frozenset[str]]:
        """Return candidate metadata fields whose values can select input groups."""
        return MappingProxyType(
            {
                field_name: frozenset(values)
                for field_name in tuple(
                    dict.fromkeys(
                        field_name
                        for candidate in candidates
                        for field_name in candidate.metadata
                    )
                )
                for values in (
                    tuple(
                        str(candidate.metadata[field_name])
                        for candidate in candidates
                        if field_name in candidate.metadata
                        and str(candidate.metadata[field_name]) in group_keys
                    ),
                )
                if (
                    values
                    and len(values) == len(candidates)
                    and group_keys.issubset(
                        {
                            str(candidate.metadata[field_name])
                            for candidate in universe_candidates
                            if field_name in candidate.metadata
                        }
                    )
                )
            }
        )

    def current_image_matches_source_candidate(
        self,
        current_image: CellProfilerCurrentImage,
        candidate: "ParsedSourceCandidate",
    ) -> bool:
        """Return whether the payload metadata names a parsed source candidate."""
        source_paths = image_payload_metadata(current_image).source_image_path_tokens
        if not source_paths:
            return False
        candidate_paths = {
            self.cellprofiler_source_order_path(candidate.path),
            self.cellprofiler_source_order_path(candidate.resolved_path),
        }
        return any(
            self.cellprofiler_source_order_path(path) in candidate_paths
            for path in source_paths
        )

    def add_measurements(
        self,
        name: str,
        rows: MeasurementRowsInput,
        *,
        object_name: str | None = None,
        fields: tuple[FieldSpec, ...] = (),
        object_id_field: str | None = None,
        source_image_name: str | None = None,
        source_path: str | None = None,
        source_component_metadata: SourceComponentMetadata | None = None,
        source_image_provenance_planes: SourceImageProvenancePlanes | None = None,
    ) -> StoredRuntimeValue:
        validation_started_at = time.perf_counter()
        if object_name is not None:
            self.require_artifact_available(
                name=object_name,
                kind=ObjectLabelsArtifactType,
            )
        AdapterProfileLog.measurement_artifact(
            "adapter_measurement_subject_validation",
            time.perf_counter() - validation_started_at,
            artifact_name=name,
            object_name=object_name,
        )
        table_started_at = time.perf_counter()
        measurement_table = MeasurementTable(
            name=name,
            rows=rows,
            object_name=object_name,
            fields=fields,
            object_id_field=object_id_field,
            source_image_name=source_image_name,
            validated_runtime_schema=bool(fields),
            source_path=source_path,
            source_component_metadata=source_component_metadata,
            source_image_provenance_planes=(
                source_image_provenance_planes
                if source_image_provenance_planes is not None
                else SourceImageProvenancePlanes()
            ),
        )
        AdapterProfileLog.measurement_artifact(
            "adapter_measurement_table_construct",
            time.perf_counter() - table_started_at,
            artifact_name=name,
            object_name=object_name,
            fields_declared=bool(fields),
        )
        record_started_at = time.perf_counter()
        stored_value = self._record_native_value(
            name,
            MeasurementsArtifactType,
            measurement_table,
        )
        AdapterProfileLog.measurement_artifact(
            "adapter_measurement_record_native",
            time.perf_counter() - record_started_at,
            artifact_name=name,
            object_name=object_name,
        )
        return stored_value

    def get_measurements(
        self,
        name: str,
        *,
        group_key: str | None = None,
        current_image: CellProfilerCurrentImage | None = None,
    ) -> MeasurementTable:
        records = RuntimeArtifactRecordResolver(
            adapter=self,
            name=name,
            artifact_type=MeasurementsArtifactType,
            group_key=group_key,
            current_image=current_image,
        ).resolve()
        return MeasurementTableUnion(
            name,
            tuple(MeasurementTable.from_runtime_value(record.value) for record in records),
        ).as_table()

    def apply_measurement_table_cache_mutation(
        self,
        table: MeasurementTable,
    ) -> None:
        """Apply registered cache policies for one measurement-table write."""
        semantics_started_at = time.perf_counter()
        table_semantics = (
            MeasurementTableObjectFeatureSemantics.from_table(table)
        )
        AdapterProfileLog.measurement_cache(
            "adapter_measurement_table_semantics",
            time.perf_counter() - semantics_started_at,
            object_count=len(table_semantics.object_names),
            feature_count=len(table_semantics.feature_names),
        )
        mutation = MeasurementTableCacheMutation(
            adapter=self,
            table=table,
            object_names=table_semantics.object_names,
            feature_names=table_semantics.feature_names,
        )
        for policy in MeasurementTableCacheMutationPolicy.registered_policies():
            policy_started_at = time.perf_counter()
            policy.apply(mutation)
            AdapterProfileLog.measurement_cache_policy(
                time.perf_counter() - policy_started_at,
                policy_name=type(policy).__name__,
                object_count=len(table_semantics.object_names),
                feature_count=len(table_semantics.feature_names),
            )

    def measurement_tables(
        self,
        *,
        group_key: str | None = None,
        match_group: bool = True,
        current_image: CellProfilerCurrentImage | None = None,
    ) -> tuple[MeasurementTable, ...]:
        """Return measurement tables visible to the current runtime scope."""
        runtime_scope = RuntimeGroupMatchScope(
            group_key=group_key,
            match_group=match_group,
        ).runtime_scope(self, current_image=current_image)
        cache_key = (
            "all",
            self.runtime_value_store.revision,
            runtime_scope.group_cache_component,
            match_group,
            runtime_scope.current_image_cache_component,
        )
        cached = self._measurement_cache.get(cache_key)
        if cached is not None:
            return cached
        query_context = runtime_scope.artifact_query_context()
        records = self.declared_measurement_input_records(
            group_key=group_key,
            match_group=match_group,
            current_image=current_image,
        )
        if not records:
            records = query_context.find(artifact_type=MeasurementsArtifactType)
        records = RuntimeRecordSourceImageSetSelector(
            self,
            current_image,
        ).select_runtime_scope(records)
        tables = tuple(
            MeasurementTable.from_runtime_value(record.value)
            for record in records
        )
        if not records:
            tables = runtime_measurement_tables(
                query_context
            )
        self._measurement_cache[cache_key] = tables
        return tables

    def declared_measurement_input_records(
        self,
        *,
        group_key: str | None = None,
        match_group: bool = True,
        current_image: CellProfilerCurrentImage | None = None,
    ) -> tuple[StoredRuntimeValue, ...]:
        """Return records for measurement artifacts declared by the runtime step."""
        records: list[StoredRuntimeValue] = []
        for input_plan in self.artifact_inputs.values():
            if input_plan.artifact_type is not MeasurementsArtifactType:
                continue
            records.extend(
                RuntimeArtifactRecordResolver(
                    adapter=self,
                    name=input_plan.name,
                    artifact_type=MeasurementsArtifactType,
                    group_key=group_key,
                    match_group=match_group,
                    current_image=current_image,
                ).resolve()
            )
        return tuple({id(record): record for record in records}.values())

    def add_relationship(
        self,
        name: str,
        *,
        parent_object_name: str,
        child_object_name: str,
        parent_ids: RelationshipIdVector,
        child_ids: RelationshipIdVector,
        slice_indices: tuple[int, ...] = (),
        slice_count: int | None = None,
        source_path: str | None = None,
        source_component_metadata: SourceComponentMetadata | None = None,
        source_image_provenance_planes: SourceImageProvenancePlanes | None = None,
    ) -> StoredRuntimeValue:
        if not self._declared_output_resolution(
            name,
            RelationshipsArtifactType,
        ).is_declared:
            self.require_artifact_available(
                name=parent_object_name,
                kind=ObjectLabelsArtifactType,
            )
            self.require_artifact_available(
                name=child_object_name,
                kind=ObjectLabelsArtifactType,
            )
        semantics = RelationshipSemantics.parent_child(
            parent_object_name,
            child_object_name,
        )
        return self._record_native_value(
            name,
            RelationshipsArtifactType,
            ObjectRelationship(
                name=name,
                source=semantics.source,
                target=semantics.target,
                source_ids=parent_ids,
                target_ids=child_ids,
                relationship_type=semantics.relationship_type,
                slice_indices=slice_indices,
                slice_count=slice_count,
                source_path=source_path,
                source_component_metadata=source_component_metadata,
                source_image_provenance_planes=(
                    source_image_provenance_planes
                    if source_image_provenance_planes is not None
                    else SourceImageProvenancePlanes()
                ),
            ),
        )

    def get_relationship(
        self,
        name: str,
        *,
        group_key: str | None = None,
        current_image: CellProfilerCurrentImage | None = None,
    ) -> ObjectRelationship:
        records = RuntimeArtifactRecordResolver(
            adapter=self,
            name=name,
            artifact_type=RelationshipsArtifactType,
            group_key=group_key,
            current_image=current_image,
        ).resolve()
        if len(records) == 1:
            return ObjectRelationship.from_runtime_value(records[0].value)
        query_context = RuntimeGroupMatchScope(
            group_key=group_key
        ).runtime_scope(self).artifact_query_context()
        return runtime_relationship(
            query_context,
            name=name,
        )

    def add_spatial_grid(
        self,
        name: str,
        grid: SpatialGridInput,
    ) -> StoredRuntimeValue:
        return self._record_native_value(
            name,
            SpatialGridArtifactType,
            SpatialGridValueAuthority.input_value(name, grid),
        )

    def get_spatial_grid(
        self,
        name: str,
        *,
        group_key: str | None = None,
    ) -> SpatialGrid | RuntimeSliceAlignedValues[SpatialGrid]:
        records = RuntimeArtifactRecordResolver(
            adapter=self,
            name=name,
            artifact_type=SpatialGridArtifactType,
            group_key=group_key,
            current_image=None,
        ).resolve()
        grids = tuple(
            SpatialGridValueAuthority.record_value(name, record)
            for record in records
        )
        return SpatialGridValueAuthority.single_spatial_grid(name, grids)

    def _record_native_value(
        self,
        name: str,
        expected_kind: ArtifactType,
        native_value: RuntimeArtifactPayloadValue,
    ) -> StoredRuntimeValue:
        total_started_at = time.perf_counter()
        profile = NativeRecordProfileContext(name, expected_kind)
        plan_started_at = time.perf_counter()
        output_plan = self._require_output_plan(name, expected_kind)
        slice_count = RuntimeSliceProjection.slice_count_from_values((native_value,))
        scoped_group_key = self.runtime_record_scoped_group_key(
            output_plan,
            native_value,
        )
        output_group_keys = output_plan.runtime_record_group_keys(
            requested_group_key=self.group_key,
            scoped_group_key=scoped_group_key,
            slice_count=slice_count,
        )
        project_runtime_slices = (
            slice_count is not None
            and slice_count > 1
            and len(output_group_keys) == slice_count
        )
        profile.event(
            "adapter_require_output_plan",
            time.perf_counter() - plan_started_at,
        )
        store_started_at = time.perf_counter()
        stored_value: StoredRuntimeValue | None = None
        for slice_index, output_group_key in enumerate(output_group_keys):
            plan = output_plan.for_group(output_group_key)
            group_native_value = (
                RuntimeSliceProjection.value_for_slice(
                    native_value,
                    RuntimeProjectionAxis(
                        slice_index=slice_index,
                        extent=slice_count,
                    ),
                )
                if project_runtime_slices
                else native_value
            )
            normalize_started_at = time.perf_counter()
            runtime_value = RuntimeGroupMatchScope(group_key=None).runtime_scope(
                self
            ).normalize_artifact_value(
                plan,
                group_native_value,
            )
            profile.normalized_value(
                time.perf_counter() - normalize_started_at,
                payload_type=type(runtime_value.data).__name__,
                group_key=output_group_key,
            )
            save_started_at = time.perf_counter()
            runtime_path = plan.path
            self._save_payload(runtime_value.data, runtime_path)
            profile.group_event(
                "adapter_save_payload",
                time.perf_counter() - save_started_at,
                output_group_key,
            )
            replace_started_at = time.perf_counter()
            stored_value = self.runtime_value_store.replace(
                runtime_value,
                path=runtime_path,
                backend=self.backend,
            )
            profile.group_event(
                "adapter_runtime_store_replace_only",
                time.perf_counter() - replace_started_at,
                output_group_key,
            )
            if expected_kind is MeasurementsArtifactType:
                table_started_at = time.perf_counter()
                measurement_table = MeasurementTable.from_runtime_value(runtime_value)
                profile.group_event(
                    "adapter_measurement_table_from_runtime_value",
                    time.perf_counter() - table_started_at,
                    output_group_key,
                )
                cache_mutation_started_at = time.perf_counter()
                self.apply_measurement_table_cache_mutation(measurement_table)
                profile.group_event(
                    "adapter_measurement_cache_mutation",
                    time.perf_counter() - cache_mutation_started_at,
                    output_group_key,
                )
        if stored_value is None:
            raise RuntimeError(
                f"No runtime artifact groups were selected for '{name}' "
                f"({expected_kind.value}); "
                f"{self._runtime_record_failure_context(output_plan, native_value, scoped_group_key, slice_count)}"
            )
        invalidation_started_at = time.perf_counter()
        self.invalidate_runtime_query_caches_for_artifact_type(expected_kind)
        profile.event(
            "adapter_runtime_query_cache_invalidation",
            time.perf_counter() - invalidation_started_at,
        )
        profile.event(
            "adapter_runtime_store_replace",
            time.perf_counter() - store_started_at,
        )
        profile.event(
            "adapter_record_native_value",
            time.perf_counter() - total_started_at,
        )
        return stored_value

    def _runtime_record_failure_context(
        self,
        output_plan: ArtifactOutputPlan,
        native_value: RuntimeArtifactPayloadValue,
        scoped_group_key: str | None,
        slice_count: int | None,
    ) -> str:
        """Return generic runtime-record diagnostics when no output group matches."""
        payload = (
            native_value.runtime_payload()
            if isinstance(native_value, NativeRuntimeValue)
            else native_value
        )
        try:
            metadata = image_payload_metadata(payload)
            source_paths = metadata.source_image_paths
            source_component_metadata = metadata.source_component_metadata
            source_plane_component_metadata = (
                metadata.source_image_provenance_planes.component_metadata
            )
        except Exception:
            source_paths = ()
            source_component_metadata = None
            source_plane_component_metadata = ()
        try:
            payload_shape = tuple(np.shape(payload))
        except Exception:
            payload_shape = ()
        return (
            f"requested_group_key={self.group_key!r}, "
            f"scoped_group_key={scoped_group_key!r}, "
            f"axis_group_key={self.axis_scope.value_text_for_component(output_plan.group_component)!r}, "
            f"output_group_keys={tuple(output_plan.group_keys or (None,))!r}, "
            f"slice_count={slice_count!r}, "
            f"native_value_type={type(native_value).__name__}, "
            f"payload_type={type(payload).__name__}, "
            f"payload_shape={payload_shape!r}, "
            f"source_paths={source_paths!r}, "
            f"source_component_metadata={source_component_metadata!r}, "
            f"source_plane_component_metadata={source_plane_component_metadata!r}."
        )

    def runtime_record_scoped_group_key(
        self,
        output_plan: ArtifactOutputPlan,
        native_value: RuntimeArtifactPayloadValue,
    ) -> str | None:
        """Return the group key that should scope recording for this output."""
        axis_group_key = self.axis_scope.value_text_for_component(
            output_plan.group_component
        )
        group_keys = frozenset(
            str(group_key)
            for group_key in (output_plan.group_keys or ())
            if group_key is not None
        )
        if not group_keys:
            return axis_group_key
        source_payload = (
            native_value.runtime_payload()
            if isinstance(native_value, NativeRuntimeValue)
            else native_value
        )
        source_group_key = self.runtime_group_key_from_source_payload(
            group_keys,
            source_payload,
            group_component=output_plan.group_component,
        )
        if source_group_key is not None:
            return source_group_key
        return axis_group_key

    def clear_runtime_query_caches(self) -> None:
        """Clear every runtime query cache owned by this adapter."""
        self._image_cache.clear()
        self._current_image_cache.clear()
        self._object_cache.clear()
        self._current_object_cache.clear()
        self._measurement_cache.clear()
        object_label_measurement_values_cache(self.runtime_value_store).clear()
        object_measurement_table_cache(self.runtime_value_store).clear()
        object_measurement_table_index_cache(self.runtime_value_store).clear()
        self._source_paths_by_image_name_cache.clear()
        self._artifact_availability_cache.clear()

    def clear_measurement_query_cache(self) -> None:
        """Clear adapter-local measurement queries after measurement writes.

        Process-wide object/feature measurement caches are mutated by
        MeasurementTableCacheMutationPolicy implementations so unrelated
        feature indexes survive derived-measurement writes.
        """
        self._measurement_cache.clear()

    def _require_output_plan(
        self,
        name: str,
        expected_kind: ArtifactType,
    ) -> ArtifactOutputPlan:
        plan = self.artifact_outputs.get(name)
        if plan is None:
            raise RuntimeError(
                f"No compiled output plan for CellProfiler artifact '{name}' "
                f"({expected_kind.value})."
            )
        if plan.artifact_type is not expected_kind:
            raise ValueError(
                f"CellProfiler artifact '{name}' expected output kind "
                f"{expected_kind.value}, got compiled kind {plan.artifact_type.value}."
            )
        return plan

    def _declared_output_resolution(
        self,
        name: str,
        kind: ArtifactType,
    ) -> DeclaredOutputResolution:
        """Return a declared output plan, or absence, without masking invalid plans."""
        if name not in self.artifact_outputs:
            return DeclaredOutputResolution()
        return DeclaredOutputResolution(self._require_output_plan(name, kind))

    @property
    def can_resolve_source_candidates(self) -> bool:
        """Return whether parser-backed source candidate resolution is available."""
        return self.processing_context is not None

    def _save_payload(self, data: RuntimeArtifactPayloadValue, path: str) -> None:
        if self.filemanager is None:
            raise RuntimeError(
                "CellProfilerRuntimeAdapter.filemanager is required for writes; "
                "adapter writes must persist through the OpenHCS VFS boundary."
            )
        replace_runtime_artifact_payload(
            self.filemanager,
            data,
            RuntimeArtifactLocation(path=path, backend=self.backend),
        )

    def source_candidates(
        self,
        file_paths: tuple[str, ...],
    ) -> tuple["ParsedSourceCandidate", ...]:
        """Return parsed source candidates for this runtime source universe.

        Source resolution may query the same step-input and pipeline-start
        universes from separate CellProfiler runtime adapters. Parsing is pure
        for the path tuple, source-binding context, metadata rules and filename
        parser, so the cache key carries those semantic inputs explicitly.
        """
        cached = self._source_candidate_cache.get(file_paths)
        if cached is not None:
            return cached
        candidates = SourceCandidateRuntimeCache(self, file_paths).candidates()
        self._source_candidate_cache[file_paths] = candidates
        return candidates

    def source_candidate_process_cache_key(
        self,
        file_paths: tuple[str, ...],
    ) -> tuple[Hashable, ...]:
        """Return the semantic cache key for one source-candidate universe."""
        file_paths = tuple(file_paths)
        cached = self._source_candidate_process_key_cache.get(file_paths)
        if cached is not None:
            return cached
        cache_key = source_candidate_process_cache_key(
            file_paths=file_paths,
            context=self.source_binding_context,
            plan=self.source_binding_plan,
            parser=RequireProcessingContextBoundaryPolicy(
                self
            ).context.microscope_handler.parser,
        )
        self._source_candidate_process_key_cache[file_paths] = cache_key
        return cache_key

    def prepare_source_resolution(self) -> None:
        """Prepare source-resolution caches owned by this adapter's source context."""
        for file_paths in self.source_binding_context.source_candidate_file_universes():
            self.source_candidates(file_paths)
        self.cellprofiler_ordered_pipeline_image_paths()
        self.cellprofiler_axis_image_number_start()
        if self.can_resolve_source_candidates:
            CellProfilerImageNumberResolver.for_adapter(self).image_number_map()
