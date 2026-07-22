"""Runtime execution helpers for FunctionStep.

This module owns callable invocation, artifact routing, and pattern-group stack
execution. FunctionStep remains responsible for step-level orchestration.
"""

from abc import ABC, abstractmethod
from functools import singledispatch
import logging
import os
import time
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass, field, replace
from threading import Lock
from pathlib import Path
from types import MappingProxyType
from typing import (
    Callable,
    ClassVar,
    Generic,
    Iterator,
    Mapping,
    Sequence,
    TypeVar,
    cast,
)

from metaclass_registry import AutoRegisterMeta
import numpy as np

from openhcs.constants.constants import AllComponents, Backend, VariableComponents
from openhcs.core.artifacts import (
    ArtifactInputPlan,
    ArtifactOutputPlan,
    NoMainFlowOutput,
    ArtifactType,
    ArtifactTypeStrategyMatchMixin,
    SpecialArtifactType,
    ImageArtifactType,
    MeasurementsArtifactType,
    ObjectLabelsArtifactType,
    SpatialGraphArtifactType,
    ArtifactSpecRef,
)
from openhcs.core.callable_contract import (
    CallableRuntimeCacheKey,
    prepare_processing_callable,
)
from openhcs.core.component_group_scope import (
    ComponentGroupScope,
    RuntimeExecutionAxisScope,
    RuntimeFixedComponentValues,
)
from openhcs.core.component_set import ComponentSet
from openhcs.core.context.processing_context import ProcessingContext
from openhcs.core.debug import (
    DebugCursor,
    DebugEvent,
    DebugEventSink,
    DebugEventType,
    DebugArtifactRefProjection,
    DebugInvocationParameter,
    debug_event_sink_from_context,
)
from openhcs.core.function_patterns import (
    CompiledFunctionGroup,
    CompiledFunctionInvocation,
    InvocationArtifactInputEdgePlan,
    InvocationArtifactInputProjectionKey,
    MainFlowInputProjection,
    RuntimeComponentValue,
    RuntimeInvocationDomain,
)
from openhcs.core.aligned_image_payload import (
    AlignedImageStack,
    AlignedImageSliceContext,
    ImagePayloadBundleContext,
    ImageOutputBundle,
    flatten_aligned_image_slice_contexts,
    flatten_aligned_image_payload_slices,
    stack_image_payload_context,
    stack_image_payload_context_from_metadata,
    unstack_image_payload_context,
)
from openhcs.core.memory import (
    convert_memory,
    stack_runtime_slices,
    unstack_runtime_slices,
)
from openhcs.core.process_local_cache import IdentityBoundProcessCache
from openhcs.core.runtime_stores import (
    RuntimeArtifactInput,
    RuntimeArtifactLocation,
    replace_runtime_artifact_payload,
)
from openhcs.core.runtime_artifact_values import RuntimeValue
from openhcs.core.runtime_output_matching import (
    RuntimeReturnedOutputMatcher,
)
from openhcs.core.runtime_adapters import (
    RuntimeAdapterRequest,
)
from openhcs.core.runtime_slice_alignment import (
    RuntimeSliceAlignedValueSet,
    RuntimeSliceAlignedValues,
)
from openhcs.core.runtime_slice_projection import (
    RuntimeSliceProjectionDeclarationError,
)
from openhcs.core.source_image_semantics import apply_source_binding_payload
from openhcs.core.source_image_provenance import SourceImageIdentity
from openhcs.core.source_workspace_projection import (
    VirtualWorkspacePathLookup,
    VirtualWorkspaceSourceProjection,
    VirtualWorkspaceSourceProjectionAuthority,
    VirtualWorkspaceSourceProjectionCache,
)
from openhcs.core.source_binding_selection import (
    SourceBindingCandidateMatcher,
    SourceBindingMatchedImageSet,
    SourceBindingRuntimeContextRequest,
    SourcePatternResolutionContext,
)
from openhcs.core.source_matching import (
    source_component_metadata_value,
    source_metadata_value,
    with_source_component_metadata,
)
from openhcs.core.source_bindings import (
    CompiledSourceBindingPlan,
    SOURCE_BINDING_ALIAS_METADATA_FIELD,
    SourceBindingRuntimeContext,
    SourceProjectionRole,
)
from openhcs.core.runtime_image_values import (
    ImagePayloadMetadata,
    ImagePayloadMetadataCarrier,
    ImagePayloadMetadataCompositionMode,
    image_payload_data,
    image_payload_mask,
    image_payload_metadata,
    image_payload_slice_context,
    preserve_declared_image_payload_axis,
    with_image_payload_data,
)
from openhcs.core.runtime_image_loading import ImagePayloadSourceMetadataContext
from openhcs.core.runtime_array_values import RuntimeArrayData
from openhcs.core.runtime_measurements import MeasurementSubject, MeasurementTable
from openhcs.core.runtime_spatial_graph import SpatialGraph
from openhcs.core.runtime_object_labels import (
    ObjectLabelSet,
    ObjectLabelValue,
    object_label_dense_array,
)
from openhcs.core.runtime_object_label_building import (
    SourceImageObjectLabelBuildRequest,
)
from openhcs.core.runtime_tabular_values import ColumnarRows
from openhcs.core.registry_strategies import (
    MostDerivedContextStrategyMixin,
    NominalTypeKeyedStrategyMixin,
)
from openhcs.core.runtime_plane_projection import (
    RuntimePlaneAxis,
    RuntimePlaneAxisProjector,
    RuntimePlaneAxisValueProjection,
    RuntimePlaneProjection,
)
from openhcs.core.runtime_slice_projection import RuntimeSliceProjection
from openhcs.core.step_dependencies import StepInputDependencyKind
from openhcs.core.steps.function_output_manifest import (
    NoStepOutputManifestMatch,
    ProducedOutputSemantics,
    step_output_manifest,
)
from openhcs.core.steps.function_output_identity import (
    FunctionOutputIdentity,
    FunctionOutputIdentityAuthority,
    FunctionOutputPathAuthority,
    FunctionOutputPathRequest,
)
from openhcs.core.compiled_step_plan import CompiledStepPlan

logger = logging.getLogger(__name__)

_PROFILE_RUNTIME_ENV = "OPENHCS_PROFILE_FUNCTION_RUNTIME"
_PROFILE_RUNTIME_PATH_ENV = "OPENHCS_PROFILE_FUNCTION_RUNTIME_PATH"
ArtifactInputPlanKeyT = TypeVar(
    "ArtifactInputPlanKeyT",
    ArtifactSpecRef,
    InvocationArtifactInputProjectionKey,
)
ArtifactInputPlanT = TypeVar(
    "ArtifactInputPlanT",
    ArtifactInputPlan,
    InvocationArtifactInputEdgePlan,
)
ArtifactInputPlans = Mapping[ArtifactInputPlanKeyT, ArtifactInputPlanT]
ArtifactOutputPlans = Mapping[ArtifactSpecRef, ArtifactOutputPlan]
JsonValue = RuntimeComponentValue | Mapping[str, "JsonValue"] | Sequence["JsonValue"]
FunctionOutputContextualizedValue = (
    RuntimeArrayData
    | ObjectLabelValue
    | ColumnarRows
    | MeasurementTable
    | SpatialGraph
    | RuntimeSliceAlignedValueSet
)
ObjectLabelContextualizableOutput = (
    RuntimeArrayData | ObjectLabelValue | RuntimeSliceAlignedValueSet
)
RuntimePayload = FunctionOutputContextualizedValue
RuntimeFunctionOutput = RuntimePayload | NoMainFlowOutput | tuple[RuntimePayload, ...]
RuntimeCallableArgument = JsonValue | RuntimePayload | ProcessingContext
RuntimeCallableKwargs = Mapping[str, RuntimeCallableArgument]
RuntimeProfileFieldValue = str | int | float | bool | None
EMPTY_ARTIFACT_PLANS: ArtifactOutputPlans = MappingProxyType({})


class FunctionInvocationCallableCache(IdentityBoundProcessCache):
    """Bound resolved callables to their exact compiled contract owners."""

    registry_key = "function_invocation_callable"


class FunctionInvocationCallableResolver:
    """Process-local resolver for compiled invocation callables.

    The compiler stores picklable ``FunctionReference`` objects in compiled
    invocations. Runtime execution needs actual callables. Resolving them during
    compiler preparation lets fork workers inherit the resolved callable cache,
    while spawn workers still resolve lazily in their own process.
    """

    _lock = Lock()

    @classmethod
    def prepare(cls, invocation: CompiledFunctionInvocation) -> None:
        """Resolve and cache one invocation callable before timed execution."""
        cls.resolve(invocation)
        prepare_processing_callable(
            invocation.contract.resolve_canonical_raw_callable()
        )

    @classmethod
    def resolve(cls, invocation: CompiledFunctionInvocation) -> Callable:
        """Return the callable for a compiled invocation."""
        cache = FunctionInvocationCallableCache.process_cache()
        with cls._lock:
            cached = cache.get_bound(invocation.contract)
        if cached is not None:
            return cached

        resolved = invocation.contract.resolve_runtime_callable()

        with cls._lock:
            cache.put_bound(invocation.contract, resolved)
        return resolved

    @classmethod
    def cache_key(
        cls,
        invocation: CompiledFunctionInvocation,
    ) -> CallableRuntimeCacheKey:
        """Return process-local callable cache key for one compiled invocation."""
        return invocation.contract.runtime_callable_cache_identity()


class RuntimeProfileSink:
    """Runtime-profile output authority backed by explicit environment settings."""

    @classmethod
    def enabled(cls) -> bool:
        raw_value = cls.environment_value(_PROFILE_RUNTIME_ENV)
        if raw_value is None:
            return False
        return raw_value.lower() in {"1", "true", "yes"}

    @staticmethod
    def environment_value(name: str) -> str | None:
        if name not in os.environ:
            return None
        return os.environ[name]

    @classmethod
    def record(
        cls,
        label: str,
        seconds: float,
        **fields: RuntimeProfileFieldValue,
    ) -> None:
        if not cls.enabled():
            return
        field_text = " ".join(f"{key}={value}" for key, value in fields.items())
        logger.info("RUNTIME_PROFILE %s %.6fs %s", label, seconds, field_text)
        profile_path = cls.environment_value(_PROFILE_RUNTIME_PATH_ENV)
        if profile_path is not None:
            with open(profile_path, "a", encoding="utf-8") as handle:
                handle.write(f"RUNTIME_PROFILE {label} {seconds:.6f}s {field_text}\n")


class FunctionOutputContextStrategy(
    ArtifactTypeStrategyMatchMixin,
    MostDerivedContextStrategyMixin[type[ArtifactType]],
    ABC,
):
    """Registered normalization for function outputs before chaining or storage."""

    artifact_type: ClassVar[type[ArtifactType] | None] = None

    @classmethod
    def for_output_plan(
        cls,
        output_plan: ArtifactOutputPlan | None,
    ) -> "FunctionOutputContextStrategy":
        output_kind = (
            ImageArtifactType if output_plan is None else output_plan.artifact_type
        )
        strategy = cls.for_context(output_kind, required=False)
        return (
            strategy
            if strategy is not None
            else UnchangedFunctionOutputContextStrategy()
        )

    @abstractmethod
    def contextualize(
        self,
        source_payload: RuntimePayload,
        output_value: RuntimePayload,
        output_plan: ArtifactOutputPlan | None,
        plane_projection: RuntimePlaneAxisValueProjection | None,
    ) -> FunctionOutputContextualizedValue:
        """Return output with source context preserved where semantics allow it."""

    def contextualize_from_projector(
        self,
        source_payload: RuntimePayload,
        output_value: RuntimePayload,
        output_plan: ArtifactOutputPlan | None,
        plane_projector: RuntimePlaneAxisProjector | None,
    ) -> FunctionOutputContextualizedValue:
        """Resolve plane projection only after the artifact strategy is selected."""

        plane_projection = (
            None
            if plane_projector is None
            else preserve_declared_image_payload_axis(
                plane_projector,
                output_value,
                source_payload=source_payload,
            )
        )
        if (
            plane_projection is None
            and output_plan is not None
            and output_plan.variable_components
        ):
            if plane_projector is None:
                if not self.output_owns_source_context(
                    source_payload,
                    output_value,
                    output_plan,
                    None,
                ):
                    raise ValueError(
                        f"Artifact output {output_plan.ref()!r} preserves variable "
                        f"components {output_plan.variable_components!r} but the "
                        "runtime invocation supplies no plane projector."
                    )
                return output_value
            plane_projection = (
                RuntimePlaneAxisValueProjection.require_from_projector(
                    plane_projector,
                    RuntimePlaneAxis.RUNTIME_SLICE,
                )
            )
        return self.contextualize(
            source_payload,
            output_value,
            output_plan,
            plane_projection,
        )

    @staticmethod
    def output_owns_source_context(
        source_payload: RuntimePayload,
        output_value: RuntimePayload,
        output_plan: ArtifactOutputPlan | None,
        plane_projection: RuntimePlaneAxisValueProjection | None,
    ) -> bool:
        """Return whether this artifact value already owns complete context."""

        del source_payload, output_value, output_plan, plane_projection
        return False


class UnchangedFunctionOutputContextStrategy(FunctionOutputContextStrategy):
    """Leave outputs unchanged when no contextual image semantics are declared."""

    artifact_type = SpecialArtifactType

    def contextualize(
        self,
        source_payload: RuntimePayload,
        output_value: RuntimePayload,
        output_plan: ArtifactOutputPlan | None,
        plane_projection: RuntimePlaneAxisValueProjection | None,
    ) -> FunctionOutputContextualizedValue:
        del source_payload, output_plan, plane_projection
        return output_value


class ImageFunctionOutputContextStrategy(FunctionOutputContextStrategy):
    """Preserve source-image metadata for image outputs derived from the main input."""

    artifact_type = ImageArtifactType

    @staticmethod
    def output_owns_source_context(
        source_payload: RuntimePayload,
        output_value: RuntimePayload,
        output_plan: ArtifactOutputPlan | None,
        plane_projection: RuntimePlaneAxisValueProjection | None,
    ) -> bool:
        """Return whether an image result already carries complete source identity."""

        if not isinstance(output_value, AlignedImageStack) and not (
            image_payload_metadata(output_value).has_complete_source_identity(
                output_value,
                plane_projection,
            )
        ):
            return False
        output_surfaces = flatten_aligned_image_payload_slices(output_value)
        if not all(
            image_payload_metadata(output_surface).has_complete_source_identity(
                output_surface
            )
            for output_surface in output_surfaces
        ):
            return False
        if output_plan is None:
            return True
        source_surfaces = flatten_aligned_image_payload_slices(source_payload)
        return len(output_surfaces) == len(source_surfaces) and all(
            image_payload_metadata(
                output_surface
            ).source_provenance.represented_source_identities
            == image_payload_metadata(
                source_surface
            ).source_provenance.represented_source_identities
            for source_surface, output_surface in zip(
                source_surfaces,
                output_surfaces,
                strict=True,
            )
        )

    def contextualize(
        self,
        source_payload: RuntimePayload,
        output_value: RuntimePayload,
        output_plan: ArtifactOutputPlan | None,
        plane_projection: RuntimePlaneAxisValueProjection | None,
    ) -> FunctionOutputContextualizedValue:
        if isinstance(output_value, RuntimeSliceAlignedValueSet):
            if (
                plane_projection is None
                or plane_projection.axis is not RuntimePlaneAxis.RUNTIME_SLICE
            ):
                raise ValueError(
                    "Runtime-slice-aligned image output requires an exact runtime "
                    "slice projection."
                )
            if plane_projection.axis_size != output_value.slice_count:
                raise ValueError(
                    "Runtime-slice-aligned image output count must exactly match "
                    "the declared runtime plane axis: "
                    f"{output_value.slice_count} != {plane_projection.axis_size}."
                )
            contextualized_slices = []
            for slice_index in range(output_value.slice_count):
                item = output_value.value_for_slice(slice_index)
                contextualized_slices.append(
                    self.contextualize(
                        RuntimeSliceProjection.value_for_slice(
                            source_payload,
                            RuntimePlaneAxisValueProjection.from_selected_plane(
                                axis=plane_projection.axis,
                                plane_index=slice_index,
                                axis_size=plane_projection.axis_size,
                            ),
                        ),
                        item.data if isinstance(item, RuntimeValue) else item,
                        output_plan,
                        None,
                    )
                )
            return RuntimeSliceAlignedValues(tuple(contextualized_slices))
        source_ref = (
            None if output_plan is None else output_plan.source_context_source()
        )
        if (
            output_plan is not None
            and output_plan.variable_components
            and plane_projection is not None
            and plane_projection.plane_index is None
        ):
            output_metadata = image_payload_metadata(output_value)
            if (
                output_metadata.plane_axis is None
                and output_metadata.source_provenance.source_plane_count
                == plane_projection.axis_size
                and plane_projection.dense_shape_carries_axis(
                    np.shape(image_payload_data(output_value))
                )
            ):
                output_value = output_metadata.replace_fields(
                    plane_axis=plane_projection.axis,
                ).attach_to(output_value)
        if output_plan is not None and not output_plan.variable_components:
            output_metadata = image_payload_metadata(output_value)
            if output_metadata.plane_axis is RuntimePlaneAxis.RUNTIME_SLICE:
                output_value = RuntimeSliceProjection.value_for_singleton_slice(
                    output_value,
                    source_description=f"Image output {output_plan.ref()!r}",
                )
            source_metadata = image_payload_metadata(source_payload)
            if source_metadata.plane_axis is RuntimePlaneAxis.RUNTIME_SLICE:
                if source_ref is None:
                    raise ValueError(
                        f"Image output {output_plan.ref()!r} consumes a runtime "
                        "stack without a declared source-context relation."
                    )
                collapsed_metadata = source_metadata.collapse_leading_plane_axis()
                source_payload = collapsed_metadata.with_source_provenance(
                    collapsed_metadata.source_provenance.with_source_image_names(
                        (source_ref.name,)
                    )
                ).attach_source_context_to(output_value)
            plane_projection = None
        source_context_strategy = ImageOutputSourceContextStrategy.for_source_payload(
            source_payload,
        )
        if self.output_owns_source_context(
            source_payload,
            output_value,
            output_plan,
            plane_projection,
        ) and not source_context_strategy.requires_plane_contextualization(
            source_payload,
            output_value,
            plane_projection,
        ):
            if isinstance(output_value, AlignedImageStack):
                return output_value
            output_metadata = image_payload_metadata(output_value)
            source_metadata = image_payload_metadata(source_payload)
            contextualized_output = output_metadata.with_source_context_from(
                source_metadata
            ).attach_source_context_to(
                output_value,
            )
            if image_payload_metadata(contextualized_output) == output_metadata:
                return output_value
            return contextualized_output
        return source_context_strategy.contextualize(
            source_payload,
            output_value,
            plane_projection,
        )


class MeasurementsFunctionOutputContextStrategy(FunctionOutputContextStrategy):
    """Own schema-bearing rows as one compiled measurement table."""

    artifact_type = MeasurementsArtifactType

    @staticmethod
    def _declared_subject(output_plan: ArtifactOutputPlan | None) -> MeasurementSubject:
        if output_plan is None:
            raise ValueError("Measurement outputs require a compiled output plan.")
        subject = output_plan.measurement_subject()
        if subject is None:
            raise ValueError(
                f"Measurement output {output_plan.ref()!r} has no declared "
                "measurement subject relation."
            )
        return subject

    @staticmethod
    def _validate_nominal_table(
        output_value: MeasurementTable,
        output_plan: ArtifactOutputPlan,
        subject: MeasurementSubject,
    ) -> None:
        output_value.validate_artifact_name(output_plan.name)
        if output_value.subject != subject:
            raise ValueError(
                f"Measurement output {output_plan.ref()!r} declares subject "
                f"{subject!r}, but returned {output_value.subject!r}."
            )

    def contextualize_from_projector(
        self,
        source_payload: RuntimePayload,
        output_value: RuntimePayload,
        output_plan: ArtifactOutputPlan | None,
        plane_projector: RuntimePlaneAxisProjector | None,
    ) -> FunctionOutputContextualizedValue:
        """Validate nominal identity before generic plane projection inspects it."""

        subject = self._declared_subject(output_plan)
        if isinstance(output_value, MeasurementTable):
            assert output_plan is not None
            self._validate_nominal_table(output_value, output_plan, subject)
        return super().contextualize_from_projector(
            source_payload,
            output_value,
            output_plan,
            plane_projector,
        )

    def contextualize(
        self,
        source_payload: RuntimePayload,
        output_value: RuntimePayload,
        output_plan: ArtifactOutputPlan | None,
        plane_projection: RuntimePlaneAxisValueProjection | None,
    ) -> FunctionOutputContextualizedValue:
        del plane_projection
        subject = self._declared_subject(output_plan)
        assert output_plan is not None
        if isinstance(output_value, MeasurementTable):
            self._validate_nominal_table(output_value, output_plan, subject)
            return output_value
        if not isinstance(output_value, ColumnarRows):
            raise TypeError(
                f"Measurement output {output_plan.ref()!r} requires ColumnarRows "
                f"or MeasurementTable, got {type(output_value).__name__}."
            )
        return MeasurementTable(
            name=output_plan.name,
            rows=output_value,
            source_image_name=subject.source_image_name,
            subject=subject,
            source_provenance=image_payload_metadata(source_payload).source_provenance,
        )


class SpatialGraphFunctionOutputContextStrategy(FunctionOutputContextStrategy):
    """Preserve invocation source identity on spatial graph outputs."""

    artifact_type = SpatialGraphArtifactType

    def contextualize(
        self,
        source_payload: RuntimePayload,
        output_value: RuntimePayload,
        output_plan: ArtifactOutputPlan | None,
        plane_projection: RuntimePlaneAxisValueProjection | None,
    ) -> FunctionOutputContextualizedValue:
        del plane_projection
        if not isinstance(output_value, SpatialGraph):
            raise TypeError(
                "Spatial graph output requires SpatialGraph, got "
                f"{type(output_value).__name__}."
            )
        if output_plan is not None:
            output_value.validate_artifact_name(output_plan.name)
        source_provenance = output_value.contextualized_source_provenance(
            image_payload_metadata(source_payload).source_provenance
        )
        contextualized_provenance = output_value.source_provenance.with_missing_from(
            source_provenance
        )
        if contextualized_provenance == output_value.source_provenance:
            return output_value
        return replace(output_value, source_provenance=contextualized_provenance)


@singledispatch
def project_declared_source_identity(
    source_payload: RuntimePayload,
    source_ref: ArtifactSpecRef,
) -> RuntimePayload:
    """Project an image payload to one exact declared source identity."""

    metadata = image_payload_metadata(source_payload)
    return metadata.project_declared_source_image(source_payload, source_ref.name)


@project_declared_source_identity.register(RuntimeSliceAlignedValueSet)
def project_aligned_declared_source_identity(
    source_payload: RuntimeSliceAlignedValueSet,
    source_ref: ArtifactSpecRef,
) -> RuntimeSliceAlignedValues:
    """Project each runtime-aligned image slice to the declared source identity."""

    return RuntimeSliceAlignedValues(
        tuple(
            project_declared_source_identity(
                source_payload.value_for_slice(slice_index),
                source_ref,
            )
            for slice_index in range(source_payload.slice_count)
        )
    )


@project_declared_source_identity.register(ObjectLabelValue)
def project_object_label_declared_source_identity(
    source_payload: ObjectLabelValue,
    source_ref: ArtifactSpecRef,
) -> ObjectLabelValue:
    """Preserve object-label context without applying image-axis projection."""

    if (
        isinstance(source_payload, ObjectLabelSet)
        and source_payload.name != source_ref.name
    ):
        raise ValueError(
            f"Object-label payload {source_payload.name!r} cannot resolve declared "
            f"source {source_ref!r}."
        )
    return source_payload


class ImageOutputSourceContextStrategy(
    NominalTypeKeyedStrategyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Registered image-output contextualization by semantic source payload type."""

    __registry_key__ = "value_type_label"
    __skip_if_no_key__ = True

    @classmethod
    def for_source_payload(
        cls,
        source_payload: RuntimePayload,
    ) -> "ImageOutputSourceContextStrategy":
        strategy = cls.for_nominal_value(source_payload)
        if strategy is None:
            return DefaultImageOutputSourceContextStrategy()
        return strategy

    def requires_plane_contextualization(
        self,
        source_payload: RuntimePayload,
        output_value: RuntimePayload,
        plane_projection: RuntimePlaneAxisValueProjection | None,
    ) -> bool:
        """Return whether this source type must bind an undeclared output axis."""

        del source_payload, output_value, plane_projection
        return False

    @abstractmethod
    def contextualize(
        self,
        source_payload: RuntimePayload,
        output_value: RuntimePayload,
        plane_projection: RuntimePlaneAxisValueProjection | None,
    ) -> RuntimePayload:
        """Return image output with source semantics attached."""


class DefaultImageOutputSourceContextStrategy(ImageOutputSourceContextStrategy):
    """Attach scalar source-image context to a derived image output."""

    def contextualize(
        self,
        source_payload: RuntimePayload,
        output_value: RuntimePayload,
        plane_projection: RuntimePlaneAxisValueProjection | None,
    ) -> RuntimePayload:
        if isinstance(output_value, AlignedImageStack):
            return output_value
        return image_payload_metadata(source_payload).derive_payload(
            source_payload,
            output_value,
            plane_projection=plane_projection,
        )


class AlignedImageStackOutputSourceContextStrategy(ImageOutputSourceContextStrategy):
    """Preserve aligned multi-source image payloads as their own source context."""

    value_type = AlignedImageStack

    def contextualize(
        self,
        source_payload: RuntimePayload,
        output_value: RuntimePayload,
        plane_projection: RuntimePlaneAxisValueProjection | None,
    ) -> RuntimePayload:
        del source_payload, plane_projection
        return output_value


class RuntimeSliceAlignedImageOutputSourceContextStrategy(
    ImageOutputSourceContextStrategy
):
    """Attach per-runtime-slice source context to derived image outputs."""

    value_type = RuntimeSliceAlignedValueSet

    def contextualize(
        self,
        source_payload: RuntimePayload,
        output_value: RuntimePayload,
        plane_projection: RuntimePlaneAxisValueProjection | None,
    ) -> RuntimePayload:
        source_values = source_payload
        if not isinstance(source_values, RuntimeSliceAlignedValueSet):
            raise TypeError(
                "Runtime-slice-aligned image output strategy requires "
                f"RuntimeSliceAlignedValueSet, got {type(source_values).__name__}."
            )
        return RuntimeSliceAlignedImageOutputContext(
            source_values=source_values,
            output_value=output_value,
            plane_projection=plane_projection,
        ).payload()


class ObjectLabelImageOutputSourceContextStrategy(
    ImageOutputSourceContextStrategy
):
    """Project an image rendered from labels onto the invocation plane axis."""

    value_type = ObjectLabelValue

    def requires_plane_contextualization(
        self,
        source_payload: RuntimePayload,
        output_value: RuntimePayload,
        plane_projection: RuntimePlaneAxisValueProjection | None,
    ) -> bool:
        """Require projection for a volume label payload, including depth one."""

        del output_value
        if not isinstance(source_payload, ObjectLabelValue):
            raise TypeError(
                "Object-label image output context requires ObjectLabelValue, got "
                f"{type(source_payload).__name__}."
            )
        return (
            plane_projection is not None
            and plane_projection.plane_index is None
            and image_payload_metadata(source_payload).plane_axis is None
            and np.ndim(object_label_dense_array(source_payload)) >= 3
        )

    def contextualize(
        self,
        source_payload: RuntimePayload,
        output_value: RuntimePayload,
        plane_projection: RuntimePlaneAxisValueProjection | None,
    ) -> RuntimePayload:
        if not isinstance(source_payload, ObjectLabelValue):
            raise TypeError(
                "Object-label image output context requires ObjectLabelValue, got "
                f"{type(source_payload).__name__}."
            )
        source_metadata = image_payload_metadata(source_payload)
        if plane_projection is None or plane_projection.plane_index is not None:
            return source_metadata.derive_payload(
                source_payload,
                output_value,
                plane_projection=plane_projection,
            )
        plane_count = source_metadata.source_provenance.source_plane_count
        if plane_count != plane_projection.axis_size:
            raise ValueError(
                "Object-label image output source-plane provenance must match the "
                "declared runtime plane axis: "
                f"{plane_count} != {plane_projection.axis_size}."
            )
        plane_projection.validate_shape(
            np.shape(image_payload_data(output_value)),
            value_name="Object-label image output payload",
        )
        output_metadata = image_payload_metadata(output_value)
        contextualized_output = output_metadata.replace_fields(
            plane_axis=plane_projection.axis,
        ).attach_to(output_value)
        return source_metadata.derive_payload(
            source_payload,
            contextualized_output,
            plane_projection=plane_projection,
        )


@dataclass(frozen=True, slots=True)
class RuntimeSliceAlignedImageOutputContext:
    """Compose stack metadata for image output derived from aligned source values."""

    source_values: RuntimeSliceAlignedValueSet
    output_value: RuntimePayload
    plane_projection: RuntimePlaneAxisValueProjection | None

    def payload(self) -> RuntimePayload:
        output_data = image_payload_data(self.output_value)
        if self.plane_projection is None:
            if self.source_values.slice_count != 1:
                raise ValueError(
                    "Runtime-slice-aligned image output has multiple source values "
                    "but no declared runtime plane projection."
                )
            source_value = self.source_values.value_for_aligned_slice(0, 1)
            return image_payload_metadata(source_value).derive_payload(
                source_value,
                self.output_value,
            )
        output_slices = self.output_slices(output_data)
        contextualized_slices = []
        for slice_index, output_slice in enumerate(output_slices):
            source_value = self.source_values.value_for_aligned_slice(
                slice_index,
                len(output_slices),
            )
            contextualized_slices.append(
                image_payload_metadata(source_value).derive_payload(
                    source_value,
                    output_slice,
                )
            )
        return stack_image_payload_context(
            tuple(contextualized_slices),
            output_data,
            metadata_mode=ImagePayloadMetadataCompositionMode.STACK,
        )

    def output_slices(
        self,
        output_data: RuntimeArrayData,
    ) -> tuple[RuntimeArrayData, ...]:
        output_array = np.asarray(output_data)
        projection = self.plane_projection
        if projection is None:
            raise RuntimeError(
                "Runtime-slice output splitting requires a plane projection."
            )
        if projection.axis_size != self.source_values.slice_count:
            raise ValueError(
                "Runtime-slice image output projection must exactly match its "
                f"aligned source count: {projection.axis_size} != "
                f"{self.source_values.slice_count}."
            )
        projection.validate_shape(
            output_array.shape,
            value_name="Runtime-slice-aligned image output",
        )
        return tuple(
            image_payload_slice_context(
                self.output_value,
                output_array[slice_index],
                slice_index,
                plane_axis=projection.axis,
            )
            for slice_index in range(projection.axis_size)
        )


class ObjectLabelsFunctionOutputContextStrategy(FunctionOutputContextStrategy):
    """Preserve source-image metadata for object-label outputs."""

    artifact_type = ObjectLabelsArtifactType

    def contextualize(
        self,
        source_payload: RuntimePayload,
        output_value: RuntimePayload,
        output_plan: ArtifactOutputPlan | None,
        plane_projection: RuntimePlaneAxisValueProjection | None,
    ) -> FunctionOutputContextualizedValue:
        return ObjectLabelOutputValueContextStrategy.for_output_value(
            output_value,
        ).contextualize(source_payload, output_value, plane_projection)


class ObjectLabelOutputValueContextStrategy(
    NominalTypeKeyedStrategyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Registered object-label output contextualization by nominal value type."""

    __registry_key__ = "value_type_label"
    __skip_if_no_key__ = True

    @classmethod
    def for_output_value(
        cls,
        output_value: ObjectLabelContextualizableOutput,
    ) -> "ObjectLabelOutputValueContextStrategy":
        return cls.require_nominal_value(
            output_value,
            context="Object-label function output",
        )

    @abstractmethod
    def contextualize(
        self,
        source_payload: RuntimePayload,
        output_value: ObjectLabelContextualizableOutput,
        plane_projection: RuntimePlaneAxisValueProjection | None,
    ) -> FunctionOutputContextualizedValue:
        """Return the output with source-image context attached when possible."""


class RuntimeSliceAlignedObjectLabelOutputValueContextStrategy(
    ObjectLabelOutputValueContextStrategy
):
    """Contextualize each runtime-slice-aligned object-label output slice."""

    value_type = RuntimeSliceAlignedValueSet

    def contextualize(
        self,
        source_payload: RuntimePayload,
        output_value: ObjectLabelContextualizableOutput,
        plane_projection: RuntimePlaneAxisValueProjection | None,
    ) -> FunctionOutputContextualizedValue:
        aligned_values = output_value
        if not isinstance(aligned_values, RuntimeSliceAlignedValueSet):
            raise TypeError(
                "Runtime-slice-aligned object-label output strategy requires "
                f"RuntimeSliceAlignedValueSet, got {type(aligned_values).__name__}."
            )
        if plane_projection is None:
            raise ValueError(
                "Runtime-slice-aligned object-label output requires a declared "
                "runtime plane projection."
            )
        if plane_projection.axis is not RuntimePlaneAxis.RUNTIME_SLICE:
            raise ValueError(
                "Runtime-slice-aligned object-label output requires the "
                f"runtime-slice axis, got {plane_projection.axis.value!r}."
            )
        if plane_projection.axis_size != aligned_values.slice_count:
            raise ValueError(
                "Runtime-slice-aligned object-label output count must exactly "
                "match the declared runtime plane axis: "
                f"{aligned_values.slice_count} != {plane_projection.axis_size}."
            )
        return RuntimeSliceAlignedValues(
            tuple(
                ObjectLabelOutputValueContextStrategy.for_output_value(
                    aligned_values.value_for_slice(slice_index)
                ).contextualize(
                    RuntimeSliceProjection.value_for_slice(
                        source_payload,
                        RuntimePlaneAxisValueProjection.from_selected_plane(
                            axis=plane_projection.axis,
                            plane_index=slice_index,
                            axis_size=plane_projection.axis_size,
                        ),
                    ),
                    aligned_values.value_for_slice(slice_index),
                    None,
                )
                for slice_index in range(aligned_values.slice_count)
            )
        )


class ContextualObjectLabelOutputValueContextStrategy(
    ObjectLabelOutputValueContextStrategy
):
    """Preserve object-label domain while filling missing source-image context."""

    value_type = ObjectLabelValue

    def contextualize(
        self,
        source_payload: RuntimePayload,
        output_value: ObjectLabelContextualizableOutput,
        plane_projection: RuntimePlaneAxisValueProjection | None,
    ) -> ObjectLabelValue:
        del plane_projection
        if not isinstance(output_value, ObjectLabelValue):
            raise TypeError(
                "Contextual object-label output strategy requires "
                f"ObjectLabelValue, got {type(output_value).__name__}."
            )
        return output_value.with_source_image_context(source_payload)


class NumpyArrayObjectLabelOutputValueContextStrategy(
    ObjectLabelOutputValueContextStrategy
):
    """Build object-label context for declared NumPy array outputs."""

    value_type = np.ndarray

    def contextualize(
        self,
        source_payload: RuntimePayload,
        output_value: ObjectLabelContextualizableOutput,
        plane_projection: RuntimePlaneAxisValueProjection | None,
    ) -> ObjectLabelValue:
        if not isinstance(output_value, np.ndarray):
            raise TypeError(
                "Runtime-array object-label output strategy requires a NumPy "
                f"array, got {type(output_value).__name__}."
            )
        return SourceImageObjectLabelBuildRequest(
            image=source_payload,
            labels=output_value,
            plane_projection=plane_projection,
        ).payload()


@dataclass(frozen=True)
class ComponentArtifactPlans(Generic[ArtifactInputPlanKeyT, ArtifactInputPlanT]):
    """Artifact plans selected for one grouped component execution."""

    inputs: ArtifactInputPlans[ArtifactInputPlanKeyT, ArtifactInputPlanT]
    outputs: ArtifactOutputPlans

    @classmethod
    def from_step_component(
        cls,
        plan: CompiledStepPlan,
        component_key: str | None,
    ) -> "ComponentArtifactPlans[ArtifactSpecRef, ArtifactInputPlan]":
        ArtifactInputPlan.require_exact_map(
            plan.artifact_inputs,
            boundary="Component artifact input",
        )
        return cls(
            inputs=dict(plan.artifact_inputs),
            outputs=cls._select_output_plans_for_component(
                plan.artifact_outputs,
                plan.execution_group_scope,
                component_key,
            ),
        )

    def select_for_invocation(
        self: "ComponentArtifactPlans[ArtifactSpecRef, ArtifactInputPlan]",
        invocation: CompiledFunctionInvocation,
        *,
        execution_scope: ComponentGroupScope,
        component_key: str | None,
    ) -> "ComponentArtifactPlans[InvocationArtifactInputProjectionKey, InvocationArtifactInputEdgePlan]":
        active_compiled_outputs = tuple(
            projected_plan
            for output_plan in invocation.artifact_output_plans
            if (
                projected_plan := self._output_plan_for_component(
                    output_plan,
                    execution_scope,
                    component_key,
                )
            )
            is not None
        )
        active_outputs = replace(
            invocation,
            artifact_output_plans=active_compiled_outputs,
        ).select_outputs(self.outputs)
        compiled_group_scope_sources = frozenset(
            source_ref
            for output_plan in invocation.artifact_output_plans
            for source_ref in output_plan.group_scope_sources()
        )
        active_group_scope_sources = frozenset(
            source_ref
            for output_plan in active_outputs.values()
            for source_ref in output_plan.group_scope_sources()
        )
        active_inputs: ArtifactInputPlans[
            InvocationArtifactInputProjectionKey,
            InvocationArtifactInputEdgePlan,
        ] = {}
        for edge_key, edge in invocation.select_inputs(self.inputs).items():
            edge_group_scope_sources = compiled_group_scope_sources.intersection(
                (edge.spec.ref(), *edge.spec.dependency_refs())
            )
            if edge_group_scope_sources and edge_group_scope_sources.isdisjoint(
                active_group_scope_sources
            ):
                continue
            active_inputs[edge_key] = edge
        return ComponentArtifactPlans(inputs=active_inputs, outputs=active_outputs)

    def select_source_bound_inputs(
        self: "ComponentArtifactPlans[InvocationArtifactInputProjectionKey, InvocationArtifactInputEdgePlan]",
        *,
        declared_source_bindings: CompiledSourceBindingPlan,
        active_source_bindings: CompiledSourceBindingPlan,
    ) -> "ComponentArtifactPlans[InvocationArtifactInputProjectionKey, InvocationArtifactInputEdgePlan]":
        """Keep only source-bound main-flow occurrences active on this component."""

        return replace(
            self,
            inputs={
                edge_key: edge
                for edge_key, edge in self.inputs.items()
                if not (
                    edge.consumes_main_flow
                    and declared_source_bindings.declares_artifact_ref(
                        edge.spec.ref()
                    )
                    and not active_source_bindings.declares_artifact_ref(
                        edge.spec.ref()
                    )
                )
            },
        )

    @classmethod
    def _select_output_plans_for_component(
        cls,
        plans: ArtifactOutputPlans,
        execution_scope: ComponentGroupScope,
        component_key: str | None,
    ) -> ArtifactOutputPlans:
        ArtifactOutputPlan.require_exact_map(
            plans,
            boundary="Component artifact output",
        )
        return {
            output_key: projected_plan
            for output_key, output_plan in plans.items()
            if (
                projected_plan := cls._output_plan_for_component(
                    output_plan,
                    execution_scope,
                    component_key,
                )
            )
            is not None
        }

    @staticmethod
    def _output_plan_for_component(
        output_plan: ArtifactOutputPlan,
        execution_scope: ComponentGroupScope,
        component_key: str | None,
    ) -> ArtifactOutputPlan | None:
        output_scope = output_plan.group_scope()
        if output_scope.is_ungrouped:
            return output_plan
        if output_scope.component is execution_scope.component:
            if not output_scope.contains_runtime_key(component_key):
                return None
            return output_plan.for_group(
                output_scope.resolve_runtime_key(component_key)
            )
        if not output_scope.is_dynamic and len(output_scope.keys) == 1:
            return output_plan.for_invocation_group(None)
        return output_plan


@dataclass(frozen=True, slots=True, kw_only=True)
class PatternGroupExecutionScope:
    """Shared pattern-group execution coordinates."""

    context: ProcessingContext
    execution_plan: CompiledStepPlan
    compiled_group: CompiledFunctionGroup
    component_value: RuntimeComponentValue = None
    fixed_component_values: RuntimeFixedComponentValues = ()

    @property
    def component_key(self) -> str | None:
        if self.component_value is None:
            return None
        return str(self.component_value)

    @property
    def unscoped_main_flow_source_binding_plan(self) -> CompiledSourceBindingPlan:
        """Return declared bindings that can contribute to main flow."""

        declared_plan = self.execution_plan.source_binding_plan
        main_flow_refs = self.compiled_group.main_flow_input_refs
        return (
            declared_plan
            if main_flow_refs is None
            else declared_plan.for_artifact_refs(main_flow_refs)
        )

    @property
    def main_flow_source_binding_plan(self) -> CompiledSourceBindingPlan:
        """Return bindings that anchor and load this group's main-flow stack."""

        return self.unscoped_main_flow_source_binding_plan.for_execution_axis_scope(
            self.axis_scope
        )

    def active_main_flow_source_binding_plan(
        self,
        payload: RuntimeArrayData,
    ) -> CompiledSourceBindingPlan:
        """Project main-flow bindings through represented payload aliases."""

        plan = self.unscoped_main_flow_source_binding_plan
        represented_names = frozenset(
            image_payload_metadata(
                payload
            ).source_provenance.represented_source_image_names
        )
        if represented_names:
            variable_components = ComponentSet.coerce(
                self.execution_plan.variable_components or ()
            )
            plan = plan.for_represented_source_stack(
                represented_names,
                variable_components=variable_components,
            )
        return plan.for_execution_axis_scope(self.axis_scope)

    @property
    def invocation_source_artifact_refs(self) -> tuple[ArtifactSpecRef, ...]:
        """Return exact source artifacts consumed by this invocation group."""

        declared_plan = self.execution_plan.source_binding_plan
        return tuple(
            dict.fromkeys(
                spec.ref()
                for invocation in self.compiled_group.invocations
                for spec in invocation.contract.artifact_inputs
                if declared_plan.binding_for_artifact_ref(spec.ref()) is not None
            )
        )

    @property
    def source_binding_plan(self) -> CompiledSourceBindingPlan:
        """Return all source bindings visible to this invocation group."""

        declared_plan = self.execution_plan.source_binding_plan
        source_refs = self.invocation_source_artifact_refs
        return (
            declared_plan.for_artifact_refs(source_refs)
            if source_refs
            else self.main_flow_source_binding_plan
        )

    @property
    def axis_component(self) -> str | None:
        if self.component_value is None:
            return None
        component = self.execution_plan.execution_group_scope.component
        return None if component is None else component.value

    @property
    def axis_component_value(self) -> str | None:
        return self.component_key

    @property
    def axis_scope(self) -> RuntimeExecutionAxisScope:
        return RuntimeExecutionAxisScope.from_raw(
            self.execution_plan.axis_id,
            component=self.axis_component,
            value=self.axis_component_value,
            fixed_component_values=self.fixed_component_values,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class FunctionRuntimeScope(PatternGroupExecutionScope):
    """Generic runtime scope shared by chain, invocation, adapter, and debug code."""

    artifacts: ComponentArtifactPlans[ArtifactSpecRef, ArtifactInputPlan]
    source_binding_context: SourceBindingRuntimeContext
    runtime_plane_index: int
    runtime_plane_count: int

    @classmethod
    def from_pattern_group(
        cls,
        request: "PatternGroupExecutionRequest",
        loaded: "PatternGroupData",
    ) -> "FunctionRuntimeScope":
        artifacts = ComponentArtifactPlans.from_step_component(
            request.execution_plan,
            request.component_key,
        )
        logger.debug(
            "Selected artifact outputs for component %s: %s",
            request.component_key,
            artifacts.outputs,
        )
        source_provenance = image_payload_metadata(
            loaded.main_data_stack
        ).source_provenance.with_common_scalar_identity_from_planes()
        common_source_metadata = source_provenance.source_component_metadata or {}
        variable_components = ComponentSet.coerce(
            request.execution_plan.variable_components or ()
        )
        execution_group_component = (
            request.execution_plan.execution_group_scope.component
        )
        fixed_components = tuple(
            component
            for component in AllComponents
            if not component.is_multiprocessing_axis()
            and component is not execution_group_component
            and component not in variable_components
            and source_component_metadata_value(
                common_source_metadata,
                component,
            )
            is not None
        )
        return cls(
            context=request.context,
            execution_plan=request.execution_plan,
            compiled_group=request.compiled_group,
            artifacts=artifacts,
            source_binding_context=loaded.source_binding_context,
            runtime_plane_index=request.component_index,
            runtime_plane_count=len(loaded.matching_files),
            component_value=request.component_value,
            fixed_component_values=source_provenance.require_common_component_values(
                fixed_components
            ),
        )

    def require_invocations(self) -> None:
        if self.compiled_group.invocations:
            return
        raise ValueError(
            f"Compiled function group {self.compiled_group.group_key} has no invocations."
        )

    def execute_chain(
        self, initial_data_stack: RuntimeArrayData
    ) -> RuntimeArrayData | NoMainFlowOutput:
        self.require_invocations()
        current_stack: RuntimeArrayData | NoMainFlowOutput = initial_data_stack
        current_memory_type = self.execution_plan.input_memory_type
        debug_sink = debug_event_sink_from_context(self.context)
        declared_source_bindings = self.execution_plan.source_binding_plan
        active_main_flow_bindings = self.active_main_flow_source_binding_plan(
            initial_data_stack
        )
        for invocation in self.compiled_group.invocations:
            group_key = invocation.key.runtime_group_key(self.component_value)
            artifacts = self.artifacts.select_for_invocation(
                invocation,
                execution_scope=self.execution_plan.execution_group_scope,
                component_key=self.component_key,
            ).select_source_bound_inputs(
                declared_source_bindings=declared_source_bindings,
                active_source_bindings=active_main_flow_bindings,
            )
            if (
                invocation.adapter_records_artifact_outputs
                and not artifacts.outputs
            ):
                continue
            runtime_invocation = invocation.for_runtime_outputs(
                output_plans=tuple(artifacts.outputs.values()),
            )
            executor = FunctionCoreExecutor(
                main_data_arg=current_stack,
                source_memory_type=current_memory_type,
                runtime_scope=self,
                invocation=runtime_invocation,
                artifacts=artifacts,
                group_key=group_key,
                plane_projection=RuntimePlaneProjection.stack(self.runtime_plane_count),
            )
            captures_debug = debug_sink.captures_invocation_events()
            if captures_debug and debug_sink.should_skip_invocation(
                executor.debug_cursor()
            ):
                continue

            invocation_started_at = time.perf_counter()
            try:
                current_stack = executor.execute(
                    debug_sink=debug_sink if captures_debug else None,
                )
            except Exception as exc:
                if captures_debug:
                    debug_sink.record(
                        executor.debug_event(
                            DebugEventType.EXCEPTION,
                            exception=exc,
                        )
                    )
                raise
            invocation_seconds = time.perf_counter() - invocation_started_at
            if captures_debug:
                after_event = executor.debug_event(
                    DebugEventType.AFTER_INVOCATION,
                    timing_seconds=invocation_seconds,
                )
                debug_sink.record(after_event)
                if debug_sink.should_stop_after_invocation(after_event):
                    break
            RuntimeProfileSink.record(
                "invocation_total",
                invocation_seconds,
                function=invocation.key.function_name,
                group=invocation.key.group_key,
                position=invocation.key.position,
            )
            if isinstance(current_stack, NoMainFlowOutput):
                return current_stack
            current_memory_type = executor.memory_types().output_type
        if self.compiled_group.preserves_input_main_flow() and all(
            invocation.adapter_records_artifact_outputs
            for invocation in self.compiled_group.invocations
        ):
            return NoMainFlowOutput()
        return current_stack


@dataclass(frozen=True, slots=True, kw_only=True)
class PatternGroupExecutionRequest(PatternGroupExecutionScope):
    """All runtime data needed to process one pattern group."""

    pattern_group_info: JsonValue
    component_index: int
    component_count: int


@dataclass(frozen=True, kw_only=True)
class PatternGroupData:
    """Loaded image data for one pattern group."""

    matching_files: list[str]
    main_data_stack: RuntimeArrayData
    source_binding_context: SourceBindingRuntimeContext = field(
        default_factory=SourceBindingRuntimeContext.empty
    )


@dataclass(frozen=True, slots=True)
class OutputPathBatchEntry:
    """Resolved identity for one output path in a runtime save batch."""

    index: int
    input_path: str | None
    output_path: str
    identity: FunctionOutputIdentity

    def diagnostic(self) -> str:
        filename_components = (
            self.identity.filename_component_values
            if self.identity.filename_component_values is not None
            else self.identity.component_values
        )
        return (
            f"#{self.index}: input={self.input_path!r}, output={self.output_path!r}, "
            f"identity={dict(self.identity.component_values)!r}, "
            f"filename_identity={dict(filename_components)!r}, "
            f"source={self.identity.source!r}"
        )


@dataclass(frozen=True, slots=True)
class OutputPathBatchUniqueness:
    """Validate that one runtime output batch has unique destination paths."""

    output_paths: Sequence[str]
    input_paths: Sequence[str]
    step_name: str
    pattern_repr: str
    entries: Sequence[OutputPathBatchEntry] = ()

    def validate(self) -> None:
        counts: dict[str, int] = {}
        for path in self.output_paths:
            if path not in counts:
                counts[path] = 1
                continue
            counts[path] += 1
        duplicates = tuple(path for path, count in counts.items() if count > 1)
        if not duplicates:
            return
        raise ValueError(
            f"Step {self.step_name!r} produced duplicate output path(s) "
            f"for pattern {self.pattern_repr}: {duplicates!r}. Input files: "
            f"{tuple(self.input_paths)!r}. Output identity details: "
            f"{tuple(entry.diagnostic() for entry in self.entries)!r}."
        )


def _save_artifact_value(
    context: ProcessingContext,
    output_plan: ArtifactOutputPlan,
    value: RuntimePayload,
    source_payload: RuntimePayload,
    *,
    execution_scope: RuntimeExecutionAxisScope,
    group_key: str | None,
    plane_projector: RuntimePlaneAxisProjector | None,
    materialization_source_metadata: ImagePayloadMetadata | None = None,
) -> RuntimePayload:
    """Validate and save one planned artifact value to the memory VFS."""
    resolved_output_plan = output_plan.for_invocation_group(group_key)
    vfs_path = resolved_output_plan.path
    contextualized_value = FunctionOutputContextStrategy.for_output_plan(
        resolved_output_plan
    ).contextualize_from_projector(
        source_payload,
        value,
        resolved_output_plan,
        plane_projector,
    )
    runtime_value = RuntimeValue.normalize_for_execution_scope(
        resolved_output_plan,
        contextualized_value,
        execution_scope=execution_scope,
        materialization_source_metadata=materialization_source_metadata,
    )

    location = RuntimeArtifactLocation(
        path=vfs_path,
        backend=Backend.MEMORY.value,
    )
    runtime_value_store = context.runtime_value_store
    runtime_value_store.replace(
        runtime_value,
        path=location.path,
        backend=location.backend,
    )
    replace_runtime_artifact_payload(
        context.filemanager,
        runtime_value.data,
        location,
    )
    return runtime_value.data

def _load_artifact_input_values(
    runtime_scope: FunctionRuntimeScope,
    input_plan: InvocationArtifactInputEdgePlan,
) -> tuple[RuntimeValue, ...]:
    """Project producer-owned runtime records into one consumer invocation."""
    context = runtime_scope.context
    return RuntimeArtifactInput(
        edge_plan=input_plan,
        axis_scope=runtime_scope.axis_scope,
        backend=Backend.MEMORY.value,
    ).projected_values(context.runtime_value_store)


def prepare_compiled_function_group(group: CompiledFunctionGroup) -> None:
    """Run optional preparation hooks for each callable in a compiled group."""
    for invocation in group.invocations:
        FunctionInvocationCallableResolver.prepare(invocation)


def prepare_compiled_context_callables(
    compiled_contexts: Mapping[str, ProcessingContext],
) -> None:
    """Prepare every compiled callable visible in the compiled contexts."""
    prepared_group_keys: set[tuple[str, int, str]] = set()
    prepared_invocation_count = 0
    for context_key, context in compiled_contexts.items():
        step_plans = context.step_plans
        if not step_plans:
            continue
        for step_plan in step_plans.values():
            compiled_pattern = step_plan.compiled_function_pattern
            if compiled_pattern is None:
                continue
            for group in compiled_pattern.groups:
                prepare_key = (
                    str(context_key),
                    int(step_plan.step_index),
                    group.group_key,
                )
                if prepare_key in prepared_group_keys:
                    continue
                prepare_compiled_function_group(group)
                prepared_invocation_count += len(group.invocations)
                prepared_group_keys.add(prepare_key)
    logger.info(
        "Prepared %d compiled callable invocations across %d groups.",
        prepared_invocation_count,
        len(prepared_group_keys),
    )


@dataclass(frozen=True, slots=True)
class FunctionCoreExecutor:
    """Execute one scoped callable invocation and route declared artifact I/O."""

    runtime_scope: FunctionRuntimeScope
    invocation: CompiledFunctionInvocation
    artifacts: ComponentArtifactPlans[
        InvocationArtifactInputProjectionKey,
        InvocationArtifactInputEdgePlan,
    ]
    group_key: str | None
    plane_projection: RuntimePlaneProjection
    main_data_arg: RuntimeArrayData
    source_memory_type: str

    @property
    def selected_artifact_input_edges(
        self,
    ) -> tuple[InvocationArtifactInputEdgePlan, ...]:
        """Return component-selected compiled occurrences in declaration order."""

        return tuple(self.artifacts.inputs.values())

    def runtime_adapter_request(
        self,
        source_payload: RuntimePayload,
    ) -> RuntimeAdapterRequest:
        return RuntimeAdapterRequest.from_runtime_scope(
            runtime_scope=self.runtime_scope,
            callable_contract=self.invocation.contract,
            artifact_inputs={
                edge.key: edge for edge in self.selected_artifact_input_edges
            },
            artifact_outputs=self.artifacts.outputs,
            group_key=self.group_key,
            plane_projection=self.plane_projection,
            source_payload=source_payload,
        )

    def debug_cursor(self) -> DebugCursor:
        return DebugCursor.from_invocation(
            step_index=self.runtime_scope.execution_plan.step_index,
            step_scope_id=self.runtime_scope.execution_plan.step_scope_id,
            invocation=self.invocation,
            pattern_group_identity=str(self.runtime_scope.runtime_plane_index),
        )

    def debug_artifacts(
        self,
        artifact_plans: (
            ArtifactInputPlans | ArtifactOutputPlans
        ),
        artifact_values: Mapping[ArtifactSpecRef, object] | None = None,
    ) -> DebugArtifactRefProjection:
        return DebugArtifactRefProjection.from_artifact_plans(
            artifact_plans=artifact_plans,
            cursor=self.debug_cursor(),
            artifact_values=artifact_values,
        )

    def debug_event(
        self,
        event_type: DebugEventType,
        *,
        exception: Exception | None = None,
        timing_seconds: float | None = None,
        invocation_parameters: tuple[DebugInvocationParameter, ...] = (),
        input_artifact_values: Mapping[ArtifactSpecRef, object] | None = None,
    ) -> DebugEvent:
        return DebugEvent.for_invocation(
            event_type=event_type,
            cursor=self.debug_cursor(),
            step_name=self.runtime_scope.execution_plan.step_name,
            callable_name=self.invocation.key.function_name,
            axis_id=self.runtime_scope.execution_plan.axis_id,
            input_artifacts=self.debug_artifacts(
                {
                    edge.storage_plan.ref(): edge.storage_plan
                    for edge in self.artifacts.inputs.values()
                    if edge.storage_plan is not None
                },
                input_artifact_values,
            ),
            output_artifacts=self.debug_artifacts(self.artifacts.outputs),
            exception=exception,
            timing_seconds=timing_seconds,
            invocation_parameters=invocation_parameters,
        )

    @property
    def func_callable(self) -> Callable:
        return FunctionInvocationCallableResolver.resolve(self.invocation)

    @property
    def base_kwargs(self) -> RuntimeCallableKwargs:
        return self.invocation.kwargs_dict

    @property
    def function_name(self) -> str:
        return self.invocation.contract.function_name

    def main_flow_output_source_payload(
        self,
        source_payload: RuntimePayload,
    ) -> RuntimePayload:
        """Project source context through the callable's nominal processing contract."""

        return self.invocation.contract.require_processing_contract().declaration.main_flow_output_source_payload(
            source_payload
        )

    def execute(
        self,
        *,
        debug_sink: DebugEventSink | None = None,
    ) -> RuntimePayload | NoMainFlowOutput:
        memory_types = self.memory_types()
        source_payload = MainFlowMemoryConversion(
            payload=self.main_data_arg,
            source_type=self.source_memory_type,
            target_type=memory_types.input_type,
            gpu_id=self.runtime_scope.execution_plan.device_id,
        ).converted_payload()
        main_data_arg = self.main_flow_call_argument(source_payload)
        final_kwargs = dict(self.base_kwargs)
        self.bind_compiled_runtime_parameters(final_kwargs)
        loads_artifact_inputs = self.should_load_artifact_inputs()
        loaded_artifact_payloads: dict[ArtifactSpecRef, RuntimePayload] = {}
        if loads_artifact_inputs:
            loaded_artifact_payloads = self.load_artifact_inputs(
                final_kwargs,
            )
        self.bind_runtime_owned_parameters(final_kwargs)
        self.bind_runtime_adapter(final_kwargs, source_payload)
        raw_output = self.invoke(
            main_data_arg,
            final_kwargs,
            loaded_artifact_payloads=loaded_artifact_payloads,
            debug_sink=debug_sink,
        )
        main_output = self.save_artifact_outputs(
            raw_output,
            source_payload,
            loaded_artifact_payloads=loaded_artifact_payloads,
        )
        if isinstance(main_output, NoMainFlowOutput):
            return main_output
        if self.invocation.adapter_records_artifact_outputs:
            return main_output
        output_source_payload = self.main_flow_output_source_payload(
            self.execution_group_source_payload(source_payload)
        )
        return FunctionOutputContextStrategy.for_output_plan(
            None
        ).contextualize_from_projector(
            output_source_payload,
            main_output,
            None,
            self.plane_projection,
        )

    def main_flow_call_argument(
        self, source_payload: RuntimePayload
    ) -> RuntimeCallableArgument:
        """Expose arrays to ordinary callables and carriers to adapter-backed calls."""

        if self.invocation.contract.runtime_adapter is not None:
            return source_payload
        return image_payload_data(source_payload)

    def memory_types(self) -> "FunctionChainInvocationMemoryTypes":
        return FunctionChainInvocationMemoryTypes.from_invocation(self.invocation)

    def execution_group_source_payload(
        self,
        source_payload: RuntimePayload,
    ) -> RuntimePayload:
        """Return source payload metadata carrying the current grouped identity."""
        component = self.runtime_scope.execution_plan.execution_group_scope.component
        if component is None or self.group_key is None:
            return source_payload
        metadata = image_payload_metadata(source_payload)
        component_metadata = dict(metadata.source_component_metadata or {})
        component_metadata = with_source_component_metadata(
            component_metadata,
            component,
            self.group_key,
        )
        return metadata.with_source_component_metadata(component_metadata).attach_to(
            source_payload
        )

    def declared_source_payload(
        self,
        source_ref: ArtifactSpecRef,
        primary_source_payload: RuntimePayload,
        *,
        loaded_artifact_payloads: Mapping[ArtifactSpecRef, RuntimePayload],
    ) -> RuntimePayload:
        input_spec = self.invocation.contract.artifact_inputs.by_ref(source_ref)
        if input_spec is None:
            raise ValueError(
                f"Invocation {self.invocation.key!r} does not declare source artifact "
                f"{source_ref!r}."
            )
        stored_payload = loaded_artifact_payloads.get(source_ref)
        source_binding = self.runtime_scope.source_binding_plan.binding_for_artifact_ref(
            source_ref
        )
        main_flow_edges = tuple(
            edge
            for edge in self.selected_artifact_input_edges
            if edge.spec.ref() == source_ref and edge.consumes_main_flow
        )
        uses_main_flow = bool(
            stored_payload is None
            and source_binding is None
            and main_flow_edges
        )
        resolved_origins = sum(
            (
                stored_payload is not None,
                source_binding is not None,
                uses_main_flow,
            )
        )
        if resolved_origins != 1:
            raise ValueError(
                f"Invocation {self.invocation.key!r} source artifact {source_ref!r} "
                "must resolve to exactly one compiled producer, source binding, or "
                f"main-flow input; resolved {resolved_origins}."
            )
        if stored_payload is not None:
            return stored_payload
        if source_binding is not None:
            return cast(
                RuntimePayload,
                self.runtime_adapter_request(
                    primary_source_payload
                ).source_artifact_payload(source_ref),
            )
        if len(main_flow_edges) != 1:
            raise ValueError(
                f"Invocation {self.invocation.key!r} source artifact {source_ref!r} "
                "must resolve through exactly one selected main-flow input edge; "
                f"resolved {len(main_flow_edges)}."
            )
        main_flow_projection = main_flow_edges[0].main_flow_projection
        if main_flow_projection is MainFlowInputProjection.COMPLETE_PAYLOAD:
            return primary_source_payload
        if main_flow_projection is not MainFlowInputProjection.DECLARED_SOURCE_IMAGE:
            raise ValueError(
                f"Invocation {self.invocation.key!r} source artifact {source_ref!r} "
                "consumes main flow without a compiled projection."
            )
        return project_declared_source_identity(primary_source_payload, source_ref)

    def load_artifact_inputs(
        self,
        final_kwargs: dict[str, RuntimeCallableArgument],
    ) -> dict[ArtifactSpecRef, RuntimePayload]:
        if not self.should_load_artifact_inputs():
            return {}
        logger.info(
            f"Artifact inputs for {self.function_name}: {self.artifacts.inputs}"
        )
        loaded_artifact_payloads: dict[ArtifactSpecRef, RuntimePayload] = {}
        parameter_values: dict[str, list[RuntimeValue]] = {}
        for input_plan in self.selected_artifact_input_edges:
            if input_plan.storage_plan is None:
                continue
            parameter_name = input_plan.spec.parameter_name
            if parameter_name is None:
                raise ValueError(
                    f"Compiled invocation {self.invocation.key!r} runtime-loaded input "
                    f"edge {input_plan.key!r} has no callable parameter."
                )
            artifact_ref = input_plan.spec.ref()
            projected_values = self.load_artifact_input(
                input_plan.spec.name,
                input_plan,
            )
            loaded_value = RuntimeValue.compose(projected_values)
            loaded_artifact_payloads[artifact_ref] = loaded_value
            parameter_values.setdefault(parameter_name, []).extend(
                projected_values
            )
        for parameter_name, projected_values in parameter_values.items():
            final_kwargs[parameter_name] = RuntimeValue.compose(tuple(projected_values))
        return loaded_artifact_payloads

    def should_load_artifact_inputs(self) -> bool:
        return bool(
            any(
                edge.storage_plan is not None
                for edge in self.artifacts.inputs.values()
            )
            and not self.invocation.adapter_manages_artifact_inputs
        )

    def load_artifact_input(
        self,
        arg_name: str,
        edge_plan: InvocationArtifactInputEdgePlan,
    ) -> tuple[RuntimeValue, ...]:
        storage_plan = edge_plan.storage_plan
        if storage_plan is None:
            raise ValueError("Artifact input loading requires a storage-backed edge.")
        logger.info(
            f"Loading artifact input '{arg_name}' from path '{storage_plan.path}' "
            "(memory backend)"
        )
        load_started_at = time.perf_counter()
        try:
            loaded_values = _load_artifact_input_values(
                self.runtime_scope,
                edge_plan,
            )
        except Exception as exc:
            logger.error(
                f"Failed to load artifact input '{arg_name}' from "
                f"'{storage_plan.path}': {exc}",
                exc_info=True,
            )
            raise
        RuntimeProfileSink.record(
            "artifact_input_load",
            time.perf_counter() - load_started_at,
            function=self.function_name,
            artifact=arg_name,
            artifact_type=storage_plan.artifact_type.value,
        )
        return loaded_values

    def bind_runtime_owned_parameters(
        self,
        final_kwargs: dict[str, RuntimeCallableArgument],
    ) -> None:
        context_parameter_name = self.invocation.contract.runtime_context_parameter
        if context_parameter_name is not None:
            final_kwargs[context_parameter_name] = self.runtime_scope.context

    def bind_compiled_runtime_parameters(
        self,
        final_kwargs: dict[str, RuntimeCallableArgument],
    ) -> None:
        for binding in self.invocation.runtime_parameter_bindings:
            final_kwargs[binding.parameter_name] = binding.value

    def bind_runtime_adapter(
        self,
        final_kwargs: dict[str, RuntimeCallableArgument],
        source_payload: RuntimePayload,
    ) -> None:
        runtime_adapter = self.invocation.contract.runtime_adapter
        if runtime_adapter is None:
            return
        adapter_parameter = runtime_adapter.require_parameter_name()
        adapter_started_at = time.perf_counter()
        final_kwargs[adapter_parameter] = runtime_adapter.factory(
            self.runtime_adapter_request(source_payload)
        )
        RuntimeProfileSink.record(
            "runtime_adapter_factory",
            time.perf_counter() - adapter_started_at,
            function=self.function_name,
            adapter=adapter_parameter,
        )

    def invoke(
        self,
        main_data_arg: RuntimeCallableArgument,
        final_kwargs: dict[str, RuntimeCallableArgument],
        *,
        loaded_artifact_payloads: Mapping[ArtifactSpecRef, RuntimePayload],
        debug_sink: DebugEventSink | None,
    ) -> RuntimeFunctionOutput:
        logger.info(f"Executing function: {self.function_name}")
        func_callable = self.func_callable
        contract = self.invocation.contract
        primary_parameter = contract.primary_input_parameter_name
        bound_parameters = dict(final_kwargs)
        if primary_parameter is not None:
            bound_parameters[primary_parameter] = main_data_arg
        plane_projector: RuntimePlaneAxisProjector = self.plane_projection
        runtime_adapter = self.invocation.contract.runtime_adapter
        if runtime_adapter is not None:
            adapter_parameter = runtime_adapter.require_parameter_name()
            adapter_value = final_kwargs[adapter_parameter]
            if isinstance(adapter_value, RuntimePlaneAxisProjector):
                plane_projector = adapter_value
        if debug_sink is not None:
            if primary_parameter is None:
                raise TypeError(
                    f"Callable {self.function_name!r} has no declared primary input "
                    "parameter for runtime invocation diagnostics."
                )
            debug_sink.record(
                self.debug_event(
                    DebugEventType.BEFORE_INVOCATION,
                    invocation_parameters=DebugInvocationParameter.from_kwargs(
                        bound_parameters,
                        plane_projector=plane_projector,
                    ),
                    input_artifact_values=loaded_artifact_payloads,
                )
            )
        call_started_at = time.perf_counter()
        try:
            raw_output = func_callable(
                main_data_arg,
                **final_kwargs,
            )
        except RuntimeSliceProjectionDeclarationError as exc:
            cursor = self.debug_cursor()
            invocation_parameters = DebugInvocationParameter.from_kwargs(
                bound_parameters,
                plane_projector=plane_projector,
            )
            selected_plane = plane_projector.runtime_slice_plane_index()
            selected_plane_status = (
                "preserved_stack" if selected_plane is None else str(selected_plane)
            )
            artifact_refs = (
                *(edge.spec.ref() for edge in self.selected_artifact_input_edges),
                *(plan.ref() for plan in self.invocation.artifact_output_plans),
            )
            raise type(exc)(
                f"{exc} Invocation boundary: step_index={cursor.step_index}; "
                f"step_name={self.runtime_scope.execution_plan.step_name!r}; "
                f"function_invocation_key={self.invocation.key!r}; "
                f"module={contract.module_name!r}; "
                f"callable={contract.function_name!r}; "
                f"artifact_spec_refs={artifact_refs!r}; "
                f"kwarg_names={tuple(sorted(final_kwargs))!r}; "
                f"nominal_values={tuple((parameter.name, parameter.value_repr) for parameter in invocation_parameters)!r}; "
                f"selected_runtime_plane={selected_plane_status}; "
                "execution_axis_cardinality="
                f"{plane_projector.runtime_slice_axis_size()!r}; "
                "image_payload_execution_mode="
                f"{contract.runtime_image_execution_mode}; "
                f"processing_contract={contract.processing_contract}."
            ) from exc
        RuntimeProfileSink.record(
            "function_call",
            time.perf_counter() - call_started_at,
            function=self.function_name,
        )
        return raw_output

    def save_artifact_outputs(
        self,
        raw_output: RuntimeFunctionOutput,
        source_payload: RuntimePayload,
        *,
        loaded_artifact_payloads: Mapping[ArtifactSpecRef, RuntimePayload],
    ) -> RuntimePayload | NoMainFlowOutput:
        if self.invocation.adapter_records_artifact_outputs:
            return self.save_module_recorded_output(raw_output)
        output_plans = tuple(self.artifacts.outputs.values())
        declared_specs = self.invocation.contract.artifact_outputs
        if not declared_specs:
            if isinstance(raw_output, tuple):
                raise TypeError(
                    "Tuple returns require declared special-output slots; multiple "
                    "main-flow images must be packed as AlignedImageStack."
                )
            return raw_output

        output_matcher = RuntimeReturnedOutputMatcher(
            callable_contract=self.invocation.contract,
            returned_output=raw_output,
        )
        _returned_values, matched_outputs = output_matcher.resolve_plan_values(
            output_plans
        )
        saved_values = {
            output_plan.ref(): self.save_artifact_output(
                output_plan.name,
                output_plan,
                output_value,
                source_payload,
                loaded_artifact_payloads=loaded_artifact_payloads,
            )
            for output_plan, _output_spec, output_value in matched_outputs
        }
        canonical_refs = frozenset(
            spec.ref()
            for spec in self.invocation.contract.canonical_return_output_specs
        )
        main_outputs = tuple(
            (output_plan, output_spec, saved_values[output_plan.ref()])
            for output_plan, output_spec, _output_value in matched_outputs
            if output_plan.ref() in canonical_refs
        )
        if main_outputs:
            output_values = tuple(
                output_value for _output_plan, _output_spec, output_value in main_outputs
            )
            if len(output_values) == 1:
                return output_values[0]
            return ImageOutputBundle(
                output_values,
                tuple(
                    AlignedImageSliceContext.main_flow(
                        output_key=output_plan.name,
                        artifact_kind=output_plan.artifact_type.value,
                    )
                    for output_plan, _output_spec, _output_value in main_outputs
                ),
            )
        return output_matcher.canonical_output

    def save_module_recorded_output(
        self,
        raw_output: RuntimeFunctionOutput,
    ) -> RuntimePayload | NoMainFlowOutput:
        """Return the main-flow value from a module that records outputs internally."""
        if isinstance(raw_output, NoMainFlowOutput):
            return raw_output
        if isinstance(raw_output, tuple):
            raise TypeError(
                "Module artifact contracts with runtime adapters record declared "
                "outputs internally; they must return one main-flow payload, not a "
                "tuple of artifact values."
            )
        return raw_output

    def save_artifact_output(
        self,
        output_key: str,
        output_plan: ArtifactOutputPlan,
        value: RuntimePayload,
        source_payload: RuntimePayload,
        *,
        loaded_artifact_payloads: Mapping[ArtifactSpecRef, RuntimePayload],
    ) -> RuntimePayload:
        logger.info(
            f"Saving artifact output '{output_key}' to VFS path '{output_plan.path}' "
            "(memory backend)"
        )
        save_started_at = time.perf_counter()
        artifact_source_payload = self.artifact_output_source_payload(
            output_plan,
            source_payload,
            loaded_artifact_payloads=loaded_artifact_payloads,
        )
        output_source_payload = self.main_flow_output_source_payload(
            self.execution_group_source_payload(artifact_source_payload)
        )
        materialization_source_metadata = None
        materialization_source_ref = output_plan.materialization_source()
        if (
            materialization_source_ref is not None
            and materialization_source_ref != output_plan.source_context_source()
        ):
            materialization_source_payload = self.declared_source_payload(
                materialization_source_ref,
                source_payload,
                loaded_artifact_payloads=loaded_artifact_payloads,
            )
            materialization_source_metadata = image_payload_metadata(
                materialization_source_payload
            )
        saved_value = _save_artifact_value(
            self.runtime_scope.context,
            output_plan,
            value,
            output_source_payload,
            execution_scope=self.runtime_scope.axis_scope,
            group_key=self.group_key,
            materialization_source_metadata=materialization_source_metadata,
            plane_projector=self.plane_projection,
        )
        RuntimeProfileSink.record(
            "artifact_output_save",
            time.perf_counter() - save_started_at,
            function=self.function_name,
            artifact=output_key,
            artifact_type=output_plan.artifact_type.value,
        )
        return saved_value

    def artifact_output_source_payload(
        self,
        output_plan: ArtifactOutputPlan,
        primary_source_payload: RuntimePayload,
        *,
        loaded_artifact_payloads: Mapping[ArtifactSpecRef, RuntimePayload],
    ) -> RuntimePayload:
        source_ref = output_plan.source_context_source()
        if source_ref is None:
            return primary_source_payload
        return self.declared_source_payload(
            source_ref,
            primary_source_payload,
            loaded_artifact_payloads=loaded_artifact_payloads,
        )


@dataclass(frozen=True, slots=True)
class FunctionChainInvocationMemoryTypes:
    """Validated memory types for one compiled invocation."""

    input_type: str
    output_type: str

    @classmethod
    def from_invocation(
        cls,
        invocation: CompiledFunctionInvocation,
    ) -> "FunctionChainInvocationMemoryTypes":
        if (
            invocation.input_memory_type is None
            or invocation.output_memory_type is None
        ):
            raise ValueError(
                f"Compiled invocation {invocation.key} is missing memory types."
            )
        return cls(invocation.input_memory_type, invocation.output_memory_type)


@dataclass(frozen=True, slots=True)
class VariableComponentNames:
    """Microscope parser variable-component names for pattern lookup."""

    components: Sequence[VariableComponents]

    @property
    def value(self) -> list[str] | None:
        if not self.components:
            return None
        return [component.value for component in self.components]


@dataclass(frozen=True, slots=True)
class MainFlowMemoryConversion:
    """Main-flow image memory conversion preserving payload context."""

    payload: RuntimeArrayData
    source_type: str
    target_type: str
    gpu_id: int | None

    def converted_payload(self) -> RuntimeArrayData:
        data = image_payload_data(self.payload)
        converted = convert_memory(
            data=data,
            source_type=self.source_type,
            target_type=self.target_type,
            gpu_id=self.gpu_id,
        )
        return with_image_payload_data(self.payload, converted)


@dataclass(slots=True)
class PatternGroupOutputData:
    """Unstacked output slices plus declared per-slice output semantics."""

    slices: list[RuntimeArrayData]
    slice_contexts: tuple[AlignedImageSliceContext, ...] = ()
    stack_payload: RuntimeArrayData | None = None

    def __post_init__(self) -> None:
        if not self.slice_contexts:
            self.slice_contexts = tuple(
                AlignedImageSliceContext.anonymous_main_flow() for _slice in self.slices
            )
        if len(self.slice_contexts) != len(self.slices):
            raise ValueError(
                "PatternGroupOutputData.slice_contexts must match slices; "
                f"got {len(self.slice_contexts)} context(s) for {len(self.slices)} slice(s)."
            )

    def __iter__(self) -> Iterator[RuntimeArrayData]:
        return iter(self.slices)

    def __len__(self) -> int:
        return len(self.slices)

    def __getitem__(self, index: int) -> RuntimeArrayData:
        return self.slices[index]

    def __eq__(self, other: object) -> bool:
        if isinstance(other, PatternGroupOutputData):
            return (
                self.slices == other.slices
                and self.slice_contexts == other.slice_contexts
            )
        if isinstance(other, SequenceABC):
            return self.slices == list(other)
        return NotImplemented


class PatternGroupRuntime:
    """Staged runtime for one pattern group."""

    def __init__(self, request: PatternGroupExecutionRequest) -> None:
        self.request = request
        self.pattern_repr = str(request.pattern_group_info)[:100]

    def source_workspace_projection_cache(
        self,
    ) -> VirtualWorkspaceSourceProjectionCache:
        """Return the per-context source-workspace projection cache."""
        return self.request.context.runtime_source_workspace_projection_cache

    def source_workspace_projection_authority(
        self,
    ) -> VirtualWorkspaceSourceProjectionAuthority:
        return VirtualWorkspaceSourceProjectionAuthority.from_context(
            self.request.context,
            cache=self.source_workspace_projection_cache(),
        )

    @staticmethod
    def _is_relative_to(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
        except ValueError:
            return False
        return True

    @classmethod
    def _input_memory_path(cls, input_dir: Path, matched_path: str) -> str:
        """Return the VFS memory path for one matched source path."""
        path = Path(matched_path)
        if path.is_absolute() or cls._is_relative_to(path, input_dir):
            return str(path)
        return str(input_dir / path)

    @classmethod
    def _input_relative_path(cls, input_dir: Path, matched_path: str) -> Path:
        """Return matched path identity relative to the step input root."""
        path = Path(matched_path)
        if cls._is_relative_to(path, input_dir):
            return path.relative_to(input_dir)
        if path.is_absolute():
            return Path(path.name)
        return path

    def run(self) -> None:
        start_time = time.time()
        plan = self.request.execution_plan
        logger.debug(f"Processing pattern {self.pattern_repr} for axis {plan.axis_id}")

        try:
            load_started_at = time.perf_counter()
            loaded = self._load_input_stack()
        except NoStepOutputManifestMatch:
            logger.debug(
                "Skipping stale pattern group %s for step %s (%s); no files "
                "belong to producer manifest.",
                self.pattern_repr,
                plan.step_index,
                plan.step_name,
            )
            return
        try:
            RuntimeProfileSink.record(
                "pattern_load_stack",
                time.perf_counter() - load_started_at,
                step=plan.step_index,
                step_name=plan.step_name,
                pattern=self.pattern_repr,
            )
            execute_started_at = time.perf_counter()
            processed_stack = self._execute_pattern(loaded)
            RuntimeProfileSink.record(
                "pattern_execute_chain",
                time.perf_counter() - execute_started_at,
                step=plan.step_index,
                step_name=plan.step_name,
                pattern=self.pattern_repr,
            )
            if isinstance(processed_stack, NoMainFlowOutput):
                self._record_main_flow_passthrough(loaded.matching_files)
                RuntimeProfileSink.record(
                    "pattern_no_main_flow_output",
                    0.0,
                    step=plan.step_index,
                    step_name=plan.step_name,
                    pattern=self.pattern_repr,
                )
                logger.debug(
                    "Pattern group %s for step %s recorded artifacts without "
                    "publishing main-flow output.",
                    self.pattern_repr,
                    plan.step_name,
                )
                return
            unstack_started_at = time.perf_counter()
            output_data = self._validate_and_unstack(processed_stack, loaded)
            RuntimeProfileSink.record(
                "pattern_validate_unstack",
                time.perf_counter() - unstack_started_at,
                step=plan.step_index,
                step_name=plan.step_name,
                pattern=self.pattern_repr,
            )
            save_started_at = time.perf_counter()
            output_records = self._save_outputs(output_data, loaded.matching_files)
            output_paths = [record.output_path for record in output_records]
            RuntimeProfileSink.record(
                "pattern_save_outputs",
                time.perf_counter() - save_started_at,
                step=plan.step_index,
                step_name=plan.step_name,
                pattern=self.pattern_repr,
            )
            cleanup_started_at = time.perf_counter()
            self._cleanup_collapsed_domains(
                output_data.slices,
                loaded.matching_files,
                output_paths,
            )
            step_output_manifest(self.request.context).record_outputs(
                plan,
                output_records,
                collapsed_input_domain=(
                    len(output_data.slices) < len(loaded.matching_files)
                ),
            )
            RuntimeProfileSink.record(
                "pattern_cleanup",
                time.perf_counter() - cleanup_started_at,
                step=plan.step_index,
                step_name=plan.step_name,
                pattern=self.pattern_repr,
            )
            logger.debug(
                f"Finished pattern group {self.pattern_repr} in {(time.time() - start_time):.2f}s."
            )
        except Exception as e:
            import traceback

            full_traceback = traceback.format_exc()
            logger.error(
                f"Error processing pattern group {self.pattern_repr}: {e}",
                exc_info=True,
            )
            logger.error(
                f"Full traceback for pattern group {self.pattern_repr}:\n{full_traceback}"
            )
            raise ValueError(
                f"Failed to process pattern group {self.pattern_repr}: {e}"
            ) from e

    def _record_main_flow_passthrough(self, matching_files: Sequence[str]) -> None:
        """Record existing main-flow anchors for artifact-only step outputs."""
        if not matching_files:
            return
        plan = self.request.execution_plan
        parser = self.request.context.microscope_handler.parser
        output_contexts = self._producer_output_contexts(matching_files)
        records = tuple(
            ProducedOutputSemantics.from_existing_main_flow_path(
                plan,
                self._input_memory_path(plan.input_dir, matching_file),
                parser,
                output_context=output_context,
            )
            for matching_file, output_context in zip(
                matching_files,
                output_contexts,
                strict=True,
            )
        )
        step_output_manifest(self.request.context).record_outputs(plan, records)

    def _producer_output_contexts(
        self,
        matching_files: Sequence[str],
    ) -> tuple[AlignedImageSliceContext, ...]:
        """Resolve exact producer contexts for the loaded main-flow paths."""

        return step_output_manifest(
            self.request.context
        ).producer_output_contexts_for_paths(
            self.request.execution_plan,
            matching_files,
            self.request.context.microscope_handler.parser,
        )

    def _load_input_stack(self) -> PatternGroupData:
        context = self.request.context
        plan = self.request.execution_plan
        request = self.request
        if not context.microscope_handler:
            raise RuntimeError("MicroscopeHandler not available in context.")

        output_manifest = step_output_manifest(context)
        producer_matching_files = output_manifest.producer_paths_matching_pattern(
            plan,
            str(request.pattern_group_info),
            context.microscope_handler.parser,
        )
        matching_files = list(producer_matching_files)
        source_projection = (
            self.source_workspace_projection_authority().projection_if_available()
        )
        if not matching_files:
            matching_files = context.microscope_handler.path_list_from_pattern(
                str(plan.input_dir),
                request.pattern_group_info,
                context.filemanager,
                Backend.MEMORY.value,
                VariableComponentNames(plan.variable_components).value,
            )
        matching_files = output_manifest.filter_to_producer_paths(
            plan,
            matching_files,
            context.microscope_handler.parser,
        )

        if not matching_files:
            raise ValueError(
                f"No matching files found for pattern group {self.pattern_repr} "
                f"in {plan.input_dir}. "
                f"This indicates either: (1) no image files exist in the directory, "
                f"(2) files don't match the pattern, or (3) pattern parsing failed. "
                f"Check that input files exist and match the expected naming convention."
            )

        matching_files = self._filter_matching_files_for_group(matching_files)

        logger.debug(
            "Pattern %s matched %d files: %s",
            self.pattern_repr,
            len(matching_files),
            [Path(f).name for f in matching_files],
        )

        matching_files.sort()
        logger.debug(
            f"Pattern {self.pattern_repr} sorted files: {[Path(f).name for f in matching_files]}"
        )
        matching_files = self._filter_matching_files_for_source_bindings(matching_files)

        full_file_paths = [
            self._input_memory_path(plan.input_dir, file_path)
            for file_path in matching_files
        ]
        workspace_path_lookups = tuple(
            VirtualWorkspacePathLookup.from_paths(
                virtual_path,
                full_virtual_path,
            )
            for virtual_path, full_virtual_path in zip(
                matching_files,
                full_file_paths,
                strict=True,
            )
        )
        workspace_source_lookups = (
            self._workspace_source_binding_lookups(
                source_projection,
                workspace_path_lookups,
            )
            if source_projection is not None
            else ()
        )
        source_binding_context = SourceBindingRuntimeContextRequest.from_context(
            context=self.request.context,
            plan=self.request.execution_plan,
            matching_files=matching_files,
            source_projection=source_projection,
        ).runtime_context()
        cached_stack = context.runtime_image_stack_cache.get(
            tuple(full_file_paths),
            memory_type=plan.input_memory_type,
        )
        RuntimeProfileSink.record(
            "runtime_stack_cache_get",
            0.0,
            step=plan.step_index,
            step_name=plan.step_name,
            hit=cached_stack is not None,
            paths=len(full_file_paths),
            memory_type=plan.input_memory_type,
        )
        if cached_stack is None:
            raw_slices = context.filemanager.load_batch(
                full_file_paths,
                Backend.MEMORY.value,
            )
            if source_projection is not None or not producer_matching_files:
                raw_slices = self._apply_source_image_loading_semantics(
                    raw_slices,
                    workspace_path_lookups,
                    workspace_source_lookups,
                    source_binding_context,
                    source_projection,
                )

            if not raw_slices:
                raise ValueError(
                    f"No valid images loaded for pattern group {self.pattern_repr} "
                    f"in {plan.input_dir}. "
                    f"Found {len(matching_files)} matching files but failed to load any valid images. "
                    f"This indicates corrupted image files, unsupported formats, or I/O errors. "
                    f"Check file integrity and format compatibility."
                )

            metadata_mode = ImagePayloadMetadataCompositionMode.STACK
            if source_projection is not None and workspace_source_lookups:
                metadata_mode = source_projection.payload_composition_mode(
                    workspace_source_lookups
                )
            if metadata_mode is ImagePayloadMetadataCompositionMode.STACK:
                raw_slice_data = tuple(
                    image_payload_data(slice_data) for slice_data in raw_slices
                )
                main_data_stack = stack_runtime_slices(
                    raw_slice_data,
                    plan.input_memory_type,
                    plan.device_id,
                )
                main_data_stack = stack_image_payload_context(
                    raw_slices,
                    main_data_stack,
                    metadata_mode=metadata_mode,
                )
            elif metadata_mode is ImagePayloadMetadataCompositionMode.BUNDLE:
                main_data_stack = ImagePayloadBundleContext.from_payloads(
                    tuple(raw_slices),
                    metadata_mode=metadata_mode,
                ).compose()
        else:
            main_data_stack = cached_stack.stack

        return PatternGroupData(
            matching_files=matching_files,
            main_data_stack=main_data_stack,
            source_binding_context=source_binding_context,
        )

    def _workspace_source_binding_lookups(
        self,
        source_projection: VirtualWorkspaceSourceProjection,
        lookups: Sequence[VirtualWorkspacePathLookup],
    ) -> tuple[VirtualWorkspacePathLookup, ...]:
        """Return workspace paths owned by this step's exact source bindings."""

        bindings = self.request.source_binding_plan.binding_declarations
        return tuple(
            lookup
            for lookup in lookups
            for projection in (source_projection.source_projection_for(lookup),)
            if projection is not None
            and any(projection.matches_binding(binding) for binding in bindings)
        )

    def _filter_matching_files_for_group(
        self,
        matching_files: list[str],
    ) -> list[str]:
        """Constrain grouped executions to files from the current component."""
        if (
            self.request.execution_plan.main_input_dependency.kind
            is StepInputDependencyKind.STEP_OUTPUT
            or self.request.compiled_group.runtime_domain
            is RuntimeInvocationDomain.ARTIFACT_MANAGED
        ):
            return matching_files

        group_component = self.request.execution_plan.execution_group_value
        component_value = self.request.component_value
        if group_component is None or component_value is None:
            return matching_files

        parser = self.request.context.microscope_handler.parser
        filtered = [
            filename
            for filename in matching_files
            if (metadata := parser.parse_filename(Path(filename).name))
            and str(metadata.get(group_component)) == str(component_value)
        ]
        if not filtered:
            raise ValueError(
                f"Pattern group {self.pattern_repr} for {group_component}="
                f"{component_value!r} matched files, but none carried the "
                f"expected grouped component. Matched files: {matching_files}"
            )
        return filtered

    def _filter_matching_files_for_source_bindings(
        self,
        matching_files: list[str],
    ) -> list[str]:
        """Constrain the loaded main stack to declared image source bindings."""

        if (
            self.request.execution_plan.main_input_dependency.kind
            is StepInputDependencyKind.STEP_OUTPUT
        ):
            return matching_files

        source_binding_plan = self.request.main_flow_source_binding_plan
        if not source_binding_plan.has_primary_content:
            return matching_files
        bindings = tuple(
            binding
            for binding in source_binding_plan.bindings
            if binding.projection_role is SourceProjectionRole.PRIMARY_PLANE
        )
        if not bindings:
            return matching_files
        selector_bindings = SourceBindingCandidateMatcher.selector_bindings(bindings)

        source_context = self._source_binding_candidate_context()
        if (
            not selector_bindings
            and not source_context.source_projections_by_virtual_path
        ):
            return matching_files
        compatible = list(
            SourceBindingMatchedImageSet.from_plan(
                bindings=bindings,
                match_plan=source_binding_plan.match_plan,
                source_context=source_context,
                identity_policy=(self.request.context.source_image_set_identity_policy),
            ).expand(
                matching_files,
                source_universe=self._source_binding_load_universe(),
            )
        )
        if compatible:
            return compatible

        raise ValueError(
            f"Source-bound step {self.request.execution_plan.step_name!r} resolved no files for "
            f"image bindings {[binding.alias for binding in bindings]!r} in pattern "
            f"{self.pattern_repr}. Matched files before source filtering: "
            f"{matching_files!r}."
        )

    def _source_binding_load_universe(self) -> tuple[str, ...]:
        """Return loadable files available for source image-set expansion."""
        source_projection = (
            self.source_workspace_projection_authority().projection_if_available()
        )
        request = SourceBindingRuntimeContextRequest.from_context(
            context=self.request.context,
            plan=self.request.execution_plan,
            matching_files=(),
            source_projection=source_projection,
        )
        return request.runtime_universe_state().require_load_universe().files

    def _source_binding_candidate_context(self) -> SourcePatternResolutionContext:
        projection = self.source_workspace_projection_authority().projection_or_empty()
        return SourcePatternResolutionContext.from_projection(
            parser=self.request.context.microscope_handler.parser,
            projection=self.source_workspace_projection_cache().filtered_by_axis(
                projection,
                axis_id=self.request.execution_plan.axis_id,
            ),
            metadata_rules=self.request.source_binding_plan.metadata_rules,
        )

    def _apply_source_image_loading_semantics(
        self,
        raw_slices: Sequence[RuntimeArrayData],
        workspace_path_lookups: Sequence[VirtualWorkspacePathLookup],
        workspace_source_lookups: Sequence[VirtualWorkspacePathLookup],
        source_binding_context: SourceBindingRuntimeContext,
        source_projection: VirtualWorkspaceSourceProjection | None,
    ) -> list[RuntimeArrayData]:
        if source_projection is not None:
            source_lookups = frozenset(workspace_source_lookups)
            return [
                (
                    self._apply_workspace_source_binding_payload(
                        payload,
                        source_projection=source_projection,
                        lookup=lookup,
                    )
                    if lookup in source_lookups
                    else self._apply_workspace_source_payload(
                        payload,
                        source_projection=source_projection,
                        lookup=lookup,
                    )
                )
                for payload, lookup in zip(
                    raw_slices,
                    workspace_path_lookups,
                    strict=True,
                )
            ]

        source_context = SourcePatternResolutionContext.from_sources(
            parser=self.request.context.microscope_handler.parser,
            source_paths_by_virtual_path=source_binding_context.step_input_source_paths,
            source_metadata_by_path=source_binding_context.source_metadata_by_path,
            metadata_rules=self.request.source_binding_plan.metadata_rules,
        )
        return [
            self._apply_source_binding_payload(
                payload,
                source_metadata=source_context.merged_metadata_for_paths(
                    (
                        lookup.virtual_path,
                        lookup.full_virtual_path,
                    )
                ),
                source_path=source_context.source_path_for(lookup.full_virtual_path),
                read_backend=source_binding_context.step_input_source_backend,
            )
            for payload, lookup in zip(
                raw_slices,
                workspace_path_lookups,
                strict=True,
            )
        ]

    def _apply_workspace_source_payload(
        self,
        payload: RuntimeArrayData,
        *,
        source_projection: VirtualWorkspaceSourceProjection,
        lookup: VirtualWorkspacePathLookup,
    ) -> RuntimeArrayData:
        """Attach workspace-owned source identity without requiring a binding."""

        source_ref = source_projection.source_ref_for(lookup)
        if source_ref is None:
            return payload
        source_context = ImagePayloadSourceMetadataContext(
            SourceImageIdentity(
                lookup.full_virtual_path,
                source_projection.source_metadata_for(lookup),
            ),
            source_ref.backend,
            self.request.context.filemanager,
            source_ref.backend_address,
        )
        metadata = source_context.metadata(payload)
        return source_projection.project_unbound_payload(
            lookup,
            metadata.payload_with(
                image_payload_data(payload),
                image_payload_mask(payload),
            ),
        )

    def _apply_workspace_source_binding_payload(
        self,
        payload: RuntimeArrayData,
        *,
        source_projection: VirtualWorkspaceSourceProjection,
        lookup: VirtualWorkspacePathLookup,
    ) -> RuntimeArrayData:
        projection = source_projection.require_source_projection_for(lookup)
        payload = source_projection.project_payload(lookup, payload)
        return self._apply_source_binding_payload(
            payload,
            source_metadata=source_projection.source_metadata_for(lookup),
            source_path=lookup.full_virtual_path,
            source_address=projection.ref.backend_address,
            read_backend=projection.ref.backend,
        )

    def _apply_source_binding_payload(
        self,
        payload: RuntimeArrayData,
        *,
        source_metadata: Mapping[str, object] | None,
        source_path: str,
        source_address: str | None = None,
        read_backend: str | None,
    ) -> RuntimeArrayData:
        source_context = ImagePayloadSourceMetadataContext(
            SourceImageIdentity(source_path, source_metadata),
            read_backend,
            self.request.context.filemanager,
            source_address,
        )
        source_bindings = self.request.source_binding_plan
        if not source_bindings.binding_declarations:
            metadata = source_context.metadata(payload)
            return metadata.payload_with(
                image_payload_data(payload),
                image_payload_mask(payload),
            )
        alias = (
            None
            if source_metadata is None
            else source_metadata_value(
                source_metadata,
                SOURCE_BINDING_ALIAS_METADATA_FIELD,
            )
        )
        if alias is None:
            raise ValueError(
                f"Source-bound payload {source_path!r} has no declared source alias."
            )
        binding = source_bindings.binding_for_alias(alias)
        if binding is None:
            raise ValueError(
                f"Source-bound payload {source_path!r} declares unknown alias "
                f"{alias!r}."
            )
        return apply_source_binding_payload(payload, binding, source_context)

    def _execute_pattern(
        self,
        loaded: PatternGroupData,
    ) -> RuntimeArrayData | NoMainFlowOutput:
        request = self.request
        runtime_scope = FunctionRuntimeScope.from_pattern_group(request, loaded)
        return runtime_scope.execute_chain(loaded.main_data_stack)

    def _validate_and_unstack(
        self,
        processed_stack: RuntimeArrayData,
        loaded: PatternGroupData,
    ) -> PatternGroupOutputData:
        if (
            isinstance(processed_stack, ImagePayloadMetadataCarrier)
            and processed_stack.metadata.plane_axis is None
        ):
            output_context = self._unwrapped_main_flow_output_context()
            scalar_data = image_payload_data(processed_stack)
            stacked_data = stack_runtime_slices(
                (scalar_data,),
                self.request.execution_plan.output_memory_type,
                self.request.execution_plan.device_id,
            )
            return PatternGroupOutputData(
                slices=(processed_stack,),
                slice_contexts=(
                    (output_context,)
                    if output_context is not None
                    else (
                        self._producer_output_contexts(loaded.matching_files)
                        if len(loaded.matching_files) == 1
                        else ()
                    )
                ),
                stack_payload=stack_image_payload_context(
                    (processed_stack,),
                    stacked_data,
                    metadata_mode=ImagePayloadMetadataCompositionMode.STACK,
                ),
            )
        if isinstance(processed_stack, AlignedImageStack):
            output_payloads = list(
                flatten_aligned_image_payload_slices(processed_stack)
            )
            output_data = tuple(
                image_payload_data(payload) for payload in output_payloads
            )
            stack_payload = None
            if len({tuple(np.shape(value)) for value in output_data}) == 1:
                stacked_data = stack_runtime_slices(
                    output_data,
                    self.request.execution_plan.output_memory_type,
                    self.request.execution_plan.device_id,
                )
                stack_payload = stack_image_payload_context(
                    output_payloads,
                    stacked_data,
                    metadata_mode=ImagePayloadMetadataCompositionMode.STACK,
                )
            return PatternGroupOutputData(
                slices=output_payloads,
                slice_contexts=flatten_aligned_image_slice_contexts(processed_stack),
                stack_payload=stack_payload,
            )
        output_context = self._unwrapped_main_flow_output_context()
        output_projection = RuntimeSliceProjection.preserved_context_for_value(
            processed_stack
        )
        if output_projection is not None:
            unstack_started_at = time.perf_counter()
            output_slices = list(
                RuntimeSliceProjection.value_for_slice(
                    processed_stack,
                    output_projection.selected_plane(slice_index),
                )
                for slice_index in range(output_projection.axis_size)
            )
            RuntimeProfileSink.record(
                "pattern_source_unstack",
                time.perf_counter() - unstack_started_at,
                step=self.request.execution_plan.step_index,
                step_name=self.request.execution_plan.step_name,
                slices=len(output_slices),
            )
            output_payloads = output_slices
        else:
            processed_data = image_payload_data(processed_stack)
            try:
                unstack_started_at = time.perf_counter()
                output_slices = list(
                    unstack_runtime_slices(
                        processed_data,
                        self.request.execution_plan.output_memory_type,
                        self.request.execution_plan.device_id,
                        expected_count=len(loaded.matching_files),
                    )
                )
                RuntimeProfileSink.record(
                    "pattern_source_unstack",
                    time.perf_counter() - unstack_started_at,
                    step=self.request.execution_plan.step_index,
                    step_name=self.request.execution_plan.step_name,
                    slices=len(output_slices),
                )
            except ValueError as exc:
                output_shape = np.shape(processed_data)
                output_ndim = np.ndim(processed_data)
                logger.error("Function output is not an OpenHCS image stack.")
                logger.error(f"Output type: {type(processed_stack)}")
                logger.error("Output shape: %s", output_shape)
                logger.error("Output ndim: %s", output_ndim)
                raise ValueError(
                    "Main processing must result in an image stack shaped "
                    f"(N, H, W) or (N, H, W, C), got "
                    f"{output_shape}"
                ) from exc

            context_started_at = time.perf_counter()
            output_payloads = unstack_image_payload_context(
                processed_stack,
                output_slices,
                default_plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
            )
            RuntimeProfileSink.record(
                "pattern_payload_context_unstack",
                time.perf_counter() - context_started_at,
                step=self.request.execution_plan.step_index,
                step_name=self.request.execution_plan.step_name,
                slices=len(output_payloads),
            )
        slice_contexts = (
            (output_context,) * len(output_payloads)
            if output_context is not None
            else ()
        )
        if not slice_contexts and len(output_payloads) == len(loaded.matching_files):
            slice_contexts = self._producer_output_contexts(loaded.matching_files)
        return PatternGroupOutputData(
            slices=output_payloads,
            slice_contexts=slice_contexts,
            stack_payload=processed_stack,
        )

    def _unwrapped_main_flow_output_context(
        self,
    ) -> AlignedImageSliceContext | None:
        declared_refs = frozenset(
            plan.ref()
            for plan in self.request.compiled_group.resulting_main_flow_output_plans()
        )
        refs = tuple(
            output_plan.ref()
            for output_plan in ComponentArtifactPlans.from_step_component(
                self.request.execution_plan,
                self.request.component_key,
            ).outputs.values()
            if output_plan.ref() in declared_refs
        )
        if not refs:
            return None
        if len(refs) != 1:
            raise ValueError(
                "Multiple named main-flow outputs require AlignedImageStack "
                f"contexts; got {tuple(refs)!r}."
            )
        ref = refs[0]
        return AlignedImageSliceContext.main_flow(
            output_key=ref.name,
            artifact_kind=ref.artifact_type.value,
        )

    def _save_outputs(
        self,
        output_data: PatternGroupOutputData,
        matching_files: list[str],
    ) -> list[ProducedOutputSemantics]:
        context = self.request.context
        output_slices = output_data.slices
        num_outputs = len(output_slices)
        num_inputs = len(matching_files)

        if num_outputs < num_inputs:
            logger.debug(
                "Function returned %d images from %d inputs - likely "
                "flattening operation",
                num_outputs,
                num_inputs,
            )
        elif num_outputs > num_inputs:
            logger.debug(
                "Function returned %s output slices from %s positional input "
                "files; extra slices must carry payload component identity.",
                num_outputs,
                num_inputs,
            )

        output_payloads = []
        output_payload_metadata = []
        output_paths_batch = []
        output_path_entries = []
        output_records = []

        overwritten_output_paths: list[str] = []
        output_directory_exists = context.filemanager.exists(
            str(self.request.execution_plan.output_dir),
            Backend.MEMORY.value,
        )
        for i, img_slice in enumerate(output_slices):
            input_filename = None
            if i < len(matching_files):
                input_filename = matching_files[i]
            output_path_request = FunctionOutputPathRequest(
                parser=context.microscope_handler.parser,
                output_dir=self.request.execution_plan.output_dir,
                output_payload=img_slice,
                input_path=input_filename,
                variable_components=self.request.execution_plan.variable_components,
                input_aligned_output=num_outputs == num_inputs,
                identity_cache=context.runtime_function_output_identity_cache,
            )
            try:
                output_identity = FunctionOutputIdentityAuthority.identity(
                    output_path_request
                )
            except ValueError as exc:
                if input_filename is None:
                    raise ValueError(
                        f"Function returned {num_outputs} output slices but only "
                        f"{num_inputs} input files were available, and output slice "
                        f"{i} does not carry payload component identity."
                    ) from exc
                raise
            output_context = output_data.slice_contexts[i]
            if not output_context.is_anonymous_main_flow:
                output_identity = output_identity.with_filename_qualifier(
                    output_context.output_key
                )
            output_path = FunctionOutputPathAuthority.output_path_for_identity(
                output_path_request,
                output_identity,
            )
            output_path_text = str(output_path)
            output_path_entries.append(
                OutputPathBatchEntry(
                    index=i,
                    input_path=input_filename,
                    output_path=output_path_text,
                    identity=output_identity,
                )
            )
            img_slice = output_context.contextualize_image_payload(img_slice)
            output_metadata = image_payload_metadata(img_slice)
            output_component_metadata = output_identity.component_metadata(
                output_metadata.source_component_metadata,
            )
            if output_metadata.source_component_metadata != output_component_metadata:
                output_metadata = output_metadata.with_source_component_metadata(
                    output_component_metadata
                )
                img_slice = output_metadata.attach_to(img_slice)
            output_record = ProducedOutputSemantics.from_output(
                self.request.execution_plan,
                output_path_text,
                output_identity,
                output_context=output_context,
                image_metadata=output_metadata,
            )

            if output_directory_exists and context.filemanager.exists(
                output_path_text,
                Backend.MEMORY.value,
            ):
                overwritten_output_paths.append(output_path_text)

            output_payloads.append(img_slice)
            output_payload_metadata.append(output_metadata)
            output_paths_batch.append(output_path_text)
            output_records.append(output_record)

        OutputPathBatchUniqueness(
            output_paths=output_paths_batch,
            input_paths=matching_files,
            step_name=self.request.execution_plan.step_name,
            pattern_repr=self.pattern_repr,
            entries=output_path_entries,
        ).validate()

        if overwritten_output_paths:
            for output_path_text in overwritten_output_paths:
                context.filemanager.delete(output_path_text, Backend.MEMORY.value)
            context.runtime_image_stack_cache.discard_paths(
                tuple(overwritten_output_paths)
            )

        context.filemanager.ensure_directory(
            str(self.request.execution_plan.output_dir),
            Backend.MEMORY.value,
        )
        context.filemanager.save_batch(
            output_payloads,
            output_paths_batch,
            Backend.MEMORY.value,
        )
        stack_payload_data = (
            image_payload_data(output_data.stack_payload)
            if output_data.stack_payload is not None
            else None
        )
        if output_data.stack_payload is not None:
            if np.shape(stack_payload_data)[:1] != (len(output_payloads),):
                raise ValueError(
                    "PatternGroupOutputData.stack_payload must match its declared "
                    f"output slice count: stack shape {np.shape(stack_payload_data)!r}, "
                    f"slice count {len(output_payloads)}."
                )
            stack_payload = stack_image_payload_context_from_metadata(
                output_payloads,
                stack_payload_data,
                output_payload_metadata,
                metadata_mode=ImagePayloadMetadataCompositionMode.STACK,
            )
            context.runtime_image_stack_cache.store(
                tuple(output_paths_batch),
                memory_type=self.request.execution_plan.output_memory_type,
                stack=stack_payload,
            )
            RuntimeProfileSink.record(
                "runtime_stack_cache_store",
                0.0,
                step=self.request.execution_plan.step_index,
                step_name=self.request.execution_plan.step_name,
                paths=len(output_paths_batch),
                memory_type=self.request.execution_plan.output_memory_type,
            )
        return output_records

    def _cleanup_collapsed_domains(
        self,
        output_slices: list[RuntimeArrayData],
        matching_files: list[str],
        output_paths: Sequence[str],
    ) -> None:
        context = self.request.context
        num_outputs = len(output_slices)
        num_inputs = len(matching_files)

        if num_outputs >= num_inputs:
            return

        if (
            self.request.execution_plan.input_dir
            == self.request.execution_plan.output_dir
        ):
            return

        retained_paths = {Path(path).as_posix() for path in output_paths}
        retained_paths.update(
            Path(record.output_path).as_posix()
            for record in step_output_manifest(context).produced_records_for(
                self.request.execution_plan
            )
        )
        for j in range(num_outputs, num_inputs):
            unused_filename = matching_files[j]
            unused_relative_path = self._input_relative_path(
                self.request.execution_plan.input_dir,
                unused_filename,
            )
            unused_path = self.request.execution_plan.output_dir / unused_relative_path
            if unused_path.as_posix() in retained_paths:
                continue
            if context.filemanager.exists(
                str(unused_path),
                Backend.MEMORY.value,
            ):
                context.runtime_image_stack_cache.discard_paths((str(unused_path),))
                context.filemanager.delete(
                    str(unused_path),
                    Backend.MEMORY.value,
                )
                logger.debug(
                    "Deleted unused collapsed-domain file after reduced "
                    "output cardinality: %s",
                    unused_path,
                )


def _process_single_pattern_group(request: PatternGroupExecutionRequest) -> None:
    """Process one image pattern group through its assigned callable pattern."""
    PatternGroupRuntime(request).run()
