"""Test construction helpers for the canonical CellProfiler runtime boundary."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from openhcs.constants.constants import AllComponents, Backend
from openhcs.core.artifacts import (
    ArtifactInputPlan,
    ArtifactInputProjectionPlan,
    ArtifactOutputPlan,
    ArtifactSpec,
)
from openhcs.core.callable_contract import CallableContract, CallableMetadata
from openhcs.core.component_group_scope import ComponentGroupScope
from openhcs.core.function_patterns import (
    DEFAULT_GROUP_KEY,
    FunctionInvocationKey,
    InvocationArtifactInputEdgePlan,
    InvocationArtifactInputProjectionKey,
)
from openhcs.core.runtime_adapters import RuntimeAdapterRequest
from openhcs.core.runtime_plane_projection import RuntimePlaneProjection
from openhcs.core.runtime_stores import RuntimeValueStore
from openhcs.core.runtime_measurements import MeasurementTable
from openhcs.core.source_matching import SourceImageSetIdentityPolicy
from openhcs.core.steps.function_output_identity import FunctionOutputIdentityCache
from openhcs.interop.cellprofiler.runtime.adapter import CellProfilerRuntimeAdapter
from openhcs.interop.cellprofiler.runtime.object_measurement_tables import (
    ObjectMeasurementTableIndex,
)
from openhcs.microscopes.imagexpress import ImageXpressFilenameParser


@dataclass(slots=True)
class CellProfilerRuntimeTestContext:
    """Minimal test implementation of the runtime services owned by ProcessingContext."""

    runtime_value_store: RuntimeValueStore
    filemanager: object | None
    microscope_handler: object
    source_image_set_identity_policy: SourceImageSetIdentityPolicy
    runtime_function_output_identity_cache: FunctionOutputIdentityCache = field(
        default_factory=FunctionOutputIdentityCache
    )


def cellprofiler_runtime_input_edge_for_test(
    storage_plan: ArtifactInputPlan,
    *,
    spec: ArtifactSpec | None = None,
    input_index: int = 0,
    invocation_scope: ComponentGroupScope,
    producer_selection_scope: ComponentGroupScope,
    component_scopes: tuple[ComponentGroupScope, ...],
    consumer_variable_components: tuple[AllComponents, ...],
) -> InvocationArtifactInputEdgePlan:
    """Build one exact compiled input edge without deriving projection semantics."""

    invocation_key = FunctionInvocationKey(
        "cellprofiler_runtime_adapter_test",
        DEFAULT_GROUP_KEY,
        0,
    )
    return InvocationArtifactInputEdgePlan(
        key=InvocationArtifactInputProjectionKey(
            invocation_key=invocation_key,
            input_index=input_index,
        ),
        spec=(
            spec
            if spec is not None
            else ArtifactSpec.input(
                storage_plan.name,
                storage_plan.artifact_type,
                sidecar_role=storage_plan.sidecar_role,
            )
        ),
        storage_plan=storage_plan,
        projection=ArtifactInputProjectionPlan(
            invocation_scope=invocation_scope,
            producer_selection_scope=producer_selection_scope,
            component_scopes=component_scopes,
            consumer_variable_components=consumer_variable_components,
        ),
    )


def cellprofiler_runtime_adapter_for_test(
    *,
    runtime_value_store: RuntimeValueStore,
    microscope_handler: object | None = None,
    filemanager: object | None = None,
    backend: str = Backend.MEMORY.value,
    **request_fields: Any,
) -> CellProfilerRuntimeAdapter:
    """Build the adapter through the same nominal request used in production."""

    return CellProfilerRuntimeAdapter(
        request=runtime_adapter_request_for_test(
            runtime_value_store=runtime_value_store,
            microscope_handler=microscope_handler,
            filemanager=filemanager,
            **request_fields,
        ),
        backend=backend,
    )


def object_measurement_tables_for_test(
    adapter: CellProfilerRuntimeAdapter,
    object_name: str,
    *,
    group_key: str | None = None,
    match_group: bool = True,
) -> tuple[MeasurementTable, ...]:
    """Query object measurement tables through their nominal index owner."""

    index = ObjectMeasurementTableIndex.from_tables(
        adapter.measurement_tables(
            group_key=group_key,
            match_group=match_group,
        )
    )
    tables = index.for_object(object_name)
    return () if tables is None else tables


def runtime_adapter_request_for_test(
    *,
    runtime_value_store: RuntimeValueStore,
    microscope_handler: object | None = None,
    filemanager: object | None = None,
    source_image_set_identity_policy: SourceImageSetIdentityPolicy = (
        SourceImageSetIdentityPolicy()
    ),
    artifact_output_bindings: Iterable[
        tuple[ArtifactSpec, ArtifactOutputPlan]
    ] = (),
    **request_fields: Any,
) -> RuntimeAdapterRequest:
    """Build the canonical runtime request with test-owned context services."""

    context = CellProfilerRuntimeTestContext(
        runtime_value_store=runtime_value_store,
        filemanager=filemanager,
        microscope_handler=(
            microscope_handler
            if microscope_handler is not None
            else SimpleNamespace(parser=ImageXpressFilenameParser())
        ),
        source_image_set_identity_policy=source_image_set_identity_policy,
    )
    output_bindings = tuple(artifact_output_bindings)
    supplied_output_plans = request_fields.get("artifact_outputs", {})
    supplied_contract = request_fields.get("callable_contract")
    if output_bindings:
        if supplied_output_plans:
            raise ValueError(
                "Test runtime requests accept artifact_output_bindings or "
                "artifact_outputs, not both."
            )
        output_plans = {}
        output_specs = []
        for spec, plan in output_bindings:
            if not isinstance(spec, ArtifactSpec) or not isinstance(
                plan, ArtifactOutputPlan
            ):
                raise TypeError(
                    "artifact_output_bindings must contain "
                    "(ArtifactSpec, ArtifactOutputPlan) pairs."
                )
            if spec.ref() != plan.ref():
                raise ValueError(
                    f"Test output declaration {spec.ref()!r} conflicts with "
                    f"compiled plan {plan.ref()!r}."
                )
            previous = output_plans.get(spec.ref())
            if previous is not None and previous != plan:
                raise ValueError(
                    f"Conflicting test output plans for {spec.ref()!r}."
                )
            output_specs.append(spec)
            output_plans[spec.ref()] = plan
        request_fields["artifact_outputs"] = output_plans
        if supplied_contract is None:
            request_fields["callable_contract"] = CallableContract(
                func=lambda image: image,
                function_name="cellprofiler_runtime_adapter_test",
                module_name=__name__,
                metadata=CallableMetadata(artifact_outputs=tuple(output_specs)),
            )
        else:
            declared_outputs = supplied_contract.artifact_outputs
            for spec in output_specs:
                if declared_outputs.by_ref(spec.ref()) != spec:
                    raise ValueError(
                        f"Test output binding {spec.ref()!r} is not declared "
                        "exactly by callable_contract."
                    )
    elif supplied_output_plans and supplied_contract is None:
        raise ValueError(
            "Nonempty test artifact_outputs require an explicit callable_contract "
            "or artifact_output_bindings."
        )
    return RuntimeAdapterRequest(
        context=context,
        plane_projection=request_fields.pop(
            "plane_projection",
            RuntimePlaneProjection.stack(1),
        ),
        **request_fields,
    )
