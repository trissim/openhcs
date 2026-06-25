"""CellProfiler measurement-image and object-label input resolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
import time

from openhcs.core.aligned_image_payload import ImagePayloadExecutionMode
from openhcs.core.artifacts import ArtifactKind, ArtifactSpec, ArtifactSpecCollection
from openhcs.interop.cellprofiler.runtime.adapter import CellProfilerRuntimeAdapter
from openhcs.interop.cellprofiler.runtime.artifact_binding import cellprofiler_image_payload
from openhcs.interop.cellprofiler.runtime.invocation import (
    CellProfilerImageRequest,
    CellProfilerMeasurementImage,
    CellProfilerMeasurementImageDomain,
)
from openhcs.interop.cellprofiler.runtime.measurement_execution_support import (
    object_measurement_runtime_inputs,
)
from openhcs.interop.cellprofiler.runtime.measurement_source_names import (
    measurement_source_name_for_specs,
)
from openhcs.interop.cellprofiler.runtime.object_measurement_execution import (
    CellProfilerPerObjectMeasurementPolicy,
)
from openhcs.interop.cellprofiler.runtime.object_only_reference_image import (
    OBJECT_ONLY_REFERENCE_IMAGE,
)
from openhcs.interop.cellprofiler.runtime.payload_types import (
    CellProfilerFunction,
    CellProfilerRuntimeValue,
)
from openhcs.interop.cellprofiler.runtime.runtime_profile import (
    CellProfilerRuntimeProfileEvent,
)

if TYPE_CHECKING:
    from openhcs.interop.cellprofiler.runtime.module_execution import (
        CellProfilerModuleExecutor,
    )


@dataclass(frozen=True, slots=True)
class CellProfilerMeasurementImageResolver:
    """Resolve CellProfiler measurement images and object-label payloads."""

    executor: CellProfilerModuleExecutor

    def measurement_image_inputs(
        self,
        func: CellProfilerFunction,
        adapter: CellProfilerRuntimeAdapter,
        current_image: CellProfilerRuntimeValue,
        image_request: "CellProfilerImageRequest | None",
    ) -> tuple["CellProfilerMeasurementImage", ...]:
        executor = self.executor
        image_inputs = self.primary_image_inputs(func)
        if not image_inputs:
            return (
                self.measurement_carrier_image(
                    current_image,
                    reference_domain=CellProfilerMeasurementImageDomain.OBJECT_LABELS,
                ),
            )

        if not CellProfilerPerObjectMeasurementPolicy.measures_images_independently(
            executor.module_name
        ):
            if image_request is None:
                raise ValueError(
                    f"{executor.module_name} requires a composed measurement image "
                    "request."
                )
            return (self.composed_measurement_image(image_request, image_inputs),)

        if image_request is not None:
            projected_images = self.measurement_images_from_image_request(
                image_request,
                image_inputs,
                reference_domain=CellProfilerMeasurementImageDomain.OBJECT_LABELS,
            )
            if projected_images is not None:
                return projected_images

        return self.resolved_measurement_images(
            func,
            image_inputs,
            adapter,
            current_image,
            reference_domain=CellProfilerMeasurementImageDomain.OBJECT_LABELS,
        )

    def independent_measurement_image_inputs(
        self,
        func: CellProfilerFunction,
        adapter: CellProfilerRuntimeAdapter,
        current_image: CellProfilerRuntimeValue,
    ) -> tuple["CellProfilerMeasurementImage", ...]:
        image_inputs = self.primary_image_inputs(func)
        if not image_inputs:
            return (
                self.measurement_carrier_image(
                    current_image,
                    reference_domain=CellProfilerMeasurementImageDomain.SOURCE_IMAGE,
                ),
            )

        return self.resolved_measurement_images(
            func,
            image_inputs,
            adapter,
            current_image,
        )

    def primary_image_inputs(
        self,
        func: CellProfilerFunction,
    ) -> tuple[ArtifactSpec, ...]:
        """Return declared primary image inputs for this measurement module."""
        return self.executor.runtime_plan(func).primary_image_inputs

    def measurement_carrier_image(
        self,
        current_image: CellProfilerRuntimeValue,
        *,
        reference_domain: "CellProfilerMeasurementImageDomain",
    ) -> "CellProfilerMeasurementImage":
        return CellProfilerMeasurementImage(
            source_image_name=None,
            source_aliases=(),
            payload=OBJECT_ONLY_REFERENCE_IMAGE.reference_image(current_image),
            reference_domain=reference_domain,
        )

    def composed_measurement_image(
        self,
        image_request: "CellProfilerImageRequest",
        image_inputs: tuple[ArtifactSpec, ...],
    ) -> "CellProfilerMeasurementImage":
        return CellProfilerMeasurementImage(
            source_image_name=measurement_source_name_for_specs(image_inputs),
            source_aliases=ArtifactSpecCollection(image_inputs).names(),
            payload=image_request.payload,
            align_to_labels=False,
            execution_mode=image_request.execution_mode,
        )

    def resolved_measurement_images(
        self,
        func: CellProfilerFunction,
        image_inputs: tuple[ArtifactSpec, ...],
        adapter: CellProfilerRuntimeAdapter,
        current_image: CellProfilerRuntimeValue,
        *,
        reference_domain: "CellProfilerMeasurementImageDomain | None" = None,
    ) -> tuple["CellProfilerMeasurementImage", ...]:
        if reference_domain is None:
            reference_domain = CellProfilerMeasurementImageDomain.SOURCE_IMAGE
        plan = self.executor.runtime_plan(func)
        runtime_image_names = plan.runtime_image_name_set
        source_aliases = plan.primary_image_source_aliases
        resolved_images: list[CellProfilerMeasurementImage] = []
        for spec in image_inputs:
            resolved_images.append(
                self.resolved_measurement_image(
                    spec,
                    adapter,
                    current_image,
                    runtime_image_names,
                    source_aliases,
                    reference_domain=reference_domain,
                )
            )
        return tuple(resolved_images)

    def measurement_images_from_image_request(
        self,
        image_request: CellProfilerImageRequest,
        image_inputs: tuple[ArtifactSpec, ...],
        *,
        reference_domain: "CellProfilerMeasurementImageDomain",
    ) -> tuple["CellProfilerMeasurementImage", ...] | None:
        """Return per-source measurement images from the already resolved image request."""
        source_aliases = ArtifactSpecCollection(image_inputs).names()
        payloads_by_name = image_request.source_payloads_for_names(source_aliases)
        if payloads_by_name is None:
            return None
        return tuple(
            CellProfilerMeasurementImage(
                source_image_name=spec.name,
                source_aliases=source_aliases,
                payload=payloads_by_name[spec.name],
                reference_domain=reference_domain,
            )
            for spec in image_inputs
        )

    def resolved_measurement_image(
        self,
        spec: ArtifactSpec,
        adapter: CellProfilerRuntimeAdapter,
        current_image: CellProfilerRuntimeValue,
        runtime_image_names: frozenset[str],
        source_aliases: tuple[str, ...],
        *,
        reference_domain: "CellProfilerMeasurementImageDomain",
    ) -> "CellProfilerMeasurementImage":
        if spec.name in runtime_image_names:
            runtime_image = adapter.get_image(spec.name)
            return CellProfilerMeasurementImage(
                source_image_name=spec.name,
                source_aliases=source_aliases,
                payload=cellprofiler_image_payload(runtime_image.data),
                reference_domain=reference_domain,
            )
        return CellProfilerMeasurementImage(
            source_image_name=spec.name,
            source_aliases=source_aliases,
            payload=cellprofiler_image_payload(
                adapter.resolve_source_image(spec.name, current_image)
            ),
            reference_domain=reference_domain,
        )

    def object_label_payload(
        self,
        spec: ArtifactSpec,
        adapter: CellProfilerRuntimeAdapter,
        measurement_image: CellProfilerRuntimeValue,
    ) -> CellProfilerRuntimeValue:
        if spec.name in self.executor.contract.external_input_names(
            ArtifactKind.OBJECT_LABELS
        ):
            return adapter.resolve_source_objects(
                spec.name,
                measurement_image,
            )
        return adapter.get_objects(spec.name)

    def object_measurement_runtime_inputs(
        self,
        *,
        func: CellProfilerFunction,
        measurement_image: "CellProfilerMeasurementImage",
        object_spec: ArtifactSpec,
        adapter: CellProfilerRuntimeAdapter,
        current_image: CellProfilerRuntimeValue,
    ) -> tuple[
        CellProfilerRuntimeValue,
        CellProfilerRuntimeValue,
        CellProfilerRuntimeValue,
        ImagePayloadExecutionMode,
        tuple["CellProfilerRuntimeProfileEvent", ...],
        float,
        float,
    ]:
        """Prepare image, labels, payload, and execution mode for object measurement."""
        return object_measurement_runtime_inputs(
            module_name=self.executor.module_name,
            func=func,
            measurement_image=measurement_image,
            object_spec=object_spec,
            label_payload=self.object_label_payload(
                object_spec,
                adapter,
                measurement_image.payload,
            ),
            adapter=adapter,
            current_image=current_image,
        )
