"""CellProfiler primary-image input policy contracts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from openhcs.core.artifacts import (
    ArtifactSpec,
)
from openhcs.core.runtime_object_labels import ObjectLabelValue
from openhcs.core.steps.function_runtime import (
    RuntimeCallableKwargs,
    RuntimeFunctionOutput,
)
from openhcs.interop.cellprofiler.runtime.invocation import CellProfilerImageRequest


class ObjectLabelDrivenPrimaryImageInputPolicy:
    """Treat declared images as carriers; object labels define the domain."""

    @classmethod
    def primary_image_inputs(
        cls,
        func: Callable[..., RuntimeFunctionOutput],
        declared_inputs: tuple[ArtifactSpec, ...],
    ) -> tuple[ArtifactSpec, ...]:
        del cls, func, declared_inputs
        return ()

    @classmethod
    def invocation_domain_inputs(
        cls,
        func: Callable[..., RuntimeFunctionOutput],
        declared_inputs: tuple[ArtifactSpec, ...],
    ) -> tuple[ArtifactSpec, ...]:
        """Return the exact object-label input that owns one invocation."""

        del func
        binding = cls.primary_image_domain_input_binding()
        parameter_name = binding.require_runtime_parameter_name()
        artifact_type = binding.require_artifact_type()
        domain_inputs = tuple(
            artifact_input
            for artifact_input in declared_inputs
            if artifact_input.parameter_name == parameter_name
            and artifact_input.artifact_type is artifact_type
        )
        if len(domain_inputs) != 1:
            raise ValueError(
                f"{cls.__name__} requires exactly one invocation-domain input "
                f"bound to {parameter_name!r}, got "
                f"{tuple(spec.ref() for spec in domain_inputs)!r}."
            )
        return domain_inputs

    @classmethod
    def project_invocation_image_request(
        cls,
        *,
        image_request: CellProfilerImageRequest,
        runtime_kwargs: RuntimeCallableKwargs,
    ) -> CellProfilerImageRequest:
        """Use the nominally declared label binding as the invocation image domain."""

        domain_binding = cls.primary_image_domain_input_binding()
        parameter_name = domain_binding.require_runtime_parameter_name()
        if parameter_name not in runtime_kwargs:
            raise ValueError(
                f"{cls.__name__} requires bound object-label parameter "
                f"{parameter_name!r} before invocation image projection."
            )
        labels = runtime_kwargs[parameter_name]
        if not isinstance(labels, ObjectLabelValue):
            raise TypeError(
                f"{cls.__name__} object-label image domain requires "
                f"ObjectLabelValue, got {type(labels).__name__}."
            )
        return replace(
            image_request,
            payload=labels.measurement_reference_image(),
            source_image_name=None,
            source_aliases=(),
            image_count=1,
            plane_projection=labels.declared_plane_projection(),
        )
