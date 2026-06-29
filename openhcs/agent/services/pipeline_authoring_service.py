"""Draft pipeline authoring service for OpenHCS agents."""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from dataclasses import replace
from itertools import count
from typing import TypeAlias

from objectstate import get_base_type_for_lazy

from openhcs.agent.dto.common import (
    AgentError,
    AgentWarning,
    JsonObject,
    JsonValue,
    RenderedSource,
    SCHEMA_VERSION,
)
from openhcs.agent.dto.config import ConfigPatch
from openhcs.agent.dto.pipeline import (
    FunctionStepAddRequest,
    FunctionSpecRef,
    FunctionStepSpec,
    PipelineConfigRefs,
    PipelineRef,
    PipelineSpec,
    PipelineValidationResult,
)
from openhcs.agent.exceptions import AgentFacingErrorMixin
from openhcs.agent.services.config_service import coerce_dataclass_patch_values
from openhcs.agent.services.function_catalog_service import (
    FunctionCatalogService,
    PARAMETER_DOCUMENTATION_POLICY,
)
from openhcs.agent.services.source_rendering_service import PythonSourceAssignmentKind
from openhcs.core.function_step_transport import FunctionStepTransportAuthority
from openhcs.core.pipeline.step_config_universe import step_config_classes_by_field_name
from openhcs.core.steps.function_step import FunctionStep


StepConfig: TypeAlias = object
DEFAULT_PIPELINE_CONFIG_REFS = PipelineConfigRefs()
PIPELINE_EMPTY_WARNING = AgentWarning(
    code="pipeline_empty",
    message="Pipeline draft has no FunctionStep entries; it will render as a no-op.",
    hint=(
        "Use openhcs_search_functions to find processing functions, then "
        "openhcs_add_function_step to add at least one step before compile or run."
    ),
)


class PipelineAuthoringError(AgentFacingErrorMixin, ValueError):
    """Base class for pipeline authoring failures intended for agents."""


class UnknownPipelineIdError(PipelineAuthoringError):
    """Raised when an in-memory pipeline draft id is not present."""

    agent_error_code = "unknown_pipeline_id"
    agent_error_hint = (
        "Create a draft with openhcs_create_pipeline in this same MCP session, "
        "then reuse the returned pipeline_id."
    )

    def __init__(self, pipeline_id: str) -> None:
        self.pipeline_id = pipeline_id
        super().__init__(f"Unknown OpenHCS pipeline_id: {pipeline_id}")


class DuplicatePipelineStepIdError(PipelineAuthoringError):
    """Raised when a draft already contains the requested step id."""

    agent_error_code = "duplicate_pipeline_step_id"
    agent_error_hint = (
        "Omit step_id to let OpenHCS generate one, or retry with a unique step_id "
        "for this pipeline draft."
    )

    def __init__(self, step_id: str) -> None:
        self.step_id = step_id
        super().__init__(f"Duplicate step_id in pipeline: {step_id}")


class InvalidFunctionKwargsError(PipelineAuthoringError):
    """Raised when a draft function spec passes kwargs not accepted by a callable."""

    agent_error_code = "invalid_function_kwargs"

    def __init__(
        self,
        function_id: str,
        invalid_kwargs: tuple[str, ...],
        accepted_kwargs: tuple[str, ...],
    ) -> None:
        self.function_id = function_id
        self.invalid_kwargs = invalid_kwargs
        self.accepted_kwargs = accepted_kwargs
        valid_text = ", ".join(accepted_kwargs) if accepted_kwargs else "<none>"
        invalid_text = ", ".join(invalid_kwargs)
        self.agent_error_hint = (
            "Call openhcs_describe_function for this function_id and use only "
            f"accepted kwargs. Accepted kwargs: {valid_text}."
        )
        super().__init__(
            "Invalid kwargs for OpenHCS function "
            f"{function_id}: {invalid_text}."
        )


class MissingFunctionKwargsError(PipelineAuthoringError):
    """Raised when required agent-supplied callable kwargs are omitted."""

    agent_error_code = "missing_function_kwargs"

    def __init__(
        self,
        function_id: str,
        missing_kwargs: tuple[str, ...],
    ) -> None:
        self.function_id = function_id
        self.missing_kwargs = missing_kwargs
        missing_text = ", ".join(missing_kwargs)
        self.agent_error_hint = (
            "Call openhcs_describe_function for this function_id and provide "
            f"required agent kwargs: {missing_text}."
        )
        super().__init__(
            "Missing required kwargs for OpenHCS function "
            f"{function_id}: {missing_text}."
        )


class PipelineAuthoringService:
    """Create and render OpenHCS FunctionStep pipelines from nominal specs."""

    def __init__(
        self,
        function_catalog: FunctionCatalogService | None = None,
    ) -> None:
        self._function_catalog = function_catalog or FunctionCatalogService()
        self._pipelines: dict[str, PipelineSpec] = {}
        self._counter = count(1)
        self._step_counter = count(1)

    def create_pipeline(
        self,
        *,
        steps: Sequence[FunctionStepSpec] = (),
        config_refs: PipelineConfigRefs = DEFAULT_PIPELINE_CONFIG_REFS,
    ) -> PipelineRef:
        pipeline_id = f"pipeline-{next(self._counter)}"
        spec = PipelineSpec(
            schema_version=SCHEMA_VERSION,
            pipeline_id=pipeline_id,
            steps=tuple(steps),
            config_refs=config_refs,
        )
        self._pipelines[pipeline_id] = spec
        return PipelineRef(
            pipeline_id=pipeline_id,
            uri=f"openhcs://pipelines/{pipeline_id}",
        )

    def make_step_spec(
        self,
        *,
        function_id: str,
        name: str | None = None,
        kwargs: JsonObject | None = None,
        step_id: str | None = None,
        description: str | None = None,
        enabled: bool = True,
        debug_pause: bool = False,
        step_config_overrides: Mapping[str, JsonObject] | None = None,
    ) -> FunctionStepSpec:
        if kwargs is None:
            function_kwargs = {}
        else:
            function_kwargs = dict(kwargs)
        function = FunctionSpecRef(
            function_id=function_id,
            kwargs=function_kwargs,
        )
        detail = self._function_catalog.get(function_id)
        return FunctionStepSpec(
            step_id=step_id or f"step-{next(self._step_counter)}",
            name=name or detail.entry.name,
            functions=(function,),
            description=description,
            enabled=enabled,
            debug_pause=debug_pause,
            step_config_overrides=_step_config_patches(step_config_overrides),
        )

    def get_pipeline(self, pipeline_ref: PipelineRef | str) -> PipelineSpec:
        pipeline_id = _pipeline_id(pipeline_ref)
        try:
            return self._pipelines[pipeline_id]
        except KeyError as exc:
            raise UnknownPipelineIdError(pipeline_id) from exc

    def add_step(
        self,
        pipeline_ref: PipelineRef | str,
        step_spec: FunctionStepSpec,
        *,
        index: int | None = None,
    ) -> PipelineSpec:
        current = self.get_pipeline(pipeline_ref)
        steps = list(current.steps)
        if any(step.step_id == step_spec.step_id for step in steps):
            raise DuplicatePipelineStepIdError(step_spec.step_id)
        if index is None:
            steps.append(step_spec)
        else:
            steps.insert(index, step_spec)
        updated = replace(current, steps=tuple(steps))
        self._pipelines[updated.pipeline_id] = updated
        return updated

    def add_function_step_from_request(
        self,
        request: FunctionStepAddRequest,
    ) -> PipelineSpec:
        step_spec = self.make_step_spec(
            function_id=request.function_id,
            name=request.name,
            kwargs=request.kwargs,
            step_id=request.step_id,
            description=request.description,
            enabled=request.enabled,
            debug_pause=request.debug_pause,
            step_config_overrides=request.step_config_overrides,
        )
        return self.add_step(
            request.pipeline_id,
            step_spec,
            index=request.index,
        )

    def validate(self, pipeline_ref: PipelineRef | str) -> PipelineValidationResult:
        ref = self._ref(pipeline_ref)
        spec = self.get_pipeline(ref)
        warnings = (PIPELINE_EMPTY_WARNING,) if not spec.steps else ()
        try:
            self.to_function_steps(ref)
        except Exception as exc:
            return PipelineValidationResult(
                schema_version=SCHEMA_VERSION,
                valid=False,
                pipeline_ref=ref,
                errors=(_pipeline_validation_error(exc),),
                warnings=warnings,
            )
        return PipelineValidationResult(
            schema_version=SCHEMA_VERSION,
            valid=True,
            pipeline_ref=ref,
            warnings=warnings,
        )

    def to_function_steps(self, pipeline_ref: PipelineRef | str) -> list[FunctionStep]:
        spec = self.get_pipeline(pipeline_ref)
        steps = [self._to_function_step(step_spec) for step_spec in spec.steps]
        return FunctionStepTransportAuthority.normalize_pipeline(steps)

    def render_source(
        self,
        pipeline_ref: PipelineRef | str,
        *,
        clean: bool = True,
    ) -> RenderedSource:
        steps = self.to_function_steps(pipeline_ref)
        return RenderedSource(
            schema_version=SCHEMA_VERSION,
            title=f"{_pipeline_id(pipeline_ref)} source",
            source=PythonSourceAssignmentKind.PIPELINE_STEPS.assignment(steps, clean).render(),
        )

    def _to_function_step(self, step_spec: FunctionStepSpec) -> FunctionStep:
        if not step_spec.functions:
            raise ValueError(f"FunctionStepSpec {step_spec.step_id!r} has no functions")

        step_kwargs = {
            "name": step_spec.name,
            "description": step_spec.description,
            "enabled": step_spec.enabled,
            "debug_pause": step_spec.debug_pause,
        }
        step_kwargs.update(_step_config_kwargs(step_spec))
        return FunctionStep(
            func=self._function_spec(step_spec.functions),
            **step_kwargs,
        )

    def _function_spec(self, refs: tuple[FunctionSpecRef, ...]):
        specs = [self._function_spec_item(ref) for ref in refs]
        if len(specs) == 1:
            return specs[0]
        return specs

    def _function_spec_item(self, ref: FunctionSpecRef):
        if ref.runtime_options:
            raise ValueError(
                "runtime_options are not accepted by the v1 agent API until "
                "nominal RuntimeInvocationOptions DTOs are added."
            )
        func = self._function_catalog.resolve(ref.function_id)
        kwargs = dict(ref.kwargs)
        _validate_callable_kwargs(ref.function_id, func, kwargs)
        if not kwargs:
            return func
        return (func, kwargs)

    def _ref(self, pipeline_ref: PipelineRef | str) -> PipelineRef:
        pipeline_id = _pipeline_id(pipeline_ref)
        self.get_pipeline(pipeline_id)
        return PipelineRef(
            pipeline_id=pipeline_id,
            uri=f"openhcs://pipelines/{pipeline_id}",
        )


def _pipeline_id(pipeline_ref: PipelineRef | str) -> str:
    return pipeline_ref.pipeline_id if isinstance(pipeline_ref, PipelineRef) else pipeline_ref


def _validate_callable_kwargs(
    function_id: str,
    func,
    kwargs: Mapping[str, JsonValue],
) -> None:
    signature = inspect.signature(func)
    accepted_kwargs = PARAMETER_DOCUMENTATION_POLICY.agent_parameter_names(func)
    invalid_kwargs = tuple(
        kwarg
        for kwarg in kwargs
        if kwarg not in accepted_kwargs
    )
    if invalid_kwargs:
        raise InvalidFunctionKwargsError(
            function_id,
            invalid_kwargs=invalid_kwargs,
            accepted_kwargs=accepted_kwargs,
        )
    missing_kwargs = tuple(
        parameter.name
        for parameter in PARAMETER_DOCUMENTATION_POLICY.parameter_specs(func)
        if parameter.required and parameter.name not in kwargs
    )
    if missing_kwargs:
        raise MissingFunctionKwargsError(
            function_id,
            missing_kwargs=missing_kwargs,
        )
    try:
        signature.bind_partial(**kwargs)
    except TypeError:
        raise


def _pipeline_validation_error(exception: Exception) -> AgentError:
    if isinstance(exception, AgentFacingErrorMixin):
        return exception.to_agent_error()
    return AgentError.from_exception("pipeline_invalid", exception)


def _step_config_kwargs(step_spec: FunctionStepSpec) -> dict[str, StepConfig]:
    return {
        kwarg: _instantiate_config_patch(patch)
        for kwarg, patch in step_spec.step_config_overrides.items()
    }


def _step_config_patches(
    values_by_field_name: Mapping[str, JsonObject] | None,
) -> dict[str, ConfigPatch]:
    if values_by_field_name is None:
        return {}
    class_by_field_name = _step_config_class_by_field_name()
    unknown = tuple(
        field_name
        for field_name in values_by_field_name
        if field_name not in class_by_field_name
    )
    if unknown:
        valid_fields = ", ".join(sorted(class_by_field_name))
        raise ValueError(
            "Unknown OpenHCS step config override field(s): "
            f"{', '.join(unknown)}. Valid fields: {valid_fields}."
        )
    return {
        field_name: ConfigPatch(
            config_type=class_by_field_name[field_name].__name__,
            values=dict(values),
        )
        for field_name, values in values_by_field_name.items()
    }


def _step_config_class_by_field_name() -> dict[str, type[StepConfig]]:
    return dict(step_config_classes_by_field_name())


def _step_config_class_by_config_type() -> dict[str, type[StepConfig]]:
    return {
        config_class.__name__: config_class
        for config_class in _step_config_class_by_field_name().values()
    }


def _step_config_class_from_patch(patch: ConfigPatch) -> type[StepConfig]:
    class_by_config_type = _step_config_class_by_config_type()
    try:
        return class_by_config_type[patch.config_type]
    except KeyError as exc:
        valid_types = ", ".join(sorted(class_by_config_type))
        raise ValueError(
            "Unknown OpenHCS step config type: "
            f"{patch.config_type}. Valid types: {valid_types}."
        ) from exc


def _instantiate_config_patch(patch: ConfigPatch) -> StepConfig:
    config_class = _step_config_class_from_patch(patch)
    coercion_class = get_base_type_for_lazy(config_class) or config_class
    values = coerce_dataclass_patch_values(coercion_class, patch.values)
    return config_class(**values)
