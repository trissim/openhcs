"""Typed ObjectState binding for the Pipeline declaration."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Self

from openhcs.config_framework.object_state import ObjectState, ObjectStateRegistry
from openhcs.core.function_patterns import (
    COMPILE_TIME_FUNCTION_KWARGS_KEY,
    CompileTimeFunctionKwarg,
)
from openhcs.core.pipeline import Pipeline
from openhcs.core.steps.function_step import FunctionSpec, FunctionStep
from openhcs.pyqt_gui.services.plate_manager_root_state import (
    root_orchestrator_scope_ids,
)
from openhcs.pyqt_gui.services.plate_scope_identity import (
    PipelineScopeIdentity,
    PlateScopeIdentity,
)
from openhcs.pyqt_gui.services.step_scope_identity import (
    FunctionStepScopeToken,
    SCOPE_SEGMENT_SEPARATOR,
)
from pyqt_reactive.services.function_pattern_code_document import (
    FunctionPatternCodeDocumentService,
)
from pyqt_reactive.services.pattern_data_manager import (
    FUNC_EDITOR_PATTERN_TOKENS_META_KEY,
    PatternDataManager,
)
from pyqt_reactive.services.scope_token_service import ScopeTokenService


PipelineFunctionPattern = FunctionSpec | dict[str, "PipelineFunctionPattern"] | None
FunctionPatternTokenTree = list[str] | dict[str, "FunctionPatternTokenTree"] | None


@dataclass(frozen=True, slots=True)
class PipelineObjectStateBinding:
    """ObjectState interface for a Pipeline's declared step_scope_ids field."""

    state: ObjectState

    def __post_init__(self) -> None:
        if not isinstance(self.state.object_instance, Pipeline):
            raise TypeError(
                "PipelineObjectStateBinding requires an ObjectState backed by Pipeline."
            )

    @property
    def pipeline(self) -> Pipeline:
        """Return the backing Pipeline declaration."""

        return self.state.object_instance

    @property
    def plate_scope(self) -> str:
        """Return the logical plate scope owning this Pipeline ObjectState."""

        return PipelineScopeIdentity.from_scope_id(self.state.scope_id).plate_scope

    @property
    def step_scope_ids(self) -> tuple[str, ...]:
        """Return the pipeline's declared step ObjectState scopes."""

        return tuple(self.state.parameters.get("step_scope_ids", []))

    @classmethod
    def for_plate(
        cls,
        plate_path: str,
        *,
        register: bool = True,
    ) -> Self | None:
        """Return the Pipeline ObjectState binding for one plate scope."""

        if not plate_path:
            return None

        pipeline_scope = PipelineScopeIdentity.from_plate_scope(plate_path).scope_id
        state = ObjectStateRegistry.get_by_scope(pipeline_scope)
        if state is None:
            identity = PlateScopeIdentity.from_scope_id(plate_path)
            state = ObjectState(
                object_instance=Pipeline(
                    name=identity.display_name,
                    step_scope_ids=[],
                ),
                scope_id=pipeline_scope,
                parent_state=ObjectStateRegistry.get_by_scope(plate_path),
            )
            if register:
                ObjectStateRegistry.register(state, _skip_snapshot=True)
        return cls(state)

    @classmethod
    def steps_for_plate(cls, plate_path: str) -> list[FunctionStep]:
        """Return FunctionStep declarations derived from a Pipeline ObjectState."""

        binding = cls.for_plate(plate_path)
        if binding is None:
            return []
        return binding.steps()

    @classmethod
    def update_plate_steps(
        cls,
        plate_path: str,
        steps: list[FunctionStep],
    ) -> None:
        """Replace one plate's Pipeline ObjectState step list."""

        binding = cls.for_plate(plate_path, register=False)
        if binding is None:
            return
        binding.replace_steps(steps)

    @classmethod
    def registered_plate_pipelines(cls) -> dict[str, list[FunctionStep]]:
        """Return all visible plate pipeline declarations from ObjectState."""

        root_state = ObjectStateRegistry.get_by_scope("__plates__")
        if root_state is None:
            return {}

        result: dict[str, list[FunctionStep]] = {}
        for plate_path in root_orchestrator_scope_ids(root_state):
            result[plate_path] = cls.steps_for_plate(plate_path)
        return result

    def steps(self) -> list[FunctionStep]:
        """Return FunctionStep declarations derived from this Pipeline ObjectState."""

        steps: list[FunctionStep] = []
        for scope_id in self.step_scope_ids:
            step_state = ObjectStateRegistry.get_by_scope(scope_id)
            if step_state is not None:
                steps.append(self.step_from_state(step_state))
        return steps

    @staticmethod
    def step_from_state(step_state: ObjectState) -> FunctionStep:
        """Return a FunctionStep with function child ObjectState values applied."""

        step = step_state.to_object()
        return step.with_function_spec(
            PipelineObjectStateBinding.function_pattern_from_child_states(
                step_state.scope_id,
                step.func,
                step_state.metadata.get(FUNC_EDITOR_PATTERN_TOKENS_META_KEY),
            )
        )

    def replace_steps(self, steps: list[FunctionStep]) -> None:
        """Replace this Pipeline's declared step ObjectState scopes."""

        existing_step_scope_ids = self.step_scope_ids
        self.transfer_existing_step_scope_tokens(
            self.plate_scope,
            existing_step_scope_ids,
            steps,
        )
        ScopeTokenService.seed_from_objects(self.plate_scope, steps)

        step_scope_ids: list[str] = []
        to_register: list[ObjectState] = []
        parent_state = ObjectStateRegistry.get_by_scope(self.plate_scope)
        for step in steps:
            scope_id = ScopeTokenService.build_scope_id(self.plate_scope, step)
            step_scope_ids.append(scope_id)
            _step_state, states = self.collect_step_registration_states(
                step=step,
                scope_id=scope_id,
                parent_state=parent_state,
            )
            to_register.extend(states)

        if ObjectStateRegistry.get_by_scope(self.state.scope_id) is None:
            ObjectStateRegistry.register(self.state)
        for state in to_register:
            ObjectStateRegistry.register(state)

        removed_step_scope_ids = set(existing_step_scope_ids).difference(
            step_scope_ids
        )
        for removed_scope_id in removed_step_scope_ids:
            ObjectStateRegistry.unregister_scope_and_descendants(
                removed_scope_id,
                _skip_snapshot=True,
            )
        self.state.update_parameter("step_scope_ids", step_scope_ids)

    @classmethod
    def register_step_state(cls, plate_path: str, step: FunctionStep) -> None:
        """Register ObjectState for one step and its function-pattern children."""

        binding = cls.for_plate(plate_path)
        if binding is None:
            return
        scope_id = ScopeTokenService.build_scope_id(plate_path, step)
        _step_state, to_register = binding.collect_step_registration_states(
            step=step,
            scope_id=scope_id,
            parent_state=ObjectStateRegistry.get_by_scope(plate_path),
        )
        for state in to_register:
            ObjectStateRegistry.register(state)

    def transfer_existing_step_scope_tokens(
        self,
        plate_path: str,
        existing_step_scope_ids: tuple[str, ...],
        steps: list[FunctionStep],
    ) -> None:
        """Reuse same-position step scope tokens for replacement updates."""

        for existing_scope_id, replacement_step in zip(existing_step_scope_ids, steps):
            if ScopeTokenService.object_token(replacement_step) is not None:
                continue
            token = FunctionStepScopeToken.from_segment(
                existing_scope_id.rsplit(SCOPE_SEGMENT_SEPARATOR, 1)[-1]
            )
            if token is None:
                continue
            ScopeTokenService.adopt_token(
                plate_path,
                replacement_step,
                token.raw,
            )

    def collect_step_registration_states(
        self,
        *,
        step: FunctionStep,
        scope_id: str,
        parent_state: ObjectState | None,
    ) -> tuple[ObjectState, list[ObjectState]]:
        """Build missing ObjectStates for one step and its function pattern."""

        step_state = ObjectStateRegistry.get_by_scope(scope_id)
        to_register: list[ObjectState] = []
        if step_state is None:
            step_state = ObjectState(
                object_instance=step,
                scope_id=scope_id,
                parent_state=parent_state,
            )
            to_register.append(step_state)
        else:
            step_state.update_object_instance(step)

        step_state.metadata[FUNC_EDITOR_PATTERN_TOKENS_META_KEY] = (
            self.scope_tokens_for_function_pattern(scope_id, step.func)
        )

        for func_obj, kwargs in self.normalize_func_items(step.func):
            func_scope_id = ScopeTokenService.build_scope_id(scope_id, func_obj)
            if ObjectStateRegistry.get_by_scope(func_scope_id) is not None:
                continue
            exclude_params = FunctionPatternCodeDocumentService.reserved_parameter_names(
                func_obj
            )
            to_register.append(
                ObjectState(
                    object_instance=func_obj,
                    scope_id=func_scope_id,
                    parent_state=step_state,
                    exclude_params=exclude_params,
                    initial_values=CompileTimeFunctionKwarg.strip_from_mapping(
                        kwargs
                    ),
                )
            )

        return step_state, to_register

    @classmethod
    def normalize_func_items(
        cls,
        func_value: PipelineFunctionPattern,
    ) -> list[tuple[Callable, dict]]:
        """Return callable/kwargs entries present in a function pattern."""

        if not func_value:
            return []
        if isinstance(func_value, dict):
            items: list[tuple[Callable, dict]] = []
            for channel_funcs in func_value.values():
                items.extend(cls.normalize_func_items(channel_funcs))
            return items
        if isinstance(func_value, list):
            items: list[tuple[Callable, dict]] = []
            for item in func_value:
                func_obj, kwargs = PatternDataManager.extract_func_and_kwargs(item)
                if func_obj:
                    items.append((func_obj, kwargs))
            return items
        func_obj, kwargs = PatternDataManager.extract_func_and_kwargs(func_value)
        if not func_obj:
            return []
        return [(func_obj, kwargs)]

    @classmethod
    def scope_tokens_for_function_pattern(
        cls,
        scope_id: str,
        func_value: PipelineFunctionPattern,
    ) -> FunctionPatternTokenTree:
        """Return child scope-token metadata for one function pattern."""

        if not func_value:
            return []
        if isinstance(func_value, dict):
            return {
                str(channel_key): cls.scope_tokens_for_function_pattern(
                    scope_id,
                    channel_funcs,
                )
                for channel_key, channel_funcs in func_value.items()
            }
        if isinstance(func_value, list):
            tokens: list[str] = []
            for item in func_value:
                func_obj, _kwargs = PatternDataManager.extract_func_and_kwargs(item)
                if func_obj:
                    tokens.append(ScopeTokenService.ensure_token(scope_id, func_obj))
            return tokens
        func_obj, _kwargs = PatternDataManager.extract_func_and_kwargs(func_value)
        if not func_obj:
            return []
        return [ScopeTokenService.ensure_token(scope_id, func_obj)]

    @classmethod
    def function_pattern_from_child_states(
        cls,
        parent_scope_id: str,
        func_value: PipelineFunctionPattern,
        tokens: FunctionPatternTokenTree,
    ) -> PipelineFunctionPattern:
        """Overlay child function ObjectState values on one function pattern."""

        if isinstance(func_value, dict):
            token_map = tokens if isinstance(tokens, dict) else {}
            return {
                channel_key: cls.function_pattern_from_child_states(
                    parent_scope_id,
                    channel_funcs,
                    token_map.get(str(channel_key)),
                )
                for channel_key, channel_funcs in func_value.items()
            }
        if isinstance(func_value, list):
            token_list = tokens if isinstance(tokens, list) else []
            return [
                cls.function_entry_from_child_state(
                    parent_scope_id,
                    item,
                    token_list[index] if index < len(token_list) else None,
                )
                for index, item in enumerate(func_value)
            ]
        token = tokens[0] if isinstance(tokens, list) and tokens else None
        return cls.function_entry_from_child_state(
            parent_scope_id,
            func_value,
            token,
        )

    @classmethod
    def function_entry_from_child_state(
        cls,
        parent_scope_id: str,
        func_item: PipelineFunctionPattern,
        token: str | None,
    ) -> PipelineFunctionPattern:
        """Overlay one child function ObjectState on a function-pattern entry."""

        if token is None:
            return func_item
        child_scope_id = f"{parent_scope_id}::{token}"
        if ObjectStateRegistry.get_by_scope(child_scope_id) is None:
            return func_item
        entry = FunctionPatternCodeDocumentService().child_scope_entry(child_scope_id)
        return cls.replace_function_entry(func_item, entry.func, entry.kwargs)

    @classmethod
    def replace_function_entry(
        cls,
        func_item: PipelineFunctionPattern,
        func_obj: Callable,
        kwargs: dict,
    ) -> PipelineFunctionPattern:
        """Return one function-pattern entry with updated callable and kwargs."""

        merged_kwargs = cls.merge_compile_time_kwargs(func_item, kwargs)
        if (
            isinstance(func_item, tuple)
            and len(func_item) == 3
            and callable(func_item[0])
            and isinstance(func_item[1], Mapping)
        ):
            return (func_obj, merged_kwargs, func_item[2])
        if (
            isinstance(func_item, tuple)
            and len(func_item) == 2
            and callable(func_item[0])
            and isinstance(func_item[1], Mapping)
        ):
            return (func_obj, merged_kwargs)
        if callable(func_item):
            return (func_obj, merged_kwargs) if merged_kwargs else func_obj
        return func_item

    @staticmethod
    def merge_compile_time_kwargs(
        func_item: PipelineFunctionPattern,
        kwargs: dict,
    ) -> dict:
        """Preserve hidden compile-time metadata while applying UI-edited kwargs."""
        if (
            isinstance(func_item, tuple)
            and len(func_item) in {2, 3}
            and callable(func_item[0])
            and isinstance(func_item[1], Mapping)
        ):
            return {
                **{
                    key: value
                    for key, value in func_item[1].items()
                    if key == COMPILE_TIME_FUNCTION_KWARGS_KEY
                },
                **kwargs,
            }
        return kwargs
