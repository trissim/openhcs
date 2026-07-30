"""GUI-local ObjectState binding for pipeline editor state and step children."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Self

from objectstate.object_state import ObjectState, ObjectStateRegistry
from openhcs.core.steps.function_step import FunctionSpec, FunctionStep
from openhcs.pyqt_gui.services.plate_manager_root_state import (
    root_orchestrator_scope_ids,
)
from openhcs.ui.shared.plate_scope_identity import (
    PipelineScopeIdentity,
    PlateScopeIdentity,
)
from openhcs.pyqt_gui.services.step_scope_identity import (
    FunctionStepScopeToken,
    SCOPE_SEGMENT_SEPARATOR,
)
from pyqt_reactive.services.function_pattern_code_document import (
    EditableFunctionPatternCallable,
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
class PipelineEditorStateRoot:
    """GUI-only text and child-scope state for one pipeline editor."""

    name: str
    description: str | None
    step_scope_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PipelineObjectStateBinding:
    """ObjectState interface for editor text and child FunctionStep states."""

    state: ObjectState

    def __post_init__(self) -> None:
        if not isinstance(self.state.object_instance, PipelineEditorStateRoot):
            raise TypeError(
                "PipelineObjectStateBinding requires an ObjectState backed by "
                "PipelineEditorStateRoot."
            )

    @property
    def _plate_scope(self) -> str:
        """Return the logical plate scope owning this editor ObjectState."""

        return PipelineScopeIdentity.from_scope_id(self.state.scope_id).plate_scope

    @classmethod
    def _for_plate(
        cls,
        plate_path: str,
        *,
        register: bool = True,
    ) -> Self:
        """Return the editor-state binding for one plate scope."""

        if not plate_path:
            raise ValueError("Pipeline editor state requires a non-empty plate path.")

        pipeline_scope = PipelineScopeIdentity.from_plate_scope(plate_path).scope_id
        state = ObjectStateRegistry.get_by_scope(pipeline_scope)
        if state is None:
            identity = PlateScopeIdentity.from_scope_id(plate_path)
            state = ObjectState(
                object_instance=PipelineEditorStateRoot(
                    name=identity.display_name,
                    description=None,
                    step_scope_ids=(),
                ),
                scope_id=pipeline_scope,
                parent_state=ObjectStateRegistry.get_by_scope(plate_path),
            )
            if register:
                ObjectStateRegistry.register(state, _skip_snapshot=True)
        return cls(state)

    @classmethod
    def steps_for_plate(cls, plate_path: str) -> list[FunctionStep]:
        """Return FunctionSteps reconstructed from one editor's child states."""

        return cls._for_plate(plate_path)._steps()

    @classmethod
    def update_plate_steps(
        cls,
        plate_path: str,
        steps: list[FunctionStep],
    ) -> None:
        """Replace one plate's editor child states from a mutable step list."""

        cls._for_plate(plate_path, register=False).replace_steps(steps)

    @classmethod
    def editor_state_for_plate(
        cls,
        plate_path: str,
    ) -> PipelineEditorStateRoot:
        """Return GUI-only text and child-scope state for one plate."""

        return cls._for_plate(plate_path)._editor_state()

    @classmethod
    def update_editor_text(
        cls,
        plate_path: str,
        *,
        name: str,
        description: str | None,
    ) -> None:
        """Replace editor display text without touching executable step state."""

        binding = cls._for_plate(plate_path)
        editor_state = binding._editor_state()
        binding.state.update_object_instance(
            PipelineEditorStateRoot(
                name=name,
                description=description,
                step_scope_ids=editor_state.step_scope_ids,
            )
        )

    @classmethod
    def registered_plate_steps(cls) -> dict[str, list[FunctionStep]]:
        """Return visible plate step lists from the shared ObjectState registry."""

        root_state = ObjectStateRegistry.get_by_scope("__plates__")
        if root_state is None:
            return {}

        result: dict[str, list[FunctionStep]] = {}
        for plate_path in root_orchestrator_scope_ids(root_state):
            result[plate_path] = cls.steps_for_plate(plate_path)
        return result

    def replace_steps(self, steps: list[FunctionStep]) -> None:
        """Replace this editor's declared FunctionStep child scopes."""

        if not isinstance(steps, list):
            raise TypeError("Pipeline editor steps must be a mutable list.")
        if not all(isinstance(step, FunctionStep) for step in steps):
            raise TypeError("Pipeline editor steps must contain FunctionStep values.")

        editor_state = self._editor_state()
        existing_step_scope_ids = editor_state.step_scope_ids
        self._transfer_existing_step_scope_tokens(
            self._plate_scope,
            existing_step_scope_ids,
            steps,
        )
        ScopeTokenService.seed_from_objects(self._plate_scope, steps)

        step_scope_ids: list[str] = []
        to_register: list[ObjectState] = []
        parent_state = ObjectStateRegistry.get_by_scope(self._plate_scope)
        for step in steps:
            scope_id = ScopeTokenService.build_scope_id(self._plate_scope, step)
            step_scope_ids.append(scope_id)
            _step_state, states = self._collect_step_registration_states(
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
        self.state.update_object_instance(
            PipelineEditorStateRoot(
                name=editor_state.name,
                description=editor_state.description,
                step_scope_ids=tuple(step_scope_ids),
            )
        )

    def _editor_state(self) -> PipelineEditorStateRoot:
        """Return the reconstructed GUI-only editor root."""

        editor_state = self.state.to_object()
        if not isinstance(editor_state, PipelineEditorStateRoot):
            raise TypeError(
                "Pipeline editor ObjectState reconstructed an unexpected object: "
                f"{type(editor_state).__name__}."
            )
        return editor_state

    def _steps(self) -> list[FunctionStep]:
        """Reconstruct FunctionSteps from the editor root's child scopes."""

        steps: list[FunctionStep] = []
        for scope_id in self._editor_state().step_scope_ids:
            step_state = ObjectStateRegistry.get_by_scope(scope_id)
            if step_state is not None:
                steps.append(self._step_from_state(step_state))
        return steps

    @staticmethod
    def _step_from_state(step_state: ObjectState) -> FunctionStep:
        """Return a FunctionStep with function child ObjectState values applied."""

        step = step_state.to_object()
        return step.with_function_spec(
            PipelineObjectStateBinding._function_pattern_from_child_states(
                step_state.scope_id,
                step.func,
                step_state.metadata.get(FUNC_EDITOR_PATTERN_TOKENS_META_KEY),
            )
        )

    def _transfer_existing_step_scope_tokens(
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

    def _collect_step_registration_states(
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
            self._scope_tokens_for_function_pattern(scope_id, step.func)
        )

        for func_obj, kwargs in self._normalize_func_items(step.func):
            func_scope_id = ScopeTokenService.build_scope_id(scope_id, func_obj)
            existing_func_state = ObjectStateRegistry.get_by_scope(func_scope_id)
            if existing_func_state is not None:
                FunctionPatternCodeDocumentService.apply_kwargs_to_state(
                    state=existing_func_state,
                    previous_kwargs=(
                        FunctionPatternCodeDocumentService.reconstruct_kwargs_from_state(
                            existing_func_state
                        )
                    ),
                    next_kwargs=kwargs,
                )
                continue
            editable_func = EditableFunctionPatternCallable.for_entry(
                func_obj,
                kwargs,
            )
            exclude_params = FunctionPatternCodeDocumentService.reserved_parameter_names(
                editable_func
            )
            to_register.append(
                ObjectState(
                    object_instance=editable_func,
                    scope_id=func_scope_id,
                    parent_state=step_state,
                    exclude_params=exclude_params,
                    initial_values=dict(kwargs),
                )
            )

        return step_state, to_register

    @classmethod
    def _normalize_func_items(
        cls,
        func_value: PipelineFunctionPattern,
    ) -> list[tuple[Callable, dict]]:
        """Return callable/kwargs entries present in a function pattern."""

        if not func_value:
            return []
        if isinstance(func_value, dict):
            items: list[tuple[Callable, dict]] = []
            for channel_funcs in func_value.values():
                items.extend(cls._normalize_func_items(channel_funcs))
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
    def _scope_tokens_for_function_pattern(
        cls,
        scope_id: str,
        func_value: PipelineFunctionPattern,
    ) -> FunctionPatternTokenTree:
        """Return child scope-token metadata for one function pattern."""

        if not func_value:
            return []
        if isinstance(func_value, dict):
            return {
                str(channel_key): cls._scope_tokens_for_function_pattern(
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
    def _function_pattern_from_child_states(
        cls,
        parent_scope_id: str,
        func_value: PipelineFunctionPattern,
        tokens: FunctionPatternTokenTree,
    ) -> PipelineFunctionPattern:
        """Overlay child function ObjectState values on one function pattern."""

        if isinstance(func_value, dict):
            token_map = tokens if isinstance(tokens, dict) else {}
            return {
                channel_key: cls._function_pattern_from_child_states(
                    parent_scope_id,
                    channel_funcs,
                    token_map.get(str(channel_key)),
                )
                for channel_key, channel_funcs in func_value.items()
            }
        if isinstance(func_value, list):
            token_list = tokens if isinstance(tokens, list) else []
            return [
                cls._function_entry_from_child_state(
                    parent_scope_id,
                    item,
                    token_list[index] if index < len(token_list) else None,
                )
                for index, item in enumerate(func_value)
            ]
        token = tokens[0] if isinstance(tokens, list) and tokens else None
        return cls._function_entry_from_child_state(
            parent_scope_id,
            func_value,
            token,
        )

    @classmethod
    def _function_entry_from_child_state(
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
        return cls._replace_function_entry(func_item, entry.func, entry.kwargs)

    @classmethod
    def _replace_function_entry(
        cls,
        func_item: PipelineFunctionPattern,
        func_obj: Callable,
        kwargs: dict,
    ) -> PipelineFunctionPattern:
        """Return one function-pattern entry with updated callable and kwargs."""

        if (
            isinstance(func_item, tuple)
            and len(func_item) == 2
            and callable(func_item[0])
            and isinstance(func_item[1], Mapping)
        ):
            return (func_obj, kwargs)
        if callable(func_item):
            return (func_obj, kwargs) if kwargs else func_obj
        return func_item
