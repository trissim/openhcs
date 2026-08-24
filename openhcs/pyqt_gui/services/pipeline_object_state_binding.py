"""GUI-local ObjectState binding for pipeline editor state and step children."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Self

from objectstate.object_state import ObjectState, ObjectStateRegistry
from openhcs.core.steps.function_step import FunctionEntry, FunctionSpec, FunctionStep
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
    FunctionPatternValue,
    function_pattern_authority,
)
from pyqt_reactive.services.pattern_data_manager import (
    FUNC_EDITOR_PATTERN_TOKENS_META_KEY,
)
from pyqt_reactive.services.scope_token_service import (
    ScopeTokenService,
    reconcile_occurrence_tokens,
)

PipelineFunctionPattern = FunctionSpec | None
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
        """Synchronize one plate's authoritative complete step declaration."""

        cls._for_plate(plate_path, register=False)._synchronize_steps(steps)

    @classmethod
    def replace_plate_step(
        cls,
        plate_path: str,
        current_step: FunctionStep,
        edited_step: FunctionStep,
    ) -> list[FunctionStep]:
        """Replace one occurrence and return the authoritative reconstructed steps."""

        if not isinstance(current_step, FunctionStep) or not isinstance(
            edited_step,
            FunctionStep,
        ):
            raise TypeError("Pipeline step replacement requires FunctionStep values.")
        binding = cls._for_plate(plate_path, register=False)
        steps = binding._steps()
        for index, step in enumerate(steps):
            if step is not current_step and not ScopeTokenService.same_object_token(
                step,
                current_step,
            ):
                continue
            ScopeTokenService.transfer_token(
                plate_path,
                current_step,
                edited_step,
            )
            steps[index] = edited_step
            binding._synchronize_steps(steps)
            return binding._steps()
        raise ValueError(
            "Edited pipeline step is not present in the authoritative state."
        )

    @classmethod
    def stage_step(cls, plate_path: str, step: FunctionStep) -> str:
        """Register an editable step without adding it to the pipeline declaration."""

        if not isinstance(step, FunctionStep):
            raise TypeError("A staged pipeline step must be a FunctionStep.")
        binding = cls._for_plate(plate_path)
        ScopeTokenService.ensure_token(plate_path, step)
        scope_id = ScopeTokenService.build_scope_id(plate_path, step)
        if scope_id in binding._editor_state().step_scope_ids:
            raise ValueError(f"Pipeline step {scope_id!r} is already committed.")

        _step_state, states = binding._collect_step_registration_states(
            step=step,
            scope_id=scope_id,
            parent_state=ObjectStateRegistry.get_by_scope(plate_path),
        )
        for state in states:
            ObjectStateRegistry.register(state)
        return scope_id

    @classmethod
    def discard_staged_step(cls, plate_path: str, scope_id: str) -> None:
        """Unregister a staged step while preserving the committed declaration."""

        binding = cls._for_plate(plate_path)
        if scope_id in binding._editor_state().step_scope_ids:
            raise ValueError(f"Pipeline step {scope_id!r} is already committed.")
        ObjectStateRegistry.unregister_scope_and_descendants(
            scope_id,
            _skip_snapshot=True,
        )

    @classmethod
    def commit_plate_state(cls, plate_path: str) -> None:
        """Advance the saved baseline for one editor's exact active state tree."""

        binding = cls._for_plate(plate_path)
        active_states: list[ObjectState] = [binding.state]
        for step_scope_id in binding._editor_state().step_scope_ids:
            step_state = ObjectStateRegistry.get_by_scope(step_scope_id)
            if step_state is None:
                raise RuntimeError(
                    f"Pipeline step state {step_scope_id!r} is not registered."
                )
            active_states.append(step_state)
            for token in binding._flatten_function_tokens(
                step_state.metadata.get(FUNC_EDITOR_PATTERN_TOKENS_META_KEY)
            ):
                function_scope_id = f"{step_scope_id}{SCOPE_SEGMENT_SEPARATOR}{token}"
                function_state = ObjectStateRegistry.get_by_scope(function_scope_id)
                if function_state is None:
                    raise RuntimeError(
                        "Pipeline function state "
                        f"{function_scope_id!r} is not registered."
                    )
                active_states.append(function_state)

        for state in active_states:
            state.mark_saved()

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

    def _synchronize_steps(self, steps: list[FunctionStep]) -> None:
        """Apply a declaration diff to this editor's FunctionStep scopes."""

        if not isinstance(steps, list):
            raise TypeError("Pipeline editor steps must be a mutable list.")
        if not all(isinstance(step, FunctionStep) for step in steps):
            raise TypeError("Pipeline editor steps must contain FunctionStep values.")

        editor_state = self._editor_state()
        existing_step_scope_ids = editor_state.step_scope_ids
        previous_steps: list[FunctionStep] = []
        previous_tokens: list[str] = []
        for scope_id in existing_step_scope_ids:
            state = ObjectStateRegistry.get_by_scope(scope_id)
            token = FunctionStepScopeToken.from_segment(
                scope_id.rsplit(SCOPE_SEGMENT_SEPARATOR, 1)[-1]
            )
            if state is None or token is None:
                continue
            previous_steps.append(self._step_from_state(state))
            previous_tokens.append(token.raw)

        generator = ScopeTokenService.get_generator(
            self._plate_scope,
            FunctionStep.__name__.lower(),
        )
        generator.seed_from_tokens(previous_tokens)
        step_tokens = reconcile_occurrence_tokens(
            previous_steps,
            previous_tokens,
            steps,
            same_declaration=lambda previous, next_step: previous.same_declaration(
                next_step
            ),
            occurrence_authorities=lambda step: step.occurrence_authorities(),
            requested_tokens=[ScopeTokenService.object_token(step) for step in steps],
            token_factory=generator.ensure,
        )
        for step, token in zip(steps, step_tokens):
            ScopeTokenService.adopt_token(self._plate_scope, step, token)

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

        removed_step_scope_ids = set(existing_step_scope_ids).difference(step_scope_ids)
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
            is_new_step = True
            step_state = ObjectState(
                object_instance=step,
                scope_id=scope_id,
                parent_state=parent_state,
            )
            to_register.append(step_state)
            previous_func = None
        else:
            is_new_step = False
            previous_step = self._step_from_state(step_state)
            previous_func = previous_step.func
            self._apply_step_declaration_parameters(step_state, step)

        previous_tokens = step_state.metadata.get(FUNC_EDITOR_PATTERN_TOKENS_META_KEY)
        pattern_tokens = self._scope_tokens_for_function_pattern(
            previous_func,
            step.func,
            previous_tokens,
        )
        step_state.metadata[FUNC_EDITOR_PATTERN_TOKENS_META_KEY] = pattern_tokens

        function_states: dict[str, ObjectState] = {}
        function_service = FunctionPatternCodeDocumentService()
        for entry in function_service.iter_tokenized_entries(
            step.func,
            pattern_tokens,
        ):
            func_obj, kwargs, token = entry.func, entry.kwargs, entry.token
            func_scope_id = f"{scope_id}{SCOPE_SEGMENT_SEPARATOR}{token}"
            existing_func_state = ObjectStateRegistry.get_by_scope(func_scope_id)
            if existing_func_state is not None:
                if function_service.same_function_authority(
                    existing_func_state.object_instance,
                    func_obj,
                ):
                    FunctionPatternCodeDocumentService.apply_kwargs_to_state(
                        state=existing_func_state,
                        previous_kwargs=(
                            FunctionPatternCodeDocumentService.reconstruct_kwargs_from_state(
                                existing_func_state
                            )
                        ),
                        next_kwargs=kwargs,
                    )
                else:
                    FunctionPatternCodeDocumentService.replace_function_state(
                        scope_id=func_scope_id,
                        parent_state=step_state,
                        entry=FunctionPatternValue(func_obj, kwargs),
                    )
                    existing_func_state = ObjectStateRegistry.get_by_scope(
                        func_scope_id
                    )
                    if existing_func_state is None:
                        raise RuntimeError(
                            "Function-pattern state replacement did not register "
                            f"{func_scope_id!r}."
                        )
                function_states[func_scope_id] = existing_func_state
                continue
            editable_func = EditableFunctionPatternCallable.for_entry(
                func_obj,
                kwargs,
            )
            exclude_params = (
                FunctionPatternCodeDocumentService.reserved_parameter_names(
                    editable_func
                )
            )
            function_state = ObjectState(
                object_instance=editable_func,
                scope_id=func_scope_id,
                parent_state=step_state,
                exclude_params=exclude_params,
                initial_values=dict(kwargs),
            )
            function_states[func_scope_id] = function_state
            to_register.append(function_state)

        active_tokens = set(self._flatten_function_tokens(pattern_tokens))
        for stale_token in set(
            self._flatten_function_tokens(previous_tokens)
        ).difference(active_tokens):
            FunctionPatternCodeDocumentService.unregister_function_state(
                scope_id,
                stale_token,
            )

        canonical_func = self._function_pattern_from_child_states(
            scope_id,
            step.func,
            step_state.metadata.get(FUNC_EDITOR_PATTERN_TOKENS_META_KEY),
            function_states=function_states,
        )
        if is_new_step:
            step_state.update_object_instance(step.with_function_spec(canonical_func))
        else:
            step_state.update_parameter("func", canonical_func)

        return step_state, to_register

    @classmethod
    def _scope_tokens_for_function_pattern(
        cls,
        previous_func_value: PipelineFunctionPattern,
        func_value: PipelineFunctionPattern,
        previous_tokens: FunctionPatternTokenTree,
    ) -> FunctionPatternTokenTree:
        """Return child scope-token metadata for one function pattern."""

        return FunctionPatternCodeDocumentService().reconcile_pattern_tokens(
            previous_func_value,
            previous_tokens,
            func_value,
        )

    @staticmethod
    def _apply_step_declaration_parameters(
        state: ObjectState,
        step: FunctionStep,
    ) -> None:
        """Apply non-function declaration fields while preserving state baselines."""

        for parameter_name, value in step.declaration_parameters().items():
            if parameter_name == "func":
                continue
            if parameter_name not in state.parameters:
                raise RuntimeError(
                    f"Step declaration field {parameter_name!r} is absent from "
                    f"ObjectState {state.scope_id!r}."
                )
            state.update_parameter(parameter_name, value)

    @classmethod
    def _flatten_function_tokens(
        cls,
        tokens: FunctionPatternTokenTree,
    ) -> tuple[str, ...]:
        """Return every occurrence token from recursive pattern metadata."""

        if isinstance(tokens, list):
            return tuple(str(token) for token in tokens if token)
        if isinstance(tokens, dict):
            return tuple(
                token
                for nested_tokens in tokens.values()
                for token in cls._flatten_function_tokens(nested_tokens)
            )
        return ()

    @classmethod
    def _function_pattern_from_child_states(
        cls,
        parent_scope_id: str,
        func_value: PipelineFunctionPattern,
        tokens: FunctionPatternTokenTree,
        *,
        function_states: Mapping[str, ObjectState] | None = None,
    ) -> PipelineFunctionPattern:
        """Overlay child function ObjectState values on one function pattern."""

        if isinstance(func_value, dict):
            token_map = tokens if isinstance(tokens, dict) else {}
            projected_by_key: dict[str, list[FunctionEntry]] = {}
            for channel_key, channel_funcs in func_value.items():
                projected = cls._function_pattern_from_child_states(
                    parent_scope_id,
                    channel_funcs,
                    token_map.get(str(channel_key)),
                    function_states=function_states,
                )
                if not isinstance(projected, list):
                    raise RuntimeError(
                        "Function-pattern channel projection must remain a list."
                    )
                projected_by_key[channel_key] = projected
            return projected_by_key
        if isinstance(func_value, list):
            token_list = tokens if isinstance(tokens, list) else []
            return [
                cls._function_entry_from_child_state(
                    parent_scope_id,
                    item,
                    token_list[index] if index < len(token_list) else None,
                    function_states=function_states,
                )
                for index, item in enumerate(func_value)
            ]
        if func_value is None:
            return None
        token = tokens[0] if isinstance(tokens, list) and tokens else None
        return cls._function_entry_from_child_state(
            parent_scope_id,
            func_value,
            token,
            function_states=function_states,
        )

    @classmethod
    def _function_entry_from_child_state(
        cls,
        parent_scope_id: str,
        func_item: FunctionEntry,
        token: str | None,
        *,
        function_states: Mapping[str, ObjectState] | None = None,
    ) -> FunctionEntry:
        """Overlay one child function ObjectState on a function-pattern entry."""

        if token is None:
            return func_item
        child_scope_id = f"{parent_scope_id}::{token}"
        child_state = (
            function_states[child_scope_id]
            if function_states is not None
            else ObjectStateRegistry.get_by_scope(child_scope_id)
        )
        if child_state is None:
            return func_item
        service = FunctionPatternCodeDocumentService()
        if function_states is not None:
            func_obj = function_pattern_authority(child_state.object_instance)
            kwargs = service.reconstruct_kwargs_from_state(child_state)
        else:
            entry = service.child_scope_entry(child_scope_id)
            func_obj = entry.func
            kwargs = entry.kwargs
        return cls._replace_function_entry(func_item, func_obj, kwargs)

    @classmethod
    def _replace_function_entry(
        cls,
        func_item: FunctionEntry,
        func_obj: Callable,
        kwargs: dict,
    ) -> FunctionEntry:
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
