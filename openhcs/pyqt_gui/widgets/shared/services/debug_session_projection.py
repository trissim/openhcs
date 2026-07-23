"""PyQt adapters for core debugger session projections."""

from __future__ import annotations

from dataclasses import dataclass

from openhcs.core.debug_session_projection import (
    DebugPauseBoundaryState,
    DebugSessionPhase,
    DebugSessionPhaseDeclarationBase,
    DebugSessionProjectionContext,
    DebugSessionTargetState,
)
from openhcs.pyqt_gui.widgets.shared.services.pipeline_debug_actions import (
    DebugActionDisabledReason,
    DebugActionPlacement,
    PipelineDebugActionDeclarationBase,
)


PipelineDebugTargetState = DebugSessionTargetState
PipelineDebugSessionPhase = DebugSessionPhase
PipelineDebugPauseBoundaryState = DebugPauseBoundaryState
PipelineDebugSessionContext = DebugSessionProjectionContext


@dataclass(frozen=True, slots=True)
class DebugActionRenderModel:
    """Projected render and invocation model for one declared debug action."""

    declaration: type[PipelineDebugActionDeclarationBase]
    action_id: str
    label: str
    tooltip: str
    placement: DebugActionPlacement
    enabled: bool
    disabled_reason: DebugActionDisabledReason | None
    phase: DebugSessionPhase
    side_effects: tuple[str, ...]
    confirmation_required: bool
    requires_active_debug_session: bool
    target_scope_ids: tuple[str, ...]


class DebugToolbarActionProjector:
    """Project declared debug actions for the current PipelineEditor context."""

    @classmethod
    def declarations(cls) -> tuple[type[PipelineDebugActionDeclarationBase], ...]:
        return PipelineDebugActionDeclarationBase.toolbar_actions()

    @classmethod
    def render_models(
        cls,
        context: DebugSessionProjectionContext,
    ) -> tuple[DebugActionRenderModel, ...]:
        return tuple(
            cls.render_model(declaration, context)
            for declaration in cls.declarations()
        )

    @classmethod
    def render_model(
        cls,
        declaration: type[PipelineDebugActionDeclarationBase],
        context: DebugSessionProjectionContext,
    ) -> DebugActionRenderModel:
        phase = cls.phase(context)
        disabled_reason = cls.disabled_reason(declaration, context)
        return DebugActionRenderModel(
            declaration=declaration,
            action_id=declaration.action_id(),
            label=declaration.label_for(context),
            tooltip=declaration.tooltip_for(context),
            placement=declaration.toolbar_placement,
            enabled=disabled_reason is None,
            disabled_reason=disabled_reason,
            phase=phase,
            side_effects=declaration.side_effects,
            confirmation_required=declaration.confirmation_required,
            requires_active_debug_session=declaration.requires_active_debug_session,
            target_scope_ids=cls.target_scope_ids(context),
        )

    @classmethod
    def target_scope_ids(cls, context: DebugSessionProjectionContext) -> tuple[str, ...]:
        if context.target is None:
            return ()
        return (context.target.pipeline_scope_id,)

    @classmethod
    def disabled_reason(
        cls,
        declaration: type[PipelineDebugActionDeclarationBase],
        context: DebugSessionProjectionContext,
    ) -> DebugActionDisabledReason | None:
        if context.target is None:
            return DebugActionDisabledReason(
                code="debug_target_required",
                message="Debug controls require a selected plate.",
                hint=(
                    "Use plate_manager.state to select a plate, initialize it, compile it, "
                    "then read pipeline_debug_toolbar.session."
                ),
            )
        if not context.target.initialized:
            return DebugActionDisabledReason(
                code="debug_initialization_required",
                message="Debug controls require an initialized selected plate.",
                hint=(
                    "Run the selected-plate init workflow before invoking debug controls."
                ),
            )
        if not context.target.compiled:
            return DebugActionDisabledReason(
                code="debug_compile_required",
                message="Debug controls require a compiled selected plate.",
                hint=(
                    "Run the selected-plate compile workflow before invoking debug controls."
                ),
            )
        if cls.execution_pending_without_session(context):
            if declaration.enabled_during_pending_execution:
                return None
            return DebugActionDisabledReason(
                code="debug_execution_pending",
                message=(
                    "A debug execution is queued or running, but no paused debug "
                    "session is available yet."
                ),
                hint=(
                    "Poll pipeline_debug_toolbar.session and plate_manager.state; "
                    "use Stop debug session if the queued execution is stuck."
                ),
            )
        override = declaration.availability_override(context)
        if override is not None:
            return override
        if (
            declaration.requires_active_or_pending_execution
            and context.active_session is None
        ):
            return DebugActionDisabledReason(
                code="debug_session_required",
                message=f"{declaration.label} requires an active or pending debug execution.",
                hint=(
                    "Run or step the compiled pipeline in debug mode before invoking "
                    f"{declaration.label!r}."
                ),
            )
        if (
            declaration.requires_active_debug_session
            and context.active_session is None
        ):
            return DebugActionDisabledReason(
                code="debug_session_required",
                message=f"{declaration.label} requires an active debug session.",
                hint=(
                    "Run or step the compiled pipeline in debug mode before invoking "
                    f"{declaration.label!r}."
                ),
            )
        return None

    @classmethod
    def execution_pending_without_session(
        cls,
        context: DebugSessionProjectionContext,
    ) -> bool:
        return context.phase is DebugSessionPhase.PENDING_EXECUTION

    @classmethod
    def phase(cls, context: DebugSessionProjectionContext) -> DebugSessionPhase:
        return context.phase


@dataclass(frozen=True, slots=True)
class DebugSessionPanelText:
    """Human-readable debugger session header derived from typed context."""

    title: str
    detail: str

    @classmethod
    def from_context(cls, context: DebugSessionProjectionContext) -> "DebugSessionPanelText":
        phase_declaration = DebugSessionPhaseDeclarationBase.for_context(context)
        cursor = None
        command_type = None
        active_session = context.active_session
        if active_session is not None:
            cursor = active_session.cursor
            command_type = active_session.command_type
        elif context.terminal_summary is not None:
            cursor = context.terminal_summary.cursor
            command_type = context.terminal_summary.command_type

        command_text = "" if command_type is None else f" / {command_type.value}"
        if cursor is None:
            return cls(
                title=phase_declaration.title,
                detail=f"{phase_declaration.detail}{command_text}",
            )
        step_text = (
            f"step {cursor.step_index}"
            if cursor.step_scope_id is None
            else f"step {cursor.step_index} / {cursor.step_scope_id}"
        )
        detail = " / ".join(
            part
            for part in (
                step_text,
                cursor.group_key,
                cursor.invocation_key,
            )
            if part
        )
        return cls(title=phase_declaration.title, detail=f"{detail}{command_text}")


__all__ = (
    "DebugActionDisabledReason",
    "DebugActionRenderModel",
    "DebugSessionPanelText",
    "DebugToolbarActionProjector",
    "PipelineDebugPauseBoundaryState",
    "PipelineDebugSessionContext",
    "PipelineDebugSessionPhase",
    "PipelineDebugTargetState",
)
