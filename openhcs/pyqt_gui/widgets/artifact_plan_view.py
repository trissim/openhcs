"""Read-only compiled artifact plan and runtime-value view."""

from __future__ import annotations

from dataclasses import dataclass, replace

from PyQt6.QtWidgets import (
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from openhcs.core.artifact_inspection import (
    CompiledArtifactInspection,
    CompiledArtifactStepInspection,
)
from openhcs.core.artifacts import ArtifactPlan, ArtifactSpec
from openhcs.core.debug import DebugArtifactIdentity, DebugArtifactRef, DebugSnapshot
from openhcs.core.runtime_stores import RuntimeArtifactAddress, RuntimeValueStore
from openhcs.pyqt_gui.widgets.shared.services.runtime_artifact_progress_service import (
    RuntimeArtifactAvailableNotification,
)


@dataclass(frozen=True, slots=True)
class ArtifactPlanViewRow:
    """One exact compiled contract occurrence with optional runtime enrichment."""

    context_id: str
    axis_id: str
    callable_name: str
    spec: ArtifactSpec | None
    plan: ArtifactPlan | None
    runtime_location: str = ""
    value_type: str = ""

    @property
    def direction(self) -> str:
        if self.spec is not None:
            return self.spec.require_plan_type().plan_role
        if self.plan is None:
            raise RuntimeError("ArtifactPlanViewRow has neither contract nor plan.")
        return type(self.plan).plan_role

    @property
    def artifact_type_name(self) -> str:
        if self.spec is not None:
            return self.spec.artifact_type.value
        if self.plan is None:
            raise RuntimeError("ArtifactPlanViewRow has neither contract nor plan.")
        return self.plan.artifact_type.value

    @property
    def artifact_name(self) -> str:
        if self.spec is not None:
            return self.spec.name
        if self.plan is None:
            raise RuntimeError("ArtifactPlanViewRow has neither contract nor plan.")
        return self.plan.name

    @property
    def required_text(self) -> str:
        return "" if self.spec is None else ("yes" if self.spec.required else "no")

    @property
    def group_scope_text(self) -> str:
        if self.plan is None:
            return ""
        scope = self.plan.group_scope()
        component = "" if scope.component is None else scope.component.value
        keys = ", ".join("default" if key is None else key for key in scope.keys)
        return keys if not component else f"{component}: {keys}"

    @property
    def planned_path(self) -> str:
        return "" if self.plan is None else self.plan.path

    def with_runtime_address(self, address: RuntimeArtifactAddress) -> "ArtifactPlanViewRow":
        return replace(
            self,
            runtime_location=f"{address.location.backend}:{address.location.path}",
            value_type="" if address.value_type is None else address.value_type,
        )

    def with_debug_ref(self, ref: DebugArtifactRef) -> "ArtifactPlanViewRow":
        shape = "" if ref.shape is None else f" shape={ref.shape}"
        dtype = "" if ref.dtype is None else f" dtype={ref.dtype}"
        backend = "" if ref.storage_backend is None else f"{ref.storage_backend}:"
        return replace(
            self,
            runtime_location=f"{backend}{ref.storage_ref}",
            value_type=f"{shape}{dtype}".strip(),
        )


@dataclass(frozen=True, slots=True)
class ArtifactPlanViewModel:
    """Renderer-independent view derived only from compiled plans and runtime facts."""

    plate_id: str | None = None
    step_index: int | None = None
    rows: tuple[ArtifactPlanViewRow, ...] = ()

    @classmethod
    def from_inspection(
        cls,
        inspection: CompiledArtifactInspection | None,
        *,
        step_index: int | None,
    ) -> "ArtifactPlanViewModel":
        if inspection is None or step_index is None:
            return cls(
                plate_id=None if inspection is None else inspection.plate_id,
                step_index=step_index,
            )
        return cls(
            plate_id=inspection.plate_id,
            step_index=step_index,
            rows=tuple(
                row
                for step in inspection.steps_for_index(step_index)
                for row in cls._rows_for_step(step)
            ),
        )

    @staticmethod
    def _rows_for_step(
        step: CompiledArtifactStepInspection,
    ) -> tuple[ArtifactPlanViewRow, ...]:
        rows: list[ArtifactPlanViewRow] = []
        claimed_plans: list[ArtifactPlan] = []
        for invocation in step.invocations:
            for edge in invocation.input_edges:
                if edge.storage_plan is not None:
                    claimed_plans.append(edge.storage_plan)
                rows.append(
                    ArtifactPlanViewRow(
                        context_id=step.context_id,
                        axis_id=step.axis_id,
                        callable_name=invocation.function_name,
                        spec=edge.spec,
                        plan=edge.storage_plan,
                    )
                )
            for spec, plan in zip(
                invocation.output_specs,
                invocation.output_plans,
                strict=True,
            ):
                claimed_plans.append(plan)
                rows.append(
                    ArtifactPlanViewRow(
                        context_id=step.context_id,
                        axis_id=step.axis_id,
                        callable_name=invocation.function_name,
                        spec=spec,
                        plan=plan,
                    )
                )
        rows.extend(
            ArtifactPlanViewRow(
                context_id=step.context_id,
                axis_id=step.axis_id,
                callable_name="Step plan",
                spec=None,
                plan=plan,
            )
            for plan in (*step.artifact_inputs, *step.artifact_outputs)
            if not any(plan == claimed for claimed in claimed_plans)
        )
        return tuple(rows)

    def with_runtime_notification(
        self,
        notification: RuntimeArtifactAvailableNotification,
    ) -> "ArtifactPlanViewModel":
        if self.plate_id is None or notification.event.plate_id != self.plate_id:
            return self
        rows = self.rows
        for address in notification.payload.addresses:
            rows = tuple(
                (
                    row.with_runtime_address(address)
                    if row.plan is not None
                    and RuntimeValueStore.address_matches_plan(
                        address,
                        row.plan,
                        axis_id=row.axis_id,
                    )
                    else row
                )
                for row in rows
            )
        return replace(self, rows=rows)

    def with_debug_snapshot(self, snapshot: DebugSnapshot) -> "ArtifactPlanViewModel":
        if self.step_index != snapshot.cursor.step_index:
            return self
        refs = (
            *snapshot.input_artifact_refs,
            *snapshot.output_artifact_refs,
            *snapshot.preview_refs,
            *snapshot.measurement_refs,
            *snapshot.relationship_refs,
        )
        rows = self.rows
        for ref in refs:
            rows = tuple(
                (
                    row.with_debug_ref(ref)
                    if row.plan is not None
                    and row.axis_id == snapshot.axis_id
                    and ref.identity is not None
                    and ref.identity.matches(
                        DebugArtifactIdentity.from_artifact_plan(row.plan)
                    )
                    else row
                )
                for row in rows
            )
        return replace(self, rows=rows)


class ArtifactPlanViewWidget(QWidget):
    """Table presentation for one step's compiled artifact occurrences."""

    _headers = (
        "Context",
        "Callable",
        "Direction",
        "Kind",
        "Name",
        "Required",
        "Group",
        "Planned path",
        "Runtime location",
        "Value",
    )

    def __init__(
        self,
        *,
        inspection: CompiledArtifactInspection | None = None,
        step_index: int | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._inspection = inspection
        self._model = ArtifactPlanViewModel.from_inspection(
            inspection,
            step_index=step_index,
        )
        self._message = QLabel(self)
        self._table = QTableWidget(0, len(self._headers), self)
        self._table.setHorizontalHeaderLabels(self._headers)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self._table.horizontalHeader().setStretchLastSection(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(self._message)
        layout.addWidget(self._table)
        self._render()

    @property
    def model(self) -> ArtifactPlanViewModel:
        return self._model

    def set_inspection(self, inspection: CompiledArtifactInspection | None) -> None:
        self._inspection = inspection
        self._model = ArtifactPlanViewModel.from_inspection(
            inspection,
            step_index=self._model.step_index,
        )
        self._render()

    def set_step_index(self, step_index: int | None) -> None:
        self._model = ArtifactPlanViewModel.from_inspection(
            self._inspection,
            step_index=step_index,
        )
        self._render()

    def apply_runtime_notification(
        self,
        notification: RuntimeArtifactAvailableNotification,
    ) -> None:
        self._model = self._model.with_runtime_notification(notification)
        self._render()

    def apply_debug_snapshot(self, snapshot: DebugSnapshot) -> None:
        self._model = self._model.with_debug_snapshot(snapshot)
        self._render()

    def _render(self) -> None:
        rows = self._model.rows
        if self._model.step_index is None:
            message = "No pipeline step selected."
        elif self._model.plate_id is None:
            message = "No compiled artifact plan."
        elif not rows:
            message = "Compiled step has no artifact contracts."
        else:
            message = f"Compiled artifact plan: step {self._model.step_index + 1}"
        self._message.setText(message)
        self._table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            values = (
                row.context_id,
                row.callable_name,
                row.direction,
                row.artifact_type_name,
                row.artifact_name,
                row.required_text,
                row.group_scope_text,
                row.planned_path,
                row.runtime_location,
                row.value_type,
            )
            for column_index, value in enumerate(values):
                self._table.setItem(
                    row_index,
                    column_index,
                    QTableWidgetItem(value),
                )
        self._table.resizeRowsToContents()
