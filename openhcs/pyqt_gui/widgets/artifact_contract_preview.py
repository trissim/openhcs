"""Read-only artifact contract preview widgets for OpenHCS function steps."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from PyQt6.QtWidgets import (
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from openhcs.core.artifact_contract_preview import (
    ArtifactContractPreview,
    SourceBindingContractAlignment,
    SourceBindingRuntimeContractGuard,
)
from openhcs.core.callable_contract import CallableContract
from openhcs.core.module_artifact_contract import ModuleArtifactContract
from openhcs.core.source_bindings import (
    EMPTY_SOURCE_BINDINGS,
    StepSourceBindingsConfig,
)


class ArtifactContractPreviewWidget(QWidget):
    """Read-only table of executable artifact contracts for a function spec."""

    _headers = ("Module", "Status", "Direction", "Origin", "Kind", "Name", "Role")

    def __init__(
        self,
        func_spec: Any = None,
        source_bindings: StepSourceBindingsConfig = EMPTY_SOURCE_BINDINGS,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._source_bindings = source_bindings
        self._message = QLabel(self)
        self._table = QTableWidget(0, len(self._headers), self)
        self._table.setHorizontalHeaderLabels(self._headers)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self._table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self._table.horizontalHeader().setStretchLastSection(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(self._message)
        layout.addWidget(self._table)
        self.set_function_spec(func_spec, source_bindings=source_bindings)

    def set_function_spec(
        self,
        func_spec: Any,
        *,
        source_bindings: StepSourceBindingsConfig | None = None,
    ) -> None:
        """Refresh the preview from a FunctionStep-compatible function spec."""
        if source_bindings is not None:
            self._source_bindings = source_bindings
        projection = ArtifactContractPreviewProjection(
            func_spec,
            source_bindings=self._source_bindings,
        )
        previews = projection.previews()
        self._message.setText(projection.message())
        rows = [
            (
                preview.module_name,
                projection.alignment_for(preview.module_name).message,
                row.direction.value,
                row.origin.value,
                row.kind.value,
                row.name,
                "" if row.sidecar_role is None else row.sidecar_role.value,
            )
            for preview in previews
            for row in preview.rows
        ]
        self._table.setRowCount(len(rows))
        for row_index, row_values in enumerate(rows):
            for column_index, value in enumerate(row_values):
                self._table.setItem(
                    row_index,
                    column_index,
                    QTableWidgetItem(value),
                )
        self._table.resizeRowsToContents()
        self._table.resizeColumnsToContents()


@dataclass(frozen=True, slots=True)
class ArtifactContractPreviewProjection:
    """Project FunctionStep-compatible function specs into preview rows."""

    func_spec: Any
    source_bindings: StepSourceBindingsConfig = EMPTY_SOURCE_BINDINGS

    def previews(self) -> tuple[ArtifactContractPreview, ...]:
        return tuple(
            ArtifactContractPreview.from_module_contract(contract)
            for contract in self.module_contracts()
        )

    def alignment_for(self, module_name: str) -> SourceBindingContractAlignment:
        for contract in self.module_contracts():
            if contract.module_name == module_name:
                return SourceBindingRuntimeContractGuard(
                    contract,
                    self.source_bindings,
                ).alignment()
        return SourceBindingContractAlignment()

    def module_contracts(self) -> Iterable[ModuleArtifactContract]:
        for func in self.callables():
            contract = CallableContract.from_callable(func).module_artifact_contract
            if contract is not None:
                yield contract

    def callables(self) -> tuple[Any, ...]:
        return tuple(self.iter_callables(self.func_spec))

    def iter_callables(self, func_spec: Any) -> Iterable[Any]:
        if callable(func_spec):
            yield func_spec
            return
        if (
            isinstance(func_spec, tuple)
            and len(func_spec) in {2, 3}
            and callable(func_spec[0])
        ):
            yield func_spec[0]
            return
        if isinstance(func_spec, list):
            for item in func_spec:
                yield from self.iter_callables(item)
            return
        if isinstance(func_spec, dict):
            for item in func_spec.values():
                yield from self.iter_callables(item)

    def message(self) -> str:
        contracts = tuple(self.module_contracts())
        if not contracts:
            return "No module artifact contract is declared for this function pattern."
        alignments = tuple(
            SourceBindingRuntimeContractGuard(
                contract,
                self.source_bindings,
            ).alignment()
            for contract in contracts
        )
        if any(not alignment.ok for alignment in alignments):
            return "Source-binding drift detected. Fix source bindings before compile."
        return "Read-only artifact contract preview. Edit source bindings in Step Settings."
