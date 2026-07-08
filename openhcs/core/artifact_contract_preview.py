"""Read-only projections for user-facing artifact contract previews."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from openhcs.core.artifacts import ArtifactSidecarRole, ArtifactType
from openhcs.core.module_artifact_contract import ModuleArtifactContract
from openhcs.core.source_bindings import StepSourceBindingsConfig

ArtifactBindingKeys = tuple[tuple[str, ArtifactType], ...]


class ArtifactPreviewDirection(str, Enum):
    """Direction of one artifact in a module contract preview."""

    INPUT = "input"
    OUTPUT = "output"


class ArtifactPreviewOrigin(str, Enum):
    """User-facing origin for artifact contract preview rows."""

    SOURCE_BINDING = "source_binding"
    RUNTIME_ARTIFACT = "runtime_artifact"
    MODULE_OUTPUT = "module_output"


@dataclass(frozen=True, slots=True)
class ArtifactContractPreviewRow:
    """Read-only row describing one artifact contract entry."""

    name: str
    kind: ArtifactType
    direction: ArtifactPreviewDirection
    origin: ArtifactPreviewOrigin
    required: bool = True
    sidecar_role: ArtifactSidecarRole | None = None
    materialized: bool = False


@dataclass(frozen=True, slots=True)
class ArtifactContractPreview:
    """Read-only artifact contract summary for UI display."""

    module_name: str
    rows: tuple[ArtifactContractPreviewRow, ...]

    @classmethod
    def from_module_contract(
        cls,
        contract: ModuleArtifactContract,
    ) -> "ArtifactContractPreview":
        """Project executable contract metadata into read-only preview rows."""
        runtime_input_keys = {
            (spec.name, spec.artifact_type) for spec in contract.runtime_artifact_inputs
        }
        rows = [
            ArtifactContractPreviewRow(
                name=spec.name,
                kind=spec.artifact_type,
                direction=ArtifactPreviewDirection.INPUT,
                origin=(
                    ArtifactPreviewOrigin.RUNTIME_ARTIFACT
                    if (spec.name, spec.artifact_type) in runtime_input_keys
                    else ArtifactPreviewOrigin.SOURCE_BINDING
                ),
                required=spec.required,
                sidecar_role=spec.sidecar_role,
                materialized=spec.materialization is not None,
            )
            for spec in contract.inputs
        ]
        rows.extend(
            ArtifactContractPreviewRow(
                name=spec.name,
                kind=spec.artifact_type,
                direction=ArtifactPreviewDirection.OUTPUT,
                origin=ArtifactPreviewOrigin.MODULE_OUTPUT,
                required=spec.required,
                sidecar_role=spec.sidecar_role,
                materialized=spec.materialization is not None,
            )
            for spec in contract.outputs
        )
        return cls(contract.module_name, tuple(rows))

    @property
    def inputs(self) -> tuple[ArtifactContractPreviewRow, ...]:
        """Return input preview rows."""
        return tuple(
            row for row in self.rows if row.direction is ArtifactPreviewDirection.INPUT
        )

    @property
    def outputs(self) -> tuple[ArtifactContractPreviewRow, ...]:
        """Return output preview rows."""
        return tuple(
            row for row in self.rows if row.direction is ArtifactPreviewDirection.OUTPUT
        )


@dataclass(frozen=True, slots=True)
class SourceBindingContractAlignment:
    """Alignment report between source-bound contract inputs and source bindings."""

    missing: ArtifactBindingKeys = ()
    unexpected: ArtifactBindingKeys = ()

    @property
    def ok(self) -> bool:
        """Return whether source bindings match source-bound contract inputs."""
        return not self.missing and not self.unexpected

    @property
    def message(self) -> str:
        """Return a compact user-facing alignment summary."""
        if self.ok:
            return "Source bindings match source-bound artifact inputs."
        parts: list[str] = []
        if self.missing:
            parts.append(f"missing: {_format_artifact_keys(self.missing)}")
        if self.unexpected:
            parts.append(f"unexpected: {_format_artifact_keys(self.unexpected)}")
        return "; ".join(parts)


@dataclass(frozen=True, slots=True)
class SourceBindingRuntimeContractGuard:
    """Validate editable source bindings against runtime artifact contracts."""

    contract: ModuleArtifactContract
    source_bindings: StepSourceBindingsConfig

    def validate(self) -> None:
        """Reject drift between source-bound contract inputs and step bindings."""
        alignment = self.alignment()
        if alignment.ok:
            return
        raise ValueError(
            "CellProfiler source bindings drifted from runtime artifact contract "
            f"for module {self.contract.module_name!r}. "
            f"{alignment.message}."
        )

    def validate_required(self) -> None:
        """Reject missing required source-bound inputs while allowing inherited supersets."""
        alignment = self.required_alignment()
        if alignment.ok:
            return
        raise ValueError(
            "CellProfiler source bindings are missing runtime artifact contract "
            f"inputs for module {self.contract.module_name!r}. "
            f"{alignment.message}."
        )

    def alignment(self) -> SourceBindingContractAlignment:
        """Return a non-throwing alignment report for UI preview surfaces."""
        expected = set(self.source_bound_input_keys)
        if not expected:
            return SourceBindingContractAlignment()
        actual = self._source_binding_specs()
        return SourceBindingContractAlignment(
            missing=tuple(sorted(expected - actual)),
            unexpected=tuple(sorted(actual - expected)),
        )

    def required_alignment(self) -> SourceBindingContractAlignment:
        """Return missing required source-bound inputs, ignoring extra inherited aliases."""
        expected = set(self.source_bound_input_keys)
        if not expected:
            return SourceBindingContractAlignment()
        actual = self._source_binding_specs()
        return SourceBindingContractAlignment(
            missing=tuple(sorted(expected - actual)),
        )

    def projected_source_bindings(self) -> StepSourceBindingsConfig:
        """Return source bindings scoped to this contract's source-bound inputs."""
        return self.source_bindings.for_artifact_keys(self.source_bound_input_keys)

    @property
    def source_bound_input_keys(self) -> ArtifactBindingKeys:
        """Return contract inputs that must be satisfied by source bindings."""
        return tuple(sorted(self._source_bound_contract_inputs()))

    def _source_bound_contract_inputs(self) -> set[tuple[str, ArtifactType]]:
        runtime_inputs = {
            (spec.name, spec.artifact_type) for spec in self.contract.runtime_artifact_inputs
        }
        return {
            (spec.name, spec.artifact_type)
            for spec in self.contract.inputs
            if (spec.name, spec.artifact_type) not in runtime_inputs
        }

    def _source_binding_specs(self) -> set[tuple[str, ArtifactType]]:
        return {
            (binding.alias, binding.artifact_kind)
            for binding in self.source_bindings.bindings
        }


def _format_artifact_keys(keys: ArtifactBindingKeys) -> str:
    return ", ".join(f"{kind.value}:{name}" for name, kind in keys)
