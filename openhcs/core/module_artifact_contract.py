"""Typed artifact contract for executable OpenHCS modules."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

from openhcs.core.artifacts import ArtifactSpec

F = TypeVar("F", bound=Callable[..., Any])
MODULE_ARTIFACT_CONTRACT_ATTR = "__openhcs_module_artifact_contract__"


@dataclass(frozen=True, slots=True)
class ModuleArtifactContract:
    """OpenHCS artifact inputs and outputs for one executable module."""

    module_name: str
    inputs: tuple[ArtifactSpec, ...] = ()
    runtime_artifact_inputs: tuple[ArtifactSpec, ...] = ()
    outputs: tuple[ArtifactSpec, ...] = ()
    declared_outputs: tuple[ArtifactSpec, ...] = ()

    def __post_init__(self) -> None:
        if not self.module_name:
            raise ValueError("ModuleArtifactContract.module_name cannot be empty.")
        object.__setattr__(self, "inputs", tuple(self.inputs))
        object.__setattr__(
            self,
            "runtime_artifact_inputs",
            tuple(self.runtime_artifact_inputs),
        )
        object.__setattr__(self, "outputs", tuple(self.outputs))
        object.__setattr__(
            self,
            "declared_outputs",
            tuple(self.declared_outputs or self.outputs),
        )
        for field_name in (
            "inputs",
            "runtime_artifact_inputs",
            "outputs",
            "declared_outputs",
        ):
            for spec in getattr(self, field_name):
                if not isinstance(spec, ArtifactSpec):
                    raise TypeError(
                        f"ModuleArtifactContract.{field_name} must contain "
                        f"ArtifactSpec values, got {type(spec).__name__}."
                    )


def module_artifact_contract(contract: ModuleArtifactContract) -> Callable[[F], F]:
    """Attach a typed module-level artifact contract to a callable."""
    if not isinstance(contract, ModuleArtifactContract):
        raise TypeError(
            "module_artifact_contract requires ModuleArtifactContract, "
            f"got {type(contract).__name__}."
        )

    def decorator(func: F) -> F:
        setattr(func, MODULE_ARTIFACT_CONTRACT_ATTR, contract)
        return func

    return decorator


def module_artifact_contract_from_namespace(
    namespace: Mapping[str, Any],
    *,
    owner_name: str,
) -> ModuleArtifactContract | None:
    """Return typed module-level artifact metadata from a callable namespace."""
    contract = namespace.get(MODULE_ARTIFACT_CONTRACT_ATTR)
    if contract is None:
        return None
    if not isinstance(contract, ModuleArtifactContract):
        raise TypeError(
            f"{owner_name}.{MODULE_ARTIFACT_CONTRACT_ATTR} must be "
            f"ModuleArtifactContract, got {type(contract).__name__}."
        )
    return contract
