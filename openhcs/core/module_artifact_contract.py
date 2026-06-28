"""Typed artifact contract for executable OpenHCS modules."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

from openhcs.constants.constants import VariableComponents
from openhcs.core.artifacts import ArtifactKind, ArtifactSpec, ArtifactSpecCollection

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
    required_variable_components: tuple[VariableComponents, ...] = ()

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
        object.__setattr__(
            self,
            "required_variable_components",
            tuple(
                component
                if isinstance(component, VariableComponents)
                else VariableComponents(component)
                for component in self.required_variable_components
            ),
        )
        self._validate_specs("inputs", self.inputs)
        self._validate_specs("runtime_artifact_inputs", self.runtime_artifact_inputs)
        self._validate_specs("outputs", self.outputs)
        self._validate_specs("declared_outputs", self.declared_outputs)

    @staticmethod
    def _validate_specs(field_name: str, specs: tuple[ArtifactSpec, ...]) -> None:
        for spec in specs:
            if not isinstance(spec, ArtifactSpec):
                raise TypeError(
                    f"ModuleArtifactContract.{field_name} must contain "
                    f"ArtifactSpec values, got {type(spec).__name__}."
                )

    def input_collection(self) -> ArtifactSpecCollection:
        """Return declared source inputs as an ordered artifact collection."""
        return ArtifactSpecCollection(self.inputs)

    def runtime_artifact_input_collection(self) -> ArtifactSpecCollection:
        """Return runtime-provided artifact inputs as an ordered collection."""
        return ArtifactSpecCollection(self.runtime_artifact_inputs)

    def output_collection(self) -> ArtifactSpecCollection:
        """Return module outputs as an ordered artifact collection."""
        return ArtifactSpecCollection(self.outputs)

    def declared_output_collection(self) -> ArtifactSpecCollection:
        """Return originally declared module outputs as an ordered collection."""
        return ArtifactSpecCollection(self.declared_outputs)

    def declared_input_specs(self) -> tuple[ArtifactSpec, ...]:
        """Return explicit inputs plus runtime-only inputs in contract order."""
        runtime_extras = tuple(
            spec for spec in self.runtime_artifact_inputs if spec not in self.inputs
        )
        return (*self.inputs, *runtime_extras)

    def declared_input_collection(self) -> ArtifactSpecCollection:
        """Return all inputs the module can resolve at execution time."""
        return ArtifactSpecCollection(self.declared_input_specs())

    def runtime_input_names(self, kind: ArtifactKind) -> tuple[str, ...]:
        """Return runtime-provided input names of one artifact kind."""
        return self.runtime_artifact_input_collection().names_of_kind(kind)

    def runtime_input_name_set(self, kind: ArtifactKind) -> frozenset[str]:
        """Return runtime-provided input names of one artifact kind as a set."""
        return self.runtime_artifact_input_collection().name_set_of_kind(kind)

    def external_input_names(self, kind: ArtifactKind) -> tuple[str, ...]:
        """Return source-resolved input names after excluding runtime inputs."""
        runtime_names = self.runtime_input_name_set(kind)
        return tuple(
            name
            for name in self.input_collection().names_of_kind(kind)
            if name not in runtime_names
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
