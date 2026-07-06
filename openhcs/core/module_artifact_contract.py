"""Typed artifact contract for executable OpenHCS modules."""

from __future__ import annotations

from abc import ABC
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, TypeVar

from metaclass_registry import AutoRegisterMeta

from openhcs.constants.constants import VariableComponents
from openhcs.core.artifact_key_selection import ArtifactPlanKeySelector
from openhcs.core.artifacts import (
    ArtifactInputPlan,
    ArtifactOutputPlan,
    ArtifactPlan,
    ArtifactSpec,
    ArtifactSpecCollection,
    ArtifactTypeValue,
)

F = TypeVar("F", bound=Callable[..., Any])
MODULE_ARTIFACT_CONTRACT_ATTR = "__openhcs_module_artifact_contract__"


class ModuleArtifactContractPartition(ABC, metaclass=AutoRegisterMeta):
    """Registered owner-local partition for one module artifact declaration."""

    __registry_key__ = "partition_key"
    __skip_if_no_key__ = True

    partition_key: ClassVar[str | None] = None
    plan_type: ClassVar[type[ArtifactPlan] | None] = None

    @classmethod
    def require_plan_type(cls) -> type[ArtifactPlan]:
        if cls.plan_type is None:
            raise TypeError(
                f"{cls.__name__} does not declare an ArtifactPlan partition role."
            )
        if not issubclass(cls.plan_type, ArtifactPlan):
            raise TypeError(
                f"{cls.__name__}.plan_type must be an ArtifactPlan type, "
                f"got {type(cls.plan_type).__name__}."
            )
        return cls.plan_type

    @classmethod
    def require_registered(
        cls,
        partition_type: type["ModuleArtifactContractPartition"],
    ) -> type["ModuleArtifactContractPartition"]:
        if not isinstance(partition_type, type) or not issubclass(partition_type, cls):
            raise TypeError(
                "Module artifact partition must be a "
                f"{cls.__name__} type, got {type(partition_type).__name__}."
            )
        if partition_type not in cls.__registry__.values():
            raise ValueError(
                f"{partition_type.__name__} is not a registered module artifact "
                "contract partition."
            )
        partition_type.require_plan_type()
        return partition_type


class SourceArtifactInputPartition(ModuleArtifactContractPartition):
    """Artifact input supplied by source binding metadata or source images."""

    partition_key: ClassVar[str] = "source_input"
    plan_type: ClassVar[type[ArtifactPlan]] = ArtifactInputPlan


class RuntimeArtifactInputPartition(ModuleArtifactContractPartition):
    """Artifact input supplied by runtime artifact storage."""

    partition_key: ClassVar[str] = "runtime_artifact_input"
    plan_type: ClassVar[type[ArtifactPlan]] = ArtifactInputPlan


class RecordedArtifactOutputPartition(ModuleArtifactContractPartition):
    """Artifact output recorded by this module after pruning."""

    partition_key: ClassVar[str] = "recorded_output"
    plan_type: ClassVar[type[ArtifactPlan]] = ArtifactOutputPlan


class DeclaredArtifactOutputPartition(ModuleArtifactContractPartition):
    """Artifact output declared before dead-output pruning."""

    partition_key: ClassVar[str] = "declared_output"
    plan_type: ClassVar[type[ArtifactPlan]] = ArtifactOutputPlan


@dataclass(frozen=True, slots=True)
class ModuleArtifactContractItem:
    """One artifact spec assigned to a registered module-contract partition."""

    partition_type: type[ModuleArtifactContractPartition]
    spec: ArtifactSpec

    def __post_init__(self) -> None:
        partition_type = ModuleArtifactContractPartition.require_registered(
            self.partition_type,
        )
        if not isinstance(self.spec, ArtifactSpec):
            raise TypeError(
                "ModuleArtifactContractItem.spec must be an ArtifactSpec, "
                f"got {type(self.spec).__name__}."
            )
        plan_type = partition_type.require_plan_type()
        if self.spec.plan_type is not plan_type:
            raise ValueError(
                f"{partition_type.__name__} requires {plan_type.plan_role} specs, "
                f"got {self.spec.plan_type.plan_role} for "
                f"{self.spec.artifact_type.value}:{self.spec.name}."
            )
        object.__setattr__(self, "partition_type", partition_type)


@dataclass(frozen=True, slots=True)
class ModuleArtifactContractItemCollection:
    """Ordered query surface over partitioned module artifact declarations."""

    items: tuple[ModuleArtifactContractItem, ...]

    def __init__(self, items: Iterable[ModuleArtifactContractItem]):
        normalized = tuple(items)
        for item in normalized:
            if not isinstance(item, ModuleArtifactContractItem):
                raise TypeError(
                    "ModuleArtifactContractItemCollection requires "
                    f"ModuleArtifactContractItem values, got {type(item).__name__}."
                )
        object.__setattr__(self, "items", normalized)

    def specs_for_partition(
        self,
        partition_type: type[ModuleArtifactContractPartition],
    ) -> tuple[ArtifactSpec, ...]:
        resolved_partition = ModuleArtifactContractPartition.require_registered(
            partition_type,
        )
        return tuple(
            item.spec
            for item in self.items
            if item.partition_type is resolved_partition
        )

    def artifact_specs(self) -> ArtifactSpecCollection:
        """Return all artifact specs from all partitions in declaration order."""
        return ArtifactSpecCollection(item.spec for item in self.items)


@dataclass(frozen=True, slots=True)
class ModuleArtifactContract(ArtifactPlanKeySelector):
    """OpenHCS artifact declarations for one executable module."""

    module_name: str
    items: tuple[ModuleArtifactContractItem, ...] = ()
    required_variable_components: tuple[VariableComponents, ...] = ()

    def __post_init__(self) -> None:
        if not self.module_name:
            raise ValueError("ModuleArtifactContract.module_name cannot be empty.")
        item_collection = ModuleArtifactContractItemCollection(self.items)
        object.__setattr__(self, "items", item_collection.items)
        object.__setattr__(
            self,
            "required_variable_components",
            tuple(
                (
                    component
                    if isinstance(component, VariableComponents)
                    else VariableComponents(component)
                )
                for component in self.required_variable_components
            ),
        )
        self.validate_artifact_relation_refs(
            owner_name=f"ModuleArtifactContract({self.module_name})",
        )

    @staticmethod
    def items_for_partition(
        partition_type: type[ModuleArtifactContractPartition],
        specs: Iterable[ArtifactSpec],
    ) -> tuple[ModuleArtifactContractItem, ...]:
        """Build partitioned contract items for one registered partition."""
        return tuple(
            ModuleArtifactContractItem(partition_type, spec) for spec in tuple(specs)
        )

    @property
    def item_collection(self) -> ModuleArtifactContractItemCollection:
        """Return the canonical partitioned artifact declarations."""
        return ModuleArtifactContractItemCollection(self.items)

    def specs_for_partition(
        self,
        partition_type: type[ModuleArtifactContractPartition],
    ) -> tuple[ArtifactSpec, ...]:
        """Return specs for one registered module-contract partition."""
        return self.item_collection.specs_for_partition(partition_type)

    def names_for_partition(
        self,
        partition_type: type[ModuleArtifactContractPartition],
    ) -> tuple[str, ...]:
        """Return artifact names declared in one registered contract partition."""
        return tuple(spec.name for spec in self.specs_for_partition(partition_type))

    def select_keys_for_partition(
        self,
        partition_type: type[ModuleArtifactContractPartition],
        keys: Iterable[str],
    ) -> tuple[str, ...]:
        """Return candidate keys owned by one registered contract partition."""
        partition_names = frozenset(self.names_for_partition(partition_type))
        return tuple(key for key in keys if key in partition_names)

    def has_keys_for_partition(
        self,
        partition_type: type[ModuleArtifactContractPartition],
        keys: Iterable[str],
    ) -> bool:
        """Return whether any candidate key belongs to a contract partition."""
        return bool(self.select_keys_for_partition(partition_type, keys))

    @property
    def inputs(self) -> tuple[ArtifactSpec, ...]:
        """Return source-bound/public artifact inputs."""
        return self.specs_for_partition(SourceArtifactInputPartition)

    @property
    def runtime_artifact_inputs(self) -> tuple[ArtifactSpec, ...]:
        """Return artifact-store inputs used at runtime."""
        return self.specs_for_partition(RuntimeArtifactInputPartition)

    @property
    def outputs(self) -> tuple[ArtifactSpec, ...]:
        """Return live output artifacts recorded by this module."""
        return self.specs_for_partition(RecordedArtifactOutputPartition)

    @property
    def declared_outputs(self) -> tuple[ArtifactSpec, ...]:
        """Return output artifacts declared before dead-output pruning."""
        return self.specs_for_partition(DeclaredArtifactOutputPartition)

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

    @property
    def artifact_specs(self) -> ArtifactSpecCollection:
        """All artifact specs declared by this module contract."""
        return self.item_collection.artifact_specs()

    def runtime_input_names(self, artifact_type: ArtifactTypeValue) -> tuple[str, ...]:
        """Return runtime-provided input names of one artifact type."""
        return self.runtime_artifact_input_collection().names_of_artifact_type(
            artifact_type
        )

    def runtime_input_name_set(
        self, artifact_type: ArtifactTypeValue
    ) -> frozenset[str]:
        """Return runtime-provided input names of one artifact type as a set."""
        return self.runtime_artifact_input_collection().name_set_of_artifact_type(
            artifact_type
        )

    def external_input_names(self, artifact_type: ArtifactTypeValue) -> tuple[str, ...]:
        """Return source-resolved input names after excluding runtime inputs."""
        runtime_names = self.runtime_input_name_set(artifact_type)
        return tuple(
            name
            for name in self.input_collection().names_of_artifact_type(artifact_type)
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
