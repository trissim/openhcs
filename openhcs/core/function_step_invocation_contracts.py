"""Typed invocation contracts carried by FunctionStep declarations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from openhcs.core.function_patterns import FunctionInvocationKey
from openhcs.core.module_artifact_contract import ModuleArtifactContract


class FunctionStepInvocationContractPayload(ABC):
    """Nominal payload for invocation contracts with richer runtime identity."""

    @property
    @abstractmethod
    def planning_contract(self) -> ModuleArtifactContract:
        """Return the aggregate module artifact contract visible to planners."""


FunctionStepInvocationContractValue = (
    ModuleArtifactContract | FunctionStepInvocationContractPayload
)


@dataclass(frozen=True, slots=True)
class FunctionStepInvocationContractBinding:
    """Runtime artifact contract for one FunctionStep function-pattern item."""

    key: FunctionInvocationKey
    contract: FunctionStepInvocationContractValue

    def __post_init__(self) -> None:
        if not isinstance(self.key, FunctionInvocationKey):
            raise TypeError(
                "FunctionStepInvocationContractBinding.key must be "
                f"FunctionInvocationKey, got {type(self.key).__name__}."
            )
        if not isinstance(
            self.contract,
            (ModuleArtifactContract, FunctionStepInvocationContractPayload),
        ):
            raise TypeError(
                "FunctionStepInvocationContractBinding.contract must be "
                "ModuleArtifactContract or FunctionStepInvocationContractPayload, "
                f"got {type(self.contract).__name__}."
            )

    @property
    def planning_contract(self) -> ModuleArtifactContract:
        """Return the aggregate contract used by generic artifact planning."""

        if isinstance(self.contract, ModuleArtifactContract):
            return self.contract
        return self.contract.planning_contract


@dataclass(frozen=True, slots=True)
class FunctionStepInvocationContracts:
    """Step-owned invocation contract declarations keyed by function-pattern item."""

    bindings: tuple[FunctionStepInvocationContractBinding, ...] = ()

    def __post_init__(self) -> None:
        normalized = tuple(self.bindings)
        seen: set[FunctionInvocationKey] = set()
        for binding in normalized:
            if not isinstance(binding, FunctionStepInvocationContractBinding):
                raise TypeError(
                    "FunctionStepInvocationContracts.bindings must contain "
                    "FunctionStepInvocationContractBinding values, got "
                    f"{type(binding).__name__}."
                )
            if binding.key in seen:
                raise ValueError(
                    "FunctionStepInvocationContracts contains duplicate key "
                    f"{binding.key!r}."
                )
            seen.add(binding.key)
        object.__setattr__(self, "bindings", normalized)

    def __bool__(self) -> bool:
        return bool(self.bindings)

    @classmethod
    def from_bindings(
        cls,
        bindings: tuple[FunctionStepInvocationContractBinding, ...],
    ) -> "FunctionStepInvocationContracts":
        return cls(tuple(bindings))

    def contract_for(
        self,
        key: FunctionInvocationKey,
    ) -> ModuleArtifactContract | None:
        """Return the planning contract bound to ``key``, if one is declared."""

        for binding in self.bindings:
            if binding.key == key:
                return binding.planning_contract
        return None


EMPTY_FUNCTION_STEP_INVOCATION_CONTRACTS = FunctionStepInvocationContracts()
