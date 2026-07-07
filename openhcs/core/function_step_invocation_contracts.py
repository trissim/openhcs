"""Typed invocation contracts carried by FunctionStep declarations."""

from __future__ import annotations

from dataclasses import dataclass

from openhcs.core.function_patterns import FunctionInvocationKey
from openhcs.core.module_artifact_contract import ModuleArtifactContract


@dataclass(frozen=True, slots=True)
class FunctionStepInvocationContractBinding:
    """Runtime artifact contract for one FunctionStep function-pattern item."""

    key: FunctionInvocationKey
    contract: ModuleArtifactContract

    def __post_init__(self) -> None:
        if not isinstance(self.key, FunctionInvocationKey):
            raise TypeError(
                "FunctionStepInvocationContractBinding.key must be "
                f"FunctionInvocationKey, got {type(self.key).__name__}."
            )
        if not isinstance(self.contract, ModuleArtifactContract):
            raise TypeError(
                "FunctionStepInvocationContractBinding.contract must be "
                f"ModuleArtifactContract, got {type(self.contract).__name__}."
            )


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
        """Return the contract bound to ``key``, if one is declared."""

        for binding in self.bindings:
            if binding.key == key:
                return binding.contract
        return None


EMPTY_FUNCTION_STEP_INVOCATION_CONTRACTS = FunctionStepInvocationContracts()
