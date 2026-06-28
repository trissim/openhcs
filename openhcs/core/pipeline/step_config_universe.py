"""Declaration-derived step config projection for compiler and agent surfaces."""

from __future__ import annotations

import inspect
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, is_dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, TypeVar, get_type_hints

from objectstate import get_base_type_for_lazy

from openhcs.core.config import CallableRuntimeConfig
from openhcs.core.runtime_invocation import RuntimeParameterBinding
from openhcs.core.steps.abstract import AbstractStep

if TYPE_CHECKING:
    from objectstate import ObjectState


ConfigT = TypeVar("ConfigT")


@dataclass(frozen=True, slots=True)
class StepConfigDeclaration:
    """One config parameter declared by AbstractStep."""

    field_name: str
    declared_type: type
    config_type: type

    def saved_value(self, step_state: "ObjectState") -> object | None:
        path = step_state.find_path_for_type(self.declared_type)
        if path is None:
            return None
        return step_state.get_saved_resolved_value(path)


@lru_cache(maxsize=1)
def step_config_declarations() -> tuple[StepConfigDeclaration, ...]:
    """Return config parameters declared by AbstractStep.__init__."""
    declarations: list[StepConfigDeclaration] = []
    type_hints = get_type_hints(AbstractStep.__init__)
    parameters = inspect.signature(AbstractStep.__init__).parameters
    for field_name, parameter in parameters.items():
        if field_name == "self":
            continue
        if parameter.kind is not inspect.Parameter.KEYWORD_ONLY:
            continue
        declared_type = type_hints.get(field_name)
        if not isinstance(declared_type, type):
            continue
        config_type = get_base_type_for_lazy(declared_type) or declared_type
        if not isinstance(config_type, type) or not is_dataclass(config_type):
            continue
        declarations.append(
            StepConfigDeclaration(
                field_name=field_name,
                declared_type=declared_type,
                config_type=config_type,
            )
        )
    return tuple(declarations)


def step_config_classes_by_field_name() -> Mapping[str, type]:
    """Return lazy config classes keyed by their AbstractStep parameter name."""
    return {
        declaration.field_name: declaration.declared_type
        for declaration in step_config_declarations()
    }


@dataclass(frozen=True, slots=True)
class StepConfigRoot:
    """One saved ObjectState-resolved step config root."""

    declaration: StepConfigDeclaration
    value: object


@dataclass(frozen=True, slots=True)
class StepConfigUniverse:
    """Saved ObjectState-resolved config roots for one step."""

    roots: tuple[StepConfigRoot, ...]

    @classmethod
    def from_object_state(cls, step_state: "ObjectState") -> "StepConfigUniverse":
        roots: list[StepConfigRoot] = []
        for declaration in step_config_declarations():
            value = declaration.saved_value(step_state)
            if value is None:
                continue
            roots.append(StepConfigRoot(declaration=declaration, value=value))
        return cls(tuple(roots))

    def find(self, config_type: type[ConfigT]) -> ConfigT | None:
        config_base = get_base_type_for_lazy(config_type) or config_type
        if not isinstance(config_base, type):
            return None
        for root in self.roots:
            if isinstance(root.value, config_base):
                return root.value
        return None

    def require(
        self,
        config_type: type[ConfigT],
        *,
        step_index: int,
    ) -> ConfigT:
        value = self.find(config_type)
        if value is None:
            raise ValueError(
                f"Step {step_index} snapshot requires saved ObjectState config "
                f"for {config_type.__name__}."
            )
        return value

    def instances_of(self, config_type: type[ConfigT]) -> Iterator[ConfigT]:
        config_base = get_base_type_for_lazy(config_type) or config_type
        if not isinstance(config_base, type):
            return
        for root in self.roots:
            if isinstance(root.value, config_base):
                yield root.value

    def runtime_parameter_bindings(self) -> tuple[RuntimeParameterBinding, ...]:
        bindings: list[RuntimeParameterBinding] = []
        for provider_type in CallableRuntimeConfig.registered_config_types():
            value = self.find(provider_type)
            if value is None:
                continue
            bindings.append(
                RuntimeParameterBinding(
                    parameter_type=provider_type.runtime_parameter_declaration(),
                    value=value,
                )
            )
        return tuple(bindings)
