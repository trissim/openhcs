"""Generic returned-output matching for runtime artifact contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, TypeAlias

from openhcs.core.aligned_image_payload import (
    AlignedImageSliceContext,
    AlignedImageStack,
)
from openhcs.core.artifacts import (
    ArtifactOutputPlan,
    ArtifactSpec,
    ArtifactSpecRef,
)
from openhcs.core.callable_contract import CallableContract


class RuntimeOutputBundle(ABC):
    """Nominal multi-output value exposed to generic runtime matching."""

    @abstractmethod
    def as_runtime_tuple(self) -> tuple[object, ...]:
        """Return the canonical main output followed by artifact outputs."""


RuntimeMatchedOutput: TypeAlias = tuple[ArtifactOutputPlan, ArtifactSpec, Any]


def runtime_output_tuple(value: Any) -> Any:
    """Lower a nominal output bundle to the generic positional output ABI."""

    if isinstance(value, RuntimeOutputBundle):
        return value.as_runtime_tuple()
    return value


def split_runtime_output(value: Any) -> tuple[Any, tuple[Any, ...]]:
    """Split one runtime return into its canonical and artifact positions."""

    positional = runtime_output_tuple(value)
    if isinstance(positional, tuple):
        if not positional:
            raise ValueError("Runtime output tuples cannot be empty.")
        return positional[0], tuple(positional[1:])
    return positional, ()


@dataclass(frozen=True, slots=True)
class RuntimeReturnedOutputMatcher:
    """Resolve one complete callable return against its declared positional ABI."""

    callable_contract: CallableContract
    returned_output: Any

    @property
    def canonical_output(self) -> Any:
        """Return the unmodified first position of the callable ABI."""

        return split_runtime_output(self.returned_output)[0]

    def resolve(self) -> dict[ArtifactSpecRef, Any]:
        """Resolve every output in the callable ABI."""

        output_specs = self.callable_contract.artifact_outputs.specs
        self._specs_by_ref(
            output_specs,
            contract_name="Callable output ABI",
        )
        canonical_output, trailing_values = split_runtime_output(self.returned_output)
        trailing_specs = self.callable_contract.trailing_return_output_specs.specs
        if len(trailing_values) != len(trailing_specs):
            raise ValueError(
                "Runtime callable trailing return count does not match its "
                f"declared trailing output slots: {len(trailing_values)} != "
                f"{len(trailing_specs)}."
            )
        resolved = {
            spec.ref(): value
            for spec, value in zip(
                trailing_specs,
                trailing_values,
                strict=True,
            )
        }
        resolved.update(
            self._canonical_values(
                self.callable_contract.canonical_return_output_specs.specs,
                canonical_output,
            )
        )
        return resolved

    def resolve_plan_values(
        self,
        selected_output_plans: tuple[ArtifactOutputPlan, ...],
    ) -> tuple[
        dict[ArtifactSpecRef, Any],
        tuple[RuntimeMatchedOutput, ...],
    ]:
        """Resolve the complete ABI and bind exact selected runtime plans once."""

        returned_values = self.resolve()
        specs_by_ref = self._specs_by_ref(
            self.callable_contract.artifact_outputs.specs,
            contract_name="Callable output ABI",
        )
        selected_refs: set[ArtifactSpecRef] = set()
        matched_outputs: list[RuntimeMatchedOutput] = []
        for plan in selected_output_plans:
            if not isinstance(plan, ArtifactOutputPlan):
                raise TypeError(
                    "Selected runtime outputs must be ArtifactOutputPlan values, "
                    f"got {type(plan).__name__}."
                )
            ref = plan.ref()
            if ref in selected_refs:
                raise ValueError(
                    f"Selected runtime output plans contain duplicate ref {ref!r}."
                )
            selected_refs.add(ref)
            spec = specs_by_ref.get(ref)
            if spec is None:
                raise ValueError(
                    f"Selected output plan {ref!r} is not declared by the callable ABI."
                )
            matched_outputs.append((plan, spec, returned_values[ref]))
        return returned_values, tuple(matched_outputs)

    @classmethod
    def _canonical_values(
        cls,
        canonical_specs: tuple[ArtifactSpec, ...],
        canonical_output: Any,
    ) -> dict[ArtifactSpecRef, Any]:
        """Resolve named outputs represented by the one canonical return slot."""

        if not canonical_specs:
            return {}
        if len(canonical_specs) == 1:
            spec = canonical_specs[0]
            return {spec.ref(): canonical_output}
        if not isinstance(canonical_output, AlignedImageStack):
            raise TypeError(
                "Multiple canonical output specs require an AlignedImageStack with "
                "one exact named slice context per output."
            )
        if not canonical_output.slice_contexts:
            raise ValueError(
                "Multiple canonical output specs require exact AlignedImageStack "
                "slice contexts; positional slice order is not artifact identity."
            )

        specs_by_context = {
            (spec.name, spec.artifact_type.value): spec for spec in canonical_specs
        }
        if len(specs_by_context) != len(canonical_specs):
            raise ValueError("Canonical output ABI contains duplicate named contexts.")
        resolved: dict[ArtifactSpecRef, Any] = {}
        for payload, context in zip(
            canonical_output.slices,
            canonical_output.slice_contexts,
            strict=True,
        ):
            if context.output_kind != AlignedImageSliceContext.MAIN_FLOW_OUTPUT_KIND:
                raise ValueError(
                    "Canonical AlignedImageStack contains a non-main-flow slice "
                    f"context: {context!r}."
                )
            context_key = (context.output_key, context.artifact_kind)
            spec = specs_by_context.get(context_key)
            if spec is None:
                raise ValueError(
                    "Canonical AlignedImageStack context is not declared by the "
                    f"callable ABI: {context!r}."
                )
            ref = spec.ref()
            if ref in resolved:
                raise ValueError(
                    "Canonical AlignedImageStack contains duplicate context for "
                    f"{ref!r}."
                )
            resolved[ref] = payload
        missing = tuple(
            spec.ref() for spec in canonical_specs if spec.ref() not in resolved
        )
        if missing:
            raise ValueError(
                "Canonical AlignedImageStack does not carry every declared output: "
                f"{missing!r}."
            )
        return resolved

    @staticmethod
    def _specs_by_ref(
        specs: tuple[ArtifactSpec, ...],
        *,
        contract_name: str,
    ) -> dict[ArtifactSpecRef, ArtifactSpec]:
        by_ref: dict[ArtifactSpecRef, ArtifactSpec] = {}
        for spec in specs:
            ref = spec.ref()
            if ref in by_ref:
                raise ValueError(
                    f"{contract_name} contains duplicate artifact ref {ref!r}."
                )
            by_ref[ref] = spec
        return by_ref
