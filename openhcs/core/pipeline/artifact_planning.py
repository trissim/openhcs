"""Artifact declaration extraction for compiled function patterns."""

from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional

from openhcs.core.artifacts import ArtifactSpec
from openhcs.core.function_patterns import (
    DEFAULT_GROUP_KEY,
    iter_enabled_function_invocations,
)


@dataclass
class ArtifactDeclarations:
    """Artifact contracts declared by a function pattern."""

    output_names: set[str] = field(default_factory=set)
    output_groups: dict[str, set[Optional[str]]] = field(
        default_factory=lambda: defaultdict(set)
    )
    inputs: OrderedDict[str, ArtifactSpec] = field(default_factory=OrderedDict)
    materializations: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def empty(cls) -> "ArtifactDeclarations":
        return cls()


def normalize_pattern(pattern: Any) -> Iterator[tuple[Callable, str, int]]:
    """Extract enabled functions from any pattern with runtime invocation positions."""
    for invocation in iter_enabled_function_invocations(pattern):
        yield (
            invocation.func,
            invocation.key.group_key,
            invocation.key.position,
        )


def extract_artifact_declarations(pattern: Any) -> ArtifactDeclarations:
    """Extract artifact metadata and per-group ownership from a function pattern."""
    declarations = ArtifactDeclarations()

    for func, group_key, _ in normalize_pattern(pattern):
        normalized_key = None if group_key == DEFAULT_GROUP_KEY else group_key

        artifact_outputs = getattr(func, "__artifact_outputs__", {})
        declarations.output_names.update(artifact_outputs.keys())
        for output in artifact_outputs:
            declarations.output_groups[output].add(normalized_key)

        declarations.inputs.update(getattr(func, "__artifact_inputs__", {}))
        declarations.materializations.update(
            {
                name: spec.materialization
                for name, spec in artifact_outputs.items()
                if spec.materialization is not None
            }
        )

    return declarations
