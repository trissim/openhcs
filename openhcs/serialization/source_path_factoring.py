"""Document-level factoring for semantically typed Python source paths."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from openhcs.core.python_source_literal import PythonSourceLiteral
from pycodify import (
    Assignment,
    BlankLine,
    CodeBlock,
    FormatContext,
    generate_python_source,
    to_source,
)


@dataclass(slots=True)
class SourcePathOccurrenceCollector:
    """Ordered render-pass collector for actual or declaration-owned paths."""

    values: list[Path] = field(default_factory=list)

    def record(self, value: Path) -> None:
        self.values.append(value)


@dataclass(frozen=True, slots=True)
class FactoredPathSourceLiteral(PythonSourceLiteral):
    """One factoring-plan-owned Python path expression."""

    code: str
    imports: frozenset[tuple[str, str]] = frozenset()

    def source_literal(self) -> str:
        return self.code

    def source_literal_imports(self) -> frozenset[tuple[str, str]]:
        return self.imports


@dataclass(frozen=True, slots=True)
class SourcePathFactoringPlan:
    """Immutable path bindings and substitutions for one source document."""

    bindings: tuple[Assignment, ...]
    _expressions: Mapping[Path, PythonSourceLiteral]

    @classmethod
    def from_occurrences(
        cls,
        occurrences: Iterable[Path],
    ) -> "SourcePathFactoringPlan":
        normalized = tuple(
            _lexically_normalized(path)
            for path in occurrences
            if path.is_absolute()
        )
        counts = Counter(normalized)
        ordered_values = tuple(dict.fromkeys(normalized))
        partitions: dict[tuple[str, str], list[Path]] = {}
        for value in ordered_values:
            first_component = value.parts[1] if len(value.parts) > 1 else ""
            partitions.setdefault((value.anchor, first_component), []).append(value)

        bindings: list[Assignment] = []
        root_binding_by_value: dict[Path, tuple[str, Path]] = {}
        root_index = 0
        for values in partitions.values():
            if len(values) < 2:
                continue
            root = _common_ancestor(values)
            if root is None or root == Path(root.anchor):
                continue
            root_index += 1
            root_name = "path_root" if root_index == 1 else f"path_root_{root_index}"
            bindings.append(
                Assignment(
                    root_name,
                    FactoredPathSourceLiteral(
                        f"Path({str(root)!r})",
                        frozenset((("pathlib", "Path"),)),
                    ),
                )
            )
            for value in values:
                root_binding_by_value[value] = (root_name, root)

        expressions: dict[Path, PythonSourceLiteral] = {}
        path_index = 0
        for value in ordered_values:
            root_binding = root_binding_by_value.get(value)
            factored_expression = (
                None
                if root_binding is None
                else _descendant_expression(root_binding[0], root_binding[1], value)
            )
            is_root_binding = root_binding is not None and value == root_binding[1]
            if counts[value] > 1 and not is_root_binding:
                path_index += 1
                path_name = f"path_{path_index}"
                binding_value = factored_expression or FactoredPathSourceLiteral(
                    f"Path({str(value)!r})",
                    frozenset((("pathlib", "Path"),)),
                )
                bindings.append(Assignment(path_name, binding_value))
                expressions[value] = FactoredPathSourceLiteral(path_name)
                continue
            if factored_expression is not None:
                expressions[value] = factored_expression

        return cls(
            bindings=tuple(bindings),
            _expressions=MappingProxyType(expressions),
        )

    def expression_for(self, value: Path) -> PythonSourceLiteral | None:
        if not value.is_absolute():
            return None
        return self._expressions.get(_lexically_normalized(value))


@dataclass(frozen=True, slots=True)
class OpenHCSPythonSourceDocument:
    """Complete OpenHCS Python document rendered with one factoring plan."""

    body: object
    header: str = ""
    clean_mode: bool = False

    def render(self) -> str:
        collector = SourcePathOccurrenceCollector()
        discovery_context = FormatContext(
            clean_mode=self.clean_mode,
            extensions=MappingProxyType(
                {SourcePathOccurrenceCollector: collector}
            ),
        )
        to_source(self.body, discovery_context)
        plan = SourcePathFactoringPlan.from_occurrences(collector.values)
        body_items = (
            self.body.items
            if isinstance(self.body, CodeBlock)
            else (self.body,)
        )
        items: tuple[object, ...] = body_items
        if plan.bindings:
            items = (*plan.bindings, BlankLine(), *body_items)
        render_context = FormatContext(
            clean_mode=self.clean_mode,
            extensions=MappingProxyType({SourcePathFactoringPlan: plan}),
        )
        return generate_python_source(
            CodeBlock.from_items(items),
            self.header,
            self.clean_mode,
            context=render_context,
        )


def _lexically_normalized(value: Path) -> Path:
    if not value.is_absolute():
        return value
    components: list[str] = []
    for component in value.parts[1:]:
        if component in ("", "."):
            continue
        if component == "..":
            if components:
                components.pop()
            continue
        components.append(component)
    return Path(value.anchor, *components)


def _common_ancestor(values: list[Path]) -> Path | None:
    first_parts = values[0].parts
    common_count = 0
    for components in zip(*(value.parts for value in values), strict=False):
        if len(set(components)) != 1:
            break
        common_count += 1
    if common_count == 0:
        return None
    return Path(first_parts[0], *first_parts[1:common_count])


def _descendant_expression(
    root_name: str,
    root: Path,
    value: Path,
) -> FactoredPathSourceLiteral:
    relative_parts = value.parts[len(root.parts):]
    code = root_name
    for part in relative_parts:
        code = f"{code} / {part!r}"
    return FactoredPathSourceLiteral(code)
