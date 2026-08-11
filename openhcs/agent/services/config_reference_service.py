"""Declaration-derived RST reference projection for OpenHCS configuration."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from openhcs.agent.dto.config import ConfigFieldSchema
from openhcs.agent.services.config_service import (
    agent_config_declaration_from_request,
)


CONFIG_REFERENCE_DIRECTIVE = "openhcs-config-reference"
_CONFIG_REFERENCE_DIRECTIVE_PATTERN = re.compile(
    rf"^\.\.\s+{CONFIG_REFERENCE_DIRECTIVE}::(?:\s+(?P<config_name>\S+))?\s*$"
)


@dataclass(frozen=True, slots=True)
class ConfigReferenceRstRenderer:
    """Render configuration reference facts from their nominal declarations."""

    def render(self, config_name: str) -> tuple[str, ...]:
        declaration = agent_config_declaration_from_request(config_name)
        fields = tuple(
            field for field in declaration.reflected_fields() if not field.ui_hidden
        )
        return (
            f"Configuration type: ``{declaration.display_name()}``",
            "",
            f"Authoring path: ``{declaration.authoring_path}``",
            "",
            *self._render_field_groups(declaration.display_name(), fields),
        )

    def _render_field_groups(
        self,
        config_type_name: str,
        fields: tuple[ConfigFieldSchema, ...],
    ) -> tuple[str, ...]:
        roots = tuple(
            dict.fromkeys(self._root_path(field.path) for field in fields)
        )
        fields_by_root = tuple(
            (
                root,
                tuple(
                    candidate
                    for candidate in fields
                    if candidate.path == root
                    or candidate.path.startswith(f"{root}.")
                    or candidate.path.startswith(f"{root}[]")
                ),
            )
            for root in roots
        )
        standalone_fields = tuple(
            group_fields[0]
            for _, group_fields in fields_by_root
            if len(group_fields) == 1
        )
        groups = (
            *(
                ((f"{config_type_name}: root fields", standalone_fields),)
                if standalone_fields
                else ()
            ),
            *(
                (f"{config_type_name}: {root}", group_fields)
                for root, group_fields in fields_by_root
                if len(group_fields) > 1
            ),
        )
        return tuple(
            line
            for group_name, group_fields in groups
            for line in self._render_group(group_name, group_fields)
        )

    @staticmethod
    def _root_path(path: str) -> str:
        return re.split(r"[.\[]", path, maxsplit=1)[0]

    def _render_group(
        self,
        group_name: str,
        fields: tuple[ConfigFieldSchema, ...],
    ) -> tuple[str, ...]:
        title = group_name
        return (
            title,
            "^" * len(title),
            "",
            *(
                line
                for field in fields
                for line in self._render_field(field)
            ),
        )

    def _render_field(self, field: ConfigFieldSchema) -> tuple[str, ...]:
        description = self._single_line(field.description)
        if not description:
            raise ValueError(
                "Configuration reference cannot render an undocumented field: "
                f"{field.path} ({field.declaring_type})"
            )
        facts = [
            f"Type: ``{field.type_repr}``",
            self._default_fact(field),
        ]
        if field.enum_values:
            facts.append(f"Accepted values: {self._literal_values(field.enum_values)}")
        if field.registry_values:
            facts.append(
                f"Registered values: {self._literal_values(field.registry_values)}"
            )
        if field.inheritable:
            facts.append("Inheritance: unresolved ``None`` inherits from the wider scope")
        if field.declaring_type:
            facts.append(f"Declared by: ``{field.declaring_type}``")
        return (
            f"``{field.path}``",
            f"  {description}",
            "",
            *(f"  * {fact}" for fact in facts),
            "",
        )

    @staticmethod
    def _default_fact(field: ConfigFieldSchema) -> str:
        if field.required:
            return "Default: required"
        if field.default_repr is None:
            return "Default: ``None``"
        return f"Default: ``{field.default_repr}``"

    @staticmethod
    def _literal_values(values: Iterable[str]) -> str:
        return ", ".join(f"``{value}``" for value in values)

    @staticmethod
    def _single_line(value: str | None) -> str:
        return " ".join((value or "").split())


def expand_config_reference_directives(
    lines: tuple[str, ...],
) -> tuple[str, ...]:
    """Expand declared config-reference directives for non-Sphinx consumers."""

    matches = tuple(
        (index, match)
        for index, line in enumerate(lines)
        if (match := _CONFIG_REFERENCE_DIRECTIVE_PATTERN.fullmatch(line)) is not None
    )
    if not matches:
        return lines

    renderer = ConfigReferenceRstRenderer()
    expanded: list[str] = []
    match_by_index = dict(matches)
    for index, line in enumerate(lines):
        match = match_by_index.get(index)
        if match is None:
            expanded.append(line)
            continue
        config_name = match.group("config_name")
        if config_name is None:
            raise ValueError(
                f"{CONFIG_REFERENCE_DIRECTIVE} requires a config owner argument"
            )
        expanded.extend(renderer.render(config_name))
    return tuple(expanded)
