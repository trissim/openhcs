"""Prompt/context projection for OpenHCS authoring agents."""

from __future__ import annotations

from enum import Enum

from openhcs.agent.dto.authoring import AuthoringContext
from openhcs.agent.dto.common import SCHEMA_VERSION
from openhcs.agent.dto.config import ConfigSchema
from openhcs.agent.services.config_service import ConfigService
from openhcs.agent.services.function_catalog_service import FunctionCatalogService


class AuthoringContextKind(Enum):
    PIPELINE = "pipeline"
    CUSTOM_FUNCTION = "custom_function"

    @classmethod
    def from_request(cls, kind: str) -> "AuthoringContextKind":
        try:
            return cls(kind.casefold())
        except ValueError as exc:
            raise ValueError("kind must be 'pipeline' or 'custom_function'") from exc

    @property
    def rules_section(self) -> str:
        if self is AuthoringContextKind.CUSTOM_FUNCTION:
            return """=== CUSTOM FUNCTION RULES ===
- Custom processing functions should have an explicit first image-like argument.
- Prefer concrete typed parameters with serializable defaults.
- Do not close over GUI, filesystem, or live viewer state."""
        return """=== PIPELINE AUTHORING RULES ===
- Author pipelines as ordered FunctionStep objects.
- Function references should come from the registry when using MCP tools.
- Search functions first, then call openhcs_describe_function before adding a step with non-default parameters.
- Rendered Python source is review/export output; the MCP draft pipeline is the canonical v1 state.
- Use LazyProcessingConfig for per-step axis/input-source semantics when needed."""


class ConfigFieldRequirementLabel(Enum):
    REQUIRED = "required"
    OPTIONAL = "optional"

    @classmethod
    def from_required(cls, required: bool) -> "ConfigFieldRequirementLabel":
        return _CONFIG_FIELD_REQUIREMENT_BY_STATE[required]


class ConfigFieldResolutionLabel(Enum):
    LAZY = ", lazy"
    EAGER = ""

    @classmethod
    def from_lazy(cls, lazy: bool) -> "ConfigFieldResolutionLabel":
        return _CONFIG_FIELD_RESOLUTION_BY_STATE[lazy]


_CONFIG_FIELD_REQUIREMENT_BY_STATE = {
    True: ConfigFieldRequirementLabel.REQUIRED,
    False: ConfigFieldRequirementLabel.OPTIONAL,
}
_CONFIG_FIELD_RESOLUTION_BY_STATE = {
    True: ConfigFieldResolutionLabel.LAZY,
    False: ConfigFieldResolutionLabel.EAGER,
}


class AgentAuthoringContextService:
    """Build bounded, registry-grounded context for agents authoring OpenHCS."""

    def __init__(
        self,
        function_catalog: FunctionCatalogService | None = None,
        config_service: ConfigService | None = None,
        *,
        max_functions: int = 25,
        max_config_fields: int = 8,
    ) -> None:
        self._function_catalog = function_catalog or FunctionCatalogService()
        self._config_service = config_service or ConfigService()
        self._max_functions = max_functions
        self._max_config_fields = max_config_fields

    def get_authoring_context(self, kind: str = "pipeline") -> AuthoringContext:
        context_kind = AuthoringContextKind.from_request(kind)

        parts = [
            _core_imports_section(),
            context_kind.rules_section,
            self._config_schema_section(),
            self._function_catalog_section(),
        ]
        return AuthoringContext(
            schema_version=SCHEMA_VERSION,
            kind=context_kind.value,
            content="\n\n".join(parts),
        )

    def _function_catalog_section(self) -> str:
        page = self._function_catalog.search(
            limit=self._max_functions,
            compact_signatures=True,
        )
        lines = [
            "=== REGISTERED OPENHCS FUNCTIONS ===",
            "Use function_id values with MCP authoring tools; use imports only when rendering reviewed Python source.",
        ]
        current_library = None
        for entry in page.items:
            if entry.library != current_library:
                current_library = entry.library
                lines.append(f"\n## {entry.library}")
            if entry.summary is None:
                summary = ""
            else:
                summary = f" - {entry.summary}"
            lines.append(f"- {entry.function_id}: `{entry.signature}`{summary}")
        if page.total > len(page.items):
            lines.append(f"\n... {page.total - len(page.items)} more functions are available through openhcs_search_functions.")
        return "\n".join(lines)

    def _config_schema_section(self) -> str:
        schemas = (
            self._config_service.describe_schema("global"),
            self._config_service.describe_schema("pipeline"),
        )
        lines = [
            "=== CONFIG SCHEMA HINTS ===",
            "Use openhcs_describe_config_schema for the full reflected schema before setting non-obvious fields.",
        ]
        for schema in schemas:
            lines.extend(self._schema_lines(schema))
        return "\n".join(lines)

    def _schema_lines(self, schema: ConfigSchema) -> list[str]:
        visible_fields = [
            field
            for field in schema.fields
            if not field.ui_hidden
        ]
        lines = [f"\n## {schema.config_type}"]
        for field in visible_fields[:self._max_config_fields]:
            requirement = ConfigFieldRequirementLabel.from_required(field.required)
            resolution = ConfigFieldResolutionLabel.from_lazy(field.lazy)
            lines.append(
                f"- {field.path}: {field.type_repr} ({requirement.value}{resolution.value})"
            )
        if len(visible_fields) > self._max_config_fields:
            remaining = len(visible_fields) - self._max_config_fields
            lines.append(f"- ... {remaining} more fields")
        return lines


def _core_imports_section() -> str:
    return """=== CORE PIPELINE IMPORTS ===
from openhcs.core.steps.function_step import FunctionStep
from openhcs.core.config import (
    LazyProcessingConfig,
    LazyDtypeConfig,
    LazyStepMaterializationConfig,
    LazyNapariStreamingConfig,
    LazyFijiStreamingConfig,
)
from openhcs.constants.constants import VariableComponents
from openhcs.constants.input_source import InputSource"""
