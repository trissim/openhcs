"""Canonical Python document contract for one OpenHCS configuration object."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import MISSING, Field, fields
from enum import Enum
from typing import Any, TypeVar, cast, get_type_hints

ConfigT = TypeVar("ConfigT")
FieldT = TypeVar("FieldT")


class ConfigDocumentField(str, Enum):
    """Exact public assignment in a configuration Python document."""

    CONFIG = "config"


class ConfigDocumentAuthority:
    """Validate, parse, and render the canonical configuration document shape."""

    HEADER = "# OpenHCS configuration"

    @classmethod
    def from_namespace(
        cls,
        namespace: Mapping[str, object],
        *,
        expected_config_type: type[ConfigT],
    ) -> ConfigT:
        """Read the exact config assignment from an executed namespace."""

        config_field = ConfigDocumentField.CONFIG.value
        if config_field not in namespace:
            raise ValueError(f"Config document must define {config_field!r}.")

        config = namespace[config_field]
        cls._require_config_type(config, expected_config_type)
        return cast(ConfigT, config)

    @classmethod
    def from_source(
        cls,
        source: str,
        *,
        expected_config_type: type[ConfigT],
    ) -> ConfigT:
        """Execute Python source and read its canonical config assignment."""

        namespace: dict[str, object] = {}
        code = compile(source, "<openhcs-config-document>", "exec")
        exec(code, namespace)  # noqa: S102 - executable Python is the document format
        return cls.from_namespace(
            namespace,
            expected_config_type=expected_config_type,
        )

    @classmethod
    def project_dataclass_field_from_source(
        cls,
        source: str,
        *,
        expected_config_type: type[ConfigT],
        expected_field_type: type[FieldT],
    ) -> FieldT:
        """Evaluate one declared field from a canonical config constructor.

        Imports remain available to the field expression, but the containing
        dataclass is not constructed. This lets headless consumers project a
        runtime-owned nested declaration without loading unrelated UI runtime
        dependencies or maintaining a second cache.
        """

        resolved_field_types = get_type_hints(expected_config_type)
        matching_fields = tuple(
            dataclass_field
            for dataclass_field in fields(cast(Any, expected_config_type))
            if resolved_field_types[dataclass_field.name] is expected_field_type
        )
        if len(matching_fields) != 1:
            raise ValueError(
                f"{expected_config_type.__name__} must declare exactly one "
                f"{expected_field_type.__name__} field."
            )
        dataclass_field = matching_fields[0]

        module = ast.parse(source, filename="<openhcs-config-document>")
        config_field = ConfigDocumentField.CONFIG.value
        assignments = tuple(
            statement
            for statement in module.body
            if isinstance(statement, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == config_field
                for target in statement.targets
            )
        )
        if len(assignments) != 1:
            raise ValueError(
                f"Config document must assign {config_field!r} exactly once."
            )
        constructor = assignments[0].value
        if not isinstance(constructor, ast.Call):
            raise TypeError(
                f"Config document {config_field!r} must be a dataclass constructor."
            )

        field_values = tuple(
            keyword.value
            for keyword in constructor.keywords
            if keyword.arg == dataclass_field.name
        )
        if len(field_values) > 1:
            raise ValueError(
                f"Config document field {dataclass_field.name!r} is assigned more "
                "than once."
            )
        if not field_values:
            value = cls._dataclass_field_default(dataclass_field)
        else:
            projection_module = ast.Module(
                body=[
                    *(
                        statement
                        for statement in module.body
                        if isinstance(statement, (ast.Import, ast.ImportFrom))
                    ),
                    ast.Assign(
                        targets=[ast.Name(id=config_field, ctx=ast.Store())],
                        value=field_values[0],
                    ),
                ],
                type_ignores=[],
            )
            namespace: dict[str, object] = {}
            exec(  # noqa: S102 - evaluate the selected Python field expression
                compile(
                    ast.fix_missing_locations(projection_module),
                    "<openhcs-config-field>",
                    "exec",
                ),
                namespace,
            )
            value = namespace[config_field]

        if type(value) is not expected_field_type:
            raise TypeError(
                f"Config document field {dataclass_field.name!r} must be "
                f"{expected_field_type.__name__}, got {type(value).__name__}."
            )
        return cast(FieldT, value)

    @classmethod
    def render(
        cls,
        config: ConfigT,
        *,
        expected_config_type: type[ConfigT],
        clean_mode: bool = True,
    ) -> str:
        """Render a validated config object as reviewable Python source."""

        from pycodify import Assignment

        import openhcs.serialization.pycodify_formatters  # noqa: F401
        from openhcs.serialization.source_path_factoring import (
            OpenHCSPythonSourceDocument,
        )

        cls._require_config_type(config, expected_config_type)
        return OpenHCSPythonSourceDocument(
            Assignment(ConfigDocumentField.CONFIG.value, config),
            header=cls.HEADER,
            clean_mode=clean_mode,
        ).render()

    @staticmethod
    def _require_config_type(
        config: object,
        expected_config_type: type[ConfigT],
    ) -> None:
        if not isinstance(config, expected_config_type):
            config_field = ConfigDocumentField.CONFIG.value
            raise TypeError(
                f"Config document variable {config_field!r} must be "
                f"{expected_config_type.__name__}, got {type(config).__name__}."
            )

    @staticmethod
    def _dataclass_field_default(dataclass_field: Field[FieldT]) -> FieldT:
        if dataclass_field.default is not MISSING:
            return cast(FieldT, dataclass_field.default)
        if dataclass_field.default_factory is not MISSING:
            return dataclass_field.default_factory()
        raise ValueError(
            f"Config document omits required field {dataclass_field.name!r}."
        )
