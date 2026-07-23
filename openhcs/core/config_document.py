"""Canonical Python document contract for one OpenHCS configuration object."""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from typing import TypeVar, cast


ConfigT = TypeVar("ConfigT")


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
        exec(code, namespace)
        return cls.from_namespace(
            namespace,
            expected_config_type=expected_config_type,
        )

    @classmethod
    def render(
        cls,
        config: ConfigT,
        *,
        expected_config_type: type[ConfigT],
        clean_mode: bool = True,
    ) -> str:
        """Render a validated config object as reviewable Python source."""

        import openhcs.serialization.pycodify_formatters  # noqa: F401
        from pycodify import Assignment

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
