"""Canonical PlateManager Python document shared by every UI frontend."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Self

from openhcs.core.config import GlobalPipelineConfig, PipelineConfig
from openhcs.core.function_step_transport import FunctionStepTransportAuthority
from openhcs.core.steps.function_step import FunctionStep
from openhcs.ui.shared.plate_scope_identity import PlateScopeIdentity


class PlateManagerCodeNamespaceField(str, Enum):
    """Exact public assignments in a PlateManager code document."""

    PLATE_PATHS = "plate_paths"
    GLOBAL_CONFIG = "global_config"
    PER_PLATE_CONFIGS = "per_plate_configs"
    PIPELINE_DATA = "pipeline_data"

    @classmethod
    def allowed_assignment_names(cls) -> frozenset[str]:
        return frozenset(field.value for field in cls)


class PlateManagerCodeNamespace(dict):
    """Nominal execution namespace for PlateManager documents."""

    @classmethod
    def from_mapping(cls, namespace: Mapping[str, object]) -> Self:
        code_namespace = cls()
        code_namespace.update(namespace)
        return code_namespace

    def require(self, field: PlateManagerCodeNamespaceField) -> object:
        if field.value not in self:
            raise ValueError(f"PlateManager code document must define {field.value!r}.")
        return self[field.value]


@dataclass(frozen=True, slots=True)
class PlateManagerOrchestratorCodePayload:
    """Validated semantic contents of one PlateManager code document."""

    plate_paths: tuple[str, ...]
    global_pipeline_config: GlobalPipelineConfig
    per_plate_configs: dict[str, PipelineConfig]
    pipeline_data: dict[str, list[FunctionStep]]


class PlateManagerCodeDocumentAuthority:
    """Normalize, render, and parse the canonical PlateManager document."""

    HEADER = "# Edit this orchestrator configuration and save to apply changes"

    @classmethod
    def from_values(
        cls,
        *,
        plate_paths: Sequence[str | Path],
        global_pipeline_config: GlobalPipelineConfig,
        per_plate_configs: Mapping[str | Path, PipelineConfig],
        pipeline_data: Mapping[str | Path, Sequence[FunctionStep]],
    ) -> PlateManagerOrchestratorCodePayload:
        namespace = PlateManagerCodeNamespace(
            {
                PlateManagerCodeNamespaceField.PLATE_PATHS.value: list(plate_paths),
                PlateManagerCodeNamespaceField.GLOBAL_CONFIG.value: (
                    global_pipeline_config
                ),
                PlateManagerCodeNamespaceField.PER_PLATE_CONFIGS.value: dict(
                    per_plate_configs
                ),
                PlateManagerCodeNamespaceField.PIPELINE_DATA.value: {
                    path: list(steps) for path, steps in pipeline_data.items()
                },
            }
        )
        return cls.from_namespace(namespace)

    @classmethod
    def from_namespace(
        cls,
        namespace: Mapping[str, object],
    ) -> PlateManagerOrchestratorCodePayload:
        values = PlateManagerCodeNamespace.from_mapping(namespace)
        plate_paths_value = values.require(PlateManagerCodeNamespaceField.PLATE_PATHS)
        global_config_value = values.require(
            PlateManagerCodeNamespaceField.GLOBAL_CONFIG
        )
        per_plate_configs_value = values.require(
            PlateManagerCodeNamespaceField.PER_PLATE_CONFIGS
        )
        pipeline_data_value = values.require(
            PlateManagerCodeNamespaceField.PIPELINE_DATA
        )

        if not isinstance(plate_paths_value, list):
            raise TypeError("plate_paths must be a list of strings or Paths.")
        plate_paths = tuple(
            cls._scope_id(path, field_name="plate_paths") for path in plate_paths_value
        )
        if len(set(plate_paths)) != len(plate_paths):
            raise ValueError("plate_paths must not contain duplicate plate scopes.")

        if not isinstance(global_config_value, GlobalPipelineConfig):
            raise TypeError("global_config must be a GlobalPipelineConfig.")

        if not isinstance(per_plate_configs_value, dict):
            raise TypeError(
                "per_plate_configs must be a dict of PipelineConfig values."
            )
        per_plate_configs: dict[str, PipelineConfig] = {}
        for path, pipeline_config in per_plate_configs_value.items():
            scope_id = cls._scope_id(path, field_name="per_plate_configs key")
            if not isinstance(pipeline_config, PipelineConfig):
                raise TypeError(
                    "per_plate_configs values must be PipelineConfig instances."
                )
            per_plate_configs[scope_id] = pipeline_config

        if not isinstance(pipeline_data_value, dict):
            raise TypeError(
                "pipeline_data must be a dict of plate paths to FunctionStep lists."
            )
        pipeline_data: dict[str, list[FunctionStep]] = {}
        for path, steps in pipeline_data_value.items():
            scope_id = cls._scope_id(path, field_name="pipeline_data key")
            if not isinstance(steps, list):
                raise TypeError("pipeline_data values must be FunctionStep lists.")
            pipeline_data[scope_id] = FunctionStepTransportAuthority.normalize_pipeline(
                steps
            )

        plate_scope_ids = frozenset(plate_paths)
        if frozenset(per_plate_configs) != plate_scope_ids:
            raise ValueError("per_plate_configs keys must exactly match plate_paths.")
        if frozenset(pipeline_data) != plate_scope_ids:
            raise ValueError("pipeline_data keys must exactly match plate_paths.")

        return PlateManagerOrchestratorCodePayload(
            plate_paths=plate_paths,
            global_pipeline_config=global_config_value,
            per_plate_configs=per_plate_configs,
            pipeline_data=pipeline_data,
        )

    @classmethod
    def from_source(cls, source: str) -> PlateManagerOrchestratorCodePayload:
        namespace = PlateManagerCodeNamespace()
        exec(compile(source, "<openhcs-plate-manager-document>", "exec"), namespace)
        return cls.from_namespace(namespace)

    @classmethod
    def to_namespace(
        cls,
        payload: PlateManagerOrchestratorCodePayload,
    ) -> PlateManagerCodeNamespace:
        normalized = cls.from_values(
            plate_paths=payload.plate_paths,
            global_pipeline_config=payload.global_pipeline_config,
            per_plate_configs=payload.per_plate_configs,
            pipeline_data=payload.pipeline_data,
        )
        return PlateManagerCodeNamespace(
            {
                PlateManagerCodeNamespaceField.PLATE_PATHS.value: list(
                    normalized.plate_paths
                ),
                PlateManagerCodeNamespaceField.GLOBAL_CONFIG.value: (
                    normalized.global_pipeline_config
                ),
                PlateManagerCodeNamespaceField.PER_PLATE_CONFIGS.value: dict(
                    normalized.per_plate_configs
                ),
                PlateManagerCodeNamespaceField.PIPELINE_DATA.value: {
                    path: list(steps)
                    for path, steps in normalized.pipeline_data.items()
                },
            }
        )

    @classmethod
    def render(
        cls,
        payload: PlateManagerOrchestratorCodePayload,
        *,
        clean_mode: bool = True,
    ) -> str:
        import openhcs.serialization.pycodify_formatters  # noqa: F401
        from pycodify import Assignment, BlankLine, CodeBlock
        from openhcs.serialization.source_path_factoring import (
            OpenHCSPythonSourceDocument,
        )

        normalized = cls.from_values(
            plate_paths=payload.plate_paths,
            global_pipeline_config=payload.global_pipeline_config,
            per_plate_configs=payload.per_plate_configs,
            pipeline_data=payload.pipeline_data,
        )
        code_value_by_scope = {
            scope_id: PlateScopeIdentity.from_scope_id(scope_id).code_value()
            for scope_id in normalized.plate_paths
        }
        body = CodeBlock.from_items(
            (
                Assignment(
                    PlateManagerCodeNamespaceField.PLATE_PATHS.value,
                    [
                        code_value_by_scope[scope_id]
                        for scope_id in normalized.plate_paths
                    ],
                ),
                BlankLine(),
                Assignment(
                    PlateManagerCodeNamespaceField.GLOBAL_CONFIG.value,
                    normalized.global_pipeline_config,
                ),
                BlankLine(),
                Assignment(
                    PlateManagerCodeNamespaceField.PER_PLATE_CONFIGS.value,
                    {
                        code_value_by_scope[scope_id]: config
                        for scope_id, config in normalized.per_plate_configs.items()
                    },
                ),
                BlankLine(),
                Assignment(
                    PlateManagerCodeNamespaceField.PIPELINE_DATA.value,
                    {
                        code_value_by_scope[scope_id]: steps
                        for scope_id, steps in normalized.pipeline_data.items()
                    },
                ),
            )
        )
        return OpenHCSPythonSourceDocument(
            body,
            header=cls.HEADER,
            clean_mode=clean_mode,
        ).render()

    @staticmethod
    def _scope_id(value: object, *, field_name: str) -> str:
        if not isinstance(value, (str, Path)):
            raise TypeError(f"{field_name} must be a plate path string or Path.")
        return PlateScopeIdentity.from_scope_id(str(value)).scope_id
