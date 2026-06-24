"""Config reflection and draft storage for OpenHCS agent integrations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import MISSING, fields, is_dataclass
from enum import Enum
from itertools import count
from types import UnionType
from typing import Annotated, TypeAlias, Union, get_args, get_origin

from objectstate import get_base_type_for_lazy

from openhcs.agent.dto.common import (
    AgentError,
    JsonValue,
    RenderedSource,
    SCHEMA_VERSION,
)
from openhcs.agent.dto.config import (
    ConfigFieldSchema,
    ConfigPatch,
    ConfigRef,
    ConfigSchema,
    ConfigValidationResult,
)
from openhcs.core.config import GlobalPipelineConfig, PipelineConfig
from openhcs.agent.services.source_rendering_service import PythonSourceAssignmentKind


AgentConfig: TypeAlias = GlobalPipelineConfig | PipelineConfig


class AgentConfigKind(Enum):
    GLOBAL = (
        "global",
        GlobalPipelineConfig,
        ("globalpipelineconfig",),
    )
    PIPELINE = (
        "pipeline",
        PipelineConfig,
        ("pipelineconfig",),
    )

    @property
    def config_class(self) -> type[AgentConfig]:
        return self.value[1]

    @classmethod
    def from_request(cls, config_type: str) -> "AgentConfigKind":
        normalized = config_type.casefold()
        for kind in cls:
            primary_name, _config_class, aliases = kind.value
            if normalized == primary_name or normalized in aliases:
                return kind
        raise ValueError(
            "config_type must be one of: global, GlobalPipelineConfig, "
            "pipeline, PipelineConfig"
        )


class ConfigService:
    """Reflect and create OpenHCS config objects without PyQt dependencies."""

    def __init__(self) -> None:
        self._configs: dict[str, AgentConfig] = {}
        self._counter = count(1)

    def describe_schema(self, config_type: str) -> ConfigSchema:
        cls = self._config_class(config_type)
        return ConfigSchema(
            schema_version=SCHEMA_VERSION,
            config_type=cls.__name__,
            fields=tuple(_field_schema(cls)),
        )

    def create(
        self,
        config_type: str,
        patch: ConfigPatch | None = None,
    ) -> ConfigRef:
        cls = self._config_class(config_type)
        values = _patch_values(cls, patch)
        instance = cls(**values)
        config_id = f"config-{next(self._counter)}"
        self._configs[config_id] = instance
        return ConfigRef(
            config_id=config_id,
            config_type=cls.__name__,
            uri=f"openhcs://configs/{config_id}",
        )

    def validate_patch(
        self,
        config_type: str,
        patch: ConfigPatch,
    ) -> ConfigValidationResult:
        try:
            config_ref = self.create(config_type, patch)
        except Exception as exc:
            return ConfigValidationResult(
                schema_version=SCHEMA_VERSION,
                valid=False,
                errors=(
                    AgentError.from_exception("config_patch_invalid", exc),
                ),
            )
        return ConfigValidationResult(
            schema_version=SCHEMA_VERSION,
            valid=True,
            config_ref=config_ref,
        )

    def resolve_ref(self, config_ref: ConfigRef | str) -> AgentConfig:
        config_id = config_ref.config_id if isinstance(config_ref, ConfigRef) else config_ref
        try:
            return self._configs[config_id]
        except KeyError as exc:
            raise KeyError(f"Unknown OpenHCS config_id: {config_id}") from exc

    def render_source(
        self,
        config_ref: ConfigRef | str,
        *,
        clean: bool = True,
    ) -> RenderedSource:
        instance = self.resolve_ref(config_ref)
        return RenderedSource(
            schema_version=SCHEMA_VERSION,
            title=f"{type(instance).__name__} source",
            source=PythonSourceAssignmentKind.CONFIG.assignment(instance, clean).render(),
        )

    def _config_class(self, config_type: str) -> type[AgentConfig]:
        return AgentConfigKind.from_request(config_type).config_class


def _patch_values(
    cls: type[AgentConfig],
    patch: ConfigPatch | None,
) -> dict[str, object]:
    if patch is None:
        return {}
    if not is_dataclass(cls):
        return dict(patch.values)
    return _coerce_dataclass_patch_values(cls, patch.values)


def _coerce_dataclass_patch_values(
    cls: type,
    values: Mapping[str, JsonValue],
) -> dict[str, object]:
    field_by_name = {field.name: field for field in fields(cls)}
    return {
        name: (
            _coerce_patch_value(field_by_name[name].type, value)
            if name in field_by_name
            else value
        )
        for name, value in values.items()
    }


def _coerce_patch_value(field_type, value: JsonValue) -> object:
    unwrapped_type = _unwrap_annotated(field_type)
    if value is None:
        return None

    dataclass_type = _dataclass_type(unwrapped_type)
    if dataclass_type is not None and isinstance(value, Mapping):
        return dataclass_type(**_coerce_dataclass_patch_values(dataclass_type, value))

    enum_type = _unwrap_enum(unwrapped_type)
    if enum_type is not None:
        return _coerce_enum_value(enum_type, value)

    return value


def _dataclass_type(field_type) -> type | None:
    field_type = _unwrap_annotated(field_type)
    if isinstance(field_type, type) and is_dataclass(field_type):
        return field_type
    origin = get_origin(field_type)
    if origin not in (Union, UnionType):
        return None
    for arg in get_args(field_type):
        nested_type = _dataclass_type(arg)
        if nested_type is not None:
            return nested_type
    return None


def _coerce_enum_value(enum_type: type[Enum], value: object) -> Enum:
    if isinstance(value, enum_type):
        return value
    for member in enum_type:
        if value == member.value or value == member.name:
            return member
    raise ValueError(
        f"{value!r} is not a valid {enum_type.__module__}.{enum_type.__name__}"
    )


def _unwrap_annotated(field_type):
    if get_origin(field_type) is Annotated:
        return get_args(field_type)[0]
    return field_type


def _field_schema(cls: type) -> tuple[ConfigFieldSchema, ...]:
    if not is_dataclass(cls):
        raise TypeError(f"{cls.__name__} is not a dataclass config type")
    return tuple(_schema_for_field(cls, field) for field in fields(cls))


def _schema_for_field(cls: type, field) -> ConfigFieldSchema:
    field_type = field.type
    lazy_base = _lazy_base_type(field_type)
    ui_hidden = False
    if "ui_hidden" in field.metadata:
        ui_hidden = bool(field.metadata["ui_hidden"])
    return ConfigFieldSchema(
        path=field.name,
        type_repr=_type_repr(field_type),
        default_repr=_default_repr(field),
        required=_is_required(field),
        description=None,
        enum_values=_enum_values(lazy_base or field_type),
        ui_hidden=ui_hidden,
        lazy=lazy_base is not None,
    )


def _lazy_base_type(field_type) -> type | None:
    if not isinstance(field_type, type):
        return None
    return get_base_type_for_lazy(field_type)


def _type_repr(field_type) -> str:
    if isinstance(field_type, type):
        return f"{field_type.__module__}.{field_type.__name__}"
    return str(field_type)


def _default_repr(field) -> str | None:
    if field.default is not MISSING:
        return repr(field.default)
    if field.default_factory is not MISSING:
        factory = field.default_factory
        name = factory.__name__
        module = factory.__module__
        return f"{module}.{name}()"
    return None


def _is_required(field) -> bool:
    return field.default is MISSING and field.default_factory is MISSING


def _enum_values(field_type) -> tuple[str, ...]:
    enum_type = _unwrap_enum(field_type)
    if enum_type is None:
        return ()
    return tuple(
        member.value if isinstance(member.value, str) else member.name
        for member in enum_type
    )


def _unwrap_enum(field_type) -> type[Enum] | None:
    field_type = _unwrap_annotated(field_type)
    if isinstance(field_type, type) and issubclass(field_type, Enum):
        return field_type
    origin = get_origin(field_type)
    if origin is None:
        return None
    for arg in get_args(field_type):
        unwrapped = _unwrap_enum(arg)
        if unwrapped is not None:
            return unwrapped
    return None
