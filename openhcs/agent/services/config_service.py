"""Config reflection and draft storage for OpenHCS agent integrations."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from abc import ABC
from dataclasses import MISSING, fields, is_dataclass
from enum import Enum
from itertools import count
from pathlib import Path
from types import UnionType
from typing import (
    Annotated,
    ClassVar,
    TypeAlias,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from metaclass_registry import AutoRegisterMeta
from objectstate import get_base_type_for_lazy
from pyqt_reactive.services.parameter_help_service import (
    class_docstring_text,
    dataclass_parameter_descriptions,
    parameter_description_from_target,
)

from openhcs.agent.dto.common import (
    AgentError,
    JsonValue,
    RenderedSource,
    SCHEMA_VERSION,
)
from openhcs.agent.dto.config import (
    ConfigFieldSchema,
    ConfigPatch,
    ConfigRegisteredType,
    ConfigRef,
    ConfigRegistrySchema,
    ConfigSchema,
    ConfigSchemaRequest,
    ConfigTypeSchema,
    ConfigValidationResult,
)
from openhcs.core.artifacts import ArtifactType
from openhcs.core.config import (
    GlobalPipelineConfig,
    PipelineConfig,
    StreamingConfig,
)
from openhcs.core.config_document import ConfigDocumentAuthority
from openhcs.core.steps.abstract import AbstractStep
from openhcs.core.steps.function_step import FunctionStep


AgentConfig: TypeAlias = GlobalPipelineConfig | PipelineConfig


class AgentConfigDeclaration(ABC, metaclass=AutoRegisterMeta):
    """Typed agent-facing declaration for one owner-derived authoring schema."""

    __registry_key__ = "config_name"
    __skip_if_no_key__ = True

    config_name: ClassVar[str | None] = None
    config_type: ClassVar[type]
    aliases: ClassVar[tuple[str, ...]] = ()
    authoring_path: ClassVar[str] = "ConfigPatch.values"
    draftable: ClassVar[bool] = True

    @classmethod
    def accepted_names(cls) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                (
                    cls.config_type.__name__,
                    cls.config_type.__name__.casefold(),
                    *cls.aliases,
                )
            )
        )

    @classmethod
    def matches(cls, requested: str) -> bool:
        normalized = requested.casefold()
        return any(alias.casefold() == normalized for alias in cls.accepted_names())

    @classmethod
    def display_name(cls) -> str:
        return cls.config_type.__name__

    @classmethod
    def reflected_fields(cls) -> tuple[ConfigFieldSchema, ...]:
        return _field_schema(cls.config_type)

    @classmethod
    def reflected_types(cls) -> tuple[type, ...]:
        return _dataclass_schema_types(cls.config_type)


class GlobalPipelineAgentConfigDeclaration(AgentConfigDeclaration):
    config_name = GlobalPipelineConfig.__name__
    config_type = GlobalPipelineConfig
    aliases = ("global",)


class PipelineAgentConfigDeclaration(AgentConfigDeclaration):
    config_name = PipelineConfig.__name__
    config_type = PipelineConfig
    aliases = ("pipeline",)


class FunctionStepAgentConfigDeclaration(AgentConfigDeclaration):
    """Project step config keys directly from the FunctionStep base signature."""

    config_name = FunctionStep.__name__
    config_type = FunctionStep
    aliases = ("step", "function_step")
    authoring_path = "FunctionStepAddRequest.step_config_overrides"
    draftable = False

    @classmethod
    def reflected_fields(cls) -> tuple[ConfigFieldSchema, ...]:
        return _function_step_config_field_schema()

    @classmethod
    def reflected_types(cls) -> tuple[type, ...]:
        return tuple(
            dict.fromkeys(
                (
                    AbstractStep,
                    *(
                        schema_type
                        for config_type in AbstractStep.config_classes_by_field_name().values()
                        for schema_type in _dataclass_schema_types(config_type)
                    ),
                )
            )
        )


def agent_config_declarations() -> tuple[type[AgentConfigDeclaration], ...]:
    return tuple(AgentConfigDeclaration.__registry__.values())


def agent_config_class_from_request(config_type: str) -> type[AgentConfig]:
    declarations = tuple(
        declaration
        for declaration in agent_config_declarations()
        if declaration.draftable
    )
    for declaration in declarations:
        if declaration.matches(config_type):
            return declaration.config_type
    accepted = ", ".join(declaration.display_name() for declaration in declarations)
    aliases = ", ".join(
        alias
        for declaration in declarations
        for alias in declaration.accepted_names()
        if alias != declaration.display_name()
    )
    raise ValueError(
        f"config_type must be one of: {accepted}; accepted aliases: {aliases}"
    )


def agent_config_declaration_from_request(
    config_type: str,
) -> type[AgentConfigDeclaration]:
    declarations = agent_config_declarations()
    for declaration in declarations:
        if declaration.matches(config_type):
            return declaration
    accepted = ", ".join(
        alias
        for declaration in declarations
        for alias in declaration.accepted_names()
    )
    raise ValueError(f"config_type must select a declared schema: {accepted}")


class ConfigService:
    """Reflect and create OpenHCS config objects without PyQt dependencies."""

    def __init__(self) -> None:
        self._configs: dict[str, AgentConfig] = {}
        self._counter = count(1)

    def describe_schema(
        self,
        config_type: str,
        path_prefix: str | None = None,
    ) -> ConfigSchema:
        declaration = agent_config_declaration_from_request(config_type)
        normalized_prefix = _normalized_schema_path_prefix(path_prefix)
        reflected_fields = declaration.reflected_fields()
        selected_fields = _selected_schema_fields(
            reflected_fields,
            normalized_prefix,
        )
        return ConfigSchema(
            schema_version=SCHEMA_VERSION,
            config_type=declaration.display_name(),
            fields=selected_fields,
            path_prefix=normalized_prefix,
            authoring_path=declaration.authoring_path,
            registries=_config_registry_schemas(selected_fields),
            types=_selected_config_type_schemas(
                declaration.reflected_types(),
                selected_fields,
            ),
        )

    def describe_schema_request(self, request: ConfigSchemaRequest) -> ConfigSchema:
        return self.describe_schema(request.config_type, request.path_prefix)

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
                errors=(AgentError.from_exception("config_patch_invalid", exc),),
            )
        return ConfigValidationResult(
            schema_version=SCHEMA_VERSION,
            valid=True,
            config_ref=config_ref,
        )

    def resolve_ref(self, config_ref: ConfigRef | str) -> AgentConfig:
        config_id = (
            config_ref.config_id if isinstance(config_ref, ConfigRef) else config_ref
        )
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
            source=ConfigDocumentAuthority.render(
                instance,
                expected_config_type=type(instance),
                clean_mode=clean,
            ),
        )

    def _config_class(self, config_type: str) -> type[AgentConfig]:
        return agent_config_class_from_request(config_type)


def _patch_values(
    cls: type[AgentConfig],
    patch: ConfigPatch | None,
) -> dict[str, object]:
    if patch is None:
        return {}
    if not is_dataclass(cls):
        return dict(patch.values)
    return coerce_dataclass_patch_values(cls, patch.values)


def coerce_dataclass_patch_values(
    cls: type,
    values: Mapping[str, JsonValue],
) -> dict[str, object]:
    field_by_name = {field.name: field for field in fields(cls)}
    resolved_types = get_type_hints(cls)
    return {
        name: (
            _coerce_patch_value(
                resolved_types.get(name, field_by_name[name].type),
                value,
            )
            if name in field_by_name
            else value
        )
        for name, value in values.items()
    }


def _coerce_patch_value(field_type, value: JsonValue) -> object:
    unwrapped_type = _unwrap_annotated(field_type)
    if value is None:
        return None

    collection_handled, collection_value = _coerce_collection_patch_value(
        unwrapped_type,
        value,
    )
    if collection_handled:
        return collection_value

    dataclass_type = _dataclass_type(unwrapped_type)
    if dataclass_type is not None and isinstance(value, Mapping):
        return dataclass_type(**coerce_dataclass_patch_values(dataclass_type, value))

    enum_type = _unwrap_enum(unwrapped_type)
    if enum_type is not None:
        return _coerce_enum_value(enum_type, value)

    path_type = _unwrap_path_type(unwrapped_type)
    if path_type is not None and isinstance(value, str):
        return path_type(value)

    registered_type = _unwrap_registered_nominal_type(unwrapped_type)
    if registered_type is not None:
        return registered_type.coerce(value)

    return value


def _coerce_collection_patch_value(field_type, value: JsonValue) -> tuple[bool, object]:
    field_type = _unwrap_annotated(field_type)
    origin = get_origin(field_type)
    if origin in (Union, UnionType):
        for arg in get_args(field_type):
            if arg is type(None):
                continue
            handled, coerced = _coerce_collection_patch_value(arg, value)
            if handled:
                return True, coerced
        return False, value
    if origin is list and isinstance(value, list):
        item_type = get_args(field_type)[0] if get_args(field_type) else object
        return True, [_coerce_patch_value(item_type, item) for item in value]
    if origin is tuple and isinstance(value, (list, tuple)):
        args = get_args(field_type)
        if len(args) == 2 and args[1] is Ellipsis:
            coerced = tuple(_coerce_patch_value(args[0], item) for item in value)
        elif args and len(args) == len(value):
            coerced = tuple(
                _coerce_patch_value(item_type, item)
                for item_type, item in zip(args, value, strict=True)
            )
        else:
            coerced = tuple(value)
        return True, coerced
    if origin in (set, frozenset) and isinstance(value, (list, tuple, set, frozenset)):
        item_type = get_args(field_type)[0] if get_args(field_type) else object
        coerced = (_coerce_patch_value(item_type, item) for item in value)
        return True, origin(coerced)
    if origin in (dict, Mapping) and isinstance(value, Mapping):
        args = get_args(field_type)
        key_type, item_type = args if len(args) == 2 else (object, object)
        return True, {
            _coerce_patch_value(key_type, key): _coerce_patch_value(item_type, item)
            for key, item in value.items()
        }
    return False, value


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


def _unwrap_path_type(field_type) -> type[Path] | None:
    field_type = _unwrap_annotated(field_type)
    if isinstance(field_type, type) and issubclass(field_type, Path):
        return field_type
    origin = get_origin(field_type)
    if origin not in (Union, UnionType):
        return None
    for member_type in get_args(field_type):
        path_type = _unwrap_path_type(member_type)
        if path_type is not None:
            return path_type
    return None


def _unwrap_registered_nominal_type(field_type) -> type[ArtifactType] | None:
    field_type = _unwrap_annotated(field_type)
    if get_origin(field_type) is not type:
        return None
    type_args = get_args(field_type)
    if len(type_args) != 1:
        return None
    nominal_root = type_args[0]
    if isinstance(nominal_root, type) and issubclass(nominal_root, ArtifactType):
        return nominal_root
    return None


def _field_schema(cls: type) -> tuple[ConfigFieldSchema, ...]:
    if not is_dataclass(cls):
        raise TypeError(f"{cls.__name__} is not a dataclass config type")
    return _field_schema_tree(
        cls,
        path_prefix="",
        ancestors=(),
        inherited_from_lazy_scope=False,
    )


def _function_step_config_field_schema() -> tuple[ConfigFieldSchema, ...]:
    """Reflect step override keys without copying AbstractStep's parameters."""

    signature = inspect.signature(AbstractStep.__init__)
    direct_fields: list[ConfigFieldSchema] = []
    nested_fields: list[ConfigFieldSchema] = []
    for field_name, declared_type in AbstractStep.config_classes_by_field_name().items():
        parameter = signature.parameters[field_name]
        source_type = _lazy_base_type(declared_type) or declared_type
        direct_fields.append(
            ConfigFieldSchema(
                path=field_name,
                type_repr=_type_repr(declared_type),
                value_type_repr=_type_repr(source_type),
                default_repr=_parameter_default_repr(parameter),
                required=parameter.default is inspect.Parameter.empty,
                description=parameter_description_from_target(
                    AbstractStep.__init__,
                    field_name,
                ),
                lazy=_lazy_base_type(declared_type) is not None,
                declaring_type=_type_repr(AbstractStep),
                default_origin=(
                    None
                    if parameter.default is inspect.Parameter.empty
                    else "parameter_default"
                ),
                nested_schema_path=field_name,
            )
        )
        nested_fields.extend(
            _field_schema_tree(
                source_type,
                path_prefix=f"{field_name}.",
                ancestors=(AbstractStep,),
                inherited_from_lazy_scope=True,
            )
        )
    return tuple((*direct_fields, *nested_fields))


def _normalized_schema_path_prefix(path_prefix: str | None) -> str | None:
    if path_prefix is None:
        return None
    normalized = path_prefix.strip().strip(".")
    if not normalized:
        return None
    return normalized


def _selected_schema_fields(
    reflected_fields: tuple[ConfigFieldSchema, ...],
    path_prefix: str | None,
) -> tuple[ConfigFieldSchema, ...]:
    if path_prefix is None:
        return tuple(
            field
            for field in reflected_fields
            if "." not in field.path and "[]" not in field.path
        )
    selected = tuple(
        field
        for field in reflected_fields
        if field.path == path_prefix
        or field.path.startswith(f"{path_prefix}.")
        or field.path.startswith(f"{path_prefix}[]")
    )
    if not selected:
        raise ValueError(f"Unknown config schema path_prefix: {path_prefix!r}")
    return selected


def _field_schema_tree(
    cls: type,
    *,
    path_prefix: str,
    ancestors: tuple[type, ...],
    inherited_from_lazy_scope: bool,
) -> tuple[ConfigFieldSchema, ...]:
    """Reflect one dataclass tree without maintaining a parallel config catalog."""
    field_descriptions = dataclass_parameter_descriptions(cls)
    try:
        resolved_types = get_type_hints(cls)
    except (NameError, TypeError):
        resolved_types = {}
    direct_fields: list[ConfigFieldSchema] = []
    nested_fields: list[ConfigFieldSchema] = []
    next_ancestors = (*ancestors, cls)
    for config_field in fields(cls):
        field_type = resolved_types.get(config_field.name, config_field.type)
        field_path = f"{path_prefix}{config_field.name}"
        direct_fields.append(
            _schema_for_field(
                config_field,
                field_descriptions,
                declaring_cls=cls,
                field_type=field_type,
                field_path=field_path,
                inheritable=inherited_from_lazy_scope,
            )
        )
        for path_suffix, nested_type in _nested_dataclass_types(field_type):
            if nested_type in next_ancestors:
                continue
            nested_fields.extend(
                _field_schema_tree(
                    nested_type,
                    path_prefix=f"{field_path}{path_suffix}.",
                    ancestors=next_ancestors,
                    inherited_from_lazy_scope=(
                        inherited_from_lazy_scope
                        or _lazy_base_type(field_type) is not None
                    ),
                )
            )
    return tuple((*direct_fields, *nested_fields))


def _schema_for_field(
    field,
    field_descriptions: Mapping[str, str],
    *,
    declaring_cls: type,
    field_type=None,
    field_path: str | None = None,
    inheritable: bool = False,
) -> ConfigFieldSchema:
    field_type = field.type if field_type is None else field_type
    lazy_base = _lazy_base_type(field_type)
    source_type = lazy_base or field_type
    nested_types = _nested_dataclass_types(field_type)
    ui_hidden = False
    if "ui_hidden" in field.metadata:
        ui_hidden = bool(field.metadata["ui_hidden"])
    default_repr, default_origin = _default_schema(field)
    return ConfigFieldSchema(
        path=field.name if field_path is None else field_path,
        type_repr=_type_repr(field_type),
        value_type_repr=_type_repr(source_type),
        default_repr=default_repr,
        required=_is_required(field),
        description=field_descriptions.get(field.name),
        enum_values=_enum_values(source_type),
        registry_values=_registered_type_values(source_type),
        ui_hidden=ui_hidden,
        lazy=lazy_base is not None,
        inheritable=(
            inheritable
            or "_inherited_default" in field.metadata
            or "_inherited_default_factory" in field.metadata
        ),
        declaring_type=_type_repr(_field_declaring_type(declaring_cls, field.name)),
        default_origin=default_origin,
        nested_schema_path=(
            (field.name if field_path is None else field_path)
            if nested_types
            else None
        ),
    )


def _nested_dataclass_types(field_type) -> tuple[tuple[str, type], ...]:
    """Return dataclass nodes reachable through one config field annotation."""
    field_type = _unwrap_annotated(field_type)
    lazy_base = _lazy_base_type(field_type)
    if lazy_base is not None:
        return (("", lazy_base),)
    if isinstance(field_type, type) and is_dataclass(field_type):
        return (("", field_type),)
    origin = get_origin(field_type)
    if origin in (Union, UnionType):
        return tuple(
            dict.fromkeys(
                nested
                for member_type in get_args(field_type)
                if member_type is not type(None)
                for nested in _nested_dataclass_types(member_type)
            )
        )
    if origin in (list, tuple, set, frozenset):
        item_types = get_args(field_type)
        if len(item_types) == 2 and item_types[1] is Ellipsis:
            item_types = item_types[:1]
        return tuple(
            dict.fromkeys(
                (f"[]{path_suffix}", nested_type)
                for item_type in item_types
                for path_suffix, nested_type in _nested_dataclass_types(item_type)
            )
        )
    return ()


def _dataclass_schema_types(
    cls: type,
    *,
    ancestors: tuple[type, ...] = (),
) -> tuple[type, ...]:
    source_cls = _lazy_base_type(cls) or cls
    if source_cls in ancestors or not is_dataclass(source_cls):
        return ()
    next_ancestors = (*ancestors, source_cls)
    try:
        resolved_types = get_type_hints(source_cls)
    except (NameError, TypeError):
        resolved_types = {}
    nested_types = tuple(
        nested_type
        for config_field in fields(source_cls)
        for _, nested_type in _nested_dataclass_types(
            resolved_types.get(config_field.name, config_field.type)
        )
    )
    dataclass_bases = tuple(
        candidate
        for candidate in source_cls.__mro__[1:]
        if candidate is not object and is_dataclass(candidate)
    )
    return tuple(
        dict.fromkeys(
            (
                source_cls,
                *dataclass_bases,
                *(
                    schema_type
                    for nested_type in nested_types
                    for schema_type in _dataclass_schema_types(
                        nested_type,
                        ancestors=next_ancestors,
                    )
                ),
            )
        )
    )


def _selected_config_type_schemas(
    schema_types: tuple[type, ...],
    selected_fields: tuple[ConfigFieldSchema, ...],
) -> tuple[ConfigTypeSchema, ...]:
    selected_type_names = {
        type_name
        for field in selected_fields
        for type_name in (field.declaring_type, field.value_type_repr)
        if type_name is not None
    }
    return tuple(
        ConfigTypeSchema(
            type_repr=_type_repr(schema_type),
            description=class_docstring_text(schema_type),
            base_types=tuple(
                _type_repr(base_type)
                for base_type in schema_type.__bases__
                if base_type is not object
            ),
        )
        for schema_type in schema_types
        if _type_repr(schema_type) in selected_type_names
    )


def _lazy_base_type(field_type) -> type | None:
    if not isinstance(field_type, type):
        return None
    return get_base_type_for_lazy(field_type)


def _type_repr(field_type) -> str:
    if isinstance(field_type, type):
        return f"{field_type.__module__}.{field_type.__qualname__}"
    return str(field_type)


def _default_schema(field) -> tuple[str | None, str | None]:
    inherited_default = field.metadata.get("_inherited_default", MISSING)
    inherited_factory = field.metadata.get("_inherited_default_factory", MISSING)
    if inherited_default is not MISSING:
        return repr(inherited_default), "inherited_default"
    if inherited_factory is not MISSING:
        return _factory_repr(inherited_factory), "inherited_factory"
    if field.default is not MISSING:
        return repr(field.default), "field_default"
    if field.default_factory is not MISSING:
        return _factory_repr(field.default_factory), "default_factory"
    return None, None


def _factory_repr(factory) -> str:
    return f"{factory.__module__}.{factory.__qualname__}()"


def _parameter_default_repr(parameter: inspect.Parameter) -> str | None:
    value = parameter.default
    if value is inspect.Parameter.empty:
        return None
    if is_dataclass(value):
        return f"{_type_repr(type(value))}()"
    return repr(value)


def _field_declaring_type(cls: type, field_name: str) -> type:
    source_cls = _lazy_base_type(cls) or cls
    inherited_metadata = frozenset(
        ("_inherited_default", "_inherited_default_factory")
    )
    fallback = source_cls
    for candidate in source_cls.__mro__:
        if field_name not in candidate.__dict__.get("__annotations__", {}):
            continue
        fallback = candidate
        if not is_dataclass(candidate):
            continue
        candidate_field = next(
            (item for item in fields(candidate) if item.name == field_name),
            None,
        )
        if candidate_field is None:
            continue
        if not inherited_metadata.intersection(candidate_field.metadata):
            return candidate
    return fallback


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


def _registered_type_values(field_type) -> tuple[str, ...]:
    nominal_root = _unwrap_registered_nominal_type(field_type)
    if nominal_root is None:
        return ()
    return tuple(str(key) for key in nominal_root.__registry__)


def _config_registry_schemas(
    selected_fields: tuple[ConfigFieldSchema, ...],
) -> tuple[ConfigRegistrySchema, ...]:
    registered_types = tuple(
        ConfigRegisteredType(
            key=config_key,
            type_repr=_type_repr(StreamingConfig.config_type_for_key(config_key)),
        )
        for config_key in StreamingConfig.supported_config_keys()
    )
    registered_type_names = {
        type_name
        for config_key in StreamingConfig.supported_config_keys()
        for registered_config_type in (
            StreamingConfig.config_type_for_key(config_key),
        )
        for type_name in (
            _type_repr(registered_config_type),
            _type_repr(
                _lazy_base_type(registered_config_type) or registered_config_type
            ),
        )
    }
    if not any(
        field.value_type_repr in registered_type_names for field in selected_fields
    ):
        return ()
    return (
        ConfigRegistrySchema(
            owner_type=_type_repr(StreamingConfig),
            registered_types=registered_types,
        ),
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
