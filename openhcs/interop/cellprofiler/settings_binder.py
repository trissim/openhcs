"""Convert CellProfiler .cppipe settings to absorbed-function kwargs."""

import logging
import re
from collections.abc import Callable
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TypeVar

from openhcs.core.artifacts import (
    ArtifactInputPlan,
    ArtifactOutputPlan,
    ArtifactPlan,
    ArtifactSidecarRole,
    ArtifactType,
)
from openhcs.core.source_bindings import resolve_source_file
from openhcs.interop.cellprofiler_setting_normalization import (
    normalize_cellprofiler_setting_name,
)

from .parser import ModuleBlock, ModuleSetting
from openhcs.interop.cellprofiler.cellprofiler_literals import (
    cellprofiler_setting_literal,
)
from openhcs.interop.cellprofiler.setting_names import (
    SettingNameFamily,
    normalized_symbol_name,
    optional_setting_value,
    setting_names,
    setting_values,
)

logger = logging.getLogger(__name__)

_EnumT = TypeVar("_EnumT", bound=Enum)
_NEGATED_ENUM_LITERALS = frozenset(("none", "no", "false", "disabled", "disable"))
_ENUM_DOMAIN_SUFFIXES = (
    "method",
    "choice",
    "option",
    "mode",
    "type",
    "style",
)

CellProfilerSettingValue = (
    bool | int | float | str | tuple[int | float, ...] | list[str] | Enum
)
SettingParser = Callable[[str], CellProfilerSettingValue]


def coerce_cellprofiler_enum(
    enum_type: type[_EnumT],
    value: _EnumT | str,
) -> _EnumT:
    """Coerce a CellProfiler literal into a nominal enum member."""
    if isinstance(value, enum_type):
        return value
    enum_value = value.value if isinstance(value, Enum) else value
    normalized_value = _normalized_enum_literal(str(enum_value))
    for member in enum_type:
        if normalized_value in _member_literals(enum_type, member):
            return member
    prefix_matches = [
        member
        for member in enum_type
        if any(
            normalized_value.startswith(candidate)
            or candidate.startswith(normalized_value)
            for candidate in _member_literals(enum_type, member)
        )
    ]
    if len(prefix_matches) == 1:
        return prefix_matches[0]
    raise ValueError(f"{enum_type.__name__} cannot be coerced from {value!r}.")


def cellprofiler_enum_setting_parser(
    enum_type: type[_EnumT],
) -> Callable[[str], _EnumT]:
    """Return a typed parser for a CellProfiler setting enum."""

    def parse(value: str) -> _EnumT:
        return coerce_cellprofiler_enum(enum_type, value)

    return parse


def cellprofiler_enum_value_setting_parser(
    enum_type: type[_EnumT],
) -> Callable[[str], str]:
    """Return a typed parser that emits an enum member's serialized value."""

    def parse(value: str) -> str:
        member = coerce_cellprofiler_enum(enum_type, value)
        if not isinstance(member.value, str):
            raise TypeError(
                f"{enum_type.__name__}.{member.name} must have a string value."
            )
        return member.value

    return parse


def parse_cellprofiler_bool(value: str) -> bool:
    """Parse a CellProfiler boolean literal."""
    normalized = value.strip().lower()
    if normalized in SettingsBinder.BOOL_TRUE:
        return True
    if normalized in SettingsBinder.BOOL_FALSE:
        return False
    raise ValueError(f"CellProfiler boolean setting must be Yes/No, got {value!r}.")


def parse_cellprofiler_float(value: str) -> float:
    """Parse a numeric CellProfiler setting as float."""
    return float(value)


def parse_cellprofiler_int(value: str) -> int:
    """Parse a numeric CellProfiler setting as int, accepting decimal spelling."""
    return int(float(value))


def _member_literals(enum_type: type[Enum], member: Enum) -> frozenset[str]:
    literals = [member.name]
    if isinstance(member.value, str):
        literals.append(member.value)
    elif isinstance(member.value, tuple):
        literals.extend(literal for literal in member.value if isinstance(literal, str))
    literals.extend(
        literal
        for literal in getattr(member, "cellprofiler_literals", ())
        if isinstance(literal, str)
    )
    normalized_literals = {_normalized_enum_literal(literal) for literal in literals}
    if normalized_literals & _NEGATED_ENUM_LITERALS:
        domain = _enum_domain_literal(enum_type)
        normalized_literals.add(f"no_{domain}")
    return frozenset(normalized_literals)


def _enum_domain_literal(enum_type: type[Enum]) -> str:
    literal = _normalized_enum_literal(enum_type.__name__)
    for suffix in _ENUM_DOMAIN_SUFFIXES:
        suffix_literal = f"_{suffix}"
        if literal.endswith(suffix_literal):
            return literal.removesuffix(suffix_literal)
    return literal


def _normalized_enum_literal(value: str) -> str:
    words = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", value.strip())
    return re.sub(r"[^a-z0-9]+", "_", words.lower()).strip("_")


@dataclass(frozen=True, slots=True)
class BoundParameter:
    """A parameter with its bound value."""

    name: str
    value: CellProfilerSettingValue
    original_key: str
    original_value: str


@dataclass(frozen=True, slots=True)
class SettingToKeywordBinding:
    """Declarative mapping from one parsed setting to one function kwarg."""

    setting_name: str | SettingNameFamily
    parameter_name: str | None = None
    parse: SettingParser | None = None
    repeated: bool = False
    artifact_plan_type: type[ArtifactPlan] | None = None
    artifact_type: type[ArtifactType] | None = None
    runtime_parameter_name: str | None = None
    sidecar_role: ArtifactSidecarRole | None = None

    def __post_init__(self) -> None:
        artifact_fields = (self.artifact_plan_type, self.artifact_type)
        if (artifact_fields[0] is None) != (artifact_fields[1] is None):
            raise ValueError(
                "SettingToKeywordBinding artifact plan and payload types must be "
                "declared together."
            )
        if self.artifact_plan_type is not None and self.artifact_plan_type not in (
            ArtifactInputPlan,
            ArtifactOutputPlan,
        ):
            raise TypeError(
                "SettingToKeywordBinding.artifact_plan_type must be "
                "ArtifactInputPlan or ArtifactOutputPlan."
            )
        if self.artifact_type is not None:
            object.__setattr__(
                self,
                "artifact_type",
                ArtifactType.coerce(self.artifact_type),
            )
        if self.runtime_parameter_name is not None:
            if self.artifact_plan_type is not ArtifactInputPlan:
                raise ValueError(
                    "SettingToKeywordBinding.runtime_parameter_name is only valid "
                    "for artifact inputs."
                )
            if not self.runtime_parameter_name:
                raise ValueError(
                    "SettingToKeywordBinding.runtime_parameter_name cannot be empty."
                )
        if self.sidecar_role is not None:
            if not isinstance(self.sidecar_role, ArtifactSidecarRole):
                raise TypeError(
                    "SettingToKeywordBinding.sidecar_role must be "
                    f"ArtifactSidecarRole, got {type(self.sidecar_role).__name__}."
                )
            if self.artifact_plan_type is not ArtifactInputPlan:
                raise ValueError(
                    "SettingToKeywordBinding.sidecar_role is only valid for "
                    "artifact inputs."
                )

    @classmethod
    def input(
        cls,
        setting_name: str | SettingNameFamily,
        artifact_type: type[ArtifactType],
        *,
        runtime_parameter_name: str | None = None,
        parse: SettingParser | None = None,
        repeated: bool = False,
        sidecar_role: ArtifactSidecarRole | None = None,
    ) -> "SettingToKeywordBinding":
        """Declare one setting-backed artifact input."""

        return cls(
            setting_name=setting_name,
            parse=parse,
            repeated=repeated,
            artifact_plan_type=ArtifactInputPlan,
            artifact_type=artifact_type,
            runtime_parameter_name=runtime_parameter_name,
            sidecar_role=sidecar_role,
        )

    @classmethod
    def output(
        cls,
        setting_name: str | SettingNameFamily,
        artifact_type: type[ArtifactType],
        parameter_name: str | None = None,
        parse: SettingParser | None = None,
        repeated: bool = False,
    ) -> "SettingToKeywordBinding":
        """Declare one setting-backed artifact output."""

        return cls(
            setting_name=setting_name,
            parameter_name=parameter_name,
            parse=parse,
            repeated=repeated,
            artifact_plan_type=ArtifactOutputPlan,
            artifact_type=artifact_type,
        )

    def artifact_input_domain_key(
        self,
    ) -> tuple[type[ArtifactType], ArtifactSidecarRole | None]:
        """Return the exact artifact domain used to reconstruct this input."""

        if self.require_artifact_plan_type() is not ArtifactInputPlan:
            raise TypeError(
                f"Setting binding {self.setting_name!r} is not an artifact input."
            )
        return (self.require_artifact_type(), self.sidecar_role)

    def preserves_artifact_input_occurrence_partitions(self) -> bool:
        """Return whether each reconstructed input occurrence stays scalar."""

        if self.require_artifact_plan_type() is not ArtifactInputPlan:
            raise TypeError(
                f"Setting binding {self.setting_name!r} is not an artifact input."
            )
        return self.runtime_parameter_name is not None

    @property
    def declares_artifact(self) -> bool:
        """Return whether this binding owns an artifact contract term."""

        return self.artifact_type is not None

    def require_artifact_plan_type(self) -> type[ArtifactPlan]:
        """Return this binding's exact artifact plan role."""

        if self.artifact_plan_type is None:
            raise TypeError(
                f"Setting binding {self.setting_name!r} does not declare an artifact."
            )
        return self.artifact_plan_type

    def require_artifact_type(self) -> type[ArtifactType]:
        """Return this binding's exact artifact payload type."""

        if self.artifact_type is None:
            raise TypeError(
                f"Setting binding {self.setting_name!r} does not declare an artifact."
            )
        return self.artifact_type

    def require_parameter_name(self) -> str:
        """Return the explicit or setting-derived callable keyword name."""

        if self.parameter_name is not None:
            if not self.parameter_name:
                raise ValueError(
                    "SettingToKeywordBinding.parameter_name cannot be empty."
                )
            return self.parameter_name
        return normalize_cellprofiler_setting_name(setting_names(self.setting_name)[0])

    def require_runtime_parameter_name(self) -> str:
        """Return the exact runtime special-input parameter or fail."""

        if self.runtime_parameter_name is None:
            raise TypeError(
                f"Setting binding {self.setting_name!r} does not inject a runtime "
                "parameter."
            )
        return self.runtime_parameter_name

    def parameter_help_description(self) -> str:
        """Return user help owned by this setting-to-parameter declaration."""

        setting_label = setting_names(self.setting_name)[0]
        repeated_prefix = "Repeated " if self.repeated else ""
        if self.artifact_plan_type is ArtifactInputPlan:
            return (
                f"{repeated_prefix}input {self.require_artifact_type().description()} "
                f"selected by the CellProfiler setting {setting_label!r}."
            )
        if self.artifact_plan_type is ArtifactOutputPlan:
            return (
                f"{repeated_prefix}output {self.require_artifact_type().description()} "
                f"named by the CellProfiler setting {setting_label!r}."
            )
        return (
            f"{repeated_prefix}value configured by the CellProfiler setting "
            f"{setting_label!r}."
        )

    def records_from_kwargs(
        self,
        kwargs: Mapping[str, object],
    ) -> tuple[ModuleSetting, ...]:
        """Reconstruct this binding's setting row from one present kwarg."""

        parameter_name = self.require_parameter_name()
        if parameter_name not in kwargs:
            return ()
        value = kwargs[parameter_name]
        values = value if isinstance(value, (tuple, list)) else (value,)
        return tuple(
            ModuleSetting(
                setting_names(self.setting_name)[0],
                cellprofiler_setting_literal(item),
            )
            for item in values
        )

    def bind(
        self,
        module: ModuleBlock,
        kwargs: dict[str, CellProfilerSettingValue],
        binder: "SettingsBinder",
    ) -> None:
        value = optional_setting_value(module, self.setting_name)
        if value is None:
            return
        setting_name = setting_names(self.setting_name)[0]
        kwargs[self.require_parameter_name()] = (
            binder.parse_value(setting_name, value)
            if self.parse is None
            else self.parse(value)
        )


@dataclass(frozen=True, slots=True)
class MeasurementFeatureSettingBinding(SettingToKeywordBinding):
    """Setting binding that owns a prior-measurement feature reference."""

    def feature_names(self, module: ModuleBlock) -> tuple[str, ...]:
        """Return the ordered measurement features selected by this binding."""

        return tuple(
            value
            for value in setting_values(module, self.setting_name)
            if normalized_symbol_name(value) is not None
        )


class SettingsBinder:
    """Bind parsed .cppipe setting strings to typed Python kwargs."""

    BOOL_TRUE = {"yes", "true", "1", "on"}
    BOOL_FALSE = {"no", "false", "0", "off"}
    GENERIC_BOOL_TRUE = {"yes", "true", "on"}
    GENERIC_BOOL_FALSE = {"no", "false", "off"}

    def __init__(
        self,
        enum_mappings: Mapping[str, type[Enum]] | None = None,
        *,
        source_root: str | Path | None = None,
    ) -> None:
        self.enum_mappings = dict(enum_mappings or {})
        self.source_root = None if source_root is None else Path(source_root)

    def resolve_source_file(self, location: str) -> Path:
        """Resolve one declared file through the generic source-path authority."""

        if self.source_root is None:
            raise ValueError(
                "CellProfiler external resource resolution requires a source root."
            )
        return resolve_source_file(location, self.source_root)

    def bind(self, module: ModuleBlock) -> dict[str, CellProfilerSettingValue]:
        """Bind one parser-owned module's declared settings."""
        return {
            parameter.name: parameter.value
            for parameter in self.bind_with_details(module)
        }

    def bind_declared(
        self,
        module: ModuleBlock,
        bindings: tuple[SettingToKeywordBinding, ...],
    ) -> dict[str, CellProfilerSettingValue]:
        """Bind an explicit setting-to-kwarg declaration for one module."""
        kwargs: dict[str, CellProfilerSettingValue] = {}
        for binding in bindings:
            binding.bind(module, kwargs, self)
        return kwargs

    def bind_with_details(self, module: ModuleBlock) -> list[BoundParameter]:
        """Bind parser-owned settings and preserve source-row provenance."""
        if not isinstance(module, ModuleBlock):
            raise TypeError(
                "SettingsBinder.bind_with_details requires a ModuleBlock, "
                f"got {type(module).__name__}."
            )
        result: list[BoundParameter] = []
        for key, value in module.settings.items():
            normalized_key = normalize_cellprofiler_setting_name(key)
            result.append(
                BoundParameter(
                    name=normalized_key,
                    value=self.parse_value(key, value),
                    original_key=key,
                    original_value=value,
                )
            )
        return result

    def parse_value(self, key: str, value: str) -> CellProfilerSettingValue:
        """Parse one CellProfiler setting value into a Python value."""
        value = value.strip()

        if value.lower() in self.GENERIC_BOOL_TRUE:
            return True
        if value.lower() in self.GENERIC_BOOL_FALSE:
            return False

        normalized_key = normalize_cellprofiler_setting_name(key)
        if normalized_key in self.enum_mappings:
            enum_type = self.enum_mappings[normalized_key]
            try:
                return enum_type[value.upper().replace(" ", "_")]
            except KeyError:
                logger.warning(f"Unknown enum value '{value}' for {normalized_key}")
                return value

        if "," in value:
            return _parse_cellprofiler_csv_value(value)

        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            return value


def _parse_cellprofiler_csv_value(value: str) -> tuple[int | float, ...] | list[str]:
    parts = [part.strip() for part in value.split(",")]
    try:
        return tuple(float(part) if "." in part else int(part) for part in parts)
    except ValueError:
        return parts
