"""CellProfiler module class declarations.

This file is the source-of-truth catalog for absorbed CellProfiler modules.
Compatibility registry payloads are derived from these classes.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from functools import lru_cache
from types import UnionType
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)
from metaclass_registry import AutoRegisterMeta, LazyDiscoveryDict, RegistryConfig
from openhcs.constants.constants import GroupBy, VariableComponents
from openhcs.core.artifacts import (
    ArtifactInputPlan,
    ArtifactOutputPlan,
    ArtifactPlan,
    ArtifactSpec,
    ArtifactSpecRef,
    ArtifactSpecRelation,
    ArtifactType,
    GroupLineageSourceRelation,
    ImageArtifactType,
    ObjectLabelsArtifactType,
    MeasurementsArtifactType,
    RelationshipsArtifactType,
    SpatialGridArtifactType,
)
from openhcs.core.equivalence.policy import normalize_runtime_identifier
from openhcs.core.runtime_invocation import RuntimeInvocationOptions
from openhcs.core.runtime_semantics import (
    MeasuredObjectAnchorFeatureMarker,
    MeasurementScope,
    MeasurementStatistic,
    ObjectCalculatedFeatureMarker,
    ObjectCountFeatureMarker,
    ObjectIdentifierFeatureMarker,
    ObjectIntensityFeatureMarker,
    ObjectLocationFeatureMarker,
    ObjectShapeDescriptorFeatureMarker,
    RuntimeMeasurementFeature,
    RuntimeMeasurementFeatureRelationDeclaration,
    RuntimeMeasurementFeatureSemanticMarker,
)
from openhcs.interop.cellprofiler.settings_binder import coerce_cellprofiler_enum
from openhcs.interop.cellprofiler_setting_normalization import (
    normalize_cellprofiler_setting_name,
)
from openhcs.interop.cellprofiler.setting_names import SettingNameFamily, setting_names
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract

_CELLPROFILER_BACKEND_PACKAGE = "openhcs.processing.backends.cellprofiler"
_CELLPROFILER_MODULE_REGISTRY = LazyDiscoveryDict(enable_cache=False)
AuthorityT = TypeVar("AuthorityT", bound="CellProfilerModuleAuthority")
if TYPE_CHECKING:
    from openhcs.core.module_artifact_contract import ModuleArtifactContract
    from openhcs.core.pipeline_image_schema import PipelineImageSchema
    from openhcs.interop.cellprofiler.module_artifact_inputs import ModuleArtifactInput
    from openhcs.interop.cellprofiler.measurement_scope import (
        CellProfilerMeasurementTargetScope,
    )
    from openhcs.interop.cellprofiler.module_function_resolution import (
        ResolvedModuleFunction,
    )
    from openhcs.interop.cellprofiler.module_processing_components import (
        ModuleProcessingComponentRequest,
        ModuleProcessingComponents,
    )
    from openhcs.interop.cellprofiler.parser import ModuleBlock, ModuleSetting
    from openhcs.interop.cellprofiler.runtime.payload_types import CellProfilerKwargs
    from openhcs.interop.cellprofiler.module_roles import ArtifactSpecKey
    from openhcs.core.runtime_exports import RuntimeImageExportSpec
    from openhcs.interop.cellprofiler.semantic_defaults import (
        CellProfilerSemanticDefaultContract,
    )
    from openhcs.interop.cellprofiler.settings_binder import SettingToKeywordBinding
    from openhcs.interop.cellprofiler.settings_binder import SettingsBinder
    from openhcs.interop.cellprofiler.symbol_table import ModuleArtifactContracts
    from openhcs.interop.cellprofiler.symbol_table import (
        CellProfilerContractAssemblyMixin,
        _SymbolTableBuilder,
    )


@dataclass(frozen=True, slots=True)
class ModuleSettingRowRecord:
    """Concrete CellProfiler setting row identity and value."""

    module_name: str
    module_num: int
    setting_name: str
    normalized_setting_name: str
    value: Any


@dataclass(frozen=True, slots=True)
class ModuleSettingCoverageRecord(ModuleSettingRowRecord):
    """Coverage status for one concrete CellProfiler setting row."""

    status: "ModuleSettingCoverageStatus"


class ModuleSettingCoverageStatus(str, Enum):
    """How one CellProfiler setting row was accounted for by import binding."""

    BOUND = "bound"
    ARTIFACT_CONTRACT = "artifact_contract"
    TYPED_IGNORE = "typed_ignore"
    CALLER_IGNORE = "caller_ignore"
    INFRASTRUCTURE = "infrastructure"
    UNMAPPED = "unmapped"

    @classmethod
    def for_setting(
        cls,
        normalized_name: str,
        *,
        binder: "SettingsBinder",
        unmapped_kwargs: Mapping[str, Any],
        ignored_unmapped_settings: frozenset[str],
        artifact_setting_names: frozenset[str],
        typed_ignore_setting_names: frozenset[str],
    ) -> "ModuleSettingCoverageStatus":
        """Return the coverage status owned by this status enum."""
        if normalized_name in binder.SKIP_SETTINGS:
            return cls.INFRASTRUCTURE
        if normalized_name not in unmapped_kwargs:
            return cls.BOUND
        if normalized_name in ignored_unmapped_settings:
            return cls.CALLER_IGNORE
        if normalized_name in artifact_setting_names:
            return cls.ARTIFACT_CONTRACT
        if normalized_name in typed_ignore_setting_names:
            return cls.TYPED_IGNORE
        return cls.UNMAPPED


@dataclass(frozen=True, slots=True)
class BoundModuleSettings:
    """Typed module-setting translation result."""

    kwargs: Mapping[str, Any]
    unmapped_kwargs: Mapping[str, Any] = field(default_factory=dict)
    invocation_options: RuntimeInvocationOptions | None = None
    setting_coverage: tuple[ModuleSettingCoverageRecord, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "kwargs", dict(self.kwargs))
        object.__setattr__(self, "unmapped_kwargs", dict(self.unmapped_kwargs))
        object.__setattr__(self, "setting_coverage", tuple(self.setting_coverage))
        if self.invocation_options is not None and (
            not isinstance(self.invocation_options, RuntimeInvocationOptions)
        ):
            raise TypeError(
                "BoundModuleSettings.invocation_options must inherit RuntimeInvocationOptions."
            )

    def with_kwargs(self, kwargs: Mapping[str, Any]) -> "BoundModuleSettings":
        """Return this binding with additional generated function kwargs."""
        return BoundModuleSettings(
            {**self.kwargs, **kwargs},
            self.unmapped_kwargs,
            self.invocation_options,
            self.setting_coverage,
        )

    def with_replaced_kwargs(self, kwargs: Mapping[str, Any]) -> "BoundModuleSettings":
        """Return this binding with the function kwargs replaced."""
        return BoundModuleSettings(
            kwargs,
            self.unmapped_kwargs,
            self.invocation_options,
            self.setting_coverage,
        )


@dataclass(frozen=True, slots=True)
class CellProfilerCompileTimeSettingsRequest:
    """Compiler request for module-owned CellProfiler setting reconstruction."""

    module_name: str
    module_num: int
    kwargs: Mapping[str, Any]
    invocation_options: RuntimeInvocationOptions | None = None
    source_bindings: Any = None
    group_key: str = "default"

    def __post_init__(self) -> None:
        object.__setattr__(self, "kwargs", dict(self.kwargs))
        if self.invocation_options is not None and not isinstance(
            self.invocation_options, RuntimeInvocationOptions
        ):
            raise TypeError(
                "CellProfilerCompileTimeSettingsRequest.invocation_options must "
                "inherit RuntimeInvocationOptions."
            )


def _enum_type_from_annotation(annotation: Any) -> type[Enum] | None:
    """Return the callable-owned Enum type declared by an annotation."""
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return annotation
    origin = get_origin(annotation)
    if origin in (Union, UnionType, tuple):
        for arg in get_args(annotation):
            enum_type = _enum_type_from_annotation(arg)
            if enum_type is not None:
                return enum_type
    return None


def _coerce_callable_enum_kwarg(value: Any, enum_type: type[Enum]) -> Any:
    """Coerce one bound kwarg value to the callable-owned Enum type."""
    if value is None:
        return None
    if isinstance(value, tuple):
        return tuple(_coerce_callable_enum_kwarg(item, enum_type) for item in value)
    if isinstance(value, enum_type):
        return value
    return coerce_cellprofiler_enum(enum_type, value)


GeneratedImportCollector = set[tuple[str, str]]


def runtime_invocation_options_source_literal(
    options: RuntimeInvocationOptions,
    *,
    import_collector: GeneratedImportCollector,
) -> str:
    """Return a Python literal for a typed runtime invocation-options dataclass."""
    if not is_dataclass(options):
        raise TypeError(
            "Runtime invocation options emitted into generated pipelines must be "
            f"dataclass instances, got {type(options).__name__}."
        )
    options_type = type(options)
    import_collector.add((options_type.__module__, options_type.__name__))
    assignments = tuple(
        f"{field.name}={runtime_invocation_options_value_literal(getattr(options, field.name), import_collector=import_collector)}"
        for field in fields(options)
    )
    return f"{options_type.__name__}({', '.join(assignments)})"


def runtime_invocation_options_value_literal(
    value: Any,
    *,
    import_collector: GeneratedImportCollector,
) -> str:
    """Return a Python literal for one invocation-options field value."""
    if isinstance(value, Enum):
        enum_type = type(value)
        import_collector.add((enum_type.__module__, enum_type.__name__))
        return f"{enum_type.__name__}.{value.name}"
    if is_dataclass(value):
        return runtime_invocation_options_source_literal(
            value,
            import_collector=import_collector,
        )
    if isinstance(value, tuple):
        inner = ", ".join(
            runtime_invocation_options_value_literal(
                item,
                import_collector=import_collector,
            )
            for item in value
        )
        return f"({inner}{',' if len(value) == 1 else ''})"
    if isinstance(value, list):
        inner = ", ".join(
            runtime_invocation_options_value_literal(
                item,
                import_collector=import_collector,
            )
            for item in value
        )
        return f"[{inner}]"
    if isinstance(value, dict):
        inner = ", ".join(
            (
                f"{runtime_invocation_options_value_literal(key, import_collector=import_collector)}: "
                f"{runtime_invocation_options_value_literal(item, import_collector=import_collector)}"
            )
            for key, item in value.items()
        )
        return f"{{{inner}}}"
    return repr(value)


@dataclass(frozen=True, slots=True)
class UnmappedModuleSetting:
    """A CellProfiler setting that no registered binding strategy consumed."""

    module_name: str
    module_num: int
    setting_name: str
    value: Any


class UnmappedModuleSettingsError(ValueError):
    """Raised when enabled module settings are not mapped or explicitly ignored."""

    def __init__(self, settings: tuple[UnmappedModuleSetting, ...]) -> None:
        self.settings = settings
        rendered = "; ".join(
            (
                f"{setting.module_name}({setting.module_num}).{setting.setting_name}={setting.value!r}"
                for setting in settings
            )
        )
        super().__init__(
            f"Enabled CellProfiler modules have unmapped settings. Add a module settings binding hook or an explicit typed ignore: {rendered}"
        )


def _required_string(value: object, name: str, owner: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{owner}.{name} must be a non-empty string.")
    return value


def _string_tuple(value: object, name: str, owner: str) -> tuple[str, ...]:
    if isinstance(value, str):
        raise TypeError(f"{owner}.{name} must be a tuple of strings, not str.")
    try:
        values = tuple(value)
    except TypeError as exc:
        raise TypeError(f"{owner}.{name} must be an iterable of strings.") from exc
    if not all((isinstance(item, str) and item.strip() for item in values)):
        raise ValueError(f"{owner}.{name} must contain only non-empty strings.")
    return values


def _module_lookup_key(name: str) -> str:
    return _normalize_setting_name(name)


def _declared_lookup_keys(module_type: type["CellProfilerModule"]) -> frozenset[str]:
    return frozenset(
        (
            _module_lookup_key(name)
            for name in (str(module_type.module_name), *module_type.aliases)
        )
    )


def _validate_unique_module_names(module_type: type["CellProfilerModule"]) -> None:
    declared_keys = _declared_lookup_keys(module_type)
    for existing_type in dict.values(CellProfilerModule.__registry__):
        if existing_type is module_type or (
            existing_type.__module__ == module_type.__module__
            and existing_type.__qualname__ == module_type.__qualname__
        ):
            continue
        overlap = declared_keys & _declared_lookup_keys(existing_type)
        if not overlap:
            continue
        names = tuple(sorted(overlap))
        raise ValueError(
            f"{module_type.__name__} duplicates CellProfiler module names or aliases declared by {existing_type.__name__}: {names!r}."
        )


class ArtifactContractModule(ABC):
    """Nominal marker for module declarations that own artifact flow."""


class CellProfilerArtifactCapability(ABC, metaclass=AutoRegisterMeta):
    """Registered CellProfiler artifact capability product term."""

    __registry_key__ = "capability_key"
    __skip_if_no_key__ = True

    capability_key: ClassVar[str | None] = None
    artifact_plan_type: ClassVar[type[ArtifactPlan] | None] = None
    artifact_type: ClassVar[type[ArtifactType] | None] = None

    @classmethod
    def require_plan_type(cls) -> type[ArtifactPlan]:
        plan_type = cls.artifact_plan_type
        if (
            not isinstance(plan_type, type)
            or not issubclass(plan_type, ArtifactPlan)
            or plan_type not in ArtifactPlan.__registry__.values()
        ):
            raise TypeError(f"{cls.__name__} must declare a registered ArtifactPlan.")
        return plan_type

    @classmethod
    def require_artifact_type(cls) -> type[ArtifactType]:
        artifact_type = cls.artifact_type
        if (
            not isinstance(artifact_type, type)
            or not issubclass(artifact_type, ArtifactType)
            or artifact_type not in ArtifactType.__registry__.values()
        ):
            raise TypeError(f"{cls.__name__} must declare a registered ArtifactType.")
        return artifact_type

    @classmethod
    def spec(cls, name: str, **kwargs: Any) -> ArtifactSpec:
        return ArtifactSpec(
            name=name,
            plan_type=cls.require_plan_type(),
            artifact_type=cls.require_artifact_type(),
            **kwargs,
        )

    @classmethod
    def bind_artifact(
        cls,
        owner_type: type["CellProfilerModule"],
        builder: "_SymbolTableBuilder",
        module: "ModuleBlock",
        spec: ArtifactSpec,
    ) -> object:
        """Bind this artifact spec through the plan role axis."""
        if not issubclass(owner_type, cls):
            raise TypeError(
                f"{owner_type.__name__} uses {cls.__name__} without inheriting it."
            )
        if (
            spec.plan_type is not cls.require_plan_type()
            or spec.artifact_type is not cls.require_artifact_type()
        ):
            raise TypeError(
                f"{cls.__name__} cannot bind {spec.plan_type.plan_role}:"
                f"{spec.artifact_type.require_value()} artifact {spec.name!r}."
            )
        return cls._bind_artifact(builder, module, spec)

    @classmethod
    @abstractmethod
    def _bind_artifact(
        cls, builder: "_SymbolTableBuilder", module: "ModuleBlock", spec: ArtifactSpec
    ) -> object:
        """Bind this artifact spec through the concrete plan role axis."""
        raise NotImplementedError


class ArtifactInputCapability(CellProfilerArtifactCapability):
    """Capability axis for CellProfiler artifacts consumed by a module."""

    artifact_plan_type = ArtifactInputPlan

    @classmethod
    def _bind_artifact(
        cls, builder: "_SymbolTableBuilder", module: "ModuleBlock", spec: ArtifactSpec
    ) -> object:
        return builder.require_artifact(spec, module)


class ArtifactOutputCapability(CellProfilerArtifactCapability):
    """Capability axis for CellProfiler artifacts produced by a module."""

    artifact_plan_type = ArtifactOutputPlan

    @classmethod
    def _bind_artifact(
        cls, builder: "_SymbolTableBuilder", module: "ModuleBlock", spec: ArtifactSpec
    ) -> object:
        return builder.declare_artifact(spec, module)


class ImageArtifactCapability(CellProfilerArtifactCapability):
    """Capability axis for image artifacts."""

    artifact_type = ImageArtifactType


class ObjectLabelArtifactCapability(CellProfilerArtifactCapability):
    """Capability axis for object-label artifacts."""

    artifact_type = ObjectLabelsArtifactType


class MeasurementArtifactCapability(CellProfilerArtifactCapability):
    """Capability axis for measurement artifacts."""

    artifact_type = MeasurementsArtifactType


class RelationshipArtifactCapability(CellProfilerArtifactCapability):
    """Capability axis for relationship artifacts."""

    artifact_type = RelationshipsArtifactType


class SpatialGridArtifactCapability(CellProfilerArtifactCapability):
    """Capability axis for spatial-grid artifacts."""

    artifact_type = SpatialGridArtifactType


class ImageArtifactInputCapability(ArtifactInputCapability, ImageArtifactCapability):
    capability_key = "image_input"


class ObjectLabelArtifactInputCapability(
    ArtifactInputCapability, ObjectLabelArtifactCapability
):
    capability_key = "object_label_input"


class MeasurementArtifactInputCapability(
    ArtifactInputCapability, MeasurementArtifactCapability
):
    capability_key = "measurement_input"


class RelationshipArtifactInputCapability(
    ArtifactInputCapability, RelationshipArtifactCapability
):
    capability_key = "relationship_input"


class SpatialGridArtifactInputCapability(
    ArtifactInputCapability, SpatialGridArtifactCapability
):
    capability_key = "spatial_grid_input"


class ImageArtifactOutputCapability(ArtifactOutputCapability, ImageArtifactCapability):
    capability_key = "image_output"


class ObjectLabelArtifactOutputCapability(
    ArtifactOutputCapability, ObjectLabelArtifactCapability
):
    capability_key = "object_label_output"


class MeasurementArtifactOutputCapability(
    ArtifactOutputCapability, MeasurementArtifactCapability
):
    capability_key = "measurement_output"


class RelationshipArtifactOutputCapability(
    ArtifactOutputCapability, RelationshipArtifactCapability
):
    capability_key = "relationship_output"


class SpatialGridArtifactOutputCapability(
    ArtifactOutputCapability, SpatialGridArtifactCapability
):
    capability_key = "spatial_grid_output"


ArtifactSettingCapability = tuple[
    str | SettingNameFamily, type[CellProfilerArtifactCapability]
]


class CellProfilerModuleAuthority:
    """Nominal semantic capability inherited by CellProfiler module declarations."""


class CellProfilerMeasurementFeatureMarker(
    RuntimeMeasurementFeatureSemanticMarker,
    CellProfilerModuleAuthority,
    ABC,
):
    """Semantic marker authority for module-owned measurement features."""

    @classmethod
    def feature_members(
        cls,
        module_type: type["CellProfilerModule"],
    ) -> tuple[RuntimeMeasurementFeature, ...]:
        """Return module feature members carrying this marker."""
        return tuple(
            feature
            for feature_type in module_type.measurement_feature_types()
            for feature in feature_type
            if cls.matches_feature(feature)
        )

    @classmethod
    def matches_feature_key(
        cls,
        module_type: type["CellProfilerModule"],
        key: Any,
        dialect: Any,
    ) -> bool:
        """Return whether ``key`` is an object value feature owned by this marker."""
        del dialect
        if key.subject.scope is not MeasurementScope.OBJECT:
            return False
        if key.statistic != MeasurementStatistic.VALUE.value:
            return False
        feature_name = normalize_runtime_identifier(key.feature_name)
        for feature in cls.feature_members(module_type):
            descriptor_declarations = feature.indexed_descriptor_declarations()
            if descriptor_declarations:
                if any(
                    declaration_type.from_feature_name(key.feature_name) is not None
                    for declaration_type in descriptor_declarations
                ):
                    return True
                continue
            family = feature.feature_family()
            if feature_name == family or feature_name.startswith(f"{family}_"):
                return True
        return False


class CurrentObjectFeatureVectorAuthority(CellProfilerModuleAuthority, ABC):
    """Authority for deriving current-object vectors from module-owned features."""

    @classmethod
    @abstractmethod
    def current_object_feature_vector(
        cls,
        feature_name: str,
        label_array: Any,
    ) -> Any | None:
        """Return a dense current-object vector for ``feature_name`` if owned."""


class ObjectCountFeature(CellProfilerMeasurementFeatureMarker, ObjectCountFeatureMarker):
    """Object-count feature marker."""

    family_qualifier = "count"


class ObjectIdentifierFeature(
    CellProfilerMeasurementFeatureMarker,
    ObjectIdentifierFeatureMarker,
):
    """Object-identifier feature marker."""

    family_qualifier = "identifier"


class MeasuredObjectAnchorFeature(
    CellProfilerMeasurementFeatureMarker,
    MeasuredObjectAnchorFeatureMarker,
):
    """Feature marker proving that an object row is measured."""

    family_qualifier = "object"


class ObjectLocationFeature(
    CellProfilerMeasurementFeatureMarker,
    ObjectLocationFeatureMarker,
):
    """Object-location feature marker."""

    family_qualifier = "location"


class IntensityFeature(CellProfilerMeasurementFeatureMarker, ObjectIntensityFeatureMarker):
    """Object-intensity feature marker."""

    family_qualifier = "intensity"


class ObjectCalculatedFeature(
    CellProfilerMeasurementFeatureMarker,
    ObjectCalculatedFeatureMarker,
):
    """Calculated object-feature marker."""

    family_qualifier = "calculated"


class ShapeDescriptorFeature(
    CellProfilerMeasurementFeatureMarker,
    ObjectShapeDescriptorFeatureMarker,
):
    """Object shape-descriptor feature marker."""

    family_qualifier = "shape"


class CellProfilerModule(
    ABC,
    metaclass=AutoRegisterMeta,
    registry_config=RegistryConfig(
        registry_dict=_CELLPROFILER_MODULE_REGISTRY,
        key_attribute="module_name",
        skip_if_no_key=True,
        registry_name="CellProfiler module",
        discovery_package=_CELLPROFILER_BACKEND_PACKAGE,
    ),
):
    """Auto-registered base class for absorbed CellProfiler modules."""

    __registry__ = _CELLPROFILER_MODULE_REGISTRY
    __registry_key__ = "module_name"
    __skip_if_no_key__ = True
    module_name: ClassVar[str | None] = None
    function_name: ClassVar[str | None] = None
    aliases: ClassVar[tuple[str, ...]] = ()
    function_variants: ClassVar[tuple[str, ...]] = ()
    setting_bindings: ClassVar[tuple["SettingToKeywordBinding", ...]] = ()
    setting_parameter_aliases: ClassVar[
        Mapping[str | "SettingNameFamily", str | list[str] | None]
    ] = {}
    ignored_settings: ClassVar[tuple[str | "SettingNameFamily", ...]] = ()
    contract: ClassVar[ProcessingContract | None] = ProcessingContract.PURE_2D
    default_variable_components: ClassVar[tuple[VariableComponents, ...]] = (
        VariableComponents.SITE,
    )
    confidence: ClassVar[float] = 0.5
    validated: ClassVar[bool] = False
    respects_masks: ClassVar[bool] = False
    required_variable_components: ClassVar[tuple[VariableComponents, ...]] = ()
    group_by: ClassVar[GroupBy] = GroupBy.CHANNEL
    allowed_group_by: ClassVar[tuple[GroupBy, ...]] = ()
    semantic_default_contract_types: ClassVar[
        tuple[type["CellProfilerSemanticDefaultContract"], ...]
    ] = ()
    semantic_default_contract_module_name: ClassVar[str | None] = None
    infrastructure_import_note: ClassVar[str | None] = None
    infrastructure_exports_tables: ClassVar[bool] = False
    infrastructure_exports_images: ClassVar[bool] = False
    measurement_feature_part_aliases: ClassVar[
        Mapping[tuple[str, ...], tuple[tuple[str, ...], ...]]
    ] = {}
    force_grouped_public_function_spec: ClassVar[bool] = False
    """Whether coalesced generated emissions must stay explicit per group.

    Most grouped emissions with identical public callable settings can be exposed
    as one normal OpenHCS callable. Modules that need per-group invocation
    contracts can opt into explicit dict-pattern emission at the declaration
    boundary.
    """
    measurement_feature_part_rewrites: ClassVar[
        Mapping[tuple[str, ...], tuple[str, ...]]
    ] = {}
    measurement_category_prefixes: ClassVar[tuple[tuple[str, ...], ...]] = ()
    measurement_source_feature_prefixes: ClassVar[tuple[tuple[str, ...], ...]] = ()
    calculated_measurement_feature_prefixes: ClassVar[tuple[tuple[str, ...], ...]] = ()
    numbered_measurement_feature_prefix_aliases: ClassVar[
        Mapping[str, tuple[str, ...]]
    ] = {}
    directional_pair_feature_aliases: ClassVar[Mapping[str, tuple[str, int]]] = {}
    scale_qualified_measurement_feature_prefixes: ClassVar[
        tuple[tuple[str, ...], ...]
    ] = ()
    pair_correlation_feature_name: ClassVar[str | None] = None
    pair_regression_slope_feature_name: ClassVar[str | None] = None
    undirected_pair_feature_names: ClassVar[frozenset[str]] = frozenset()
    threshold_sensitive_pair_feature_names: ClassVar[frozenset[str]] = frozenset()

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if "source_qualified_measurement_feature_enum_types" in cls.__dict__:
            raise TypeError(
                f"{cls.__name__} must inherit SourceQualifiedMeasurementFeatureModule "
                "instead of declaring source_qualified_measurement_feature_enum_types."
            )
        if "measurement_feature_enum_types" in cls.__dict__:
            raise TypeError(
                f"{cls.__name__} must declare RuntimeMeasurementFeature enum classes "
                "on the module class instead of maintaining measurement_feature_enum_types."
            )
        module_name = cls.__dict__.get("module_name")
        if module_name is None:
            return
        cls.module_name = _required_string(module_name, "module_name", cls.__name__)
        cls.function_name = _required_string(
            cls.function_name, "function_name", cls.__name__
        )
        cls.aliases = _string_tuple(cls.aliases, "aliases", cls.__name__)
        cls.function_variants = _string_tuple(
            cls.function_variants, "function_variants", cls.__name__
        )
        if cls.function_name in cls.function_variants:
            raise ValueError(
                f"{cls.__name__} declares primary function {cls.function_name!r} as a variant."
            )
        if cls.contract is not None and not isinstance(cls.contract, ProcessingContract):
            raise TypeError(
                f"{cls.__name__}.contract must be a ProcessingContract or None, "
                f"got {cls.contract!r}."
            )
        if cls.function_variants and cls.contract is not ProcessingContract.FLEXIBLE:
            raise ValueError(
                f"{cls.__name__} declares function_variants and must use "
                "contract=ProcessingContract.FLEXIBLE."
            )
        cls.default_variable_components = tuple(
            (
                component
                if isinstance(component, VariableComponents)
                else VariableComponents(component)
            )
            for component in cls.default_variable_components
        )
        cls.confidence = float(cls.confidence)
        cls.validated = bool(cls.validated)
        cls.required_variable_components = tuple(
            (
                (
                    component
                    if isinstance(component, VariableComponents)
                    else VariableComponents(component)
                )
                for component in cls.required_variable_components
            )
        )
        if cls.group_by is None:
            raise TypeError(
                f"{cls.__name__}.group_by must be a GroupBy value; "
                "GroupBy.NONE is generated only by the variable-component "
                "collision rule."
            )
        cls.group_by = (
            cls.group_by
            if isinstance(cls.group_by, GroupBy)
            else GroupBy(cls.group_by)
        )
        cls.allowed_group_by = tuple(
            (
                group_by if isinstance(group_by, GroupBy) else GroupBy(group_by)
                for group_by in cls.allowed_group_by
            )
        )
        _validate_unique_module_names(cls)
        CellProfilerModule.for_module.__func__.cache_clear()
        CellProfilerModule._measurement_feature_marker_types_for_key_payload.__func__.cache_clear()
        CellProfilerModule.measurement_feature_types.__func__.cache_clear()
        CellProfilerModule.alternative_measurement_feature_part_aliases.__func__.cache_clear()
        CellProfilerModule.measurement_feature_part_rewrite_declarations.__func__.cache_clear()
        CellProfilerModule.measurement_category_prefix_declarations.__func__.cache_clear()
        CellProfilerModule.measurement_source_feature_prefix_declarations.__func__.cache_clear()
        CellProfilerModule.calculated_measurement_feature_prefix_declarations.__func__.cache_clear()
        CellProfilerModule.numbered_measurement_feature_prefix_alias_declarations.__func__.cache_clear()
        CellProfilerModule.scale_qualified_measurement_feature_prefix_declarations.__func__.cache_clear()

    @classmethod
    def declared_function_names(cls) -> tuple[str, ...]:
        """Return the primary and variant function names declared by this module."""
        return (str(cls.function_name), *cls.function_variants)

    @classmethod
    def normalize_setting_name(cls, setting_name: str) -> str:
        """Return the canonical lookup key for CellProfiler setting labels."""
        del cls
        return normalize_cellprofiler_setting_name(setting_name)

    @classmethod
    @lru_cache(maxsize=512)
    def for_module(cls, module_name: str) -> type["CellProfilerModule"] | None:
        """Return the registered module class for a canonical name or alias."""
        lookup_key = _module_lookup_key(module_name)
        for module_type in cls.__registry__.values():
            if _module_lookup_key(str(module_type.module_name)) == lookup_key:
                return module_type
            if lookup_key in {
                _module_lookup_key(alias) for alias in module_type.aliases
            }:
                return module_type
        return None

    @classmethod
    def canonical_module_name(cls, module_name: str) -> str:
        """Return the canonical module name declared by the module class root."""
        module_type = cls.for_module(module_name)
        if module_type is None:
            return _required_string(module_name, "module_name", cls.__name__)
        return str(module_type.module_name)

    @classmethod
    def require_authority_type(
        cls,
        authority_type: type[AuthorityT],
    ) -> type[AuthorityT]:
        """Return a valid CellProfiler module authority root."""
        if not isinstance(authority_type, type) or not issubclass(
            authority_type,
            CellProfilerModuleAuthority,
        ):
            raise TypeError(
                f"{cls.__name__} authority must inherit CellProfilerModuleAuthority."
            )
        return authority_type

    @classmethod
    def declared_authority_types(
        cls,
        authority_root: type[AuthorityT],
    ) -> tuple[type[AuthorityT], ...]:
        """Return most-derived authority types declared by this module's MRO."""
        cls.require_authority_type(authority_root)
        matching_authority_types = tuple(
            dict.fromkeys(
                candidate_type
                for candidate_type in cls.__mro__
                if candidate_type is not cls
                and candidate_type is not authority_root
                and candidate_type is not CellProfilerModuleAuthority
                and issubclass(candidate_type, authority_root)
            )
        ) + tuple(
            dict.fromkeys(
                nested_type
                for owner_type in cls.__mro__
                for nested_type in owner_type.__dict__.values()
                if isinstance(nested_type, type)
                and nested_type is not authority_root
                and nested_type is not CellProfilerModuleAuthority
                and issubclass(nested_type, authority_root)
            )
        )
        return tuple(
            candidate_type
            for candidate_type in matching_authority_types
            if not any(
                other_type is not candidate_type
                and issubclass(other_type, candidate_type)
                for other_type in matching_authority_types
            )
        )

    @classmethod
    def derived_measurement_feature_relation_declarations(
        cls,
    ) -> tuple[RuntimeMeasurementFeatureRelationDeclaration, ...]:
        """Return measurement-feature relations derived from module-owned markers."""
        return ()

    @classmethod
    def measurement_feature_relation_declarations(
        cls,
    ) -> tuple[RuntimeMeasurementFeatureRelationDeclaration, ...]:
        """Return producer-owned measurement-feature relations from modules."""
        return tuple(
            (
                relation
                for module_type in cls.__registry__.values()
                for feature_type in module_type.measurement_feature_types()
                for feature in feature_type
                for relation in feature.relation_declarations()
            )
        ) + tuple(
            relation
            for module_type in cls.__registry__.values()
            for relation in module_type.derived_measurement_feature_relation_declarations()
        )

    @classmethod
    def measurement_feature_family_parts(cls) -> tuple[tuple[str, ...], ...]:
        """Return feature-family parts declared by registered modules."""
        return tuple(
            (
                tuple((part for part in feature.feature_family().split("_") if part))
                for module_type in cls.__registry__.values()
                for feature_type in module_type.measurement_feature_types()
                for feature in feature_type
            )
        )

    @classmethod
    def source_qualified_measurement_feature_family_parts(
        cls,
    ) -> tuple[tuple[str, ...], ...]:
        """Return module-owned feature families that carry source names."""
        return tuple(
            (
                tuple((part for part in feature.feature_family().split("_") if part))
                for module_type in cls.__registry__.values()
                for feature_type in module_type.source_qualified_measurement_feature_types()
                for feature in feature_type
            )
        )

    @classmethod
    def source_qualified_measurement_feature_types(
        cls,
    ) -> tuple[type[RuntimeMeasurementFeature], ...]:
        """Return measurement feature enums whose rows are source-qualified."""
        return ()

    @classmethod
    def measurement_feature_marker_types_for_key(
        cls,
        key: Any,
        dialect: Any,
    ) -> tuple[type[RuntimeMeasurementFeatureSemanticMarker], ...]:
        """Return module-owned semantic marker authorities matching ``key``."""
        from openhcs.core.equivalence.keys import RuntimeMeasurementFeatureKey

        del dialect
        if not isinstance(key, RuntimeMeasurementFeatureKey):
            raise TypeError(
                "CellProfilerModule.measurement_feature_marker_types_for_key "
                "requires RuntimeMeasurementFeatureKey."
            )
        return cls._measurement_feature_marker_types_for_key_payload(
            key.to_cache_payload()
        )

    @classmethod
    @lru_cache(maxsize=4096)
    def _measurement_feature_marker_types_for_key_payload(
        cls,
        key_payload: object,
    ) -> tuple[type[RuntimeMeasurementFeatureSemanticMarker], ...]:
        """Return module-owned marker authorities keyed by semantic feature payload."""
        from openhcs.core.equivalence.keys import RuntimeMeasurementFeatureKey

        key = RuntimeMeasurementFeatureKey.from_cache_payload(key_payload)
        return tuple(
            dict.fromkeys(
                authority_type
                for module_type in cls.__registry__.values()
                for authority_type in module_type.__mro__
                if authority_type is not CellProfilerMeasurementFeatureMarker
                and authority_type is not RuntimeMeasurementFeatureSemanticMarker
                and isinstance(authority_type, type)
                and issubclass(authority_type, CellProfilerMeasurementFeatureMarker)
                if authority_type.matches_feature_key(module_type, key, None)
            )
        )

    @classmethod
    @lru_cache(maxsize=None)
    def measurement_feature_types(
        cls,
    ) -> tuple[type[RuntimeMeasurementFeature], ...]:
        """Return measurement feature enums declared on this module's MRO."""
        return tuple(
            dict.fromkeys(
                feature_type
                for owner_type in cls.__mro__
                for feature_type in owner_type.__dict__.values()
                if isinstance(feature_type, type)
                and issubclass(feature_type, RuntimeMeasurementFeature)
                and feature_type is not RuntimeMeasurementFeature
            )
        )

    @classmethod
    @lru_cache(maxsize=1)
    def alternative_measurement_feature_part_aliases(
        cls,
    ) -> Mapping[tuple[str, ...], tuple[tuple[str, ...], ...]]:
        """Return module-owned alternative feature-family aliases."""
        aliases: dict[tuple[str, ...], tuple[tuple[str, ...], ...]] = {}
        for module_type in cls.__registry__.values():
            for (
                source,
                alternatives,
            ) in module_type.measurement_feature_part_aliases.items():
                aliases[tuple(source)] = tuple((tuple(alias) for alias in alternatives))
        return aliases

    @classmethod
    @lru_cache(maxsize=1)
    def measurement_feature_part_rewrite_declarations(
        cls,
    ) -> Mapping[tuple[str, ...], tuple[str, ...]]:
        """Return module-owned direct feature-family rewrites."""
        aliases: dict[tuple[str, ...], tuple[str, ...]] = {}
        for module_type in cls.__registry__.values():
            for source, target in module_type.measurement_feature_part_rewrites.items():
                aliases[tuple(source)] = tuple(target)
        return aliases

    @classmethod
    @lru_cache(maxsize=1)
    def measurement_category_prefix_declarations(cls) -> tuple[tuple[str, ...], ...]:
        """Return module-owned CellProfiler measurement category prefixes."""
        return tuple(
            dict.fromkeys(
                prefix
                for module_type in cls.__registry__.values()
                for prefix in module_type.measurement_category_prefixes
            )
        )

    @classmethod
    @lru_cache(maxsize=1)
    def measurement_source_feature_prefix_declarations(
        cls,
    ) -> tuple[tuple[str, ...], ...]:
        """Return module-owned source-qualified feature prefixes."""
        return tuple(
            dict.fromkeys(
                prefix
                for module_type in cls.__registry__.values()
                for prefix in module_type.measurement_source_feature_prefixes
            )
        )

    @classmethod
    @lru_cache(maxsize=1)
    def calculated_measurement_feature_prefix_declarations(
        cls,
    ) -> tuple[tuple[str, ...], ...]:
        """Return module-owned calculated feature prefixes."""
        return tuple(
            dict.fromkeys(
                prefix
                for module_type in cls.__registry__.values()
                for prefix in module_type.calculated_measurement_feature_prefixes
            )
        )

    @classmethod
    @lru_cache(maxsize=1)
    def numbered_measurement_feature_prefix_alias_declarations(
        cls,
    ) -> Mapping[str, tuple[str, ...]]:
        """Return module-owned numbered feature-prefix aliases."""
        aliases: dict[str, tuple[str, ...]] = {}
        for module_type in cls.__registry__.values():
            aliases.update(module_type.numbered_measurement_feature_prefix_aliases)
        return aliases

    @classmethod
    @lru_cache(maxsize=1)
    def scale_qualified_measurement_feature_prefix_declarations(
        cls,
    ) -> tuple[tuple[str, ...], ...]:
        """Return module-owned scale-qualified feature prefixes."""
        return tuple(
            dict.fromkeys(
                prefix
                for module_type in cls.__registry__.values()
                for prefix in module_type.scale_qualified_measurement_feature_prefixes
            )
        )

    @classmethod
    def directional_pair_feature_alias_declarations(
        cls,
    ) -> Mapping[str, tuple[str, int]]:
        """Return module-owned directional pair aliases."""
        aliases: dict[str, tuple[str, int]] = {}
        for module_type in cls.__registry__.values():
            aliases.update(module_type.directional_pair_feature_aliases)
        return aliases

    @classmethod
    def pair_correlation_feature_name_declaration(cls) -> str | None:
        """Return the module-owned pair correlation feature name."""
        names = tuple(
            module_type.pair_correlation_feature_name
            for module_type in cls.__registry__.values()
            if module_type.pair_correlation_feature_name is not None
        )
        if len(set(names)) > 1:
            raise ValueError(f"Conflicting pair correlation feature names: {names!r}.")
        return names[0] if names else None

    @classmethod
    def pair_regression_slope_feature_name_declaration(cls) -> str | None:
        """Return the module-owned pair regression-slope feature name."""
        names = tuple(
            module_type.pair_regression_slope_feature_name
            for module_type in cls.__registry__.values()
            if module_type.pair_regression_slope_feature_name is not None
        )
        if len(set(names)) > 1:
            raise ValueError(f"Conflicting pair regression feature names: {names!r}.")
        return names[0] if names else None

    @classmethod
    def undirected_pair_feature_name_declarations(cls) -> frozenset[str]:
        """Return module-owned pair features that have no direction."""
        return frozenset(
            feature_name
            for module_type in cls.__registry__.values()
            for feature_name in module_type.undirected_pair_feature_names
        )

    @classmethod
    def threshold_sensitive_pair_feature_name_declarations(cls) -> frozenset[str]:
        """Return module-owned threshold-sensitive pair feature names."""
        return frozenset(
            feature_name
            for module_type in cls.__registry__.values()
            for feature_name in module_type.threshold_sensitive_pair_feature_names
        )

    @classmethod
    def semantic_default_contracts(
        cls,
    ) -> tuple["CellProfilerSemanticDefaultContract", ...]:
        """Return source-validation contracts owned by this module declaration."""
        contracts = []
        for contract_type in cls.semantic_default_contract_types:
            contract = contract_type()
            if contract.module_name is None:
                contract.module_name = cls.semantic_default_contract_module_name or str(
                    cls.module_name
                )
            contracts.append(contract)
        return tuple(contracts)

    @classmethod
    def _bind_declared_settings(
        cls,
        module: "ModuleBlock",
        *,
        binder: "SettingsBinder",
        param_mapping: Mapping[str, Any],
    ) -> "BoundModuleSettings":
        """Bind rows using setting declarations inherited by this module class."""
        del param_mapping
        setting_bindings = tuple(cls.setting_bindings)
        ignored_settings = tuple(cls.ignored_settings_for(module))
        bound_details = binder.bind_with_details(module.settings)
        kwargs = binder.bind_declared(module, setting_bindings)
        mapped_settings = {
            _normalize_setting_name(setting_name)
            for binding in setting_bindings
            for setting_name in setting_names(binding.setting_name)
        }
        mapped_settings.update(
            (
                _normalize_setting_name(concrete_setting_name)
                for setting_name in ignored_settings
                for concrete_setting_name in setting_names(setting_name)
            )
        )
        unmapped_kwargs = {
            detail.name: detail.original_value
            for detail in bound_details
            if detail.name not in mapped_settings
        }
        return BoundModuleSettings(kwargs, unmapped_kwargs)

    @classmethod
    def _bind_generic_settings(
        cls,
        module: "ModuleBlock",
        *,
        binder: "SettingsBinder",
        param_mapping: Mapping[str, Any],
        use_declaration: bool = True,
    ) -> "BoundModuleSettings":
        """Bind docstring/signature-mapped settings through this declaration."""
        from enum import Enum as EnumType
        from inspect import signature
        from typing import Literal, get_args, get_origin, get_type_hints
        from openhcs.processing.backends.cellprofiler.function_documentation import (
            cellprofiler_source_setting_parameter_mapping,
        )
        from openhcs.processing.backends.cellprofiler.library import require_function

        setting_parameter_mapping = dict(param_mapping)
        if use_declaration:
            absorbed_function = require_function(
                str(cls.module_name), function_name=str(cls.function_name)
            )
            parameter_names = tuple(signature(absorbed_function).parameters)
            setting_parameter_mapping.update({name: name for name in parameter_names})
            setting_parameter_mapping.update(
                cellprofiler_source_setting_parameter_mapping(
                    str(cls.module_name), parameter_names
                )
            )
            for setting_name, parameter_name in cls.setting_parameter_aliases.items():
                for concrete_setting_name in setting_names(setting_name):
                    setting_parameter_mapping[
                        _normalize_setting_name(concrete_setting_name)
                    ] = parameter_name
            for setting_name in cls.ignored_settings_for(module):
                for concrete_setting_name in setting_names(setting_name):
                    setting_parameter_mapping[
                        _normalize_setting_name(concrete_setting_name)
                    ] = None
            annotations = get_type_hints(absorbed_function)
        else:
            annotations = {}
        bound_kwargs = binder.bind(module.settings)
        coerced_kwargs: dict[str, Any] = {}
        for setting_name, value in bound_kwargs.items():
            parameter_name = setting_parameter_mapping.get(setting_name)
            annotation = (
                annotations.get(parameter_name)
                if isinstance(parameter_name, str)
                else None
            )
            if annotation is None:
                coerced_kwargs[setting_name] = value
                continue
            origin = get_origin(annotation)
            if origin is Literal and isinstance(value, str):
                normalized_value = _normalize_setting_name(value)
                for literal in get_args(annotation):
                    if isinstance(
                        literal, str
                    ) and normalized_value == _normalize_setting_name(literal):
                        value = literal
                        break
            elif isinstance(annotation, type) and issubclass(annotation, EnumType):
                value = coerce_cellprofiler_enum(annotation, value)
            coerced_kwargs[setting_name] = value
        translated_kwargs: dict[str, Any] = {}
        unmapped_kwargs: dict[str, Any] = {}
        for cp_setting, value in coerced_kwargs.items():
            if cp_setting not in setting_parameter_mapping:
                unmapped_kwargs[cp_setting] = value
                continue
            py_param = setting_parameter_mapping[cp_setting]
            if py_param is None:
                continue
            if isinstance(py_param, list):
                if isinstance(value, tuple) and len(value) == len(py_param):
                    for index, param_name in enumerate(py_param):
                        translated_kwargs[param_name] = value[index]
                else:
                    translated_kwargs[py_param[0]] = value
            else:
                translated_kwargs[py_param] = value
        return BoundModuleSettings(translated_kwargs, unmapped_kwargs)

    @classmethod
    def _finalize_bound_settings(
        cls,
        module: "ModuleBlock",
        *,
        binder: "SettingsBinder",
        bound: "BoundModuleSettings",
        ignored_unmapped_settings: frozenset[str] = frozenset(),
    ) -> "BoundModuleSettings":
        """Validate and annotate a binding result produced by this declaration."""
        from openhcs.interop.cellprofiler.artifact_semantics import (
            artifact_setting_symbols,
        )

        runtime_kwargs = cls.runtime_kwargs(module)
        if runtime_kwargs:
            bound = bound.with_kwargs(runtime_kwargs)
        bound = bound.with_replaced_kwargs(
            cls._coerce_kwargs_to_callable_signature(bound.kwargs)
        )
        artifact_setting_names = frozenset(
            (
                _normalize_setting_name(symbol.setting_name)
                for symbol in artifact_setting_symbols(module)
            )
        )
        typed_ignore_setting_names = frozenset(
            (
                _normalize_setting_name(concrete_name)
                for setting_name in cls.ignored_settings_for(module)
                for concrete_name in setting_names(setting_name)
            )
        )
        unmapped_kwargs = {
            setting_name: value
            for setting_name, value in bound.unmapped_kwargs.items()
            if setting_name not in ignored_unmapped_settings
            and setting_name not in artifact_setting_names
            and (setting_name not in typed_ignore_setting_names)
        }
        setting_coverage: list[ModuleSettingCoverageRecord] = []
        for setting in module.iter_settings():
            normalized_name = _normalize_setting_name(setting.name)
            setting_coverage.append(
                ModuleSettingCoverageRecord(
                    module_name=module.name,
                    module_num=module.module_num,
                    setting_name=setting.name,
                    normalized_setting_name=normalized_name,
                    value=setting.value,
                    status=ModuleSettingCoverageStatus.for_setting(
                        normalized_name,
                        binder=binder,
                        unmapped_kwargs=bound.unmapped_kwargs,
                        ignored_unmapped_settings=ignored_unmapped_settings,
                        artifact_setting_names=artifact_setting_names,
                        typed_ignore_setting_names=typed_ignore_setting_names,
                    ),
                )
            )
        if unmapped_kwargs:
            raise UnmappedModuleSettingsError(
                tuple(
                    (
                        UnmappedModuleSetting(
                            module_name=module.name,
                            module_num=module.module_num,
                            setting_name=setting_name,
                            value=value,
                        )
                        for setting_name, value in sorted(unmapped_kwargs.items())
                    )
                )
            )
        return BoundModuleSettings(
            bound.kwargs,
            unmapped_kwargs,
            bound.invocation_options,
            tuple(setting_coverage),
        )

    @classmethod
    def _coerce_kwargs_to_callable_signature(
        cls,
        kwargs: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Return kwargs using enum classes declared by the runtime callable."""
        from inspect import Parameter, signature
        from openhcs.processing.backends.cellprofiler.library import require_function

        if not kwargs:
            return kwargs
        absorbed_function = require_function(
            str(cls.module_name), function_name=str(cls.function_name)
        )
        annotations = get_type_hints(absorbed_function)
        parameters = signature(absorbed_function).parameters
        coerced: dict[str, Any] = {}
        for parameter_name, value in kwargs.items():
            parameter = parameters.get(parameter_name)
            if parameter is None:
                coerced[parameter_name] = value
                continue
            annotation = annotations.get(parameter_name)
            enum_type = (
                _enum_type_from_annotation(annotation)
                if annotation is not None
                else None
            )
            if (
                enum_type is None
                and parameter.default is not Parameter.empty
                and isinstance(parameter.default, Enum)
            ):
                enum_type = type(parameter.default)
            if enum_type is None:
                coerced[parameter_name] = value
                continue
            coerced[parameter_name] = _coerce_callable_enum_kwarg(value, enum_type)
        return coerced

    @classmethod
    def bind_settings(
        cls,
        module: "ModuleBlock",
        *,
        binder: "SettingsBinder",
        param_mapping: Mapping[str, Any],
        ignored_unmapped_settings: frozenset[str] = frozenset(),
    ) -> "BoundModuleSettings":
        """Bind parsed module settings through this module declaration."""
        if cls.setting_parameter_aliases:
            bound = cls._bind_generic_settings(
                module, binder=binder, param_mapping=param_mapping
            )
            return cls._finalize_bound_settings(
                module,
                binder=binder,
                bound=cls.postprocess_bound_settings(module, bound),
                ignored_unmapped_settings=ignored_unmapped_settings,
            )
        if cls.setting_bindings:
            bound = cls._bind_declared_settings(
                module, binder=binder, param_mapping=param_mapping
            )
            return cls._finalize_bound_settings(
                module,
                binder=binder,
                bound=cls.postprocess_bound_settings(module, bound),
                ignored_unmapped_settings=ignored_unmapped_settings,
            )
        bound = cls._bind_generic_settings(
            module,
            binder=binder,
            param_mapping=param_mapping,
            use_declaration=bool(cls.ignored_settings),
        )
        return cls._finalize_bound_settings(
            module,
            binder=binder,
            bound=cls.postprocess_bound_settings(module, bound),
            ignored_unmapped_settings=ignored_unmapped_settings,
        )

    @classmethod
    def postprocess_bound_settings(
        cls, module: "ModuleBlock", bound: "BoundModuleSettings"
    ) -> "BoundModuleSettings":
        """Apply module-local binding semantics after declared settings bind."""
        del module
        return bound

    @classmethod
    def generated_module_blocks(cls, module: "ModuleBlock") -> tuple["ModuleBlock", ...]:
        """Return module-owned OpenHCS execution blocks for generated pipelines."""
        return (module,)

    @classmethod
    def runtime_kwargs(cls, module: "ModuleBlock") -> Mapping[str, Any]:
        """Return declaration-owned runtime-selection kwargs for this module."""
        del module
        return {}

    @classmethod
    def compile_time_setting_records_from_kwargs(
        cls, kwargs: Mapping[str, Any]
    ) -> tuple["ModuleSetting", ...]:
        """Return module-owned CP setting rows reconstructed from public kwargs."""
        from openhcs.interop.cellprofiler.cellprofiler_literals import (
            cellprofiler_setting_literal,
        )
        from openhcs.interop.cellprofiler.parser import ModuleSetting

        return tuple(
            ModuleSetting(
                setting_names(binding.setting_name)[0],
                cellprofiler_setting_literal(kwargs[binding.parameter_name]),
            )
            for binding in cls.setting_bindings
            if binding.parameter_name in kwargs
        )

    @classmethod
    def compile_time_setting_records_for_invocation(
        cls,
        request: CellProfilerCompileTimeSettingsRequest,
    ) -> tuple["ModuleSetting", ...]:
        """Return all compiler-only CellProfiler setting rows for one invocation."""
        records = [
            *cls.compile_time_setting_records_from_kwargs(request.kwargs),
            *cls.compile_time_public_setting_records_from_kwargs(request.kwargs),
        ]
        records.extend(
            cls.compile_time_source_binding_input_setting_records(
                request,
                existing_records=tuple(records),
            )
        )
        return tuple(records)

    @classmethod
    def compile_time_source_binding_input_setting_records(
        cls,
        request: CellProfilerCompileTimeSettingsRequest,
        *,
        existing_records: tuple["ModuleSetting", ...],
    ) -> tuple["ModuleSetting", ...]:
        """Infer missing declared input settings from step source bindings."""
        from openhcs.core.source_bindings import StepSourceBindingsConfig
        from openhcs.interop.cellprofiler.parser import ModuleSetting

        source_bindings = request.source_bindings
        if not isinstance(source_bindings, StepSourceBindingsConfig):
            return ()
        if not source_bindings.enabled:
            return ()

        missing_settings = cls._missing_declared_input_settings(existing_records)
        if not missing_settings:
            return ()

        binding_declarations = cls._source_binding_declarations_for_group(request)
        bindings_by_artifact_type: dict[type[ArtifactType], list[Any]] = {}
        for binding in binding_declarations:
            bindings_by_artifact_type.setdefault(binding.artifact_kind, []).append(
                binding
            )

        inferred_records: list[ModuleSetting] = []
        consumed_by_artifact_type: dict[type[ArtifactType], int] = {}
        for setting_name, artifact_type in missing_settings:
            candidates = bindings_by_artifact_type.get(artifact_type, [])
            consumed = consumed_by_artifact_type.get(artifact_type, 0)
            if consumed >= len(candidates):
                raise ValueError(
                    f"Module {request.module_name}({request.module_num}) is missing "
                    f"CellProfiler input setting {setting_name!r}; source bindings "
                    f"declare only {[binding.alias for binding in candidates]!r} "
                    f"for {artifact_type.require_value()} artifacts."
                )
            inferred_records.append(ModuleSetting(setting_name, candidates[consumed].alias))
            consumed_by_artifact_type[artifact_type] = consumed + 1

        for artifact_type, consumed in consumed_by_artifact_type.items():
            candidates = bindings_by_artifact_type.get(artifact_type, [])
            if consumed != len(candidates):
                unused_aliases = [binding.alias for binding in candidates[consumed:]]
                raise ValueError(
                    f"Module {request.module_name}({request.module_num}) has "
                    f"ambiguous source bindings for {artifact_type.require_value()} "
                    f"inputs; unused aliases={unused_aliases!r}."
                )
        return tuple(inferred_records)

    @classmethod
    def _source_binding_declarations_for_group(
        cls,
        request: CellProfilerCompileTimeSettingsRequest,
    ) -> tuple[Any, ...]:
        """Return source bindings addressed by this function-pattern group."""
        return request.source_bindings.bindings_for_group_key(request.group_key)

    @classmethod
    def _missing_declared_input_settings(
        cls, existing_records: tuple["ModuleSetting", ...]
    ) -> tuple[tuple[str, type[ArtifactType]], ...]:
        existing_setting_names = {record.name for record in existing_records}
        missing_settings: list[tuple[str, type[ArtifactType]]] = []
        for setting, capability_type in cls.compile_time_required_artifact_input_settings():
            concrete_names = setting_names(setting)
            if any(name in existing_setting_names for name in concrete_names):
                continue
            missing_settings.append(
                (concrete_names[0], capability_type.require_artifact_type())
            )
        return tuple(missing_settings)

    @classmethod
    def compile_time_required_artifact_input_settings(
        cls,
    ) -> tuple[ArtifactSettingCapability, ...]:
        """Return artifact input settings that must be reconstructable from steps."""
        return cls.declared_artifact_input_settings()

    @classmethod
    def compile_time_public_setting_names(
        cls,
    ) -> tuple[str | "SettingNameFamily", ...]:
        """Return declared CP setting families projected as public compile kwargs."""
        return tuple(
            setting
            for setting, _capability_type in (
                *cls.declared_artifact_input_settings(),
                *cls.declared_artifact_output_settings(),
            )
        )

    @classmethod
    def compile_time_public_kwarg_names(cls) -> tuple[str, ...]:
        """Return public compile-only kwarg names derived from CP setting names."""
        return tuple(
            dict.fromkeys(
                normalize_cellprofiler_setting_name(concrete_name)
                for setting_name in cls.compile_time_public_setting_names()
                for concrete_name in setting_names(setting_name)
            )
        )

    @classmethod
    def compile_time_public_setting_records_from_kwargs(
        cls, kwargs: Mapping[str, Any]
    ) -> tuple["ModuleSetting", ...]:
        """Return compile-time-only CP setting rows from public kwargs."""
        from openhcs.interop.cellprofiler.cellprofiler_literals import (
            cellprofiler_setting_literal,
        )
        from openhcs.interop.cellprofiler.parser import ModuleSetting

        records: list[ModuleSetting] = []
        for setting_name in cls.compile_time_public_setting_names():
            for concrete_name in setting_names(setting_name):
                key = normalize_cellprofiler_setting_name(concrete_name)
                if key not in kwargs:
                    continue
                value = kwargs[key]
                if isinstance(value, tuple):
                    records.extend(
                        ModuleSetting(concrete_name, cellprofiler_setting_literal(item))
                        for item in value
                    )
                else:
                    records.append(
                        ModuleSetting(concrete_name, cellprofiler_setting_literal(value))
                    )
        return tuple(records)

    @classmethod
    def compile_time_public_setting_records(
        cls, module: "ModuleBlock", source_schema: "PipelineImageSchema | None" = None
    ) -> tuple["ModuleSetting", ...]:
        """Return CP setting rows needed by contracts but not runtime kwargs."""
        del module, source_schema
        return ()

    @classmethod
    def generated_invocation_options_literal(
        cls,
        options: RuntimeInvocationOptions | None,
        *,
        import_collector: GeneratedImportCollector,
    ) -> str | None:
        """Return generated-source literal for declaration-owned invocation options."""
        if options is None:
            return None
        return runtime_invocation_options_source_literal(
            options,
            import_collector=import_collector,
        )

    @classmethod
    def ignored_settings_for(
        cls, module: "ModuleBlock"
    ) -> tuple[str | "SettingNameFamily", ...]:
        """Return settings consumed outside direct runtime kwargs."""
        del module
        return cls.ignored_settings

    @classmethod
    def setting_value(
        cls,
        module: "ModuleBlock",
        setting_name: str | "SettingNameFamily",
        *,
        include_blank: bool = False,
    ) -> str | None:
        """Return a module setting value through the module declaration boundary."""
        if not include_blank:
            return _optional_setting_value(module, setting_name)
        for setting in module.iter_settings():
            if _setting_name_matches(setting.name, setting_name):
                return setting.value.strip()
        for candidate_name, value in module.settings.items():
            if _setting_name_matches(candidate_name, setting_name):
                return value.strip()
        return None

    @classmethod
    def artifact_inputs(
        cls, module: "ModuleBlock", source_schema: "PipelineImageSchema"
    ) -> tuple["ModuleArtifactInput", ...]:
        """Return artifact inputs declared directly by this module class."""
        del source_schema
        from openhcs.interop.cellprofiler.module_artifact_inputs import (
            ModuleArtifactInput,
        )

        inputs: list[ModuleArtifactInput] = []
        for setting_name, capability_type in cls.declared_artifact_input_settings():
            setting_value = _optional_setting_value(
                module, cls.declared_setting_name(setting_name)
            )
            if setting_value is None:
                continue
            artifact_name = _normalized_symbol_name(setting_value)
            if artifact_name is not None:
                inputs.append(
                    ModuleArtifactInput(
                        artifact_name, capability_type.require_artifact_type()
                    )
                )
        return tuple(inputs)

    @classmethod
    def declared_artifact_input_settings(cls) -> tuple[ArtifactSettingCapability, ...]:
        """Return declared CellProfiler setting families and artifact capabilities."""
        return ()

    @classmethod
    def declared_artifact_output_settings(cls) -> tuple[ArtifactSettingCapability, ...]:
        """Return declared CellProfiler output setting families and capabilities."""
        return ()

    @classmethod
    def source_image_types_by_alias(cls, module: "ModuleBlock") -> Mapping[str, str]:
        """Return source-image role refinements implied by this module's inputs."""
        del module
        return {}

    @classmethod
    def resolve_function(
        cls, module: "ModuleBlock", *, default_function_name: str | None = None
    ) -> "ResolvedModuleFunction":
        """Return the function selected by this module declaration."""
        del module
        from openhcs.interop.cellprofiler.module_function_resolution import (
            ResolvedModuleFunction,
        )

        return ResolvedModuleFunction(default_function_name or str(cls.function_name))

    @classmethod
    def resolve_semantic_function(
        cls,
        module: "ModuleBlock",
        *,
        default_function_name: str | None = None,
        request: "ModuleProcessingComponentRequest",
    ) -> "ResolvedModuleFunction":
        """Return the function selected from module settings plus source semantics."""
        del request
        return cls.resolve_function(module, default_function_name=default_function_name)

    @classmethod
    def processing_components(
        cls, request: "ModuleProcessingComponentRequest"
    ) -> "ModuleProcessingComponents":
        """Return generated FunctionStep component semantics for this module."""
        from openhcs.interop.cellprofiler.module_processing_components import (
            default_module_processing_components,
        )

        components = default_module_processing_components(request)
        if cls.required_variable_components:
            if request.has_direct_source_bindings():
                components = components.validate_required_variable_components(
                    cls.required_variable_components,
                    module_name=str(cls.module_name),
                )
            else:
                components = components.with_required_variable_components(
                    cls.required_variable_components,
                    module_name=str(cls.module_name),
                )
        if components.has_group_by_resolution():
            return components
        return cls.with_generated_group_by(components)

    @classmethod
    def with_generated_group_by(
        cls, components: "ModuleProcessingComponents"
    ) -> "ModuleProcessingComponents":
        """Return components with this module's default group_by applied."""
        group_by = cls.generated_group_by(components)
        return components.with_group_by(group_by)

    @classmethod
    def generated_group_by(
        cls, components: "ModuleProcessingComponents"
    ) -> GroupBy:
        """Return this module's generated group_by after validation."""
        group_by = cls.group_by
        if group_by.value is not None and any(
            component.value == group_by.value
            for component in components.variable_components
        ):
            return GroupBy.NONE
        from openhcs.core.pipeline.funcstep_contract_validator import (
            FuncStepContractValidator,
        )

        return FuncStepContractValidator.normalized_group_by(
            group_by,
            components.variable_components,
            str(cls.module_name or cls.__name__),
        )

    @classmethod
    def runtime_object_measurement_row_policy(cls):
        """Return the object-measurement row policy declared by this module."""
        from openhcs.interop.cellprofiler.runtime.object_measurement_row_policies import (
            CellProfilerObjectMeasurementRowPolicy,
            DefaultObjectMeasurementRowPolicy,
        )

        if issubclass(cls, CellProfilerObjectMeasurementRowPolicy):
            return cls()
        return DefaultObjectMeasurementRowPolicy()

    @classmethod
    def infrastructure_retained_artifacts(
        cls,
        module: "ModuleBlock",
        *,
        contracts_by_module_num: Mapping[int, "ModuleArtifactContracts"],
    ) -> frozenset["ArtifactSpecKey"]:
        """Return artifacts this module keeps alive when handled as infrastructure."""
        del module, contracts_by_module_num
        return frozenset()

    @classmethod
    def image_export_specs(
        cls, module: "ModuleBlock"
    ) -> tuple["RuntimeImageExportSpec", ...]:
        """Return runtime image-export expectations declared by this module."""
        del module
        return ()

    @classmethod
    def measurement_artifact_name(cls, module: "ModuleBlock") -> str:
        """Return the standard CellProfiler measurement artifact name."""
        return f"{module.name}_{module.module_num}_measurements"

    @classmethod
    def measurement_output_relations(
        cls, builder: "_SymbolTableBuilder", module: "ModuleBlock"
    ) -> tuple[ArtifactSpecRelation, ...]:
        """Return semantic relations for the standard measurement output."""
        del builder, module
        return ()

    @classmethod
    def declared_setting_name(
        cls, setting: str | "SettingNameFamily"
    ) -> str | "SettingNameFamily":
        """Return a concrete setting declaration."""
        return setting

    @staticmethod
    def declared_setting_value(
        setting: str | "SettingNameFamily" | Callable[[], str | "SettingNameFamily"],
    ) -> str | "SettingNameFamily":
        """Resolve a declaration supplied as a value or lazy class hook."""
        return setting() if callable(setting) else setting

    @classmethod
    def artifact_inputs_from_setting(
        cls,
        builder: "_SymbolTableBuilder",
        module: "ModuleBlock",
        setting: str | "SettingNameFamily" | None,
        capability_type: type[CellProfilerArtifactCapability],
    ) -> tuple[object, ...]:
        """Require artifacts named by one declared CellProfiler setting family."""
        if setting is None:
            return ()
        return tuple(
            (
                capability_type.bind_artifact(
                    cls,
                    builder,
                    module,
                    capability_type.spec(name),
                )
                for name in cls.artifact_input_names_from_setting(module, setting)
            )
        )

    @classmethod
    def artifact_input_names_from_setting(
        cls,
        module: "ModuleBlock",
        setting: str | "SettingNameFamily" | None,
    ) -> tuple[str, ...]:
        """Return artifact names selected by one declared CellProfiler setting."""
        from openhcs.interop.cellprofiler.setting_names import (
            setting_values,
            split_symbol_names,
        )

        if setting is None:
            return ()
        declared_setting = cls.declared_setting_name(
            cls.declared_setting_value(setting)
        )
        return tuple(
            name
            for value in setting_values(module, declared_setting)
            for name in split_symbol_names(value)
        )

    @classmethod
    def measurement_artifact_contract_from_declared_settings(
        cls,
        assembler: "CellProfilerContractAssemblyMixin",
        builder: "_SymbolTableBuilder",
        module: "ModuleBlock",
    ) -> "ModuleArtifactContracts":
        """Assemble the standard measurement contract from declared settings."""
        inputs = cls.measurement_artifact_inputs(builder, module)
        outputs = [
            MeasurementArtifactOutputCapability.bind_artifact(
                cls,
                builder,
                module,
                MeasurementArtifactOutputCapability.spec(
                    cls.measurement_artifact_name(module)
                ),
            )
        ]
        return assembler.assemble_contract(
            module, builder, inputs=inputs, outputs=outputs
        )

    @classmethod
    def measurement_artifact_inputs(
        cls, builder: "_SymbolTableBuilder", module: "ModuleBlock"
    ) -> tuple[object, ...]:
        """Return measurement-input artifacts from declared module settings."""
        return tuple(
            (
                artifact
                for setting, capability_type in cls.declared_artifact_input_settings()
                for artifact in cls.artifact_inputs_from_setting(
                    builder, module, setting, capability_type
                )
            )
        )

    @classmethod
    def artifact_contract_inputs(
        cls, builder: "_SymbolTableBuilder", module: "ModuleBlock"
    ) -> tuple[object, ...]:
        """Return artifacts consumed by this module's declared contract."""
        return cls.measurement_artifact_inputs(builder, module)

    @classmethod
    def artifact_contract_outputs(
        cls, builder: "_SymbolTableBuilder", module: "ModuleBlock"
    ) -> tuple[object, ...]:
        """Return artifacts produced by this module's declared contract."""
        return cls.declared_output_artifacts_from_settings(builder, module)

    @classmethod
    def declared_output_artifacts_from_settings(
        cls, builder: "_SymbolTableBuilder", module: "ModuleBlock"
    ) -> tuple[object, ...]:
        """Declare output artifacts named by CellProfiler setting families."""
        from openhcs.interop.cellprofiler.setting_names import (
            setting_values,
            split_symbol_names,
        )

        return tuple(
            (
                capability_type.bind_artifact(
                    cls,
                    builder,
                    module,
                    capability_type.spec(
                        name,
                        relations=cls.declared_output_artifact_relations(
                            builder,
                            module,
                            setting=setting,
                            capability_type=capability_type,
                            name=name,
                        ),
                    ),
                )
                for setting, capability_type in cls.declared_artifact_output_settings()
                for value in setting_values(
                    module,
                    cls.declared_setting_name(cls.declared_setting_value(setting)),
                )
                for name in split_symbol_names(value)
            )
        )

    @classmethod
    def declared_output_artifact_relations(
        cls,
        builder: "_SymbolTableBuilder",
        module: "ModuleBlock",
        *,
        setting: str | "SettingNameFamily",
        capability_type: type[CellProfilerArtifactCapability],
        name: str,
    ) -> tuple[ArtifactSpecRelation, ...]:
        """Return semantic relations for one declared output artifact."""
        del builder, module, setting, capability_type, name
        return ()

    @classmethod
    def measurement_output_artifact(
        cls, builder: "_SymbolTableBuilder", module: "ModuleBlock"
    ) -> object:
        """Declare the standard CellProfiler measurement output artifact."""
        return MeasurementArtifactOutputCapability.bind_artifact(
            cls,
            builder,
            module,
            MeasurementArtifactOutputCapability.spec(
                cls.measurement_artifact_name(module),
                relations=cls.measurement_output_relations(builder, module),
            ),
        )

    @classmethod
    def parent_child_relationship_output_artifact(
        cls,
        builder: "_SymbolTableBuilder",
        module: "ModuleBlock",
        *,
        parent_name: str,
        child_name: str,
    ) -> object:
        """Declare a relationship artifact from parent/child object artifacts."""
        from openhcs.core.runtime_semantics import (
            parent_child_relationship_artifact_name,
        )

        return RelationshipArtifactOutputCapability.bind_artifact(
            cls,
            builder,
            module,
            RelationshipArtifactOutputCapability.spec(
                parent_child_relationship_artifact_name(parent_name, child_name)
            ),
        )

    @classmethod
    def artifact_contract(
        cls,
        assembler: "CellProfilerContractAssemblyMixin",
        builder: "_SymbolTableBuilder",
        module: "ModuleBlock",
    ) -> "ModuleArtifactContracts | None":
        """Return typed declaration-owned artifact flow, when this module owns it."""
        inputs = cls.artifact_contract_inputs(builder, module)
        outputs = cls.artifact_contract_outputs(builder, module)
        if not inputs and (not outputs):
            return None
        return assembler.assemble_contract(
            module, builder, inputs=inputs, outputs=outputs
        )

    @classmethod
    def preserve_duplicate_artifact_inputs(cls, module: "ModuleBlock") -> bool:
        """Return whether repeated same-name inputs are distinct module roles."""
        del module
        return False

    @classmethod
    def source_binding_participates_in_image_stack(
        cls,
        module: "ModuleBlock",
        symbol: "CellProfilerSymbol",
        input_symbols: tuple["CellProfilerSymbol", ...],
    ) -> bool:
        """Return whether a source-bound symbol anchors image-stack execution."""
        del module, symbol, input_symbols
        return True

    @classmethod
    def relationship_measurement_rows(cls, request: object) -> object:
        """Return the relationship-row projector owned by this module declaration."""
        from openhcs.interop.cellprofiler.runtime.relationship_measurement_rows import (
            GenericRelationshipMeasurementRows,
        )

        return GenericRelationshipMeasurementRows(request)

    @classmethod
    def relationship_endpoint_contract(
        cls, resolver: object, relationship_spec: object
    ) -> object | None:
        """Return a declaration-owned endpoint contract for one relationship output."""
        del resolver, relationship_spec
        return None

    @classmethod
    def relationship_distance_measurements_apply(
        cls, resolver: object, relationship_spec: object
    ) -> bool:
        """Return whether this relationship output owns distance measurement rows."""
        del resolver, relationship_spec
        return False

    @classmethod
    def measurement_record(cls, request: object) -> object:
        """Return the measurement record declared by this module."""
        from openhcs.interop.cellprofiler.runtime.measurement_recording import (
            DefaultMeasurementRecordModule,
        )

        return DefaultMeasurementRecordModule.measurement_record(request)


class PlaneRuntimeArtifactModule(ABC):
    """Parent for modules that consume source-aligned runtime artifacts by plane."""

    allowed_group_by: ClassVar[tuple[GroupBy, ...]] = tuple(GroupBy)


class PerObjectMeasurementExecutionModule(PlaneRuntimeArtifactModule):
    """Parent for modules invoked once per measured object set."""


class ComposedImageObjectMeasurementExecutionModule(
    PerObjectMeasurementExecutionModule
):
    """Parent for object measurements that consume composed image payloads."""


class ObjectMeasurementRowsModule(CellProfilerModule):
    """Parent for modules whose declaration is also the object-row policy."""

    @classmethod
    def object_measurement_setting(cls) -> "SettingNameFamily":
        from openhcs.interop.cellprofiler.setting_names import SettingNameFamily

        return SettingNameFamily(
            "Select object sets to measure",
            aliases=("Select objects to measure", "Select an object to measure"),
        )


class InfrastructureCellProfilerModule(CellProfilerModule):
    """Parent for modules handled as OpenHCS import/runtime infrastructure."""

    @classmethod
    def declared_function_names(cls) -> tuple[str, ...]:
        """Infrastructure declarations are not executable backend functions."""
        return ()


class ModuleSettingsSourceModule(CellProfilerModule):
    """Parent for declarations that lower settings without binder context."""

    invocation_options_source: ClassVar[
        Callable[["ModuleBlock"], RuntimeInvocationOptions] | None
    ] = None

    @classmethod
    @abstractmethod
    def settings_source(cls, module: "ModuleBlock") -> "CellProfilerKwargs":
        """Return absorbed-function kwargs owned by this module declaration."""
        raise NotImplementedError

    @classmethod
    def bind_settings(
        cls,
        module: "ModuleBlock",
        *,
        binder: "SettingsBinder",
        param_mapping: Mapping[str, Any],
        ignored_unmapped_settings: frozenset[str] = frozenset(),
    ) -> "BoundModuleSettings":
        del param_mapping
        return cls._finalize_bound_settings(
            module,
            binder=binder,
            bound=BoundModuleSettings(
                cls.settings_source(module),
                {},
                (
                    cls.invocation_options_source(module)
                    if cls.invocation_options_source is not None
                    else None
                ),
            ),
            ignored_unmapped_settings=ignored_unmapped_settings,
        )


class BinderSettingsSourceModule(CellProfilerModule):
    """Parent for declarations that lower settings with binder parsing."""

    invocation_options_source: ClassVar[
        Callable[["ModuleBlock"], RuntimeInvocationOptions] | None
    ] = None

    @classmethod
    @abstractmethod
    def settings_source(
        cls, module: "ModuleBlock", binder: "SettingsBinder"
    ) -> "CellProfilerKwargs":
        """Return absorbed-function kwargs owned by this module declaration."""
        raise NotImplementedError

    @classmethod
    def bind_settings(
        cls,
        module: "ModuleBlock",
        *,
        binder: "SettingsBinder",
        param_mapping: Mapping[str, Any],
        ignored_unmapped_settings: frozenset[str] = frozenset(),
    ) -> "BoundModuleSettings":
        del param_mapping
        return cls._finalize_bound_settings(
            module,
            binder=binder,
            bound=BoundModuleSettings(
                cls.settings_source(module, binder),
                {},
                (
                    cls.invocation_options_source(module)
                    if cls.invocation_options_source is not None
                    else None
                ),
            ),
            ignored_unmapped_settings=ignored_unmapped_settings,
        )


class CellProfilerStructuringElement(Enum):
    """CellProfiler morphology structuring-element shape literal."""

    DISK = "disk"
    SQUARE = "square"
    DIAMOND = "diamond"
    OCTAGON = "octagon"
    STAR = "star"
    BALL = "ball"
    CUBE = "cube"
    OCTAHEDRON = "octahedron"


STRUCTURING_ELEMENT_SETTING_NAME = "Structuring element"
DEFAULT_STRUCTURING_ELEMENT_SETTING = "disk,3"


@dataclass(frozen=True, slots=True)
class StructuringElementSetting:
    """Typed CellProfiler morphology footprint setting."""

    structuring_element: CellProfilerStructuringElement
    size: int

    @classmethod
    def from_cellprofiler_value(cls, value: Any) -> "StructuringElementSetting":
        from openhcs.interop.cellprofiler.settings_binder import (
            coerce_cellprofiler_enum,
        )

        shape, size = _structuring_element_parts(value)
        return cls(
            structuring_element=coerce_cellprofiler_enum(
                CellProfilerStructuringElement, shape
            ),
            size=_positive_size(size),
        )

    def bound_kwargs(
        self, *, shape_keyword: str = "structuring_element", size_keyword: str = "size"
    ) -> dict[str, str | int]:
        """Return generated-code-safe absorbed-function kwargs."""
        return {shape_keyword: self.structuring_element.value, size_keyword: self.size}


@dataclass(frozen=True, slots=True)
class StructuringElementSettingBinding:
    """Bind one named CellProfiler structuring-element setting to kwargs."""

    setting_name: str | "SettingNameFamily" = STRUCTURING_ELEMENT_SETTING_NAME
    legacy_size_setting_name: str | "SettingNameFamily" | None = "Size"
    default_value: str = DEFAULT_STRUCTURING_ELEMENT_SETTING
    shape_keyword: str = "structuring_element"
    size_keyword: str = "size"

    @property
    def normalized_setting_names(self) -> frozenset[str]:
        from openhcs.interop.cellprofiler.setting_names import setting_names

        names = set(
            (
                normalize_cellprofiler_setting_name(setting_name)
                for setting_name in setting_names(self.setting_name)
            )
        )
        if self.legacy_size_setting_name is not None:
            names.update(
                (
                    normalize_cellprofiler_setting_name(setting_name)
                    for setting_name in setting_names(self.legacy_size_setting_name)
                )
            )
        return frozenset(names)

    def bound_kwargs(
        self, module: "ModuleBlock", binder: "SettingsBinder"
    ) -> dict[str, str | int]:
        parsed_value = self.parsed_setting(module, binder)
        return StructuringElementSetting.from_cellprofiler_value(
            parsed_value
        ).bound_kwargs(shape_keyword=self.shape_keyword, size_keyword=self.size_keyword)

    def parsed_setting(
        self, module: "ModuleBlock", binder: "SettingsBinder"
    ) -> tuple[Any, Any]:
        from openhcs.interop.cellprofiler.setting_names import (
            optional_setting_value,
            setting_names,
        )

        raw_value = optional_setting_value(module, self.setting_name)
        if raw_value is not None:
            return _structuring_element_parts(
                binder.parse_value(setting_names(self.setting_name)[0], raw_value)
            )
        legacy_size = self.legacy_size(module, binder)
        if legacy_size is None:
            return _structuring_element_parts(
                binder.parse_value(
                    setting_names(self.setting_name)[0], self.default_value
                )
            )
        default_shape, _default_size = _structuring_element_parts(self.default_value)
        return (default_shape, legacy_size)

    def legacy_size(
        self, module: "ModuleBlock", binder: "SettingsBinder"
    ) -> Any | None:
        from openhcs.interop.cellprofiler.setting_names import (
            optional_setting_value,
            setting_names,
        )

        if self.legacy_size_setting_name is None:
            return None
        raw_value = optional_setting_value(module, self.legacy_size_setting_name)
        if raw_value is None:
            return None
        return binder.parse_value(
            setting_names(self.legacy_size_setting_name)[0], raw_value
        )


def structuring_element_bound_kwargs(
    module: "ModuleBlock",
    binder: "SettingsBinder",
    binding: StructuringElementSettingBinding = StructuringElementSettingBinding(),
) -> dict[str, str | int]:
    """Lower the common CellProfiler morphology setting into function kwargs."""
    return binding.bound_kwargs(module, binder)


def _structuring_element_parts(value: Any) -> tuple[Any, Any]:
    if isinstance(value, str):
        parts = tuple((part.strip() for part in value.split(",")))
    elif isinstance(value, (list, tuple)):
        parts = tuple(value)
    else:
        raise TypeError(
            f"Structuring element setting must be a comma-separated string or sequence, got {type(value).__name__}."
        )
    if len(parts) != 2:
        raise ValueError(
            f"Structuring element setting must contain shape and size, got {value!r}."
        )
    return (parts[0], parts[1])


def _positive_size(value: Any) -> int:
    size = int(value)
    if size <= 0:
        raise ValueError(f"Structuring element size must be positive: {size!r}")
    return size


class StructuringElementSettingsModule(BinderSettingsSourceModule):
    """Parent for modules sharing CellProfiler structuring-element lowering."""

    structuring_element_binding: ClassVar["StructuringElementSettingBinding | None"] = (
        None
    )

    @classmethod
    def _resolved_structuring_element_binding(
        cls,
    ) -> "StructuringElementSettingBinding":
        if cls.structuring_element_binding is not None:
            return cls.structuring_element_binding
        return StructuringElementSettingBinding()

    @classmethod
    def settings_source(
        cls, module: "ModuleBlock", binder: "SettingsBinder"
    ) -> "CellProfilerKwargs":
        binding = cls._resolved_structuring_element_binding()
        return binding.bound_kwargs(module, binder)

    @classmethod
    def bind_settings(
        cls,
        module: "ModuleBlock",
        *,
        binder: "SettingsBinder",
        param_mapping: Mapping[str, Any],
        ignored_unmapped_settings: frozenset[str] = frozenset(),
    ) -> "BoundModuleSettings":
        binding = cls._resolved_structuring_element_binding()
        if cls.setting_bindings:
            bound = cls._bind_declared_settings(
                module, binder=binder, param_mapping=param_mapping
            )
        else:
            bound = cls._bind_generic_settings(
                module,
                binder=binder,
                param_mapping=param_mapping,
                use_declaration=bool(cls.ignored_settings),
            )
        kwargs = dict(bound.kwargs)
        kwargs.update(binding.bound_kwargs(module, binder))
        unmapped_kwargs = dict(bound.unmapped_kwargs)
        for setting_name in binding.normalized_setting_names:
            unmapped_kwargs.pop(setting_name, None)
        return cls._finalize_bound_settings(
            module,
            binder=binder,
            bound=cls.postprocess_bound_settings(
                module,
                BoundModuleSettings(kwargs, unmapped_kwargs, bound.invocation_options),
            ),
            ignored_unmapped_settings=ignored_unmapped_settings,
        )


def _normalize_setting_name(value: str) -> str:
    return CellProfilerModule.normalize_setting_name(str(value))


def _setting_name_matches(
    candidate_name: str, declared_name: str | SettingNameFamily
) -> bool:
    normalized_candidate = _normalize_setting_name(candidate_name)
    return any(
        (
            normalized_candidate == _normalize_setting_name(name)
            for name in setting_names(declared_name)
        )
    )


def _setting_values(
    module: "ModuleBlock", setting_name: str | SettingNameFamily
) -> tuple[str, ...]:
    values: list[str] = []
    for setting in module.iter_settings():
        if _setting_name_matches(setting.name, setting_name):
            values.append(setting.value.strip())
    if values:
        return tuple(values)
    for candidate_name, value in module.settings.items():
        if _setting_name_matches(candidate_name, setting_name):
            values.append(value.strip())
    return tuple(values)


def _optional_setting_value(module: "ModuleBlock", setting_name: object) -> str | None:
    values = _setting_values(module, setting_name)
    return values[-1] if values else None


def _normalized_symbol_name(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _parse_cellprofiler_float(value: str) -> float:
    return float(str(value).strip())


def _parse_cellprofiler_int(value: str) -> int:
    return int(float(str(value).strip()))


def _parse_cellprofiler_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"yes", "true", "1"}:
        return True
    if normalized in {"no", "false", "0"}:
        return False
    raise ValueError(f"Unsupported CellProfiler boolean literal: {value!r}.")


def _cellprofiler_setting_token(value: Any) -> str:
    """Return a stable comparison token for parsed CellProfiler settings."""
    if isinstance(value, Enum) and isinstance(value.value, str):
        value = value.value
    return " ".join(str(value).strip().lower().replace("-", " ").split())


class RepeatedSettingValuePolicy(ABC, metaclass=AutoRegisterMeta):
    """Nominal resolver for CellProfiler settings that reuse the same label."""

    __registry_key__ = "policy_key"
    __skip_if_no_key__ = True
    setting_name: ClassVar[str | None] = None
    policy_key: ClassVar[str | None] = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls.__dict__.get("policy_key") is not None:
            return
        setting_name = cls.__dict__.get("setting_name")
        if isinstance(setting_name, str):
            cls.policy_key = setting_name

    @classmethod
    def for_setting(cls, setting_name: str) -> "RepeatedSettingValuePolicy":
        lookup_key = CellProfilerModule.normalize_setting_name(setting_name)
        strategy_type = next(
            (
                policy_type
                for policy_type in cls.__registry__.values()
                if CellProfilerModule.normalize_setting_name(
                    str(policy_type.policy_key)
                )
                == lookup_key
            ),
            LastRepeatedSettingValuePolicy,
        )
        return strategy_type()

    def value(
        self, module: "ModuleBlock", setting_name: str | "SettingNameFamily"
    ) -> str | None:
        values = _setting_values(module, setting_name)
        if not values:
            return None
        if len(values) == 1:
            return values[0]
        return self._resolve_repeated_value(module, setting_name, tuple(values))

    @abstractmethod
    def _resolve_repeated_value(
        self,
        module: "ModuleBlock",
        setting_name: str | "SettingNameFamily",
        values: tuple[str, ...],
    ) -> str:
        """Return the semantically active value for a repeated setting label."""


class LastRepeatedSettingValuePolicy(RepeatedSettingValuePolicy):
    """Default CellProfiler scalar behavior: the later row is authoritative."""

    def _resolve_repeated_value(
        self,
        module: "ModuleBlock",
        setting_name: str | "SettingNameFamily",
        values: tuple[str, ...],
    ) -> str:
        del module, setting_name
        return values[-1]


class ImageArtifactInputModule(
    CellProfilerModule,
    ArtifactContractModule,
    ImageArtifactInputCapability,
):
    """Parent for modules that consume image artifacts through declared settings."""

    image_input_settings: ClassVar[tuple[str | "SettingNameFamily", ...]] = ()

    @classmethod
    def image_input_setting_names(cls) -> tuple[str | "SettingNameFamily", ...]:
        """Return image-input setting families declared by this module."""
        return cls.image_input_settings

    @classmethod
    def declared_artifact_input_settings(cls) -> tuple[ArtifactSettingCapability, ...]:
        return (
            *super().declared_artifact_input_settings(),
            *(
                (setting, ImageArtifactInputCapability)
                for setting in cls.image_input_setting_names()
            ),
        )


class ObjectArtifactInputModule(
    CellProfilerModule,
    ArtifactContractModule,
    ObjectLabelArtifactInputCapability,
):
    """Parent for modules that consume object-label artifacts through declared settings."""

    object_input_settings: ClassVar[tuple[str | "SettingNameFamily", ...]] = ()

    @classmethod
    def object_input_setting_names(cls) -> tuple[str | "SettingNameFamily", ...]:
        """Return object-label input setting families declared by this module."""
        return cls.object_input_settings

    @classmethod
    def declared_artifact_input_settings(cls) -> tuple[ArtifactSettingCapability, ...]:
        return (
            *super().declared_artifact_input_settings(),
            *(
                (setting, ObjectLabelArtifactInputCapability)
                for setting in cls.object_input_setting_names()
            ),
        )


class ImageArtifactOutputModule(
    CellProfilerModule,
    ArtifactContractModule,
    ImageArtifactOutputCapability,
):
    """Parent for modules that emit image artifacts through declared settings."""

    image_output_settings: ClassVar[tuple[str | "SettingNameFamily", ...]] = ()

    @classmethod
    def image_output_setting_names(cls) -> tuple[str | "SettingNameFamily", ...]:
        """Return image-output setting families declared by this module."""
        return cls.image_output_settings

    @classmethod
    def declared_artifact_output_settings(cls) -> tuple[ArtifactSettingCapability, ...]:
        return (
            *super().declared_artifact_output_settings(),
            *(
                (setting, ImageArtifactOutputCapability)
                for setting in cls.image_output_setting_names()
            ),
        )

    @classmethod
    def declared_output_artifact_relations(
        cls,
        builder: "_SymbolTableBuilder",
        module: "ModuleBlock",
        *,
        setting: str | "SettingNameFamily",
        capability_type: type[CellProfilerArtifactCapability],
        name: str,
    ) -> tuple[ArtifactSpecRelation, ...]:
        relations = super().declared_output_artifact_relations(
            builder,
            module,
            setting=setting,
            capability_type=capability_type,
            name=name,
        )
        if capability_type is not ImageArtifactOutputCapability:
            return relations
        if not issubclass(cls, ImageArtifactInputModule):
            return relations
        source_name = cls.single_image_transform_runtime_input_name(builder, module)
        if source_name is None:
            return relations
        return (
            *relations,
            GroupLineageSourceRelation(
                source=ArtifactSpecRef.input(source_name, ImageArtifactType)
            ),
        )

    @classmethod
    def single_image_transform_runtime_input_name(
        cls,
        builder: "_SymbolTableBuilder",
        module: "ModuleBlock",
    ) -> str | None:
        """Return the only runtime image input name for a simple image transform."""
        from openhcs.interop.cellprofiler.setting_names import (
            setting_values,
            split_symbol_names,
        )

        input_settings = cls.image_input_setting_names()
        output_settings = cls.image_output_setting_names()
        if len(input_settings) != 1 or len(output_settings) != 1:
            return None
        input_names = tuple(
            name
            for value in setting_values(
                module,
                cls.declared_setting_name(
                    cls.declared_setting_value(input_settings[0])
                ),
            )
            for name in split_symbol_names(value)
        )
        if len(input_names) != 1:
            return None
        input_symbol = builder.optional_artifact(
            ArtifactSpec.input(input_names[0], ImageArtifactType)
        )
        if input_symbol is None:
            return None
        return input_symbol.name


class ObjectArtifactOutputModule(
    CellProfilerModule,
    ArtifactContractModule,
    ObjectLabelArtifactOutputCapability,
):
    """Parent for modules that emit object-label artifacts through declared settings."""

    object_output_settings: ClassVar[tuple[str | "SettingNameFamily", ...]] = ()

    @classmethod
    def object_output_setting_names(cls) -> tuple[str | "SettingNameFamily", ...]:
        """Return object-label output setting families declared by this module."""
        return cls.object_output_settings

    @classmethod
    def declared_artifact_output_settings(cls) -> tuple[ArtifactSettingCapability, ...]:
        return (
            *super().declared_artifact_output_settings(),
            *(
                (setting, ObjectLabelArtifactOutputCapability)
                for setting in cls.object_output_setting_names()
            ),
        )


class MeasurementArtifactInputModule(
    CellProfilerModule,
    ArtifactContractModule,
    MeasurementArtifactInputCapability,
):
    """Parent for modules that consume measurement artifacts."""


class PriorMeasurementArtifactInputModule(MeasurementArtifactInputModule):
    """Parent for modules that consume feature-addressed prior measurements."""

    @classmethod
    def prior_measurement_artifact_inputs(
        cls, builder: "_SymbolTableBuilder"
    ) -> tuple[object, ...]:
        """Return prior measurement artifacts available to feature queries."""
        return builder.measurement_outputs()

    @classmethod
    def artifact_contract_inputs(
        cls, builder: "_SymbolTableBuilder", module: "ModuleBlock"
    ) -> tuple[object, ...]:
        return (
            *super().artifact_contract_inputs(builder, module),
            *cls.prior_measurement_artifact_inputs(builder),
        )


class MeasurementArtifactOutputModule(
    CellProfilerModule,
    ArtifactContractModule,
    MeasurementArtifactOutputCapability,
):
    """Parent for modules that emit the standard measurement artifact."""

    @classmethod
    def measurement_output_relations(
        cls, builder: "_SymbolTableBuilder", module: "ModuleBlock"
    ) -> tuple[ArtifactSpecRelation, ...]:
        relations = super().measurement_output_relations(builder, module)
        lineage_source = cls.single_measurement_lineage_input(module)
        if lineage_source is None:
            return relations
        return (
            *relations,
            GroupLineageSourceRelation(source=lineage_source),
        )

    @classmethod
    def single_measurement_lineage_input(
        cls,
        module: "ModuleBlock",
    ) -> ArtifactSpecRef | None:
        """Return an unambiguous declared input scope for measurement outputs."""
        candidates: list[ArtifactSpecRef] = []
        if issubclass(cls, ImageArtifactInputModule):
            candidates.extend(
                ArtifactSpecRef.input(name, ImageArtifactType)
                for name in cls.single_declared_input_setting_names(
                    module,
                    cls.image_input_setting_names(),
                )
            )
        if issubclass(cls, ObjectArtifactInputModule):
            candidates.extend(
                ArtifactSpecRef.input(name, ObjectLabelsArtifactType)
                for name in cls.single_declared_input_setting_names(
                    module,
                    cls.object_input_setting_names(),
                )
            )
        if len(candidates) != 1:
            return None
        return candidates[0]

    @classmethod
    def single_declared_input_setting_names(
        cls,
        module: "ModuleBlock",
        settings: tuple[str | "SettingNameFamily", ...],
    ) -> tuple[str, ...]:
        if len(settings) != 1:
            return ()
        names = cls.artifact_input_names_from_setting(module, settings[0])
        if len(names) != 1:
            return ()
        return names

    @classmethod
    def artifact_contract_outputs(
        cls, builder: "_SymbolTableBuilder", module: "ModuleBlock"
    ) -> tuple[object, ...]:
        return (
            cls.measurement_output_artifact(builder, module),
            *super().artifact_contract_outputs(builder, module),
        )


class SourceQualifiedMeasurementFeatureModule(CellProfilerModule):
    """Trait for modules whose emitted measurement features include source names."""

    @classmethod
    def source_qualified_measurement_feature_types(
        cls,
    ) -> tuple[type[RuntimeMeasurementFeature], ...]:
        """Source-qualified modules qualify all emitted feature families."""
        return cls.measurement_feature_types()


class RelationshipArtifactInputModule(
    CellProfilerModule,
    ArtifactContractModule,
    RelationshipArtifactInputCapability,
):
    """Parent for modules that consume relationship artifacts."""


class RelationshipArtifactOutputModule(
    CellProfilerModule,
    ArtifactContractModule,
    RelationshipArtifactOutputCapability,
):
    """Parent for modules that emit relationship artifacts."""


class SpatialGridArtifactInputModule(
    CellProfilerModule,
    ArtifactContractModule,
    SpatialGridArtifactInputCapability,
):
    """Parent for modules that consume spatial-grid artifacts."""


class SpatialGridArtifactOutputModule(
    CellProfilerModule,
    ArtifactContractModule,
    SpatialGridArtifactOutputCapability,
):
    """Parent for modules that emit spatial-grid artifacts."""


class ObjectLineageTransformContractModule(
    PlaneRuntimeArtifactModule,
    RelationshipArtifactOutputModule,
    MeasurementArtifactOutputModule,
    ObjectArtifactInputModule,
    ObjectArtifactOutputModule,
):
    """Parent for one-object-input modules that emit object lineage."""

    input_objects_setting: ClassVar[str | "SettingNameFamily"]
    output_objects_setting: ClassVar[str | "SettingNameFamily"]

    @classmethod
    def object_input_setting_names(cls) -> tuple[str | "SettingNameFamily", ...]:
        return (cls.input_objects_setting,)

    @classmethod
    def object_output_setting_names(cls) -> tuple[str | "SettingNameFamily", ...]:
        return (cls.output_objects_setting,)

    @classmethod
    def artifact_contract_outputs(
        cls, builder: "_SymbolTableBuilder", module: "ModuleBlock"
    ) -> tuple[object, ...]:
        from openhcs.interop.cellprofiler.setting_names import required_setting_value

        parent_name = required_setting_value(module, cls.input_objects_setting)
        child_name = required_setting_value(module, cls.output_objects_setting)
        return (
            cls.measurement_output_artifact(builder, module),
            cls.parent_child_relationship_output_artifact(
                builder, module, parent_name=parent_name, child_name=child_name
            ),
            *cls.declared_output_artifacts_from_settings(builder, module),
        )


class ImageMeasurementInputModule(
    SourceQualifiedMeasurementFeatureModule,
    MeasurementArtifactOutputModule,
    ImageArtifactInputCapability,
):
    """Parent for measurement modules that consume image measurement inputs."""

    @classmethod
    def image_measurement_setting(cls) -> "SettingNameFamily":
        from openhcs.interop.cellprofiler.setting_names import SettingNameFamily

        return SettingNameFamily(
            "Select images to measure",
            aliases=("Select an image to measure", "Select the image to measure"),
        )

    @classmethod
    def measurement_artifact_inputs(
        cls, builder: "_SymbolTableBuilder", module: "ModuleBlock"
    ) -> tuple[object, ...]:
        return (
            *cls.artifact_inputs_from_setting(
                builder,
                module,
                cls.declared_setting_value(cls.image_measurement_setting),
                ImageArtifactInputCapability,
            ),
            *super().measurement_artifact_inputs(builder, module),
        )

    @classmethod
    def measurement_output_relations(
        cls, builder: "_SymbolTableBuilder", module: "ModuleBlock"
    ) -> tuple[ArtifactSpecRelation, ...]:
        image_names = cls.artifact_input_names_from_setting(
            module,
            cls.declared_setting_value(cls.image_measurement_setting),
        )
        if not image_names:
            return super().measurement_output_relations(builder, module)
        return (
            GroupLineageSourceRelation(
                source=ArtifactSpecRef.input(image_names[0], ImageArtifactType)
            ),
        )


class ObjectMeasurementInputModule(
    MeasurementArtifactOutputModule,
    ObjectLabelArtifactInputCapability,
):
    """Parent for measurement modules that consume object-label measurement inputs."""

    @classmethod
    def object_measurement_setting(cls) -> "SettingNameFamily":
        from openhcs.interop.cellprofiler.setting_names import SettingNameFamily

        return SettingNameFamily(
            "Select object sets to measure",
            aliases=("Select objects to measure", "Select an object to measure"),
        )

    @classmethod
    def measurement_artifact_inputs(
        cls, builder: "_SymbolTableBuilder", module: "ModuleBlock"
    ) -> tuple[object, ...]:
        return (
            *super().measurement_artifact_inputs(builder, module),
            *cls.artifact_inputs_from_setting(
                builder,
                module,
                cls.declared_setting_value(cls.object_measurement_setting),
                ObjectLabelArtifactInputCapability,
            ),
        )

    @classmethod
    def measurement_output_relations(
        cls, builder: "_SymbolTableBuilder", module: "ModuleBlock"
    ) -> tuple[ArtifactSpecRelation, ...]:
        relations = super().measurement_output_relations(builder, module)
        object_names = cls.artifact_input_names_from_setting(
            module,
            cls.declared_setting_value(cls.object_measurement_setting),
        )
        if not object_names:
            return relations
        return (
            *relations,
            GroupLineageSourceRelation(
                source=ArtifactSpecRef.input(object_names[0], ObjectLabelsArtifactType)
            ),
        )


class ScopedMeasurementModule(
    ImageMeasurementInputModule, ObjectMeasurementInputModule
):
    """Module declaration parent for CellProfiler modules with target-scope settings."""

    measurement_scope_setting: ClassVar["SettingNameFamily"]

    @classmethod
    @abstractmethod
    def measurement_target_scope(cls, module: "ModuleBlock") -> Enum:
        """Return the declaration-owned typed measurement target scope."""
        raise NotImplementedError

    @classmethod
    def runtime_measurement_target_scope(
        cls, module: "ModuleBlock"
    ) -> "CellProfilerMeasurementTargetScope":
        """Lower the module-local target-scope enum to the runtime scope enum."""
        from openhcs.interop.cellprofiler.measurement_scope import (
            CellProfilerMeasurementTargetScope,
        )

        target_scope = cls.measurement_target_scope(module)
        member_name = "OBJECT" if target_scope.name == "objects" else target_scope.name
        return CellProfilerMeasurementTargetScope[member_name.upper()]

    @classmethod
    def postprocess_bound_settings(
        cls, module: "ModuleBlock", bound: "BoundModuleSettings"
    ) -> "BoundModuleSettings":
        """Attach the declaration-owned measurement target scope to runtime kwargs."""
        from openhcs.interop.cellprofiler.runtime.binding_authorities import (
            CellProfilerInvocationOverrideKwarg,
        )

        return (
            super()
            .postprocess_bound_settings(module, bound)
            .with_kwargs(
                {
                    CellProfilerInvocationOverrideKwarg.measurement_target_scope: cls.runtime_measurement_target_scope(
                        module
                    )
                }
            )
        )


__all__ = (
    "CellProfilerModule",
    "ModuleSettingsSourceModule",
    "BinderSettingsSourceModule",
    "CellProfilerStructuringElement",
    "DEFAULT_STRUCTURING_ELEMENT_SETTING",
    "STRUCTURING_ELEMENT_SETTING_NAME",
    "StructuringElementSetting",
    "StructuringElementSettingBinding",
    "StructuringElementSettingsModule",
    "structuring_element_bound_kwargs",
    "ScopedMeasurementModule",
    "PerObjectMeasurementExecutionModule",
    "ComposedImageObjectMeasurementExecutionModule",
    "PlaneRuntimeArtifactModule",
    "ObjectMeasurementRowsModule",
)
