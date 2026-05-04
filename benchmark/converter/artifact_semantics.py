"""Generic CellProfiler setting-to-artifact semantic classifiers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar

from metaclass_registry import AutoRegisterMeta

from openhcs.core.artifacts import ArtifactKind
from openhcs.processing.materialization import (
    CsvOptions,
    MaterializationSpec,
    ROIOptions,
    TiffStackOptions,
)

from .parser import ModuleBlock, ModuleSetting
from .settings_binder import normalize_cellprofiler_setting_name


class ArtifactSettingDirection(str, Enum):
    """Whether one setting names a consumed or produced artifact."""

    INPUT = "input"
    OUTPUT = "output"


class ArtifactSettingRole(Enum):
    """Closed semantic roles for CellProfiler artifact-name settings."""

    INPUT_IMAGE = (ArtifactSettingDirection.INPUT, ArtifactKind.IMAGE)
    INPUT_OBJECTS = (ArtifactSettingDirection.INPUT, ArtifactKind.OBJECT_LABELS)
    OUTPUT_IMAGE = (ArtifactSettingDirection.OUTPUT, ArtifactKind.IMAGE)
    OUTPUT_OBJECTS = (
        ArtifactSettingDirection.OUTPUT,
        ArtifactKind.OBJECT_LABELS,
    )
    INPUT_SPATIAL_GRID = (
        ArtifactSettingDirection.INPUT,
        ArtifactKind.SPATIAL_GRID,
    )
    OUTPUT_SPATIAL_GRID = (
        ArtifactSettingDirection.OUTPUT,
        ArtifactKind.SPATIAL_GRID,
    )

    def __init__(
        self,
        direction: ArtifactSettingDirection,
        artifact_kind: ArtifactKind,
    ) -> None:
        self._direction = direction
        self._artifact_kind = artifact_kind

    @property
    def direction(self) -> ArtifactSettingDirection:
        return self._direction

    @property
    def artifact_kind(self) -> ArtifactKind:
        return self._artifact_kind

    @property
    def is_input(self) -> bool:
        return self.direction is ArtifactSettingDirection.INPUT


@dataclass(frozen=True, slots=True)
class ArtifactSettingSymbol:
    """One CellProfiler setting value classified as an artifact symbol."""

    role: ArtifactSettingRole
    name: str
    setting_name: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "name",
            _normalized_nonempty_name(
                self.name,
                "ArtifactSettingSymbol.name",
            ),
        )


@dataclass(frozen=True, slots=True)
class FunctionSpecialOutput:
    """One function-declared auxiliary output projected onto artifact kind."""

    name: str
    kind: ArtifactKind

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "name",
            _normalized_nonempty_name(
                self.name,
                "FunctionSpecialOutput.name",
            ),
        )


class ArtifactSettingClassifier(ABC, metaclass=AutoRegisterMeta):
    """Nominal setting-label classifier for CellProfiler artifact semantics."""

    __registry_key__ = "classifier_name"
    __skip_if_no_key__ = True
    classifier_name: ClassVar[str | None] = None
    priority: ClassVar[int] = 100

    @classmethod
    def role_for(cls, setting: ModuleSetting) -> ArtifactSettingRole | None:
        for classifier_type in sorted(
            cls.__registry__.values(),
            key=lambda candidate: candidate.priority,
        ):
            role = classifier_type().classify(setting)
            if role is not None:
                return role
        return None

    @abstractmethod
    def classify(self, setting: ModuleSetting) -> ArtifactSettingRole | None:
        """Return a role when this classifier owns the setting label."""


class OutputImageSettingClassifier(ArtifactSettingClassifier):
    """Classify output image name settings."""

    classifier_name = "output_image"
    priority = 10

    def classify(self, setting: ModuleSetting) -> ArtifactSettingRole | None:
        name = _normalized_setting(setting.name)
        if name.startswith("name_the_output_image"):
            return ArtifactSettingRole.OUTPUT_IMAGE
        if name.startswith("name_the_image_to_save"):
            return ArtifactSettingRole.OUTPUT_IMAGE
        return None


class OutputObjectsSettingClassifier(ArtifactSettingClassifier):
    """Classify output object-label name settings."""

    classifier_name = "output_objects"
    priority = 20

    def classify(self, setting: ModuleSetting) -> ArtifactSettingRole | None:
        name = _normalized_setting(setting.name)
        if not name.startswith("name_"):
            return None
        tokens = _tokens(name)
        if "object" not in tokens and "objects" not in tokens:
            return None
        if any(
            phrase in name
            for phrase in (
                "masked_objects",
                "output_objects",
                "objects_to_be_identified",
                "primary_objects_to_be_identified",
                "secondary_objects_to_be_identified",
                "tertiary_objects_to_be_identified",
                "new_primary_objects",
            )
        ):
            return ArtifactSettingRole.OUTPUT_OBJECTS
        if name.startswith("name_the_output"):
            return ArtifactSettingRole.OUTPUT_OBJECTS
        return None


class OutputSpatialGridSettingClassifier(ArtifactSettingClassifier):
    """Classify named spatial-grid definitions."""

    classifier_name = "output_spatial_grid"
    priority = 25

    def classify(self, setting: ModuleSetting) -> ArtifactSettingRole | None:
        if _normalized_setting(setting.name) == "name_the_grid":
            return ArtifactSettingRole.OUTPUT_SPATIAL_GRID
        return None


class InputImageSettingClassifier(ArtifactSettingClassifier):
    """Classify source or produced image name inputs."""

    classifier_name = "input_image"
    priority = 30

    def classify(self, setting: ModuleSetting) -> ArtifactSettingRole | None:
        name = _normalized_setting(setting.name)
        tokens = _tokens(name)
        if "image" not in tokens and "images" not in tokens:
            return None
        if not name.startswith("select_"):
            return None
        if _contains_any(
            name,
            (
                "image_type",
                "image_set",
                "rule_criteria",
                "thresholding_method",
            ),
        ):
            return None
        return ArtifactSettingRole.INPUT_IMAGE


class InputObjectsSettingClassifier(ArtifactSettingClassifier):
    """Classify object-label name inputs."""

    classifier_name = "input_objects"
    priority = 40

    def classify(self, setting: ModuleSetting) -> ArtifactSettingRole | None:
        name = _normalized_setting(setting.name)
        if "object" not in _tokens(name) and "objects" not in _tokens(name):
            return None
        if not name.startswith("select_"):
            return None
        if _contains_any(
            name,
            ("location", "method", "module", "measurement", "shape"),
        ):
            return None
        return ArtifactSettingRole.INPUT_OBJECTS


class InputSpatialGridSettingClassifier(ArtifactSettingClassifier):
    """Classify named spatial-grid inputs."""

    classifier_name = "input_spatial_grid"
    priority = 35

    def classify(self, setting: ModuleSetting) -> ArtifactSettingRole | None:
        if _normalized_setting(setting.name) == "select_the_defined_grid":
            return ArtifactSettingRole.INPUT_SPATIAL_GRID
        return None


class SpecialOutputKindClassifier(ABC, metaclass=AutoRegisterMeta):
    """Nominal classifier for function-declared special output specs."""

    __registry_key__ = "classifier_name"
    __skip_if_no_key__ = True
    classifier_name: ClassVar[str | None] = None
    priority: ClassVar[int] = 100

    @classmethod
    def kind_for(cls, spec: object) -> ArtifactKind:
        for classifier_type in sorted(
            cls.__registry__.values(),
            key=lambda candidate: candidate.priority,
        ):
            kind = classifier_type().classify(spec)
            if kind is not None:
                return kind
        raise ValueError(f"Cannot infer artifact kind for special output {spec!r}.")

    @abstractmethod
    def classify(self, spec: object) -> ArtifactKind | None:
        """Return an artifact kind when this classifier owns the output spec."""


class MaterializationOptionSpecialOutputKindClassifier(SpecialOutputKindClassifier):
    """Base for classifiers keyed by MaterializationSpec option type."""

    option_type: ClassVar[type[Any] | None] = None
    output_kind: ClassVar[ArtifactKind | None] = None

    def classify(self, spec: object) -> ArtifactKind | None:
        materialization = _special_output_materialization(spec)
        if materialization is None:
            return None
        option_type = type(self).option_type
        output_kind = type(self).output_kind
        if option_type is None or output_kind is None:
            raise TypeError(
                f"{type(self).__name__} must define option_type and output_kind."
            )
        if any(isinstance(option, option_type) for option in materialization.outputs):
            return output_kind
        return None


class SpatialGridSpecialOutputKindClassifier(SpecialOutputKindClassifier):
    """Classify special outputs that carry grid geometry, not measurement rows."""

    classifier_name = "spatial_grid"
    priority = 5

    def classify(self, spec: object) -> ArtifactKind | None:
        normalized = _normalized_setting(_special_output_name(spec))
        if normalized in {"grid_info", "grid_definition", "spatial_grid"}:
            return ArtifactKind.SPATIAL_GRID
        return None


@dataclass(frozen=True, slots=True)
class MaterializationOptionSpecialOutputKindClassifierSpec:
    """Declarative registration row for materialization-option classifiers."""

    class_name: str
    classifier_name: str
    priority: int
    option_type: type[Any]
    output_kind: ArtifactKind


def _declare_materialization_option_classifier(
    spec: MaterializationOptionSpecialOutputKindClassifierSpec,
) -> None:
    globals()[spec.class_name] = type(
        spec.class_name,
        (MaterializationOptionSpecialOutputKindClassifier,),
        {
            "__module__": __name__,
            "classifier_name": spec.classifier_name,
            "priority": spec.priority,
            "option_type": spec.option_type,
            "output_kind": spec.output_kind,
        },
    )


for _materialization_classifier_spec in (
    MaterializationOptionSpecialOutputKindClassifierSpec(
        class_name="RoiSpecialOutputKindClassifier",
        classifier_name="roi",
        priority=10,
        option_type=ROIOptions,
        output_kind=ArtifactKind.OBJECT_LABELS,
    ),
    MaterializationOptionSpecialOutputKindClassifierSpec(
        class_name="CsvSpecialOutputKindClassifier",
        classifier_name="csv",
        priority=20,
        option_type=CsvOptions,
        output_kind=ArtifactKind.MEASUREMENTS,
    ),
    MaterializationOptionSpecialOutputKindClassifierSpec(
        class_name="TiffSpecialOutputKindClassifier",
        classifier_name="tiff",
        priority=30,
        option_type=TiffStackOptions,
        output_kind=ArtifactKind.IMAGE,
    ),
):
    _declare_materialization_option_classifier(_materialization_classifier_spec)


class NameSpecialOutputKindClassifier(SpecialOutputKindClassifier):
    """Name-based classifier for legacy special_outputs without materialization."""

    classifier_name = "name"
    priority = 40

    def classify(self, spec: object) -> ArtifactKind | None:
        name = _special_output_name(spec)
        normalized = _normalized_setting(name)
        if "label" in normalized or "labels" in normalized:
            return ArtifactKind.OBJECT_LABELS
        if "relationship" in normalized:
            return ArtifactKind.RELATIONSHIPS
        if "image" in normalized:
            return ArtifactKind.IMAGE
        return ArtifactKind.MEASUREMENTS


def artifact_setting_symbols(module: ModuleBlock) -> tuple[ArtifactSettingSymbol, ...]:
    """Return artifact-name settings in .cppipe order."""
    symbols: list[ArtifactSettingSymbol] = []
    for setting in _iter_module_settings(module):
        role = ArtifactSettingClassifier.role_for(setting)
        if role is None:
            continue
        for name in _symbol_names_from_setting(setting):
            symbols.append(
                ArtifactSettingSymbol(
                    role=role,
                    name=name,
                    setting_name=setting.name,
                )
            )
    return tuple(symbols)


def _iter_module_settings(module: ModuleBlock) -> tuple[ModuleSetting, ...]:
    records = module.iter_settings()
    if records:
        return records
    return tuple(
        ModuleSetting(name=name, value=value)
        for name, value in module.settings.items()
    )


def function_special_outputs(module_name: str) -> tuple[FunctionSpecialOutput, ...]:
    """Return function-declared auxiliary outputs with semantic artifact kinds."""
    from openhcs.processing.backends.cellprofiler import require_cellprofiler_function

    raw_outputs = vars(require_cellprofiler_function(module_name)).get(
        "__special_outputs__",
        (),
    )
    if not isinstance(raw_outputs, tuple):
        raise TypeError(
            f"{module_name}.__special_outputs__ must be a tuple, "
            f"got {type(raw_outputs).__name__}."
        )
    return tuple(
        FunctionSpecialOutput(
            name=_special_output_name(spec),
            kind=SpecialOutputKindClassifier.kind_for(spec),
        )
        for spec in raw_outputs
    )


def _special_output_name(spec: object) -> str:
    if isinstance(spec, str):
        return spec
    if isinstance(spec, tuple) and len(spec) == 2 and isinstance(spec[0], str):
        return spec[0]
    raise ValueError(f"Invalid special output declaration: {spec!r}.")


def _special_output_materialization(spec: object) -> MaterializationSpec | None:
    if isinstance(spec, tuple) and len(spec) == 2:
        materialization = spec[1]
        if materialization is None:
            return None
        if not isinstance(materialization, MaterializationSpec):
            raise TypeError(
                "special_outputs materialization must be MaterializationSpec "
                f"or None, got {type(materialization).__name__}."
            )
        return materialization
    return None


def _symbol_names_from_setting(setting: ModuleSetting) -> tuple[str, ...]:
    return tuple(
        value
        for value in (
            part.strip()
            for part in setting.value.split(",")
        )
        if value and not _is_blank_symbol(value)
    )


def _is_blank_symbol(value: str) -> bool:
    return _normalized_setting(value) in {
        "leave_this_black",
        "none",
        "do_not_use",
        "no",
        "not_using",
    }


def _normalized_nonempty_name(value: str, field_name: str) -> str:
    normalized_name = value.strip()
    if not normalized_name:
        raise ValueError(f"{field_name} cannot be empty.")
    return normalized_name


def _normalized_setting(value: str) -> str:
    return normalize_cellprofiler_setting_name(value)


def _tokens(value: str) -> frozenset[str]:
    return frozenset(value.split("_"))


def _contains_any(value: str, fragments: tuple[str, ...]) -> bool:
    return any(fragment in value for fragment in fragments)
