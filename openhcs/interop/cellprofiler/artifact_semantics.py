"""CellProfiler setting-to-artifact semantic classifiers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

from metaclass_registry import AutoRegisterMeta

from openhcs.core.artifacts import ArtifactKind
from openhcs.core.special_outputs import (
    SpecialOutputKindClassifier,
    special_output_name,
)
from openhcs.interop.cellprofiler.parser import ModuleBlock, ModuleSetting
from openhcs.interop.cellprofiler.settings_binder import (
    normalize_cellprofiler_setting_name,
)


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
                "combined_object_set",
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
            (
                "how_to_handle",
                "location",
                "method",
                "module",
                "measurement",
                "shape",
            ),
        ):
            return None
        return ArtifactSettingRole.INPUT_OBJECTS


class BareObjectsInputSettingClassifier(ArtifactSettingClassifier):
    """Classify CellProfiler LabelSubscriber settings named simply ``Objects``."""

    classifier_name = "bare_input_objects"
    priority = 41

    def classify(self, setting: ModuleSetting) -> ArtifactSettingRole | None:
        if _normalized_setting(setting.name) == "objects":
            return ArtifactSettingRole.INPUT_OBJECTS
        return None


class InputSpatialGridSettingClassifier(ArtifactSettingClassifier):
    """Classify named spatial-grid inputs."""

    classifier_name = "input_spatial_grid"
    priority = 35

    def classify(self, setting: ModuleSetting) -> ArtifactSettingRole | None:
        if _normalized_setting(setting.name) == "select_the_defined_grid":
            return ArtifactSettingRole.INPUT_SPATIAL_GRID
        return None


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
            name=special_output_name(spec),
            kind=SpecialOutputKindClassifier.kind_for(spec),
        )
        for spec in raw_outputs
    )


def _iter_module_settings(module: ModuleBlock) -> tuple[ModuleSetting, ...]:
    records = module.iter_settings()
    if records:
        return records
    return tuple(
        ModuleSetting(name=name, value=value)
        for name, value in module.settings.items()
    )


def _symbol_names_from_setting(setting: ModuleSetting) -> tuple[str, ...]:
    return tuple(
        value
        for value in (part.strip() for part in setting.value.split(","))
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
