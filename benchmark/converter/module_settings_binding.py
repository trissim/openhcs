"""Module-level settings-to-kwargs translation for generated CellProfiler steps."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

from metaclass_registry import AutoRegisterMeta

from benchmark.cellprofiler_library import canonical_module_name
from benchmark.cellprofiler_library.functions._enum import _coerce_function_enum
from benchmark.cellprofiler_library.functions.correctilluminationapply import (
    IlluminationCorrectionMethod,
)
from benchmark.cellprofiler_library.functions.relateobjects import DistanceMethod
from benchmark.cellprofiler_library.functions.measureobjectintensitydistribution import (
    CenterChoice,
    ZernikeMode,
)
from openhcs.interop.cellprofiler.measurement_scope import (
    CELLPROFILER_MEASUREMENT_TARGET_SCOPE_KWARG,
)

from .align_settings import align_bound_kwargs
from .area_occupied_settings import area_occupied_bound_kwargs
from .calculate_math_settings import calculate_math_bound_kwargs
from .classify_objects_settings import classify_objects_bound_kwargs
from .color_to_gray_settings import color_to_gray_bound_kwargs
from .crop_settings import crop_bound_kwargs
from .display_data_settings import display_data_on_image_bound_kwargs
from .enhance_edges_settings import ENHANCE_EDGES_SETTINGS
from .expand_or_shrink_settings import expand_or_shrink_bound_kwargs
from .filter_objects_settings import filter_objects_bound_kwargs
from .grid_settings import (
    define_grid_bound_kwargs,
    identify_objects_in_grid_bound_kwargs,
)
from .gray_to_color_settings import (
    GRAY_TO_COLOR_CMYK_IMAGE_SETTINGS,
    GRAY_TO_COLOR_CMYK_WEIGHT_SETTINGS,
    GRAY_TO_COLOR_RGB_IMAGE_SETTINGS,
    GRAY_TO_COLOR_RGB_WEIGHT_SETTINGS,
    GRAY_TO_COLOR_RESCALE_SETTING,
    GrayToColorScheme,
    coerce_gray_to_color_scheme,
    gray_to_color_rescale_default,
    gray_to_color_stack_channels,
    is_blank_gray_to_color_source,
)
from .illumination_settings import (
    CORRECT_ILLUMINATION_APPLY_SETTINGS,
    CORRECT_ILLUMINATION_CALCULATE_SETTINGS,
)
from .image_math_settings import image_math_bound_kwargs
from .mask_objects_settings import MASK_OBJECTS_SETTINGS
from .module_function_resolution import measurement_target_scope
from .overlay_outlines_settings import overlay_outlines_bound_kwargs
from .parser import ModuleBlock
from .settings_binder import (
    parse_cellprofiler_bool,
    parse_cellprofiler_float,
    parse_cellprofiler_int,
    SettingToKeywordBinding,
    SettingsBinder,
    normalize_cellprofiler_setting_name,
)
from openhcs.interop.cellprofiler.setting_names import (
    SettingNameFamily,
    optional_setting_value,
    setting_names,
    setting_values,
)
from .smooth_settings import SMOOTH_SETTINGS
from .straighten_worms_settings import straighten_worms_bound_kwargs
from .structuring_element_settings import structuring_element_bound_kwargs
from .untangle_worms_settings import untangle_worms_bound_kwargs
from .unmix_colors_settings import unmix_colors_bound_kwargs


@dataclass(frozen=True, slots=True)
class BoundModuleSettings:
    """Typed module-setting translation result."""

    kwargs: Mapping[str, Any]
    unmapped_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kwargs", dict(self.kwargs))
        object.__setattr__(self, "unmapped_kwargs", dict(self.unmapped_kwargs))


class ModuleSettingsBindingStrategy(ABC, metaclass=AutoRegisterMeta):
    """Nominal family for converting one module's settings into function kwargs."""

    __registry_key__ = "module_name"
    __skip_if_no_key__ = True
    module_name: ClassVar[str | None] = None

    @classmethod
    def for_module(cls, module_name: str) -> "ModuleSettingsBindingStrategy":
        strategy_type = cls.__registry__.get(
            canonical_module_name(module_name),
            GenericModuleSettingsBindingStrategy,
        )
        return strategy_type()

    @abstractmethod
    def bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        """Bind one parsed module into generated function kwargs."""


class GenericModuleSettingsBindingStrategy(ModuleSettingsBindingStrategy):
    """Default docstring-mapped module-setting binder."""

    def bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        return _translate_bound_kwargs(
            binder.bind(module.settings),
            param_mapping,
        )


_CELLPROFILER_THRESHOLD_SETTINGS: Mapping[str, str] = {
    "Threshold strategy": "threshold_scope",
    "Thresholding method": "threshold_method",
    "Threshold smoothing scale": "threshold_smoothing_scale",
    "Threshold correction factor": "threshold_correction_factor",
    "Two-class or three-class thresholding?": "otsu_class_count",
    "Assign pixels in the middle intensity class to the foreground or the background?": (
        "assign_middle_to_foreground"
    ),
    "Log transform before thresholding?": "log_transform",
    "Size of adaptive window": "adaptive_window_size",
    "Lower outlier fraction": "lower_outlier_fraction",
    "Upper outlier fraction": "upper_outlier_fraction",
    "Averaging method": "averaging_method",
    "Variance method": "variance_method",
    "# of deviations": "number_of_deviations",
    "Manual threshold": "manual_threshold",
}
_IGNORED_CELLPROFILER_THRESHOLD_SETTINGS: tuple[str, ...] = (
    "Threshold setting version",
    "Select the measurement to threshold with",
)
_CELLPROFILER_THRESHOLD_SETTING_VERSION = "Threshold setting version"
_CELLPROFILER_THRESHOLD_METHOD_SETTING = "Thresholding method"
_CELLPROFILER_OTSU_METHOD = "otsu"
_CELLPROFILER_THREE_CLASS_OTSU = "three classes"
_CELLPROFILER_TWO_CLASS_OTSU = "two classes"
_LEGACY_CELLPROFILER_THRESHOLD_METHOD_NAMES: Mapping[str, str] = {
    "robustbackground": "Robust Background",
    "minimum cross entropy": "Minimum Cross-Entropy",
}
_FLOAT_CELLPROFILER_THRESHOLD_SETTINGS: frozenset[str] = frozenset(
    {
        "Threshold smoothing scale",
        "Threshold correction factor",
        "Lower outlier fraction",
        "Upper outlier fraction",
        "# of deviations",
        "Manual threshold",
    }
)
_INT_CELLPROFILER_THRESHOLD_SETTINGS: frozenset[str] = frozenset(
    {
        "Size of adaptive window",
    }
)
_BOOL_CELLPROFILER_THRESHOLD_SETTINGS: frozenset[str] = frozenset(
    {
        "Log transform before thresholding?",
    }
)


def _parse_cellprofiler_threshold_setting(
    binder: SettingsBinder,
    setting_name: str,
    value: str,
) -> Any:
    """Parse threshold settings by semantic field, not generic literal shape."""
    if setting_name in _FLOAT_CELLPROFILER_THRESHOLD_SETTINGS:
        return parse_cellprofiler_float(value)
    if setting_name in _INT_CELLPROFILER_THRESHOLD_SETTINGS:
        return parse_cellprofiler_int(value)
    if setting_name in _BOOL_CELLPROFILER_THRESHOLD_SETTINGS:
        return parse_cellprofiler_bool(value)
    return binder.parse_value(setting_name, value)


def _last_optional_setting_value(
    module: ModuleBlock,
    setting_name: str | SettingNameFamily,
) -> str | None:
    """Return the last ordered value for legacy scalar settings."""
    values = setting_values(module, setting_name)
    if values:
        return values[-1]
    return None


def _active_threshold_setting_value(
    module: ModuleBlock,
    setting_name: str,
) -> str | None:
    """Return the active ordered value for threshold settings.

    Legacy threshold version 10 pipelines can carry duplicate method rows.
    The public pipeline semantics for the cached CP-compatible examples follow
    the first method row, not an inferred migration rewrite.
    """
    values = setting_values(module, setting_name)
    if not values:
        return None
    legacy_version = _legacy_cellprofiler_threshold_version(module)
    if (
        setting_name == _CELLPROFILER_THRESHOLD_METHOD_SETTING
        and len(values) > 1
        and legacy_version is not None
        and legacy_version <= 10
    ):
        return values[0]
    return values[-1]


def _cellprofiler_threshold_setting_token(value: Any) -> str:
    """Return a stable comparison token for parsed CellProfiler settings."""
    if hasattr(value, "value") and isinstance(value.value, str):
        value = value.value
    return " ".join(str(value).strip().lower().replace("-", " ").split())


def _legacy_cellprofiler_threshold_version(module: ModuleBlock) -> int | None:
    value = optional_setting_value(module, _CELLPROFILER_THRESHOLD_SETTING_VERSION)
    if value is None:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def _legacy_cellprofiler_log_transform_default(
    module: ModuleBlock,
    kwargs: Mapping[str, Any],
) -> bool | None:
    """Return CP's synthesized log-transform default for old threshold blocks.

    Native CellProfiler inserts ``Log transform before thresholding?`` while
    upgrading threshold setting version 10 to version 11. The inserted value is
    true only for three-class Otsu thresholding.
    """
    version = _legacy_cellprofiler_threshold_version(module)
    if version is None or version > 10:
        return None
    threshold_method = _cellprofiler_threshold_setting_token(
        kwargs.get("threshold_method", "")
    )
    otsu_class_count = _cellprofiler_threshold_setting_token(
        kwargs.get("otsu_class_count", "")
    )
    return (
        threshold_method == _CELLPROFILER_OTSU_METHOD
        and otsu_class_count == _CELLPROFILER_THREE_CLASS_OTSU
    )


def _upgrade_legacy_cellprofiler_threshold_kwargs(
    module: ModuleBlock,
    kwargs: dict[str, Any],
) -> None:
    version = _legacy_cellprofiler_threshold_version(module)
    if version is None or version > 10:
        return

    threshold_method = kwargs.get("threshold_method")
    if threshold_method is not None:
        method_token = _cellprofiler_threshold_setting_token(threshold_method)
        kwargs["threshold_method"] = _LEGACY_CELLPROFILER_THRESHOLD_METHOD_NAMES.get(
            method_token,
            threshold_method,
        )

    if "log_transform" not in kwargs:
        log_transform_default = _legacy_cellprofiler_log_transform_default(
            module,
            kwargs,
        )
        if log_transform_default is not None:
            kwargs["log_transform"] = log_transform_default


def _bind_cellprofiler_threshold_settings(
    *,
    module: ModuleBlock,
    binder: SettingsBinder,
    kwargs: dict[str, Any],
    unmapped_kwargs: dict[str, Any],
    include_advanced_setting: bool,
) -> None:
    if include_advanced_setting:
        value = _last_optional_setting_value(module, "Use advanced settings?")
        if value is not None:
            kwargs["use_advanced_settings"] = binder.parse_value(
                "Use advanced settings?",
                value,
            )
        unmapped_kwargs.pop(
            normalize_cellprofiler_setting_name("Use advanced settings?"),
            None,
        )

    for setting_name, parameter_name in _CELLPROFILER_THRESHOLD_SETTINGS.items():
        value = _active_threshold_setting_value(module, setting_name)
        if value is not None:
            kwargs[parameter_name] = _parse_cellprofiler_threshold_setting(
                binder,
                setting_name,
                value,
        )
        unmapped_kwargs.pop(normalize_cellprofiler_setting_name(setting_name), None)

    _upgrade_legacy_cellprofiler_threshold_kwargs(module, kwargs)

    bounds = _last_optional_setting_value(module, "Lower and upper bounds on threshold")
    if bounds is not None:
        parsed_bounds = binder.parse_value(
            "Lower and upper bounds on threshold",
            bounds,
        )
        if not isinstance(parsed_bounds, tuple) or len(parsed_bounds) != 2:
            raise ValueError(
                f"{module.name} threshold bounds must contain two values, "
                f"got {bounds!r}."
            )
        kwargs["threshold_min"], kwargs["threshold_max"] = parsed_bounds
    unmapped_kwargs.pop(
        normalize_cellprofiler_setting_name("Lower and upper bounds on threshold"),
        None,
    )

    for setting_name in _IGNORED_CELLPROFILER_THRESHOLD_SETTINGS:
        unmapped_kwargs.pop(normalize_cellprofiler_setting_name(setting_name), None)


class IdentifyPrimaryObjectsModuleSettingsBindingStrategy(
    GenericModuleSettingsBindingStrategy
):
    """Bind primary-object threshold settings using ordered CellProfiler settings."""

    module_name = "IdentifyPrimaryObjects"

    def bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        bound = super().bind(module, binder=binder, param_mapping=param_mapping)
        kwargs = dict(bound.kwargs)
        unmapped_kwargs = dict(bound.unmapped_kwargs)
        _bind_cellprofiler_threshold_settings(
            module=module,
            binder=binder,
            kwargs=kwargs,
            unmapped_kwargs=unmapped_kwargs,
            include_advanced_setting=True,
        )

        return BoundModuleSettings(kwargs, unmapped_kwargs)


class EnhanceOrSuppressFeaturesModuleSettingsBindingStrategy(
    GenericModuleSettingsBindingStrategy
):
    """Bind EnhanceOrSuppressFeatures settings to native library semantics."""

    module_name = "EnhanceOrSuppressFeatures"
    explicit_settings: ClassVar[Mapping[str, str]] = {
        "Select the operation": "method",
        "Feature type": "enhance_method",
        "Smoothing scale": "smoothing_value",
        "Shear angle": "dic_angle",
        "Decay": "dic_decay",
        "Enhancement method": "neurite_method",
        "Speed and accuracy": "speckle_accuracy",
    }
    ignored_settings: ClassVar[tuple[str, ...]] = (
        "Select the input image",
        "Name the output image",
    )

    def bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        bound = super().bind(module, binder=binder, param_mapping=param_mapping)
        kwargs = dict(bound.kwargs)
        unmapped_kwargs = dict(bound.unmapped_kwargs)

        for setting_name, parameter_name in type(self).explicit_settings.items():
            value = optional_setting_value(module, setting_name)
            if value is not None:
                kwargs[parameter_name] = binder.parse_value(setting_name, value)
            unmapped_kwargs.pop(normalize_cellprofiler_setting_name(setting_name), None)

        hole_sizes = optional_setting_value(module, "Range of hole sizes")
        if hole_sizes is not None:
            parsed_hole_sizes = binder.parse_value("Range of hole sizes", hole_sizes)
            if (
                not isinstance(parsed_hole_sizes, tuple)
                or len(parsed_hole_sizes) != 2
            ):
                raise ValueError(
                    f"{module.name} hole size range must contain two values, "
                    f"got {hole_sizes!r}."
                )
            kwargs["dark_hole_radius_min"], kwargs["dark_hole_radius_max"] = (
                parsed_hole_sizes
            )
        unmapped_kwargs.pop(
            normalize_cellprofiler_setting_name("Range of hole sizes"),
            None,
        )

        feature_size = optional_setting_value(module, "Feature size")
        if feature_size is not None:
            kwargs["radius"] = binder.parse_value("Feature size", feature_size) / 2
        unmapped_kwargs.pop(
            normalize_cellprofiler_setting_name("Feature size"),
            None,
        )

        for setting_name in type(self).ignored_settings:
            unmapped_kwargs.pop(normalize_cellprofiler_setting_name(setting_name), None)

        return BoundModuleSettings(kwargs, unmapped_kwargs)


class IdentifySecondaryObjectsModuleSettingsBindingStrategy(
    GenericModuleSettingsBindingStrategy
):
    """Bind secondary-object method and shared threshold settings."""

    module_name = "IdentifySecondaryObjects"

    explicit_settings: ClassVar[Mapping[str, str]] = {
        "Select the method to identify the secondary objects": "method",
        "Number of pixels by which to expand the primary objects": (
            "distance_to_dilate"
        ),
        "Regularization factor": "regularization_factor",
        "Fill holes in identified objects?": "fill_holes",
        "Discard secondary objects touching the border of the image?": (
            "discard_edge_objects"
        ),
    }
    ignored_settings: ClassVar[tuple[str, ...]] = (
        "Select the input objects",
        "Name the objects to be identified",
        "Select the input image",
        "Discard the associated primary objects?",
        "Name the new primary objects",
    )

    def bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        bound = super().bind(module, binder=binder, param_mapping=param_mapping)
        kwargs = dict(bound.kwargs)
        unmapped_kwargs = dict(bound.unmapped_kwargs)

        for setting_name, parameter_name in type(self).explicit_settings.items():
            value = optional_setting_value(module, setting_name)
            if value is not None:
                kwargs[parameter_name] = binder.parse_value(setting_name, value)
            unmapped_kwargs.pop(normalize_cellprofiler_setting_name(setting_name), None)

        _bind_cellprofiler_threshold_settings(
            module=module,
            binder=binder,
            kwargs=kwargs,
            unmapped_kwargs=unmapped_kwargs,
            include_advanced_setting=False,
        )

        for setting_name in type(self).ignored_settings:
            unmapped_kwargs.pop(normalize_cellprofiler_setting_name(setting_name), None)

        return BoundModuleSettings(kwargs, unmapped_kwargs)


class ScopedMeasurementModuleSettingsBindingStrategy(
    GenericModuleSettingsBindingStrategy
):
    """Bind generic measurement-scope intent alongside normal kwargs."""

    scope_setting_name: ClassVar[SettingNameFamily | None] = None
    default_scope_value: ClassVar[str | None] = None

    def bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        bound = super().bind(module, binder=binder, param_mapping=param_mapping)
        kwargs = dict(bound.kwargs)
        unmapped_kwargs = dict(bound.unmapped_kwargs)
        kwargs[CELLPROFILER_MEASUREMENT_TARGET_SCOPE_KWARG] = (
            measurement_target_scope(
                module,
                setting=_required_class_attr(
                    type(self).scope_setting_name,
                    "scope_setting_name",
                ),
                default=_required_class_attr(
                    type(self).default_scope_value,
                    "default_scope_value",
                ),
            ).value
        )
        for setting_name in setting_names(
            _required_class_attr(type(self).scope_setting_name, "scope_setting_name")
        ):
            unmapped_kwargs.pop(normalize_cellprofiler_setting_name(setting_name), None)
        return BoundModuleSettings(kwargs, unmapped_kwargs)


class MeasureTextureModuleSettingsBindingStrategy(
    ScopedMeasurementModuleSettingsBindingStrategy
):
    """Bind MeasureTexture's image/object target scope."""

    module_name = "MeasureTexture"
    scope_setting_name = SettingNameFamily(
        "Measure images or objects?",
        aliases=("Measure whole images or objects?",),
    )
    default_scope_value = "Images"

    def bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        bound = super().bind(module, binder=binder, param_mapping=param_mapping)
        kwargs = dict(bound.kwargs)
        unmapped_kwargs = dict(bound.unmapped_kwargs)

        texture_scales = setting_values(module, "Texture scale to measure")
        if texture_scales:
            parsed_scales = tuple(
                parse_cellprofiler_int(value)
                for value in texture_scales
            )
            kwargs["scale"] = (
                parsed_scales[0] if len(parsed_scales) == 1 else parsed_scales
            )
            unmapped_kwargs.pop(
                normalize_cellprofiler_setting_name("Texture scale to measure"),
                None,
            )

        gray_levels = optional_setting_value(
            module,
            "Enter how many gray levels to measure the texture at",
        )
        if gray_levels is not None:
            kwargs["gray_levels"] = parse_cellprofiler_int(gray_levels)
            unmapped_kwargs.pop(
                normalize_cellprofiler_setting_name(
                    "Enter how many gray levels to measure the texture at"
                ),
                None,
            )

        return BoundModuleSettings(kwargs, unmapped_kwargs)


class MeasureObjectSizeShapeModuleSettingsBindingStrategy(
    GenericModuleSettingsBindingStrategy
):
    """Bind MeasureObjectSizeShape semantic toggles and routing-only settings."""

    module_name = "MeasureObjectSizeShape"
    explicit_settings: ClassVar[Mapping[str, str]] = {
        "Calculate the Zernike features?": "calculate_zernikes",
    }
    ignored_settings: ClassVar[tuple[str, ...]] = (
        "Select objects to measure",
    )

    def bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        bound = super().bind(module, binder=binder, param_mapping=param_mapping)
        kwargs = dict(bound.kwargs)
        unmapped_kwargs = dict(bound.unmapped_kwargs)

        for setting_name, parameter_name in type(self).explicit_settings.items():
            value = optional_setting_value(module, setting_name)
            if value is not None:
                kwargs[parameter_name] = parse_cellprofiler_bool(value)
            unmapped_kwargs.pop(normalize_cellprofiler_setting_name(setting_name), None)

        for setting_name in type(self).ignored_settings:
            unmapped_kwargs.pop(normalize_cellprofiler_setting_name(setting_name), None)

        return BoundModuleSettings(kwargs, unmapped_kwargs)


class MeasureObjectIntensityDistributionModuleSettingsBindingStrategy(
    GenericModuleSettingsBindingStrategy
):
    """Bind radial-distribution settings not named like function parameters."""

    module_name = "MeasureObjectIntensityDistribution"
    zernike_setting_name: ClassVar[str] = "Calculate intensity Zernikes?"
    zernike_degree_setting_name: ClassVar[str | SettingNameFamily] = SettingNameFamily(
        "Maximum Zernike moment",
        aliases=("Maximum zernike moment",),
    )
    scalar_settings: ClassVar[Mapping[str, tuple[str, Any]]] = {
        "Scale the bins?": ("wants_scaled", parse_cellprofiler_bool),
        "Number of bins": ("bin_count", parse_cellprofiler_int),
        "Maximum radius": ("maximum_radius", parse_cellprofiler_int),
    }

    def bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        bound = super().bind(module, binder=binder, param_mapping=param_mapping)
        kwargs = dict(bound.kwargs)
        unmapped_kwargs = dict(bound.unmapped_kwargs)

        zernike_value = optional_setting_value(module, type(self).zernike_setting_name)
        if zernike_value is not None:
            kwargs["wants_zernikes"] = _parse_zernike_mode_setting(zernike_value)
            unmapped_kwargs.pop(
                normalize_cellprofiler_setting_name(type(self).zernike_setting_name),
                None,
            )

        zernike_degree = optional_setting_value(
            module,
            type(self).zernike_degree_setting_name,
        )
        if zernike_degree is not None:
            kwargs["zernike_degree"] = parse_cellprofiler_int(zernike_degree)
            unmapped_kwargs.pop(
                normalize_cellprofiler_setting_name(
                    setting_names(type(self).zernike_degree_setting_name)[0]
                ),
                None,
            )

        for setting_name, (parameter_name, parse) in type(self).scalar_settings.items():
            value = optional_setting_value(module, setting_name)
            if value is None:
                continue
            kwargs[parameter_name] = parse(value)
            unmapped_kwargs.pop(normalize_cellprofiler_setting_name(setting_name), None)

        center_choice = optional_setting_value(module, "Object to use as center?")
        if center_choice is not None:
            kwargs["center_choice"] = _parse_radial_center_choice_setting(center_choice)
            unmapped_kwargs.pop(
                normalize_cellprofiler_setting_name("Object to use as center?"),
                None,
            )

        return BoundModuleSettings(kwargs, unmapped_kwargs)


def _parse_zernike_mode_setting(value: str) -> str:
    """Return the absorbed-function Zernike mode literal for a CP setting."""
    normalized = normalize_cellprofiler_setting_name(value)
    if normalized == "magnitudes_only":
        return ZernikeMode.MAGNITUDES.value
    return _coerce_function_enum(ZernikeMode, value).value


def _parse_radial_center_choice_setting(value: str) -> str:
    """Return the absorbed-function center-choice literal for a CP setting."""
    normalized = normalize_cellprofiler_setting_name(value)
    if normalized in {"these_objects", "self"}:
        return CenterChoice.SELF.value
    if normalized in {"centers_of_other_objects", "centers_of_other"}:
        return CenterChoice.CENTERS_OF_OTHER.value
    if normalized in {"edges_of_other_objects", "edges_of_other"}:
        return CenterChoice.EDGES_OF_OTHER.value
    return _coerce_function_enum(CenterChoice, value).value


class MeasureGranularityModuleSettingsBindingStrategy(
    GenericModuleSettingsBindingStrategy
):
    """Bind granularity spectrum settings shared across image/object rows."""

    module_name = "MeasureGranularity"
    scalar_settings: ClassVar[Mapping[str, tuple[str, Any]]] = {
        "Subsampling factor for granularity measurements": (
            "subsample_size",
            parse_cellprofiler_float,
        ),
        "Subsampling factor for background reduction": (
            "background_subsample_size",
            parse_cellprofiler_float,
        ),
        "Radius of structuring element": (
            "element_radius",
            parse_cellprofiler_int,
        ),
        "Range of the granular spectrum": (
            "spectrum_length",
            parse_cellprofiler_int,
        ),
    }

    def bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        bound = super().bind(module, binder=binder, param_mapping=param_mapping)
        kwargs = dict(bound.kwargs)
        unmapped_kwargs = dict(bound.unmapped_kwargs)

        for setting_name, (parameter_name, parse) in type(self).scalar_settings.items():
            values = setting_values(module, setting_name)
            if not values:
                continue
            parsed_values = tuple(parse(value) for value in values)
            first_value = parsed_values[0]
            if any(value != first_value for value in parsed_values[1:]):
                raise ValueError(
                    f"Module {module.name}({module.module_num}) has per-row "
                    f"{setting_name!r} values {parsed_values!r}; OpenHCS "
                    "currently binds one granularity setting set per module."
                )
            kwargs[parameter_name] = first_value
            unmapped_kwargs.pop(normalize_cellprofiler_setting_name(setting_name), None)

        return BoundModuleSettings(kwargs, unmapped_kwargs)


class MeasureColocalizationModuleSettingsBindingStrategy(
    ScopedMeasurementModuleSettingsBindingStrategy
):
    """Bind MeasureColocalization's image/object target scope."""

    module_name = "MeasureColocalization"
    scope_setting_name = SettingNameFamily("Select where to measure correlation")
    default_scope_value = "Across entire image"
    metric_settings: ClassVar[Mapping[str, str]] = {
        "Set threshold as percentage of maximum intensity for the images": (
            "threshold_percent"
        ),
        "Calculate correlation and slope metrics?": "do_correlation",
        "Calculate the Manders coefficients?": "do_manders",
        "Calculate the Rank Weighted Colocalization coefficients?": "do_rwc",
        "Calculate the Overlap coefficients?": "do_overlap",
        "Calculate the Manders coefficients using Costes auto threshold?": (
            "do_costes"
        ),
    }
    metric_flags: ClassVar[tuple[str, ...]] = (
        "do_correlation",
        "do_manders",
        "do_rwc",
        "do_overlap",
        "do_costes",
    )

    def bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        bound = super().bind(module, binder=binder, param_mapping=param_mapping)
        kwargs = dict(bound.kwargs)
        unmapped_kwargs = dict(bound.unmapped_kwargs)

        for setting_name, parameter_name in type(self).metric_settings.items():
            value = optional_setting_value(module, setting_name)
            if value is not None:
                kwargs[parameter_name] = binder.parse_value(setting_name, value)
            unmapped_kwargs.pop(normalize_cellprofiler_setting_name(setting_name), None)

        run_all_value = optional_setting_value(module, "Run all metrics?")
        if run_all_value is not None:
            if _parse_colocalization_run_all_metrics(run_all_value):
                kwargs.update({flag: True for flag in type(self).metric_flags})
            unmapped_kwargs.pop(
                normalize_cellprofiler_setting_name("Run all metrics?"),
                None,
            )

        costes_method = optional_setting_value(module, "Method for Costes thresholding")
        if costes_method is not None:
            kwargs["costes_method"] = _parse_colocalization_costes_method(
                costes_method
            )
            unmapped_kwargs.pop(
                normalize_cellprofiler_setting_name("Method for Costes thresholding"),
                None,
            )

        return BoundModuleSettings(kwargs, unmapped_kwargs)


class DeclarativeModuleSettingsBindingStrategy(ModuleSettingsBindingStrategy):
    """Bind modules described by explicit setting-to-kwarg declarations."""

    setting_bindings: ClassVar[tuple[SettingToKeywordBinding, ...]] = ()

    def bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        del param_mapping
        return BoundModuleSettings(
            binder.bind_declared(module, type(self).setting_bindings)
        )


class GrayToColorModuleSettingsBindingStrategy(ModuleSettingsBindingStrategy):
    """Scheme-aware binder for GrayToColor's closed family of input layouts."""

    module_name = "GrayToColor"

    def bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        del param_mapping
        scheme = coerce_gray_to_color_scheme(
            module.get_setting("Select a color scheme", GrayToColorScheme.RGB.value)
        )
        return GrayToColorSchemeBindingStrategy.for_scheme(scheme).bind(
            module,
            binder=binder,
        )


class UnmixColorsModuleSettingsBindingStrategy(ModuleSettingsBindingStrategy):
    """Bind UnmixColors repeated output rows into one multi-output call."""

    module_name = "UnmixColors"

    def bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        del binder, param_mapping
        return BoundModuleSettings(unmix_colors_bound_kwargs(module))


class ColorToGrayModuleSettingsBindingStrategy(ModuleSettingsBindingStrategy):
    """Bind ColorToGray's mode-dependent channel plan."""

    module_name = "ColorToGray"

    def bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        del param_mapping
        return BoundModuleSettings(color_to_gray_bound_kwargs(module, binder))


class MeasureImageAreaOccupiedModuleSettingsBindingStrategy(
    ModuleSettingsBindingStrategy
):
    """Bind ordered area-occupied rows into one generic multi-row call."""

    module_name = "MeasureImageAreaOccupiedBinary"

    def bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        del binder, param_mapping
        return BoundModuleSettings(area_occupied_bound_kwargs(module))


class AlignModuleSettingsBindingStrategy(ModuleSettingsBindingStrategy):
    """Bind legacy Align settings into the absorbed registration function."""

    module_name = "Align"

    def bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        del binder, param_mapping
        return BoundModuleSettings(align_bound_kwargs(module))


class OverlayOutlinesModuleSettingsBindingStrategy(ModuleSettingsBindingStrategy):
    """Bind ordered OverlayOutlines rows into one generic overlay call."""

    module_name = "OverlayOutlines"

    def bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        del binder, param_mapping
        return BoundModuleSettings(overlay_outlines_bound_kwargs(module))


class TileModuleSettingsBindingStrategy(ModuleSettingsBindingStrategy):
    """Bind legacy Tile assembly modes into explicit montage geometry."""

    module_name = "Tile"

    def bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        del binder, param_mapping
        assembly_method = module.get_setting(
            "Tile assembly method",
            "Within cycles",
        ).strip()
        normalized_method = normalize_cellprofiler_setting_name(assembly_method)
        if normalized_method != "within_cycles":
            raise NotImplementedError(
                "Tile assembly method is not supported by the converter: "
                f"{assembly_method!r}"
            )
        return BoundModuleSettings(
            {
                "rows": 1,
                "columns": 1,
                "place_first": "top_left",
                "tile_style": "row",
                "meander": False,
                "auto_rows": False,
                "auto_columns": True,
            }
        )


class TrackObjectsModuleSettingsBindingStrategy(ModuleSettingsBindingStrategy):
    """Bind TrackObjects settings that define tracking identity and feature names."""

    module_name = "TrackObjects"

    def bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        generic = GenericModuleSettingsBindingStrategy().bind(
            module,
            binder=binder,
            param_mapping=param_mapping,
        )
        method = module.get_setting("Choose a tracking method", "Overlap").strip()
        normalized_method = normalize_cellprofiler_setting_name(method)
        if normalized_method not in {"overlap", "distance"}:
            raise NotImplementedError(
                "TrackObjects tracking method is not supported by the converter: "
                f"{method!r}"
            )
        object_name = module.get_setting("Select the objects to track", "Objects").strip()
        if not object_name:
            raise ValueError("TrackObjects requires a non-empty tracked object name.")
        pixel_radius = parse_cellprofiler_int(
            module.get_setting("Maximum pixel distance to consider matches", "50")
        )
        return BoundModuleSettings(
            {
                **dict(generic.kwargs),
                "object_name": object_name,
                "tracking_method": normalized_method,
                "pixel_radius": pixel_radius,
            },
            generic.unmapped_kwargs,
        )


class ImageMathModuleSettingsBindingStrategy(ModuleSettingsBindingStrategy):
    """Bind ImageMath operation and arithmetic modifiers."""

    module_name = "ImageMath"

    def bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        del param_mapping
        return BoundModuleSettings(image_math_bound_kwargs(module, binder))


class MeasureObjectNeighborsModuleSettingsBindingStrategy(
    ModuleSettingsBindingStrategy
):
    """Bind retained neighbor-image settings alongside topology measurements."""

    module_name = "MeasureObjectNeighbors"

    def bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        bound = GenericModuleSettingsBindingStrategy().bind(
            module,
            binder=binder,
            param_mapping=param_mapping,
        )
        colormaps = module.get_setting_values("Select colormap")
        kwargs = dict(bound.kwargs)
        kwargs.update(
            {
                "retain_neighbor_count_image": parse_cellprofiler_bool(
                    module.get_setting(
                        "Retain the image of objects colored by numbers of neighbors?",
                        "No",
                    )
                ),
                "neighbor_count_colormap": colormaps[0] if colormaps else "Default",
                "retain_percent_touching_image": parse_cellprofiler_bool(
                    module.get_setting(
                        "Retain the image of objects colored by percent of touching pixels?",
                        "No",
                    )
                ),
                "percent_touching_colormap": (
                    colormaps[1]
                    if len(colormaps) > 1
                    else colormaps[0] if colormaps else "Default"
                ),
            }
        )
        unmapped = {
            name: value
            for name, value in bound.unmapped_kwargs.items()
            if name
            not in {
                "retain_the_image_of_objects_colored_by_numbers_of_neighbors",
                "retain_the_image_of_objects_colored_by_percent_of_touching_pixels",
                "name_the_output_image",
                "select_colormap",
            }
        }
        return BoundModuleSettings(kwargs, unmapped)


class FilterObjectsModuleSettingsBindingStrategy(ModuleSettingsBindingStrategy):
    """Bind FilterObjects rows into one generic multi-output object filter."""

    module_name = "FilterObjects"

    def bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        del binder, param_mapping
        return BoundModuleSettings(filter_objects_bound_kwargs(module))


class RelateObjectsModuleSettingsBindingStrategy(GenericModuleSettingsBindingStrategy):
    """Bind RelateObjects distance settings instead of using expensive defaults."""

    module_name = "RelateObjects"
    distance_setting_name: ClassVar[str] = "Calculate child-parent distances?"

    def bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        bound = super().bind(module, binder=binder, param_mapping=param_mapping)
        kwargs = dict(bound.kwargs)
        unmapped_kwargs = dict(bound.unmapped_kwargs)
        value = optional_setting_value(module, type(self).distance_setting_name)
        if value is not None:
            kwargs["calculate_distances"] = _coerce_function_enum(
                DistanceMethod,
                value,
            ).value
            unmapped_kwargs.pop(
                normalize_cellprofiler_setting_name(type(self).distance_setting_name),
                None,
            )
        return BoundModuleSettings(kwargs, unmapped_kwargs)


class DisplayDataOnImageModuleSettingsBindingStrategy(ModuleSettingsBindingStrategy):
    """Bind DisplayDataOnImage measurement-selection settings."""

    module_name = "DisplayDataOnImage"

    def bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        del binder, param_mapping
        return BoundModuleSettings(display_data_on_image_bound_kwargs(module))


class CalculateMathModuleSettingsBindingStrategy(ModuleSettingsBindingStrategy):
    """Bind CalculateMath operand and arithmetic settings."""

    module_name = "CalculateMath"

    def bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        del param_mapping
        return BoundModuleSettings(calculate_math_bound_kwargs(module, binder))


class ClassifyObjectsModuleSettingsBindingStrategy(ModuleSettingsBindingStrategy):
    """Bind ClassifyObjects settings into absorbed classification kwargs."""

    module_name = "ClassifyObjectsSingleMeasurement"

    def bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        del param_mapping
        return BoundModuleSettings(classify_objects_bound_kwargs(module, binder))


class CropModuleSettingsBindingStrategy(ModuleSettingsBindingStrategy):
    """Bind Crop's coordinate/mask mode settings into absorbed Crop kwargs."""

    module_name = "Crop"

    def bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        del param_mapping
        return BoundModuleSettings(crop_bound_kwargs(module, binder))


class CorrectIlluminationCalculateModuleSettingsBindingStrategy(
    DeclarativeModuleSettingsBindingStrategy
):
    """Bind illumination-function calculation settings without bool/enum loss."""

    module_name = "CorrectIlluminationCalculate"
    setting_bindings = CORRECT_ILLUMINATION_CALCULATE_SETTINGS


class CorrectIlluminationApplyModuleSettingsBindingStrategy(
    DeclarativeModuleSettingsBindingStrategy
):
    """Bind illumination application settings for image+function pairs."""

    module_name = "CorrectIlluminationApply"
    setting_bindings = CORRECT_ILLUMINATION_APPLY_SETTINGS

    def bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        del param_mapping
        kwargs = binder.bind_declared(module, type(self).setting_bindings)
        _bind_repeated_correct_illumination_apply_settings(module, kwargs)
        return BoundModuleSettings(kwargs)


def _bind_repeated_correct_illumination_apply_settings(
    module: ModuleBlock,
    kwargs: dict[str, Any],
) -> None:
    """Preserve per-image-pair settings when one CP module has several pairs."""
    method_values = setting_values(
        module,
        "Select how the illumination function is applied",
    )
    if len(method_values) > 1:
        kwargs["method"] = tuple(
            _coerce_function_enum(IlluminationCorrectionMethod, value).value
            for value in method_values
        )
    _bind_repeated_bool_setting(
        module,
        "Set output image values less than 0 equal to 0?",
        "truncate_low",
        kwargs,
    )
    _bind_repeated_bool_setting(
        module,
        "Set output image values greater than 1 equal to 1?",
        "truncate_high",
        kwargs,
    )


def _bind_repeated_bool_setting(
    module: ModuleBlock,
    setting_name: str,
    parameter_name: str,
    kwargs: dict[str, Any],
) -> None:
    values = setting_values(module, setting_name)
    if len(values) > 1:
        kwargs[parameter_name] = tuple(parse_cellprofiler_bool(value) for value in values)


class SmoothModuleSettingsBindingStrategy(DeclarativeModuleSettingsBindingStrategy):
    """Bind Smooth's filter and scale settings into absorbed image smoothing."""

    module_name = "Smooth"
    setting_bindings = SMOOTH_SETTINGS


class EnhanceEdgesModuleSettingsBindingStrategy(
    DeclarativeModuleSettingsBindingStrategy
):
    """Bind EnhanceEdges method and threshold settings into edge filtering."""

    module_name = "EnhanceEdges"
    setting_bindings = ENHANCE_EDGES_SETTINGS


class ExpandOrShrinkObjectsModuleSettingsBindingStrategy(ModuleSettingsBindingStrategy):
    """Bind ExpandOrShrinkObjects operation settings into morphology kwargs."""

    module_name = "ExpandOrShrinkObjects"

    def bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        del param_mapping
        return BoundModuleSettings(expand_or_shrink_bound_kwargs(module, binder))


class MaskObjectsModuleSettingsBindingStrategy(
    DeclarativeModuleSettingsBindingStrategy
):
    """Bind object-mask policy settings into absorbed MaskObjects kwargs."""

    module_name = "MaskObjects"
    setting_bindings = MASK_OBJECTS_SETTINGS


class StructuringElementModuleSettingsBindingStrategy(ModuleSettingsBindingStrategy):
    """Bind shared morphology structuring-element settings."""

    def bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        del param_mapping
        return BoundModuleSettings(structuring_element_bound_kwargs(module, binder))


STRUCTURING_ELEMENT_MODULE_NAMES = (
    "Opening",
    "Closing",
    "ErodeImage",
    "DilateImage",
)


def _declare_module_settings_binding_strategy(
    module_name: str,
    base: type[ModuleSettingsBindingStrategy],
) -> type[ModuleSettingsBindingStrategy]:
    class_name = f"{module_name}ModuleSettingsBindingStrategy"
    return type(base)(
        class_name,
        (base,),
        {
            "__module__": __name__,
            "__qualname__": class_name,
            "module_name": module_name,
        },
    )


globals().update(
    {
        f"{module_name}ModuleSettingsBindingStrategy": (
            _declare_module_settings_binding_strategy(
                module_name,
                StructuringElementModuleSettingsBindingStrategy,
            )
        )
        for module_name in STRUCTURING_ELEMENT_MODULE_NAMES
    }
)


class DefineGridModuleSettingsBindingStrategy(ModuleSettingsBindingStrategy):
    """Bind DefineGrid settings into absorbed grid-definition kwargs."""

    module_name = "DefineGridManual"

    def bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        del param_mapping
        return BoundModuleSettings(define_grid_bound_kwargs(module, binder))


class IdentifyObjectsInGridModuleSettingsBindingStrategy(
    ModuleSettingsBindingStrategy
):
    """Bind IdentifyObjectsInGrid settings into absorbed grid-object kwargs."""

    module_name = "IdentifyObjectsInGrid"

    def bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        del param_mapping
        return BoundModuleSettings(
            identify_objects_in_grid_bound_kwargs(module, binder)
        )


class UntangleWormsModuleSettingsBindingStrategy(ModuleSettingsBindingStrategy):
    """Bind UntangleWorms output-mode settings into typed runtime kwargs."""

    module_name = "UntangleWorms"

    def bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        del binder, param_mapping
        return BoundModuleSettings(untangle_worms_bound_kwargs(module))


class StraightenWormsModuleSettingsBindingStrategy(ModuleSettingsBindingStrategy):
    """Bind StraightenWorms geometry and measurement settings."""

    module_name = "StraightenWorms"

    def bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
        param_mapping: Mapping[str, Any],
    ) -> BoundModuleSettings:
        del binder, param_mapping
        return BoundModuleSettings(straighten_worms_bound_kwargs(module))


class GrayToColorSchemeBindingStrategy(ABC, metaclass=AutoRegisterMeta):
    """Closed nominal family for GrayToColor scheme-specific kwarg lowering."""

    __registry_key__ = "scheme_literal"
    __skip_if_no_key__ = True
    scheme_literal: ClassVar[str | None] = None

    @classmethod
    def for_scheme(
        cls,
        scheme: GrayToColorScheme,
    ) -> "GrayToColorSchemeBindingStrategy":
        strategy_type = cls.__registry__.get(scheme.value)
        if strategy_type is None:
            raise ValueError(f"Unsupported GrayToColor scheme: {scheme.value!r}")
        return strategy_type()

    @abstractmethod
    def bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
    ) -> BoundModuleSettings:
        """Bind one GrayToColor module for a specific color scheme."""


class _GrayToColorIndexedSchemeBindingStrategy(GrayToColorSchemeBindingStrategy):
    """Base class for schemes whose payload is indexed into ordered image planes."""

    image_settings: ClassVar[tuple[tuple[str, str], ...]] = ()
    weight_settings: ClassVar[tuple[tuple[str, str], ...]] = ()

    def bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
    ) -> BoundModuleSettings:
        kwargs: dict[str, Any] = {
            "color_scheme": type(self).scheme_literal,
            "rescale_intensity": _typed_setting_value(
                binder,
                GRAY_TO_COLOR_RESCALE_SETTING,
                module.get_setting(
                    GRAY_TO_COLOR_RESCALE_SETTING,
                    gray_to_color_rescale_default(module),
                ),
            ),
        }
        channel_index = 0
        for setting_name, parameter_name in type(self).image_settings:
            kwargs[parameter_name] = -1
            image_name = module.get_setting(setting_name, "").strip()
            if is_blank_gray_to_color_source(image_name):
                continue
            kwargs[parameter_name] = channel_index
            channel_index += 1
        for setting_name, parameter_name in type(self).weight_settings:
            kwargs[parameter_name] = _typed_setting_value(
                binder,
                setting_name,
                module.get_setting(setting_name, "1.0"),
            )
        return BoundModuleSettings(kwargs)


class GrayToColorRgbBindingStrategy(_GrayToColorIndexedSchemeBindingStrategy):
    scheme_literal = GrayToColorScheme.RGB.value
    image_settings = tuple(
        zip(
            GRAY_TO_COLOR_RGB_IMAGE_SETTINGS,
            ("red_channel", "green_channel", "blue_channel"),
            strict=True,
        )
    )
    weight_settings = tuple(
        zip(
            GRAY_TO_COLOR_RGB_WEIGHT_SETTINGS,
            ("red_weight", "green_weight", "blue_weight"),
            strict=True,
        )
    )


class GrayToColorCmykBindingStrategy(_GrayToColorIndexedSchemeBindingStrategy):
    scheme_literal = GrayToColorScheme.CMYK.value
    image_settings = tuple(
        zip(
            GRAY_TO_COLOR_CMYK_IMAGE_SETTINGS,
            ("cyan_channel", "magenta_channel", "yellow_channel", "gray_channel"),
            strict=True,
        )
    )
    weight_settings = tuple(
        zip(
            GRAY_TO_COLOR_CMYK_WEIGHT_SETTINGS,
            ("cyan_weight", "magenta_weight", "yellow_weight", "gray_weight"),
            strict=True,
        )
    )


class _GrayToColorStackFamilyBindingStrategy(GrayToColorSchemeBindingStrategy):
    """Base class for Stack/Composite repeated per-image settings."""

    def bind(
        self,
        module: ModuleBlock,
        *,
        binder: SettingsBinder,
    ) -> BoundModuleSettings:
        channels = gray_to_color_stack_channels(module)
        kwargs: dict[str, Any] = {
            "color_scheme": type(self).scheme_literal,
            "rescale_intensity": _typed_setting_value(
                binder,
                GRAY_TO_COLOR_RESCALE_SETTING,
                module.get_setting(
                    GRAY_TO_COLOR_RESCALE_SETTING,
                    gray_to_color_rescale_default(module),
                ),
            ),
            "channel_weights": tuple(
                float(_typed_setting_value(binder, "Weight", channel.weight))
                for channel in channels
            ),
        }
        if type(self).scheme_literal == GrayToColorScheme.COMPOSITE.value:
            kwargs["channel_colors"] = tuple(
                channel.color for channel in channels
            )
        return BoundModuleSettings(kwargs)


class GrayToColorStackBindingStrategy(_GrayToColorStackFamilyBindingStrategy):
    scheme_literal = GrayToColorScheme.STACK.value


class GrayToColorCompositeBindingStrategy(_GrayToColorStackFamilyBindingStrategy):
    scheme_literal = GrayToColorScheme.COMPOSITE.value


def _translate_bound_kwargs(
    kwargs: Mapping[str, Any],
    param_mapping: Mapping[str, Any],
) -> BoundModuleSettings:
    translated_kwargs: dict[str, Any] = {}
    unmapped_kwargs: dict[str, Any] = {}

    for cp_setting, value in kwargs.items():
        if cp_setting in param_mapping:
            py_param = param_mapping[cp_setting]
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
            continue
        unmapped_kwargs[cp_setting] = value

    return BoundModuleSettings(
        kwargs=translated_kwargs,
        unmapped_kwargs=unmapped_kwargs,
    )


def _parse_colocalization_run_all_metrics(value: str) -> bool:
    """Parse modern and legacy MeasureColocalization all-metric settings."""
    try:
        return parse_cellprofiler_bool(value)
    except ValueError:
        # Older pipelines may store legacy accuracy choices such as "Accurate"
        # in this slot. Those choices do not disable any metric family.
        return bool(value.strip())


def _parse_colocalization_costes_method(value: str) -> str:
    """Normalize CellProfiler Costes method choices to absorbed-function values."""
    normalized = value.strip().lower()
    if normalized not in {"faster", "fast", "accurate"}:
        raise ValueError(f"Unsupported Costes thresholding method: {value!r}.")
    return normalized


def _typed_setting_value(
    binder: SettingsBinder,
    key: str,
    value: str,
) -> Any:
    return binder.parse_value(key, value)


def _required_class_attr[T](value: T | None, name: str) -> T:
    if value is None:
        raise TypeError(f"Measurement settings strategy must define {name}.")
    return value
