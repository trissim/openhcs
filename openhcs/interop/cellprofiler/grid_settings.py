"""Typed lowering for CellProfiler grid-module variants."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar

from metaclass_registry import AutoRegisterMeta

from openhcs.interop.cellprofiler.runtime import (
    CellProfilerGridCycleScope,
    CellProfilerInvocationOptions,
)

from .parser import ModuleBlock
from .setting_names import is_blank_symbol_name, optional_setting_value
from .settings_binder import SettingsBinder


class DefineGridVariant(str, Enum):
    """Absorbed DefineGrid function variants."""

    MANUAL = "define_grid_manual"
    AUTOMATIC = "define_grid_automatic"

    @classmethod
    def from_module(cls, module: ModuleBlock) -> "DefineGridVariant":
        value = _setting_value(
            module,
            "Select the method to define the grid",
            default="Manual",
        ).lower()
        if "automatic" in value:
            return cls.AUTOMATIC
        if "manual" in value:
            return cls.MANUAL
        raise ValueError(f"Unsupported DefineGrid method: {value!r}.")


class IdentifyObjectsInGridVariant(str, Enum):
    """Absorbed IdentifyObjectsInGrid function variants."""

    GRID_ONLY = "identify_objects_in_grid"
    WITH_GUIDES = "identify_objects_in_grid_with_guides"

    @classmethod
    def from_module(cls, module: ModuleBlock) -> "IdentifyObjectsInGridVariant":
        guiding_objects = _setting_value(
            module,
            "Select the guiding objects",
            default="None",
        )
        if is_blank_symbol_name(guiding_objects):
            return cls.GRID_ONLY
        return cls.WITH_GUIDES


class FunctionNameVariantResolver(ABC, metaclass=AutoRegisterMeta):
    """Registered module-name authority for grid function variant selection."""

    __registry_key__ = "module_name"
    __skip_if_no_key__ = True
    module_name: ClassVar[str | None] = None

    @classmethod
    def for_module_name(cls, module_name: str) -> "FunctionNameVariantResolver":
        resolver_type = cls.__registry__.get(module_name)
        if resolver_type is None:
            raise ValueError(f"Unsupported grid module name: {module_name!r}.")
        return resolver_type()

    @abstractmethod
    def function_name(self, module: ModuleBlock) -> str:
        """Return the absorbed function name selected by one module."""


class DefineGridFunctionNameVariantResolver(FunctionNameVariantResolver):
    """Resolve DefineGrid function variants from typed module settings."""

    module_name = "DefineGrid"

    def function_name(self, module: ModuleBlock) -> str:
        return DefineGridVariant.from_module(module).value


class IdentifyObjectsInGridFunctionNameVariantResolver(FunctionNameVariantResolver):
    """Resolve IdentifyObjectsInGrid function variants from typed module settings."""

    module_name = "IdentifyObjectsInGrid"

    def function_name(self, module: ModuleBlock) -> str:
        return IdentifyObjectsInGridVariant.from_module(module).value


def define_grid_bound_kwargs(
    module: ModuleBlock,
    binder: SettingsBinder,
) -> dict[str, Any]:
    """Return kwargs for the absorbed DefineGrid variant."""
    kwargs = {
        "grid_rows": TypedGridSetting(
            module,
            binder,
            "Number of rows",
            default="8",
        ).value,
        "grid_columns": TypedGridSetting(
            module,
            binder,
            "Number of columns",
            default="12",
        ).value,
        "origin": _grid_origin(
            _setting_value(module, "Location of the first spot", default="Top left")
        ),
        "ordering": _grid_ordering(
            _setting_value(module, "Order of the spots", default="Rows")
        ),
    }
    if DefineGridVariant.from_module(module) is DefineGridVariant.MANUAL:
        first_x, first_y = _coordinate_pair(
            _setting_value(module, "Coordinates of the first cell", default="100,100")
        )
        second_x, second_y = _coordinate_pair(
            _setting_value(module, "Coordinates of the second cell", default="200,200")
        )
        kwargs.update(
            {
                "first_spot_x": first_x,
                "first_spot_y": first_y,
                "first_spot_row": TypedGridSetting(
                    module,
                    binder,
                    "Row number of the first cell",
                    default="1",
                ).value,
                "first_spot_col": TypedGridSetting(
                    module,
                    binder,
                    "Column number of the first cell",
                    default="1",
                ).value,
                "second_spot_x": second_x,
                "second_spot_y": second_y,
                "second_spot_row": TypedGridSetting(
                    module,
                    binder,
                    "Row number of the second cell",
                    default="8",
                ).value,
                "second_spot_col": TypedGridSetting(
                    module,
                    binder,
                    "Column number of the second cell",
                    default="12",
                ).value,
            }
        )
    return kwargs


def define_grid_invocation_options(module: ModuleBlock) -> CellProfilerInvocationOptions:
    """Return typed runtime controls for a DefineGrid invocation."""
    return CellProfilerInvocationOptions(
        grid_cycle_scope=_grid_cycle_scope(
            _setting_value(module, "Define a grid for which cycle?", default="Each cycle")
        )
    )


def identify_objects_in_grid_bound_kwargs(
    module: ModuleBlock,
    binder: SettingsBinder,
) -> dict[str, Any]:
    """Return kwargs for the absorbed IdentifyObjectsInGrid variant."""
    return {
        "shape_choice": _shape_choice(
            _setting_value(
                module,
                "Select object shapes and locations",
                default="Rectangle Forced Location",
            )
        ),
        "diameter_choice": _diameter_choice(
            _setting_value(
                module,
                "Specify the circle diameter automatically?",
                default="Manual",
            )
        ),
        "circle_diameter": TypedGridSetting(
            module,
            binder,
            "Circle diameter",
            default="20",
        ).value,
    }


@dataclass(frozen=True, slots=True)
class TypedGridSetting:
    """Nominal parser request for one typed grid setting."""

    module: ModuleBlock
    binder: SettingsBinder
    setting_name: str
    default: str

    @property
    def value(self) -> Any:
        return self.binder.parse_value(
            self.setting_name,
            _setting_value(self.module, self.setting_name, default=self.default),
        )


@dataclass(frozen=True, slots=True)
class FragmentMatchedLiteral:
    """Nominal owner for CP grid literal matching by normalized word fragments."""

    value: str
    fragments_to_literal: dict[tuple[str, ...], str]

    @property
    def literal(self) -> str:
        normalized = self.value.strip().lower()
        for fragments, literal in self.fragments_to_literal.items():
            if all(fragment in normalized for fragment in fragments):
                return literal
        raise ValueError(f"Unsupported grid setting value: {self.value!r}.")


def _setting_value(
    module: ModuleBlock,
    setting_name: str,
    *,
    default: str,
) -> str:
    return optional_setting_value(module, setting_name) or default


def _coordinate_pair(value: str) -> tuple[int, int]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        raise ValueError(f"Grid coordinate must be x,y, got {value!r}.")
    return int(float(parts[0])), int(float(parts[1]))


def _grid_origin(value: str) -> str:
    return FragmentMatchedLiteral(
        value=value,
        fragments_to_literal={
            ("top", "left"): "top_left",
            ("bottom", "left"): "bottom_left",
            ("top", "right"): "top_right",
            ("bottom", "right"): "bottom_right",
        },
    ).literal


def _grid_ordering(value: str) -> str:
    return FragmentMatchedLiteral(
        value=value,
        fragments_to_literal={
            ("row",): "rows",
            ("column",): "columns",
        },
    ).literal


def _grid_cycle_scope(value: str) -> CellProfilerGridCycleScope:
    return CellProfilerGridCycleScope(
        FragmentMatchedLiteral(
            value=value,
            fragments_to_literal={
                ("once",): "once",
                ("each",): "each_cycle",
            },
        ).literal
    )


def _shape_choice(value: str) -> str:
    return FragmentMatchedLiteral(
        value=value,
        fragments_to_literal={
            ("rectangle",): "rectangle_forced_location",
            ("circle", "forced"): "circle_forced_location",
            ("circle", "natural"): "circle_natural_location",
            ("natural",): "natural_shape_and_location",
        },
    ).literal


def _diameter_choice(value: str) -> str:
    normalized = value.strip().lower()
    if "automatic" in normalized or normalized in {"yes", "true"}:
        return "automatic"
    if "manual" in normalized or normalized in {"no", "false"}:
        return "manual"
    raise ValueError(f"Unsupported grid diameter choice: {value!r}.")
