"""
Display configuration factory for creating viewer-specific config dataclasses.

Provides generic infrastructure for creating display configuration dataclasses
with component-specific dimension modes, supporting both Napari and Fiji viewers.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional, Sequence, TypeAlias

from polystore.streaming.viewer_transport import ViewerDisplayConfigABC


DisplayComponentName: TypeAlias = str | Enum
DisplayFieldDefault: TypeAlias = str | int | float | bool | Enum | None
DisplayBaseFields: TypeAlias = dict[
    str,
    tuple[type, DisplayFieldDefault, str],
]
DisplayModeDefaults: TypeAlias = dict[str, Enum]
DisplayMethods: TypeAlias = dict[str, Callable]


class ViewerDisplayConfigObject(ViewerDisplayConfigABC, ABC):
    """Object display config contract emitted by OpenHCS config factories."""

    COMPONENT_ORDER: tuple[str, ...]

    @abstractmethod
    def component_modes(self) -> dict[str, str]:
        """Return component-name to mode-name mapping."""


def component_value(component: DisplayComponentName) -> str:
    if isinstance(component, Enum):
        return str(component.value)
    return str(component).lower()


def display_component_default(
    component: str,
    component_defaults: DisplayModeDefaults,
    default_mode: Enum,
) -> Enum:
    if component in component_defaults:
        return component_defaults[component]
    return default_mode


def create_display_config(
    name: str,
    base_fields: DisplayBaseFields,
    component_mode_enum: type[Enum],
    component_defaults: DisplayModeDefaults,
    virtual_components: Sequence[Enum],
    component_order: Sequence[str],
    default_mode: Enum,
    methods: DisplayMethods,
    docstring: str,
) -> type:
    """
    Generic factory for creating display configuration dataclasses.

    Creates a frozen dataclass with:
    - Base fields (e.g., colormap, variable_size_handling)
    - Component-specific mode fields (e.g., channel_mode, z_index_mode, well_mode)
    - Custom methods (e.g., get_dimension_mode, get_colormap_name)
    - COMPONENT_ORDER class attribute for canonical layer naming order

    Args:
        name: Name of the dataclass to create
        base_fields: Dict mapping field names to (type, default_value) tuples
        component_mode_enum: Enum class for component dimension modes
        component_defaults: Dict mapping component names to default modes
        virtual_components: Extra non-filename components.
        component_order: Canonical order for layer naming (e.g., ['well', 'channel'])
        default_mode: Default mode for components not specified in component_defaults
        methods: Dict mapping method names to method implementations
        docstring: Docstring for the created class

    Returns:
        Dynamically created frozen dataclass

    Example:
        >>> NapariDisplayConfig = create_display_config(
        ...     name='NapariDisplayConfig',
        ...     base_fields={
        ...         'colormap': (NapariColormap, NapariColormap.GRAY),
        ...         'variable_size_handling': (NapariVariableSizeHandling, NapariVariableSizeHandling.SEPARATE_LAYERS)
        ...     },
        ...     component_mode_enum=NapariDimensionMode,
        ...     component_defaults={'channel': NapariDimensionMode.SLICE},
        ...     component_order=['well', 'channel'],
        ...     methods={'get_colormap_name': lambda self: self.colormap.value}
        ... )
    """
    from openhcs.constants import AllComponents

    component_order = tuple(component_order)
    annotations: dict[str, type] = {}
    defaults: dict[str, DisplayFieldDefault] = {}

    for field_name, (field_type, default_value, description) in base_fields.items():
        annotations[field_name] = field_type
        defaults[field_name] = field(
            default=default_value,
            metadata={"description": description},
        )

    for component in (*AllComponents, *virtual_components):
        component_name = component_value(component)
        annotations[f"{component_name}_mode"] = component_mode_enum
        defaults[f"{component_name}_mode"] = field(
            default=display_component_default(
                component_name,
                component_defaults,
                default_mode,
            ),
            metadata={
                "description": (
                    f"Viewer layout mode for the {component_name} axis; choose "
                    f"one of {', '.join(member.name for member in component_mode_enum)}."
                )
            },
        )

    def component_modes(self) -> dict[str, str]:
        modes = {}
        storage = vars(self)
        for component in component_order:
            mode = storage[f"{component}_mode"]
            if mode is None:
                mode = display_component_default(
                    component,
                    component_defaults,
                    default_mode,
                )
            modes[component] = mode.value if isinstance(mode, Enum) else str(mode)
        return modes

    class_attrs = {
        "__annotations__": annotations,
        "component_modes": component_modes,
        "__doc__": docstring,
        "COMPONENT_ORDER": component_order,
    }
    class_attrs.update(defaults)
    class_attrs.update(methods)

    display_config = type(name, (ViewerDisplayConfigObject,), class_attrs)
    display_config = dataclass(frozen=True)(display_config)
    display_config.__module__ = "openhcs.core.config"
    display_config.__qualname__ = name
    return display_config


def create_napari_display_config(
    colormap_enum: type[Enum],
    dimension_mode_enum: type[Enum],
    variable_size_handling_enum: type[Enum],
    visualization_dtype_enum: type[Enum],
    component_order: Sequence[str],
    virtual_components: Optional[type[Enum]] = None,
    virtual_component_defaults: Optional[DisplayModeDefaults] = None,
    default_visualization_dtype: Optional[Enum] = None
) -> type:
    """
    Create NapariDisplayConfig with component-specific fields.

    Args:
        colormap_enum: Enum for colormap options
        dimension_mode_enum: Enum for dimension modes (SLICE/STACK)
        variable_size_handling_enum: Enum for variable size handling
        visualization_dtype_enum: Enum for dtype normalization (UINT8/UINT16/FLOAT32)
        virtual_components: Optional enum for non-filename display components.
        component_order: Canonical order for layer naming
        virtual_component_defaults: Optional dict mapping virtual component names to default modes
        default_visualization_dtype: Default dtype for visualization normalization (defaults to UINT16)

    Returns:
        NapariDisplayConfig dataclass
    """
    def get_dimension_mode(self, component):
        field_name = f"{component_value(component)}_mode"
        mode = vars(self)[field_name]

        if mode is None:
            # Default: all components are STACK (well, channel, site, z_index, timepoint)
            return dimension_mode_enum.STACK

        return mode

    def get_colormap_name(self):
        return self.colormap.value

    from openhcs.constants import AllComponents

    component_defaults = {
        component.value: dimension_mode_enum.STACK
        for component in AllComponents
    }
    if virtual_component_defaults is not None:
        component_defaults.update(virtual_component_defaults)

    # Default visualization dtype to UINT16 (microscopy standard) if not specified
    if default_visualization_dtype is None:
        default_visualization_dtype = visualization_dtype_enum.UINT16

    virtual_component_members: tuple[Enum, ...] = ()
    if virtual_components is not None:
        virtual_component_members = tuple(virtual_components)

    return create_display_config(
        name='NapariDisplayConfig',
        base_fields={
            'colormap': (
                colormap_enum,
                colormap_enum.GRAY,
                "Colormap applied to grayscale image layers in napari.",
            ),
            'variable_size_handling': (
                variable_size_handling_enum,
                variable_size_handling_enum.PAD_TO_MAX,
                "How napari combines images with different spatial dimensions into layers.",
            ),
            'visualization_dtype': (
                visualization_dtype_enum,
                default_visualization_dtype,
                "Target dtype used for contrast-preserving visualization scaling.",
            ),
        },
        component_mode_enum=dimension_mode_enum,
        component_defaults=component_defaults,
        virtual_components=virtual_component_members,
        component_order=component_order,
        default_mode=dimension_mode_enum.STACK,
        methods={
            'get_dimension_mode': get_dimension_mode,
            'get_colormap_name': get_colormap_name,
        },
        docstring="""Configuration for napari display behavior for all OpenHCS components.

        This class is dynamically generated with individual fields for each component dimension.
        Each component has a corresponding {component}_mode field that controls whether
        it's displayed as a slice or stack in napari.

        Includes ALL dimensions (site, channel, z_index, timepoint, well) regardless of
        which dimension is used as the multiprocessing axis.

        visualization_dtype controls dtype normalization for stacking - all images in a stack
        are normalized to this dtype using contrast-preserving scaling (not simple casting).
        Defaults to UINT16 (microscopy standard) for optimal precision preservation.
        """
    )


def create_fiji_display_config(
    lut_enum: type[Enum],
    dimension_mode_enum: type[Enum],
    component_order: Sequence[str],
    virtual_components: Optional[type[Enum]] = None,
    virtual_component_defaults: Optional[DisplayModeDefaults] = None
) -> type:
    """
    Create FijiDisplayConfig with component-specific fields.

    Maps OpenHCS dimensions to ImageJ hyperstack dimensions (C, Z, T).
    Default mapping:
    - well → FRAME (wells become frames)
    - site → FRAME (sites become frames)
    - channel → CHANNEL (channels become channels)
    - z_index → SLICE (z-planes become slices)
    - timepoint → FRAME (timepoints become frames)

    Args:
        lut_enum: Enum for Fiji LUT options
        dimension_mode_enum: Enum for dimension modes (WINDOW/CHANNEL/SLICE/FRAME)
        virtual_components: Optional enum for non-filename display components.
        component_order: Canonical order for layer naming
        virtual_component_defaults: Optional dict mapping virtual component names to default modes

    Returns:
        FijiDisplayConfig dataclass
    """
    def get_dimension_mode(self, component):
        key = component_value(component)
        field_name = f"{key}_mode"
        mode = vars(self)[field_name]

        if mode is None:
            return display_component_default(
                key,
                fiji_component_defaults,
                dimension_mode_enum.CHANNEL,
            )

        return mode

    def get_lut_name(self):
        return self.lut.value

    from openhcs.constants import AllComponents

    component_defaults = {
        component.value: dimension_mode_enum.CHANNEL
        for component in AllComponents
    }
    component_defaults.update(
        {
            AllComponents.WELL.value: dimension_mode_enum.FRAME,
            AllComponents.SITE.value: dimension_mode_enum.FRAME,
            AllComponents.Z_INDEX.value: dimension_mode_enum.SLICE,
            AllComponents.TIMEPOINT.value: dimension_mode_enum.FRAME,
        }
    )
    fiji_component_defaults = dict(component_defaults)
    if virtual_component_defaults is not None:
        component_defaults.update(virtual_component_defaults)

    virtual_component_members: tuple[Enum, ...] = ()
    if virtual_components is not None:
        virtual_component_members = tuple(virtual_components)

    return create_display_config(
        name='FijiDisplayConfig',
        base_fields={
            'lut': (
                lut_enum,
                lut_enum.GRAYS,
                "Fiji lookup table applied when image data is displayed.",
            ),
            'auto_contrast': (
                bool,
                True,
                "Automatically set Fiji display limits from the streamed image data.",
            ),
        },
        component_mode_enum=dimension_mode_enum,
        component_defaults=component_defaults,
        virtual_components=virtual_component_members,
        component_order=component_order,
        default_mode=dimension_mode_enum.CHANNEL,
        methods={
            'get_dimension_mode': get_dimension_mode,
            'get_lut_name': get_lut_name,
        },
        docstring="""Configuration for Fiji display behavior for all OpenHCS components.

        This class is dynamically generated with individual fields for each component dimension.
        Each component has a corresponding {component}_mode field that controls how it maps
        to ImageJ hyperstack dimensions (WINDOW/CHANNEL/SLICE/FRAME).

        Includes ALL dimensions (site, channel, z_index, timepoint, well) regardless of
        which dimension is used as the multiprocessing axis.

        ImageJ hyperstacks have 3 dimensions:
        - Channels (C): Color channels or sites
        - Slices (Z): Z-planes or depth
        - Frames (T): Time points or temporal dimension

        WINDOW mode creates separate windows instead of combining into hyperstack.
        """
    )
