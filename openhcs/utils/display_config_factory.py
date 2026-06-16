"""
Display configuration factory for creating viewer-specific config dataclasses.

Provides generic infrastructure for creating display configuration dataclasses
with component-specific dimension modes, supporting both Napari and Fiji viewers.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Callable, Optional, Type


STANDARD_COMPONENTS = ("well", "site", "channel", "z_index", "timepoint")


def component_value(component: object) -> str:
    if isinstance(component, Enum):
        return str(component.value)
    return str(component).lower()


def standard_component_defaults(
    *,
    well: Any,
    site: Any,
    channel: Any,
    z_index: Any,
    timepoint: Any,
) -> dict[str, Any]:
    return dict(
        zip(
            STANDARD_COMPONENTS,
            (well, site, channel, z_index, timepoint),
            strict=True,
        )
    )


def create_display_config(
    name: str,
    base_fields: Dict[str, tuple[Type, Any]],
    component_mode_enum: Type[Enum],
    component_defaults: Optional[Dict[str, Any]] = None,
    virtual_components: Optional[Type[Enum]] = None,
    component_order: Optional[list[str]] = None,
    default_mode: Optional[Any] = None,
    methods: Optional[Dict[str, Callable]] = None,
    docstring: Optional[str] = None
) -> Type:
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
        component_defaults: Optional dict mapping component names to default modes
        virtual_components: Optional enum for non-filename components.
        component_order: Canonical order for layer naming (e.g., ['well', 'channel'])
        default_mode: Default mode for components not specified in component_defaults (required)
        methods: Optional dict mapping method names to method implementations
        docstring: Optional docstring for the created class

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

    # Use AllComponents instead of VariableComponents so display configs include ALL dimensions
    # (including the multiprocessing axis). Display configuration should be independent of
    # multiprocessing axis choice - users should be able to control how wells/any dimension
    # are displayed regardless of which dimension is used for parallelization.
    all_components = list(AllComponents)
    component_defaults = component_defaults or {}

    # Require explicit default_mode - no magic fallbacks
    if default_mode is None:
        raise ValueError("default_mode is required - specify the default mode for unspecified components")

    annotations = {}
    defaults = {}

    for field_name, (field_type, default_value) in base_fields.items():
        annotations[field_name] = field_type
        defaults[field_name] = default_value

    # Generate mode fields for filename components
    for component in all_components:
        field_name = f"{component.value}_mode"
        annotations[field_name] = component_mode_enum
        defaults[field_name] = component_defaults.get(component.value, default_mode)

    # Generate mode fields for virtual components
    if virtual_components:
        for component in virtual_components:
            field_name = f"{component.value}_mode"
            annotations[field_name] = component_mode_enum
            defaults[field_name] = component_defaults.get(component.value, default_mode)

    def __init__(self, **kwargs):
        for field_name, default_value in defaults.items():
            if field_name not in kwargs:
                kwargs[field_name] = default_value
        for field_name, value in kwargs.items():
            object.__setattr__(self, field_name, value)

    def component_modes(self) -> dict[str, str]:
        modes = {}
        storage = vars(self)
        for component in component_order or ():
            mode = storage[f"{component}_mode"]
            if mode is not None:
                modes[component] = mode.value
        return modes

    class_attrs = {
        '__annotations__': annotations,
        '__init__': __init__,
        'component_modes': component_modes,
        '__doc__': docstring or f"Display configuration for {name}",
    }

    for field_name, default_value in defaults.items():
        class_attrs[field_name] = default_value

    # Add component order as class attribute
    if component_order:
        class_attrs['COMPONENT_ORDER'] = component_order

    if methods:
        class_attrs.update(methods)

    DisplayConfig = type(name, (), class_attrs)
    DisplayConfig = dataclass(frozen=True)(DisplayConfig)

    # Set proper module and qualname for pickling support
    DisplayConfig.__module__ = 'openhcs.core.config'
    DisplayConfig.__qualname__ = name

    return DisplayConfig


def create_napari_display_config(
    colormap_enum: Type[Enum],
    dimension_mode_enum: Type[Enum],
    variable_size_handling_enum: Type[Enum],
    visualization_dtype_enum: Type[Enum],
    virtual_components: Optional[Type[Enum]] = None,
    component_order: Optional[list[str]] = None,
    virtual_component_defaults: Optional[Dict[str, Any]] = None,
    default_visualization_dtype: Optional[Any] = None
) -> Type:
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
        mode = vars(self).get(field_name)

        if mode is None:
            # Default: all components are STACK (well, channel, site, z_index, timepoint)
            return dimension_mode_enum.STACK

        return mode

    def get_colormap_name(self):
        return self.colormap.value

    # Merge component defaults - all components default to STACK
    component_defaults = standard_component_defaults(
        well=dimension_mode_enum.STACK,
        site=dimension_mode_enum.STACK,
        channel=dimension_mode_enum.STACK,
        z_index=dimension_mode_enum.STACK,
        timepoint=dimension_mode_enum.STACK,
    )
    if virtual_component_defaults:
        component_defaults.update(virtual_component_defaults)

    # Default visualization dtype to UINT16 (microscopy standard) if not specified
    if default_visualization_dtype is None:
        default_visualization_dtype = visualization_dtype_enum.UINT16

    return create_display_config(
        name='NapariDisplayConfig',
        base_fields={
            'colormap': (colormap_enum, colormap_enum.GRAY),
            'variable_size_handling': (variable_size_handling_enum, variable_size_handling_enum.PAD_TO_MAX),
            'visualization_dtype': (visualization_dtype_enum, default_visualization_dtype),
        },
        component_mode_enum=dimension_mode_enum,
        component_defaults=component_defaults,
        virtual_components=virtual_components,
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
    lut_enum: Type[Enum],
    dimension_mode_enum: Type[Enum],
    virtual_components: Optional[Type[Enum]] = None,
    component_order: Optional[list[str]] = None,
    virtual_component_defaults: Optional[Dict[str, Any]] = None
) -> Type:
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
        mode = vars(self).get(field_name)

        if mode is None:
            return fiji_component_defaults.get(key, dimension_mode_enum.CHANNEL)

        return mode

    def get_lut_name(self):
        return self.lut.value

    # Merge component defaults
    component_defaults = standard_component_defaults(
        well=dimension_mode_enum.FRAME,
        site=dimension_mode_enum.FRAME,
        channel=dimension_mode_enum.CHANNEL,
        z_index=dimension_mode_enum.SLICE,
        timepoint=dimension_mode_enum.FRAME,
    )
    fiji_component_defaults = dict(component_defaults)
    if virtual_component_defaults:
        component_defaults.update(virtual_component_defaults)

    return create_display_config(
        name='FijiDisplayConfig',
        base_fields={
            'lut': (lut_enum, lut_enum.GRAYS),
            'auto_contrast': (bool, True),
        },
        component_mode_enum=dimension_mode_enum,
        component_defaults=component_defaults,
        virtual_components=virtual_components,
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
