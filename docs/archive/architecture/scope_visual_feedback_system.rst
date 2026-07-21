Scope Visual Feedback System
============================

**CIELAB perceptual colors and scope-based visual styling.**

*Modules: openhcs.pyqt_gui.widgets.shared.scope_color_*, openhcs.pyqt_gui.widgets.shared.scope_visual_config*

Overview
--------

The scope visual feedback system provides color-coded visual feedback based on
hierarchical scope IDs. Each orchestrator (plate) gets a distinct base color,
and steps within that orchestrator inherit the base color with tint/pattern variations.

Color Strategy Architecture
---------------------------

Colors are generated via pluggable strategies:

.. list-table::
   :header-rows: 1

   * - Strategy
     - Description
   * - ``IndexBasedStrategy``
     - Primary palette (12 colors) assigned in discovery order
   * - ``MD5HashStrategy``
     - Fallback using distinctipy palette (50 colors)
   * - ``ManualColorStrategy``
     - User-selected colors with persistence

``ScopeColorService`` (singleton) manages strategy selection and caching.

CIELAB Perceptual Colors
------------------------

Border tints use CIELAB color space for perceptual uniformity. L* values
(lightness) are equidistant in perceptual space:

.. code-block:: python

   # Perceptually equidistant L* levels (0-100 scale)
   BORDER_LAB_LIGHTNESS = (30.0, 55.0, 80.0)  # dark, mid, light

The ``tint_color_perceptual()`` function converts RGB → LAB, adjusts L* to
target level while preserving a*/b* (chromatic channels), then converts back.

Scope Hierarchy
---------------

.. code-block:: text

   Orchestrator (plate_path)
   ├── Gets BASE color from palette (by discovery order)
   └── Steps (plate_path::functionstep_N)
       └── Inherit BASE color, vary by tint index + border pattern

Border patterns for steps: 3 tints × 4 patterns = 12 visual combinations.

ScopeColorScheme
----------------

Computed color scheme for a scope:

.. code-block:: python

   @dataclass
   class ScopeColorScheme:
       scope_id: Optional[str]
       hue: int
       orchestrator_item_bg_rgb: tuple[int, int, int]
       orchestrator_item_border_rgb: tuple[int, int, int]
       step_window_border_rgb: tuple[int, int, int]
       step_item_bg_rgb: Optional[tuple[int, int, int]]
       step_border_width: int
       step_border_layers: list  # [(width, tint_idx, pattern), ...]
       base_color_rgb: tuple[int, int, int]

Usage
-----

.. code-block:: python

   from openhcs.pyqt_gui.widgets.shared.scope_color_utils import get_scope_color_scheme

   # Get color scheme for a scope
   scheme = get_scope_color_scheme("plate_path::functionstep_0")

   # Access colors
   bg_color = scheme.to_qcolor_orchestrator_bg()
   border_color = scheme.to_qcolor_step_window_border()

   # Or with explicit step index (for list item position)
   scheme = get_scope_color_scheme("plate_path::functionstep_0", step_index=5)

ScopeVisualConfig
-----------------

Static configuration for visual parameters:

.. code-block:: python

   @dataclass
   class ScopeVisualConfig:
       ORCHESTRATOR_ITEM_BG_OPACITY: float = 0.10
       STEP_ITEM_BG_OPACITY: float = 0.10
       GROUPBOX_BG_OPACITY: float = 0.05
       STEP_WINDOW_BORDER_WIDTH_PX: int = 4
       FLASH_DURATION_MS: int = 300
       # ...

Access via ``get_scope_visual_config()`` singleton.

Integration with SccopedBorderMixin
-----------------------------------

Widgets use ``ScopedBorderMixin`` to apply scope-based border styling:

.. code-block:: python

   from openhcs.pyqt_gui.widgets.shared.scoped_border_mixin import ScopedBorderMixin

   class StepEditorDialog(QDialog, ScopedBorderMixin):
       def __init__(self, scope_id: str):
           super().__init__()
           self._apply_scope_border(scope_id)

See Also
--------

- :doc:`flash_animation_system` - Flash animations use scope colors
- :doc:`abstract_manager_widget` - List items use scope-based backgrounds
