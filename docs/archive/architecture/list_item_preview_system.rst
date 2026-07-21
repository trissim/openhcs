List Item Preview System
========================

The list item preview system provides declarative, per-field styling for list items in manager widgets (PlateManager, PipelineEditor). It displays configuration previews with visual indicators for dirty state and signature differences.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
--------

List items in OpenHCS show rich previews of configuration state:

.. code-block:: text

   ▶ MyPlate (W:8 | Seq:C,Z)           ← Inline format
     /path/to/plate
     └─ wf:[3] | mat:ZARR | configs=[NAP, FIJI]

**Visual semantics:**

- **Underline** = field differs from signature default (explicitly set)
- **Asterisk (*)** = resolved value differs from saved (dirty/unsaved)
- Name inherits styling if ANY field is dirty/sig-diff

Declarative Configuration
-------------------------

Each widget declares its format using ``ListItemFormat``:

.. code-block:: python

   from openhcs.pyqt_gui.widgets.shared.abstract_manager_widget import ListItemFormat

   class PlateManagerWidget(AbstractManagerWidget):
       LIST_ITEM_FORMAT = ListItemFormat(
           first_line=(),                    # Fields after name on line 1
           preview_line=(                    # Fields on └─ preview line
               'num_workers',
               'vfs_config.materialization_backend',
               'path_planning_config.well_filter',
           ),
           detail_line_field='path',         # Field for detail line
           show_config_indicators=True,      # Show NAP/FIJI/MAT configs
           formatters={                      # Custom formatters (optional)
               'my_field': lambda v: f"custom:{v}" if v else None,
           },
       )

Field abbreviations are declared on config classes via ``@global_pipeline_config``:

.. code-block:: python

   @global_pipeline_config(field_abbreviations={'well_filter': 'wf', 'output_dir_suffix': 'out'})
   @dataclass(frozen=True)
   class PathPlanningConfig:
       well_filter: Optional[str] = None
       output_dir_suffix: str = "_openhcs"

ListItemFormat Fields
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Field
     - Default
     - Description
   * - ``first_line``
     - ``()``
     - Tuple of field paths shown after name on line 1
   * - ``preview_line``
     - ``()``
     - Tuple of field paths shown on └─ preview line
   * - ``detail_line_field``
     - ``None``
     - Single field path for detail line (e.g., ``'path'``)
   * - ``show_config_indicators``
     - ``True``
     - Whether to show config indicators (NAP, FIJI, MAT)
   * - ``formatters``
     - ``{}``
     - Dict mapping field path to formatter function

Visual Layout
-------------

Multiline Format
~~~~~~~~~~~~~~~~

.. code-block:: text

   ▶ ItemName (first_line_segments)     ← Line 1: name + first_line fields
     detail_line                         ← Line 2: e.g., path
     └─ preview_segments | configs=[NAP, FIJI]  ← Line 3: preview + configs

Inline Format
~~~~~~~~~~~~~

.. code-block:: text

   ▶ ItemName  (seg1 | seg2 | seg3)     ← Single line with all segments

The format is determined by the ``StyledTextLayout.multiline`` flag, set automatically based on whether detail_line or preview_line are populated.

Field Types
-----------

Permanent vs Dynamic Fields
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Type
     - Source
     - Behavior
   * - **Declared**
     - ``first_line``, ``preview_line``
     - Always shown (permanent)
   * - **Config indicators**
     - ``@global_pipeline_config(preview_label='NAP')``
     - Shown if config's ``enabled=True``
   * - **Sig-diff fields**
     - Auto-detected
     - Dynamically added when value differs from signature default

The auto-include logic (in ``_build_multiline_styled_text``) automatically adds any field with a non-default value to the preview line, even if not explicitly declared.

Value Formatting
----------------

All field values are formatted by ``format_preview_value()`` in ``config_preview_formatters.py``:

.. code-block:: python

   def format_preview_value(value: Any) -> Optional[str]:
       if value is None:
           return None
       if isinstance(value, Enum):
           return value.name if value.value else None
       if isinstance(value, list):
           if value and isinstance(value[0], Enum):
               return ','.join(v.value for v in value)
           return f'[{len(value)}]'
       if callable(value) and not isinstance(value, type):
           return getattr(value, '__name__', str(value))
       return str(value)

Field abbreviations are declared on config classes via ``@global_pipeline_config(field_abbreviations=...)``:

.. code-block:: python

   @global_pipeline_config(field_abbreviations={'well_filter': 'wf', 'output_dir_suffix': 'out'})
   @dataclass(frozen=True)
   class PathPlanningConfig:
       well_filter: Optional[str] = None
       output_dir_suffix: str = "_openhcs"

Adding a New Preview Field
--------------------------

1. **Add to ListItemFormat** (in widget):

.. code-block:: python

   LIST_ITEM_FORMAT = ListItemFormat(
       preview_line=(
           'existing_field',
           'my_new_config.some_field',  # Add field path
       ),
       formatters={
           'some_field': lambda v: f"custom:{v}" if v else None,  # Optional formatter
       },
   )

2. **Add abbreviation** (on config class):

.. code-block:: python

   @global_pipeline_config(field_abbreviations={'some_field': 'sf'})
   @dataclass(frozen=True)
   class MyNewConfig:
       some_field: str = "default"

Adding a Config Indicator
-------------------------

Config indicators (NAP, FIJI, MAT) and field abbreviations are both declared via ``@global_pipeline_config``:

.. code-block:: python

   from openhcs.config_framework import global_pipeline_config

   @global_pipeline_config(
       preview_label='NEW',
       field_abbreviations={'some_setting': 'set'}
   )
   @dataclass
   class MyNewConfig:
       enabled: bool = False
       some_setting: int = 10

The config indicator appears in ``configs=[...]`` when ``enabled=True``.

Structured Rendering Architecture
---------------------------------

The system uses structured ``StyledTextLayout`` objects instead of string parsing:

.. code-block:: python

   @dataclass(frozen=True)
   class Segment:
       """A styled text segment with field path for dirty/sig-diff matching."""
       text: str
       field_path: Optional[str] = None  # None=no styling, ''=root path

   @dataclass
   class StyledTextLayout:
       """Structured layout for styled text rendering."""
       name: Segment
       status_prefix: str = ""
       first_line_segments: List[Segment] = field(default_factory=list)
       detail_line: str = ""
       preview_segments: List[Segment] = field(default_factory=list)
       config_segments: List[Segment] = field(default_factory=list)
       multiline: bool = False

The delegate iterates segments directly and applies styling per-segment without any string matching:

1. Build ``StyledTextLayout`` with all segments
2. Store layout in item data (``LAYOUT_ROLE``)
3. Delegate reads layout and renders segments with appropriate styling
4. No string parsing required

See Also
--------

- :doc:`abstract_manager_widget` - AbstractManagerWidget base class
- :doc:`gui_performance_patterns` - Cross-window preview system
- :doc:`configuration_framework` - Config framework and lazy configs

**Implementation References:**

- ``openhcs/pyqt_gui/widgets/shared/list_item_delegate.py`` - StyledTextLayout and rendering
- ``openhcs/pyqt_gui/widgets/shared/abstract_manager_widget.py`` - ListItemFormat and build methods
- ``openhcs/pyqt_gui/widgets/config_preview_formatters.py`` - Centralized formatters
