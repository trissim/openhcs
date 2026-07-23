Compact Window Patterns
=======================

Compact window patterns enable minimal UI layouts that work well on smaller screens and when tiling multiple windows side-by-side.

Overview
--------

Traditional form dialogs often require large minimum widths to accommodate titles, help buttons, and action buttons in a single row. The compact window pattern separates these elements into multiple rows, allowing windows to be much narrower while maintaining usability.

Two-Row Header Pattern
----------------------

The two-row header pattern separates the window title from action buttons:

**Row 1**: Title and help button
**Row 2**: Action buttons (Save, Cancel, Reset, etc.)

Traditional Single-Row Layout::

  [Configure MyClass] [?]          [View Code] [Reset] [Cancel] [Save]
  
  Minimum width: 600-800px

Compact Two-Row Layout::

  [Configure MyClass] [?]
  [View Code] [Reset] [Cancel] [Save]
  
  Minimum width: 150-400px

Implementation
--------------

Using _create_compact_header Helper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``BaseManagedWindow`` class provides a helper for creating compact headers:

.. code-block:: python

   from pyqt_reactive.widgets.shared.base_form_dialog import BaseManagedWindow
   
   class MyConfigWindow(BaseManagedWindow):
       def setup_ui(self):
           layout = QVBoxLayout(self)
           
           # Create compact two-row header
           title_label, button_layout = self._create_compact_header(
               layout,
               title_text="Configure MyClass",
               title_color=self.color_scheme.to_hex(
                   self.color_scheme.text_accent
               )
           )
           
           # Add action buttons to button row
           save_btn = QPushButton("Save")
           cancel_btn = QPushButton("Cancel")
           button_layout.addWidget(save_btn)
           button_layout.addWidget(cancel_btn)

Manual Implementation
~~~~~~~~~~~~~~~~~~~~~

You can also implement the pattern manually:

.. code-block:: python

   def setup_ui(self):
       layout = QVBoxLayout(self)
       
       # Row 1: Title
       title_widget = QWidget()
       title_layout = QHBoxLayout(title_widget)
       
       self.header_label = QLabel("Window Title")
       self.header_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
       title_layout.addWidget(self.header_label)
       title_layout.addStretch()
       
       layout.addWidget(title_widget)
       
       # Row 2: Buttons
       button_widget = QWidget()
       button_layout = QHBoxLayout(button_widget)
       button_layout.addStretch()
       
       save_btn = QPushButton("Save")
       cancel_btn = QPushButton("Cancel")
       button_layout.addWidget(save_btn)
       button_layout.addWidget(cancel_btn)
       
       layout.addWidget(button_widget)

Window Size Guidelines
----------------------

Minimum Sizes
~~~~~~~~~~~~~

For compact windows, use these minimum sizes:

- **Minimum width**: 150px (ultra-minimal, allows narrow tiling)
- **Default width**: 400-500px (comfortable for most content)
- **Minimum height**: 150px (just enough for header + some content)

Example from ConfigWindow:

.. code-block:: python

   self.setMinimumSize(150, 150)  # Ultra minimal
   self.resize(400, 400)          # Default size

Benefits
--------

Screen Space Efficiency
~~~~~~~~~~~~~~~~~~~~~~~

- **Side-by-side tiling**: Two windows fit comfortably on a 1920px wide monitor
- **Laptop screens**: Works well on smaller screens (1366px, 1440px)
- **Multi-monitor setups**: Allows dense window arrangements

Improved Usability
~~~~~~~~~~~~~~~~~~

- **Reduced eye movement**: Buttons are closer to content
- **Better focus**: Title is prominent at top
- **Scannable layout**: Clear visual hierarchy

Integration with Responsive Widgets
-----------------------------------

Combine compact headers with responsive layouts for maximum flexibility:

.. code-block:: python

   from pyqt_reactive.widgets.shared.responsive_layout_widgets import (
       set_wrapping_enabled
   )
   
   # Enable responsive wrapping
   set_wrapping_enabled(True)
   
   # Create compact window
   window = ConfigWindow(config_class=MyConfig)
   
   # Parameter rows will wrap when narrow
   # Window can be very narrow while remaining usable

Examples
--------

ConfigWindow
~~~~~~~~~~~~

.. code-block:: python

   class ConfigWindow(BaseFormDialog):
       def setup_ui(self):
           self.setMinimumSize(150, 150)
           self.resize(400, 400)
           
           layout = QVBoxLayout(self)
           
           # Compact two-row header
           title_text = f"Configure {self.config_class.__name__}"
           title_color = self.color_scheme.to_hex(
               self.color_scheme.text_accent
           )
           self._header_label, button_layout = self._create_compact_header(
               layout, title_text, title_color
           )
           
           # Add help button to title row
           help_btn = HelpButton(help_target=self.config_class)
           title_row = self._header_label.parent().layout()
           title_row.insertWidget(1, help_btn)
           
           # Add action buttons to button row
           button_layout.addWidget(view_code_btn)
           button_layout.addWidget(reset_btn)
           button_layout.addWidget(cancel_btn)
           button_layout.addWidget(save_btn)

DualEditorWindow
~~~~~~~~~~~~~~~~

.. code-block:: python

   class DualEditorWindow(BaseFormDialog):
       def setup_ui(self):
           self.setMinimumSize(150, 150)
           self.resize(400, 400)
           
           layout = QVBoxLayout(self)
           
           # Row 1: Title only
           title_widget = QWidget()
           title_layout = QHBoxLayout(title_widget)
           
           self.header_label = QLabel()
           self.header_label.setFont(
               QFont("Arial", 14, QFont.Weight.Bold)
           )
           title_layout.addWidget(self.header_label)
           title_layout.addStretch()
           
           layout.addWidget(title_widget)
           
           # Row 2: Tabs and buttons
           control_row = QHBoxLayout()
           control_row.addWidget(self.tab_bar, 0)
           control_row.addStretch()
           control_row.addWidget(cancel_btn)
           control_row.addWidget(save_btn)
           
           layout.addLayout(control_row)

See Also
--------

- :doc:`/architecture/parameter_form_service_architecture` - Form architecture
- :doc:`window_manager_usage` - Multi-window management
