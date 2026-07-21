AbstractTableBrowser ABC
========================

**Generic searchable table widget pattern for consistent browser UIs.**

*Module: openhcs.pyqt_gui.widgets.shared.abstract_table_browser*

Overview
--------

``AbstractTableBrowser`` is an abstract base class for table-based browser widgets.
It provides common infrastructure for displaying searchable, filterable table views
of item collections. Subclasses implement abstract methods to customize column layout,
row population, and event handling.

Each subclass requires <100 lines of code to implement a fully-functional browser.

Core Components
---------------

.. code-block:: text

   AbstractTableBrowser[T]
   ├── search_input: QLineEdit       # Search box
   ├── table_widget: QTableWidget    # Main table
   ├── status_label: QLabel          # "Showing N/M items"
   ├── _search_service: SearchService # Filter logic
   └── Signals:
       ├── item_selected(key, item)
       ├── item_double_clicked(key, item)
       └── items_selected(keys)        # Multi-select mode

ColumnDef Configuration
-----------------------

Columns are declaratively defined using ``ColumnDef``:

.. code-block:: python

   @dataclass
   class ColumnDef:
       name: str                       # Header text
       key: str                        # Internal key
       width: Optional[int] = None     # Fixed width (None = stretch)
       sortable: bool = True           # Enable column sorting
       resizable: bool = True          # Enable resize handle

Abstract Methods
----------------

Subclasses must implement three methods:

.. code-block:: python

   class MyBrowser(AbstractTableBrowser[MyItemType]):

       def get_columns(self) -> List[ColumnDef]:
           """Return column definitions for the table."""
           return [
               ColumnDef("Name", "name", width=200),
               ColumnDef("Status", "status", width=100),
           ]

       def extract_row_data(self, item: MyItemType) -> List[str]:
           """Extract display values for a table row."""
           return [item.name, item.status]

       def get_searchable_text(self, item: MyItemType) -> str:
           """Return searchable text for an item."""
           return f"{item.name} {item.status}"

Implementations
---------------

.. list-table::
   :header-rows: 1

   * - Browser
     - Columns
     - Used By
   * - ``FunctionTableBrowser``
     - Name, Module, Backend, Contract, Tags
     - FunctionSelectorDialog
   * - ``ImageTableBrowser``
     - Filename + dynamic metadata keys
     - ImageBrowserWidget
   * - ``SnapshotTableBrowser``
     - Index, Time, Label, States
     - SnapshotBrowserWindow

Dynamic Columns
---------------

``ImageTableBrowser`` demonstrates dynamic column configuration:

.. code-block:: python

   browser = ImageTableBrowser()
   browser.set_metadata_keys(['well', 'channel', 'z_slice'])  # Add columns
   browser.set_items(file_metadata_dict)

The ``set_metadata_keys()`` method calls ``reconfigure_columns()`` to rebuild
the table header before populating items.

Selection Modes
---------------

Two modes are supported:

- **single** (default): One item at a time, emits ``item_selected``
- **multi**: Multiple items, emits ``items_selected`` with list of keys

.. code-block:: python

   # Multi-select for batch operations
   browser = ImageTableBrowser(selection_mode='multi')
   browser.items_selected.connect(self._on_files_selected)

SearchService Integration
-------------------------

The ``SearchService`` (from ``openhcs.ui.shared.search_service``) provides
framework-agnostic search with:

- Minimum character threshold (default: 2)
- Case-insensitive substring matching
- Performance optimization for short queries

See Also
--------

- :doc:`list_item_preview_system` - List widget styling patterns
- :doc:`widget_protocol_system` - ABC contracts for widgets
