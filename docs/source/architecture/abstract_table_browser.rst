Table browsers
==============

The generic searchable table, selection lifecycle, column filters, context
surface, and visibility/order presentation state moved to
`pyqt-reactive <https://github.com/OpenHCSDev/PyQT-reactive>`_. See
:external+pyqt-reactive:doc:`the AbstractTableBrowser owner documentation
<architecture/abstract_table_browser>` for that public mechanism.

OpenHCS supplies domain column declarations, row extraction, search text, and
base/context filters for consumers such as Image Browser and Function Selector.
Their ObjectState and workflow composition is documented in
:doc:`ui_services_architecture`; package ownership and migration policy are in
:doc:`external_foundations`.
