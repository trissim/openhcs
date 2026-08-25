Declarative windows
===================

Generic live-window discovery, registration, scope-based reuse, navigation,
parentage, and close cleanup moved to
`pyqt-reactive <https://github.com/OpenHCSDev/PyQT-reactive>`_.
See :external+pyqt-reactive:doc:`window manager usage
<development/window_manager_usage>` and
:external+pyqt-reactive:doc:`the scope-window factory
<architecture/scope_window_factory>`.

OpenHCS retains stable product window identities, domain factories, ObjectState
scope selection, and pipeline/execution actions. Their cross-package composition
is documented in :doc:`ui_services_architecture`; package ownership and
migration policy are in :doc:`external_foundations`.
