Step-editor infrastructure
==========================

Generic ObjectState-backed forms, nested field projection, reset/help chrome,
responsive layout, and editor-window lifecycle moved to
`pyqt-reactive <https://github.com/OpenHCSDev/PyQT-reactive>`_. See
:external+pyqt-reactive:doc:`parameter-form services
<architecture/parameter_form_service_architecture>` and
:external+pyqt-reactive:doc:`responsive layout widgets
<responsive_layout_widgets>`.

OpenHCS retains FunctionStep scope registration, callable/config exclusions,
domain validation, semantic code documents, compile/debug actions, and pipeline
workflow composition. See :doc:`ui_services_architecture` for the UI boundary,
:doc:`code_ui_interconversion` for code round trips, and
:doc:`external_foundations` for package ownership.
