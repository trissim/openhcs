UI services
===========

`pyqt-reactive <https://github.com/OpenHCSDev/PyQT-reactive>`_ owns reusable
forms, widget descriptors, responsive layouts, parameter-help projection, and
window-management mechanics. OpenHCS owns the domain state topology and the
adapters that attach those mechanisms to pipeline, execution, knowledge, and
viewer services.

Composition boundary
--------------------

The desktop is a domain composition over two reusable owners:

.. code-block:: text

   OpenHCS declaration or workflow
       -> scoped ObjectState
          (working values, saved/live resolution, provenance, history)
       -> pyqt-reactive form, manager, table, or managed window
       -> OpenHCS action adapter
          (compile, execute, inspect, stream, or open domain documentation)

ObjectState owns editable state independently of a window. pyqt-reactive owns
how that state is projected into generic Qt controls and how those controls are
laid out, refreshed, scrolled, and closed. OpenHCS decides which pipeline or
microscopy object is editable, assigns its scope and parent, supplies domain
validation and code documents, and connects semantic actions to orchestrators.

This boundary is deliberately one-way. OpenHCS must not copy generic form
metadata, column-presentation preferences, responsive breakpoints, or window
lifecycle tables. pyqt-reactive must not import OpenHCS configuration classes or
pipeline workflows. See
:external+objectstate:doc:`ObjectState state management <state_management>` and
:external+pyqt-reactive:doc:`pyqt-reactive parameter-form services
<architecture/parameter_form_service_architecture>` for the reusable owners.

Window and action composition
-----------------------------

OpenHCS declares stable window identities and factories for Plate Manager,
Pipeline Editor, Image Browser, configuration, Results, logs, and server
management. pyqt-reactive's ``WindowManager`` and ``ScopeWindowFactory`` own
generic registration, focus/reuse, parentage, navigation, and close cleanup.
Reopening a scoped editor therefore focuses the existing window; it does not
create a second domain state or a second window registry.

Form headers and field rows are likewise generic layout surfaces. OpenHCS
contributes semantic actions such as Help, Cancel, Save, Reset, and View Code;
pyqt-reactive keeps the title/help group together, pins commit actions to the
right, wraps auxiliary actions only when their real minimum widths require it,
and reserves scroll-bar space around form content. Domain windows do not carry
their own pixel breakpoint or scroll-overlap fixes.

See :external+pyqt-reactive:doc:`window manager usage
<development/window_manager_usage>` and
:external+pyqt-reactive:doc:`responsive layout widgets
<responsive_layout_widgets>`.

Table-browser composition
-------------------------

OpenHCS table browsers supply domain column declarations, row extraction,
search text, and optional context or base filtering. pyqt-reactive's
``AbstractTableBrowser`` owns the table, search/selection lifecycle, automatic
per-column filters, and user presentation state keyed by stable column identity.
Column visibility and order are therefore generic preferences rather than
Image-Browser- or Function-Selector-specific maps.

The Image Browser contributes its folder tree as context above the generic
column-filter surface and composes folder/plate selection as a base filter. It
does not rearrange the generic filter panel or implement another visibility
dialog. See :external+pyqt-reactive:doc:`the table-browser owner
<architecture/abstract_table_browser>`.

Image Browser state topology
----------------------------

For a selected plate, the Image Browser registers one
``<plate>::image_browser`` ObjectState whose parent is the plate's
PipelineConfig state. The Image Browser and FunctionStep states are siblings:
both resolve through the selected PipelineConfig and then process-global
configuration. Viewer tabs come from the registered ``StreamingConfig`` field
keys rather than a copied Napari/Fiji list.

Image Browser streaming controls are a live surface, not a Save/Cancel editor.
Each field edit advances the same ObjectState's working value and saved baseline
together, so the next image open uses the current configuration without a false
unsaved marker. This behavior is specific to that live browser surface; ordinary
configuration windows retain explicit Save and Cancel semantics.

Process configuration topology
------------------------------

``UIConfig`` is the process-global desktop declaration. Its ``zmq`` field is an
``OpenHCSZMQConfig`` owned by the runtime transport package; it is not mirrored
as a widget settings table. ``PyQtGuiRuntimeContext`` carries the current
``UIConfig`` beside ``GlobalPipelineConfig``. The global configuration window
edits their registered ObjectState scopes and its save handlers return the new
typed values to the main window.

``OpenHCSMainWindow.set_ui_config()`` is the live application boundary. It
replaces the runtime-context value, updates the shared widget service, forwards
the ZMQ declaration to Plate Manager and ZMQ Manager, and emits
``ui_config_changed``. Plate Manager then updates its existing execution-client
service. A saved host, transport, timeout, or port setting therefore affects
future connections through the same declaration; consumers must not cache a
second port table or restart a hidden client with defaults.

That handoff is not a promise that every ``UIConfig`` field is applied live.
The current behavior is owner-specific:

- the application and main-window runtime contexts retain the newly saved
  object;
- Plate Manager adopts ``zmq`` for future execution connections;
- the embedded and already-open managed server browsers rebuild their ZMQ scan
  configuration, and an already-open Image Browser adopts the new ZMQ
  declaration;
- existing menu actions and the time-travel event filter are not rebound when
  ``shortcuts`` changes;
- progress coalescing currently constructs its default interval instead of
  consuming ``UIConfig.progress``;
- the system monitor receives the exact saved ``performance_monitor``
  declaration through pyqt-reactive's ``update_config()`` owner boundary; it
  rebuilds sampling only when derived sampling behavior changes and applies
  plot presentation idempotently; and
- style/theme, window policy, logging, and agent bridge lifecycle are not
  generally rebuilt by ``set_ui_config()``.

UIConfig edits also have no process-round-trip persistence path at present. The
GlobalPipelineConfig tab writes through the config cache, while the UIConfig tab
updates only the running process. Main-window QSettings restore/save is disabled,
and a restart constructs the default UIConfig again. Until each declared field
has an explicit behavior owner and persistence policy, documentation and callers
must distinguish **live**, **next operation or future instance**, and
**not currently applied** behavior instead of describing UIConfig as globally
live.

UI-thread mutation boundary
---------------------------

Agent and background services never mutate Qt widgets directly.
``UiThreadDispatcher`` owns one Qt-affine proxy. ``call()`` provides bounded
request/response dispatch when a result is required, while ``post()`` queues
fire-and-forget rendering or selection work through the same proxy. Both reject
work after shutdown. Domain workflows choose whether a response is required;
they do not grow their own signal/thread fallback chains.

Contextual Help routes
----------------------

pyqt-reactive owns the generic declaration-to-parameter-help projection.
OpenHCS manager classes declare a ``KnowledgeBaseDocumentTarget`` and install a
Help action beside the responsive title. The action opens the one managed
knowledge-base window through ``WindowManager`` and navigates that existing
window to the declared document/section. It does not create a per-widget help
browser or copy knowledge prose into the manager.

Help is not yet one unified content/presentation system. The main Knowledge Base
action and Plate/Pipeline manager Help use the managed, source-backed OpenHCS
knowledge window. Configuration, function, nested-config, parameter, and field
Help use pyqt-reactive's docstring/parameter projection and its help windows.
Those routes preserve their respective authoritative content, but they do not
currently combine a declaration docstring with a related knowledge-base page in
one request or managed window. New integrations must not claim that unification
or add a copied target-to-prose table.

Results and selected-plate ownership
------------------------------------

Plate Manager owns the visible plate selection and delegates pipeline editing,
compile/run actions, Results, and viewer launch to the orchestrator registered
for that row. Output rows resolve through the declared source/output relation so
Results can use the producing orchestrator rather than treating the output plate
as an unrelated session.

The Results window is measurement-first. It opens immediately over retained
typed live-measurement previews and selects the latest semantic entry. The image
browser is represented by a placeholder and is constructed only when the user
selects its tab or requests an artifact. This keeps measurement inspection
available without paying image/viewer startup cost and preserves the selected
orchestrator as the single source for later image navigation.

Package boundary
----------------

See :doc:`external_foundations` for package ownership,
:doc:`orchestrator_configuration_management` for compile-time configuration
resolution, and :doc:`code_ui_interconversion` for ObjectState/code-document
revision ownership.
