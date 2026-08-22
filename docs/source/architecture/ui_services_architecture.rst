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

OpenHCS declares stable window identities and routes for Plate Manager,
Pipeline Editor, Image Browser, configuration, Results, logs, and server
management. pyqt-reactive's ``WindowManager`` and ``ScopeWindowRegistry`` own
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
OpenHCS-owned ``OpenHCSZMQConfig`` subtype of ZMQRuntime's generic
``ZMQConfig``; it is not mirrored as a widget settings table.
``PyQtGuiRuntimeContext`` carries the current
``UIConfig`` beside ``GlobalPipelineConfig``. The global configuration window
edits their registered ObjectState scopes. The declaration contains only
settings with real lifecycle owners: performance-monitor sampling and
presentation, progress-update coalescing, application shortcuts, execution ZMQ,
process logging, and the local agent UI bridge. Theme, generic window policy,
plugin payloads, bridge reserve limits, and shortcut action descriptions are not
duplicated as editable UIConfig fields. The update-check preference is itself a
``UIConfig`` field consumed by the desktop update lifecycle.

``OpenHCSMainWindow.set_ui_config()`` is the live application boundary. It
first reconciles every live consumer and only then publishes the runtime-context
value and ``ui_config_changed`` signal. The system monitor applies
``performance_monitor`` through its idempotent update owner. Plate Manager
updates its execution connection declaration and the already-materialized
progress timer; a not-yet-materialized progress service receives the same value
when constructed. The main-window shortcut lifecycle validates the complete key
set before rebinding its concrete QActions and time-travel event filter. ZMQ
Manager rebuilds its scan inputs from the current transport declaration and the
bridge's actual running binding, never from an unstarted configured port.
The logging lifecycle rebuilds its owned console and rotating-file handlers from
the exact ``UIConfig.logging`` declaration. Launcher flags are ephemeral
overrides rather than persisted mirrors. Log discovery derives its directory
from that same live declaration, so changing the location does not require a
second registry or copied path setting.

``MainWindowUiBridgeLifecycle`` owns bridge enable/disable and exact
configuration reconciliation. An unchanged running configuration is a no-op.
A changed configuration stops the old server and starts a candidate; factory or
partial-start failure cleans the candidate and restarts the prior server before
the error reaches the save transaction. The outer UIConfig handoff similarly
reapplies the prior configuration if any later live consumer rejects the new
one.

Both process configuration roots use the same generic typed config-cache
boundary. Writes serialize before opening a same-directory temporary file,
flush and fsync it, atomically replace the cache target, and fsync the containing
directory where supported. Startup loads and schema-migrates UIConfig using
resolved type hints, then applies authoritative environment overrides for the
agent bridge. There is no parallel Qt cache adapter and no main-window
``QSettings`` timer or disabled window-state persistence surface.

An isolated launch or test may set ``OPENHCS_UI_CONFIG_CACHE_FILE`` to one
absolute cache path. ``UIConfigCacheEnvironment`` resolves that process-local
boundary before the ordinary XDG path is chosen; it does not add another
configuration value or persistence implementation.

The two-tab configuration window is one transaction, not two independent Save
callbacks. It validates all materialized forms and runs mutation guards before
reconstruction. It then checkpoints every affected ObjectState plus the
independent saved/live global contexts, commits all states, and invokes explicit
reversible persistence/live participants. A participant or saved-state failure
first compensates attempted external participants in reverse order, restores the
exact saved/live global-context split, and only then restores ObjectState's exact
dirty/saved/provenance state. ObjectState observers therefore cannot see a
rolled-back state beside a failed external candidate. Observer signals, token
notification, and window close occur after the durable transaction and cannot
roll it back. The first tab is materialized initially; inactive config forms and
their actions are constructed once on first activation.

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

pyqt-reactive owns the generic ``HelpDocument`` model, rich
``HelpDocumentBrowser``, and declaration-to-docstring/parameter-help
projection. The model carries an explicit plain-text, Markdown, or
reStructuredText format; the browser owns safe rendering, theme hierarchy,
wrapping, links, code blocks, scrolling, and layout-derived sizing. Config,
nested-config, parameter, and field popups all use this one renderer. The old
hand-built label/text-edit and plain-text browser paths no longer exist.

OpenHCS manager classes declare a ``KnowledgeBaseDocumentTarget`` and install a
Help action beside the responsive title. The action opens the one managed help
window through ``WindowManager`` and navigates that existing window to the
declared source-backed document/section. Its Guides tab resolves content only
through ``KnowledgeBaseService``. Its Functions tab enumerates and searches the
authoritative function registry through ``FunctionCatalogService``, resolves
the exact callable, then passes pyqt-reactive's structured docstring projection
to the same rich renderer. Thus help now has one content/presentation
abstraction without merging its semantic authorities: authored knowledge stays
in the knowledge manifest and callable/config help stays on the declarations
and docstrings. No target-to-prose table, copied function list, compatibility
renderer, or fallback help system is retained.

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
