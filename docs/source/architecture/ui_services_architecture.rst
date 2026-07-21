UI services
===========

`pyqt-reactive <https://github.com/OpenHCSDev/PyQT-reactive>`_ owns reusable
forms, widget descriptors, responsive layouts, parameter-help projection, and
window-management mechanics. OpenHCS owns the domain state topology and the
adapters that attach those mechanisms to pipeline, execution, knowledge, and
viewer services.

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

UI-thread mutation boundary
---------------------------

Agent and background services never mutate Qt widgets directly.
``UiThreadDispatcher`` owns one Qt-affine proxy. ``call()`` provides bounded
request/response dispatch when a result is required, while ``post()`` queues
fire-and-forget rendering or selection work through the same proxy. Both reject
work after shutdown. Domain workflows choose whether a response is required;
they do not grow their own signal/thread fallback chains.

Managed contextual help
-----------------------

pyqt-reactive owns the generic declaration-to-parameter-help projection.
OpenHCS manager classes declare a ``KnowledgeBaseDocumentTarget`` and install a
Help action beside the responsive title. The action opens the one managed
knowledge-base window through ``WindowManager`` and navigates that existing
window to the declared document/section. It does not create a per-widget help
browser or copy knowledge prose into the manager.

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
