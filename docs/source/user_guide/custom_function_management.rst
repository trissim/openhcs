Managing custom functions
=========================

Choose **Tools > Custom Functions > Manage Functions** to inspect, edit, reload,
or remove persisted custom functions. Changes refresh the function catalog used
by the Pipeline Editor and local MCP authoring tools. Refresh invalidates the
shared endpoint projection and requests the updated catalogue asynchronously,
so the Function Selector remains responsive while the execution server exposes
the new declaration.

.. openhcs-gallery:: ui-custom-function-manager

``CustomFunctionManager`` owns persisted source and coordinates create, load,
replace, and delete transactions. It stores source under the platform-specific
OpenHCS data directory, validates code before execution, requires a supported
memory decorator, and rejects name collisions. The process-local
``CustomFunctionRuntimeRegistry`` atomically publishes the derived callable
metadata, source lifetime, and public module exports, so readers see either the
previous complete catalogue or the validated replacement. Reloading persisted
source retains ephemeral declarations registered by the current process. Do
not edit runtime projections, registry caches, or generated catalogue data
directly.

Custom-function changes are published as a headless domain event. The desktop
loads a Qt signal adapter for that event, while local MCP and execution-server
processes use the same manager without importing Qt. The event observes
subscribers without owning their lifetime, so closing a temporary desktop
adapter cannot leave an invalid GUI callback in the headless domain owner.

Programmatic registration is available for tooling:

.. code-block:: python

   from openhcs.processing.custom_functions import CustomFunctionManager

   manager = CustomFunctionManager()
   functions = manager.register_from_code(source_text, persist=True)

Treat registered source as executable code. Review it before loading, and use
the desktop editor's validation result to resolve rejected imports, missing
decorators, signature problems, or collisions.

See :doc:`custom_functions` for the callable contract and
:doc:`../development/callable_artifact_authoring` for runtime-bound parameters
and artifacts.
