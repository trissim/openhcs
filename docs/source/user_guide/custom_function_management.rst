Managing custom functions
=========================

Choose **Tools > Custom Functions > Manage Functions** to inspect, edit, reload,
or remove persisted custom functions. Changes refresh the function catalog used
by the Pipeline Editor.

``CustomFunctionManager`` owns persistence and registration. It stores source
under the platform-specific OpenHCS data directory, validates code before
execution, requires a supported memory decorator, and rejects name collisions.
Do not edit registry caches or generated catalog data directly.

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
