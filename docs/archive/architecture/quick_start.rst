Architecture Quick Start
=======================

This short guide helps new contributors and maintainers pick the right path through the architecture documentation.

Pick one path below depending on what you need to do; each path links to 3–5 targeted architecture pages with a one-line reason why you should read them next.

Core systems (read to understand the runtime internals)
------------------------------------------------------

1. :doc:`function_pattern_system` — How functions are wrapped, validated and executed (core processing model)
2. :doc:`pipeline_compilation_system` — The multi-pass compiler that plans memory, GPU, and materialization
3. :doc:`storage_and_memory_system` — VFS, memory staging and materialization behavior (where data lives)

Why these three? Together they explain how OpenHCS turns Python functions into efficient, multi-backend pipelines.

Integrations (if you need to connect external tools)
---------------------------------------------------

1. :doc:`external_integrations_overview` — Patterns and protocols used for all external integrations
2. :doc:`napari_integration_architecture` — Napari-specific viewer integration and streaming details
3. :doc:`omero_backend_system` — OMERO storage backend and file-annotation patterns

Why these three? They show the common messaging patterns and backend adapters used to connect OpenHCS with external viewers and platforms.

User interface / developer UX (if you build on the UI)
------------------------------------------------------

1. :doc:`parameter_form_lifecycle` — How parameter forms are synchronized and validated
2. :doc:`service-layer-architecture` — Service/adapter patterns used by UI components
3. :doc:`tui_system` — Terminal UI architecture and component composition (desktop GUI concepts map to similar services)

Why these three? They describe the UI lifecycle, validation, and the thin service layer you can extend to add new editors or viewers.

Quick checklist for new contributors
-----------------------------------

- Run the unit/integration tests mentioned on the relevant pages (see "Implementation files" sections).
- Look for the `Status:` tag at the top of pages to know what is stable vs draft.
- If you only need a single example, search for `examples/` in the repository and the "Quick Start" snippets in the Getting Started guide.

What to do next
----------------

- If you want, I can add small runnable examples for each path (examples/quickstart_core.py, examples/quickstart_integration.py, examples/quickstart_ui.py) and wire them into the docs.
