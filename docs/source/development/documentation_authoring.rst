Documentation authoring
=======================

OpenHCS documentation has two kinds of authority: declaration-owned facts and
human-authored guidance. Keep each fact at its nominal owner and project it to
the surfaces that need it.

Choose the authority first
--------------------------

Configuration field names, types, defaults, accepted values, inheritance, and
field help belong to the typed dataclass or constructor declaration. The
desktop ObjectState forms, ``openhcs_describe_config_schema``,
:doc:`../reference/configuration`, and the knowledge base all reflect those
declarations. Do not add a documentation-only field table, tooltip registry, or
MCP description map.

Callable summaries and parameter meaning belong to the callable docstring or
the generic decorator declaration that introduces a parameter. Function forms
and the agent function catalogue consume the same introspection service. A
backend adapter may project external declaration help, but it must not copy
that help into OpenHCS metadata.

Task instructions and conceptual explanations belong to one maintained RTD
source page. The knowledge-base manifest allowlists that source; it does not
contain a second copy of the prose. Use links between pages when one reader
need depends on another.

Every material active-documentation revision must update the corresponding
entry in the repository's ``docs/audits`` ledger. This includes the root README
selected by ``project.readme`` in ``pyproject.toml``, because that source also
becomes the package-index long description. The ledger records review evidence
and the digest of the exact reviewed source; it is not another product-fact
authority. Follow ``docs/audits/README.md`` rather than copying its schema into
this guide.

Choose one reader need per page
-------------------------------

Use the Diataxis mode that matches what the reader is doing:

Tutorial
  Lead a learner through one bounded path to a visible result. Control the
  variables and explain only what is needed to complete the lesson.

How-to guide
  Help a competent reader achieve one concrete outcome. Start with the goal,
  use imperative steps, and state checkpoints and recovery actions.

Reference
  Support lookup with exact, current facts and stable organisation. Generate
  repetitive declaration facts rather than maintaining them by hand.

Explanation
  Build a mental model, clarify boundaries, and discuss why the system behaves
  as it does. Link to a how-to guide instead of embedding a task procedure.

A page can link across modes, but it should not switch modes halfway through.
Do not manufacture an empty page merely to fill a category.

Add a workflow visual
---------------------

Use a screenshot only when it helps the reader recognise a state, control, or
result needed to complete the page's task. Prefer a focused figure in a tutorial
or how-to guide over a decorative screenshot or an image-heavy reference page.

Public UI media belongs to a nominal gallery scenario. The scenario declaration
owns its stable ID, representative still or video poster, caption, description,
alt text, dimensions, capture target, derivative contract, and release evidence.
It also declares whether the scenario is available to the website, the
documentation, or both. An RTD page names only that ID:

.. code-block:: rst

   .. openhcs-gallery:: napari-roi-navigation

The Sphinx projection resolves every other figure property from the scenario.
Do not add a second image path, caption, alt text, media-type branch, or
documentation-only screenshot registry. A documentation-only scenario remains
part of the same media catalogue, asset inventory, and checksum record; its
typed publication target merely keeps it out of the website card projection.
If the available scenario does not truthfully depict the page's workflow,
capture and declare a new scenario rather than placing a misleading image.

Follow :doc:`media_gallery_capture` to capture and verify the real application
surface. Build the documentation with warnings treated as errors, then inspect
the rendered figure at desktop and narrow widths. Confirm that the image remains
legible, the caption matches what is visible, and the alt text conveys the
task-relevant information without relying on the image.

Update a configuration description
----------------------------------

1. Find the dataclass field or constructor parameter that owns the value.
2. Describe its effect, scope, units or accepted shape, inheritance behaviour,
   and important distinction from adjacent settings.
3. Keep generic transport, storage, UI, or introspection semantics in the
   extracted package that owns them. Keep OpenHCS microscopy and pipeline
   meaning in OpenHCS.
4. Run the help-projection and schema tests.
5. Build the documentation with warnings treated as errors.
6. Open the rendered reference and the live field help. Confirm that both show
   the exact owner text and a useful default.
7. Retrieve the corresponding knowledge-base section through a fresh current-
   source MCP process.

Verification
------------

From the repository root:

.. code-block:: bash

   python -m pytest -q \
     tests/unit/agent/test_config_reference_service.py \
     tests/unit/agent/test_config_schema_authority.py \
     tests/unit/agent/test_knowledge_base_service.py \
     tests/unit/pyqt_gui/test_user_facing_help_surfaces.py
   python -m sphinx -E -W --keep-going -b html docs/source docs/build/html

For live UI and MCP checks, follow
:doc:`agent_workflow_validation`. Treat an MCP process from another checkout as
drift evidence, not validation of the current source tree.
