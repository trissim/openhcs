Developer guide
===============

This section describes the current extension boundaries. Start with the
workflow guide, then follow the owner for the concept you are changing.

Architecture and extension workflows
------------------------------------

.. toctree::
   :maxdepth: 2

   respecting_codebase_architecture
   documentation_authoring
   extension_workflows
   callable_artifact_authoring
   cellprofiler_module_authoring
   source_binding_extension
   compiler_extension
   runtime_value_extension
   runtime_system_assembly_rules
   repository_setup
   ast_refactoring_workflow

Focused development guides
--------------------------

.. toctree::
   :maxdepth: 1

   pipeline_debugging_guide
   pyclesperanto_simple_implementation
   mcp_development
   agent_workflow_validation
   mcp_knowledge_base
   mcp_release
   git_worktree_testing
   omero_testing
   ../guides/testing_guide

Repository setup and verification
---------------------------------

Clone recursively and install the extracted packages before OpenHCS itself.
The exact commands are in :doc:`repository_setup`.

Before opening a change:

1. Run the focused tests for the owning declaration, strategy, or compiler
   phase.
2. Run an importer or integration test when declaration lowering changes.
3. Build these docs with warnings treated as errors.
4. Check that generic code did not acquire a concrete backend import or a
   parallel semantic table.

Related architecture
--------------------

- :doc:`../architecture/system_overview`
- :doc:`../architecture/nominal_ownership`
- :doc:`../architecture/external_foundations`
