MCP Knowledge Base
==================

OpenHCS exposes a read-only, source-backed documentation knowledge base through
the agent API and MCP server. The knowledge base is owned by
``openhcs.agent.services.knowledge_base_service.KnowledgeBaseService``; the MCP
server delegates to that service and does not accept arbitrary filesystem paths
from clients.

Current MCP surface:

* resource: ``openhcs://knowledge``
* tool: ``openhcs_list_knowledge_documents``
* tool: ``openhcs_get_knowledge_document``
* tool: ``openhcs_search_knowledge``

Fresh-Checkout Commands
-----------------------

From an active checkout with its environment sourced:

.. code-block:: bash

   . .venv/bin/activate
   python -m openhcs.mcp.dev_client knowledge
   python -m openhcs.mcp.dev_client knowledge-search "plate well site channel" --limit 5
   python -m openhcs.mcp.dev_client knowledge-search "CellProfiler examples cppipe native OpenHCS examples" --limit 5
   python -m openhcs.mcp.dev_client knowledge-search "ExampleHuman OpenHCS Python" --limit 5
   python -m openhcs.mcp.dev_client knowledge-search "GlobalPipelineConfig ObjectState" --limit 5
   python -m openhcs.mcp.dev_client knowledge-document openhcs_domain_expert_onboarding --max-chars 4000
   python -m openhcs.mcp.dev_client knowledge-document openhcs_example_corpus_map --max-chars 4000
   python -m openhcs.mcp.dev_client knowledge-document openhcs_official30_benchmark_recipes --section-id examplehuman-openhcs-python --max-chars 50000

First Query Sequence
--------------------

Use this sequence when starting from no context:

1. Run ``knowledge`` and choose a ``document_id`` from the catalog.
2. Run ``knowledge-search <term>`` for a domain term such as ``microscopy``,
   ``plate layout``, ``segmentation``, or ``fluorescence``.
3. Search for existing examples before authoring a new workflow. For an imported
   recipe, use the biological task name plus ``OpenHCS Python`` or
   ``official30 recipe``. For native presets, search ``MFD preset``.
4. Run another ``knowledge-search <term>`` for the technical operator concept
   you need, such as ``GlobalPipelineConfig``, ``ObjectState``,
   ``custom function``, ``Napari``, ``FunctionStep``, or ``MaterializationSpec``.
5. Use the returned ``section_id`` with ``knowledge-document`` to read only the
   relevant section.
6. Follow source paths and capability names in the result before editing code.

For a new MCP session, the preceding knowledge sequence follows
``openhcs_health_check``,
``openhcs_get_authoring_context(kind="first_use")``, and
``openhcs_search_capabilities`` with task-relevant filters. The returned
``surface_profile`` and declaration-owned workflow metadata are the authority
for what that server exposes; use ``openhcs_list_capabilities`` only when the
complete selected registry is needed. Before setting non-obvious configuration
fields, call ``openhcs_describe_config_schema`` with ``pipeline`` or ``global``.
For nested step overrides, construct the mutation value from each field's
``authoring_value_path`` rather than treating its schema-navigation ``path`` as
a dotted mutation key.

Official30 Converted Examples
-----------------------------

The ``openhcs_official30_benchmark_recipes`` document derives its case catalog
directly from ``benchmark/manifests/official30_portable_axis1.json``. Every case
advertises an ``<case>-openhcs-python`` child section. Requesting that exact
section resolves the manifest-declared ``.cppipe`` and dataset roots, calls the
canonical public CellProfiler importer, and returns one factored, importable
Python document defining ``pipeline_config`` and ``pipeline_steps``.

Request converted source with ``max_chars=50000``. Fifteen current recipes are
larger than the default 12,000-character response bound; a truncated response
is useful for inspection but is not importable Python.

Catalog listing, search, and ordinary recipe reads do not convert all 30 cases.
Conversion happens only for the requested source section and is recomputed from
the manifest-resolved ``.cppipe`` and dataset source on each request. The public
importer can depend on dataset metadata and external source files, and no existing
cache owner exposes that complete identity. The projection therefore does not
cache or store generated Python files, and it never returns raw ``.cppipe`` text
as a fallback; missing sources and conversion failures are structured response
errors.

Each current documentation source is registered under one canonical document id.
Search tags live on that canonical entry rather than compatibility aliases that
would expose the same file as multiple semantic authorities.

Response Shape
--------------

The catalog response contains document summaries:

.. code-block:: json

   {
     "schema_version": "openhcs.agent.v1",
     "documents": [
       {
         "document_id": "openhcs_architecture_quick_start",
         "title": "OpenHCS architecture quick start",
         "source_path": "docs/source/architecture/quick_start.rst",
         "tags": ["architecture quick start", "first use", "MCP quick start"]
       }
     ],
     "warnings": []
   }

Document reads include the bounded ``content`` plus a section list. The
``max_chars`` bound applies to returned document content; metadata remains
available so agents can choose follow-up section ids.

Current And Historical Documents
--------------------------------

The primary current architecture starts at
``docs/source/architecture/system_overview.rst`` and continues through the
nominal ownership, processing, source, artifact, runtime, equivalence, and
CellProfiler pages in that section. Domain-facing guidance starts at
``docs/source/guide_for_biologists/domain_expert_onboarding.rst``.
Example-corpus guidance starts at ``docs/source/guides/example_corpus_map.rst``
and points agents to existing CellProfiler examples, native benchmark pipelines,
presets, and current complete examples. Technical operator guidance is
allowlisted from canonical compiler, source, artifact, runtime, ownership,
extension, custom-function, and viewer pages. Historical MCP plans are excluded
from the active knowledge manifest so an agent cannot mistake a frozen tool
census or design proposal for the current registry-backed surface.

Freshness
---------

Allowlisted documentation files are part of the MCP stale-source watchlist.
When a running MCP process has stale code or watched docs, normal tools and
resources report a structured stale-server payload. ``openhcs_health_check``
remains callable so a client can see which paths changed and restart the MCP
server.
