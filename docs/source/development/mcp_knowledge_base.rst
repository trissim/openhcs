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
   python -m openhcs.mcp.dev_client knowledge-search "GlobalPipelineConfig ObjectState" --limit 5
   python -m openhcs.mcp.dev_client knowledge-document openhcs_domain_expert_onboarding --max-chars 4000
   python -m openhcs.mcp.dev_client knowledge-document openhcs_example_corpus_map --max-chars 4000

First Query Sequence
--------------------

Use this sequence when starting from no context:

1. Run ``knowledge`` and choose a ``document_id`` from the catalog.
2. Run ``knowledge-search <term>`` for a domain term such as ``microscopy``,
   ``plate layout``, ``segmentation``, or ``fluorescence``.
3. Search for existing examples before authoring a new workflow, using terms
   such as ``CellProfiler examples``, ``cppipe``, ``native OpenHCS examples``,
   ``BBBC021``, ``ExampleHuman``, ``production examples``, or
   ``preset pipelines``.
4. Run another ``knowledge-search <term>`` for the technical operator concept
   you need, such as ``GlobalPipelineConfig``, ``ObjectState``,
   ``custom function``, ``Napari``, ``FunctionStep``, or ``MaterializationSpec``.
5. Use the returned ``section_id`` with ``knowledge-document`` to read only the
   relevant section.
6. Follow source paths and capability names in the result before editing code.

Response Shape
--------------

The catalog response contains document summaries:

.. code-block:: json

   {
     "schema_version": "openhcs.agent.v1",
     "documents": [
       {
         "document_id": "openhcs_agent_mcp_overview",
         "title": "OpenHCS MCP agent knowledge base",
         "source_path": "docs/plans/openhcs_mcp_agent_knowledge_base_20260625.md",
         "tags": ["mcp", "agent", "architecture", "overview"]
       }
     ],
     "warnings": []
   }

Document reads include the bounded ``content`` plus a section list. The
``max_chars`` bound applies to returned document content; metadata remains
available so agents can choose follow-up section ids.

Current And Historical Documents
--------------------------------

The primary current overview is
``docs/plans/openhcs_mcp_agent_knowledge_base_20260625.md``. Domain-facing
guidance starts at ``docs/source/guide_for_biologists/domain_expert_onboarding.rst``.
Example-corpus guidance starts at ``docs/source/guides/example_corpus_map.rst``
and points agents to existing CellProfiler examples, native benchmark pipelines,
presets, complete examples, and production examples. Technical operator guidance
is allowlisted from the configuration, compiler, component, storage,
custom-function, and viewer architecture docs. The KB also allowlists older MCP
planning documents because they preserve design rationale; treat those plan
documents as historical evidence unless current capability registry output or
service code confirms the behavior.

Freshness
---------

Allowlisted documentation files are part of the MCP stale-source watchlist.
When a running MCP process has stale code or watched docs, normal tools and
resources report a structured stale-server payload. ``openhcs_health_check``
remains callable so a client can see which paths changed and restart the MCP
server.
