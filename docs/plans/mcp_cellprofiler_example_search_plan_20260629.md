# MCP CellProfiler Example Search Plan

Date: 2026-06-29

## Problem

Agents usually know CellProfiler. OpenHCS should make it obvious that
CellProfiler examples and `.cppipe` pipelines are useful, searchable evidence
for OpenHCS workflows.

The solution must not add a static MCP list of the official 30 pipelines or a
new CellProfiler module catalog. Existing docs, manifests, parser/generator
code, and module declarations are the authorities.

## Existing Authorities

- `docs/source/guides/example_corpus_map.rst`
- `docs/source/development/mcp_knowledge_base_manifest.json`
- `KnowledgeBaseService`
- official30 manifest under `benchmark/manifests`
- native CellProfiler reference artifacts under `benchmark/native_refs`
- CellProfiler parser/generator code
- CellProfiler module declarations
- native OpenHCS examples and preset pipelines

## Target Shape

Knowledge search should make examples easy to find:

```text
example corpus docs and manifests
    -> KnowledgeBaseService dynamic official30 projection
    -> searchable sections by case, module, assay, source data, cppipe path
    -> agent docs/renderers
```

Agents should be able to search:

- "nuclei segmentation CellProfiler example"
- "MeasureObjectSizeShape official30"
- "illumination correction cppipe"
- "native OpenHCS example source bindings"

and get source-backed documents or examples, not MCP-invented summaries.

## Nominal Iteration Authority

If implementation needs official30 cases, iterate the official30 manifest
through `KnowledgeBaseService`'s existing official30 projection helpers. Do not
copy the 30 case list into MCP or docs code.

If implementation needs CellProfiler module usage for examples, iterate module
inventories derived from the manifest's resolved `.cppipe` paths and existing
parser/generator/module declaration code. Do not maintain module inventories in
MCP.

If implementation needs native OpenHCS examples, iterate knowledge-base document
specs and source paths. Do not copy example source into the MCP layer.

## Implementation Dry Run

See `docs/plans/mcp_agent_experience_implementation_dry_run_20260629.md`.
The dry run confirmed `default_document_specs()` currently exposes both
`openhcs_official30_benchmark_recipes` and `openhcs_example_corpus_map`, and
`KnowledgeBaseService` already owns official30 helper methods.

## Implementation Steps

1. Audit current knowledge search for official30 and examples.
   - Confirm case names, module inventories, assay tags, source roots, and
     `.cppipe` paths are searchable.
2. Improve manifest tags and section generation.
   - Use the existing dynamic official30 rendering in `KnowledgeBaseService`.
   - Add tags to source docs if search misses common terms.
3. Add native OpenHCS example indexing if missing.
   - Use existing example docs/source projection paths.
   - Do not duplicate example lists in MCP.
4. Improve first-use and CellProfiler translation context.
   - Tell agents to search the example corpus before inventing pipelines.
5. Add tests for search recall.
   - Specific queries should return the example corpus map or official30 recipes.

## Mirror Traps To Avoid

- Do not add a hardcoded list of 30 pipelines in MCP.
- Do not duplicate CellProfiler module inventories outside existing parser,
  generator, and module declaration authorities.
- Do not copy `.cppipe` content into docs.
- Do not make search tags influence runtime semantics.

## Semantic Mirroring Audit

Audit questions:

- Do official30 case names, module inventories, and paths come from the manifest
  or existing native reference locations?
- Does search ranking stay inside `KnowledgeBaseService` and docs metadata?
- Do CellProfiler module semantics remain on parser/generator/module declaration
  authorities?
- Are native OpenHCS examples indexed by source path, not copied into MCP code?

Hard failures:

- MCP or agent code contains a hardcoded official30 pipeline list.
- MCP or agent code contains CellProfiler module inventory tables for example
  search.
- Docs copy full `.cppipe` contents instead of pointing to source artifacts.
- Knowledge tags are consumed by compiler/runtime behavior.

AST/rg audit:

```bash
rg -n "official30|CellProfiler examples|cppipe|native OpenHCS examples" docs/source/development/mcp_knowledge_base_manifest.json docs/source openhcs/agent/services/knowledge_base_service.py
rg -n "official30.*= \\{|CellProfiler.*pipelines.*= \\{|cppipe.*= \\[" openhcs/mcp openhcs/agent
rg -n "KnowledgeBaseService|_official30|example_corpus" openhcs/agent/services/knowledge_base_service.py docs/source
```

Allowed static values are manifest metadata, source paths, tags, and search
policy. They must not drive runtime translation or module behavior.

## Verification

Search gates:

```bash
rg -n "official30|CellProfiler examples|cppipe|native OpenHCS examples" docs/source/development/mcp_knowledge_base_manifest.json docs/source openhcs/agent/services/knowledge_base_service.py
rg -n "official30.*=|CellProfiler.*pipelines.*=" openhcs/mcp openhcs/agent
```

Expected result:

- Example lookup is source-backed through knowledge docs/manifests.
- MCP has no static official30 list.

Focused tests:

```bash
XDG_CACHE_HOME=/tmp/openhcs-test-cache .venv/bin/python -m pytest \
  tests/unit/agent/test_knowledge_base_service.py \
  tests/unit/agent/test_agent_services.py
```

Fresh MCP checks:

```bash
.venv/bin/python -m openhcs.mcp.dev_client knowledge-search "CellProfiler examples MeasureObjectSizeShape official30"
.venv/bin/python -m openhcs.mcp.dev_client knowledge-document openhcs_official30_benchmark_recipes
```
