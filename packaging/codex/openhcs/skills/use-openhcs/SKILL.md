---
name: use-openhcs
description: Operate local OpenHCS microscopy workflows through the bundled MCP server. Use when Codex needs to inspect plate data, discover processing functions, author or validate pipelines, compile and execute jobs, control viewers, or interact with a running OpenHCS GUI.
---

# Use OpenHCS

1. Call `openhcs_health_check` before relying on other tools. Report a bootstrap or stale-server failure instead of working around it.
2. If you do not already know OpenHCS, call `openhcs_get_authoring_context` with `kind="first_use"`. Retain its execution, step-pattern, `variable_components`/`group_by`, artifact, and CellProfiler mental model while completing the task.
3. Treat Python pipeline source as one complete `PipelineDocument`: `pipeline_config` plus ordered `pipeline_steps`. Never send a steps-only fragment, mirror configuration through a second argument, or strip source bindings from the reviewed document. UI-visible and headless routes use these same declarations even though their process and state owners differ.
4. Call `openhcs_search_capabilities` with task-relevant workflow, target, or text filters. Its current `surface_profile`, registry-owned workflow metadata, side effects, and security metadata—not a remembered tool list—decide whether to use a UI-visible or exposed headless route. Use `openhcs_list_capabilities` only when the complete selected surface is required.
5. Search knowledge and examples before authoring. For CellProfiler or benchmark work, search the biological task plus `OpenHCS Python` and retrieve the returned Official30 section with `max_chars=50000`. CellProfiler image, object, measurement, relationship, and export semantics lower into the same OpenHCS declarations and runtime.
6. Inspect real plate data and registered function declarations before authoring. If the microscope is unsupported, use typed pipeline-level `SourceBindingsConfig` declarations to filter files, extract metadata, name semantic sources, and project a virtual workspace; make each consuming `FunctionStep` select those aliases through its step-local source bindings. Never parse filenames inside processing functions. Reflect `global` or `pipeline` configuration with `openhcs_describe_config_schema` before setting non-obvious fields. Keep filesystem operations inside configured read and write roots.
7. Validate and compile before execution. Do not infer that source code, UI state, or an earlier validation result implies a current compiled plan.
8. Start read-only. Use capability-registry metadata as the authority for mutation and exposure; before mutation, execution, UI actions, viewer launch, network use, or external data exposure, show the target/change, obtain approval, and refresh revision or request tokens.
9. Preserve the active ownership route. Discover the GUI bridge before UI tools and apply code/state changes with current tokens. Use headless tools only when their workflow group is exposed. Treat structured errors and recovery hints as authoritative; never bypass path policy, stale-process checks, compile requirements, or bridge authentication.
