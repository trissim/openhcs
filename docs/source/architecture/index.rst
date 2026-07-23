Architecture reference
======================

This section documents the current OpenHCS-owned architecture. Generic behavior
provided by a first-party dependency is documented by that package; OpenHCS
pages describe only the integration boundary and cross-package invariants.

Start with :doc:`quick_start`. It routes desktop, Python, CellProfiler, and MCP
users into the same declaration-to-runtime model. Continue with
:doc:`system_overview` for the expanded system path.

Architecture front door
-----------------------

.. toctree::
   :maxdepth: 1

   quick_start

Declaration and compilation
---------------------------

.. toctree::
   :maxdepth: 1

   system_overview
   nominal_ownership
   abstraction_lattices
   processing_semantics
   source_model
   artifact_contract_system
   pipeline_compilation_system

Runtime
-------

.. toctree::
   :maxdepth: 1

   runtime_value_system
   measurement_equivalence_system
   progress_runtime_projection_system
   concurrency_model
   orchestrator_cleanup_guarantees

Interop and application boundaries
----------------------------------

.. toctree::
   :maxdepth: 1

   cellprofiler_interop
   external_foundations
   streaming_boundary_and_wrappers
   external_integrations_overview
   microscope_handler_integration
   code_ui_interconversion
   serialization_boundaries
   plate_manager_services
   batch_workflow_service
   zmq_server_browser_system
   mcp_distribution

Specialized subsystems
----------------------

.. toctree::
   :maxdepth: 1

   gpu_resource_management
   multiprocessing_coordination_system
   analysis_consolidation_system
   experimental_analysis_system
   component_validation_system
   component_system_integration
   orchestrator_configuration_management

Documentation status
--------------------

Older pages that describe a fixed five-phase compiler, string-keyed step plans,
generated CellProfiler semantic sidecars, OpenHCS-owned generic configuration or
UI internals, or the deprecated TUI are intentionally absent from this
navigation. Their durable content has been moved to the owning package or
replaced by a transition page, and the originals are archived as migration
history. The completed disposition is recorded in
``docs/plans/documentation_overhaul_disposition_20260717.md``.

.. toctree::
   :hidden:
   :caption: Transition URLs

   compilation_system_detailed
   compilation_service
   configuration_framework
   context_system
   dynamic_dataclass_factory
   component_configuration_framework
   memory_type_system
   storage_and_memory_system
   roi_system
   plugin_registry_system
   plugin_registry_advanced
   function_registry_system
   function_pattern_system
   function_reference_pattern
   custom_function_registration_system
   parser_metaprogramming_system
   component_processor_metaprogramming
   pattern_detection_system
   dict_pattern_case_study
   special_io_system
   pattern_grouping_and_special_outputs
   abstract_manager_widget
   abstract_table_browser
   cross_window_update_optimization
   declarative_window_system
   field_change_dispatcher
   flash_animation_system
   gui_performance_patterns
   list_item_preview_system
   parameter_form_lifecycle
   parameter_form_service_architecture
   parametric_widget_creation
   scope_visual_feedback_system
   scope_window_factory_system
   service-layer-architecture
   service_registry_integration
   step-editor-generalization
   ui_services_architecture
   widget_protocol_system
   system_integration
   tui_system
   zmq_execution_service_extracted
   fiji_streaming_system
   napari_integration_architecture
   napari_streaming_system
   omero_backend_system

Architecture invariants
-----------------------

- Public pipeline declarations are ``PipelineConfig`` plus
  ``list[FunctionStep]``.
- ObjectState resolves declaration configuration once before compiler stages
  consume it.
- ``CompilationSession`` and typed ``CompiledStepPlan`` fields are compiler
  authorities; string-keyed semantic dictionaries are not.
- Callable, module, artifact, source, measurement, and strategy declarations own
  their respective semantics.
- Generic consumers query nominal registries or shared strategy mixins and do
  not import concrete backends to discover names or behavior.
- ``variable_components``, ``group_by``, and ``ProcessingContract`` are
  independent declarations with different meanings.
- Runtime workers consume a ``CompiledExecutionBundle`` and validated typed
  runtime values; they do not reconstruct semantic contracts from sidecars.
