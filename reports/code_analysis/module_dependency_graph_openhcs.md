# Module Dependency Graph for `openhcs`
Generated on: 2025-05-23 02:53:07

## `openhcs/__init__.py`
### No local dependencies found or file has issues.

## `openhcs/constants/__init__.py`
### Depends on:
- `openhcs.constants.constants`
- `openhcs.constants.clauses`

## `openhcs/constants/clauses.py`
### No local dependencies found or file has issues.

## `openhcs/constants/constants.py`
### No local dependencies found or file has issues.

## `openhcs/core/__init__.py`
### Depends on:
- `openhcs.core.context.processing_context`
- `openhcs.core.steps.abstract`

## `openhcs/core/config.py`
### Depends on:
- `openhcs.constants`

## `openhcs/core/context/__init__.py`
### Depends on:
- `openhcs.core.context.processing_context`

## `openhcs/core/context/processing_context.py`
### Depends on:
- `openhcs.core.config`

## `openhcs/core/exceptions.py`
### No local dependencies found or file has issues.

## `openhcs/core/memory/__init__.py`
### Depends on:
- `openhcs.core.memory.wrapper`
- `openhcs.constants.constants`
- `openhcs.core.memory.decorators`

## `openhcs/core/memory/conversion_functions.py`
### Depends on:
- `openhcs.core.memory.exceptions`
- `openhcs.core.memory.utils`
- `openhcs.constants.constants`

## `openhcs/core/memory/converters.py`
### Depends on:
- `openhcs.core.memory.conversion_functions`
- `openhcs.constants.constants`

## `openhcs/core/memory/decorators.py`
### Depends on:
- `openhcs.constants.constants`

## `openhcs/core/memory/exceptions.py`
### No local dependencies found or file has issues.

## `openhcs/core/memory/gpu_utils.py`
### Depends on:
- `openhcs.core.utils`

## `openhcs/core/memory/stack_utils.py`
### Depends on:
- `openhcs.core.memory`
- `openhcs.constants.constants`
- `openhcs.core.utils`

## `openhcs/core/memory/trackers/cupy_tracker.py`
### No local dependencies found or file has issues.

## `openhcs/core/memory/trackers/memory_tracker.py`
### No local dependencies found or file has issues.

## `openhcs/core/memory/trackers/memory_tracker_registry.py`
### No local dependencies found or file has issues.

## `openhcs/core/memory/trackers/numpy_tracker.py`
### No local dependencies found or file has issues.

## `openhcs/core/memory/trackers/tf_tracker.py`
### No local dependencies found or file has issues.

## `openhcs/core/memory/trackers/torch_tracker.py`
### Depends on:
- `openhcs.core.utils`

## `openhcs/core/memory/utils.py`
### Depends on:
- `openhcs.core.memory.exceptions`
- `openhcs.constants.constants`

## `openhcs/core/memory/wrapper.py`
### Depends on:
- `openhcs.core.memory.exceptions`
- `openhcs.core.memory.converters`
- `openhcs.core.memory.utils`
- `openhcs.constants.constants`

## `openhcs/core/orchestrator/__init__.py`
### Depends on:
- `openhcs.core.orchestrator.gpu_scheduler`
- `openhcs.core.orchestrator.orchestrator`

## `openhcs/core/orchestrator/gpu_scheduler.py`
### Depends on:
- `openhcs.core.config`
- `openhcs.core.memory.gpu_utils`

## `openhcs/core/orchestrator/orchestrator.py`
### Depends on:
- `openhcs.io.exceptions`
- `openhcs.io.filemanager`
- `openhcs.microscopes.microscope_interfaces`
- `openhcs.runtime.napari_stream_visualizer`
- `openhcs.core.pipeline.compiler`
- `openhcs.core.config`
- `openhcs.core.context.processing_context`
- `openhcs.io.base`
- `openhcs.core.steps.abstract`
- `openhcs.core.pipeline.step_attribute_stripper`
- `openhcs.constants.constants`

## `openhcs/core/pipeline/__init__.py`
### Depends on:
- `openhcs.core.pipeline.step_attribute_stripper`
- `openhcs.core.pipeline.funcstep_contract_validator`
- `openhcs.core.pipeline.materialization_flag_planner`
- `openhcs.core.pipeline.path_planner`
- `openhcs.core.pipeline.pipeline`
- `openhcs.constants.constants`

## `openhcs/core/pipeline/compiler.py`
### Depends on:
- `openhcs.core.steps.function_step`
- `openhcs.core.pipeline.gpu_memory_validator`
- `openhcs.core.pipeline.funcstep_contract_validator`
- `openhcs.core.context.processing_context`
- `openhcs.core.steps.abstract`
- `openhcs.core.pipeline.materialization_flag_planner`
- `openhcs.core.pipeline.path_planner`
- `openhcs.constants.constants`

## `openhcs/core/pipeline/executor.py`
### Depends on:
- `openhcs.core.context.processing_context`
- `openhcs.core.steps.abstract`
- `openhcs.constants.constants`

## `openhcs/core/pipeline/funcstep_contract_validator.py`
### Depends on:
- `openhcs.core.steps.function_step`
- `openhcs.constants.constants`

## `openhcs/core/pipeline/function_contracts.py`
### No local dependencies found or file has issues.

## `openhcs/core/pipeline/gpu_memory_validator.py`
### Depends on:
- `openhcs.core.orchestrator.gpu_scheduler`
- `openhcs.constants.constants`
- `openhcs.core.utils`

## `openhcs/core/pipeline/materialization_flag_planner.py`
### Depends on:
- `openhcs.core.context.processing_context`
- `openhcs.core.steps.function_step`
- `openhcs.core.steps.abstract`
- `openhcs.constants.constants`

## `openhcs/core/pipeline/path_planner.py`
### Depends on:
- `openhcs.core.pipeline.pipeline_utils`
- `openhcs.core.context.processing_context`
- `openhcs.core.steps.function_step`
- `openhcs.core.steps.abstract`

## `openhcs/core/pipeline/pipeline.py`
### Depends on:
- `openhcs.core.pipeline.executor`
- `openhcs.core.pipeline.compiler`

## `openhcs/core/pipeline/pipeline_factories.py`
### Depends on:
- `openhcs.core.pipeline`
- `openhcs.core.steps`

## `openhcs/core/pipeline/pipeline_utils.py`
### No local dependencies found or file has issues.

## `openhcs/core/pipeline/step_attribute_stripper.py`
### No local dependencies found or file has issues.

## `openhcs/core/steps/__init__.py`
### Depends on:
- `openhcs.core.steps.function_step`
- `openhcs.core.steps.abstract`

## `openhcs/core/steps/abstract.py`
### Depends on:
- `openhcs.core.context.processing_context`

## `openhcs/core/steps/function_step.py`
### Depends on:
- `openhcs.core.memory.stack_utils`
- `openhcs.formats.func_arg_prep`
- `openhcs.core.context.processing_context`
- `openhcs.core.steps.abstract`
- `openhcs.constants.constants`

## `openhcs/core/steps/specialized/__init__.py`
### Depends on:
- `openhcs.core.steps.specialized.composite_step`
- `openhcs.core.steps.specialized.norm_step`
- `openhcs.core.steps.specialized.focus_step`
- `openhcs.core.steps.specialized.zflat_step`

## `openhcs/core/steps/specialized/composite_step.py`
### Depends on:
- `openhcs.core.steps.function_step`

## `openhcs/core/steps/specialized/focus_step.py`
### Depends on:
- `openhcs.processing.backends.enhance.dl_edof_unsupervised`
- `openhcs.core.steps.function_step`

## `openhcs/core/steps/specialized/norm_step.py`
### Depends on:
- `openhcs.core.steps.function_step`

## `openhcs/core/steps/specialized/zflat_step.py`
### Depends on:
- `openhcs.core.steps.function_step`
- `openhcs.processing.backends.processors.numpy_processor`

## `openhcs/core/utils.py`
### No local dependencies found or file has issues.

## `openhcs/core/validation/__init__.py`
### No local dependencies found or file has issues.

## `openhcs/formats/func_arg_prep.py`
### No local dependencies found or file has issues.

## `openhcs/formats/pattern/__init__.py`
### No local dependencies found or file has issues.

## `openhcs/formats/pattern/pattern_discovery.py`
### Depends on:
- `openhcs.io.filemanager`
- `openhcs.constants.constants`
- `openhcs.microscopes.microscope_interfaces_base`

## `openhcs/formats/pattern/pattern_resolver.py`
### No local dependencies found or file has issues.

## `openhcs/formats/position_format.py`
### No local dependencies found or file has issues.

## `openhcs/io/__init__.py`
### Depends on:
- `openhcs.io.filemanager`
- `openhcs.io.zarr`
- `openhcs.io.base`
- `openhcs.io.memory`
- `openhcs.io.disk`

## `openhcs/io/base.py`
### Depends on:
- `openhcs.io.memory`
- `openhcs.io.zarr`
- `openhcs.io.disk`
- `openhcs.constants.constants`

## `openhcs/io/disk.py`
### Depends on:
- `openhcs.io.base`
- `openhcs.constants.constants`

## `openhcs/io/exceptions.py`
### No local dependencies found or file has issues.

## `openhcs/io/filemanager.py`
### Depends on:
- `openhcs.io.exceptions`
- `openhcs.validation`
- `openhcs.io.base`
- `openhcs.constants.constants`

## `openhcs/io/memory.py`
### Depends on:
- `openhcs.io.base`

## `openhcs/io/zarr.py`
### Depends on:
- `openhcs.io.base`

## `openhcs/microscopes/__init__.py`
### Depends on:
- `openhcs.microscopes.imagexpress`
- `openhcs.microscopes.opera_phenix`

## `openhcs/microscopes/imagexpress.py`
### Depends on:
- `openhcs.io.filemanager`
- `openhcs.io.exceptions`
- `openhcs.microscopes.microscope_interfaces_base`
- `openhcs.microscopes.microscope_base`
- `openhcs.constants.constants`

## `openhcs/microscopes/microscope_base.py`
### Depends on:
- `openhcs.io.filemanager`
- `openhcs.formats.pattern.pattern_discovery`
- `openhcs.constants.constants`
- `openhcs.microscopes.microscope_interfaces_base`

## `openhcs/microscopes/microscope_interfaces.py`
### Depends on:
- `openhcs.io.filemanager`
- `openhcs.microscopes.opera_phenix`
- `openhcs.microscopes.imagexpress`
- `openhcs.microscopes.microscope_base`
- `openhcs.constants.constants`

## `openhcs/microscopes/microscope_interfaces_base.py`
### No local dependencies found or file has issues.

## `openhcs/microscopes/opera_phenix.py`
### Depends on:
- `openhcs.microscopes.microscope_base`
- `openhcs.io.filemanager`
- `openhcs.constants.constants`
- `openhcs.microscopes.microscope_interfaces_base`

## `openhcs/microscopes/opera_phenix_xml_parser.py`
### No local dependencies found or file has issues.

## `openhcs/processing/__init__.py`
### Depends on:
- `openhcs.processing.function_registry`
- `openhcs.processing.backends`
- `openhcs.processing.func_registry`

## `openhcs/processing/backends/__init__.py`
### No local dependencies found or file has issues.

## `openhcs/processing/backends/analysis/__init__.py`
### Depends on:
- `openhcs.processing.backends.analysis.focus_analyzer`
- `openhcs.processing.backends.analysis.dxf_mask_pipeline`

## `openhcs/processing/backends/analysis/dxf_mask_pipeline.py`
### Depends on:
- `openhcs.utils.import_utils`
- `openhcs.core.memory.decorators`

## `openhcs/processing/backends/analysis/focus_analyzer.py`
### No local dependencies found or file has issues.

## `openhcs/processing/backends/analysis/rrs_vectorized_tracer.py`
### Depends on:
- `openhcs.core.memory.decorators`
- `openhcs.core.utils`

## `openhcs/processing/backends/analysis/self_supervised_segmentation_3d.py`
### Depends on:
- `openhcs.core.memory.decorators`
- `openhcs.core.utils`

## `openhcs/processing/backends/analysis/straighten_object_3d.py`
### Depends on:
- `openhcs.core.memory.decorators`
- `openhcs.core.utils`

## `openhcs/processing/backends/assemblers/__init__.py`
### Depends on:
- `openhcs.processing.backends.assemblers.assemble_stack_cupy`
- `openhcs.processing.backends.assemblers.assemble_stack_cpu`

## `openhcs/processing/backends/assemblers/assemble_stack_cpu.py`
### Depends on:
- `openhcs.core.pipeline.function_contracts`
- `openhcs.core.memory.decorators`

## `openhcs/processing/backends/assemblers/assemble_stack_cupy.py`
### Depends on:
- `openhcs.core.utils`
- `openhcs.core.pipeline.function_contracts`
- `openhcs.core.memory.decorators`

## `openhcs/processing/backends/assemblers/self_supervised_stitcher.py`
### Depends on:
- `openhcs.core.memory.decorators`
- `openhcs.core.utils`

## `openhcs/processing/backends/enhance/__init__.py`
### Depends on:
- `openhcs.processing.backends.enhance.n2v2_processor_torch`
- `openhcs.processing.backends.enhance.basic_processor_cupy`
- `openhcs.processing.backends.enhance.basic_processor_numpy`

## `openhcs/processing/backends/enhance/basic_processor_cupy.py`
### Depends on:
- `openhcs.core.utils`
- `openhcs.core.memory.decorators`

## `openhcs/processing/backends/enhance/basic_processor_numpy.py`
### Depends on:
- `openhcs.core.memory`

## `openhcs/processing/backends/enhance/dl_edof_unsupervised.py`
### Depends on:
- `openhcs.core.utils`
- `openhcs.core.memory.decorators`

## `openhcs/processing/backends/enhance/focus_torch.py`
### Depends on:
- `openhcs.core.memory.decorators`
- `openhcs.core.utils`

## `openhcs/processing/backends/enhance/n2v2_processor_torch.py`
### Depends on:
- `openhcs.utils.import_utils`
- `openhcs.core.memory.decorators`

## `openhcs/processing/backends/enhance/self_supervised_3d_deconvolution.py`
### Depends on:
- `openhcs.core.memory.decorators`
- `openhcs.core.utils`

## `openhcs/processing/backends/pos_gen/__init__.py`
### Depends on:
- `openhcs.processing.backends.pos_gen.ashlar_processor_cupy`
- `openhcs.processing.backends.pos_gen.mist_processor_cupy`

## `openhcs/processing/backends/pos_gen/ashlar_processor_cupy.py`
### Depends on:
- `openhcs.core.utils`
- `openhcs.core.pipeline.function_contracts`
- `openhcs.core.memory.decorators`

## `openhcs/processing/backends/pos_gen/mist_processor_cupy.py`
### Depends on:
- `openhcs.core.utils`
- `openhcs.core.pipeline.function_contracts`
- `openhcs.constants.constants`
- `openhcs.core.memory.decorators`

## `openhcs/processing/backends/processors/cupy_processor.py`
### Depends on:
- `openhcs.core.utils`
- `openhcs.core.memory.decorators`

## `openhcs/processing/backends/processors/jax_processor.py`
### Depends on:
- `openhcs.core.utils`
- `openhcs.core.memory.decorators`

## `openhcs/processing/backends/processors/numpy_processor.py`
### Depends on:
- `openhcs.core.memory.decorators`

## `openhcs/processing/backends/processors/tensorflow_processor.py`
### Depends on:
- `openhcs.core.utils`
- `openhcs.core.memory.decorators`

## `openhcs/processing/backends/processors/torch_processor.py`
### Depends on:
- `openhcs.core.memory.decorators`
- `openhcs.core.utils`

## `openhcs/processing/func_registry.py`
### Depends on:
- `openhcs`

## `openhcs/processing/function_registry.py`
### Depends on:
- `openhcs.processing.func_registry`
- `openhcs.constants.constants`
- `openhcs.core.memory.decorators`

## `openhcs/processing/registry_base.py`
### No local dependencies found or file has issues.

## `openhcs/runtime/napari_stream_visualizer.py`
### Depends on:
- `openhcs.io.filemanager`

## `openhcs/tests/__init__.py`
### No local dependencies found or file has issues.

## `openhcs/tests/generators/__init__.py`
### No local dependencies found or file has issues.

## `openhcs/tests/generators/generate_synthetic_data.py`
### No local dependencies found or file has issues.

## `openhcs/tests/helpers/unsafe_registry.py`
### Depends on:
- `openhcs.core.exceptions`

## `openhcs/tui/__init__.py`
### Depends on:
- `openhcs.tui.function_pattern_editor`

## `openhcs/tui/__main__.py`
### Depends on:
- `openhcs.tui.tui_launcher`
- `openhcs.core.config`
- `openhcs.core.orchestrator.gpu_scheduler`

## `openhcs/tui/commands.py`
### Depends on:
- `openhcs.core.steps.function_step`
- `openhcs.core.orchestrator.orchestrator`
- `openhcs.tui.dialogs.global_settings_editor`
- `openhcs.tui.dialogs.plate_dialog_manager`
- `openhcs.tui.tui_architecture`
- `openhcs.core.config`
- `openhcs.core.context.processing_context`
- `openhcs.tui.utils`
- `openhcs.core.steps.abstract`
- `openhcs.constants.constants`

## `openhcs/tui/components.py`
### No local dependencies found or file has issues.

## `openhcs/tui/dialogs/global_settings_editor.py`
### Depends on:
- `openhcs.core.config`
- `openhcs.constants.constants`

## `openhcs/tui/dialogs/plate_config_editor.py`
### Depends on:
- `openhcs.tui.tui_architecture`
- `openhcs.core.config`
- `openhcs.core.orchestrator.orchestrator`
- `openhcs.constants`

## `openhcs/tui/dialogs/plate_dialog_manager.py`
### Depends on:
- `openhcs.tui.file_browser`
- `openhcs.io.filemanager`
- `openhcs.constants.constants`

## `openhcs/tui/dual_step_func_editor.py`
### Depends on:
- `openhcs.core.steps.function_step`
- `openhcs.tui.function_pattern_editor`
- `openhcs.tui.utils`
- `openhcs.core.steps.abstract`
- `openhcs.constants.constants`

## `openhcs/tui/file_browser.py`
### Depends on:
- `openhcs.io.filemanager`
- `openhcs.constants.constants`

## `openhcs/tui/function_pattern_editor.py`
### Depends on:
- `openhcs.processing.func_registry`
- `openhcs.tui.components`
- `openhcs.core.pipeline.funcstep_contract_validator`
- `openhcs.tui.services.external_editor_service`
- `openhcs.tui.utils`
- `openhcs.constants.constants`

## `openhcs/tui/menu_bar.py`
### Depends on:
- `openhcs.core.orchestrator.orchestrator`
- `openhcs.tui.tui_architecture`
- `openhcs.core.config`
- `openhcs.core.context.processing_context`
- `openhcs.core.steps.abstract`
- `openhcs.tui.commands`

## `openhcs/tui/plate_manager_core.py`
### Depends on:
- `openhcs.io.filemanager`
- `openhcs.core.orchestrator.orchestrator`
- `openhcs.tui.dialogs.plate_dialog_manager`
- `openhcs.tui.components`
- `openhcs.tui.status_bar`
- `openhcs.core.context.processing_context`
- `openhcs.io.base`
- `openhcs.tui.services.plate_validation`
- `openhcs.constants.constants`

## `openhcs/tui/services/external_editor_service.py`
### Depends on:
- `openhcs.core.pipeline.funcstep_contract_validator`

## `openhcs/tui/services/plate_validation.py`
### Depends on:
- `openhcs.core.context.processing_context`
- `openhcs.io.filemanager`

## `openhcs/tui/status_bar.py`
### No local dependencies found or file has issues.

## `openhcs/tui/step_viewer.py`
### Depends on:
- `openhcs.core.steps.function_step`
- `openhcs.tui.components`
- `openhcs.core.context.processing_context`
- `openhcs.tui.utils`
- `openhcs.core.steps.abstract`

## `openhcs/tui/tui_architecture.py`
### Depends on:
- `openhcs.tui.plate_manager_core`
- `openhcs.io.filemanager`
- `openhcs.core.steps.function_step`
- `openhcs.core.orchestrator.orchestrator`
- `openhcs.processing.func_registry`
- `openhcs.tui.function_pattern_editor`
- `openhcs.core.config`
- `openhcs.tui.dual_step_func_editor`
- `openhcs.tui.status_bar`
- `openhcs.tui.menu_bar`
- `openhcs.io.base`
- `openhcs.core.context.processing_context`
- `openhcs.tui.step_viewer`
- `openhcs.tui.dialogs.plate_config_editor`

## `openhcs/tui/tui_launcher.py`
### Depends on:
- `openhcs.io.filemanager`
- `openhcs.core.orchestrator.orchestrator`
- `openhcs.tui.tui_architecture`
- `openhcs.core.config`
- `openhcs.core.context.processing_context`
- `openhcs.io.base`

## `openhcs/tui/utils.py`
### No local dependencies found or file has issues.

## `openhcs/utils/import_utils.py`
### No local dependencies found or file has issues.

## `openhcs/validation/__init__.py`
### Depends on:
- `openhcs.validation.ast_validator`

## `openhcs/validation/ast_validator.py`
### No local dependencies found or file has issues.

## `openhcs/validation/validate.py`
### Depends on:
- `openhcs.validation.ast_validator`

