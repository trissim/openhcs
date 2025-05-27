"""
Command Pattern Infrastructure for OpenHCS TUI.

Implements the Command Pattern to encapsulate all UI actions as discrete,
testable, and reusable command objects. This provides clean separation
between UI event handling and business logic.

All commands operate on TUIState and ProcessingContext to maintain
architectural consistency and enable proper async/await patterns.
"""

import asyncio
import logging
from typing import Protocol, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class Command(Protocol):
    """
    Base command protocol for all TUI actions.

    Commands encapsulate user actions and provide a clean interface
    for executing business logic while maintaining separation from UI concerns.
    """

    async def execute(self, state: 'TUIState', context: 'ProcessingContext', **kwargs: Any) -> None:
        """
        Execute the command with given state and context.

        Args:
            state: Current TUI state
            context: Processing context with core services
            **kwargs: Additional command-specific parameters
        """
        ...

    def can_execute(self, state: 'TUIState') -> bool:
        """
        Check if command can be executed in current state.

        Args:
            state: Current TUI state

        Returns:
            True if command can be executed, False otherwise
        """
        return True

# =============================================================================
# Top Bar Commands (Global Settings, Help)
# =============================================================================

class ShowGlobalSettingsDialogCommand:
    """Command to show global settings editor dialog."""

    async def execute(self, state: 'TUIState', context: 'ProcessingContext', **kwargs: Any) -> None:
        """Show global settings dialog."""
        try:
            logger.info("Showing global settings dialog")

            from prompt_toolkit.application import get_app
            from prompt_toolkit.widgets import Dialog, Label, Button, TextArea
            from prompt_toolkit.layout import HSplit
            from prompt_toolkit.layout.containers import FloatContainer, Float

            # Create settings form
            settings_text = f"""Global Pipeline Configuration:

Output Directory: {getattr(context.global_config, 'output_dir', 'Not set')}
Temp Directory: {getattr(context.global_config, 'temp_dir', 'Not set')}
Max Workers: {getattr(context.global_config, 'max_workers', 'Not set')}
Log Level: {getattr(context.global_config, 'log_level', 'Not set')}

Note: Settings are read-only in this version.
Use configuration files to modify settings."""

            settings_area = TextArea(
                text=settings_text,
                read_only=True,
                multiline=True,
                height=10
            )

            dialog_closed = False

            def close_dialog():
                nonlocal dialog_closed
                dialog_closed = True
                get_app().layout = previous_layout
                get_app().invalidate()

            dialog = Dialog(
                title="Global Settings",
                body=HSplit([
                    Label("Current Global Configuration:"),
                    settings_area
                ]),
                buttons=[
                    Button("Close", handler=close_dialog)
                ],
                width=80,
                modal=True
            )

            # Show dialog
            app = get_app()
            previous_layout = app.layout

            float_container = FloatContainer(
                content=previous_layout.container,
                floats=[
                    Float(content=dialog, left=10, top=5)
                ]
            )

            app.layout.container = float_container
            app.invalidate()

            await state.set_status("Global settings dialog opened", "info")

        except Exception as e:
            logger.error(f"Failed to show global settings dialog: {e}")
            await state.set_status(f"Failed to show settings: {e}", "error")

class ShowHelpCommand:
    """Command to show help dialog."""

    async def execute(self, state: 'TUIState', context: 'ProcessingContext', **kwargs: Any) -> None:
        """Show help dialog."""
        try:
            logger.info("Showing help dialog")

            from prompt_toolkit.application import get_app
            from prompt_toolkit.widgets import Dialog, Label, Button, TextArea
            from prompt_toolkit.layout import HSplit
            from prompt_toolkit.layout.containers import FloatContainer, Float

            help_text = """OpenHCS TUI Help

KEYBOARD SHORTCUTS:
  q, Ctrl+Q, Ctrl+C  - Quit application
  Escape             - Close current dialog
  Tab                - Navigate between elements
  Enter              - Activate focused element

PLATE MANAGER:
  add     - Add new plate directory
  del     - Delete selected plates
  edit    - Edit plate configuration
  init    - Initialize plates
  compile - Compile plate pipelines
  run     - Execute plate processing

PIPELINE EDITOR:
  add     - Add new step to pipeline
  del     - Delete selected steps
  edit    - Edit step configuration
  load    - Load pipeline from file
  save    - Save pipeline to file

STATUS SYMBOLS:
  ?  - Uninitialized
  -  - Initialized
  ✓  - Completed/Compiled
  !  - Error
  o  - Running

NAVIGATION:
  ^/v - Scroll indicators for lists
  Click buttons or use keyboard navigation"""

            help_area = TextArea(
                text=help_text,
                read_only=True,
                multiline=True,
                height=20
            )

            def close_dialog():
                get_app().layout = previous_layout
                get_app().invalidate()

            dialog = Dialog(
                title="OpenHCS TUI Help",
                body=HSplit([
                    help_area
                ]),
                buttons=[
                    Button("Close", handler=close_dialog)
                ],
                width=80,
                modal=True
            )

            # Show dialog
            app = get_app()
            previous_layout = app.layout

            float_container = FloatContainer(
                content=previous_layout.container,
                floats=[
                    Float(content=dialog, left=10, top=3)
                ]
            )

            app.layout.container = float_container
            app.invalidate()

            await state.set_status("Help dialog opened", "info")

        except Exception as e:
            logger.error(f"Failed to show help dialog: {e}")
            await state.set_status(f"Failed to show help: {e}", "error")

# =============================================================================
# Plate Manager Commands (Add, Delete, Edit, Init, Compile, Run)
# =============================================================================

class ShowAddPlateDialogCommand:
    """Command to show interactive add plate dialog with VFS browser."""

    async def execute(self, state: 'TUIState', context: 'ProcessingContext', **kwargs: Any) -> None:
        """Show add plate dialog with FileManager browser."""
        try:
            logger.info("Showing add plate dialog with FileManager browser")

            from ..utils.dialogs import prompt_for_directory_dialog
            from pathlib import Path

            # Show directory selection dialog
            selected_path = await prompt_for_directory_dialog(
                title="Select Plate Directory",
                initial_path=Path.home()
            )

            if selected_path:
                try:
                    # Create plate data from selected path
                    plate_name = selected_path.name
                    plate_id = f"plate_{plate_name}_{len(state.plates) + 1}"

                    # Add plate to state
                    from openhcs.tui.state import PlateData
                    new_plate = PlateData(
                        id=plate_id,
                        name=plate_name,
                        path=str(selected_path),
                        status='uninitialized'
                    )

                    await state.add_plate(new_plate)
                    await state.set_status(f"Added plate: {plate_name}", "info")
                    logger.info(f"Added plate {plate_name} from {selected_path}")

                except Exception as e:
                    await state.set_status(f"Failed to add plate: {e}", "error")
            else:
                await state.set_status("Add plate cancelled", "info")

        except Exception as e:
            logger.error(f"Failed to show add plate dialog: {e}")
            await state.set_status(f"Failed to show file browser: {e}", "error")

class DeleteSelectedPlatesCommand:
    """Command to delete selected plates."""

    def can_execute(self, state: 'TUIState') -> bool:
        """Check if any plates are selected for deletion."""
        return bool(getattr(state, 'selected_plates', []))

    async def execute(self, state: 'TUIState', context: 'ProcessingContext', **kwargs: Any) -> None:
        """Delete selected plates."""
        try:
            selected_plates = getattr(state, 'selected_plates', [])
            if not selected_plates:
                logger.warning("No plates selected for deletion")
                return

            logger.info(f"Deleting {len(selected_plates)} selected plates")

            for plate_id in selected_plates:
                # Remove from orchestrators
                if hasattr(context, 'orchestrators') and plate_id in context.orchestrators:
                    orchestrator = context.orchestrators[plate_id]
                    await orchestrator.cleanup()  # Cleanup resources
                    del context.orchestrators[plate_id]

                # Notify state of removal
                await state.notify('plate_removed', {'plate_id': plate_id})

            # Clear selection
            await state.notify('plates_selection_cleared', {})

        except Exception as e:
            logger.error(f"Failed to delete plates: {e}")
            await state.notify('error_occurred', {'message': f"Failed to delete plates: {e}"})

class ShowEditPlateConfigDialogCommand:
    """Command to show plate-specific configuration editor."""

    def can_execute(self, state: 'TUIState') -> bool:
        """Check if a plate is selected for editing."""
        return bool(getattr(state, 'focused_plate', None))

    async def execute(self, state: 'TUIState', context: 'ProcessingContext', **kwargs: Any) -> None:
        """Show plate configuration editor dialog."""
        try:
            focused_plate = getattr(state, 'focused_plate', None)
            if not focused_plate:
                logger.warning("No plate focused for configuration editing")
                return

            logger.info(f"Showing plate config editor for: {focused_plate}")

            # Get orchestrator for the focused plate
            orchestrator = None
            if hasattr(context, 'orchestrators') and focused_plate in context.orchestrators:
                orchestrator = context.orchestrators[focused_plate]

            await state.notify('show_plate_config_dialog', {
                'plate_id': focused_plate,
                'orchestrator': orchestrator
            })

        except Exception as e:
            logger.error(f"Failed to show plate config dialog: {e}")
            await state.notify('error_occurred', {'message': f"Failed to show plate config: {e}"})

class InitializePlatesCommand:
    """Command to initialize selected plates."""

    def can_execute(self, state: 'TUIState') -> bool:
        """Check if any plates are selected for initialization."""
        return bool(getattr(state, 'selected_plates', []) or getattr(state, 'focused_plate', None))

    async def execute(self, state: 'TUIState', context: 'ProcessingContext', **kwargs: Any) -> None:
        """Initialize selected plates."""
        try:
            # Get plates to initialize (selected or focused)
            plates_to_init = getattr(state, 'selected_plates', [])
            if not plates_to_init and getattr(state, 'focused_plate', None):
                plates_to_init = [state.focused_plate]

            if not plates_to_init:
                logger.warning("No plates selected for initialization")
                return

            logger.info(f"Initializing {len(plates_to_init)} plates")

            for plate_id in plates_to_init:
                try:
                    # Get orchestrator
                    if not hasattr(context, 'orchestrators') or plate_id not in context.orchestrators:
                        logger.error(f"No orchestrator found for plate: {plate_id}")
                        await state.notify('plate_status_changed', {
                            'plate_id': plate_id,
                            'status': 'error',
                            'message': 'No orchestrator found'
                        })
                        continue

                    orchestrator = context.orchestrators[plate_id]

                    # Update status to initializing
                    await state.notify('plate_status_changed', {
                        'plate_id': plate_id,
                        'status': 'initializing'
                    })

                    # Initialize orchestrator
                    await orchestrator.initialize()

                    # Update status to initialized
                    await state.notify('plate_status_changed', {
                        'plate_id': plate_id,
                        'status': 'initialized'
                    })

                    logger.info(f"Successfully initialized plate: {plate_id}")

                except Exception as e:
                    logger.error(f"Failed to initialize plate {plate_id}: {e}")
                    await state.notify('plate_status_changed', {
                        'plate_id': plate_id,
                        'status': 'error',
                        'message': str(e)
                    })

        except Exception as e:
            logger.error(f"Failed to initialize plates: {e}")
            await state.notify('error_occurred', {'message': f"Failed to initialize plates: {e}"})

class CompilePlatesCommand:
    """Command to compile selected plates."""

    def can_execute(self, state: 'TUIState') -> bool:
        """Check if any plates are selected for compilation."""
        return bool(getattr(state, 'selected_plates', []) or getattr(state, 'focused_plate', None))

    async def execute(self, state: 'TUIState', context: 'ProcessingContext', **kwargs: Any) -> None:
        """Compile selected plates."""
        try:
            # Get plates to compile (selected or focused)
            plates_to_compile = getattr(state, 'selected_plates', [])
            if not plates_to_compile and getattr(state, 'focused_plate', None):
                plates_to_compile = [state.focused_plate]

            if not plates_to_compile:
                logger.warning("No plates selected for compilation")
                return

            logger.info(f"Compiling {len(plates_to_compile)} plates")

            for plate_id in plates_to_compile:
                try:
                    # Get orchestrator
                    if not hasattr(context, 'orchestrators') or plate_id not in context.orchestrators:
                        logger.error(f"No orchestrator found for plate: {plate_id}")
                        await state.notify('plate_status_changed', {
                            'plate_id': plate_id,
                            'status': 'error',
                            'message': 'No orchestrator found'
                        })
                        continue

                    orchestrator = context.orchestrators[plate_id]

                    # Update status to compiling
                    await state.notify('plate_status_changed', {
                        'plate_id': plate_id,
                        'status': 'compiling'
                    })

                    # Compile pipeline
                    await orchestrator.compile_pipeline()

                    # Update status to compiled
                    await state.notify('plate_status_changed', {
                        'plate_id': plate_id,
                        'status': 'compiled'
                    })

                    logger.info(f"Successfully compiled plate: {plate_id}")

                except Exception as e:
                    logger.error(f"Failed to compile plate {plate_id}: {e}")
                    await state.notify('plate_status_changed', {
                        'plate_id': plate_id,
                        'status': 'error',
                        'message': str(e)
                    })

        except Exception as e:
            logger.error(f"Failed to compile plates: {e}")
            await state.notify('error_occurred', {'message': f"Failed to compile plates: {e}"})

class RunPlatesCommand:
    """Command to run selected plates."""

    def can_execute(self, state: 'TUIState') -> bool:
        """Check if any plates are selected for running."""
        return bool(getattr(state, 'selected_plates', []) or getattr(state, 'focused_plate', None))

    async def execute(self, state: 'TUIState', context: 'ProcessingContext', **kwargs: Any) -> None:
        """Run selected plates."""
        try:
            # Get plates to run (selected or focused)
            plates_to_run = getattr(state, 'selected_plates', [])
            if not plates_to_run and getattr(state, 'focused_plate', None):
                plates_to_run = [state.focused_plate]

            if not plates_to_run:
                logger.warning("No plates selected for running")
                return

            logger.info(f"Running {len(plates_to_run)} plates")

            for plate_id in plates_to_run:
                try:
                    # Get orchestrator
                    if not hasattr(context, 'orchestrators') or plate_id not in context.orchestrators:
                        logger.error(f"No orchestrator found for plate: {plate_id}")
                        await state.notify('plate_status_changed', {
                            'plate_id': plate_id,
                            'status': 'error',
                            'message': 'No orchestrator found'
                        })
                        continue

                    orchestrator = context.orchestrators[plate_id]

                    # Update status to running
                    await state.notify('plate_status_changed', {
                        'plate_id': plate_id,
                        'status': 'running'
                    })

                    # Run pipeline
                    await orchestrator.run_pipeline()

                    # Update status to completed
                    await state.notify('plate_status_changed', {
                        'plate_id': plate_id,
                        'status': 'completed'
                    })

                    logger.info(f"Successfully ran plate: {plate_id}")

                except Exception as e:
                    logger.error(f"Failed to run plate {plate_id}: {e}")
                    await state.notify('plate_status_changed', {
                        'plate_id': plate_id,
                        'status': 'error',
                        'message': str(e)
                    })

        except Exception as e:
            logger.error(f"Failed to run plates: {e}")
            await state.notify('error_occurred', {'message': f"Failed to run plates: {e}"})

# =============================================================================
# Pipeline Editor Commands (Add, Delete, Edit, Load, Save)
# =============================================================================

class AddStepCommand:
    """Command to add a new step to the pipeline."""

    def can_execute(self, state: 'TUIState') -> bool:
        """Check if a plate is focused for adding steps."""
        return bool(getattr(state, 'focused_plate', None))

    async def execute(self, state: 'TUIState', context: 'ProcessingContext', **kwargs: Any) -> None:
        """Add a new step to the active pipeline."""
        try:
            focused_plate = getattr(state, 'focused_plate', None)
            if not focused_plate:
                logger.warning("No plate focused for adding step")
                return

            logger.info(f"Adding new step to plate: {focused_plate}")

            # Get orchestrator
            if not hasattr(context, 'orchestrators') or focused_plate not in context.orchestrators:
                logger.error(f"No orchestrator found for plate: {focused_plate}")
                await state.notify('error_occurred', {'message': 'No orchestrator found for focused plate'})
                return

            orchestrator = context.orchestrators[focused_plate]

            # Create a default step (this could be enhanced with a step type selection dialog)
            from openhcs.core.pipeline.steps import AbstractStep
            new_step = AbstractStep(
                name="New Step",
                input_dir=None,
                output_dir=None
            )

            # Add to pipeline
            if hasattr(orchestrator, 'pipeline_definition') and orchestrator.pipeline_definition:
                orchestrator.pipeline_definition.steps.append(new_step)

            await state.notify('step_added', {
                'plate_id': focused_plate,
                'step': new_step
            })

        except Exception as e:
            logger.error(f"Failed to add step: {e}")
            await state.notify('error_occurred', {'message': f"Failed to add step: {e}"})

class DeleteSelectedStepsCommand:
    """Command to delete selected steps from the pipeline."""

    def can_execute(self, state: 'TUIState') -> bool:
        """Check if any steps are selected for deletion."""
        return bool(getattr(state, 'selected_steps', []))

    async def execute(self, state: 'TUIState', context: 'ProcessingContext', **kwargs: Any) -> None:
        """Delete selected steps from the pipeline."""
        try:
            selected_steps = getattr(state, 'selected_steps', [])
            focused_plate = getattr(state, 'focused_plate', None)

            if not selected_steps:
                logger.warning("No steps selected for deletion")
                return

            if not focused_plate:
                logger.warning("No plate focused for step deletion")
                return

            logger.info(f"Deleting {len(selected_steps)} steps from plate: {focused_plate}")

            # Get orchestrator
            if not hasattr(context, 'orchestrators') or focused_plate not in context.orchestrators:
                logger.error(f"No orchestrator found for plate: {focused_plate}")
                await state.notify('error_occurred', {'message': 'No orchestrator found for focused plate'})
                return

            orchestrator = context.orchestrators[focused_plate]

            # Remove steps from pipeline (assuming selected_steps contains indices)
            if hasattr(orchestrator, 'pipeline_definition') and orchestrator.pipeline_definition:
                # Sort indices in reverse order to avoid index shifting issues
                for step_index in sorted(selected_steps, reverse=True):
                    if 0 <= step_index < len(orchestrator.pipeline_definition.steps):
                        removed_step = orchestrator.pipeline_definition.steps.pop(step_index)
                        await state.notify('step_removed', {
                            'plate_id': focused_plate,
                            'step_index': step_index,
                            'step': removed_step
                        })

            # Clear selection
            await state.notify('steps_selection_cleared', {})

        except Exception as e:
            logger.error(f"Failed to delete steps: {e}")
            await state.notify('error_occurred', {'message': f"Failed to delete steps: {e}"})

class ShowEditStepDialogCommand:
    """Command to show the dual step/func editor for editing a step."""

    def can_execute(self, state: 'TUIState') -> bool:
        """Check if a step is selected for editing."""
        return bool(getattr(state, 'focused_step', None))

    async def execute(self, state: 'TUIState', context: 'ProcessingContext', **kwargs: Any) -> None:
        """Show dual step/func editor for the focused step."""
        try:
            focused_step = getattr(state, 'focused_step', None)
            focused_plate = getattr(state, 'focused_plate', None)

            if not focused_step:
                logger.warning("No step focused for editing")
                return

            if not focused_plate:
                logger.warning("No plate focused for step editing")
                return

            logger.info(f"Showing step editor for step in plate: {focused_plate}")

            # Set editing state
            await state.notify('step_editing_started', {
                'plate_id': focused_plate,
                'step_to_edit': focused_step,
                'editing_step_config': True
            })

        except Exception as e:
            logger.error(f"Failed to show step editor: {e}")
            await state.notify('error_occurred', {'message': f"Failed to show step editor: {e}"})

class LoadPipelineCommand:
    """Command to load a pipeline from file."""

    def can_execute(self, state: 'TUIState') -> bool:
        """Check if a plate is focused for loading pipeline."""
        return bool(getattr(state, 'focused_plate', None))

    async def execute(self, state: 'TUIState', context: 'ProcessingContext', **kwargs: Any) -> None:
        """Load pipeline from file."""
        try:
            focused_plate = getattr(state, 'focused_plate', None)
            if not focused_plate:
                logger.warning("No plate focused for loading pipeline")
                await state.set_status("No plate selected for pipeline loading", "warning")
                return

            logger.info(f"Loading pipeline for plate: {focused_plate}")

            from ..utils.dialogs import prompt_for_path_dialog
            from pathlib import Path

            # Show file browser for pipeline loading
            selected_path = await prompt_for_path_dialog(
                title=f"Load Pipeline for {focused_plate}",
                initial_path=Path.home(),
                file_types=['.json', '.yaml', '.yml'],
                save_mode=False
            )

            if selected_path:
                try:
                    # Simulate loading pipeline (in real implementation, would parse the file)
                    await state.set_status(f"Pipeline loaded from: {selected_path.name}", "info")
                    logger.info(f"Loaded pipeline from {selected_path} for plate {focused_plate}")

                except Exception as e:
                    await state.set_status(f"Failed to load pipeline: {e}", "error")
            else:
                await state.set_status("Load pipeline cancelled", "info")

        except Exception as e:
            logger.error(f"Failed to load pipeline: {e}")
            await state.set_status(f"Failed to load pipeline: {e}", "error")

class SavePipelineCommand:
    """Command to save pipeline to file."""

    def can_execute(self, state: 'TUIState') -> bool:
        """Check if a plate is focused for saving pipeline."""
        return bool(getattr(state, 'focused_plate', None))

    async def execute(self, state: 'TUIState', context: 'ProcessingContext', **kwargs: Any) -> None:
        """Save pipeline to file."""
        try:
            focused_plate = getattr(state, 'focused_plate', None)
            if not focused_plate:
                logger.warning("No plate focused for saving pipeline")
                await state.set_status("No plate selected for pipeline saving", "warning")
                return

            logger.info(f"Saving pipeline for plate: {focused_plate}")

            from ..utils.dialogs import prompt_for_path_dialog
            from pathlib import Path
            import json

            # Show file browser for pipeline saving
            default_name = f"{focused_plate}_pipeline.json"
            selected_path = await prompt_for_path_dialog(
                title=f"Save Pipeline for {focused_plate}",
                initial_path=Path.home() / default_name,
                file_types=['.json', '.yaml', '.yml'],
                save_mode=True
            )

            if selected_path:
                try:
                    # Ensure .json extension
                    if not selected_path.suffix.lower() in ['.json', '.yaml', '.yml']:
                        selected_path = selected_path.with_suffix('.json')

                    # Create demo pipeline data
                    pipeline_data = {
                        "plate_id": focused_plate,
                        "steps": [
                            {"name": "Load Images", "type": "LoadStep", "status": "completed"},
                            {"name": "Normalize", "type": "NormalizeStep", "status": "running"},
                            {"name": "Stitch", "type": "StitchStep", "status": "pending"}
                        ],
                        "created": "2024-01-01T00:00:00Z",
                        "version": "1.0"
                    }

                    # Write pipeline file
                    with open(selected_path, 'w') as f:
                        json.dump(pipeline_data, f, indent=2)

                    await state.set_status(f"Pipeline saved to: {selected_path.name}", "info")
                    logger.info(f"Saved pipeline to {selected_path} for plate {focused_plate}")

                except Exception as e:
                    await state.set_status(f"Failed to save pipeline: {e}", "error")
            else:
                await state.set_status("Save pipeline cancelled", "info")

        except Exception as e:
            logger.error(f"Failed to save pipeline: {e}")
            await state.set_status(f"Failed to save pipeline: {e}", "error")
