"""Multi-orchestrator configuration window for OpenHCS Textual TUI."""

from typing import Type, Any, Callable, Optional, List, Dict
from textual.app import ComposeResult
from textual.widgets import Button, Static
from textual.containers import Container, Horizontal, ScrollableContainer
import dataclasses

from openhcs.textual_tui.windows.base_window import BaseOpenHCSWindow
from openhcs.textual_tui.widgets.config_form import ConfigFormWidget
from openhcs.core.config import GlobalPipelineConfig
from openhcs.textual_tui.widgets.shared.signature_analyzer import SignatureAnalyzer


class MultiOrchestratorConfigWindow(BaseOpenHCSWindow):
    """Multi-orchestrator configuration window using textual-window system."""

    DEFAULT_CSS = """
    MultiOrchestratorConfigWindow {
        width: 80; height: 35;
        min-width: 60; min-height: 20;
    }
    
    #config_form {
        height: 1fr;
        overflow-y: auto;
    }
    
    #status_bar {
        color: $text-muted;
        text-align: center;
        height: 1;
        margin: 1 0;
    }
    """

    def __init__(self, 
                 orchestrators: List[Any],
                 on_save_callback: Optional[Callable] = None, 
                 **kwargs):
        """
        Initialize multi-orchestrator config window.

        Args:
            orchestrators: List of orchestrators to configure
            on_save_callback: Function to call when config is saved
        """
        super().__init__(
            window_id="multi_orchestrator_config",
            title=f"Multi-Orchestrator Configuration ({len(orchestrators)} orchestrators)",
            mode="temporary",
            **kwargs
        )
        self.orchestrators = orchestrators
        self.on_save_callback = on_save_callback
        
        # Analyze configs across orchestrators
        self.config_analysis = self._analyze_orchestrator_configs()
        
        # Create base config for form (use first orchestrator's config as template)
        self.base_config = orchestrators[0].global_config if orchestrators else None
        
        # Create the form widget using unified parameter analysis
        if self.base_config:
            self.config_form = ConfigFormWidget.from_dataclass(
                GlobalPipelineConfig, 
                self.base_config
            )
            # Attach config analysis to form manager for different values handling
            if hasattr(self.config_form, 'form_manager'):
                self.config_form.form_manager.config_analysis = self.config_analysis

    def _analyze_orchestrator_configs(self) -> Dict[str, Any]:
        """Analyze configs across orchestrators to find same/different values."""
        if not self.orchestrators:
            return {}

        # Get parameter info for defaults
        param_info = SignatureAnalyzer.analyze(GlobalPipelineConfig)
        config_analysis = {}

        # Analyze each field in GlobalPipelineConfig
        for field in dataclasses.fields(GlobalPipelineConfig):
            field_name = field.name

            # Get values from all orchestrators
            values = []
            for orch in self.orchestrators:
                try:
                    value = getattr(orch.global_config, field_name)
                    values.append(value)
                except AttributeError:
                    # Field doesn't exist in this config, skip
                    continue

            if not values:
                continue

            # Get default value from parameter info
            param_details = param_info.get(field_name)
            default_value = param_details.default_value if param_details else None

            # Check if all values are the same
            if all(self._values_equal(v, values[0]) for v in values):
                config_analysis[field_name] = {
                    "type": "same",
                    "value": values[0],
                    "default": default_value
                }
            else:
                config_analysis[field_name] = {
                    "type": "different",
                    "values": values,
                    "default": default_value
                }

        return config_analysis

    def _values_equal(self, val1: Any, val2: Any) -> bool:
        """Check if two values are equal, handling dataclasses and complex types."""
        # Handle dataclass comparison
        if dataclasses.is_dataclass(val1) and dataclasses.is_dataclass(val2):
            return dataclasses.asdict(val1) == dataclasses.asdict(val2)
        
        # Handle enum comparison
        if hasattr(val1, 'value') and hasattr(val2, 'value'):
            return val1.value == val2.value
        
        # Standard comparison
        return val1 == val2

    def compose(self) -> ComposeResult:
        """Compose the config window content."""
        # Status bar
        yield Static("Ready", id="status_bar")
        
        # Scrollable form area
        with ScrollableContainer(id="config_form"):
            if hasattr(self, 'config_form'):
                yield self.config_form
            else:
                yield Static("No configuration available")

        # Buttons
        with Horizontal(classes="dialog-buttons"):
            yield Button("Save", id="save", compact=True)
            yield Button("Cancel", id="cancel", compact=True)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "save":
            self._handle_save()
        elif event.button.id == "cancel":
            self.close_window()

    def _handle_save(self):
        """Handle save button - apply config to all orchestrators."""
        if not hasattr(self, 'config_form'):
            self.close_window()
            return
            
        try:
            # Get form values
            form_values = self.config_form.get_config_values()

            # Create new config instance
            new_config = GlobalPipelineConfig(**form_values)

            # Update thread-local storage for MaterializationPathConfig defaults
            from openhcs.core.config import set_current_pipeline_config
            set_current_pipeline_config(new_config)

            # Apply to all orchestrators
            import asyncio
            async def apply_to_all():
                for orchestrator in self.orchestrators:
                    await orchestrator.apply_new_global_config(new_config)

            # Run the async operation
            asyncio.create_task(apply_to_all())

            # Call the callback if provided
            if self.on_save_callback:
                self.on_save_callback(new_config, len(self.orchestrators))

            self.close_window()

        except Exception as e:
            # Update status bar with error
            status_bar = self.query_one("#status_bar", Static)
            status_bar.update(f"Error: {e}")


async def show_multi_orchestrator_config(app, orchestrators: List[Any], 
                                        on_save_callback: Optional[Callable] = None):
    """
    Show multi-orchestrator config window.
    
    Args:
        app: The Textual app instance
        orchestrators: List of orchestrators to configure
        on_save_callback: Optional callback when config is saved
    """
    from textual.css.query import NoMatches
    
    # Try to find existing window
    try:
        window = app.query_one(MultiOrchestratorConfigWindow)
        # Window exists, just open it
        window.open_state = True
    except NoMatches:
        # Create new window
        window = MultiOrchestratorConfigWindow(
            orchestrators=orchestrators,
            on_save_callback=on_save_callback
        )
        await app.mount(window)
        window.open_state = True
    
    return window
