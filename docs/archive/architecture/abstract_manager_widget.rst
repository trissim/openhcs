AbstractManagerWidget Architecture
===================================

The Problem: Duplicated Manager Widget Code
--------------------------------------------

PlateManagerWidget and PipelineEditorWidget implement nearly identical CRUD operations (add, delete, edit, list items) with only domain-specific differences. This duplication (~1000 lines) creates maintenance burden: bug fixes must be applied twice, and adding new features requires changes in multiple places. Additionally, both widgets use duck-typing (implicit interfaces), making it hard to understand what methods subclasses must implement.

The Solution: Template Method Pattern with Declarative Configuration
---------------------------------------------------------------------

AbstractManagerWidget uses the template method pattern to define the CRUD workflow once, with declarative configuration via class attributes. Subclasses specify their domain-specific behavior (button configs, item hooks, preview fields) as class attributes rather than implementing methods. This eliminates duplication, makes the interface explicit (ABC contracts), and enables easy extension.

Overview
--------

The ``AbstractManagerWidget`` is a PyQt6 ABC that eliminates duck-typing and code duplication
between ``PlateManagerWidget`` and ``PipelineEditorWidget`` through declarative configuration
and the template method pattern.

Architecture Pattern
--------------------

The ABC uses the **template method pattern** with declarative configuration:

.. code-block:: python

    from openhcs.pyqt_gui.widgets.shared.abstract_manager_widget import AbstractManagerWidget

    class PipelineEditorWidget(AbstractManagerWidget):
        # Declarative configuration via class attributes
        TITLE = "Pipeline Editor"

        BUTTON_CONFIGS = [
            ButtonConfig(text="Add Step", action="add", icon="plus"),
            ButtonConfig(text="Delete Step", action="delete", icon="trash"),
            ButtonConfig(text="Edit Step", action="edit", icon="edit"),
        ]

        ITEM_HOOKS = ItemHooks(
            get_items=lambda self: self.pipeline_steps,
            set_items=lambda self, items: setattr(self, 'pipeline_steps', items),
            get_selected_index=lambda self: self.step_list.currentRow(),
        )

        PREVIEW_FIELD_CONFIGS = [
            ('napari_streaming_config.enabled', lambda v: 'NAP' if v else None, 'step'),
            ('fiji_streaming_config.enabled', lambda v: 'FIJI' if v else None, 'step'),
        ]

        # Implement abstract hooks for domain-specific behavior
        def _perform_delete(self, index: int) -> None:
            """Delete step at index."""
            del self.pipeline_steps[index]

        def _show_item_editor(self, item: Any, index: int) -> None:
            """Show step editor dialog."""
            dialog = StepEditorDialog(item, parent=self)
            dialog.exec()

        def _format_list_item(self, item: Any, index: int) -> str:
            """Format step for display in list."""
            return f"{index + 1}. {item.name}"

Declarative Configuration
--------------------------

**Class Attributes**:

- ``TITLE``: Widget title (str)
- ``BUTTON_CONFIGS``: List of ``ButtonConfig`` objects defining toolbar buttons
- ``ITEM_HOOKS``: ``ItemHooks`` dataclass with lambdas for item access
- ``PREVIEW_FIELD_CONFIGS``: List of tuples ``(field_path, formatter, scope_root)`` for cross-window previews
- ``CODE_EDITOR_CONFIG``: Optional ``CodeEditorConfig`` for code editing support

**ButtonConfig**:

.. code-block:: python

    @dataclass
    class ButtonConfig:
        text: str
        action: str  # Maps to action_{action} method
        icon: Optional[str] = None
        tooltip: Optional[str] = None

**ItemHooks**:

.. code-block:: python

    @dataclass
    class ItemHooks:
        get_items: Callable[[Any], List[Any]]
        set_items: Callable[[Any, List[Any]], None]
        get_selected_index: Callable[[Any], int]
        get_item_at_index: Optional[Callable[[Any, int], Any]] = None

Template Methods
-----------------

The ABC provides template methods that orchestrate the workflow:

**CRUD Operations**:

- ``action_add()``: Add new item (calls ``_create_new_item()`` hook)
- ``action_delete()``: Delete selected item (calls ``_perform_delete()`` hook)
- ``action_edit()``: Edit selected item (calls ``_show_item_editor()`` hook)
- ``update_item_list()``: Refresh list widget (calls ``_format_list_item()`` hook)

**Code Editing**:

- ``action_view_code()``: Show code editor dialog
- ``_handle_edited_code(code)``: Execute edited code and apply to widget state

**Cross-Window Previews**:

- ``_init_cross_window_preview_mixin()``: Initialize preview system
- ``_process_pending_preview_updates()``: Apply incremental preview updates

Abstract Hooks
--------------

Subclasses must implement these abstract methods:

.. code-block:: python

    @abstractmethod
    def _perform_delete(self, index: int) -> None:
        """Delete item at index."""
        ...

    @abstractmethod
    def _show_item_editor(self, item: Any, index: int) -> None:
        """Show editor dialog for item."""
        ...

    @abstractmethod
    def _format_list_item(self, item: Any, index: int) -> str:
        """Format item for display in list widget."""
        ...

    @abstractmethod
    def _get_context_stack_for_resolution(self) -> List[Any]:
        """Get context stack for lazy config resolution."""
        ...

Optional hooks with default implementations:

- ``_create_new_item() -> Any``: Create new item (default: None)
- ``_get_code_editor_title() -> str``: Code editor title (default: "Code Editor")
- ``_apply_extracted_variables(vars: Dict[str, Any])``: Apply code execution results

See Also
--------

- :doc:`widget_protocol_system` - ABC contracts for widget operations
- :doc:`ui_services_architecture` - Service layer for ParameterFormManager
- :doc:`gui_performance_patterns` - Cross-window preview system
- :doc:`batch_workflow_service` - Unified compile/execute orchestration service
- :doc:`zmq_server_browser_system` - ZMQ browser abstraction and OpenHCS adapter
- :doc:`parametric_widget_creation` - Widget creation configuration
