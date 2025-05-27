# Code Reference Analysis

Found 14 files with references:

## openhcs/tui/components.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 82 | InteractiveListItem._create_container | `VSplit` |
| 82 | InteractiveListItem._create_container | `VSplit` |
| 97 | InteractiveListItem._create_container | `VSplit` |
| 68 | InteractiveListItem._create_container | `VSplit` |
| 189 | ParameterEditor._build_ui | `HSplit` |
| 164 | ParameterEditor._build_ui | `Container` |
| 226 | ParameterEditor._create_parameter_field | `VSplit` |
| 213 | ParameterEditor._create_parameter_field | `Container` |

### String References
| Line | Reference |
| ---- | --------- |
| 7 | `from prompt_toolkit.layout import HSplit, ScrollablePane, VSplit, Container # Ad...` |
| 9 | `TextArea, RadioList as Dropdown)` |
| 68 | `def _create_container(self) -> VSplit: # Changed to VSplit for better layout con...` |
| 82 | `buttons_container = VSplit(move_buttons_children, width=3, padding=0) if move_bu...` |
| 97 | `return VSplit([` |
| 164 | `def _build_ui(self) -> Container:` |
| 189 | `return HSplit(param_fields)` |
| 213 | `def _create_parameter_field(self, name: str, default: Any, current_value: Any, r...` |
| 226 | `return VSplit([label, input_field, Box(reset_button, width=8)], padding=1)` |

## openhcs/tui/file_browser.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 123 | FileManagerBrowser._build_ui | `VSplit` |
| 128 | FileManagerBrowser._build_ui | `VSplit` |
| 134 | FileManagerBrowser._build_ui | `VSplit` |
| 149 | FileManagerBrowser._build_ui | `HSplit` |
| 119 | FileManagerBrowser._build_ui | `Container` |
| 239 | FileManagerBrowser._build_item_list_ui | `HSplit` |
| 180 | FileManagerBrowser._build_item_list_ui | `Container` |

### String References
| Line | Reference |
| ---- | --------- |
| 16 | `HSplit,` |
| 17 | `VSplit,` |
| 18 | `DynamicContainer,` |
| 22 | `ConditionalContainer # Added ConditionalContainer` |
| 74 | `self.item_list_container = DynamicContainer(lambda: self._build_item_list_ui())` |
| 119 | `def _build_ui(self) -> Container:` |
| 123 | `action_buttons = VSplit([` |
| 128 | `nav_buttons = VSplit([` |
| 134 | `bottom_bar = VSplit([` |
| 144 | `error_container = ConditionalContainer(` |
| 149 | `return HSplit([` |
| 155 | `bottom_bar, # Use the new VSplit for buttons` |
| 180 | `def _build_item_list_ui(self) -> Container:` |
| 239 | `return HSplit(items_ui)` |

## openhcs/tui/dual_step_func_editor.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 63 | DualStepFuncEditorPane.__init__ | `HSplit` |
| 96 | DualStepFuncEditorPane._initialize_ui | `VSplit` |
| 121 | DualStepFuncEditorPane.get_current_view_container | `HSplit` |
| 125 | DualStepFuncEditorPane._initialize_ui | `HSplit` |
| 159 | DualStepFuncEditorPane._create_step_settings_view | `CheckboxList` |
| 176 | DualStepFuncEditorPane._create_step_settings_view | `RadioList` |
| 191 | DualStepFuncEditorPane._create_step_settings_view | `RadioList` |
| 217 | DualStepFuncEditorPane._create_step_settings_view | `VSplit` |
| 223 | DualStepFuncEditorPane._create_step_settings_view | `HSplit` |
| 230 | DualStepFuncEditorPane._create_step_settings_view | `VSplit` |
| 238 | DualStepFuncEditorPane._create_step_settings_view | `HSplit` |
| 256 | DualStepFuncEditorPane._create_func_pattern_view | `HSplit` |
| 349 | DualStepFuncEditorPane._save_changes | `RadioList` |
| 355 | DualStepFuncEditorPane._save_changes | `RadioList` |
| 361 | DualStepFuncEditorPane._save_changes | `CheckboxList` |
| 404 | DualStepFuncEditorPane.container | `HSplit` |
| 401 | DualStepFuncEditorPane.container | `Container` |
| 503 | DualStepFuncEditorPane._reset_step_parameter_field | `CheckboxList` |
| 506 | DualStepFuncEditorPane._reset_step_parameter_field | `RadioList` |
| 530 | DualStepFuncEditorPane._reset_step_parameter_field | `RadioList` |

### String References
| Line | Reference |
| ---- | --------- |
| 21 | `from prompt_toolkit.layout import HSplit, VSplit, DynamicContainer, FormattedTex...` |
| 22 | `from prompt_toolkit.widgets import Box, Button, Frame, Label, TextArea, RadioLis...` |
| 63 | `self._container: Optional[HSplit] = None` |
| 96 | `menu_bar = VSplit([` |
| 121 | `return self.func_pattern_container if self.func_pattern_container else HSplit([L...` |
| 123 | `dynamic_content_area = DynamicContainer(get_current_view_container)` |
| 125 | `self._container = HSplit([` |
| 159 | `widget = CheckboxList(values=[(param_name, "")])` |
| 162 | `# For CheckboxList, changes are typically handled on save or via a dedicated cal...` |
| 167 | `# Determine initial selection for RadioList` |
| 176 | `widget = RadioList(values=options, default=initial_selection)` |
| 177 | `# The handler for RadioList is set directly on the widget instance` |
| 191 | `widget = RadioList(values=options, default=initial_selection)` |
| 217 | `rows.append(VSplit([` |
| 223 | `parameter_fields_container = HSplit(rows)` |
| 230 | `step_settings_toolbar = VSplit([` |
| 238 | `view_content = HSplit([` |
| 243 | `# Removed the old step_object_buttons VSplit from here` |
| 256 | `return HSplit([Label("Error: Function Pattern Editor component is not loaded.")]...` |
| 302 | `# Handle variable_components specifically if it was changed by RadioList` |
| 349 | `if param_name == "variable_components" and isinstance(widget, RadioList):` |
| 355 | `elif param_name == "group_by" and isinstance(widget, RadioList):` |
| 361 | `elif isinstance(widget, CheckboxList) and actual_type is bool: # Existing bool h...` |
| 401 | `def container(self) -> Container: # Ensure Container is imported from prompt_too...` |
| 403 | `logger.error("DualStepFuncEditorPane: Container accessed before initialization."...` |
| 404 | `return HSplit([Label("Error: Editor not initialized.")]) # Fallback` |
| 503 | `elif isinstance(associated_widget, CheckboxList):` |
| 504 | `# Assuming param_name_to_reset is the value for boolean CheckboxList` |
| 506 | `elif isinstance(associated_widget, RadioList):` |
| 524 | `logger.warning(f"Cannot reset RadioList for unknown enum param: {param_name_to_r...` |
| 530 | `self._something_changed(param_name=param_name_to_reset, widget_value=original_va...` |

## openhcs/tui/menu_bar.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 373 | MenuBar.None | `Container` |
| 417 | MenuBar.__init__ | `VSplit` |
| 622 | MenuBar._create_submenu_float | `HSplit` |
| 892 | MenuBar._create_submenu_container | `HSplit` |
| 834 | MenuBar._create_submenu_container | `Container` |
| 1198 | MenuBar.__pt_container__ | `Container` |

### String References
| Line | Reference |
| ---- | --------- |
| 31 | `from prompt_toolkit.layout import (Container, FormattedTextControl, HSplit,` |
| 32 | `VSplit, Window)` |
| 33 | `from prompt_toolkit.layout.containers import (AnyContainer,` |
| 34 | `ConditionalContainer, Float)` |
| 373 | `class MenuBar(Container):` |
| 417 | `self.container = VSplit(self.menu_labels)` |
| 622 | `submenu_container = HSplit([])` |
| 834 | `def _create_submenu_container(self, menu_items: List[MenuItem]) -> Container:` |
| 842 | `A Container for the submenu` |
| 892 | `return HSplit(labels)` |
| 1198 | `def __pt_container__(self) -> Container:` |

## openhcs/tui/utils.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 23 | show_error_dialog | `HSplit` |
| 97 | prompt_for_path_dialog | `HSplit` |

### String References
| Line | Reference |
| ---- | --------- |
| 3 | `from prompt_toolkit.layout.containers import HSplit` |
| 23 | `body=HSplit([` |
| 97 | `body=HSplit([` |

## openhcs/tui/function_pattern_editor.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 129 | FunctionPatternEditor.__init__ | `HSplit` |
| 130 | FunctionPatternEditor.__init__ | `HSplit` |
| 137 | FunctionPatternEditor.__init__ | `HSplit` |
| 145 | FunctionPatternEditor.container | `Container` |
| 285 | FunctionPatternEditor._create_header | `VSplit` |
| 354 | FunctionPatternEditor._refresh_key_selector | `VSplit` |
| 358 | FunctionPatternEditor._refresh_key_selector | `VSplit` |
| 376 | FunctionPatternEditor._refresh_function_list | `HSplit` |
| 436 | FunctionPatternEditor._create_function_item | `HSplit` |
| 438 | FunctionPatternEditor._create_function_item | `VSplit` |
| 735 | FunctionPatternEditor._create_parameter_editor | `HSplit` |
| 755 | FunctionPatternEditor._create_parameter_field | `VSplit` |

### String References
| Line | Reference |
| ---- | --------- |
| 26 | `from prompt_toolkit.layout import HSplit, ScrollablePane, VSplit, Container` |
| 27 | `from prompt_toolkit.widgets import (Box, Button, Dialog, Label, TextArea, RadioL...` |
| 129 | `self.key_selector_container = HSplit([])` |
| 130 | `self.function_list_container = ScrollablePane(HSplit([])) # This will contain it...` |
| 137 | `self._container = HSplit([` |
| 145 | `def container(self) -> Container:` |
| 285 | `return VSplit([` |
| 354 | `key_management_buttons = VSplit([add_key_button, remove_key_button], padding=1)` |
| 358 | `VSplit([` |
| 376 | `self.function_list_container.content = HSplit(function_items)` |
| 436 | `HSplit([` |
| 438 | `VSplit([` |
| 735 | `return HSplit(param_fields)` |
| 755 | `return VSplit([` |

## openhcs/tui/plate_manager_core.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 95 | PlateManagerPane.__init__ | `HSplit` |
| 148 | PlateManagerPane._initialize_ui | `HSplit` |
| 184 | PlateManagerPane.container | `Container` |
| 195 | PlateManagerPane.get_buttons_container | `HSplit` |
| 197 | PlateManagerPane.get_buttons_container | `HSplit` |
| 198 | PlateManagerPane.get_buttons_container | `VSplit` |
| 191 | PlateManagerPane.get_buttons_container | `Container` |
| 301 | PlateManagerPane._build_plate_items_container | `HSplit` |
| 278 | PlateManagerPane._build_plate_items_container | `HSplit` |

### String References
| Line | Reference |
| ---- | --------- |
| 38 | `from prompt_toolkit.layout import Container, HSplit, VSplit, DynamicContainer, D...` |
| 95 | `self.plate_items_container_widget: Optional[HSplit] = None` |
| 96 | `self._dynamic_plate_list_wrapper: Optional[DynamicContainer] = None` |
| 148 | `self.get_current_plate_list_container = lambda: self.plate_items_container_widge...` |
| 149 | `self._dynamic_plate_list_wrapper = DynamicContainer(self.get_current_plate_list_...` |
| 184 | `def container(self) -> Container:` |
| 191 | `def get_buttons_container(self) -> Container:` |
| 195 | `return HSplit([Label("Buttons not ready.")])` |
| 197 | `return HSplit([` |
| 198 | `VSplit([ # VSplit for horizontal arrangement of buttons` |
| 278 | `async def _build_plate_items_container(self) -> HSplit:` |
| 279 | `"""Builds the HSplit container holding individual InteractiveListItem widgets fo...` |
| 300 | `# Ensure HSplit always has children, even if it's just a placeholder label` |
| 301 | `return HSplit(item_widgets if item_widgets else [Label(" ")], width=Dimension(we...` |
| 614 | `"""Placeholder. Relies on PTK default scroll-to-focus for Frame/DynamicContainer...` |

## openhcs/tui/tui_architecture.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 296 | OpenHCSTUI._create_root_container | `HSplit` |
| 298 | OpenHCSTUI._create_root_container | `VSplit` |
| 305 | OpenHCSTUI._create_root_container | `VSplit` |
| 310 | OpenHCSTUI._create_root_container | `VSplit` |
| 323 | OpenHCSTUI._create_root_container | `VSplit` |
| 291 | OpenHCSTUI._create_root_container | `Container` |
| 331 | OpenHCSTUI._get_left_pane | `Container` |
| 420 | OpenHCSTUI._get_step_viewer | `Container` |
| 437 | OpenHCSTUI._get_status_bar | `Container` |
| 451 | OpenHCSTUI._get_menu_bar | `Container` |

### String References
| Line | Reference |
| ---- | --------- |
| 25 | `from typing import Any, Callable, Container, Dict, List, Optional, Union, TYPE_C...` |
| 61 | `from prompt_toolkit.layout import Container, HSplit, Layout, VSplit, Window` |
| 62 | `from prompt_toolkit.layout.containers import (DynamicContainer, Float,` |
| 63 | `FloatContainer)` |
| 291 | `def _create_root_container(self) -> Container:` |
| 296 | `return HSplit([` |
| 298 | `VSplit([` |
| 305 | `VSplit([` |
| 310 | `VSplit([` |
| 311 | `DynamicContainer(` |
| 316 | `DynamicContainer(` |
| 322 | `# Main Panes (Plate Manager \| Pipeline Editor) - This should be a VSplit` |
| 323 | `VSplit([` |
| 331 | `def _get_left_pane(self) -> Container:` |
| 420 | `def _get_step_viewer(self) -> Container:` |
| 437 | `def _get_status_bar(self) -> Container:` |
| 451 | `def _get_menu_bar(self) -> Container:` |

## openhcs/tui/step_viewer.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 48 | PipelineEditorPane.__init__ | `HSplit` |
| 124 | PipelineEditorPane.setup | `HSplit` |
| 146 | PipelineEditorPane.container | `Container` |
| 155 | PipelineEditorPane.get_buttons_container | `VSplit` |
| 157 | PipelineEditorPane.get_buttons_container | `VSplit` |
| 150 | PipelineEditorPane.get_buttons_container | `Container` |
| 203 | PipelineEditorPane._build_step_items_container | `HSplit` |
| 165 | PipelineEditorPane._build_step_items_container | `HSplit` |

### String References
| Line | Reference |
| ---- | --------- |
| 8 | `from prompt_toolkit.layout import HSplit, VSplit` |
| 10 | `from prompt_toolkit.layout import HSplit, VSplit, DynamicContainer, Container, D...` |
| 48 | `self.step_items_container_widget: Optional[HSplit] = None # Will hold HSplit of ...` |
| 55 | `self._dynamic_step_list_wrapper: Optional[DynamicContainer] = None` |
| 123 | `# We use a DynamicContainer to allow replacing the HSplit of items easily` |
| 124 | `self.get_current_step_list_container = lambda: self.step_items_container_widget ...` |
| 125 | `self._dynamic_step_list_wrapper = DynamicContainer(self.get_current_step_list_co...` |
| 146 | `def container(self) -> Container:` |
| 150 | `def get_buttons_container(self) -> Container:` |
| 155 | `return VSplit([Label("Pipeline Buttons not ready.")])` |
| 157 | `return VSplit([ # Use VSplit for horizontal button layout` |
| 165 | `async def _build_step_items_container(self) -> HSplit:` |
| 167 | `Builds the HSplit container holding individual InteractiveStepItem widgets.` |
| 203 | `return HSplit(item_widgets if item_widgets else [Label("No steps to display.")],...` |
| 394 | `# Rebuild the HSplit with new InteractiveListItem instances` |
| 396 | `# The DynamicContainer (self._dynamic_step_list_wrapper) will pick up the change` |
| 404 | `get_app().invalidate() # Crucial to trigger a redraw of the DynamicContainer` |

## openhcs/tui/status_bar.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 216 | StatusBar.None | `Container` |
| 238 | StatusBar.__init__ | `HSplit` |

### String References
| Line | Reference |
| ---- | --------- |
| 39 | `from prompt_toolkit.layout import ConditionalContainer, Container, HSplit` |
| 216 | `class StatusBar(Container):` |
| 233 | `self.log_drawer_container = ConditionalContainer(` |
| 238 | `self.container = HSplit([` |

## openhcs/tui/dialogs/global_settings_editor.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 57 | GlobalSettingsEditorDialog._build_dialog | `RadioList` |
| 65 | GlobalSettingsEditorDialog._build_dialog | `RadioList` |
| 75 | GlobalSettingsEditorDialog._build_dialog | `HSplit` |
| 77 | GlobalSettingsEditorDialog._build_dialog | `HSplit` |
| 79 | GlobalSettingsEditorDialog._build_dialog | `HSplit` |
| 90 | GlobalSettingsEditorDialog._build_dialog | `HSplit` |

### String References
| Line | Reference |
| ---- | --------- |
| 12 | `from prompt_toolkit.layout import HSplit, VSplit` |
| 13 | `from prompt_toolkit.widgets import Button, Dialog, Label, TextArea, RadioList, C...` |
| 55 | `# Example: VFS default_storage_backend (using RadioList as Dropdown)` |
| 57 | `self.vfs_backend_selector = RadioList(` |
| 65 | `self.microscope_selector = RadioList(` |
| 75 | `HSplit([Label("Num Workers:", width=25), self.num_workers_input]),` |
| 77 | `HSplit([Label("  Default Storage Backend:", width=25), self.vfs_backend_selector...` |
| 79 | `HSplit([Label("  Default Microscope Type:", width=25), self.microscope_selector]...` |
| 90 | `body=HSplit(body_content, padding=1),` |

## openhcs/tui/dialogs/plate_config_editor.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 170 | PlateConfigEditorPane._create_config_widgets | `RadioList` |
| 178 | PlateConfigEditorPane._create_config_widgets | `HSplit` |
| 180 | PlateConfigEditorPane._create_config_widgets | `HSplit` |
| 189 | PlateConfigEditorPane._create_config_widgets | `VSplit` |
| 204 | PlateConfigEditorPane._build_layout | `HSplit` |
| 206 | PlateConfigEditorPane._build_layout | `VSplit` |
| 214 | PlateConfigEditorPane._build_layout | `HSplit` |
| 192 | PlateConfigEditorPane._build_layout | `Container` |
| 223 | PlateConfigEditorPane.container | `Container` |
| 247 | PlateConfigEditorPane._handle_save | `RadioList` |

### String References
| Line | Reference |
| ---- | --------- |
| 14 | `from prompt_toolkit.layout import HSplit, VSplit, DynamicContainer, Container, D...` |
| 15 | `from prompt_toolkit.widgets import Box, Button, Frame, Label, TextArea, RadioLis...` |
| 97 | `elif current_value_type == bool: # For Checkbox or RadioList representing bool` |
| 102 | `elif issubclass(current_value_type, Enum): # For RadioList returning string memb...` |
| 170 | `input_widget = RadioList(values=enum_values, current_value=current_value.name)` |
| 171 | `# RadioList changes are typically polled. For live update:` |
| 173 | `#     # This is tricky as RadioList doesn't have a direct on_change.` |
| 174 | `#     # This would require a custom RadioList or polling.` |
| 178 | `widgets.append(HSplit([` |
| 180 | `Box(HSplit(self._create_config_widgets(current_value, full_path)), padding_left=...` |
| 189 | `widgets.append(VSplit([label, input_widget], padding=0, width=Dimension(max=100)...` |
| 192 | `def _build_layout(self) -> Container:` |
| 204 | `self.config_view_container = ScrollablePane(HSplit(config_widgets_list))` |
| 206 | `buttons = VSplit([` |
| 214 | `HSplit([` |
| 223 | `def container(self) -> Container:` |
| 235 | `# This loop explicitly processes RadioLists and Checkboxes.` |
| 247 | `if widget_identifier == "radiolist" and isinstance(widget, RadioList):` |

## openhcs/tui/dialogs/plate_dialog_manager.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 54 | ErrorBanner.None | `Container` |
| 471 | PlateDialogManager._create_error_dialog | `HSplit` |
| 485 | PlateDialogManager._create_error_dialog | `HSplit` |
| 517 | PlateDialogManager.show_remove_plate_dialog | `HSplit` |

### String References
| Line | Reference |
| ---- | --------- |
| 29 | `from prompt_toolkit.layout import (ConditionalContainer, Container, Float,` |
| 30 | `HSplit, Dimension) # Added Dimension` |
| 32 | `from prompt_toolkit.widgets import RadioList as Dropdown` |
| 54 | `class ErrorBanner(Container):` |
| 79 | `self.filtered_container = ConditionalContainer(` |
| 139 | `container: Container to search in` |
| 471 | `body = HSplit([` |
| 485 | `body = HSplit([` |
| 517 | `body=HSplit([` |

## openhcs/tui/services/external_editor_service.py
**Imports target:** True

### AST References
| Line | Context | Reference |
| ---- | ------- | --------- |
| 125 | ExternalEditorService._show_error_dialog | `HSplit` |

### String References
| Line | Reference |
| ---- | --------- |
| 9 | `from prompt_toolkit.layout import HSplit` |
| 125 | `body=HSplit([Label(message)]),` |
