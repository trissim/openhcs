# Code Reference Analysis

Found 15 files with references:

## openhcs/tui/components.py
**Imports target:** False

### String References
| Line | Reference |
| ---- | --------- |
| 5 | `from prompt_toolkit.application import get_app` |
| 6 | `from prompt_toolkit.formatted_text import HTML` |
| 7 | `from prompt_toolkit.layout import HSplit, ScrollablePane, VSplit, Container # Ad...` |
| 8 | `from prompt_toolkit.widgets import (Box, Button, Dialog, Label,` |
| 10 | `from prompt_toolkit.key_binding import KeyBindings # Added KeyBindings` |
| 14 | `A custom prompt_toolkit widget that represents an interactive item in a list.` |

## openhcs/tui/file_browser.py
**Imports target:** False

### String References
| Line | Reference |
| ---- | --------- |
| 13 | `from prompt_toolkit.application import get_app` |
| 14 | `from prompt_toolkit.layout import (` |
| 24 | `from prompt_toolkit.filters import Condition # Added Condition` |
| 25 | `from prompt_toolkit.widgets import Box, Button, Label, TextArea` |
| 26 | `from prompt_toolkit.key_binding import KeyBindings` |
| 27 | `from prompt_toolkit.formatted_text import HTML, FormattedText, to_formatted_text` |
| 479 | `# Example Usage (requires a running prompt_toolkit application)` |
| 481 | `from prompt_toolkit.application import Application` |
| 482 | `from prompt_toolkit.layout.layout import Layout` |

## openhcs/tui/dual_step_func_editor.py
**Imports target:** False

### String References
| Line | Reference |
| ---- | --------- |
| 19 | `from prompt_toolkit.application import get_app` |
| 20 | `from prompt_toolkit.key_binding import KeyBindings` |
| 21 | `from prompt_toolkit.layout import HSplit, VSplit, DynamicContainer, FormattedTex...` |
| 22 | `from prompt_toolkit.widgets import Box, Button, Frame, Label, TextArea, RadioLis...` |
| 401 | `def container(self) -> Container: # Ensure Container is imported from prompt_too...` |

## openhcs/tui/menu_bar.py
**Imports target:** False

### String References
| Line | Reference |
| ---- | --------- |
| 27 | `from prompt_toolkit.application import get_app` |
| 28 | `from prompt_toolkit.filters import Condition, has_focus` |
| 29 | `from prompt_toolkit.formatted_text import HTML` |
| 30 | `from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent` |
| 31 | `from prompt_toolkit.layout import (Container, FormattedTextControl, HSplit,` |
| 33 | `from prompt_toolkit.layout.containers import (AnyContainer,` |
| 35 | `from prompt_toolkit.mouse_events import MouseEventType` |
| 36 | `from prompt_toolkit.widgets import Box, Frame, Label` |

## openhcs/tui/utils.py
**Imports target:** False

### String References
| Line | Reference |
| ---- | --------- |
| 2 | `from prompt_toolkit.application import get_app` |
| 3 | `from prompt_toolkit.layout.containers import HSplit` |
| 4 | `from prompt_toolkit.widgets import Button, Dialog, Label, TextArea # Added TextA...` |
| 34 | `# Otherwise, use the general prompt_toolkit way.` |
| 37 | `# For now, we'll make it compatible with direct prompt_toolkit usage too.` |

## openhcs/tui/function_pattern_editor.py
**Imports target:** False

### String References
| Line | Reference |
| ---- | --------- |
| 24 | `from prompt_toolkit.application import get_app` |
| 25 | `from prompt_toolkit.formatted_text import HTML` |
| 26 | `from prompt_toolkit.layout import HSplit, ScrollablePane, VSplit, Container` |
| 27 | `from prompt_toolkit.widgets import (Box, Button, Dialog, Label, TextArea, RadioL...` |
| 400 | `from prompt_toolkit.widgets import Frame` |
| 496 | `from prompt_toolkit.formatted_text import HTML` |

## openhcs/tui/plate_manager_core.py
**Imports target:** False

### String References
| Line | Reference |
| ---- | --------- |
| 35 | `from prompt_toolkit.application import get_app` |
| 36 | `from prompt_toolkit.filters import Condition, has_focus` |
| 37 | `from prompt_toolkit.key_binding import KeyBindings` |
| 38 | `from prompt_toolkit.layout import Container, HSplit, VSplit, DynamicContainer, D...` |
| 39 | `from prompt_toolkit.widgets import Box, Button, Frame, Label # Removed TextArea` |

## openhcs/tui/tui_architecture.py
**Imports target:** False

### String References
| Line | Reference |
| ---- | --------- |
| 57 | `from prompt_toolkit import Application` |
| 58 | `from prompt_toolkit.application import get_app` |
| 59 | `from prompt_toolkit.filters import Condition, has_focus` |
| 60 | `from prompt_toolkit.key_binding import KeyBindings` |
| 61 | `from prompt_toolkit.layout import Container, HSplit, Layout, VSplit, Window` |
| 62 | `from prompt_toolkit.layout.containers import (DynamicContainer, Float,` |
| 64 | `from prompt_toolkit.widgets import Box, Button, Frame, Label, TextArea` |

## openhcs/tui/commands.py
**Imports target:** False

### String References
| Line | Reference |
| ---- | --------- |
| 65 | `from prompt_toolkit.shortcuts import message_dialog, DialogResult` |
| 66 | `from prompt_toolkit.application import get_app` |

## openhcs/tui/step_viewer.py
**Imports target:** False

### String References
| Line | Reference |
| ---- | --------- |
| 5 | `from prompt_toolkit.application import get_app` |
| 6 | `from prompt_toolkit.filters import has_focus` |
| 7 | `from prompt_toolkit.key_binding import KeyBindings` |
| 8 | `from prompt_toolkit.layout import HSplit, VSplit` |
| 9 | `from prompt_toolkit.widgets import Button, Frame, Label, TextArea, Box` |
| 10 | `from prompt_toolkit.layout import HSplit, VSplit, DynamicContainer, Container, D...` |

## openhcs/tui/status_bar.py
**Imports target:** False

### String References
| Line | Reference |
| ---- | --------- |
| 36 | `from prompt_toolkit.application import get_app` |
| 37 | `from prompt_toolkit.filters import Condition` |
| 38 | `from prompt_toolkit.formatted_text import FormattedText` |
| 39 | `from prompt_toolkit.layout import ConditionalContainer, Container, HSplit` |
| 40 | `from prompt_toolkit.mouse_events import MouseEventType` |
| 41 | `from prompt_toolkit.widgets import Box, Frame, Label` |

## openhcs/tui/dialogs/global_settings_editor.py
**Imports target:** False

### String References
| Line | Reference |
| ---- | --------- |
| 11 | `from prompt_toolkit.application import get_app` |
| 12 | `from prompt_toolkit.layout import HSplit, VSplit` |
| 13 | `from prompt_toolkit.widgets import Button, Dialog, Label, TextArea, RadioList, C...` |

## openhcs/tui/dialogs/plate_config_editor.py
**Imports target:** False

### String References
| Line | Reference |
| ---- | --------- |
| 13 | `from prompt_toolkit.application import get_app` |
| 14 | `from prompt_toolkit.layout import HSplit, VSplit, DynamicContainer, Container, D...` |
| 15 | `from prompt_toolkit.widgets import Box, Button, Frame, Label, TextArea, RadioLis...` |
| 16 | `from prompt_toolkit.formatted_text import HTML` |

## openhcs/tui/dialogs/plate_dialog_manager.py
**Imports target:** False

### String References
| Line | Reference |
| ---- | --------- |
| 26 | `from prompt_toolkit.application import get_app` |
| 27 | `from prompt_toolkit.filters import Condition, is_done` |
| 28 | `from prompt_toolkit.formatted_text import HTML` |
| 29 | `from prompt_toolkit.layout import (ConditionalContainer, Container, Float,` |
| 31 | `from prompt_toolkit.widgets import Box, Button, Dialog, Label` |
| 32 | `from prompt_toolkit.widgets import RadioList as Dropdown` |
| 33 | `from prompt_toolkit.widgets import TextArea` |

## openhcs/tui/services/external_editor_service.py
**Imports target:** False

### String References
| Line | Reference |
| ---- | --------- |
| 7 | `from prompt_toolkit.application import get_app` |
| 8 | `from prompt_toolkit.widgets import Dialog, Label, Button` |
| 9 | `from prompt_toolkit.layout import HSplit` |
| 41 | `# Use get_app().run_system_command for integration with prompt_toolkit's event l...` |
