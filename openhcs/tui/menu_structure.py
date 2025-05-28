"""
Menu structure definitions for OpenHCS TUI.

This module contains the declarative menu structure that defines
the hierarchical menu system for the TUI application.

🔒 Clause 3: Declarative Primacy
Menu structure is defined declaratively and loaded by MenuBar.

🔒 Clause 245: Modular Architecture
Menu structure is separated from menu logic for better modularity.
"""

# Default menu structure for OpenHCS TUI
# Matches tui_final.md specification for hierarchical menu system
DEFAULT_MENU_STRUCTURE = {
    "File": {
        "mnemonic": "F",  # For Alt+F activation
        "items": [
            {"type": "command", "label": "&New Pipeline", "handler": "_on_new_pipeline", "shortcut": "Ctrl+N"},
            {"type": "command", "label": "&Open Pipeline...", "handler": "_on_open_pipeline", "shortcut": "Ctrl+O"},
            {"type": "command", "label": "&Save Pipeline", "handler": "_on_save_pipeline", "enabled": "is_compiled"},
            {"type": "separator"},
            {"type": "command", "label": "E&xit", "handler": "_on_exit"}
        ]
    },
    "Edit": {
        "mnemonic": "E",
        "items": [
            {"type": "command", "label": "&Add Step", "handler": "_on_add_step", "enabled": "has_selected_step"},
            {"type": "command", "label": "Edit Ste&p", "handler": "_on_edit_step", "enabled": "has_selected_step"},
            {"type": "command", "label": "&Remove Step", "handler": "_on_remove_step", "enabled": "has_selected_step"},
        ]
    },
    "View": {
        "mnemonic": "V",
        "items": [
            {"type": "checkbox", "label": "&Vim Mode", "handler": "_on_toggle_vim_mode", "checked": "vim_mode"},
            {"type": "checkbox", "label": "&Log Drawer", "handler": "_on_toggle_log_drawer", "checked": "log_drawer_expanded"},
            {"type": "separator"},
            {"type": "submenu", "label": "&Theme", "children": [
                {"type": "checkbox", "label": "&Light", "handler": "_on_set_theme_light", "checked": "theme_is_light"},
                {"type": "checkbox", "label": "&Dark", "handler": "_on_set_theme_dark", "checked": "theme_is_dark"},
                {"type": "checkbox", "label": "&System", "handler": "_on_set_theme_system", "checked": "theme_is_system"},
            ]}
        ]
    },
    "Pipeline": {
        "mnemonic": "P",
        "items": [
            {"type": "command", "label": "Pre-&compile", "handler": "_on_pre_compile"},
            {"type": "command", "label": "&Compile", "handler": "_on_compile"},
            {"type": "command", "label": "&Run", "handler": "_on_run", "enabled": "is_compiled"},
            {"type": "separator"},
            {"type": "command", "label": "Global Se&ttings...", "handler": "_on_settings"}
        ]
    },
    "Help": {
        "mnemonic": "H",
        "items": [
            # Consolidated into a single "Help" item per tui_final.md
            {"type": "command", "label": "&View Help", "handler": "_on_show_help"}
        ]
    }
}

# Menu condition definitions
# These define when menu items should be enabled/checked
MENU_CONDITIONS = {
    "is_compiled": "state.is_compiled",
    "has_selected_step": "state.pipeline_editor.has_selected_step",
    "vim_mode": "state.vim_mode_enabled",
    "log_drawer_expanded": "state.log_drawer_expanded",
    "theme_is_light": "state.theme == 'light'",
    "theme_is_dark": "state.theme == 'dark'",
    "theme_is_system": "state.theme == 'system'"
}

# Menu handler mappings
# Maps handler names to actual method names
MENU_HANDLERS = {
    "_on_new_pipeline": "new_pipeline",
    "_on_open_pipeline": "open_pipeline", 
    "_on_save_pipeline": "save_pipeline",
    "_on_exit": "exit_application",
    "_on_add_step": "add_step",
    "_on_edit_step": "edit_step",
    "_on_remove_step": "remove_step",
    "_on_toggle_vim_mode": "toggle_vim_mode",
    "_on_toggle_log_drawer": "toggle_log_drawer",
    "_on_set_theme_light": "set_theme_light",
    "_on_set_theme_dark": "set_theme_dark",
    "_on_set_theme_system": "set_theme_system",
    "_on_pre_compile": "pre_compile",
    "_on_compile": "compile_pipeline",
    "_on_run": "run_pipeline",
    "_on_settings": "show_global_settings",
    "_on_show_help": "show_help"
}
