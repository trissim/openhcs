# Async Pattern Analysis for menu_bar.py

## Async Functions

| Name | Line | Class | Has Return Type |
|------|------|-------|----------------|
| __aenter__ | 65 | ReentrantLock | False |
| __aexit__ | 81 | ReentrantLock | False |
| handler | 441 | MenuBar | True |
| _activate_menu | 720 | MenuBar | True |
| _close_menu | 758 | MenuBar | True |
| _navigate_menu | 778 | MenuBar | True |
| _navigate_submenu | 801 | MenuBar | True |
| _select_current_item | 843 | MenuBar | True |
| _handle_menu_item | 917 | MenuBar | True |
| _on_new_pipeline | 953 | MenuBar | True |
| _on_open_pipeline | 959 | MenuBar | True |
| _on_save_pipeline | 965 | MenuBar | True |
| _on_save_pipeline_as | 1025 | MenuBar | True |
| _on_exit | 1031 | MenuBar | True |
| _on_add_step | 1035 | MenuBar | True |
| _on_edit_step | 1041 | MenuBar | True |
| _on_remove_step | 1079 | MenuBar | True |
| _on_move_step_up | 1126 | MenuBar | True |
| _on_move_step_down | 1132 | MenuBar | True |
| _on_toggle_log_drawer | 1138 | MenuBar | True |
| _on_toggle_vim_mode | 1142 | MenuBar | True |
| _on_set_theme | 1146 | MenuBar | True |
| _on_pre_compile | 1155 | MenuBar | True |
| _on_compile | 1159 | MenuBar | True |
| _on_run | 1163 | MenuBar | True |
| _on_test | 1167 | MenuBar | True |
| _on_operation_status_changed | 1194 | MenuBar | True |
| _on_plate_selected | 1204 | MenuBar | True |
| _on_is_compiled_changed | 1214 | MenuBar | True |
| shutdown | 1241 | MenuBar | False |

## Async Functions Without Return Type Annotations

| Name | Line | Class |
|------|------|-------|
| __aenter__ | 65 | ReentrantLock |
| __aexit__ | 81 | ReentrantLock |
| shutdown | 1241 | MenuBar |

## Unawaited Coroutines

No unawaited coroutines found.

## Awaited Coroutines

| Name | Line | Function | Class |
|------|------|----------|-------|
| execute | 443 | handler | MenuBar |
| execute | 936 | _handle_menu_item | MenuBar |
