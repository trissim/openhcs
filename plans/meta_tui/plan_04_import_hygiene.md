# Plan 04: TUI Import Hygiene and Structure Refinement

**Version**: 1.0
**Date**: 2025-05-23
**Author**: MasterMind Architect

## 1. Introduction & Goal

**Problem**: The `openhcs.tui` package, as revealed by the `reports/code_analysis/tui_comprehensive.md/import_analysis.md` report, suffers from numerous import-related issues:
    *   **Missing Imports**: Many symbols are used without being explicitly imported (e.g., in `components.py`, `pipeline_editor.py`). This often relies on symbols being implicitly available from wildcard imports or being injected into a module's namespace, leading to code that is hard to understand and prone to breakage.
    *   **Unused Imports**: Several modules import symbols that are never used (e.g., `Dialog`, `HTML`, `ast` in `components.py`; `FunctionPatternEditor` in `openhcs/tui/__init__.py`). This clutters the namespace and can mislead developers about module dependencies.
    *   **Module Structure Issues**: Some imports might point to internal structures of other modules or use non-standard import paths (e.g., `prompt_toolkit.layout.Container` being noted as an issue, though this might be a linter false positive if it's the canonical path).
    *   **Disorganized Imports**: Imports are not consistently grouped or ordered according to PEP 8, making it harder to quickly assess module dependencies.

**Goal**: To improve the clarity, maintainability, and robustness of the `openhcs.tui` package by:
    1.  Systematically addressing all missing and unused imports identified in the analysis report.
    2.  Organizing imports according to PEP 8 guidelines (standard library, third-party, local application).
    3.  Consolidating common TUI utility functions into dedicated utility modules to reduce scattered helper code and improve import paths.
    4.  Enforcing clearer module boundaries by ensuring modules only import what they directly need from public interfaces of other modules.

**Architectural Principles**:
*   **Explicitness**: Imports should clearly declare all external symbols a module uses.
*   **Minimality (Least Privilege for Imports)**: Modules should only import what they need.
*   **Standardization**: Adherence to PEP 8 for import formatting improves readability.
*   **Modularity**: Well-defined utility modules promote reuse and reduce code duplication.

## 2. Refactoring Steps

### 2.1. Address Missing Imports

*   **Action**: For each file listed in `import_analysis.md` under "Missing Imports":
    1.  Identify the source module for each missing symbol.
    2.  Add an explicit `from module import symbol` or `import module` statement.
    3.  Prioritize importing from public APIs of modules rather than internal submodules if possible.
*   **Key Files from Report**:
    *   `openhcs/tui/components.py`: Has a very large list of missing symbols (e.g., `buffer`, `current_kwargs`, `display_text_func`, `on_parameter_change`, `sig`). These likely originate from `prompt_toolkit` widgets or custom base classes whose attributes are accessed directly without being passed or imported. This requires careful investigation to determine if these are attributes of `self` or genuinely missing imports. If they are attributes, the linter might be misinterpreting. If they are from elsewhere, they must be imported.
    *   `openhcs/tui/pipeline_editor.py`: Missing `PlaceholderCommand`, `active_orchestrator`, etc.
    *   Other files as listed in the report.
*   **Tooling**: Use an IDE with import resolution capabilities or manually trace symbol origins.

### 2.2. Remove Unused Imports

*   **Action**: For each file listed in `import_analysis.md` under "Unused Imports":
    1.  Verify that the symbol is genuinely unused within the module.
    2.  Remove the corresponding import statement.
*   **Key Files from Report**:
    *   `openhcs/tui/components.py`: Unused `Dialog`, `HTML`, `KeyBindings`, `ScrollablePane`, `Tuple`, `Union`, `ast`.
    *   `openhcs/tui/__init__.py`: Unused `FunctionPatternEditor`.
    *   Other files as listed.
*   **Tooling**: Linters like `flake8` (with `flake8-import-order` and `flake8-unused-imports`) or `pylint` can automate detection. Autofixers like `autoflake` can remove them.

### 2.3. Organize Imports (PEP 8)

*   **Action**: For every Python file in `openhcs.tui`:
    1.  Group imports into three sections:
        *   Standard library imports (e.g., `os`, `typing`, `asyncio`).
        *   Third-party library imports (e.g., `prompt_toolkit`).
        *   Local application/library imports (e.g., `from openhcs.core.config import ...`, `from .components import ...`).
    2.  Within each section, sort imports alphabetically.
    3.  Separate sections with a blank line.
*   **Tooling**: `isort` is the standard tool for automatically sorting and formatting imports. Configure it project-wide.

### 2.4. Consolidate TUI Utilities

*   **Problem**: The current `openhcs/tui/utils.py` and `openhcs/tui/utils/dialog_helpers.py`, `openhcs/tui/utils/error_handling.py` exist. Review their content and the overall TUI for scattered utility functions.
*   **Action**:
    1.  **Review `openhcs/tui/utils/__init__.py`, `dialog_helpers.py`, `error_handling.py`**:
        *   Ensure `utils/__init__.py` correctly exports symbols from its submodules if intended for public use by other TUI components.
        *   Consolidate genuinely general TUI utilities into `openhcs.tui.utils.py` or keep them in specific submodules like `dialog_helpers.py` if they are cohesive.
    2.  **Identify other scattered utilities**: Look for small helper functions within larger component files that could be generalized and moved to a relevant utility module.
    3.  **Refactor imports**: Update TUI components to import utilities from these centralized locations.
*   **Example**: If `show_error_dialog` is used by many components, ensure it's easily importable from `openhcs.tui.utils`.

### 2.5. Address Module Structure Issues (from report)

*   **File**: `openhcs/tui/components.py`
*   **Issue**: `Line 1: prompt_toolkit.layout.Container - Container is imported from prompt_toolkit.layout` (and similar on line 8).
*   **Analysis**: This specific message from the linter might be a false positive or a style suggestion if `prompt_toolkit.layout.Container` is the canonical way to import `Container`. However, it prompts a review:
    1.  Verify the recommended import path for `Container` from `prompt_toolkit` documentation.
    2.  If `from prompt_toolkit.layout import Container` is standard, this "issue" can be ignored or the linter rule adjusted.
    3.  If there's a more direct or preferred import path (e.g., `from prompt_toolkit.widgets import Container` if it exists and is equivalent), update the import.
*   **General Action**: For any other "Module Structure Issues" identified by linters or manual review, ensure imports use the most direct and public paths to symbols. Avoid importing from `_internal` or deeply nested modules of external libraries if a more public API is available.

## 3. Verification

1.  **Static Analysis (Primary)**:
    *   Re-run `python tools/code_analysis/meta_analyzer.py comprehensive openhcs/tui -o reports/code_analysis/tui_comprehensive.md` (or specifically the import analysis part: `python tools/code_analysis/../import_analysis/import_validator.py openhcs/tui -o updated_import_analysis.md`).
    *   Verify that the number of "Missing Imports" and "Unused Imports" is zero or acceptably minimal.
    *   Manually inspect a selection of files to confirm PEP 8 import ordering.
2.  **Linter Pass**: Run `flake8` and `pylint` (if configured) over the `openhcs.tui` package. Address any new import-related warnings or errors.
3.  **Functionality Tests**: Run existing unit and integration tests for the TUI. While import changes should ideally not break functionality if done correctly, they can sometimes reveal hidden issues or incorrect assumptions about symbol availability.
4.  **Code Review**:
    *   Focus on the clarity of imports.
    *   Ensure utility functions are appropriately placed and imported.
    *   Check for any remaining reliance on implicitly available symbols.

This plan will lead to a cleaner, more explicit, and more maintainable import structure for the `openhcs.tui` package.