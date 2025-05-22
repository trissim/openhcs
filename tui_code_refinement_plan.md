# Plan to Address Identified TUI Code Stubs & Implementation Areas

## I. Overview & Goal
The primary goal is to enhance the robustness, completeness, and user experience of the OpenHCS TUI by addressing potential stubs, unimplemented logic, and areas for improvement identified in the Python source code.

## II. Categorization & Prioritization of Issues

```mermaid
graph TD
    A[TUI Code Refinement Plan] --> B{Categories};
    B --> C[1. Resource Management & Cleanup];
    B --> D[2. Core Functionality Gaps];
    B --> E[3. UX & Error Handling];
    B --> F[4. Design/Config Considerations];

    C --> C1(["tui_architecture.py:363 - FPE Cleanup"]);
    C --> C2(["tui_launcher.py:187 - Orchestrator Instance Cleanup"]);
    C --> C3(["tui_launcher.py:265 - Orchestrator General Cleanup"]);

    D --> D1(["tui_architecture.py:53 - TUIState 'pass'"]);
    D --> D2(["action_menu_pane.py:324 - Process execution_results"]);
    D --> D3(["plate_manager_core.py:553 - Status update in 'finally'"]);

    E --> E1(["action_menu_pane.py:728 - In-dialog error display"]);
    E2(["plate_manager_core.py:402 - 'pass' in 'except' (review)"]);
    E --> E3(["dialogs/plate_dialog_manager.py:334,397 - 'pass' for focus error (review)"]);

    F --> F1(["tui_launcher.py:53 - FileManager VFSConfig"]);

    subgraph Legend
        L1[P1: Critical]
        L2[P2: High]
        L3[P3: Medium]
        L4[P4: Low]
    end

    style C fill:#ffcccb,stroke:#333,stroke-width:2px
    style D fill:#ffe0b3,stroke:#333,stroke-width:2px
    style E fill:#b3e0ff,stroke:#333,stroke-width:2px
    style F fill:#e6ccff,stroke:#333,stroke-width:2px

    linkStyle 0 stroke-width:0px;
    linkStyle 1 stroke-width:0px;
    linkStyle 2 stroke-width:0px;
    linkStyle 3 stroke-width:0px;
    linkStyle 4 stroke-width:0px;
    linkStyle 5 stroke-width:0px;
    linkStyle 6 stroke-width:0px;
    linkStyle 7 stroke-width:0px;
    linkStyle 8 stroke-width:0px;
    linkStyle 9 stroke-width:0px;
    linkStyle 10 stroke-width:0px;
    linkStyle 11 stroke-width:0px;
    linkStyle 12 stroke-width:0px;
    linkStyle 13 stroke-width:0px;
    linkStyle 14 stroke-width:0px;
    linkStyle 15 stroke-width:0px;
    linkStyle 16 stroke-width:0px;
    linkStyle 17 stroke-width:0px;
    linkStyle 18 stroke-width:0px;
    linkStyle 19 stroke-width:0px;
```

*   **Priority 1: Resource Management & Cleanup (Critical for Stability)**
*   **Priority 2: Core Functionality Gaps (Essential for Features)**
*   **Priority 3: UX & Error Handling (Important for Usability)**
*   **Priority 4: Design/Configuration Considerations (Refinements)**

*(Note: `openhcs/tui/function_pattern_editor.py:78` `pass` in `PatternValidationError` is standard for custom exceptions and likely requires no action, so it's omitted from this active plan).*

## III. Detailed Action Plan

**A. Category 1: Resource Management & Cleanup**
    1.  **Item:** `openhcs/tui/tui_architecture.py:363` - `TODO: If FPE has a close/cleanup method, call it here.`
        *   **Action:** Investigate `FunctionPatternEditor` (FPE) class. Determine if it manages resources that require explicit cleanup (e.g., event listeners, temporary files, background tasks).
        *   **If yes:** Implement a `close()` or `cleanup()` method in FPE and call it at the specified TODO location.
        *   **If no:** Document this finding and remove the TODO.
    2.  **Item:** `openhcs/tui/tui_launcher.py:187` - `TODO: Add any specific cleanup for the orchestrator instance if needed`
        *   **Action:** Analyze the `Orchestrator` class. Identify any resources held by an instance (e.g., open files, network connections, child processes/threads not managed by `atexit` or similar).
        *   **If resources exist:** Implement necessary cleanup logic when an orchestrator instance is removed from `self.orchestrators`.
        *   **If no specific cleanup needed:** Document and remove TODO.
    3.  **Item:** `openhcs/tui/tui_launcher.py:265` - `TODO: Implement proper cleanup for each orchestrator if needed (e.g., shutting down thread pools, releasing resources they might hold)`
        *   **Action:** This is a broader version of the above. Review the overall lifecycle of orchestrators. If they use shared resources like thread pools that need explicit shutdown when *all* orchestrators are done or the TUI exits, implement this logic. This might involve a dedicated shutdown sequence in `TuiLauncher`.

**B. Category 2: Core Functionality Gaps/Placeholders**
    1.  **Item:** `openhcs/tui/tui_architecture.py:53` - `pass` statement within the `TUIState` class definition.
        *   **Action:** Review the intended purpose of `TUIState`. If it's meant to hold shared TUI state, define necessary attributes and methods. If it's a simple marker or base class that's complete as is, add a comment clarifying its purpose and that the `pass` is intentional for an empty class body.
    2.  **Item:** `openhcs/tui/action_menu_pane.py:324` - `TODO: Process execution_results if it contains meaningful data or error summaries per well.`
        *   **Action:** Determine the structure of `execution_results`. Implement logic to parse this data, extract meaningful information (e.g., success/failure per well, error messages), and update the TUI accordingly (e.g., display summaries, update status indicators).
    3.  **Item:** `openhcs/tui/plate_manager_core.py:553` - `pass` statement in a `finally` block with comment "For now, we'll just keep the current status."
        *   **Action:** Re-evaluate if any status update or cleanup action is *always* required in this `finally` block, regardless of success or failure of the `try` block. If so, implement it. If not, clarify the comment or remove the `pass` if the `finally` block becomes non-empty or is deemed unnecessary.

**C. Category 3: UX & Error Handling Improvements**
    1.  **Item:** `openhcs/tui/action_menu_pane.py:728` - `TODO: Display these errors within the dialog itself for better UX.`
        *   **Action:** Design and implement a mechanism to display validation errors (currently sent to the pane's error banner) directly within the active dialog that triggered them. This might involve passing error messages back to the dialog or having dialogs subscribe to specific error events.
    2.  **Item:** `openhcs/tui/plate_manager_core.py:402` - `pass` in `except` block (comment: "Error already handled by validation service").
        *   **Action:** Confirm that the validation service's handling is comprehensive and that no additional logging (e.g., for debugging or audit trails at this specific point) or state change is beneficial here. If truly redundant, the `pass` is acceptable but could be commented more clearly.
    3.  **Item:** `openhcs/tui/dialogs/plate_dialog_manager.py:334` & `:397` - `pass` in `except KeyError` for focus restoration.
        *   **Action:** Evaluate if "silently continuing" on focus restoration failure is always the best user experience. Consider adding minimal logging (e.g., `logger.debug("Focus restoration failed for X")`) to aid in diagnosing potential focus issues, without disrupting the user.

**D. Category 4: Design/Configuration Considerations**
    1.  **Item:** `openhcs/tui/tui_launcher.py:53` - `TODO: Consider if FileManager needs VFSConfig from core_global_config`
        *   **Action:** Analyze the `FileManager`'s responsibilities and how it interacts with the virtual file system (VFS). Determine if passing `VFSConfig` (presumably from a global configuration) would enable more flexible or correct behavior. If so, refactor to include it. If not, document why it's not needed and remove the TODO.

## IV. Implementation & Verification
*   Address items category by category, or by individual priority if preferred.
*   For each item, once changes are made:
    *   Perform unit/integration testing relevant to the change.
    *   Manually test the TUI to observe the impact of the change, especially for UX items.
    *   Ensure no regressions are introduced.

## V. Documentation
*   Update any relevant code comments.
*   If significant design decisions are made, consider documenting them in appropriate project documentation.