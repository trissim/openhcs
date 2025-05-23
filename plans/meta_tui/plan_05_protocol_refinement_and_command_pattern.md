# Plan 05: TUI Protocol Refinement and Command Pattern Solidification

**Version**: 1.0
**Date**: 2025-05-23
**Author**: MasterMind Architect

## 1. Introduction & Goal

**Problem**: The `openhcs.tui` package uses Python `Protocol`s for defining interfaces, notably for commands and callbacks. However, the `reports/code_analysis/tui_comprehensive.md/interface_analysis.md` report indicates that some classes implement a large number of these protocols (e.g., command classes implementing `Command`, `PlateEventHandler`, `DialogResultCallback`, `ErrorCallback`, `ValidationResultCallback`). This suggests that the current protocols might be too broad or that classes are taking on too many disparate roles. Furthermore, the `Command` protocol itself is very generic. The `semantic_role_analysis.md` report also shows a mix of roles within UI components, some of which might be better encapsulated by more specific command objects or delegated through clearer interfaces.

**Goal**: To refine the existing TUI protocols and solidify the Command pattern by:
    1.  Reviewing and potentially specializing the `Command` protocol and other key protocols (like `DialogResultCallback`, `ErrorCallback`, `PlateEventHandler`) to better reflect specific interaction types.
    2.  Ensuring that classes implement only the protocols relevant to their primary responsibility, potentially by delegating secondary responsibilities to collaborator objects.
    3.  Clarifying the data contracts (arguments and return types) for protocol methods, using `CorePlateData`, `CoreStepData` (from Plan 01), or other TUI-specific data transfer objects (DTOs) instead of raw `Dict[str, Any]` where appropriate.
    4.  Ensuring command objects are pure encapsulations of an action, taking necessary context (like adapters and `TUIState`) during execution rather than holding excessive state themselves.

**Architectural Principles**:
*   **Interface Segregation Principle (ISP)**: Clients should not be forced to depend on interfaces they do not use. Protocols should be fine-grained.
*   **Single Responsibility Principle (SRP)**: Classes (including commands) should have one primary responsibility.
*   **Command Pattern**: Encapsulate a request as an object, thereby letting you parameterize clients with different requests, queue or log requests, and support undoable operations.
*   **Clear Contracts**: Interfaces should clearly define the expected inputs and outputs.

## 2. Proposed Protocol and Command Refinements

### 2.1. Review and Specialize `Command` Protocol

*   **Current**: `Command(Protocol)` with `async def execute(self, state: "TUIState", context: "ProcessingContext", **kwargs: Any)` and `def can_execute(self, state: "TUIState")`.
*   **Observation**: The `context: "ProcessingContext"` and `**kwargs` are very general. Plan 01 proposes changing `execute` to receive adapter interfaces.
*   **Proposed Action**:
    1.  **Base `TUICommand(Protocol)`**:
        ```python
        # In openhcs.tui.interfaces.py (or a new openhcs.tui.commands.protocols.py)
        class TUICommand(Protocol):
            async def execute(self, app_adapter: "CoreApplicationAdapterInterface",
                              plate_adapter: Optional["CoreOrchestratorAdapterInterface"], # If relevant to current context
                              ui_state: "TUIState", # Renamed for clarity
                              **kwargs: Any) -> None: ... # kwargs for command-specific params

            def can_execute(self, ui_state: "TUIState",
                            app_adapter: Optional["CoreApplicationAdapterInterface"] = None, # Optional for simple checks
                            plate_adapter: Optional["CoreOrchestratorAdapterInterface"] = None
                           ) -> bool:
                return True # Default
        ```
    2.  **Specialized Command Protocols (Examples, if beneficial)**:
        *   `DialogCommand(TUICommand)`: For commands that primarily show dialogs. Might have specific methods or properties related to dialog management if common patterns emerge.
        *   `PlateOperationCommand(TUICommand)`: For commands operating on the active plate/orchestrator. `plate_adapter` would be non-optional in `execute`.
        *   `StepOperationCommand(TUICommand)`: For commands operating on steps within a plate.
    *   **Rationale**: Specialization can make command types more explicit and allow for type-checking specific command categories if needed by invokers (e.g., a `MenuBuilder` might only accept `MenuCommand` types). However, over-specialization should be avoided if the base `TUICommand` is sufficient. Start with the refined `TUICommand` and introduce specializations only if clear benefits arise.

### 2.2. Refine Callback Protocols

*   **Current**: `DialogResultCallback`, `ErrorCallback`, `ValidationResultCallback`, `PlateEventHandler` are defined as `Protocol`s. Many classes implement all of them.
*   **Observation**: This wide implementation suggests these might be too generic or that components are acting as catch-alls for many event types.
*   **Proposed Action**:
    1.  **Review Necessity**: Determine if all these distinct callback protocols are truly needed or if some can be consolidated or made more specific to the component that *emits* the event.
    2.  **`DialogResultCallback(data: Dict[str, Any]) -> None`**:
        *   Refine `data` to be more specific if possible (e.g., `DialogOutputData(TypedDict)` or a Pydantic model).
        *   Ensure that only components directly responsible for handling a dialog's outcome implement this.
    3.  **`ErrorCallback(message: str, details: str = "") -> None`**:
        *   This seems reasonable. Ensure it's used consistently for UI-level error reporting.
    4.  **`ValidationResultCallback(data: Dict[str, Any]) -> None`**:
        *   Refine `data` (e.g., `ValidationOutputData(TypedDict)`).
    5.  **`PlateEventHandler`**:
        *   `async def on_plate_added(self, plate: Dict[str, Any]) -> None:` -> `plate: CorePlateData`
        *   `async def on_plate_removed(self, plate: Dict[str, Any]) -> None:` -> `plate_id: str` (or `CorePlateData` if full data is useful)
        *   `async def on_plate_selected(self, plate: Dict[str, Any]) -> None:` -> `plate_data: Optional[CorePlateData]`
        *   `async def on_plate_status_changed(self, plate_id: str, status: str) -> None:` (seems okay, maybe add `message: Optional[str]`)
    *   **Decoupling Implementation**: Instead of one class implementing many callbacks, consider:
        *   Using the `TUIState` event bus more: Components emit specific events, and interested controllers subscribe.
        *   If a component truly needs to handle multiple types of results, it can have distinct methods that are registered as callbacks, rather than formally implementing many broad protocols.

### 2.3. Data Contracts for Protocol Methods

*   **Action**: Review all protocol methods (especially in `CoreApplicationAdapterInterface` and `CoreOrchestratorAdapterInterface` from Plan 01, and the callback protocols).
*   Replace generic `Dict[str, Any]`, `Any`, `List[Dict]` with more specific types:
    *   `CorePlateData` (defined in Plan 01 interfaces)
    *   `CoreStepData` (defined in Plan 01 interfaces)
    *   New `TypedDict` or Pydantic models for other structured data passed between TUI and adapters (e.g., for function details, configuration deltas).
*   **Example**:
    *   `CoreOrchestratorAdapterInterface.update_config(self, config_delta: Dict[str, Any])` could become `update_config(self, config_delta: PlateConfigUpdateData)`.

### 2.4. Command Object State and Responsibilities

*   **Action**:
    1.  **Statelessness**: Commands should ideally be stateless or hold minimal state directly related to their specific invocation (e.g., parameters passed via `**kwargs` to `execute`). They should derive necessary information from `ui_state` and adapters during execution.
    2.  **Single Action**: Each command class should encapsulate a single, well-defined user action. If a command becomes too complex, consider breaking it into smaller, more focused commands that can be composed.
    3.  **Constructor**: Command `__init__` methods should be lightweight, primarily for setting up any fixed parameters. Avoid complex logic or state loading in constructors. The `state` and `context` (now adapters) are passed to `execute`.
    4.  **`can_execute` Logic**: This method should rely on `ui_state` and potentially query adapters (if the condition depends on core state). It should be efficient as it might be called frequently to update UI element enablement.

## 3. Refactoring Steps

1.  **Create/Update Interface Files**:
    *   Modify `openhcs/tui/interfaces.py` (from Plan 01) to include the refined `TUICommand` protocol and any specialized command protocols.
    *   Update or create new files for refined callback protocols if necessary (e.g., `openhcs.tui.callbacks.py`).
    *   Define `TypedDict` or Pydantic models for data contracts in a shared TUI types module (e.g., `openhcs.tui.types.py`).
2.  **Refactor Command Classes (`openhcs.tui.commands.py` and submodules)**:
    *   Update command classes to inherit from the new `TUICommand` (or specialized versions).
    *   Modify `execute` and `can_execute` signatures.
    *   Replace direct core interactions with calls to adapter methods.
    *   Use new DTOs/TypedDicts for data exchange with adapters.
3.  **Refactor Components Implementing Callbacks**:
    *   Review classes that implement multiple callback protocols.
    *   If a class is genuinely responsible for handling diverse events, ensure its handler methods are clear and focused.
    *   If a class is acting as a "catch-all," refactor to delegate handling to more appropriate components or use more specific event subscriptions via `TUIState`.
    *   Update callback method signatures to use refined data contracts.
4.  **Update Command Instantiation and Execution**:
    *   Modify `CommandRegistry` (if used) or any direct command invokers (e.g., in `MenuBar`, toolbars from Plan 02) to pass the correct adapter interfaces and `TUIState` to `command.execute()`.
    *   Ensure `can_execute` is called with the necessary arguments.

## 4. Verification

1.  **Static Type Checking**: Run `mypy` on the `openhcs.tui` package. Verify that type hints for protocols, commands, and callbacks are consistent and that type errors are resolved.
2.  **Interface Analysis**: Re-run `python tools/code_analysis/interface_classifier.py openhcs/tui -o updated_interface_analysis.md`.
    *   Check that the new/refined protocols are correctly identified.
    *   Analyze the "Implementations" section to see if classes now implement a more focused set of protocols.
3.  **Semantic Role Analysis**: Re-run `python tools/code_analysis/semantic_role_analyzer.py openhcs/tui -o updated_semantic_role_analysis.md`.
    *   Assess if command classes now primarily contain `ACTION_DISPATCH` or `LOGIC_EXECUTION` roles related to their specific command, and less of other roles.
4.  **Code Review**:
    *   Ensure protocol methods have clear, typed signatures.
    *   Verify that commands are largely stateless and encapsulate single actions.
    *   Check that callback implementations are focused.
5.  **Functional Tests**: Ensure TUI interactions, dialogs, and command-driven actions still function correctly.

This plan will lead to a more robust and maintainable command and event handling system within the TUI, with clearer contracts and better separation of responsibilities.