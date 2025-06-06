# plan_03_state_synchronization.md
## Component: Unified State Management and Synchronization

### Objective
Establish a unified state synchronization system that keeps SimpleTUIState, PlateManagerPane state, OrchestratorManager state, and individual orchestrator states properly synchronized through event-driven coordination.

### Plan
1. **Analyze Current State Architecture (1 hour)**
   - Map all state objects and their responsibilities
   - Trace event flow between state objects
   - Identify synchronization gaps and conflicts
   - Document current observer pattern implementations

2. **Standardize State Event Contracts (1.5 hours)**
   - Define canonical event signatures for state changes
   - Ensure SimpleTUIState and TUIState compatibility
   - Standardize plate status change events
   - Create consistent orchestrator status events

3. **Implement State Synchronization Coordinator (2 hours)**
   - Create StateSynchronizationCoordinator class
   - Implement bidirectional state sync between components
   - Add conflict resolution for competing state updates
   - Ensure atomic state transitions

4. **Fix Component State Integration (1.5 hours)**
   - Fix PlateManagerPane state sync with SimpleTUIState
   - Ensure OrchestratorManager state reflects in UI
   - Sync individual orchestrator status with plate status
   - Fix active_orchestrator management

5. **Add State Validation and Recovery (1 hour)**
   - Add state consistency validation
   - Implement recovery mechanisms for state conflicts
   - Add logging for state synchronization issues
   - Test state recovery scenarios

### Findings

#### Current State Architecture Analysis

**State Objects Identified (HIGH CONFIDENCE)**
1. **SimpleTUIState** - Main TUI state in simple_launcher.py
   - Observer pattern implementation
   - Core properties: selected_plate, is_compiled, is_running
   - Event emission via notify() method

2. **PlateManagerPane Internal State** - In service/controller/view layers
   - Plate list management
   - Selected plate tracking
   - Status management per plate

3. **OrchestratorManager State** - Thread-safe orchestrator storage
   - Dictionary of plate_id → PipelineOrchestrator
   - Thread-safe access with RLock
   - Status tracking methods

4. **Individual Orchestrator State** - PipelineOrchestrator internal state
   - Initialization status
   - Compilation results (last_compiled_contexts)
   - Pipeline definition

**Event Flow Architecture (MEDIUM CONFIDENCE)**
```
PlateManagerService → 'plate_added' → SimpleTUIState
  → PlateOrchestratorCoordinationBridge → OrchestratorManager

Commands → orchestrator operations → status events → SimpleTUIState
  → UI components (StatusBar, PlateManagerPane)

UI interactions → SimpleTUIState → component updates
```

#### Critical Synchronization Issues

**Issue 1: State Object Compatibility**
- SimpleTUIState vs TUIState interface differences
- Event signature mismatches between components
- Observer pattern implementations not fully compatible
- Type hints reference different state classes

**Issue 2: Bidirectional Sync Gaps**
- Orchestrator status changes don't always update plate status
- Plate status changes may not reflect orchestrator state
- UI state changes may not propagate to backend state
- Active orchestrator selection sync unclear

**Issue 3: Race Conditions**
- Multiple async operations updating state simultaneously
- Event emission during state transitions
- Observer callbacks modifying state during iteration
- Thread safety between UI and background operations

**Issue 4: State Consistency Validation**
- No validation that state objects remain consistent
- Conflicting state updates may overwrite each other
- No recovery mechanism for inconsistent state
- State corruption detection missing

#### Component State Integration Analysis

**SimpleTUIState Integration (70% confidence)**
- Well-designed observer pattern
- Clean event emission interface
- Used by multiple components
- May have compatibility issues with production components

**PlateManagerPane State Integration (50% confidence)**
- MVC architecture with internal state management
- Emits events to SimpleTUIState
- State sync with service layer unclear
- UI update coordination unclear

**OrchestratorManager Integration (60% confidence)**
- Thread-safe state management
- Registration/unregistration methods exist
- Status tracking basic but functional
- Integration with event system unclear

**StatusBar State Integration (80% confidence)**
- Well-designed immutable state pattern
- Event-driven updates from SimpleTUIState
- Clean separation of concerns
- Good integration patterns

#### State Synchronization Confidence Assessment
- **Event Flow Architecture**: 65% - Basic patterns exist but gaps identified
- **State Consistency**: 30% - No validation or conflict resolution
- **Bidirectional Sync**: 40% - Some sync exists but incomplete
- **Race Condition Handling**: 25% - Basic async patterns but no coordination
- **Error Recovery**: 20% - No recovery mechanisms identified

### Implementation Draft
*Implementation will be added after smell loop approval*
