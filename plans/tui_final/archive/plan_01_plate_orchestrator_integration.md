# plan_01_plate_orchestrator_integration.md
## Component: Plate-Orchestrator Integration Bridge

### Objective
Fix the critical integration gap between PlateManagerPane and OrchestratorManager to ensure proper orchestrator lifecycle management and state synchronization when plates are added, removed, or selected in the TUI.

### Plan
1. **Analyze Current Integration State (30 minutes)**
   - Verify PlateManagerService creates orchestrators correctly
   - Confirm PlateOrchestratorCoordinationBridge exists and functions
   - Test event flow from plate addition to orchestrator registration
   - Identify specific integration failures

2. **Fix PlateManagerService Integration (1 hour)**
   - Ensure PlateManagerService.add_plate() creates orchestrators properly
   - Verify orchestrator instances are stored in plate data
   - Fix any issues with orchestrator creation parameters
   - Ensure proper error handling for orchestrator creation failures

3. **Enhance PlateOrchestratorCoordinationBridge (1.5 hours)**
   - Verify bridge initialization in canonical_layout.py
   - Fix event registration and handler methods
   - Ensure proper orchestrator registration with OrchestratorManager
   - Add robust error handling and logging

4. **Fix State Synchronization (1 hour)**
   - Ensure SimpleTUIState.active_orchestrator is properly set
   - Fix plate selection event flow
   - Verify status updates propagate correctly
   - Test orchestrator status changes update plate UI

5. **Add Integration Tests (1 hour)**
   - Test plate addition → orchestrator creation flow
   - Test plate selection → active orchestrator update
   - Test orchestrator status → plate status synchronization
   - Verify error handling paths

### Findings

#### Current Integration Architecture
Based on code analysis, the integration architecture is more sophisticated than initially assessed:

**PlateManagerService Integration (HIGH CONFIDENCE)**
- PlateManagerService.add_plate() DOES create PipelineOrchestrator instances
- Orchestrators are stored in plate data: `{'orchestrator': orchestrator}`
- Service emits 'plate_added' events with orchestrator data

**PlateOrchestratorCoordinationBridge (MEDIUM CONFIDENCE)**  
- Bridge exists and is initialized in canonical_layout.py
- Registers for 'plate_added', 'plates_removed', 'plate_selected' events
- Has register_existing_orchestrator() method for coordination

**Event Flow Architecture (MEDIUM CONFIDENCE)**
```
PlateManagerService.add_plate() 
  → Creates PipelineOrchestrator
  → Emits 'plate_added' event with orchestrator
  → PlateOrchestratorCoordinationBridge.on_plate_added()
  → OrchestratorManager.register_existing_orchestrator()
  → Updates SimpleTUIState.active_orchestrator
```

#### Critical Issues Identified

**Issue 1: Bridge Initialization Timing**
- Bridge is initialized asynchronously in canonical_layout.py
- May not be ready when first plates are added
- Could cause missed events during startup

**Issue 2: State Object Compatibility**
- SimpleTUIState vs TUIState compatibility unclear
- Bridge expects specific state structure
- Event signatures may not match

**Issue 3: Error Handling Gaps**
- Orchestrator creation failures may not propagate properly
- Bridge errors may be silently ignored
- No fallback mechanisms for failed integration

**Issue 4: Active Orchestrator Management**
- SimpleTUIState.active_orchestrator setting unclear
- Plate selection → active orchestrator flow untested
- Multiple plates selection handling missing

#### Integration Confidence Assessment
- **PlateManagerService**: 85% - Creates orchestrators correctly
- **CoordinationBridge**: 60% - Exists but integration untested  
- **Event Flow**: 40% - Architecture sound but execution unclear
- **State Sync**: 30% - Multiple state objects, sync mechanism unclear

### Implementation Draft
*Implementation will be added after smell loop approval*
