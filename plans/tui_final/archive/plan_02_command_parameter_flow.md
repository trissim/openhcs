# plan_02_command_parameter_flow.md
## Component: Command Parameter Flow and Orchestrator Access

### Objective
Fix the command execution flow to ensure commands can properly access orchestrator instances for init/compile/run operations, with clear parameter passing and error handling.

### Plan
1. **Analyze Current Command Architecture (45 minutes)**
   - Map command registration in canonical_layout.py
   - Trace parameter flow from button clicks to command execution
   - Identify how commands access orchestrator instances
   - Test enhanced compilation command integration

2. **Fix Command Parameter Passing (1.5 hours)**
   - Ensure commands receive proper orchestrator references
   - Fix _get_selected_orchestrators() methods in commands
   - Verify OrchestratorManager integration in command execution
   - Add proper parameter validation

3. **Enhance Command-Orchestrator Integration (2 hours)**
   - Fix InitializePlatesCommand orchestrator access
   - Verify CompilePlatesCommand/EnhancedCompilePlatesCommand integration
   - Fix RunPlatesCommand orchestrator and context access
   - Ensure proper async execution patterns

4. **Fix Command Error Handling (1 hour)**
   - Add robust error propagation from commands to UI
   - Ensure failed operations update plate status correctly
   - Add user feedback for command failures
   - Test error recovery scenarios

5. **Verify Command State Updates (45 minutes)**
   - Test command execution updates SimpleTUIState correctly
   - Verify status changes propagate to UI components
   - Test is_compiled and is_running state management
   - Ensure proper operation lifecycle tracking

### Findings

#### Current Command Architecture Analysis

**Command Registration (HIGH CONFIDENCE)**
- Commands registered in canonical_layout._register_commands()
- Uses OrchestratorAwareInitializeCommand wrapper for orchestrator access
- EnhancedCompilePlatesCommand used for compilation
- Basic RunPlatesCommand as fallback

**Command Parameter Flow (MEDIUM CONFIDENCE)**
```
Button Click → Command.execute(state, context, **kwargs)
  → Command accesses state.active_orchestrator
  → OR Command gets orchestrators from OrchestratorManager
  → Command executes orchestrator methods
  → Command updates state and emits events
```

**Enhanced Compilation Integration (HIGH CONFIDENCE)**
- EnhancedCompilePlatesCommand exists and is registered
- Uses PipelineCompilationBridge for validation and execution
- Has proper error handling and user feedback
- Integrates with OrchestratorManager

#### Critical Issues Identified

**Issue 1: Orchestrator Access Inconsistency**
- Some commands use state.active_orchestrator
- Others expect orchestrators_to_init/compile/run in kwargs
- _get_selected_orchestrators() methods may fail
- No clear contract for orchestrator parameter passing

**Issue 2: Command Registration Complexity**
- OrchestratorAwareInitializeCommand wrapper may not work
- Enhanced vs basic command registration unclear
- Fallback command registration may override enhanced commands
- Command dependencies may cause import failures

**Issue 3: State Update Coordination**
- Commands update state but coordination with UI unclear
- Multiple state objects (SimpleTUIState vs TUIState) compatibility
- Event emission from commands may not reach UI components
- Operation status tracking inconsistent

**Issue 4: Error Handling Gaps**
- Command errors may not propagate to status bar
- Failed operations may not update plate status
- User feedback for command failures inconsistent
- No recovery mechanisms for partial failures

#### Command Integration Confidence Assessment
- **Command Registration**: 70% - Architecture exists but complexity high
- **Parameter Passing**: 40% - Multiple patterns, unclear which works
- **Orchestrator Access**: 50% - Multiple access methods, consistency unclear
- **Error Handling**: 35% - Basic patterns exist but coordination unclear
- **State Updates**: 45% - Events emitted but propagation unclear

#### Specific Command Analysis

**InitializePlatesCommand (60% confidence)**
- Uses OrchestratorAwareInitializeCommand wrapper
- Accesses orchestrators via _get_selected_orchestrators()
- May fail if state.selected_plate structure incorrect

**EnhancedCompilePlatesCommand (80% confidence)**
- Well-designed with PipelineCompilationBridge
- Proper error handling and validation
- Good integration with OrchestratorManager

**RunPlatesCommand (40% confidence)**
- Basic implementation as fallback
- Unclear orchestrator access pattern
- May not integrate with enhanced compilation results

### Implementation Draft
*Implementation will be added after smell loop approval*
