Integration Testing Framework Overview
====================================

Composable workflow foundation for PyQt6 GUI integration testing with mathematical simplification, real widget interactions, and comprehensive validation.

Overview
--------

The integration testing framework provides a sophisticated yet simple approach to testing complex GUI workflows in OpenHCS. It replaces scattered test files with a unified, modular system that can handle real PyQt6 widget interactions.

**The Challenge:** GUI integration testing is inherently complex, involving multiple components, timing dependencies, and state management. Traditional approaches often use mocks or simplified scenarios that don't catch real-world issues.

**The Solution:** A composable workflow system with atomic operations, background monitoring, and parameterized testing that works with real PyQt6 widgets and synthetic data generation.

**Real-World Impact:** Enabled detection of critical lazy dataclass bugs, provided foundation for comprehensive GUI testing, and established patterns for future test development.

WorkflowStep Atomic Operations
------------------------------

The foundation of the testing framework is the WorkflowStep - atomic operations with clear input/output contracts.

WorkflowStep Design
~~~~~~~~~~~~~~~~~~~

Each WorkflowStep represents a single, testable operation in a GUI workflow:

.. code-block:: python

    @dataclass
    class WorkflowStep:
        """Atomic workflow operation with clear input/output contract."""
        name: str
        operation: Callable[[WorkflowContext], WorkflowContext]
        description: str = ""
        timing_delay: Optional[float] = None
    
        def execute(self, context: WorkflowContext) -> WorkflowContext:
            """Execute step with timing and logging."""
            print(f"  {self.name}...")
            result = self.operation(context)
            if self.timing_delay:
                _wait_for_gui(self.timing_delay)
            print(f"  ✅ {self.name} completed")
            return result

**Key Principles:**

1. **Atomic Operations**: Each step does one thing and does it well
2. **Immutable Context**: Steps receive context and return new context
3. **Clear Contracts**: Input and output types are well-defined
4. **Timing Control**: Optional delays for GUI operations
5. **Logging Integration**: Automatic progress tracking

Atomic Operation Examples
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Application lifecycle operations
    def _launch_application(context: WorkflowContext) -> WorkflowContext:
        """Launch OpenHCS main application."""
        app = QApplication.instance() or QApplication([])
        main_window = OpenHCSMainWindow()
        main_window.show()
        QApplication.processEvents()
        
        return context.with_updates(main_window=main_window)
    
    # Widget access operations
    def _access_plate_manager(context: WorkflowContext) -> WorkflowContext:
        """Access plate manager widget from main window."""
        plate_manager = context.main_window.findChild(PlateManagerWidget)
        if not plate_manager:
            raise AssertionError("Plate manager widget not found")
        
        return context.with_updates(plate_manager_widget=plate_manager)
    
    # Configuration operations
    def _apply_orchestrator_config(context: WorkflowContext) -> WorkflowContext:
        """Apply parameterized orchestrator configuration."""
        orchestrator = context.plate_manager_widget.orchestrators[str(context.synthetic_plate_dir)]
        
        config_params = context.test_scenario.orchestrator_config
        orchestrator_config = PipelineConfig(
            materialization_defaults=LazyStepMaterializationConfig(
                output_dir_suffix=config_params.get("output_dir_suffix"),
                sub_dir=config_params.get("sub_dir"),
                well_filter=config_params.get("well_filter")
            )
        )
        
        orchestrator.set_config(orchestrator_config)
        return context.with_updates(orchestrator=orchestrator)

Immutable Context Pattern
~~~~~~~~~~~~~~~~~~~~~~~~~

The WorkflowContext uses an immutable pattern for safe state management:

.. code-block:: python

    @dataclass
    class WorkflowContext:
        """Immutable context passed between workflow steps."""
        main_window: Optional[OpenHCSMainWindow] = None
        plate_manager_widget: Optional[PlateManagerWidget] = None
        config_window: Optional[QDialog] = None
        synthetic_plate_dir: Optional[Path] = None
        orchestrator: Optional[PipelineOrchestrator] = None
        validation_results: Dict[str, Any] = field(default_factory=dict)
        test_scenario: Optional[TestScenario] = None
    
        def with_updates(self, **kwargs) -> 'WorkflowContext':
            """Create new context with updates (immutable pattern)."""
            from dataclasses import replace
            return replace(self, **kwargs)

**Benefits of Immutable Context:**

1. **Thread Safety**: No shared mutable state between operations
2. **Debugging**: Clear state transitions between steps
3. **Composability**: Steps can be reordered without side effects
4. **Testability**: Each step's input/output is explicit

WorkflowBuilder Step Sequencing
-------------------------------

The WorkflowBuilder provides composable step sequencing with assertion injection.

Fluent Interface Design
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    class WorkflowBuilder:
        """Composable workflow builder for GUI test scenarios."""
    
        def __init__(self):
            self.steps: List[WorkflowStep] = []
            self.assertions: List[Callable[[WorkflowContext], None]] = []
    
        def add_step(self, step: WorkflowStep) -> 'WorkflowBuilder':
            """Add workflow step (fluent interface)."""
            self.steps.append(step)
            return self
    
        def add_assertion(self, assertion: Callable[[WorkflowContext], None]) -> 'WorkflowBuilder':
            """Add assertion to be checked after workflow completion."""
            self.assertions.append(assertion)
            return self
    
        def execute(self, initial_context: WorkflowContext) -> WorkflowContext:
            """Execute workflow steps sequentially."""
            context = initial_context
            for step in self.steps:
                context = step.execute(context)
    
            # Run all assertions
            for assertion in self.assertions:
                assertion(context)
    
            return context

Composable Workflow Construction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Build complex workflows using fluent interface
    workflow = (WorkflowBuilder()
        .add_step(WorkflowStep(
            name="Launch OpenHCS Application",
            operation=_launch_application,
            timing_delay=TIMING.WINDOW_DELAY
        ))
        .add_step(WorkflowStep(
            name="Access Plate Manager",
            operation=_access_plate_manager
        ))
        .add_step(WorkflowStep(
            name="Add and Select Plate",
            operation=_add_and_select_plate,
            timing_delay=TIMING.ACTION_DELAY
        ))
        .add_step(WorkflowStep(
            name="Initialize Plate",
            operation=_initialize_plate,
            timing_delay=TIMING.SAVE_DELAY
        ))
        .add_step(WorkflowStep(
            name="Apply Parameterized Orchestrator Configuration",
            operation=_apply_orchestrator_config,
            timing_delay=TIMING.ACTION_DELAY
        ))
        .add_assertion(lambda ctx: assert ctx.orchestrator is not None)
        .add_assertion(lambda ctx: assert ctx.synthetic_plate_dir.exists())
    )

Assertion Integration
~~~~~~~~~~~~~~~~~~~~~

Assertions can be added at any point and are executed after workflow completion:

.. code-block:: python

    def validate_placeholder_behavior(context: WorkflowContext):
        """Validate that placeholder text reflects inheritance hierarchy."""
        config_window = context.config_window
        form_manager = config_window.findChild(ParameterFormManager)
        
        # Check specific field placeholders
        output_dir_widget = form_manager.findChild(QLineEdit, "config_output_dir_suffix")
        placeholder_text = output_dir_widget.placeholderText()
        
        expected_value = context.test_scenario.expected_values["output_dir_suffix"]
        assert expected_value in placeholder_text, f"Expected {expected_value} in placeholder: {placeholder_text}"
    
    # Add assertion to workflow
    workflow.add_assertion(validate_placeholder_behavior)

TestOrchestrator Patterns
-------------------------

The TestOrchestrator provides central coordination for complex test scenarios with parameterized testing.

Parameterized Test Scenarios
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    @dataclass
    class TestScenario:
        """Parameterized test scenario configuration."""
        name: str
        orchestrator_config: Dict[str, Any]
        expected_values: Dict[str, Any]
        field_to_test: 'FieldModificationSpec'
    
    @dataclass
    class FieldModificationSpec:
        """Specification for field modification testing."""
        field_name: str
        modification_value: Any
    
    # Define test scenarios
    DEFAULT_SCENARIO = TestScenario(
        name="default_hierarchy",
        orchestrator_config={
            "output_dir_suffix": "_outputs",
            "sub_dir": "images",
            "well_filter": 5
        },
        expected_values={
            "output_dir_suffix": "_outputs",
            "sub_dir": "images",
            "well_filter": 5
        },
        field_to_test=FieldModificationSpec(
            field_name="well_filter",
            modification_value=4
        )
    )

Test Orchestration
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    @pytest.mark.parametrize("test_scenario", [DEFAULT_SCENARIO, ALTERNATIVE_SCENARIO])
    def test_lazy_config_inheritance_comprehensive(test_scenario):
        """Comprehensive test using parameterized scenarios."""
        
        # Create synthetic data
        synthetic_plate_dir = create_synthetic_plate()
        
        # Build workflow with parameterized context
        workflow = build_comprehensive_workflow(test_scenario)
        
        # Execute with scenario-specific context
        initial_context = WorkflowContext(
            synthetic_plate_dir=synthetic_plate_dir,
            test_scenario=test_scenario
        )
        
        final_context = workflow.execute(initial_context)
        
        # Scenario-specific validation
        validate_scenario_results(final_context, test_scenario)

Real PyQt6 Widget Testing
-------------------------

The framework tests actual PyQt6 widgets rather than mocks, providing authentic integration testing.

Widget Interaction Patterns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def _modify_field_value(context: WorkflowContext) -> WorkflowContext:
        """Modify field value using real widget interactions."""
        config_window = context.config_window
        form_manager = config_window.findChild(ParameterFormManager)

        # Find the actual QLineEdit widget
        field_name = context.test_scenario.field_to_test.field_name
        widget_id = f"config_{field_name}"
        target_widget = form_manager.findChild(QLineEdit, widget_id)

        if not target_widget:
            raise AssertionError(f"Widget {widget_id} not found in form")

        # Perform real widget interaction
        target_widget.clear()
        target_widget.setText(str(context.test_scenario.field_to_test.modification_value))

        # Trigger change events
        target_widget.editingFinished.emit()
        QApplication.processEvents()

        return context

Widget Discovery and Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def validate_widget_state(context: WorkflowContext):
        """Validate actual widget state and properties."""
        form_manager = context.config_window.findChild(ParameterFormManager)

        # Test real widget properties
        for field_name, expected_value in context.test_scenario.expected_values.items():
            widget_id = f"config_{field_name}"
            widget = form_manager.findChild(QLineEdit, widget_id)

            # Validate placeholder text (inheritance behavior)
            placeholder = widget.placeholderText()
            assert expected_value in placeholder, f"Expected {expected_value} in {placeholder}"

            # Validate widget state
            assert widget.isEnabled(), f"Widget {widget_id} should be enabled"
            assert widget.isVisible(), f"Widget {widget_id} should be visible"

Real Event Processing
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def _wait_for_gui(delay_seconds: float = 0.1):
        """Wait for GUI events to process with real event loop."""
        import time

        start_time = time.time()
        while time.time() - start_time < delay_seconds:
            QApplication.processEvents()
            time.sleep(0.01)  # Small sleep to prevent CPU spinning

    def _trigger_widget_events(widget, event_type="change"):
        """Trigger real widget events."""
        if event_type == "change" and hasattr(widget, 'editingFinished'):
            widget.editingFinished.emit()
        elif event_type == "click" and hasattr(widget, 'clicked'):
            widget.clicked.emit()

        # Process events immediately
        QApplication.processEvents()

Background Monitoring
---------------------

The framework includes sophisticated background monitoring for error detection and freeze prevention.

Error Dialog Detection
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    class ErrorDialogMonitor(QObject):
        """Background monitor that continuously watches for error dialogs."""

        error_detected = pyqtSignal(str)

        def __init__(self):
            super().__init__()
            self.timer = QTimer()
            self.timer.timeout.connect(self.check_for_error_dialogs)
            self.monitoring = False
            self.detected_error = None

        def start_monitoring(self, check_interval_ms: int = 100):
            """Start continuous monitoring for error dialogs."""
            print("  Starting background error dialog monitor...")
            self.monitoring = True
            self.detected_error = None
            self.timer.start(check_interval_ms)

        def check_for_error_dialogs(self):
            """Check for error dialogs and handle them immediately."""
            if not self.monitoring:
                return

            try:
                error_dialogs = self._find_error_dialogs_immediate()
                if error_dialogs and not self.detected_error:
                    error_details = self._close_error_dialogs_immediate(error_dialogs)
                    error_message = (
                        f"LAZY CONFIG BUG DETECTED: Error dialog appeared! "
                        f"Error dialogs: {error_details}"
                    )
                    self.detected_error = error_message
                    self.error_detected.emit(error_message)
                    self.stop_monitoring()
            except Exception as e:
                print(f"  Error in background monitor: {e}")

Timeout and Error Handling
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def with_timeout_and_error_handling(timeout_seconds: int = 10, operation_name: str = "operation"):
        """Decorator for timeout handling with background error dialog monitoring."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                import time
                start_time = time.time()

                # Start background error monitoring
                monitor = get_error_monitor()
                monitor.start_monitoring(check_interval_ms=50)

                try:
                    print(f"  {operation_name.title()}...")
                    result = func(*args, **kwargs)

                    # Check if error was detected during operation
                    if monitor.detected_error:
                        raise AssertionError(monitor.detected_error)

                    elapsed = time.time() - start_time
                    print(f"  {operation_name.title()} completed successfully in {elapsed:.2f}s")
                    return result
                finally:
                    monitor.stop_monitoring()

            return wrapper
        return decorator

Synthetic Data Generation
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def create_synthetic_plate() -> Path:
        """Generate synthetic microscopy data for testing."""
        generator = SyntheticMicroscopyGenerator()

        # Create realistic plate structure
        plate_config = {
            'wells': ['A01', 'A02', 'B01', 'B02'],
            'sites': [1, 2, 3, 4],
            'channels': ['DAPI', 'GFP', 'RFP'],
            'image_size': (512, 512),
            'bit_depth': 16
        }

        synthetic_plate_dir = generator.create_plate(plate_config)

        # Validate synthetic data structure
        assert synthetic_plate_dir.exists()
        assert len(list(synthetic_plate_dir.glob('**/*.tiff'))) > 0

        return synthetic_plate_dir

Real-World Usage Examples
-------------------------

These examples show how the integration testing framework handles complex scenarios.

Example 1: Lazy Configuration Bug Detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def test_lazy_config_inheritance_bug():
        """Test that detects the lazy configuration inheritance bug."""

        # This test detected a critical bug where modifying path_planning.output_dir_suffix
        # caused materialization_defaults.output_dir_suffix to show static defaults
        # instead of inheriting the new path_planning value

        workflow = (WorkflowBuilder()
            .add_step(WorkflowStep("Launch Application", _launch_application))
            .add_step(WorkflowStep("Setup Plate", _setup_test_plate))
            .add_step(WorkflowStep("Open Config Window", _open_config_window))
            .add_step(WorkflowStep("Modify Path Planning Field", _modify_path_planning_field))
            .add_step(WorkflowStep("Validate Inheritance", _validate_sibling_inheritance))
        )

        # The test would fail before the fix, detecting the inheritance bug
        context = workflow.execute(WorkflowContext())

Example 2: Multi-Step Configuration Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def test_complete_configuration_workflow():
        """Test complete configuration workflow with multiple modifications."""

        workflow = (WorkflowBuilder()
            .add_step(WorkflowStep("Initialize Application", _launch_application))
            .add_step(WorkflowStep("Create Test Plate", _create_synthetic_plate))
            .add_step(WorkflowStep("Configure Orchestrator", _apply_orchestrator_config))
            .add_step(WorkflowStep("Open Configuration UI", _open_config_window))
            .add_step(WorkflowStep("Modify Multiple Fields", _modify_multiple_fields))
            .add_step(WorkflowStep("Save Configuration", _save_configuration))
            .add_step(WorkflowStep("Reload and Validate", _reload_and_validate))
            .add_assertion(validate_configuration_persistence)
            .add_assertion(validate_inheritance_chains)
        )

        context = workflow.execute(WorkflowContext(test_scenario=COMPLEX_SCENARIO))

Example 3: Error Recovery Testing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def test_error_recovery_workflow():
        """Test error recovery and graceful failure handling."""

        workflow = (WorkflowBuilder()
            .add_step(WorkflowStep("Setup Normal State", _setup_normal_state))
            .add_step(WorkflowStep("Introduce Error Condition", _introduce_error))
            .add_step(WorkflowStep("Attempt Recovery", _attempt_recovery))
            .add_step(WorkflowStep("Validate Error Handling", _validate_error_handling))
        )

        # Background monitor will detect any error dialogs
        with pytest.raises(AssertionError, match="LAZY CONFIG BUG DETECTED"):
            workflow.execute(WorkflowContext())

Benefits
--------

- **Composable Testing**: Atomic operations can be combined for complex scenarios
- **Real Widget Interactions**: Tests actual PyQt6 widgets, not mocks
- **Parameterized Scenarios**: Single test framework handles multiple configurations
- **Background Monitoring**: Detects errors and freezes during test execution
- **Synthetic Data Generation**: Reproducible test data for consistent results
- **Mathematical Simplification**: Clear contracts and immutable state management
- **Comprehensive Validation**: Flexible assertion framework for any workflow state
- **Timing Control**: Configurable delays for GUI operations and state transitions
- **Error Detection**: Automatic detection of error dialogs and application freezes
- **Bug Discovery**: Capable of detecting subtle inheritance and configuration bugs
- **Workflow Reusability**: Steps can be reused across different test scenarios

See Also
--------

- :doc:`ui-patterns` - UI patterns and functional dispatch tested by this framework
- :doc:`ui-utilities-migration` - Functional utilities validated through integration testing
- :doc:`../architecture/lazy-class-system` - Lazy dataclass patterns tested for inheritance bugs
- :doc:`../architecture/configuration-resolution` - Context management patterns tested in workflows
- :doc:`../architecture/step-editor-generalization` - Step editor functionality validated through testing
