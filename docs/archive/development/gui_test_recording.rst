.. _gui_test_recording:

======================
GUI Test Recording
======================

OpenHCS includes a **GUI test recording system** that captures manual interactions with the PyQt6 GUI and generates pytest-qt test code that can replay those exact interactions.

This system solves two critical testing challenges:

1. **GUI Testing in CI**: Generated tests run in headless CI environments
2. **GUI ↔ CLI Equivalence**: Validates that both interfaces produce identical results

Overview
========

The recording system works by:

1. Installing a Qt event filter that captures user interactions
2. Recording clicks, text input, selections, and timing
3. Generating pytest-qt test code that replays the exact sequence
4. Validating that GUI and CLI produce identical outputs

Quick Start
===========

Record a Workflow
-----------------

Start the GUI in recording mode::

    python -m openhcs.pyqt_gui.launch --record-test my_workflow

Interact with the GUI normally:

- Add plates
- Configure pipeline
- Run processing
- etc.

When you close the application, a test file is automatically generated at::

    tests/pyqt_gui/recorded/test_my_workflow.py

Run the Recorded Test
---------------------

Run the generated test::

    # Run specific test
    pytest tests/pyqt_gui/recorded/test_my_workflow.py -v

    # Run all recorded tests
    pytest tests/pyqt_gui/recorded/ -v

Architecture
============

Event Recorder
--------------

The :class:`~openhcs.pyqt_gui.testing.event_recorder.EventRecorder` class captures GUI events:

.. code-block:: python

    from openhcs.pyqt_gui.testing import EventRecorder, install_recorder
    
    # Install on application
    recorder = install_recorder(app, "my_test")
    
    # Events are automatically captured
    # Test code is generated on app close

**Captured Events**:

- Button clicks (``QPushButton``)
- Text input (``QLineEdit``)
- Dropdown selections (``QComboBox``)
- Checkbox toggles (``QCheckBox``)

**Timing Strategy**:

The recorder does **not** capture exact timing delays. Instead, generated tests use:

- ``_wait_for_gui(TIMING.ACTION_DELAY)`` after each action
- ``TimingConfig.from_environment()`` for configurable delays
- Environment variables to adjust timing for slower machines

This ensures tests work reliably across different machine speeds without flakiness

**Generated Code**:

.. code-block:: python

    def test_my_workflow(qtbot):
        """Auto-generated from GUI recording."""
        main_window = OpenHCSMainWindow()
        qtbot.addWidget(main_window)
        main_window.show()
        qtbot.wait(1500)
        
        # Click button
        qtbot.mouseClick(button, Qt.MouseButton.LeftButton)
        
        # Type text
        qtbot.keyClicks(text_field, "value")
        
        # Select dropdown
        combo.setCurrentText("option")

Test Validator
--------------

The :class:`~openhcs.pyqt_gui.testing.test_validator.TestValidator` class validates GUI ↔ CLI equivalence:

.. code-block:: python

    from openhcs.pyqt_gui.testing import TestValidator
    
    validator = TestValidator("my_workflow", tmp_path)
    
    # Capture GUI output
    gui_snapshot = validator.capture_gui_snapshot(plate_dir, config)
    
    # Run equivalent CLI command
    cli_snapshot = validator.run_cli_equivalent(cli_command)
    
    # Validate equivalence
    assert validator.validate_equivalence()

**Validation Process**:

1. Hash all output files from GUI workflow
2. Run equivalent CLI command
3. Hash all output files from CLI workflow
4. Compare file hashes and metadata
5. Assert identical results

Timing Configuration
====================

Handling Slower Machines
-------------------------

Generated tests use **environment-configurable timing** to work reliably on machines of different speeds.

**Default Timing** (from ``TimingConfig``)::

    ACTION_DELAY = 1.5 seconds
    WINDOW_DELAY = 1.5 seconds
    SAVE_DELAY = 1.5 seconds

**Adjust for Slower Machines**::

    # Increase delays for slower CI runners or VMs
    export OPENHCS_TEST_ACTION_DELAY=3.0
    export OPENHCS_TEST_WINDOW_DELAY=3.0
    export OPENHCS_TEST_SAVE_DELAY=3.0

    pytest tests/pyqt_gui/recorded/ -v

**Adjust for Faster Machines**::

    # Decrease delays for faster local testing
    export OPENHCS_TEST_ACTION_DELAY=0.5
    export OPENHCS_TEST_WINDOW_DELAY=0.5
    export OPENHCS_TEST_SAVE_DELAY=0.5

    pytest tests/pyqt_gui/recorded/ -v

**Why This Works**:

- Tests don't use fixed ``qtbot.wait(1000)`` delays
- Instead use ``_wait_for_gui(TIMING.ACTION_DELAY)``
- ``TimingConfig.from_environment()`` reads env vars
- Same test code works on all machines by adjusting env vars

CI Integration
==============

Headless Display Setup
----------------------

OpenHCS uses the same approach as Napari for headless GUI testing in CI.

**GitHub Actions Workflow**:

.. code-block:: yaml

    gui-tests:
      runs-on: ${{ matrix.os }}
      strategy:
        matrix:
          os: [ubuntu-latest, windows-latest, macos-latest]
      
      steps:
        - uses: actions/checkout@v4
        
        - uses: actions/setup-python@v5
          with:
            python-version: "3.12"
        
        - name: Setup headless display
          uses: pyvista/setup-headless-display-action@v4.2
          with:
            qt: true
            wm: herbstluftwm
        
        - name: Install dependencies
          run: pip install -e ".[dev,gui]"
        
        - name: Run recorded GUI tests
          env:
            PYVISTA_OFF_SCREEN: true
            QT_QPA_PLATFORM: offscreen
          run: pytest tests/pyqt_gui/recorded/ -v

**Platform Support**:

- **Linux**: Uses Xvfb + herbstluftwm window manager
- **macOS**: Native GUI support (no virtual display needed)
- **Windows**: Native GUI support (no virtual display needed)

**OpenGL Support**:

For tests requiring Napari visualization::

    - name: Setup headless display with OpenGL
      uses: pyvista/setup-headless-display-action@v4.2
      with:
        qt: true
        wm: herbstluftwm
        pyvista: true  # Enable software OpenGL rendering
    
    - name: Run integration tests with Napari
      env:
        PYVISTA_OFF_SCREEN: true
        LIBGL_ALWAYS_SOFTWARE: 1
      run: pytest tests/integration/ --it-visualizers napari -v

Example Workflow
================

Recording a Plate Addition
---------------------------

**Step 1: Start Recording**::

    python -m openhcs.pyqt_gui.launch --record-test add_plate_workflow

**Step 2: Interact with GUI**:

1. Click "Add Plate" button
2. Select directory ``/path/to/plate``
3. Click "OK"
4. Click "Init Plate"
5. Close application

**Step 3: Generated Test**:

.. code-block:: python

    # tests/pyqt_gui/recorded/test_add_plate_workflow.py
    
    import pytest
    import os
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QWidget, QPushButton
    
    if os.getenv('OPENHCS_CPU_ONLY', 'false').lower() == 'true':
        pytest.skip('PyQt6 GUI tests skipped in CPU-only mode', 
                    allow_module_level=True)
    
    from openhcs.pyqt_gui.main import OpenHCSMainWindow
    from tests.pyqt_gui.integration.test_end_to_end_workflow_foundation import (
        WidgetFinder
    )
    
    def test_add_plate_workflow(qtbot):
        """
        Auto-generated test from GUI recording: add_plate_workflow
        Recorded on: 2025-10-31 14:30:00
        Total events: 5
        """
        # Create main window
        main_window = OpenHCSMainWindow()
        qtbot.addWidget(main_window)
        main_window.show()
        qtbot.wait(1500)
        
        # Replay recorded interactions
        add_button = WidgetFinder.find_button_by_text(
            main_window, ["add plate"]
        )
        qtbot.mouseClick(add_button, Qt.MouseButton.LeftButton)
        qtbot.wait(500)
        
        # ... more interactions

**Step 4: Run in CI**::

    pytest tests/pyqt_gui/recorded/test_add_plate_workflow.py -v

GUI ↔ CLI Equivalence Testing
==============================

Concept
-------

The same workflow should produce **identical results** whether run through:

- **GUI**: User clicks buttons, fills forms
- **CLI**: User runs command with arguments

Implementation
--------------

.. code-block:: python

    def test_workflow_gui_cli_equivalence(qtbot, tmp_path):
        """Validate GUI and CLI produce identical results."""
        
        from openhcs.pyqt_gui.testing import TestValidator
        
        validator = TestValidator("my_workflow", tmp_path)
        
        # 1. Run GUI workflow (from recorded test)
        # ... GUI interactions ...
        gui_snapshot = validator.capture_gui_snapshot(plate_dir, config)
        
        # 2. Generate equivalent CLI command
        cli_command = generate_cli_command_from_config(config, plate_dir)
        
        # 3. Run CLI
        cli_snapshot = validator.run_cli_equivalent(cli_command)
        
        # 4. Validate equivalence
        assert validator.validate_equivalence()
        # ✅ Compares file hashes
        # ✅ Compares metadata
        # ✅ Ensures identical outputs

Benefits
========

✅ **No Manual Test Writing**
    Record once, replay forever. No need to manually write qtbot interactions.

✅ **GUI + CLI Tested Simultaneously**
    Same workflow runs through both interfaces. Validates they produce identical results.

✅ **CI Integration**
    Generated tests run in headless CI using battle-tested ``pyvista/setup-headless-display-action``.

✅ **Regression Detection**
    Replay recorded workflows after code changes to catch UI regressions automatically.

✅ **Real User Workflows**
    Captures actual usage patterns, not artificial test scenarios.

API Reference
=============

.. automodule:: openhcs.pyqt_gui.testing.event_recorder
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: openhcs.pyqt_gui.testing.test_validator
   :members:
   :undoc-members:
   :show-inheritance:

See Also
========

- :doc:`../guides/testing_guide` - General testing guide
- :doc:`omero_testing` - OMERO-specific testing
- :doc:`../user_guide/cpu_only_mode` - CPU-only testing mode
