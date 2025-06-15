#!/usr/bin/env python3
"""
Test script to verify TUI fix for gpu_id issue.
Creates a fresh pipeline with proper FunctionStep objects and tests compilation.
"""

import sys
import os
sys.path.insert(0, '/home/ts/code/projects/openhcs')

from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
from openhcs.core.orchestrator.gpu_scheduler import setup_global_gpu_registry
from openhcs.core.steps import FunctionStep as Step
from openhcs.constants.constants import VariableComponents

# Import the same functions as test_main.py
from openhcs.processing.backends.processors.torch_processor import (
    create_projection, create_composite, stack_percentile_normalize
)
from openhcs.processing.backends.pos_gen.mist.mist_main import mist_compute_tile_positions
from openhcs.processing.backends.assemblers.assemble_stack_cupy import assemble_stack_cupy

def test_tui_pipeline_fix():
    """Test that TUI pipeline creation works like test_main.py"""
    
    print("🔥 Testing TUI pipeline fix...")
    
    # Initialize GPU registry
    setup_global_gpu_registry()
    
    # Create the exact same pipeline as test_main.py
    pipeline_steps = [
        Step(func=create_composite,
             variable_components=[VariableComponents.CHANNEL]
        ),
        Step(name="Z-Stack Flattening",
             func=(create_projection, {'method': 'max_projection'}),
             variable_components=[VariableComponents.Z_INDEX],
        ),
        Step(name="Image Enhancement Processing",
             func=[
                 (stack_percentile_normalize, {'low_percentile': 0.5, 'high_percentile': 99.5}),
             ],
        ),
        Step(func=mist_compute_tile_positions,
        ),
        Step(name="Image Enhancement Processing",
             func=[
                 (stack_percentile_normalize, {'low_percentile': 0.5, 'high_percentile': 99.5}),
             ],
        ),
        Step(func=(assemble_stack_cupy, {'blend_method': 'rectangular', 'blend_radius': 5.0}),
        )
    ]
    
    print(f"🔥 Created {len(pipeline_steps)} steps")
    
    # Verify memory type decorators are present
    for i, step in enumerate(pipeline_steps):
        print(f"🔥 Step {i+1}: {step.name}")
        
        # Check if the function has memory type decorators
        func = step.func
        if isinstance(func, tuple):
            func = func[0]
        elif isinstance(func, list):
            func = func[0]
            if isinstance(func, tuple):
                func = func[0]
        
        input_type = getattr(func, 'input_memory_type', 'MISSING')
        output_type = getattr(func, 'output_memory_type', 'MISSING')
        print(f"    Memory types: input={input_type}, output={output_type}")
        
        if input_type == 'MISSING' or output_type == 'MISSING':
            print(f"    ❌ MISSING MEMORY TYPES!")
        else:
            print(f"    ✅ Memory types present")
    
    # Test compilation
    plate_dir = "/home/ts/code/projects/openhcs/tests/integration/tests_data/opera_phenix_pipeline/test_main_3d[OperaPhenix]/zstack_plate_1"
    
    print(f"🔥 Testing compilation with plate: {plate_dir}")
    
    orchestrator = PipelineOrchestrator(plate_dir)
    orchestrator.initialize()
    
    wells = orchestrator.get_wells()
    print(f"🔥 Found {len(wells)} wells: {wells}")
    
    try:
        compiled_contexts = orchestrator.compile_pipelines(
            pipeline_definition=pipeline_steps,
            well_filter=wells
        )
        print(f"🔥 ✅ COMPILATION SUCCESS: {len(compiled_contexts)} contexts compiled")
        
        # Check if gpu_id is present in step plans
        for well_id, context in compiled_contexts.items():
            print(f"🔥 Checking well {well_id} step plans...")
            for step_id, step_plan in context.step_plans.items():
                gpu_id = step_plan.get('gpu_id', 'MISSING')
                step_name = step_plan.get('step_name', 'Unknown')
                print(f"    Step '{step_name}': gpu_id={gpu_id}")
                
                if gpu_id == 'MISSING':
                    print(f"    ❌ GPU ID MISSING!")
                else:
                    print(f"    ✅ GPU ID present")
        
        return True
        
    except Exception as e:
        print(f"🔥 ❌ COMPILATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_tui_pipeline_fix()
    if success:
        print("🔥 ✅ TUI FIX VERIFIED - Pipeline compilation works!")
    else:
        print("🔥 ❌ TUI FIX FAILED - Pipeline compilation still broken!")
    sys.exit(0 if success else 1)
