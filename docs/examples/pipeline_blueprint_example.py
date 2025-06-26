"""
Example of using the PipelineBlueprint class.

This example demonstrates how to use the PipelineBlueprint class to create a
memory-aware pipeline blueprint and convert it to executable steps.
"""

from pathlib import Path
from typing import List, Dict, Any

from ezstitcher.core.pipeline.pipeline_blueprint import PipelineBlueprint, StepDeclaration
from ezstitcher.core.steps import FunctionStep, ZFlatStep, NormStep, CompositeStep
from ezstitcher.io.virtual_path.factory import VirtualPathFactory
from ezstitcher.constants.memory import MemoryType


def create_pipeline_blueprint() -> PipelineBlueprint:
    """
    Create a pipeline blueprint with explicit memory typing.

    Returns:
        A PipelineBlueprint object
    """
    # Create input and output paths
    input_path = VirtualPathFactory.from_path("input")
    output_path = VirtualPathFactory.from_path("output")

    # Create step declarations
    step_declarations = [
        StepDeclaration(
            step_cls=ZFlatStep,
            input_path=input_path,
            output_path=output_path.joinpath("zflat"),
            input_memory_type=MemoryType.NUMPY.value,
            output_memory_type=MemoryType.NUMPY.value,
            metadata={
                "name": "Z-Stack Flattening",
                "variable_components": ["z_index"]
            }
        ),
        StepDeclaration(
            step_cls=NormStep,
            input_path=output_path.joinpath("zflat"),
            output_path=output_path.joinpath("norm"),
            input_memory_type=MEMORY_TYPE_NUMPY,
            output_memory_type=MEMORY_TYPE_CUPY,
            metadata={
                "name": "Normalization",
                "variable_components": []
            }
        ),
        StepDeclaration(
            step_cls=CompositeStep,
            input_path=output_path.joinpath("norm"),
            output_path=output_path.joinpath("composite"),
            input_memory_type=MEMORY_TYPE_CUPY,
            output_memory_type=MEMORY_TYPE_NUMPY,
            metadata={
                "name": "Channel Compositing",
                "variable_components": ["channel"]
            }
        )
    ]

    # Create a pipeline blueprint
    return PipelineBlueprint(steps=step_declarations, name="Example Pipeline")


def main():
    """Main function."""
    # Create a pipeline blueprint
    blueprint = create_pipeline_blueprint()

    # Print the blueprint
    print(f"Pipeline Blueprint: {blueprint.name}")
    print(f"Number of steps: {len(blueprint.steps)}")

    for i, step in enumerate(blueprint.steps):
        print(f"\nStep {i+1}: {step.metadata.get('name', step.step_cls.__name__)}")
        print(f"  Class: {step.step_cls.__name__}")
        print(f"  Input Path: {step.input_path}")
        print(f"  Output Path: {step.output_path}")
        print(f"  Input Memory Type: {step.input_memory_type}")
        print(f"  Output Memory Type: {step.output_memory_type}")
        print(f"  Variable Components: {step.metadata.get('variable_components', [])}")

    # Convert to executable steps
    executable_steps = blueprint.as_executable_steps()

    # Create a new blueprint from the executable steps with automatic conversion steps
    new_blueprint = PipelineBlueprint.from_steps(executable_steps)

    # Get the final steps
    final_steps = new_blueprint.as_executable_steps()

    # Print the final pipeline
    print("\nFinal Pipeline:")
    print(f"Number of steps: {len(final_steps)}")

    for i, step in enumerate(final_steps):
        print(f"\nStep {i+1}: {step.name}")
        print(f"  Class: {step.__class__.__name__}")
        print(f"  Input Memory Type: {step.input_memory_type}")
        print(f"  Output Memory Type: {step.output_memory_type}")


if __name__ == "__main__":
    main()
