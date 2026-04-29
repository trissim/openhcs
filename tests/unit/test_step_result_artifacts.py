import pytest
import numpy as np

from openhcs.core.artifacts import (
    CROP_MASK_ARTIFACT_SIDECAR,
    ArtifactInputPlan,
    ArtifactKind,
    ArtifactOutputPlan,
    StepResult,
)
from openhcs.core.runtime_stores import RuntimeValueStore
from openhcs.core.image_shapes import is_image_stack
from openhcs.core.image_stack_layout import ImageStackLayout
from openhcs.core.steps.function_runtime import (
    FunctionExecutionRequest,
    _execute_function_core,
)


class MemoryBackend:
    def __init__(self):
        self._memory_store = {}


class FileManagerStub:
    def __init__(self):
        self.memory = MemoryBackend()
        self.saved = {}
        self.directories = set()

    def _get_backend(self, backend):
        return self.memory

    def ensure_directory(self, path, backend):
        self.directories.add((path, backend))

    def save(self, value, path, backend):
        self.saved[(path, backend)] = value
        self.memory._memory_store[path] = value

    def load(self, path, backend):
        return self.memory._memory_store[path]


class ContextStub:
    def __init__(self):
        self.axis_id = "A01"
        self.filemanager = FileManagerStub()
        self.runtime_value_store = RuntimeValueStore()


def test_crop_mask_sidecar_names_derive_from_core_artifact_role():
    assert CROP_MASK_ARTIFACT_SIDECAR.name_for("CroppedImage") == (
        "CroppedImage__crop_mask"
    )


def test_execute_function_core_saves_named_step_result_artifacts():
    context = ContextStub()

    def analyze(image):
        return StepResult(
            image=image + 1,
            artifacts={"measurements": [{"count": 2}]},
        )

    result = _execute_function_core(
        FunctionExecutionRequest(
            func_callable=analyze,
            main_data_arg=41,
            base_kwargs={},
            context=context,
            artifact_inputs={},
            artifact_outputs={
                "measurements": ArtifactOutputPlan(
                    name="measurements",
                    path="/memory/measurements.pkl",
                )
            },
        )
    )

    assert result == 42
    assert context.filemanager.saved[
        ("/memory/measurements.pkl", "memory")
    ] == [{"count": 2}]
    stored = context.runtime_value_store.find(
        name="measurements",
        axis_id="A01",
    )
    assert len(stored) == 1
    assert stored[0].value.data == [{"count": 2}]


def test_execute_function_core_loads_artifact_input_from_vfs_via_store_record():
    context = ContextStub()

    def produce(image):
        return StepResult(image=image, artifacts={"positions": {"x": 1}})

    _execute_function_core(
        FunctionExecutionRequest(
            func_callable=produce,
            main_data_arg=41,
            base_kwargs={},
            context=context,
            artifact_inputs={},
            artifact_outputs={
                "positions": ArtifactOutputPlan(
                    name="positions",
                    path="/memory/positions.pkl",
                )
            },
        )
    )

    context.filemanager.memory._memory_store["/memory/positions.pkl"] = {
        "x": "from-vfs"
    }

    loaded_inputs = []

    def consume(image, positions):
        loaded_inputs.append(positions)
        return image

    result = _execute_function_core(
        FunctionExecutionRequest(
            func_callable=consume,
            main_data_arg=41,
            base_kwargs={},
            context=context,
            artifact_inputs={
                "positions": ArtifactInputPlan(
                    name="positions",
                    path="/memory/positions.pkl",
                )
            },
            artifact_outputs={},
        )
    )

    assert result == 41
    assert loaded_inputs == [{"x": "from-vfs"}]


def test_execute_function_core_refuses_direct_vfs_artifact_input_fallback():
    context = ContextStub()
    context.filemanager.memory._memory_store["/memory/positions.pkl"] = {"x": 1}

    def consume(image, positions):
        return image

    with pytest.raises(RuntimeError, match="Refusing direct VFS fallback"):
        _execute_function_core(
            FunctionExecutionRequest(
                func_callable=consume,
                main_data_arg=41,
                base_kwargs={},
                context=context,
                artifact_inputs={
                    "positions": ArtifactInputPlan(
                        name="positions",
                        path="/memory/positions.pkl",
                    )
                },
                artifact_outputs={},
            )
        )


def test_execute_function_core_requires_planned_step_result_artifacts():
    context = ContextStub()

    def analyze(image):
        return StepResult(image=image, artifacts={})

    with pytest.raises(ValueError, match="planned artifact 'measurements'"):
        _execute_function_core(
            FunctionExecutionRequest(
                func_callable=analyze,
                main_data_arg=41,
                base_kwargs={},
                context=context,
                artifact_inputs={},
                artifact_outputs={
                    "measurements": ArtifactOutputPlan(
                        name="measurements",
                        path="/memory/measurements.pkl",
                    )
                },
            )
        )


def test_execute_function_core_validates_step_result_artifact_kind():
    context = ContextStub()

    def analyze(image):
        return StepResult(image=image, artifacts={"metadata": ["not", "metadata"]})

    with pytest.raises(TypeError, match="expected metadata mapping"):
        _execute_function_core(
            FunctionExecutionRequest(
                func_callable=analyze,
                main_data_arg=41,
                base_kwargs={},
                context=context,
                artifact_inputs={},
                artifact_outputs={
                    "metadata": ArtifactOutputPlan(
                        name="metadata",
                        path="/memory/metadata.pkl",
                        kind=ArtifactKind.METADATA,
                    )
                },
            )
        )


def test_execute_function_core_validates_tuple_artifact_kind():
    context = ContextStub()

    def analyze(image):
        return image, {"not": "labels"}

    with pytest.raises(TypeError, match="expected object_labels payload"):
        _execute_function_core(
            FunctionExecutionRequest(
                func_callable=analyze,
                main_data_arg=41,
                base_kwargs={},
                context=context,
                artifact_inputs={},
                artifact_outputs={
                    "nuclei": ArtifactOutputPlan(
                        name="nuclei",
                        path="/memory/nuclei.pkl",
                        kind=ArtifactKind.OBJECT_LABELS,
                    )
                },
        )
    )


def test_function_runtime_stacks_and_unstacks_color_image_slices():
    slices = [
        np.zeros((4, 5, 3), dtype=np.float32),
        np.ones((4, 5, 3), dtype=np.float32),
    ]

    stack = ImageStackLayout.for_slices(slices).stack(
        slices=slices,
        memory_type="numpy",
        gpu_id=0,
    )
    unstacked = ImageStackLayout.for_stack(stack).unstack(
        array=stack,
        memory_type="numpy",
        gpu_id=0,
    )

    assert is_image_stack(stack)
    assert stack.shape == (2, 4, 5, 3)
    assert [slice_data.shape for slice_data in unstacked] == [(4, 5, 3), (4, 5, 3)]
    np.testing.assert_array_equal(unstacked[1], slices[1])
