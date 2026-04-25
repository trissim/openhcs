import pytest

from openhcs.core.artifacts import ArtifactKind, ArtifactOutputPlan, StepResult
from openhcs.core.runtime_stores import RuntimeValueStore
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


class ContextStub:
    def __init__(self):
        self.axis_id = "A01"
        self.filemanager = FileManagerStub()
        self.runtime_value_store = RuntimeValueStore()


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
