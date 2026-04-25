import pytest

from openhcs.core.artifacts import ArtifactOutputPlan, StepResult
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
        self.filemanager = FileManagerStub()


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
