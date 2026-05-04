from types import SimpleNamespace

from openhcs.constants.constants import Backend
from openhcs.constants.input_source import InputSource
from openhcs.core.artifacts import ArtifactKind, ArtifactOutputPlan
from openhcs.core.compiled_step_plan import CompiledStepPlan
from openhcs.core.config import MaterializationBackend, VFSConfig
from openhcs.core.pipeline.materialization_flag_planner import MaterializationFlagPlanner


def _pipeline_config() -> SimpleNamespace:
    return SimpleNamespace(
        vfs_config=VFSConfig(
            read_backend=Backend.DISK,
            materialization_backend=MaterializationBackend.DISK,
        )
    )


def _step(input_source: InputSource = InputSource.PREVIOUS_STEP) -> SimpleNamespace:
    return SimpleNamespace(processing_config=SimpleNamespace(input_source=input_source))


def test_final_measurement_only_step_keeps_images_in_memory() -> None:
    context = SimpleNamespace(
        step_plans=[
            CompiledStepPlan(0, "image", "FunctionStep", "A01"),
            CompiledStepPlan(
                1,
                "measure",
                "FunctionStep",
                "A01",
                artifact_outputs={
                    "measurements": ArtifactOutputPlan(
                        "measurements",
                        "/memory/measurements.pkl",
                        ArtifactKind.MEASUREMENTS,
                    )
                },
            ),
        ]
    )

    MaterializationFlagPlanner.prepare_pipeline_flags(
        context,
        [_step(), _step()],
        plate_path=None,
        pipeline_config=_pipeline_config(),
    )

    assert context.step_plans[1].write_backend == Backend.MEMORY.value


def test_final_image_artifact_step_materializes_images() -> None:
    context = SimpleNamespace(
        step_plans=[
            CompiledStepPlan(
                0,
                "segment",
                "FunctionStep",
                "A01",
                artifact_outputs={
                    "labels": ArtifactOutputPlan(
                        "labels",
                        "/memory/labels.pkl",
                        ArtifactKind.OBJECT_LABELS,
                    )
                },
            )
        ]
    )

    MaterializationFlagPlanner.prepare_pipeline_flags(
        context,
        [_step()],
        plate_path=None,
        pipeline_config=_pipeline_config(),
    )

    assert context.step_plans[0].write_backend == Backend.DISK.value


def test_final_uncontracted_step_preserves_legacy_image_materialization() -> None:
    context = SimpleNamespace(
        step_plans=[CompiledStepPlan(0, "process", "FunctionStep", "A01")]
    )

    MaterializationFlagPlanner.prepare_pipeline_flags(
        context,
        [_step()],
        plate_path=None,
        pipeline_config=_pipeline_config(),
    )

    assert context.step_plans[0].write_backend == Backend.DISK.value
