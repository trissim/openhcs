"""Source-owned scope regression for public CombineObjects compilation."""

from __future__ import annotations

from pathlib import Path
from queue import SimpleQueue
from types import SimpleNamespace

import numpy as np
import tifffile

from objectstate.lazy_factory import ensure_global_config_context
from openhcs.constants.constants import (
    AllComponents,
    GroupBy,
    VariableComponents,
)
from openhcs.constants.input_source import InputSource
from openhcs.core.artifacts import (
    ArtifactSpec,
    ArtifactSpecCollection,
    ObjectLabelsArtifactType,
)
from openhcs.core.callable_contract import CallableContract
from openhcs.core.config import GlobalPipelineConfig, LazyProcessingConfig
from openhcs.core.invocation_artifacts import ArtifactDeclarationStepContext
from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
from openhcs.core.pipeline.function_contracts import artifact_inputs, artifact_outputs
from openhcs.core.pipeline.path_planner import (
    PathPlannerExecutionGroups,
    PathPlannerGroupScope,
)
from openhcs.core.pipeline.step_snapshot import StepSnapshot
from openhcs.core.progress import set_progress_queue
from openhcs.core.source_bindings import (
    NamedSourceBinding,
    SourceProjectionRole,
    StepSourceBindingsConfig,
)
from openhcs.core.steps.function_step import FunctionStep
from openhcs.interop.cellprofiler.plate_workspace import (
    CellProfilerPlateWorkspacePreparer,
)

COMBINE_OBJECTS_CPIPE = """CellProfiler Pipeline: http://www.cellprofiler.org
Version:5
ModuleCount:3
HasImagePlaneDetails:False

Images:[module_num:1|enabled:True]
    Filter images?:Images only
    Select the rule criteria:and (extension does isimage)

NamesAndTypes:[module_num:2|enabled:True]
    Image set matching method:Order
    Assignments count:2
    Single images count:0
    Select the rule criteria:and (file does contain "A")
    Name to assign these objects:A
    Select the image type:Objects
    Select the rule criteria:and (file does contain "B")
    Name to assign these objects:B
    Select the image type:Objects

CombineObjects:[module_num:3|enabled:True]
    Select initial object set:A
    Select object set to combine:B
    Select how to handle overlapping objects:Merge
    Name the combined object set:CombinedObjects
"""


def test_artifact_owned_scope_uses_exact_source_artifact_component_identity() -> None:
    source = ArtifactSpec.input("source_labels", ObjectLabelsArtifactType)
    output = ArtifactSpec.output_inheriting_group_scope(
        "combined_labels",
        ObjectLabelsArtifactType,
        source,
    )

    @artifact_inputs(source)
    @artifact_outputs(output)
    def combine(object_labels):
        return object_labels

    source_bindings = StepSourceBindingsConfig(
        enabled=True,
        bindings=(
            NamedSourceBinding(
                alias=source.name,
                artifact_kind=ObjectLabelsArtifactType,
                projection_role=SourceProjectionRole.SOURCE_ARTIFACT,
            ),
        ),
    )
    step = FunctionStep(
        func=combine,
        name="combine",
        processing_config=LazyProcessingConfig(
            variable_components=[VariableComponents.SITE],
            group_by=GroupBy.CHANNEL,
            input_source=InputSource.PIPELINE_START,
        ),
        source_bindings=source_bindings,
    )
    snapshot = StepSnapshot(index=0, scope_id="combine", step=step)
    planner = SimpleNamespace(
        declared={},
        artifact_context=ArtifactDeclarationStepContext(
            available_artifacts=ArtifactSpecCollection((source,)),
        ),
        session=SimpleNamespace(
            realized_source_metadata=({"source_alias": source.name, "channel": "2"},),
        ),
        source_bindings_for_snapshot=lambda _snapshot: source_bindings,
    )

    scope = PathPlannerExecutionGroups(planner).artifact_owned_execution_scope(
        snapshot,
        (CallableContract.from_callable(combine),),
        consumer_scope=PathPlannerGroupScope.from_raw(
            ("1", "2"),
            component=AllComponents.CHANNEL,
        ),
    )

    assert scope == PathPlannerGroupScope.from_raw(
        ("2",),
        component=AllComponents.CHANNEL,
    )


def test_public_combineobjects_import_compiles_source_owned_scope(
    tmp_path: Path,
) -> None:
    tifffile.imwrite(tmp_path / "A.tif", np.ones((4, 4), dtype=np.uint16))
    tifffile.imwrite(tmp_path / "B.tif", np.full((4, 4), 2, dtype=np.uint16))
    cppipe_path = tmp_path / "CombineObjectsDemo.cppipe"
    cppipe_path.write_text(COMBINE_OBJECTS_CPIPE, encoding="utf-8")

    prepared = CellProfilerPlateWorkspacePreparer(
        tmp_path,
        cppipe_path=cppipe_path,
    ).prepare()

    assert prepared.pipeline_import_error is None
    assert prepared.pipeline_steps is not None
    assert prepared.pipeline_config is not None

    ensure_global_config_context(GlobalPipelineConfig, GlobalPipelineConfig())
    orchestrator = PipelineOrchestrator(
        prepared.execution_plate_path,
        pipeline_config=prepared.pipeline_config,
    )
    set_progress_queue(SimpleQueue())
    try:
        orchestrator.initialize()
        compilation = orchestrator.compile_pipelines(
            pipeline_definition=prepared.pipeline_steps,
            well_filter=["A01"],
            is_zmq_execution=True,
        )
    finally:
        set_progress_queue(None)

    context = compilation.runtime_contexts["A01"]
    step_plan = context.step_plans[0]
    assert tuple(step_plan.variable_components) == (VariableComponents.SITE,)
    assert step_plan.group_by is GroupBy.CHANNEL
    assert step_plan.execution_group_scope == PathPlannerGroupScope.from_raw(
        ("1",),
        component=AllComponents.CHANNEL,
    )
    assert tuple(
        binding.input_spec().ref()
        for binding in step_plan.source_binding_plan.binding_declarations
    ) == tuple(
        edge.spec.ref()
        for edge in next(
            step_plan.compiled_function_pattern.iter_invocations()
        ).artifact_input_edges
    )

    edges = next(
        step_plan.compiled_function_pattern.iter_invocations()
    ).artifact_input_edges
    assert tuple(edge.spec.name for edge in edges) == ("A", "B")
    assert all(edge.spec.artifact_type is ObjectLabelsArtifactType for edge in edges)
    assert all(edge.spec.parameter_name == "object_labels" for edge in edges)
    assert all(edge.storage_plan is None and edge.projection is None for edge in edges)
