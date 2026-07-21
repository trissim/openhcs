"""Generic lifecycle-anchor selection contracts for FunctionStep execution."""

from __future__ import annotations

from types import SimpleNamespace

from openhcs.constants.constants import AllComponents
from openhcs.core.artifacts import (
    ArtifactInputPlan,
    ArtifactSpec,
    MeasurementsArtifactType,
)
from openhcs.core.component_group_scope import ComponentGroupScope
from openhcs.core.component_set import ComponentSet
from openhcs.core.function_patterns import (
    RuntimeInvocationDomain,
    compile_function_pattern,
)
from openhcs.core.pipeline.function_contracts import artifact_inputs
from openhcs.core.pipeline.path_planner import PathPlanner, PathPlannerArtifactStage
from openhcs.core.runtime_adapters import runtime_adapter
from openhcs.core.source_bindings import CompiledSourceBindingPlan
from openhcs.core.step_dependencies import StepInputDependency
from openhcs.core.steps.function_execution import PatternGroups, StepAnchorPatternFilter


def _anchor_filter(plan: object, output_manifest: object) -> StepAnchorPatternFilter:
    return StepAnchorPatternFilter(
        plan=plan,
        parser=object(),
        output_manifest=output_manifest,
        source_workspace_authority=None,
        source_workspace_projection_cache=None,
    )


def test_storage_backed_cross_group_uses_producer_lifecycle_anchor() -> None:
    measurements = ArtifactSpec.input(
        "measurements",
        MeasurementsArtifactType,
    )

    @artifact_inputs(measurements)
    @runtime_adapter(
        "runtime",
        lambda _request: object(),
        manages_artifact_inputs=True,
    )
    def consume_measurements(image, *, runtime):
        del runtime
        return image

    measurements_plan = ArtifactInputPlan(
        name=measurements.name,
        path="/memory/measurements.pkl",
        artifact_type=MeasurementsArtifactType,
        source_step_id=7,
        source_step_scope_id="measurement-producer",
        group_keys=("2",),
        group_component=AllComponents.CHANNEL,
        paths_by_group={"2": "/memory/measurements_w2.pkl"},
    )
    compiled_pattern = compile_function_pattern(
        consume_measurements,
        {plan.ref(): plan for plan in (measurements_plan,)},
        {},
    )
    compiled_pattern = PathPlannerArtifactStage(
        PathPlanner.__new__(PathPlanner)
    ).compile_invocation_input_edges(
        compiled_pattern,
        artifact_inputs={measurements_plan.ref(): measurements_plan},
        relation_source_scopes={
            measurements.ref(): measurements_plan.producer_group_scope(),
        },
        execution_group_scope=ComponentGroupScope.from_raw(
            ("1",),
            component=AllComponents.CHANNEL,
        ),
        consumer_variable_components=ComponentSet(),
    )
    invocation = next(compiled_pattern.iter_invocations())
    assert invocation.runtime_domain is RuntimeInvocationDomain.ARTIFACT_MANAGED
    assert all(
        edge.storage_plan is not None and not edge.consumes_main_flow
        for edge in invocation.artifact_input_edges
    )

    plan = SimpleNamespace(
        step_index=9,
        step_name="StorageBackedConsumer",
        main_input_dependency=StepInputDependency.step_output(
            source_step_index=4,
            source_step_scope_id="non-adjacent-main-flow-producer",
        ),
        source_binding_plan=CompiledSourceBindingPlan.empty(),
        execution_group_value="channel",
        execution_group_scope=ComponentGroupScope.from_raw(
            ("1",),
            component=AllComponents.CHANNEL,
        ),
        compiled_function_pattern=compiled_pattern,
    )

    class ExactProducerManifest:
        calls = 0

        def filter_to_producer_paths(self, _plan, paths, _parser):
            self.calls += 1
            return [path for path in paths if "second" in path]

    manifest = ExactProducerManifest()
    filtered = _anchor_filter(plan, manifest).filtered(
        PatternGroups(
            {
                "1": ("first-anchor.tif",),
                "2": ("second-anchor.tif",),
            }
        )
    )

    assert manifest.calls == 2
    assert filtered.groups == {"1": ("second-anchor.tif",)}


def test_source_anchored_group_uses_exact_main_flow_producer_manifest() -> None:
    compiled_pattern = compile_function_pattern(lambda image: image, {}, {})
    invocation = next(compiled_pattern.iter_invocations())
    assert invocation.runtime_domain is RuntimeInvocationDomain.SOURCE_ANCHORED
    plan = SimpleNamespace(
        step_index=6,
        step_name="SourceConsumer",
        main_input_dependency=StepInputDependency.step_output(
            source_step_index=2,
            source_step_scope_id="exact-main-flow-producer",
        ),
        source_binding_plan=CompiledSourceBindingPlan.empty(),
        execution_group_value=None,
        execution_group_scope=ComponentGroupScope.ungrouped(),
        compiled_function_pattern=compiled_pattern,
    )

    class ExactProducerManifest:
        calls = 0

        def filter_to_producer_paths(self, selected_plan, paths, _parser):
            self.calls += 1
            assert selected_plan.main_input_dependency.source_step_index == 2
            return [path for path in paths if "producer" in path]

    manifest = ExactProducerManifest()
    filtered = _anchor_filter(plan, manifest).filtered(
        PatternGroups(
            {None: ("unrelated-anchor.tif", "producer-anchor.tif")}
        )
    )

    assert manifest.calls == 1
    assert filtered.groups == {None: ("producer-anchor.tif",)}
