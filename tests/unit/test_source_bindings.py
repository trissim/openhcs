import pickle
import subprocess
import sys

import pytest

from openhcs.config_framework.lazy_factory import ensure_global_config_context
from openhcs.config_framework.object_state import ObjectState
from openhcs.config_framework.object_state_registry import ObjectStateRegistry
from openhcs.constants import Microscope
from openhcs.constants.constants import AllComponents, GroupBy, VariableComponents
from openhcs.core.artifacts import ArtifactKind
from openhcs.core.config import (
    GlobalPipelineConfig,
    LazyStepSourceBindingsConfig,
    PipelineConfig,
)
from openhcs.core.pipeline.step_snapshot import StepSnapshot
from openhcs.core.source_bindings import (
    CompiledSourceBindingPlan,
    CompiledSourceUniversePlan,
    ComponentSelector,
    MetadataExtractionRule,
    MetadataSource,
    MetadataSelector,
    NamedSourceBinding,
    SourceBindingMatchDimension,
    SourceBindingMatchField,
    SourceBindingMatchMethod,
    SourceBindingMatchPlan,
    SourceBindingOrigin,
    SourceFilterClause,
    SourceFilterMatchType,
    SourceFilterSubject,
    SourceBindingRuntimeContext,
    SourceSelector,
    SourceBindingsConfig,
    StepSourceBindingsConfig,
)
from openhcs.core.steps.function_step import FunctionStep
from openhcs.core.source_workspace_projection import VirtualWorkspaceSourceProjection


def test_component_selector_coerces_existing_component_vocabulary():
    selector = ComponentSelector(component=GroupBy.CHANNEL, value=1)

    assert selector.component is AllComponents.CHANNEL
    assert selector.value == "1"

    variable_selector = ComponentSelector(
        component=VariableComponents.SITE,
        value="3",
    )

    assert variable_selector.component is AllComponents.SITE


def test_named_source_binding_normalizes_origin_and_requires_alias():
    binding = NamedSourceBinding(
        alias="OrigBlue",
        origin="pipeline_start",
    )

    assert binding.origin is SourceBindingOrigin.PIPELINE_START
    assert binding.artifact_kind is ArtifactKind.IMAGE

    objects_binding = NamedSourceBinding(
        alias="Nuclei",
        artifact_kind="object_labels",
    )

    assert objects_binding.artifact_kind is ArtifactKind.OBJECT_LABELS

    with pytest.raises(ValueError, match="alias cannot be empty"):
        NamedSourceBinding(alias="")


def test_step_source_bindings_reject_duplicate_aliases():
    with pytest.raises(ValueError, match="duplicate alias"):
        StepSourceBindingsConfig(
            bindings=(
                NamedSourceBinding(alias="OrigBlue"),
                NamedSourceBinding(alias="OrigBlue"),
            )
        )


def test_step_source_bindings_global_config_merge_resolves_lazy_default():
    ensure_global_config_context(GlobalPipelineConfig, GlobalPipelineConfig())
    merged = ObjectState(PipelineConfig()).to_saved_resolved_object()

    assert isinstance(merged, GlobalPipelineConfig)
    assert isinstance(merged.step_source_bindings_config, StepSourceBindingsConfig)
    assert merged.step_source_bindings_config.is_empty
    assert merged.step_source_bindings_config.bindings == ()
    assert merged.step_source_bindings_config.metadata_rules == ()
    assert merged.step_source_bindings_config.match_plan is None


def test_lazy_step_source_bindings_preserve_inherited_payload_sentinels():
    config = LazyStepSourceBindingsConfig()

    assert object.__getattribute__(config, "bindings") is None
    assert object.__getattribute__(config, "source_filters") is None
    assert object.__getattribute__(config, "metadata_rules") is None
    assert object.__getattribute__(config, "match_plan") is None


def test_step_source_bindings_keeps_class_identity_after_config_import():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "\n".join(
                [
                    "from openhcs.core import invocation_artifacts, source_bindings",
                    "from openhcs.core.config import StepSourceBindingsConfig as cfg_type",
                    "obj = cfg_type()",
                    "assert invocation_artifacts.StepSourceBindingsConfig is source_bindings.StepSourceBindingsConfig",
                    "assert cfg_type is source_bindings.StepSourceBindingsConfig",
                    "assert isinstance(obj, invocation_artifacts.StepSourceBindingsConfig)",
                ]
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_step_source_bindings_inherit_plate_source_bindings_for_snapshot():
    ObjectStateRegistry.clear()
    binding = NamedSourceBinding(alias="DNA")
    ensure_global_config_context(GlobalPipelineConfig, GlobalPipelineConfig())
    pipeline_state = ObjectState(
        PipelineConfig(
            source_bindings_config=SourceBindingsConfig(bindings=(binding,)),
        ),
        scope_id="plate",
    )

    try:
        ObjectStateRegistry.register(pipeline_state, _skip_snapshot=True)
        step = FunctionStep(func=lambda image: image)
        step_state = ObjectState(
            step,
            scope_id="plate::step_0",
            parent_state=pipeline_state,
            exclude_params=["func"],
        )
        ObjectStateRegistry.register(step_state, _skip_snapshot=True)
        snapshot = StepSnapshot.from_resolved_step(
            index=0,
            step=step_state.to_object(),
            step_state=step_state,
        )
    finally:
        ObjectStateRegistry.clear()

    assert snapshot.source_bindings.bindings == (binding,)
    assert CompiledSourceBindingPlan.from_config(
        snapshot.source_bindings,
    ).is_empty


def test_enabled_step_source_bindings_compile_inherited_bindings():
    ObjectStateRegistry.clear()
    binding = NamedSourceBinding(alias="DNA")
    ensure_global_config_context(GlobalPipelineConfig, GlobalPipelineConfig())
    pipeline_state = ObjectState(
        PipelineConfig(
            source_bindings_config=SourceBindingsConfig(bindings=(binding,)),
        ),
        scope_id="plate",
    )

    try:
        ObjectStateRegistry.register(pipeline_state, _skip_snapshot=True)
        step = FunctionStep(
            func=lambda image: image,
            source_bindings=LazyStepSourceBindingsConfig(enabled=True),
        )
        step_state = ObjectState(
            step,
            scope_id="plate::step_0",
            parent_state=pipeline_state,
            exclude_params=["func"],
        )
        ObjectStateRegistry.register(step_state, _skip_snapshot=True)
        snapshot = StepSnapshot.from_resolved_step(
            index=0,
            step=step_state.to_object(),
            step_state=step_state,
        )
    finally:
        ObjectStateRegistry.clear()

    assert snapshot.source_bindings.bindings == (binding,)
    assert CompiledSourceBindingPlan.from_config(
        snapshot.source_bindings,
    ).bindings == (binding,)


def test_pipeline_step_source_bindings_enabled_inherits_to_function_steps():
    ObjectStateRegistry.clear()
    binding = NamedSourceBinding(alias="DNA")
    ensure_global_config_context(GlobalPipelineConfig, GlobalPipelineConfig())
    pipeline_state = ObjectState(
        PipelineConfig(
            step_source_bindings_config=LazyStepSourceBindingsConfig(
                bindings=(binding,),
                enabled=True,
            ),
        ),
        scope_id="plate",
    )

    try:
        ObjectStateRegistry.register(pipeline_state, _skip_snapshot=True)
        step = FunctionStep(func=lambda image: image)
        step_state = ObjectState(
            step,
            scope_id="plate::step_0",
            parent_state=pipeline_state,
            exclude_params=["func"],
        )
        ObjectStateRegistry.register(step_state, _skip_snapshot=True)
        snapshot = StepSnapshot.from_resolved_step(
            index=0,
            step=step_state.to_object(),
            step_state=step_state,
        )
    finally:
        ObjectStateRegistry.clear()

    assert snapshot.source_bindings.enabled is True
    assert snapshot.source_bindings.bindings == (binding,)
    assert CompiledSourceBindingPlan.from_config(
        snapshot.source_bindings,
    ).bindings == (binding,)


def test_saved_resolved_pipeline_config_preserves_scalar_override():
    ensure_global_config_context(
        GlobalPipelineConfig,
        GlobalPipelineConfig(microscope=Microscope.AUTO, num_workers=7),
    )

    merged = ObjectState(
        PipelineConfig(microscope=Microscope.OPENHCS)
    ).to_saved_resolved_object()

    assert isinstance(merged, GlobalPipelineConfig)
    assert merged.microscope is Microscope.OPENHCS
    assert merged.num_workers == 7


def test_orchestrator_microscope_init_uses_saved_resolved_pipeline_config(
    tmp_path,
    monkeypatch,
):
    from openhcs.core.orchestrator import orchestrator as orchestrator_module
    from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator

    class DummyHandler:
        microscope_type = "dummy"

    ensure_global_config_context(
        GlobalPipelineConfig,
        GlobalPipelineConfig(microscope=Microscope.AUTO),
    )
    captured_kwargs = {}

    def fake_create_microscope_handler(**kwargs):
        captured_kwargs.update(kwargs)
        return DummyHandler()

    monkeypatch.setattr(
        orchestrator_module,
        "create_microscope_handler",
        fake_create_microscope_handler,
    )

    orchestrator = PipelineOrchestrator(
        tmp_path,
        pipeline_config=PipelineConfig(microscope=Microscope.OPENHCS),
    )
    orchestrator.initialize_microscope_handler()

    assert captured_kwargs["microscope_type"] == Microscope.OPENHCS.value


def test_source_bindings_expose_generic_resolution_requirements():
    config = StepSourceBindingsConfig(
        bindings=(
            NamedSourceBinding(
                alias="DNA",
                selector=SourceSelector(
                    components=(ComponentSelector(AllComponents.CHANNEL, "1"),)
                ),
            ),
            NamedSourceBinding(
                alias="IllumDNA",
                origin=SourceBindingOrigin.PIPELINE_START,
            ),
        )
    )

    assert config.requires_step_input_channel_stack
    assert config.requires_pipeline_start_resolution
    assert config.bindings[0].requires_selector_resolution
    assert not config.bindings[1].requires_step_input_channel_stack


def test_compiled_source_binding_plan_preserves_named_selectors():
    config = StepSourceBindingsConfig(
        bindings=(
            NamedSourceBinding(
                alias="OrigBlue",
                selector=SourceSelector(
                    components=(
                        ComponentSelector("channel", "1"),
                        ComponentSelector(AllComponents.SITE, "3"),
                    ),
                    metadata=(MetadataSelector("stain", "DAPI"),),
                ),
            ),
        ),
        metadata_rules=(
            MetadataExtractionRule(
                source=MetadataSource.FILE_NAME,
                pattern=r".*(?P<plate>PlateA)\.tif",
                filters=(
                    SourceFilterClause(
                        subject=SourceFilterSubject.FILE,
                        match_type=SourceFilterMatchType.CONTAINS,
                        value="PlateA",
                    ),
                ),
            ),
        ),
        match_plan=SourceBindingMatchPlan(
            method=SourceBindingMatchMethod.METADATA,
            dimensions=(
                SourceBindingMatchDimension(
                    fields=(
                        SourceBindingMatchField(
                            alias="OrigBlue",
                            metadata_field="plate",
                        ),
                        SourceBindingMatchField(
                            alias="IllumBlue",
                            metadata_field="plate_illum",
                        ),
                    ),
                ),
            ),
        ),
        enabled=True,
    )

    plan = CompiledSourceBindingPlan.from_config(config)

    assert not plan.is_empty
    assert len(plan.bindings) == 1
    assert plan.metadata_rules[0].source is MetadataSource.FILE_NAME
    assert plan.match_plan is not None
    assert plan.match_plan.method is SourceBindingMatchMethod.METADATA
    binding = plan.bindings[0]
    assert binding.alias == "OrigBlue"
    assert binding.selector.components[0].component is AllComponents.CHANNEL
    assert binding.selector.metadata[0].field == "stain"
    assert plan.binding_for_alias("OrigBlue") == binding
    assert plan.binding_for_alias("Missing") is None


def test_pipeline_start_binding_does_not_force_full_source_universe():
    binding_plan = CompiledSourceBindingPlan.from_config(
        StepSourceBindingsConfig(
            bindings=(
                NamedSourceBinding(
                    alias="OrigBlue",
                    selector=SourceSelector(
                        components=(ComponentSelector(AllComponents.CHANNEL, "1"),),
                    ),
                    origin=SourceBindingOrigin.PIPELINE_START,
                ),
            ),
            enabled=True,
        )
    )

    universe_plan = CompiledSourceUniversePlan.from_source_binding_plan(binding_plan)

    assert universe_plan.uses_pipeline_start_binding_origin
    assert not universe_plan.requires_full_pipeline_source_universe


def test_virtual_workspace_projection_filters_source_metadata_by_axis():
    projection = VirtualWorkspaceSourceProjection(
        source_paths_by_virtual_path={
            "A01_s001_w1_z001_t001.tif": "/source/A01_w1.tif",
            "A02_s001_w1_z001_t001.tif": "/source/A02_w1.tif",
        },
        source_metadata_by_path={
            "A01_s001_w1_z001_t001.tif": {"well": "A01", "channel": "1"},
            "/source/A01_w1.tif": {"well": "A01", "channel": "1"},
            "A02_s001_w1_z001_t001.tif": {"well": "A02", "channel": "1"},
            "/source/A02_w1.tif": {"well": "A02", "channel": "1"},
        },
    )

    filtered = projection.filtered_by_axis(axis_id="A01")

    assert tuple(filtered.source_paths_by_virtual_path) == (
        "A01_s001_w1_z001_t001.tif",
    )
    assert filtered.source_metadata_by_path["/source/A01_w1.tif"]["well"] == "A01"
    assert "/source/A02_w1.tif" not in filtered.source_metadata_by_path


def test_compiled_source_binding_plan_round_trips_through_pickle():
    plan = CompiledSourceBindingPlan.from_config(
        StepSourceBindingsConfig(
            bindings=(
                NamedSourceBinding(
                    alias="OrigBlue",
                    selector=SourceSelector(
                        components=(ComponentSelector("channel", "1"),),
                    ),
                    origin=SourceBindingOrigin.STEP_INPUT,
                ),
            ),
            enabled=True,
            metadata_rules=(
                MetadataExtractionRule(
                    source=MetadataSource.FOLDER_NAME,
                    pattern=r"(?P<plate>PlateA)",
                ),
            ),
            match_plan=SourceBindingMatchPlan(
                method=SourceBindingMatchMethod.METADATA,
                dimensions=(
                    SourceBindingMatchDimension(
                        fields=(
                            SourceBindingMatchField(
                                alias="OrigBlue",
                                metadata_field="plate",
                            ),
                        ),
                    ),
                ),
            ),
        )
    )

    restored = pickle.loads(pickle.dumps(plan))

    assert restored == plan
    assert restored.metadata_rules == plan.metadata_rules
    assert restored.match_plan == plan.match_plan
    assert restored.binding_for_alias("OrigBlue") == plan.binding_for_alias("OrigBlue")


def test_source_binding_runtime_context_preserves_source_provenance_through_pickle():
    context = SourceBindingRuntimeContext(
        step_input_files=("A01_s001_w1_z001_t001.tif",),
        step_input_dir="/workspace",
        step_input_source_paths={
            "A01_s001_w1_z001_t001.tif": "/real/source_C20_w1.tif",
        },
        source_metadata_by_path={
            "A01_s001_w1_z001_t001.tif": {"Compound": "DMSO"},
        },
        pipeline_input_files=("/real/source_C20_w1.tif",),
        pipeline_input_backend="disk",
    )

    restored = pickle.loads(pickle.dumps(context))

    assert restored == context
    assert dict(restored.step_input_source_paths) == {
        "A01_s001_w1_z001_t001.tif": "/real/source_C20_w1.tif",
    }
    assert dict(restored.source_metadata_by_path["A01_s001_w1_z001_t001.tif"]) == {
        "Compound": "DMSO",
    }
