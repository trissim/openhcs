import pickle

import pytest

from openhcs.constants.constants import AllComponents, GroupBy, VariableComponents
from openhcs.core.artifacts import ArtifactKind
from openhcs.core.config import GlobalPipelineConfig, PipelineConfig
from openhcs.core.orchestrator.orchestrator import _create_merged_config
from openhcs.core.source_bindings import (
    CompiledSourceBindingPlan,
    ComponentSelector,
    GroupedSourceBindings,
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
    StepSourceBindingsConfig,
)


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


def test_step_source_bindings_reject_duplicate_aliases_and_group_keys():
    with pytest.raises(ValueError, match="duplicate alias"):
        GroupedSourceBindings(
            bindings=(
                NamedSourceBinding(alias="OrigBlue"),
                NamedSourceBinding(alias="OrigBlue"),
            )
        )

    with pytest.raises(ValueError, match="duplicate group key"):
        StepSourceBindingsConfig(
            groups=(
                GroupedSourceBindings(group_key="dna"),
                GroupedSourceBindings(group_key="dna"),
            )
        )


def test_step_source_bindings_global_config_merge_resolves_lazy_default():
    merged = _create_merged_config(PipelineConfig(), GlobalPipelineConfig())

    assert isinstance(merged.step_source_bindings_config, StepSourceBindingsConfig)
    assert merged.step_source_bindings_config.is_empty
    assert merged.step_source_bindings_config.groups == ()
    assert merged.step_source_bindings_config.metadata_rules == ()
    assert merged.step_source_bindings_config.match_plan is None


def test_source_bindings_expose_generic_resolution_requirements():
    config = StepSourceBindingsConfig(
        groups=(
            GroupedSourceBindings(
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
                ),
            ),
        )
    )

    assert config.requires_step_input_channel_stack
    assert config.requires_pipeline_start_resolution
    assert config.groups[0].bindings[0].requires_selector_resolution
    assert not config.groups[0].bindings[1].requires_step_input_channel_stack


def test_compiled_source_binding_plan_preserves_grouped_named_selectors():
    config = StepSourceBindingsConfig(
        groups=(
            GroupedSourceBindings(
                group_key="dna",
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
            ),
        )
        ,
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
    )

    plan = CompiledSourceBindingPlan.from_config(config)

    assert not plan.is_empty
    assert tuple(plan.bindings_by_group) == ("dna",)
    assert plan.metadata_rules[0].source is MetadataSource.FILE_NAME
    assert plan.match_plan is not None
    assert plan.match_plan.method is SourceBindingMatchMethod.METADATA
    binding = plan.bindings_by_group["dna"][0]
    assert binding.alias == "OrigBlue"
    assert binding.selector.components[0].component is AllComponents.CHANNEL
    assert binding.selector.metadata[0].field == "stain"
    assert plan.binding_for_alias("OrigBlue", "dna") == binding
    assert plan.binding_for_alias("Missing", "dna") is None


def test_compiled_source_binding_plan_round_trips_through_pickle():
    plan = CompiledSourceBindingPlan.from_config(
        StepSourceBindingsConfig(
            groups=(
                GroupedSourceBindings(
                    group_key="dna",
                    bindings=(
                        NamedSourceBinding(
                            alias="OrigBlue",
                            selector=SourceSelector(
                                components=(ComponentSelector("channel", "1"),),
                            ),
                            origin=SourceBindingOrigin.STEP_INPUT,
                        ),
                    ),
                ),
            ),
            metadata_rules=(
                MetadataExtractionRule(
                    source=MetadataSource.FOLDER_NAME,
                    pattern=r".*/(?P<plate>PlateA)/.*",
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
    assert restored.binding_for_alias("OrigBlue", "dna") == plan.binding_for_alias(
        "OrigBlue",
        "dna",
    )


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
