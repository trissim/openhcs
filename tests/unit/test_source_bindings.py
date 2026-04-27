import pickle

import pytest

from openhcs.constants.constants import AllComponents, GroupBy, VariableComponents
from openhcs.core.source_bindings import (
    CompiledSourceBindingPlan,
    ComponentSelector,
    GroupedSourceBindings,
    MetadataSelector,
    NamedSourceBinding,
    SourceBindingOrigin,
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
    )

    plan = CompiledSourceBindingPlan.from_config(config)

    assert not plan.is_empty
    assert tuple(plan.bindings_by_group) == ("dna",)
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
        )
    )

    restored = pickle.loads(pickle.dumps(plan))

    assert restored == plan
    assert restored.binding_for_alias("OrigBlue", "dna") == plan.binding_for_alias(
        "OrigBlue",
        "dna",
    )
