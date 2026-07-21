import pytest

from openhcs.constants.constants import AllComponents
from openhcs.core.artifacts import (
    ArtifactInputPlan,
    ArtifactInputProjectionPlan,
    ImageArtifactType,
    MeasurementsArtifactType,
    ObjectLabelsArtifactType,
)
from openhcs.core.component_group_scope import ComponentGroupScope
from openhcs.core.component_set import ComponentSet


def _projection(
    storage_plan: ArtifactInputPlan,
    *,
    invocation_scope: ComponentGroupScope,
    producer_selection_scope: ComponentGroupScope,
    component_scopes: tuple[ComponentGroupScope, ...] = (),
    consumer_variable_components: tuple[AllComponents, ...] = (),
) -> ArtifactInputProjectionPlan:
    return ArtifactInputProjectionPlan(
        invocation_scope=invocation_scope,
        producer_selection_scope=producer_selection_scope,
        component_scopes=component_scopes,
        consumer_variable_components=consumer_variable_components,
    )


@pytest.mark.parametrize(
    "invocation_scope",
    (
        ComponentGroupScope.ungrouped(),
        ComponentGroupScope.dynamic(AllComponents.CHANNEL),
    ),
    ids=("ungrouped-invocation", "dynamic-dispatch-invocation"),
)
def test_grouped_producer_selection_domain_is_independent_of_invocation_dispatch(
    invocation_scope: ComponentGroupScope,
) -> None:
    storage_plan = ArtifactInputPlan(
        name="measurements",
        path="/memory/measurements.pkl",
        artifact_type=MeasurementsArtifactType,
        group_keys=("1", "2"),
        group_component=AllComponents.CHANNEL,
    )
    projection = _projection(
        storage_plan,
        invocation_scope=invocation_scope,
        producer_selection_scope=storage_plan.producer_group_scope(),
        consumer_variable_components=(AllComponents.SITE,),
    )

    projection.validate_axis_projection(storage_plan)


def test_dynamic_producer_coordinate_requires_matching_invocation_dispatch() -> None:
    storage_plan = ArtifactInputPlan(
        name="objects",
        path="/memory/objects.pkl",
        artifact_type=ObjectLabelsArtifactType,
        group_keys=(None,),
        group_component=AllComponents.CHANNEL,
    )
    projection = _projection(
        storage_plan,
        invocation_scope=ComponentGroupScope.dynamic(AllComponents.SITE),
        producer_selection_scope=ComponentGroupScope.dynamic(
            AllComponents.CHANNEL
        ),
        consumer_variable_components=(AllComponents.TIMEPOINT,),
    )

    with pytest.raises(ValueError, match="not owned by invocation scope"):
        projection.validate_axis_projection(storage_plan)


def test_existing_producer_stack_is_retained_when_consumer_relabels_third_axis() -> None:
    storage_plan = ArtifactInputPlan(
        name="objects",
        path="/memory/objects.pkl",
        artifact_type=ObjectLabelsArtifactType,
        group_keys=("1",),
        group_component=AllComponents.CHANNEL,
        variable_components=(AllComponents.SITE,),
    )
    consumer_components = ComponentSet((AllComponents.TIMEPOINT,))
    projection = _projection(
        storage_plan,
        invocation_scope=ComponentGroupScope.ungrouped(),
        producer_selection_scope=storage_plan.producer_group_scope(),
        consumer_variable_components=consumer_components.as_tuple(),
    )

    assert storage_plan.retains_producer_stack(consumer_components)
    assert storage_plan.runtime_variable_components(
        consumer_components
    ) == consumer_components
    assert projection.projected_variable_components(storage_plan) == ComponentSet()
    projection.validate_axis_projection(storage_plan)


def test_scalar_consumer_requires_coordinate_for_each_producer_stack_component() -> None:
    storage_plan = ArtifactInputPlan(
        name="objects",
        path="/memory/objects.pkl",
        artifact_type=ObjectLabelsArtifactType,
        group_keys=("1",),
        group_component=AllComponents.CHANNEL,
        variable_components=(AllComponents.SITE,),
    )
    projection = _projection(
        storage_plan,
        invocation_scope=ComponentGroupScope.ungrouped(),
        producer_selection_scope=storage_plan.producer_group_scope(),
    )

    with pytest.raises(ValueError, match="site.*without an exact coordinate"):
        projection.validate_axis_projection(storage_plan)


def test_transposed_producer_group_axis_requires_old_stack_coordinate() -> None:
    storage_plan = ArtifactInputPlan(
        name="image",
        path="/memory/image.pkl",
        artifact_type=ImageArtifactType,
        group_keys=("1", "2"),
        group_component=AllComponents.CHANNEL,
        variable_components=(AllComponents.SITE,),
    )
    consumer_components = ComponentSet((AllComponents.CHANNEL,))
    projection = _projection(
        storage_plan,
        invocation_scope=ComponentGroupScope.dynamic(AllComponents.CHANNEL),
        producer_selection_scope=storage_plan.producer_group_scope(),
        component_scopes=(
            ComponentGroupScope.dynamic(AllComponents.CHANNEL),
        ),
        consumer_variable_components=consumer_components.as_tuple(),
    )

    assert storage_plan.composes_producer_groups(consumer_components)
    assert not storage_plan.retains_producer_stack(consumer_components)
    assert storage_plan.runtime_variable_components(
        consumer_components
    ) == consumer_components
    with pytest.raises(ValueError, match="site.*without an exact coordinate"):
        projection.validate_axis_projection(storage_plan)


def test_transposed_producer_stack_rejects_multi_coordinate_projection() -> None:
    storage_plan = ArtifactInputPlan(
        name="image",
        path="/memory/image.pkl",
        artifact_type=ImageArtifactType,
        group_keys=("1", "2"),
        group_component=AllComponents.CHANNEL,
        variable_components=(AllComponents.SITE,),
    )
    projection = _projection(
        storage_plan,
        invocation_scope=ComponentGroupScope.dynamic(AllComponents.CHANNEL),
        producer_selection_scope=storage_plan.producer_group_scope(),
        component_scopes=(
            ComponentGroupScope.from_raw(
                ("1", "2", "3"),
                component=AllComponents.SITE,
            ),
        ),
        consumer_variable_components=(AllComponents.CHANNEL,),
    )

    with pytest.raises(ValueError, match="site.*not a single exact coordinate"):
        projection.validate_axis_projection(storage_plan)
