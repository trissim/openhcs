from types import SimpleNamespace

import pytest

from openhcs.constants.constants import AllComponents
from openhcs.core.artifacts import (
    ArtifactOutputPlan,
    ArtifactSpec,
    GroupLineageSourceRelation,
    MeasurementsArtifactType,
)
from openhcs.core.component_group_scope import ComponentGroupScope
from openhcs.core.function_patterns import compile_function_pattern
from openhcs.core.pipeline.function_contracts import (
    artifact_inputs,
    artifact_outputs,
    composed_image_payload,
)
from openhcs.core.source_binding_selection import SourcePatternResolutionContext
from openhcs.core.source_bindings import (
    ComponentSelector,
    CompiledSourceBindingPlan,
    NamedSourceBinding,
    SourceBindingMatchDimension,
    SourceBindingMatchField,
    SourceBindingMatchMethod,
    SourceBindingMatchPlan,
    SourceBindingOrigin,
    SourceSelector,
)
from openhcs.core.step_dependencies import StepInputDependency
from openhcs.core.steps.function_execution import (
    PatternGroups,
    StepAnchorPatternFilter,
)
from openhcs.microscopes.source_schema import SourceSchemaFilenameParser


SOURCE_ALIASES = (
    ("OrigER", "2"),
    ("OrigHoechst", "1"),
    ("OrigMito", "5"),
    ("OrigPh_golgi", "4"),
    ("OrigSyto", "3"),
)


def _source_bindings() -> tuple[NamedSourceBinding, ...]:
    return tuple(
        NamedSourceBinding(
            alias=alias,
            selector=SourceSelector(
                components=(ComponentSelector(AllComponents.CHANNEL, channel),),
            ),
            origin=SourceBindingOrigin.PIPELINE_START,
            component_identity=(ComponentSelector(AllComponents.CHANNEL, channel),),
        )
        for alias, channel in SOURCE_ALIASES
    )


def _match_dimensions(
    aliases: tuple[str, ...],
) -> tuple[SourceBindingMatchDimension, ...]:
    return tuple(
        SourceBindingMatchDimension(
            fields=tuple(
                SourceBindingMatchField(alias=alias, metadata_field=field)
                for alias in aliases
            )
        )
        for field in ("Plate", "Well", "Site")
    )


def _filter_source_anchors(
    monkeypatch: pytest.MonkeyPatch,
    *,
    artifact_aliases: tuple[str, ...],
    sites: tuple[int, ...] = (1,),
    dimension_aliases: tuple[str, ...] | None = None,
    shared_output: bool = True,
    composed_image_set: bool = True,
) -> PatternGroups:
    available_bindings = _source_bindings()
    aliases = tuple(binding.alias for binding in available_bindings)
    compiled_bindings = tuple(
        binding for binding in available_bindings if binding.alias in artifact_aliases
    )
    source_binding_plan = CompiledSourceBindingPlan(
        bindings=compiled_bindings,
        match_plan=SourceBindingMatchPlan(
            method=SourceBindingMatchMethod.METADATA,
            dimensions=_match_dimensions(dimension_aliases or aliases),
        ),
    )
    declared_specs = tuple(binding.input_spec() for binding in compiled_bindings)
    source_aligned_outputs = tuple(
        ArtifactSpec.output(
            "Measurements" if shared_output else f"{binding.alias}Measurements",
            MeasurementsArtifactType,
            relations=tuple(
                GroupLineageSourceRelation(source=source_spec.ref())
                for source_spec in (
                    declared_specs if shared_output else (binding.input_spec(),)
                )
            ),
        )
        for binding in (compiled_bindings[:1] if shared_output else compiled_bindings)
    )
    output_specs = (
        source_aligned_outputs
        if shared_output
        else (
            *source_aligned_outputs,
            ArtifactSpec.output(
                "AggregateMeasurements",
                MeasurementsArtifactType,
                relations=tuple(
                    GroupLineageSourceRelation(source=source_spec.ref())
                    for source_spec in declared_specs
                ),
            ),
        )
    )

    @artifact_inputs(*declared_specs)
    @artifact_outputs(*output_specs)
    def measure_images(image):
        return image

    if composed_image_set:
        measure_images = composed_image_payload(measure_images)

    output_bindings = (
        compiled_bindings[:1]
        if shared_output
        else (*compiled_bindings, compiled_bindings[0])
    )
    output_plans = {
        output_spec.ref(): ArtifactOutputPlan(
            name=output_spec.name,
            path=f"/memory/{output_spec.name}.pkl",
            artifact_type=output_spec.artifact_type,
            group_keys=(
                tuple(binding.component_identity[0].value for binding in compiled_bindings)
                if shared_output or output_spec.name == "AggregateMeasurements"
                else (binding.component_identity[0].value,)
            ),
            group_component=AllComponents.CHANNEL,
            relations=output_spec.relations,
        )
        for output_spec, binding in zip(
            output_specs,
            output_bindings,
            strict=True,
        )
    }
    compiled_pattern = compile_function_pattern(
        measure_images,
        {},
        output_plans,
    )
    plan = SimpleNamespace(
        axis_id="A01",
        step_index=0,
        step_name="MeasureImages",
        main_input_dependency=StepInputDependency.pipeline_start(),
        source_binding_plan=source_binding_plan,
        execution_group_scope=ComponentGroupScope.from_raw(
            tuple(
                binding.component_identity[0].value
                for binding in compiled_bindings
            ),
            component=AllComponents.CHANNEL,
        ),
        compiled_function_pattern=compiled_pattern,
    )
    patterns = PatternGroups(
        {
            channel: tuple(
                f"A01_s{site:03d}_w{channel}_z001_t001.tif" for site in sites
            )
            for _alias, channel in SOURCE_ALIASES
        }
    )
    source_paths = {
        pattern: f"/source/{pattern}"
        for pattern_list in patterns.values()
        for pattern in pattern_list
    }
    source_metadata = {
        pattern: {
            "Plate": "Plate",
            "Well": "A01",
            "Site": str(site),
        }
        for pattern_list in patterns.values()
        for site, pattern in zip(sites, pattern_list, strict=True)
    }
    source_context = SourcePatternResolutionContext.from_sources(
        parser=SourceSchemaFilenameParser(),
        source_paths_by_virtual_path=source_paths,
        source_metadata_by_path=source_metadata,
    )
    monkeypatch.setattr(
        StepAnchorPatternFilter,
        "source_pattern_context",
        lambda _self: source_context,
    )
    pattern_filter = StepAnchorPatternFilter(
        plan=plan,
        parser=SourceSchemaFilenameParser(),
        output_manifest=None,
        source_workspace_authority=None,
        source_workspace_projection_cache=None,
    )
    return pattern_filter.source_bound_anchor_patterns(patterns)


def test_all_loaded_contract_collapses_sibling_alias_groups_by_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    filtered = _filter_source_anchors(
        monkeypatch,
        artifact_aliases=tuple(alias for alias, _channel in SOURCE_ALIASES),
    )

    assert filtered.groups == {
        "2": ("A01_s001_w2_z001_t001.tif",),
        "1": (),
        "5": (),
        "4": (),
        "3": (),
    }


def test_natural_callable_keeps_each_source_component_anchor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    filtered = _filter_source_anchors(
        monkeypatch,
        artifact_aliases=tuple(alias for alias, _channel in SOURCE_ALIASES),
        composed_image_set=False,
    )

    assert filtered.groups == {
        channel: (f"A01_s001_w{channel}_z001_t001.tif",)
        for _alias, channel in SOURCE_ALIASES
    }


def test_selected_source_contract_keeps_only_its_exact_alias_anchor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    filtered = _filter_source_anchors(
        monkeypatch,
        artifact_aliases=("OrigHoechst",),
    )

    assert filtered.groups == {
        "2": (),
        "1": ("A01_s001_w1_z001_t001.tif",),
        "5": (),
        "4": (),
        "3": (),
    }


def test_distinct_metadata_sets_each_keep_one_source_representative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    filtered = _filter_source_anchors(
        monkeypatch,
        artifact_aliases=tuple(alias for alias, _channel in SOURCE_ALIASES),
        sites=(1, 2),
    )

    assert filtered.groups == {
        "2": (
            "A01_s001_w2_z001_t001.tif",
            "A01_s002_w2_z001_t001.tif",
        ),
        "1": (),
        "5": (),
        "4": (),
        "3": (),
    }


def test_independent_source_aligned_outputs_keep_each_execution_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    filtered = _filter_source_anchors(
        monkeypatch,
        artifact_aliases=("OrigER", "OrigHoechst"),
        shared_output=False,
    )

    assert filtered.groups == {
        "2": ("A01_s001_w2_z001_t001.tif",),
        "1": ("A01_s001_w1_z001_t001.tif",),
        "5": (),
        "4": (),
        "3": (),
    }


def test_missing_alias_in_match_dimension_keeps_existing_strict_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(
        ValueError,
        match="METADATA source binding dimension is missing alias 'OrigER'",
    ):
        _filter_source_anchors(
            monkeypatch,
            artifact_aliases=tuple(alias for alias, _channel in SOURCE_ALIASES),
            dimension_aliases=tuple(
                alias for alias, _channel in SOURCE_ALIASES if alias != "OrigER"
            ),
        )


def test_grouped_branches_project_their_exact_contract_bindings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiled_bindings = _source_bindings()[:2]
    aliases = tuple(binding.alias for binding in compiled_bindings)
    source_binding_plan = CompiledSourceBindingPlan(
        bindings=compiled_bindings,
        match_plan=SourceBindingMatchPlan(
            method=SourceBindingMatchMethod.METADATA,
            dimensions=_match_dimensions(aliases),
        ),
    )

    @artifact_inputs(compiled_bindings[0].input_spec())
    def measure_er(image):
        return image

    @artifact_inputs(compiled_bindings[1].input_spec())
    def measure_hoechst(image):
        return image

    compiled_pattern = compile_function_pattern(
        {"2": measure_er, "1": measure_hoechst},
        {},
        {},
    )
    plan = SimpleNamespace(
        axis_id="A01",
        step_index=0,
        step_name="GroupedMeasurements",
        main_input_dependency=StepInputDependency.pipeline_start(),
        source_binding_plan=source_binding_plan,
        execution_group_scope=ComponentGroupScope.from_raw(
            ("2", "1"),
            component=AllComponents.CHANNEL,
        ),
        compiled_function_pattern=compiled_pattern,
    )
    patterns = PatternGroups(
        {
            "2": ("A01_s001_w2_z001_t001.tif",),
            "1": ("A01_s001_w1_z001_t001.tif",),
            "5": ("A01_s001_w5_z001_t001.tif",),
        }
    )
    source_context = SourcePatternResolutionContext.from_sources(
        parser=SourceSchemaFilenameParser(),
        source_paths_by_virtual_path={
            pattern: f"/source/{pattern}"
            for pattern_list in patterns.values()
            for pattern in pattern_list
        },
        source_metadata_by_path={
            pattern: {"Plate": "Plate", "Well": "A01", "Site": "1"}
            for pattern_list in patterns.values()
            for pattern in pattern_list
        },
    )
    monkeypatch.setattr(
        StepAnchorPatternFilter,
        "source_pattern_context",
        lambda _self: source_context,
    )
    pattern_filter = StepAnchorPatternFilter(
        plan=plan,
        parser=SourceSchemaFilenameParser(),
        output_manifest=None,
        source_workspace_authority=None,
        source_workspace_projection_cache=None,
    )

    filtered = pattern_filter.source_bound_anchor_patterns(patterns)

    assert filtered.groups == {
        "2": ("A01_s001_w2_z001_t001.tif",),
        "1": ("A01_s001_w1_z001_t001.tif",),
        "5": (),
    }
