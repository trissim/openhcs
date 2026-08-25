import ast
import hashlib
import io
import pickle
import subprocess
import sys
from dataclasses import fields as dataclass_fields
from pathlib import Path
from types import SimpleNamespace

import pytest
from objectstate.lazy_factory import ensure_global_config_context
from objectstate.object_state import ObjectState
from objectstate.object_state_registry import ObjectStateRegistry
from polystore.virtual_workspace import SourcePixelRef

from openhcs.constants import Microscope
from openhcs.constants.constants import AllComponents, GroupBy, VariableComponents
from openhcs.constants.input_source import InputSource
from openhcs.core import source_bindings as source_bindings_module
from openhcs.core.artifacts import (
    ArtifactInputPlan,
    ArtifactSpec,
    ArtifactSpecCollection,
    GroupLineageSourceRelation,
    ImageArtifactType,
    ObjectLabelsArtifactType,
)
from openhcs.core.config import (
    GlobalPipelineConfig,
    LazySourceBindingsConfig,
    LazyStepSourceBindingsConfig,
    PipelineConfig,
)
from openhcs.core.pipeline.step_snapshot import StepSnapshot
from openhcs.core.runtime_tabular_values import FieldSpec
from openhcs.core.source_binding_selection import (
    SourceBindingCandidateMatcher,
    SourceBindingMatchedImageSet,
    SourcePatternResolutionContext,
)
from openhcs.core.source_bindings import (
    EMPTY_SOURCE_BINDINGS,
    CompiledSourceBindingPlan,
    CompiledSourceUniversePlan,
    ComponentSelector,
    ImagePlaneSource,
    ImportedMetadataJoin,
    ImportedMetadataTable,
    MetadataExtractionRule,
    MetadataSelector,
    MetadataSource,
    NamedSourceBinding,
    SourceBindingMatchDimension,
    SourceBindingMatchField,
    SourceBindingMatchMethod,
    SourceBindingMatchPlan,
    SourceBindingOrigin,
    SourceBindingRuntimeContext,
    SourceBindingsConfig,
    SourceFilterClause,
    SourceFilterMatchType,
    SourceFilterSubject,
    SourceProjectionRole,
    SourceSelector,
    SourceSetRole,
    StepSourceBindingsConfig,
    source_binding_group_keys_for_group_by,
)
from openhcs.core.source_image_provenance import (
    RuntimeSourceImageProvenancePlane,
    SourceImageIdentity,
    SourceImageProvenance,
    SourceImageProvenanceContributor,
    SourceImageProvenancePlanes,
)
from openhcs.core.source_matching import (
    SourceImageSetComponentRole,
    SourceImageSetIdentityPolicy,
)
from openhcs.core.source_metadata import (
    ORIGINAL_SOURCE_METADATA_FIELD,
    SourceFilterPathMetadata,
    SourceVoxelSpacing,
)
from openhcs.core.source_projection import (
    OpenHCSPlaneAddress,
    SourceArtifactProjection,
    SourcePlaneProjection,
)
from openhcs.core.source_workspace_projection import VirtualWorkspaceSourceProjection
from openhcs.core.steps.function_step import FunctionStep
from openhcs.microscopes.source_schema import SourceSchemaFilenameParser


def test_component_selector_coerces_existing_component_vocabulary():
    selector = ComponentSelector(component=GroupBy.CHANNEL, value=1)

    assert selector.component is AllComponents.CHANNEL
    assert selector.value == "1"

    variable_selector = ComponentSelector(
        component=VariableComponents.SITE,
        value="3",
    )

    assert variable_selector.component is AllComponents.SITE


def test_metadata_selector_preserves_and_uses_declared_scalar_type():
    config = SourceBindingsConfig(
        metadata_fields=(FieldSpec("Dose", float, required=False),),
        bindings=(
            NamedSourceBinding(
                alias="DNA",
                selector=SourceSelector(
                    metadata=(MetadataSelector("Dose", "0.25"),),
                ),
            ),
        ),
    )

    selector = config.bindings[0].selector.metadata[0]
    assert selector.value == 0.25
    assert isinstance(selector.value, float)


def test_source_image_set_plane_members_are_exactly_the_step_stack_axes():
    site_stack = SourceImageSetIdentityPolicy.from_plane_member_fields(
        frozenset((AllComponents.SITE.value,))
    )

    assert (
        site_stack.role(AllComponents.SITE)
        is SourceImageSetComponentRole.IMAGE_PLANE_MEMBER
    )
    assert (
        site_stack.role(AllComponents.CHANNEL)
        is SourceImageSetComponentRole.IMAGE_SET_AXIS
    )


def test_source_image_set_policy_uses_binding_and_source_stack_declarations():
    source_bindings = StepSourceBindingsConfig(
        bindings=(
            NamedSourceBinding(
                alias="DNA",
                component_identity=(ComponentSelector("channel", "1"),),
            ),
        ),
        source_stack_components=(AllComponents.Z_INDEX,),
    )

    policy = SourceImageSetIdentityPolicy.from_source_bindings(source_bindings)

    assert policy.plane_member_components == frozenset(
        (AllComponents.CHANNEL, AllComponents.Z_INDEX)
    )


def test_named_source_binding_normalizes_origin_and_requires_alias():
    binding = NamedSourceBinding(
        alias="OrigBlue",
        origin="pipeline_start",
    )

    assert binding.origin is SourceBindingOrigin.PIPELINE_START
    assert binding.artifact_kind is ImageArtifactType

    objects_binding = NamedSourceBinding(
        alias="Nuclei",
        artifact_kind="object_labels",
        projection_role=SourceProjectionRole.SOURCE_ARTIFACT,
    )

    assert objects_binding.artifact_kind is ObjectLabelsArtifactType

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

    assert all(
        object.__getattribute__(config, item.name) is None
        for item in dataclass_fields(SourceBindingsConfig)
        if item.init
    )


def test_empty_step_source_bindings_remains_concrete_after_config_registration():
    assert EMPTY_SOURCE_BINDINGS.is_empty
    assert EMPTY_SOURCE_BINDINGS.source_voxel_spacing == SourceVoxelSpacing()
    assert EMPTY_SOURCE_BINDINGS.metadata_rules == ()
    assert EMPTY_SOURCE_BINDINGS.source_filters == ()
    assert EMPTY_SOURCE_BINDINGS.bindings == ()
    assert EMPTY_SOURCE_BINDINGS.image_plane_sources == ()
    assert EMPTY_SOURCE_BINDINGS.imported_metadata_tables == ()
    assert EMPTY_SOURCE_BINDINGS.source_stack_components == ()
    assert EMPTY_SOURCE_BINDINGS.grouping_metadata_fields == ()


def test_source_locations_resolve_local_and_http_sources_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    local_image = tmp_path / "local.tif"
    local_image.write_bytes(b"local")
    metadata_table = tmp_path / "metadata.csv"
    metadata_table.write_text("Well\nA01\n", encoding="utf-8")
    remote_payload = b"remote-image"
    remote_uri = "https://example.invalid/images/remote.TIF"
    requests: list[str] = []

    def urlopen(request, timeout):
        requests.append(request.full_url)
        assert timeout == 60
        return io.BytesIO(remote_payload)

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr(source_bindings_module.urllib.request, "urlopen", urlopen)
    config = SourceBindingsConfig(
        image_plane_sources=(
            ImagePlaneSource(uri="local.tif"),
            ImagePlaneSource(uri=remote_uri),
        ),
        imported_metadata_tables=(
            ImportedMetadataTable(location=metadata_table.as_uri()),
        ),
    )

    resolved = config.resolved_source_locations(tmp_path)

    assert resolved.image_plane_sources[0].uri == str(local_image)
    remote_path = Path(resolved.image_plane_sources[1].uri)
    assert remote_path.name == f"{hashlib.sha256(remote_payload).hexdigest()}.TIF"
    assert remote_path.read_bytes() == remote_payload
    assert resolved.imported_metadata_tables[0].location == str(metadata_table)
    assert requests == [remote_uri]

    monkeypatch.setattr(
        source_bindings_module.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: pytest.fail(
            "resolved HTTP source was downloaded twice"
        ),
    )
    assert config.resolved_source_locations(tmp_path) == resolved


def test_source_location_rejects_unsupported_uri_scheme(tmp_path: Path):
    config = SourceBindingsConfig(
        image_plane_sources=(ImagePlaneSource(uri="s3://bucket/image.tif"),),
    )

    with pytest.raises(ValueError, match="Unsupported source URI scheme 's3'"):
        config.resolved_source_locations(tmp_path)


def test_imported_metadata_resolves_bare_name_from_explicit_portable_root(
    tmp_path: Path,
):
    source_root = tmp_path / "images"
    portable_root = tmp_path / "pipeline"
    source_root.mkdir()
    portable_root.mkdir()
    metadata_table = portable_root / "plate.csv"
    metadata_table.write_text("Well\nA01\n", encoding="utf-8")

    resolved = ImportedMetadataTable(location="plate.csv").resolved(
        source_root,
        portable_roots=(portable_root,),
    )

    assert resolved.location == str(metadata_table)


def test_imported_metadata_prefers_primary_root_over_portable_roots(tmp_path: Path):
    source_root = tmp_path / "images"
    portable_root = tmp_path / "pipeline"
    source_root.mkdir()
    portable_root.mkdir()
    primary_table = source_root / "plate.csv"
    primary_table.write_text("Well\nA01\n", encoding="utf-8")
    (portable_root / "plate.csv").write_text("Well\nB01\n", encoding="utf-8")

    resolved = ImportedMetadataTable(location="plate.csv").resolved(
        source_root,
        portable_roots=(portable_root,),
    )

    assert resolved.location == str(primary_table)


def test_imported_metadata_rejects_ambiguous_portable_roots(tmp_path: Path):
    source_root = tmp_path / "images"
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    source_root.mkdir()
    first_root.mkdir()
    second_root.mkdir()
    (first_root / "plate.csv").write_text("Well\nA01\n", encoding="utf-8")
    (second_root / "plate.csv").write_text("Well\nB01\n", encoding="utf-8")

    with pytest.raises(ValueError, match="matches multiple portable roots"):
        ImportedMetadataTable(location="plate.csv").resolved(
            source_root,
            portable_roots=(first_root, second_root),
        )


def test_production_code_has_no_unresolved_empty_step_source_config_shortcuts():
    root = Path(__file__).parents[2] / "openhcs"
    violations: list[tuple[str, int]] = []
    for path in root.rglob("*.py"):
        if path == root / "core" / "source_bindings.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        violations.extend(
            (str(path.relative_to(root)), node.lineno)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "StepSourceBindingsConfig"
            and not node.args
            and not node.keywords
        )

    assert violations == []


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
    metadata_fields = (FieldSpec("Dose", float, required=False),)
    ensure_global_config_context(GlobalPipelineConfig, GlobalPipelineConfig())
    pipeline_state = ObjectState(
        PipelineConfig(
            source_bindings_config=LazySourceBindingsConfig(
                bindings=(binding,),
                source_stack_components=(AllComponents.Z_INDEX,),
                grouping_metadata_fields=("Plate",),
                metadata_fields=metadata_fields,
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
        snapshot = StepSnapshot(
            index=0,
            scope_id=step_state.scope_id,
            step=step_state.to_saved_resolved_object(),
        )
    finally:
        ObjectStateRegistry.clear()

    assert snapshot.step.source_bindings.bindings == (binding,)
    assert snapshot.step.source_bindings.source_stack_components == (
        AllComponents.Z_INDEX,
    )
    assert snapshot.step.source_bindings.grouping_metadata_fields == ("Plate",)
    assert snapshot.step.source_bindings.metadata_fields == metadata_fields
    compiled = CompiledSourceBindingPlan.from_config(
        snapshot.step.source_bindings,
    )
    assert compiled.bindings == (binding,)
    assert compiled.source_stack_components == (AllComponents.Z_INDEX,)
    assert compiled.metadata_fields == metadata_fields
    assert compiled.has_primary_content
    assert pickle.loads(pickle.dumps(compiled)) == compiled


def test_compiled_source_bindings_include_realized_original_metadata_schema():
    config = StepSourceBindingsConfig(
        metadata_fields=(FieldSpec("Dose", float, required=False),),
        imported_metadata_tables=(
            ImportedMetadataTable(
                joins=(ImportedMetadataJoin("Well", "CSVWell"),),
            ),
        ),
    )

    compiled = CompiledSourceBindingPlan.from_config(
        config,
        realized_source_metadata=(
            {
                ORIGINAL_SOURCE_METADATA_FIELD: {
                    "Dose": "0.25",
                    "Compound": "DMSO",
                    "Replicate": 1,
                    "CSVWell": "A01",
                }
            },
            {
                ORIGINAL_SOURCE_METADATA_FIELD: {
                    "Dose": "0.50",
                    "Compound": "Drug",
                    "Replicate": 2,
                    "CSVWell": "A02",
                }
            },
        ),
    )

    assert compiled.metadata_fields == (
        FieldSpec("Dose", float, required=False),
        FieldSpec("Compound", str, required=False),
        FieldSpec("Replicate", int, required=False),
    )


def test_enabled_step_source_bindings_compile_inherited_bindings():
    ObjectStateRegistry.clear()
    binding = NamedSourceBinding(alias="DNA")
    ensure_global_config_context(GlobalPipelineConfig, GlobalPipelineConfig())
    pipeline_state = ObjectState(
        PipelineConfig(
            source_bindings_config=LazySourceBindingsConfig(bindings=(binding,)),
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
        snapshot = StepSnapshot(
            index=0,
            scope_id=step_state.scope_id,
            step=step_state.to_saved_resolved_object(),
        )
    finally:
        ObjectStateRegistry.clear()

    assert snapshot.step.source_bindings.bindings == (binding,)
    assert CompiledSourceBindingPlan.from_config(
        snapshot.step.source_bindings,
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
        snapshot = StepSnapshot(
            index=0,
            scope_id=step_state.scope_id,
            step=step_state.to_saved_resolved_object(),
        )
    finally:
        ObjectStateRegistry.clear()

    assert snapshot.step.source_bindings.enabled is True
    assert snapshot.step.source_bindings.bindings == (binding,)
    assert CompiledSourceBindingPlan.from_config(
        snapshot.step.source_bindings,
    ).bindings == (binding,)


def test_pipeline_start_binding_groups_do_not_mutate_resolved_enabled_state():
    binding = NamedSourceBinding(
        alias="DNA",
        component_identity=(
            ComponentSelector(component=AllComponents.CHANNEL, value="1"),
        ),
    )
    source_bindings = StepSourceBindingsConfig(
        enabled=False,
        bindings=(binding,),
    )

    compiled = CompiledSourceBindingPlan.from_config(
        source_bindings,
    )

    assert compiled.bindings == (binding,)
    assert source_bindings.enabled is False
    assert source_binding_group_keys_for_group_by(
        source_bindings,
        GroupBy.CHANNEL,
    ) == ("1",)
    assert (
        source_bindings.for_input_source(InputSource.PIPELINE_START) is source_bindings
    )
    assert source_bindings.for_input_source(InputSource.PREVIOUS_STEP).is_empty


def test_source_lineage_requires_its_declared_cursor_artifact() -> None:
    source = ArtifactSpec.input("DNA", ImageArtifactType)
    derived = ArtifactSpec.output(
        "CorrectedDNA",
        ImageArtifactType,
        relations=(GroupLineageSourceRelation(source.ref()),),
    )
    source_bindings = StepSourceBindingsConfig(
        enabled=True,
        bindings=(
            NamedSourceBinding(
                alias="DNA",
                component_identity=(
                    ComponentSelector(
                        component=AllComponents.CHANNEL,
                        value="1",
                    ),
                ),
            ),
        ),
    )

    with pytest.raises(ValueError, match="lineage.*unavailable source"):
        source_bindings.component_group_keys_for_artifact_specs(
            AllComponents.CHANNEL,
            (derived,),
            ArtifactSpecCollection(()),
        )


def test_source_lineage_resolves_consumer_view_to_active_artifact_binding() -> None:
    source = ArtifactSpec.input("DNA", ImageArtifactType)
    derived = ArtifactSpec.output(
        "CorrectedDNA",
        ImageArtifactType,
        relations=(GroupLineageSourceRelation(source.ref()),),
    )
    source_bindings = StepSourceBindingsConfig(
        enabled=True,
        bindings=(
            NamedSourceBinding(
                alias="DNA",
                component_identity=(
                    ComponentSelector(
                        component=AllComponents.CHANNEL,
                        value="1",
                    ),
                ),
            ),
        ),
    )

    assert source_bindings.component_group_keys_for_artifact_specs(
        AllComponents.CHANNEL,
        (derived.for_plan_type(ArtifactInputPlan),),
        ArtifactSpecCollection((source, derived)),
    ) == ("1",)


def test_source_lineage_projection_preserves_declaration_order() -> None:
    blue = NamedSourceBinding(alias="Blue")
    green = NamedSourceBinding(alias="Green")
    source_bindings = StepSourceBindingsConfig(
        enabled=True,
        bindings=(blue, green),
    )
    available = ArtifactSpecCollection((blue.input_spec(), green.input_spec()))

    projected = source_bindings.bindings_for_artifact_specs(
        (green.input_spec(), blue.input_spec()),
        available,
    )

    assert projected == (blue, green)


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


def test_component_identity_owns_realized_source_group_values() -> None:
    binding = NamedSourceBinding(
        alias="DNA",
        selector=SourceSelector(
            components=(ComponentSelector(AllComponents.CHANNEL, "1"),),
        ),
        component_identity=(ComponentSelector(AllComponents.CHANNEL, "DNA"),),
    )

    assert binding.component_values(
        AllComponents.CHANNEL,
        realized_source_metadata=(
            {"channel": 1},
            {"channel": 2},
        ),
    ) == ("DNA",)


def test_realized_component_values_are_scoped_by_source_selector() -> None:
    binding = NamedSourceBinding(
        alias="DNA",
        selector=SourceSelector(
            components=(ComponentSelector(AllComponents.CHANNEL, "1"),),
        ),
    )

    assert binding.component_values(
        AllComponents.SITE,
        realized_source_metadata=(
            {"channel": 1, "site": 3},
            {"channel": 2, "site": 7},
        ),
    ) == ("3",)


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

    plan = CompiledSourceBindingPlan.from_config(
        config,
    )

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
        ),
    )

    universe_plan = CompiledSourceUniversePlan.from_source_binding_plan(binding_plan)

    assert universe_plan.uses_pipeline_start_binding_origin
    assert not universe_plan.requires_full_pipeline_source_universe


def test_matched_image_set_without_match_plan_uses_selector_compatible_sources():
    context = SourcePatternResolutionContext.from_sources(
        parser=SourceSchemaFilenameParser(),
        source_paths_by_virtual_path={},
    )
    matched_set = SourceBindingMatchedImageSet.from_plan(
        bindings=(
            NamedSourceBinding(
                alias="DNA",
                selector=SourceSelector(
                    filters=(
                        SourceFilterClause(
                            subject=SourceFilterSubject.FILE,
                            match_type=SourceFilterMatchType.CONTAINS,
                            value="w1",
                        ),
                    )
                ),
                origin=SourceBindingOrigin.PIPELINE_START,
            ),
            NamedSourceBinding(
                alias="PH3",
                selector=SourceSelector(
                    filters=(
                        SourceFilterClause(
                            subject=SourceFilterSubject.FILE,
                            match_type=SourceFilterMatchType.CONTAINS,
                            value="w2",
                        ),
                    )
                ),
                origin=SourceBindingOrigin.PIPELINE_START,
            ),
        ),
        match_plan=None,
        source_context=context,
        identity_policy=SourceImageSetIdentityPolicy(),
    )

    assert matched_set.expand(
        ("A01_s001_w1_z001_t001.tif",),
        source_universe=(
            "A01_s001_w1_z001_t001.tif",
            "A01_s001_w2_z001_t001.tif",
            "A01_s001_w3_z001_t001.tif",
        ),
    ) == ("A01_s001_w1_z001_t001.tif",)


def test_alias_only_step_bindings_filter_and_order_workspace_projections() -> None:
    declarations = (
        NamedSourceBinding(alias="Hoechst"),
        NamedSourceBinding(alias="MAP2"),
        NamedSourceBinding(alias="SMI312"),
    )
    virtual_paths = tuple(
        f"R04C09_s011_w{channel}_z001_t001.tif" for channel in ("1", "2", "4")
    )
    source_projections = {
        virtual_path: SourcePlaneProjection(
            address=OpenHCSPlaneAddress.from_values(
                well="R04C09",
                site="11",
                channel=channel,
                z_index="1",
                timepoint="1",
            ),
            ref=SourcePixelRef("disk", f"/source/ch{channel}.tiff"),
            source_alias=binding.alias,
        )
        for virtual_path, channel, binding in zip(
            virtual_paths,
            ("1", "2", "4"),
            declarations,
            strict=True,
        )
    }
    projection = VirtualWorkspaceSourceProjection(
        source_refs_by_virtual_path={
            path: source_projection.ref
            for path, source_projection in source_projections.items()
        },
        source_metadata_by_path={},
        source_projections_by_virtual_path=source_projections,
    )
    context = SourcePatternResolutionContext.from_projection(
        parser=SourceSchemaFilenameParser(),
        projection=projection,
    )
    matched_set = SourceBindingMatchedImageSet.from_plan(
        bindings=(
            NamedSourceBinding(alias="SMI312"),
            NamedSourceBinding(alias="Hoechst"),
        ),
        match_plan=None,
        source_context=context,
        identity_policy=SourceImageSetIdentityPolicy(),
    )

    assert matched_set.expand(
        virtual_paths,
        source_universe=virtual_paths,
    ) == (virtual_paths[2], virtual_paths[0])


def test_source_set_without_matched_bindings_preserves_anchor_candidates() -> None:
    context = SourcePatternResolutionContext.from_sources(
        parser=SourceSchemaFilenameParser(),
        source_paths_by_virtual_path={},
    )
    matched_set = SourceBindingMatchedImageSet.from_plan(
        bindings=(
            NamedSourceBinding(
                alias="PlateTemplate",
                source_set_role=SourceSetRole.BROADCAST,
            ),
        ),
        match_plan=None,
        source_context=context,
        identity_policy=SourceImageSetIdentityPolicy(),
    )
    anchors = (
        "A01_s001_w1_z001_t001.tif",
        "A01_s002_w1_z001_t001.tif",
    )

    assert matched_set.expand(anchors, source_universe=anchors) == anchors


def test_narrow_step_binding_uses_exact_workspace_provenance_identity():
    bindings = tuple(
        NamedSourceBinding(
            alias=alias,
            selector=SourceSelector(
                metadata=(MetadataSelector("ChannelNumber", channel),),
            ),
            component_identity=(ComponentSelector(AllComponents.CHANNEL, channel),),
        )
        for alias, channel in (("DNA", "2"), ("Mito", "1"), ("Membrane", "0"))
    )
    virtual_paths = tuple(
        f"A01_s001_w{channel}_z00{z_index}_t001.tif"
        for channel in ("2", "1", "0")
        for z_index in (1, 2, 3)
    )
    source_paths = {
        virtual_path: f"/source/channel_{virtual_path.split('_w', 1)[1][0]}.tif"
        for virtual_path in virtual_paths
    }
    source_metadata = {
        virtual_path: {
            "well": "A01",
            "site": "1",
            "channel": virtual_path.split("_w", 1)[1][0],
            "ChannelNumber": virtual_path.split("_w", 1)[1][0],
            "z_index": int(virtual_path.split("_z", 1)[1].split("_", 1)[0]),
            "timepoint": "1",
        }
        for virtual_path in virtual_paths
    }
    projection = VirtualWorkspaceSourceProjection(
        source_refs_by_virtual_path={
            virtual_path: SourcePixelRef(
                "disk",
                source_path,
                source_axis_indices=(
                    int(virtual_path.split("_z", 1)[1].split("_", 1)[0]) - 1,
                ),
            )
            for virtual_path, source_path in source_paths.items()
        },
        source_metadata_by_path=source_metadata,
    )
    context = SourcePatternResolutionContext.from_projection(
        parser=SourceSchemaFilenameParser(),
        projection=projection,
    )
    matched_set = SourceBindingMatchedImageSet.from_plan(
        bindings=(bindings[2],),
        match_plan=SourceBindingMatchPlan(SourceBindingMatchMethod.ORDER),
        source_context=context,
        identity_policy=SourceImageSetIdentityPolicy.from_source_bindings(
            SourceBindingsConfig(bindings=bindings)
        ),
    )

    assert matched_set.members_for_binding(
        bindings[2],
        anchor_provenance=SourceImageProvenance(
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=tuple(
                    source_paths[f"A01_s001_w2_z00{z_index}_t001.tif"]
                    for z_index in (1, 2, 3)
                ),
                component_metadata=tuple(
                    source_metadata[f"A01_s001_w2_z00{z_index}_t001.tif"]
                    for z_index in (1, 2, 3)
                ),
            )
        ),
        source_universe=virtual_paths,
    ) == tuple(f"A01_s001_w0_z00{z_index}_t001.tif" for z_index in (1, 2, 3))

    assert matched_set.members_for_binding(
        bindings[2],
        anchor_provenance=SourceImageProvenance(
            source_image_provenance_planes=SourceImageProvenancePlanes(
                (
                    RuntimeSourceImageProvenancePlane(
                        contributors=tuple(
                            SourceImageProvenanceContributor(
                                SourceImageIdentity(
                                    source_paths[f"A01_s001_w2_z00{z_index}_t001.tif"],
                                    source_metadata[
                                        f"A01_s001_w2_z00{z_index}_t001.tif"
                                    ],
                                ),
                                source_image_name="DNA",
                            )
                            for z_index in (1, 2, 3)
                        )
                    ),
                )
            )
        ),
        source_universe=virtual_paths,
    ) == tuple(f"A01_s001_w0_z00{z_index}_t001.tif" for z_index in (1, 2, 3))

    with pytest.raises(ValueError, match="one exact declared source-set position"):
        matched_set.members_for_binding(
            bindings[2],
            anchor_provenance=SourceImageProvenance(
                source_path=source_paths["A01_s001_w2_z001_t001.tif"]
            ),
            source_universe=virtual_paths,
        )

    for unresolved_provenance in (
        SourceImageProvenance(),
        SourceImageProvenance(source_image_names=("DNA",)),
    ):
        with pytest.raises(
            ValueError,
            match="requires addressable runtime provenance",
        ):
            matched_set.members_for_binding(
                bindings[2],
                anchor_provenance=unresolved_provenance,
                source_universe=virtual_paths,
            )


def test_source_binding_members_load_one_store_for_multiple_matching_identities():
    binding = NamedSourceBinding(
        alias="Membrane",
        selector=SourceSelector(
            metadata=(MetadataSelector("ChannelNumber", "0"),),
        ),
        component_identity=(ComponentSelector(AllComponents.CHANNEL, "0"),),
    )
    virtual_path = "A01_s001_w0_z001_t001.tif"
    source_path = "/source/channel_0.tif"
    plane_metadata = {
        "well": "A01",
        "site": "1",
        "channel": "0",
        "ChannelNumber": "0",
        "z_index": "1",
        "timepoint": "1",
    }
    store_metadata = {
        "well": "A01",
        "channel": "0",
        "ChannelNumber": "0",
    }
    projection = VirtualWorkspaceSourceProjection(
        source_refs_by_virtual_path={
            virtual_path: SourcePixelRef(
                "disk",
                source_path,
                source_axis_indices=(0,),
            ),
        },
        source_metadata_by_path={
            virtual_path: plane_metadata,
            source_path: store_metadata,
        },
    )
    matched_set = SourceBindingMatchedImageSet.from_plan(
        bindings=(binding,),
        match_plan=SourceBindingMatchPlan(SourceBindingMatchMethod.ORDER),
        source_context=SourcePatternResolutionContext.from_projection(
            parser=SourceSchemaFilenameParser(),
            projection=projection,
        ),
        identity_policy=SourceImageSetIdentityPolicy.from_source_bindings(
            SourceBindingsConfig(bindings=(binding,))
        ),
    )
    provenance = SourceImageProvenance(
        source_image_provenance_planes=SourceImageProvenancePlanes(
            (
                RuntimeSourceImageProvenancePlane(
                    SourceImageIdentity(source_path, plane_metadata),
                    contributors=(
                        SourceImageProvenanceContributor(
                            SourceImageIdentity(source_path, store_metadata),
                            source_image_name="Membrane",
                        ),
                    ),
                    source_image_name="DerivedMembrane",
                ),
            )
        )
    )

    assert matched_set.members_for_binding(
        binding,
        anchor_provenance=provenance,
        source_universe=(virtual_path,),
    ) == (virtual_path,)


def test_virtual_workspace_projection_filters_source_metadata_by_axis():
    projection = VirtualWorkspaceSourceProjection(
        source_refs_by_virtual_path={
            "A01_s001_w1_z001_t001.tif": SourcePixelRef("disk", "/source/A01_w1.tif"),
            "A02_s001_w1_z001_t001.tif": SourcePixelRef("disk", "/source/A02_w1.tif"),
        },
        source_metadata_by_path={
            "A01_s001_w1_z001_t001.tif": {"well": "A01", "channel": "1"},
            "/source/A01_w1.tif": {"well": "A01", "channel": "1"},
            "A02_s001_w1_z001_t001.tif": {"well": "A02", "channel": "1"},
            "/source/A02_w1.tif": {"well": "A02", "channel": "1"},
        },
    )

    filtered = projection.filtered_by_axis(axis_id="A01")

    assert tuple(filtered.source_refs_by_virtual_path) == ("A01_s001_w1_z001_t001.tif",)
    assert filtered.source_metadata_by_path["/source/A01_w1.tif"]["well"] == "A01"
    assert "/source/A02_w1.tif" not in filtered.source_metadata_by_path


def test_virtual_workspace_source_matching_uses_declared_filter_identity():
    virtual_path = "A01_s001_w1_z001_t001.tif"
    source_path = "/source/0_1_N_R.png"
    source_metadata: dict[str, object] = {"channel": "1"}
    SourceFilterPathMetadata.from_paths((source_path,)).merge_into(
        source_metadata,
        path=virtual_path,
    )
    projection = VirtualWorkspaceSourceProjection(
        source_refs_by_virtual_path={
            virtual_path: SourcePixelRef(
                "opaque_backend",
                '{"opaque":"address-without-source-name"}',
            ),
        },
        source_metadata_by_path={virtual_path: source_metadata},
    )
    context = SourcePatternResolutionContext.from_projection(
        parser=SimpleNamespace(parse_filename=lambda _path: None),
        projection=projection,
    )
    binding = NamedSourceBinding(
        alias="OrigStain1",
        selector=SourceSelector(
            filters=(
                SourceFilterClause(
                    subject=SourceFilterSubject.FILE,
                    match_type=SourceFilterMatchType.CONTAINS,
                    value="N_R",
                ),
            ),
        ),
    )

    assert context.candidate_filter_paths(virtual_path) == (source_path,)
    assert SourceBindingCandidateMatcher.matches(
        virtual_path,
        binding=binding,
        source_context=context,
    )


def test_virtual_workspace_source_matching_requires_exact_projection_binding():
    address = OpenHCSPlaneAddress.from_values(
        well="A01",
        site="1",
        channel="1",
        z_index="1",
        timepoint="1",
    )
    original_path = "A01_s001_w1_z001_t001.tif"
    illumination_path = f"_source/Illumination/{original_path}"
    original = SourcePlaneProjection(
        address=address,
        ref=SourcePixelRef("disk", "/source/original.tif"),
        source_alias="Original",
    )
    illumination = SourceArtifactProjection(
        address=address,
        ref=SourcePixelRef("disk", "/source/illumination.npy"),
        source_alias="Illumination",
        artifact_kind=ImageArtifactType,
    )
    projection = VirtualWorkspaceSourceProjection(
        source_refs_by_virtual_path={
            original_path: original.ref,
            illumination_path: illumination.ref,
        },
        source_metadata_by_path={
            original_path: {"channel": "1"},
            illumination_path: {"channel": "1"},
        },
        source_projections_by_virtual_path={
            original_path: original,
            illumination_path: illumination,
        },
    )
    context = SourcePatternResolutionContext.from_projection(
        parser=SourceSchemaFilenameParser(),
        projection=projection,
    )
    original_binding = NamedSourceBinding(alias="Original")

    assert SourceBindingCandidateMatcher.compatible_candidates(
        (original_path, illumination_path),
        bindings=(original_binding,),
        source_context=context,
    ) == (original_path,)


def test_rule_metadata_preserves_original_source_metadata_mapping():
    path = "monolayer_1.tif"
    context = SourcePatternResolutionContext.from_sources(
        parser=SimpleNamespace(parse_filename=lambda _path: None),
        source_paths_by_virtual_path={},
        metadata_rules=(
            MetadataExtractionRule(
                source=MetadataSource.FILE_NAME,
                pattern=r"(?P<Plate>[^_]+)_(?P<Site>\d+)",
            ),
        ),
    )

    metadata = context.metadata_for_path(path)

    assert metadata is not None
    assert metadata[ORIGINAL_SOURCE_METADATA_FIELD] == {
        "Plate": "monolayer",
        "Site": "1",
    }


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
        ),
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
            "A01_s001_w1_z001_t001.tif": {
                "Compound": "DMSO",
                ORIGINAL_SOURCE_METADATA_FIELD: {
                    "Plate": "PlateA",
                    "ChannelNumber": "1",
                },
            },
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
        ORIGINAL_SOURCE_METADATA_FIELD: {
            "Plate": "PlateA",
            "ChannelNumber": "1",
        },
    }
