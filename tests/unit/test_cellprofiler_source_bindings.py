"""CellProfiler setup lowering through generic source-binding authorities."""

from __future__ import annotations

from dataclasses import fields
import inspect
from pathlib import Path
import subprocess
import sys
import warnings

import numpy as np
import pytest

import openhcs.core.source_bindings as source_bindings_module
from openhcs.constants.constants import AllComponents
from openhcs.core.artifacts import ImageArtifactType, ObjectLabelsArtifactType
from openhcs.core.config import (
    LazySourceBindingsConfig,
    LazyStepSourceBindingsConfig,
    PipelineConfig,
)
from openhcs.core.source_metadata import SourceVoxelSpacing
from openhcs.core.source_image_semantics import apply_source_binding_payload
from openhcs.core.runtime_image_loading import ImagePayloadSourceMetadataContext
from openhcs.core.runtime_image_values import (
    image_payload_data,
    image_payload_metadata,
)
from openhcs.core.runtime_tabular_values import FieldSpec
from openhcs.core.source_image_provenance import SourceImageIdentity
from openhcs.core.source_matching import source_filters_match
from openhcs.core.source_bindings import (
    ComponentSelector,
    ImagePlaneSource,
    MetadataSource,
    NamedSourceBinding,
    SourceBindingDeclarationsMixin,
    SourceBindingMatchMethod,
    SourceBindingOrigin,
    SourceBindingsConfig,
    SourceFilterClause,
    SourceFilterMatchType,
    SourceFilterSubject,
    SourceProjectionRole,
    SourceSetRole,
    StepSourceBindingsConfig,
)
from openhcs.interop.cellprofiler.module_declarations import CellProfilerModule
from openhcs.interop.cellprofiler.parser import (
    CPPipeParser,
    ModuleBlock,
    ModuleSetting,
)


def _field_names(record_type: type[object]) -> frozenset[str]:
    return frozenset(field.name for field in fields(record_type))


def _module(
    module_num: int,
    name: str,
    records: tuple[tuple[str, str], ...],
    *,
    enabled: bool = True,
) -> ModuleBlock:
    setting_records = [ModuleSetting(key, value) for key, value in records]
    return ModuleBlock(
        name=name,
        module_num=module_num,
        enabled=enabled,
        setting_records=setting_records,
    )


def _fold_setup_modules(*modules: ModuleBlock) -> SourceBindingsConfig:
    config = SourceBindingsConfig()
    for module in modules:
        if not module.enabled:
            continue
        module_type = CellProfilerModule.require_module(module.name)
        assert module_type.emits_function_step() is False
        config = module_type.contribute_source_bindings(module, config)
        assert isinstance(config, SourceBindingsConfig)
    return config


def test_named_source_binding_contains_generic_payload_loading_declaration() -> None:
    with pytest.raises(ValueError, match="channel axis"):
        NamedSourceBinding(
            alias="Color",
            source_channel_counts=frozenset({3}),
        )


def test_setup_modules_share_cellprofiler_module_registry() -> None:
    for module_name in ("Images", "LoadImages", "Metadata", "NamesAndTypes", "Groups"):
        module_type = CellProfilerModule.require_module(module_name)
        assert module_type.emits_function_step() is False
        assert callable(module_type.contribute_source_bindings)


def test_source_binding_queries_live_on_shared_declaration_mixin() -> None:
    for method_name in (
        "binding_for_alias",
        "binding_for_artifact_ref",
        "declares_artifact_ref",
        "bindings_for_artifact_refs",
        "for_artifact_refs",
        "primary_plane_aliases",
        "measurement_source_names",
        "bindings_for_component_group",
    ):
        assert method_name in vars(SourceBindingDeclarationsMixin)


def test_component_group_binding_scope_filters_its_axis_and_broadcasts_others() -> None:
    config = SourceBindingsConfig(
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
            NamedSourceBinding(
                alias="RNA",
                component_identity=(
                    ComponentSelector(
                        component=AllComponents.CHANNEL,
                        value="2",
                    ),
                ),
            ),
        )
    )

    assert tuple(
        binding.alias
        for binding in config.bindings_for_component_group(
            AllComponents.CHANNEL,
            "2",
        )
    ) == ("RNA",)
    assert tuple(
        binding.alias
        for binding in config.bindings_for_component_group(
            AllComponents.SITE,
            "1",
        )
    ) == ("DNA", "RNA")
    with pytest.raises(ValueError, match="channel|3"):
        config.bindings_for_component_group(AllComponents.CHANNEL, "3")


def test_lazy_source_configs_cover_every_inherited_dataclass_field() -> None:
    source_fields = _field_names(SourceBindingsConfig)

    assert source_fields <= _field_names(LazySourceBindingsConfig)
    assert source_fields <= _field_names(LazyStepSourceBindingsConfig)
    assert source_fields <= _field_names(StepSourceBindingsConfig)


def test_compiled_source_binding_plan_accepts_input_source() -> None:
    assert (
        "input_source"
        in inspect.signature(
            source_bindings_module.CompiledSourceBindingPlan.from_config
        ).parameters
    )


def test_complete_lazy_setup_facts_pycodify_and_reconstruct_in_fresh_process() -> None:
    from pycodify import Assignment, generate_python_source

    import openhcs.serialization.pycodify_formatters  # noqa: F401

    image_plane_source_type = getattr(
        source_bindings_module,
        "ImagePlaneSource",
        None,
    )
    imported_join_type = getattr(
        source_bindings_module,
        "ImportedMetadataJoin",
        None,
    )
    imported_table_type = getattr(
        source_bindings_module,
        "ImportedMetadataTable",
        None,
    )
    assert image_plane_source_type is not None
    assert imported_join_type is not None
    assert imported_table_type is not None

    config = PipelineConfig(
        source_bindings_config=LazySourceBindingsConfig(
            bindings=(
                NamedSourceBinding(
                    alias="Mask",
                    artifact_kind=ObjectLabelsArtifactType,
                    required=False,
                    projection_role=SourceProjectionRole.SOURCE_ARTIFACT,
                    load_as_mask=True,
                    source_channel_axis=-1,
                    source_channel_counts=frozenset({3}),
                ),
            ),
            image_plane_sources=(
                image_plane_source_type(
                    uri="/data/explicit.npy",
                    series="0",
                    index="2",
                    channel="1",
                ),
            ),
            imported_metadata_tables=(
                imported_table_type(
                    location="/data/metadata.csv",
                    joins=(
                        imported_join_type(
                            image_metadata_field="Well",
                            imported_metadata_field="WellID",
                        ),
                    ),
                ),
            ),
            metadata_fields=(
                FieldSpec("Dose", float, required=False),
                FieldSpec("Frame", int, required=False),
            ),
            source_stack_components=(AllComponents.TIMEPOINT,),
            grouping_metadata_fields=("Plate", "Well"),
            source_voxel_spacing=SourceVoxelSpacing((2.0, 1.0, 1.0)),
        )
    )
    source = generate_python_source(
        Assignment("pipeline_config", config),
        clean_mode=True,
    )
    script = "\n".join(
        (
            source,
            "from openhcs.constants.constants import AllComponents",
            "from openhcs.core.artifacts import ObjectLabelsArtifactType",
            "cfg = pipeline_config.source_bindings_config",
            "assert cfg.bindings[0].artifact_kind is ObjectLabelsArtifactType",
            "assert cfg.bindings[0].load_as_mask is True",
            "assert cfg.bindings[0].source_channel_axis == -1",
            "assert cfg.bindings[0].source_channel_counts == frozenset({3})",
            "assert cfg.image_plane_sources[0].uri == '/data/explicit.npy'",
            "assert cfg.imported_metadata_tables[0].joins[0].imported_metadata_field == 'WellID'",
            "assert cfg.metadata_fields[0].dtype is float",
            "assert cfg.metadata_fields[1].dtype is int",
            "assert cfg.source_stack_components == (AllComponents.TIMEPOINT,)",
            "assert cfg.grouping_metadata_fields == ('Plate', 'Well')",
            "assert cfg.source_voxel_spacing.values_zyx == (2.0, 1.0, 1.0)",
        )
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_cppipe_parser_preserves_repeated_settings_in_source_order(
    tmp_path: Path,
) -> None:
    cppipe_path = tmp_path / "repeated.cppipe"
    cppipe_path.write_text(
        "\n".join(
            (
                "CellProfiler Pipeline: http://www.cellprofiler.org",
                "Version:5",
                "",
                "NamesAndTypes:[module_num:3|enabled:True]",
                "    Assignments count:2",
                '    Select the rule criteria:and (metadata does channel "1")',
                "    Name to assign these images:DAPI",
                '    Select the rule criteria:and (metadata does channel "2")',
                "    Name to assign these images:Actin",
                "",
            )
        ),
        encoding="utf-8",
    )

    (module,) = CPPipeParser().parse(cppipe_path)

    assert module.settings["Name to assign these images"] == "Actin"
    assert module.get_setting_values("Name to assign these images") == (
        "DAPI",
        "Actin",
    )
    assert tuple(
        record.value for record in module.iter_settings("Select the rule criteria")
    ) == (
        'and (metadata does channel "1")',
        'and (metadata does channel "2")',
    )


def test_cppipe_parser_preserves_embedded_image_plane_rows(tmp_path: Path) -> None:
    cppipe_path = tmp_path / "planes.cppipe"
    cppipe_path.write_text(
        "\n".join(
            (
                "CellProfiler Pipeline: http://www.cellprofiler.org",
                "Version:3",
                "HasImagePlaneDetails:True",
                "",
                "Images:[module_num:1|enabled:True]",
                "    Filter images?:Images only",
                "",
                '"Version":"1","PlaneCount":"2"',
                '"URL","Series","Index","Channel"',
                '"https://example.invalid/A_D.TIF",,,',
                '"file:/tmp/A_F.TIF","0","1","2"',
            )
        ),
        encoding="utf-8",
    )
    parser = CPPipeParser()

    (module,) = parser.parse(cppipe_path)

    assert parser.image_plane_sources == (
        ImagePlaneSource(uri="https://example.invalid/A_D.TIF"),
        ImagePlaneSource(
            uri="file:/tmp/A_F.TIF",
            series="0",
            index="1",
            channel="2",
        ),
    )
    assert "image_plane_sources" not in module.metadata


def test_images_setup_preserves_pipeline_owned_embedded_planes() -> None:
    module = _module(
        1,
        "Images",
        (
            ("Filter images?", "Images only"),
            ("Select the rule criteria", 'and (file does containregexp "A01")'),
        ),
    )
    config = CellProfilerModule.require_module(module.name).contribute_source_bindings(
        module,
        SourceBindingsConfig(
            image_plane_sources=(ImagePlaneSource(uri="/data/A01.npy", index="2"),)
        ),
    )

    assert config.source_filters[0].match_type is SourceFilterMatchType.IS_IMAGE
    assert config.source_filters[1].subject is SourceFilterSubject.FILE
    assert config.source_filters[1].match_type is SourceFilterMatchType.CONTAINS_REGEX
    assert config.source_filters[1].value == "A01"
    assert config.image_plane_sources[0].uri == "/data/A01.npy"
    assert config.image_plane_sources[0].index == "2"


def test_images_setup_preserves_cellprofiler_regex_escapes_without_python_reparse() -> (
    None
):
    module = _module(
        1,
        "Images",
        (
            ("Filter images?", "Images only"),
            (
                "Select the rule criteria",
                r'and (directory doesnot containregexp "[\\/]\.")',
            ),
        ),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", SyntaxWarning)
        config = _fold_setup_modules(module)

    assert config.source_filters[1].value == r"[\/]\."


def test_images_setup_projects_disjunction_to_grouped_source_filters() -> None:
    module = _module(
        1,
        "Images",
        (
            ("Filter images?", "Images only"),
            (
                "Select the rule criteria",
                'or (extension does isimage) (file does endwith ".npy")',
            ),
        ),
    )

    config = _fold_setup_modules(module)

    assert len(config.source_filters) == 2
    assert {clause.any_group for clause in config.source_filters} == {0}
    assert source_filters_match("/data/image.png", config.source_filters)
    assert source_filters_match("/data/array.npy", config.source_filters)
    assert not source_filters_match("/data/notes.txt", config.source_filters)


def test_names_and_types_keeps_metadata_presence_out_of_path_filters() -> None:
    module = _module(
        1,
        "NamesAndTypes",
        (
            ("Assignments count", "1"),
            ("Select the image type", "Grayscale image"),
            ("Name to assign these images", "DNA"),
            (
                "Select the rule criteria",
                'and (metadata does Well) (file does contain "dapi")',
            ),
        ),
    )

    config = _fold_setup_modules(module)

    assert config.bindings[0].selector.filters == (
        SourceFilterClause(
            SourceFilterSubject.FILE,
            SourceFilterMatchType.CONTAINS,
            "dapi",
        ),
    )


def test_metadata_setup_contributes_path_rules_and_imported_tables() -> None:
    metadata = _module(
        1,
        "Metadata",
        (
            ("Extract metadata?", "Yes"),
            ("Metadata extraction method", "Extract from file/folder names"),
            ("Metadata source", "File name"),
            ("Extract metadata from", "Images matching a rule"),
            (
                "Regular expression to extract from file name",
                r".*(?P<Well>[A-H][0-9]{2})_s(?P<Site>[0-9]+)",
            ),
            ("Select the filtering criteria", 'and (file does contain "DNA")'),
            ("Metadata extraction method", "Import from file"),
            ("Metadata file location", "Default Input Folder|metadata"),
            ("Metadata file name", "plate.csv"),
            (
                "Match file and image metadata",
                "[{'Image Metadata': 'Well', 'CSV Metadata': 'WellID'}]",
            ),
        ),
    )

    config = _fold_setup_modules(metadata)

    assert len(config.metadata_rules) == 1
    assert config.metadata_rules[0].source is MetadataSource.FILE_NAME
    assert "Well" in config.metadata_rules[0].pattern
    assert config.metadata_rules[0].filters[0].value == "DNA"
    assert config.imported_metadata_tables[0].location == "metadata/plate.csv"
    assert config.imported_metadata_tables[0].joins[0].image_metadata_field == "Well"
    assert (
        config.imported_metadata_tables[0].joins[0].imported_metadata_field == "WellID"
    )
    assert config.metadata_fields == (
        FieldSpec("FileLocation", str, required=False),
        FieldSpec("Frame", str, required=False),
        FieldSpec("Series", str, required=False),
        FieldSpec("Well", str, required=False),
        FieldSpec("Site", str, required=False),
    )


def test_metadata_all_images_ignores_inactive_filtering_criteria() -> None:
    metadata = _module(
        1,
        "Metadata",
        (
            ("Extract metadata?", "Yes"),
            ("Metadata extraction method", "Extract from file/folder names"),
            ("Metadata source", "File name"),
            ("Extract metadata from", "All images"),
            (
                "Regular expression to extract from file name",
                r".*_w(?P<Channel>[0-9]+)",
            ),
            (
                "Select the filtering criteria",
                'and (file does containregexp "inactive\\.mat$")',
            ),
        ),
    )

    config = _fold_setup_modules(metadata)

    assert config.metadata_rules[0].filters == ()


def test_metadata_setup_preserves_declared_field_types_in_source_config() -> None:
    metadata = _module(
        1,
        "Metadata",
        (
            ("Extract metadata?", "Yes"),
            ("Metadata extraction method", "Extract from file/folder names"),
            ("Metadata source", "File name"),
            ("Extract metadata from", "All images"),
            (
                "Regular expression to extract from file name",
                r"^(?P<Dose>[0-9.]+)_(?P<Plate>[^.]+)",
            ),
            ("Metadata data type", "Choose for each"),
            (
                "Metadata types",
                '{"Dose":"float","Frame":"integer","Plate":"text"}',
            ),
        ),
    )

    config = _fold_setup_modules(metadata)

    assert config.metadata_fields == (
        FieldSpec("FileLocation", str, required=False),
        FieldSpec("Frame", int, required=False),
        FieldSpec("Series", int, required=False),
        FieldSpec("Dose", float, required=False),
        FieldSpec("Plate", str, required=False),
    )


def test_metadata_choose_mode_owns_reserved_types_and_omits_none_fields() -> None:
    metadata = _module(
        1,
        "Metadata",
        (
            ("Extract metadata?", "Yes"),
            ("Metadata extraction method", "Extract from file/folder names"),
            ("Metadata source", "File name"),
            ("Extract metadata from", "All images"),
            (
                "Regular expression to extract from file name",
                r"^(?P<Dose>[0-9.]+)_(?P<Ignored>[^.]+)",
            ),
            ("Metadata data type", "Choose for each"),
            (
                "Metadata types",
                '{"Frame":"text","Series":"float","Dose":"float",' '"Ignored":"none"}',
            ),
        ),
    )

    config = _fold_setup_modules(metadata)

    assert config.metadata_fields == (
        FieldSpec("FileLocation", str, required=False),
        FieldSpec("Frame", int, required=False),
        FieldSpec("Series", int, required=False),
        FieldSpec("Dose", float, required=False),
    )


def test_names_and_types_selector_uses_metadata_module_declared_dtype() -> None:
    metadata = _module(
        1,
        "Metadata",
        (
            ("Extract metadata?", "Yes"),
            ("Metadata extraction method", "Extract from file/folder names"),
            ("Metadata source", "File name"),
            ("Extract metadata from", "All images"),
            (
                "Regular expression to extract from file name",
                r"^(?P<Dose>[0-9.]+)",
            ),
            ("Metadata data type", "Choose for each"),
            ("Metadata types", '{"Dose":"float"}'),
        ),
    )
    names_and_types = _module(
        2,
        "NamesAndTypes",
        (
            ("Assignments count", "1"),
            ("Single images count", "0"),
            ("Select the rule criteria", 'and (metadata does Dose "0.25")'),
            ("Name to assign these images", "DoseImage"),
            ("Select the image type", "Grayscale image"),
            ("Image set matching method", "Order"),
        ),
    )

    config = _fold_setup_modules(metadata, names_and_types)

    (selector,) = config.binding_for_alias("DoseImage").selector.metadata
    assert selector.value == 0.25
    assert isinstance(selector.value, float)


def test_disabled_metadata_does_not_infer_path_identity_from_inactive_settings() -> (
    None
):
    metadata = _module(
        1,
        "Metadata",
        (
            ("Extract metadata?", "No"),
            ("Metadata extraction method", "Extract from file/folder names"),
            ("Metadata source", "File name"),
            ("Extract metadata from", "All images"),
            (
                "Regular expression to extract from file name",
                r"^(?P<Plate>.*)_(?P<Well>[A-H][0-9]{2})_s(?P<Site>[0-9]+)",
            ),
        ),
    )

    assert _fold_setup_modules(metadata).metadata_rules == ()


def test_disabled_metadata_accepts_declared_image_header_method_as_inactive() -> None:
    metadata = _module(
        1,
        "Metadata",
        (
            ("Extract metadata?", "No"),
            ("Metadata data type", "Text"),
            ("Metadata types", "{}"),
            ("Extraction method count", "1"),
            ("Metadata extraction method", "Extract from image file headers"),
            ("Metadata source", "File name"),
            (
                "Regular expression to extract from file name",
                r"^(?P<Plate>.*)_(?P<Well>[A-P][0-9]{2})_s(?P<Site>[0-9])_w(?P<ChannelNumber>[0-9])",
            ),
            ("Extract metadata from", "All images"),
        ),
    )

    config = _fold_setup_modules(metadata)

    assert config.metadata_rules == ()
    assert config.imported_metadata_tables == ()
    assert config.metadata_fields == (FieldSpec("FileLocation", str, required=False),)


@pytest.mark.parametrize(
    ("extraction_enabled", "method", "error"),
    (
        (
            True,
            "Extract from image file headers",
            "Unsupported active CellProfiler Metadata extraction method",
        ),
        (False, "Undeclared extraction method", "is not a valid"),
    ),
)
def test_metadata_rejects_unsupported_extraction_method_states(
    extraction_enabled: bool,
    method: str,
    error: str,
) -> None:
    metadata = _module(
        1,
        "Metadata",
        (
            ("Extract metadata?", "Yes" if extraction_enabled else "No"),
            ("Metadata extraction method", method),
        ),
    )

    with pytest.raises(ValueError, match=error):
        _fold_setup_modules(metadata)


def test_names_and_types_contributes_typed_bindings_and_metadata_match_plan() -> None:
    module = _module(
        2,
        "NamesAndTypes",
        (
            ("Assignments count", "2"),
            ("Select the rule criteria", 'and (metadata does channel "1")'),
            ("Name to assign these images", "DAPI"),
            ("Select the image type", "Grayscale image"),
            ("Select the rule criteria", 'and (metadata does illum "DAPI")'),
            ("Name to assign these images", "DAPIillum"),
            ("Select the image type", "Illumination function"),
            ("Match metadata", "[{'DAPI': 'folder', 'DAPIillum': 'folder_illum'}]"),
            ("Image set matching method", "Metadata"),
        ),
    )

    config = _fold_setup_modules(module)
    dapi = config.binding_for_alias("DAPI")
    illumination = config.binding_for_alias("DAPIillum")

    assert dapi.artifact_kind is ImageArtifactType
    assert dapi.origin is SourceBindingOrigin.PIPELINE_START
    assert dapi.selector.metadata[0].field == "channel"
    assert dapi.selector.metadata[0].value == "1"
    assert illumination.artifact_kind is ImageArtifactType
    assert illumination.origin is SourceBindingOrigin.PIPELINE_START
    assert illumination.source_set_role is SourceSetRole.MATCHED
    assert illumination.projection_role is SourceProjectionRole.SOURCE_ARTIFACT
    assert config.match_plan is not None
    assert config.match_plan.method is SourceBindingMatchMethod.METADATA
    assert config.match_plan.dimensions[0].field_for_alias("DAPI") == "folder"
    assert (
        config.match_plan.dimensions[0].field_for_alias("DAPIillum") == "folder_illum"
    )


def test_names_and_types_parses_declared_single_image_as_broadcast_member() -> None:
    module = _module(
        2,
        "NamesAndTypes",
        (
            ("Assignments count", "1"),
            ("Single images count", "1"),
            ("Select the rule criteria", 'and (file does contain "DAPI")'),
            ("Name to assign these images", "DAPI"),
            ("Select the image type", "Grayscale image"),
            ("Single image location", "file:/data/flatfield.tif 2 3 4"),
            ("Name to assign this image", "Flatfield"),
            ("Select the image type", "Illumination function"),
            ("Image set matching method", "Order"),
        ),
    )

    config = _fold_setup_modules(module)
    dapi = config.binding_for_alias("DAPI")
    flatfield = config.binding_for_alias("Flatfield")

    assert dapi.source_set_role is SourceSetRole.MATCHED
    assert dapi.projection_role is SourceProjectionRole.PRIMARY_PLANE
    assert flatfield.source_set_role is SourceSetRole.BROADCAST
    assert flatfield.projection_role is SourceProjectionRole.SOURCE_ARTIFACT
    assert flatfield.explicit_source == ImagePlaneSource(
        uri="file:/data/flatfield.tif",
        series="2",
        index="3",
        channel="4",
    )
    assert config.image_plane_sources == (flatfield.explicit_source,)


def test_names_and_types_does_not_override_metadata_declared_channel_identity() -> None:
    metadata = _module(
        1,
        "Metadata",
        (
            ("Extract metadata?", "Yes"),
            ("Metadata extraction method", "Extract from file/folder names"),
            ("Metadata source", "File name"),
            ("Extract metadata from", "All images"),
            (
                "Regular expression to extract from file name",
                r"^.*_w(?P<ChannelNumber>[0-9]+)",
            ),
        ),
    )
    names_and_types = _module(
        2,
        "NamesAndTypes",
        (
            ("Assignments count", "2"),
            ("Select the rule criteria", 'and (file does contain "w2")'),
            ("Name to assign these images", "rawDNA"),
            ("Select the image type", "Grayscale image"),
            ("Select the rule criteria", 'and (file does contain "w1")'),
            ("Name to assign these images", "rawGFP"),
            ("Select the image type", "Grayscale image"),
        ),
    )

    config = _fold_setup_modules(metadata, names_and_types)

    assert config.metadata_rules[0].capture_fields == ("ChannelNumber",)
    assert config.binding_for_alias("rawDNA").component_identity == ()
    assert config.binding_for_alias("rawGFP").component_identity == ()


def test_names_and_types_contributes_payload_loading_semantics(
    tmp_path: Path,
) -> None:
    import tifffile

    module = _module(
        1,
        "NamesAndTypes",
        (
            ("Assignments count", "3"),
            ("Select the rule criteria", 'and (metadata does channel "1")'),
            ("Name to assign these images", "Color"),
            ("Name to assign these objects", "Cell"),
            ("Select the image type", "Color image"),
            ("Select the rule criteria", 'and (metadata does channel "2")'),
            ("Name to assign these images", "Mask"),
            ("Name to assign these objects", "Cell"),
            ("Select the image type", "Binary mask"),
            ("Select the rule criteria", 'and (metadata does channel "3")'),
            ("Name to assign these images", "DNA"),
            ("Name to assign these objects", "LoadedNuclei"),
            ("Select the image type", "Objects"),
        ),
    )

    config = _fold_setup_modules(module)
    color = config.binding_for_alias("Color")
    mask = config.binding_for_alias("Mask")
    objects = config.binding_for_alias("LoadedNuclei")

    assert color.artifact_kind is ImageArtifactType
    assert color.source_channel_axis == -1
    assert color.source_channel_counts is None
    assert color.source_channel_axis_for_shape((4, 5)) is None
    assert color.source_channel_axis_for_shape((4, 5, 2)) == -1
    assert color.source_channel_axis_for_shape((4, 5, 3)) == -1
    multiband_data = np.zeros((5, 6, 2), dtype=np.float32)
    multiband_path = tmp_path / "multiband.tiff"
    tifffile.imwrite(multiband_path, multiband_data)
    multiband = apply_source_binding_payload(
        tifffile.imread(multiband_path),
        color,
        ImagePayloadSourceMetadataContext(
            SourceImageIdentity(str(multiband_path)),
        ),
    )
    multiband_metadata = image_payload_metadata(multiband)
    assert multiband_metadata.source_channel_axis == -1
    assert multiband_metadata.source_spatial_shape_yx == (5, 6)
    grayscale = apply_source_binding_payload(
        np.zeros((512, 512), dtype=np.uint8),
        color,
        None,
    )
    assert image_payload_metadata(grayscale).source_channel_axis is None
    assert mask.artifact_kind is ImageArtifactType
    assert mask.load_as_mask is True
    assert objects.artifact_kind is ObjectLabelsArtifactType
    assert objects.source_set_role is SourceSetRole.MATCHED
    assert objects.projection_role is SourceProjectionRole.SOURCE_ARTIFACT


def test_monochrome_source_normalizes_before_collapsing_rgb_channels() -> None:
    from skimage.color import rgb2gray

    codes = np.asarray(
        (
            (7, 98, 128, 254),
            (23, 46, 115, 205),
        ),
        dtype=np.uint8,
    )
    rgb = np.repeat(codes[..., np.newaxis], 3, axis=-1)
    binding = NamedSourceBinding(
        alias="phase",
        load_as_monochrome=True,
        source_channel_axis=-1,
        source_channel_counts=frozenset((3, 4)),
    )

    observed = apply_source_binding_payload(rgb, binding, None)
    expected = rgb2gray(rgb.astype(np.float32) / np.float32(255))

    np.testing.assert_array_equal(image_payload_data(observed), expected)
    metadata = image_payload_metadata(observed)
    assert image_payload_data(observed).dtype == np.float32
    assert metadata.source_channel_axis is None
    assert metadata.unit_interval_intensity_scale is None


def test_names_and_types_repeated_columns_require_exact_cardinality() -> None:
    module = _module(
        1,
        "NamesAndTypes",
        (
            ("Assignments count", "2"),
            ("Select the rule criteria", 'and (metadata does channel "1")'),
            ("Name to assign these images", "DAPI"),
            ("Name to assign these images", "Actin"),
            ("Select the image type", "Grayscale image"),
            ("Select the image type", "Binary mask"),
            ("Select the image type", "Color image"),
        ),
    )

    with pytest.raises(ValueError, match="cardinality|assignment|3"):
        _fold_setup_modules(module)


def test_names_and_types_contributes_3d_axis_and_voxel_spacing() -> None:
    module = _module(
        1,
        "NamesAndTypes",
        (
            ("Assignments count", "1"),
            ("Select the rule criteria", 'and (metadata does channel "1")'),
            ("Name to assign these images", "DNA"),
            ("Select the image type", "Grayscale image"),
            ("Process as 3D?", "Yes"),
            ("Relative pixel spacing in X", "0.5"),
            ("Relative pixel spacing in Y", "1.0"),
            ("Relative pixel spacing in Z", "2.0"),
        ),
    )

    config = _fold_setup_modules(module)

    assert config.source_stack_components == (AllComponents.Z_INDEX,)
    assert config.source_voxel_spacing.values_zyx == (2.0, 1.0, 0.5)


def test_load_images_contributes_binding_filters_metadata_and_grouping() -> None:
    module = _module(
        1,
        "LoadImages",
        (
            ("What type of files are you loading?", "individual images"),
            ("How do you want to load these files?", "Text-Exact match"),
            ("Do you want to exclude certain files?", "Yes"),
            ("Type the text that the excluded images have in common", "ILLUM"),
            ("Do you want to group image sets by metadata?", "Yes"),
            ("What metadata fields do you want to group by?", "WellRow,WellCol"),
            (
                "Type the text that these images have in common (case-sensitive)",
                "Channel2",
            ),
            ("What do you want to call this image in CellProfiler?", "DNA"),
            (
                "Do you want to extract metadata from the file name, the subfolder path or both?",
                "File name",
            ),
            (
                "Type the regular expression that finds metadata in the file name\x3a",
                r"^.*-(?P<WellRow>.+)-(?P<WellCol>\x5B0-9\x5D{2})",
            ),
        ),
    )

    config = _fold_setup_modules(module)
    binding = config.binding_for_alias("DNA")

    assert binding.origin is SourceBindingOrigin.PIPELINE_START
    assert binding.selector.filters[0].subject is SourceFilterSubject.FILE
    assert binding.selector.filters[0].match_type is SourceFilterMatchType.CONTAINS
    assert binding.selector.filters[0].value == "Channel2"
    assert (
        binding.selector.filters[1].match_type is SourceFilterMatchType.DOES_NOT_CONTAIN
    )
    assert binding.selector.filters[1].value == "ILLUM"
    assert config.metadata_rules[0].source is MetadataSource.FILE_NAME
    assert config.metadata_rules[0].pattern == (
        r"^.*-(?P<WellRow>.+)-(?P<WellCol>[0-9]{2})"
    )
    assert config.grouping_metadata_fields == ("WellRow", "WellCol")


def test_groups_contributes_grouping_fields() -> None:
    module = _module(
        3,
        "Groups",
        (
            ("Do you want to group your images?", "Yes"),
            ("Metadata category", "Plate"),
            ("Metadata category", "Well"),
        ),
    )

    config = _fold_setup_modules(module)

    assert config.grouping_metadata_fields == ("Plate", "Well")


def test_setup_fold_is_source_ordered_and_uses_module_registry() -> None:
    images = _module(
        1,
        "Images",
        (("Filter images?", "Images only"),),
    )
    names = _module(
        2,
        "NamesAndTypes",
        (
            ("Assignments count", "1"),
            ("Select the rule criteria", 'and (file does contain "DNA")'),
            ("Name to assign these images", "DNA"),
            ("Select the image type", "Grayscale image"),
        ),
    )
    groups = _module(
        3,
        "Groups",
        (
            ("Do you want to group your images?", "Yes"),
            ("Metadata category", "Well"),
        ),
    )

    config = _fold_setup_modules(images, names, groups)

    assert tuple(binding.alias for binding in config.bindings) == ("DNA",)
    assert config.source_filters[0].match_type is SourceFilterMatchType.IS_IMAGE
    assert config.grouping_metadata_fields == ("Well",)
    assert all(
        CellProfilerModule.require_module(module.name).emits_function_step() is False
        for module in (images, names, groups)
    )
