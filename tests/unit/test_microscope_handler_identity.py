from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from openhcs.agent.dto.config import ConfigPatch
from openhcs.agent.dto.execution import PipelineSourceArtifactPlanInspectionRequest
from openhcs.agent.dto.plate import PlatePathInspectionRequest
from openhcs.agent.path_policy import AgentPathPolicy
from openhcs.agent.services.config_service import ConfigService
from openhcs.agent.services.execution_session_service import ExecutionSessionService
from openhcs.agent.services.plate_inspection_service import PlateInspectionService
from openhcs.constants.constants import Microscope
from openhcs.core.config import LazyWellFilterConfig, PipelineConfig
from openhcs.core.config_document import ConfigDocumentAuthority
from openhcs.core.pipeline_document import PipelineDocumentAuthority
from openhcs.core.steps.function_step import FunctionStep
from openhcs.microscopes import create_microscope_handler, get_all_handler_types
from openhcs.microscopes.microscope_base import (
    MicroscopeHandler,
    MicroscopeSourceSelectionRole,
)
from openhcs.microscopes.opera_phenix import OperaPhenixHandler
from openhcs.processing.backends.processors.numpy_processor import percentile_normalize
from openhcs.demo.synthetic_data import (
    SyntheticMicroscopyGenerator,
)
from tests.unit.bioformats_fixture import bioformats_filemanager


def _write_valid_opera_phenix_plate(root: Path) -> None:
    generator = SyntheticMicroscopyGenerator(
        output_dir=str(root),
        grid_size=(2, 2),
        tile_size=(8, 8),
        overlap_percent=0,
        stage_error_px=1,
        wavelengths=3,
        z_stack_levels=1,
        num_cells=0,
        wells=["D09"],
        format="OperaPhenix",
        random_seed=1,
    )
    for channel in (1, 2, 3):
        tifffile.imwrite(
            generator.images_dir / f"r04c09f1p01-ch{channel}sk1fk1fl1.tiff",
            np.full((8, 8), channel, dtype=np.uint16),
        )
    generator.generate_opera_phenix_index_xml(root.name)


def test_typed_microscope_values_are_exact_registered_handler_keys() -> None:
    handler_types = set(get_all_handler_types())
    configured_types = {
        microscope.value
        for microscope in Microscope
        if microscope is not Microscope.AUTO
    }

    assert configured_types == handler_types
    assert {
        handler_type._microscope_type
        for handler_type in MicroscopeHandler.__registry__.values()
        if handler_type._microscope_type in configured_types
    } == configured_types


def test_config_schema_patch_and_source_share_opera_handler_identity() -> None:
    service = ConfigService()
    schema = service.describe_schema("pipeline")
    microscope_field = next(
        field for field in schema.fields if field.path == "microscope"
    )

    assert microscope_field.enum_values == tuple(
        microscope.value for microscope in Microscope
    )
    assert Microscope.OPERAPHENIX.value == "opera_phenix"
    assert Microscope.OPERAPHENIX.value in microscope_field.enum_values
    assert "OperaPhenix" not in microscope_field.enum_values

    config_ref = service.create(
        "pipeline",
        ConfigPatch(
            config_type="PipelineConfig",
            values={"microscope": Microscope.OPERAPHENIX.value},
        ),
    )
    config = service.resolve_ref(config_ref)
    rendered = service.render_source(config_ref)

    assert config.microscope is Microscope.OPERAPHENIX
    assert "microscope=Microscope.OPERAPHENIX" in rendered.source
    assert (
        ConfigDocumentAuthority.from_source(
            rendered.source,
            expected_config_type=PipelineConfig,
        ).microscope
        is Microscope.OPERAPHENIX
    )


def test_handler_factory_uses_exact_declared_identity(tmp_path: Path) -> None:
    handler = create_microscope_handler(
        microscope_type=Microscope.OPERAPHENIX.value,
        plate_folder=tmp_path,
        filemanager=bioformats_filemanager(),
    )

    assert isinstance(handler, OperaPhenixHandler)
    assert handler.microscope_type == Microscope.OPERAPHENIX.value
    with pytest.raises(ValueError, match="Unsupported microscope type: OperaPhenix"):
        create_microscope_handler(
            microscope_type="OperaPhenix",
            plate_folder=tmp_path,
            filemanager=bioformats_filemanager(),
        )


def test_source_selection_role_owns_local_availability_contract(
    tmp_path: Path,
) -> None:
    missing_path = tmp_path / "missing"
    file_path = tmp_path / "image.tif"
    file_path.touch()

    with pytest.raises(FileNotFoundError, match=str(missing_path)):
        MicroscopeSourceSelectionRole.FORMAT_SPECIFIC.require_available_source(
            missing_path
        )
    with pytest.raises(NotADirectoryError, match=str(file_path)):
        MicroscopeSourceSelectionRole.FORMAT_SPECIFIC.require_available_source(
            file_path
        )

    MicroscopeSourceSelectionRole.REMOTE_SERVICE.require_available_source(missing_path)


def test_explicit_inspection_and_artifact_plan_reach_opera_axes(
    tmp_path: Path,
) -> None:
    _write_valid_opera_phenix_plate(tmp_path)
    path_policy = AgentPathPolicy.with_roots(
        readable_roots=(tmp_path,),
        writable_roots=(tmp_path,),
    )
    inspection = PlateInspectionService(
        path_policy,
        filemanager_factory=type(
            "BioFormatsFileManagerFactory",
            (),
            {"create": staticmethod(bioformats_filemanager)},
        )(),
    ).inspect(
        PlatePathInspectionRequest.from_fields(
            plate_path=str(tmp_path),
            microscope_type=Microscope.OPERAPHENIX.value,
        )
    )

    assert inspection.errors == ()
    assert inspection.detected_microscope_type == Microscope.OPERAPHENIX.value
    assert inspection.available_microscope_types == tuple(
        sorted(get_all_handler_types())
    )
    assert inspection.image_files.count == 3

    config = PipelineConfig(
        microscope=Microscope.OPERAPHENIX,
        well_filter_config=LazyWellFilterConfig(well_filter="R04C09"),
    )
    pipeline_source = PipelineDocumentAuthority.render(
        PipelineDocumentAuthority.from_values(
            pipeline_config=config,
            pipeline_steps=[FunctionStep(func=percentile_normalize)],
        )
    )
    artifact_plan = ExecutionSessionService(
        path_policy=path_policy,
        pipeline_service=object(),
        config_service=ConfigService(),
    ).inspect_pipeline_source_artifact_plan_request(
        PipelineSourceArtifactPlanInspectionRequest.from_fields(
            plate_path=str(tmp_path),
            pipeline_source=pipeline_source,
            axis_filter=["R04C09"],
        )
    )

    assert artifact_plan.errors == ()
    assert artifact_plan.axes == ("R04C09",)
    assert artifact_plan.source_workspace.axis_file_counts == {"R04C09": 3}
