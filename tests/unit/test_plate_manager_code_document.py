from pathlib import Path

import pytest

from openhcs.core.config import GlobalPipelineConfig, PipelineConfig
from openhcs.ui.shared.plate_manager_code_document import (
    PlateManagerCodeDocumentAuthority,
)


def test_plate_manager_document_round_trips_all_semantic_fields(tmp_path: Path):
    plate_a = tmp_path / "plate-a"
    plate_b = tmp_path / "plate-b"
    global_config = GlobalPipelineConfig(num_workers=7)
    configs = {
        plate_a: PipelineConfig(num_workers=2),
        plate_b: PipelineConfig(num_workers=3),
    }
    payload = PlateManagerCodeDocumentAuthority.from_values(
        plate_paths=(plate_a, plate_b),
        global_pipeline_config=global_config,
        per_plate_configs=configs,
        pipeline_data={plate_a: [], plate_b: []},
    )

    source = PlateManagerCodeDocumentAuthority.render(payload)
    restored = PlateManagerCodeDocumentAuthority.from_source(source)

    assert "plate_paths =" in source
    assert "global_config = GlobalPipelineConfig(" in source
    assert "per_plate_configs =" in source
    assert "pipeline_data =" in source
    assert restored.plate_paths == (str(plate_a), str(plate_b))
    assert restored.global_pipeline_config == global_config
    assert restored.per_plate_configs == {
        str(path): config for path, config in configs.items()
    }
    assert restored.pipeline_data == {str(plate_a): [], str(plate_b): []}


@pytest.mark.parametrize(
    "missing_field",
    ("plate_paths", "global_config", "per_plate_configs", "pipeline_data"),
)
def test_plate_manager_document_requires_complete_payload(missing_field: str):
    namespace = {
        "plate_paths": ["/plate"],
        "global_config": GlobalPipelineConfig(),
        "per_plate_configs": {"/plate": PipelineConfig()},
        "pipeline_data": {"/plate": []},
    }
    del namespace[missing_field]

    with pytest.raises(ValueError, match=missing_field):
        PlateManagerCodeDocumentAuthority.from_namespace(namespace)


def test_plate_manager_document_rejects_misaligned_plate_keys():
    with pytest.raises(ValueError, match="per_plate_configs keys"):
        PlateManagerCodeDocumentAuthority.from_values(
            plate_paths=("/plate-a",),
            global_pipeline_config=GlobalPipelineConfig(),
            per_plate_configs={"/plate-b": PipelineConfig()},
            pipeline_data={"/plate-a": []},
        )
