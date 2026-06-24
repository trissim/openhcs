from openhcs.runtime.zmq_orchestrator_environment import (
    ZMQOrchestratorEnvironmentRequest,
)


def test_pycodified_execution_initializes_from_source_plate_path():
    request = ZMQOrchestratorEnvironmentRequest(
        execution_id="exec-1",
        plate_id="/tmp/source-plate",
        execution_plate_id="/tmp/new-output-plate",
        selected_pipeline_path=None,
        global_config=object(),
        config_params=None,
    )

    assert request.prepared_plate_path({}) == "/tmp/source-plate"


def test_selected_external_pipeline_initializes_from_prepared_workspace_path():
    request = ZMQOrchestratorEnvironmentRequest(
        execution_id="exec-1",
        plate_id="/tmp/source-plate#openhcs-cppipe=segmentation_final.cppipe",
        execution_plate_id="/tmp/source-plate/.openhcs_cellprofiler/segmentation_final",
        selected_pipeline_path="/tmp/source-plate/segmentation_final.cppipe",
        global_config=object(),
        config_params=None,
    )

    assert (
        request.prepared_plate_path({})
        == "/tmp/source-plate/.openhcs_cellprofiler/segmentation_final"
    )
