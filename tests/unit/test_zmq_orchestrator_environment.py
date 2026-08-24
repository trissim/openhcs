from openhcs.runtime.zmq_orchestrator_environment import (
    ZMQOrchestratorEnvironmentRequest,
)


def test_pycodified_execution_initializes_from_source_plate_path():
    request = ZMQOrchestratorEnvironmentRequest(
        execution_id="exec-1",
        plate_id="/tmp/source-plate",
        execution_plate_id="/tmp/new-output-plate",
        selected_pipeline_path=None,
        debug_execution_config=None,
    )

    assert request.prepared_plate_path({}) == "/tmp/source-plate"


def test_selected_external_pipeline_initializes_from_prepared_workspace_path():
    request = ZMQOrchestratorEnvironmentRequest(
        execution_id="exec-1",
        plate_id="/tmp/source-plate#openhcs-cppipe=segmentation_final.cppipe",
        execution_plate_id="/tmp/source-plate/.openhcs_cellprofiler/segmentation_final",
        selected_pipeline_path="/tmp/source-plate/segmentation_final.cppipe",
        debug_execution_config=None,
    )

    assert (
        request.prepared_plate_path({})
        == "/tmp/source-plate/.openhcs_cellprofiler/segmentation_final"
    )


def test_omero_plate_uses_backend_metadata_namespace_authority(monkeypatch):
    from polystore import omero_local

    from openhcs.runtime import omero_instance_manager

    connection = object()
    backend_kwargs = {}

    class FakeOMEROManager:
        conn = connection

        def connect(self, timeout):
            assert timeout == 60
            return True

        def close(self):
            pass

    class FakeOMEROBackend:
        def __init__(self, **kwargs):
            backend_kwargs.update(kwargs)

    monkeypatch.setattr(
        omero_instance_manager,
        "OMEROInstanceManager",
        FakeOMEROManager,
    )
    monkeypatch.setattr(omero_local, "OMEROLocalBackend", FakeOMEROBackend)
    storage_registry = {}
    request = ZMQOrchestratorEnvironmentRequest(
        execution_id="exec-1",
        plate_id="17",
        execution_plate_id=None,
        selected_pipeline_path=None,
        debug_execution_config=None,
    )

    assert request.prepared_plate_path(storage_registry) == "/omero/plate_17"
    assert backend_kwargs == {
        "omero_conn": connection,
        "lock_dir_name": ".openhcs",
    }
    assert isinstance(storage_registry["omero_local"], FakeOMEROBackend)
