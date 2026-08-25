from types import SimpleNamespace

from objectstate import ObjectStateRegistry

from openhcs.constants.constants import GroupBy, OrchestratorState
from openhcs.core.orchestrator import PipelineOrchestrator
from openhcs.pyqt_gui.services.reactor_providers import (
    OpenHCSComponentSelectionProvider,
)


class _ComponentOrchestrator(PipelineOrchestrator):
    def get_component_keys(self, group_by, component_filter=None):
        del component_filter
        assert group_by is GroupBy.CHANNEL
        return ["1", "2"]


def test_component_provider_resolves_the_public_orchestrator_declaration(
    monkeypatch,
    tmp_path,
) -> None:
    orchestrator = _ComponentOrchestrator(tmp_path)
    provider = OpenHCSComponentSelectionProvider()
    monkeypatch.setattr(
        provider,
        "_get_plate_manager",
        lambda: SimpleNamespace(selected_plate_path=str(tmp_path)),
    )
    monkeypatch.setattr(
        ObjectStateRegistry,
        "get_object",
        lambda _scope_id: orchestrator,
    )

    assert provider.has_components_available(GroupBy.CHANNEL) is False

    orchestrator.state = OrchestratorState.READY

    assert provider.has_components_available(GroupBy.CHANNEL) is True
    assert provider.get_component_keys(GroupBy.CHANNEL) == ["1", "2"]
