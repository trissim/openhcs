from __future__ import annotations

import pickle
from types import MappingProxyType

from openhcs.core.orchestrator.execution_result import ExecutionResult


def test_execution_result_transport_preserves_mapping_proxy_metadata() -> None:
    result = ExecutionResult.success(axis_id="W001")
    payload = {"result": result, "metadata": MappingProxyType({"well": "W001"})}

    restored = pickle.loads(pickle.dumps(payload))

    assert restored["result"].is_success()
    assert restored["metadata"] == {"well": "W001"}
    assert isinstance(restored["metadata"], MappingProxyType)
