from pathlib import Path

import pytest

from openhcs.agent.path_policy import AgentPathPolicy, AgentPathPolicyError


def test_path_policy_allows_paths_under_roots(tmp_path: Path):
    readable = tmp_path / "readable.txt"
    readable.write_text("ok")
    writable = tmp_path / "nested" / "output.py"
    policy = AgentPathPolicy.with_roots(
        readable_roots=(tmp_path.resolve(),),
        writable_roots=(tmp_path.resolve(),),
    )

    assert policy.assert_readable(readable) == readable.resolve()
    assert policy.assert_writable(writable) == writable.resolve(strict=False)


def test_path_policy_rejects_paths_outside_roots(tmp_path: Path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("no")
    policy = AgentPathPolicy.with_roots(
        readable_roots=(allowed.resolve(),),
        writable_roots=(allowed.resolve(),),
    )

    with pytest.raises(AgentPathPolicyError):
        policy.assert_readable(outside)

    with pytest.raises(AgentPathPolicyError):
        policy.assert_writable(outside)
