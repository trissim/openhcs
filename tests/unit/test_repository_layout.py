"""Repository-layout regression gates."""

from pathlib import Path

TEST_ROOT = Path(__file__).resolve().parents[1]
MAX_TEST_ROOT_FILE_BYTES = 1_000_000


def test_test_root_does_not_hold_large_fixture_payloads() -> None:
    """Keep large fixtures in declared test-data trees instead of the test root."""

    oversized = {
        path.name: path.stat().st_size
        for path in TEST_ROOT.iterdir()
        if path.is_file() and path.stat().st_size > MAX_TEST_ROOT_FILE_BYTES
    }

    assert oversized == {}
