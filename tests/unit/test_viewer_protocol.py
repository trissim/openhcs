import subprocess
import sys

import pytest

from openhcs.runtime.viewer_protocol import ViewerProcessHandle


def test_viewer_process_handle_wraps_subprocess_lifecycle():
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        handle = ViewerProcessHandle.from_process(process)

        assert handle.pid == process.pid
        assert handle.pid_label == str(process.pid)
        assert handle.is_alive()
        assert not handle.terminate(timeout=1, kill_timeout=1)
        assert not handle.is_alive()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=1)


def test_viewer_process_handle_rejects_structural_process_lookalikes():
    class ProcessLike:
        def is_alive(self):
            return True

    with pytest.raises(TypeError, match="Unsupported viewer process handle"):
        ViewerProcessHandle.from_process(ProcessLike())
