from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import numpy as np


def test_declared_library_submodules_are_imported_from_the_package_authority(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Lazy package exports do not have to be preloaded before discovery."""

    package_root = tmp_path / "lazy_registry_package"
    package_root.mkdir()
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    (package_root / "restoration.py").write_text(
        textwrap.dedent(
            """
            def _denoise(image):
                return image

            def __dir__():
                return ["denoise"]

            def __getattr__(name):
                if name == "denoise":
                    return _denoise
                raise AttributeError(name)
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    import importlib

    lazy_package = importlib.import_module("lazy_registry_package")
    assert "restoration" not in vars(lazy_package)

    from openhcs.processing.backends.lib_registry import scikit_image_registry

    monkeypatch.setattr(scikit_image_registry, "skimage", lazy_package)
    registry = scikit_image_registry.SkimageRegistry()
    registry.MODULES_TO_SCAN = ["restoration"]

    modules = registry.get_modules_to_scan()

    assert modules == [
        (
            "restoration",
            importlib.import_module("lazy_registry_package.restoration"),
        )
    ]
    functions = registry.discover_functions()
    assert tuple(functions) == ("restoration.denoise",)


def test_runtime_discovery_requires_an_array_main_flow_output() -> None:
    """Successful calls with non-image returns are not processing functions."""

    from openhcs.processing.backends.lib_registry.scikit_image_registry import (
        SkimageRegistry,
    )

    registry = SkimageRegistry()

    def figure_result(image):
        del image
        return object(), object()

    def image_with_auxiliary_result(image):
        return image, {"mean": float(np.mean(image))}

    rejected_contract, rejected = registry.classify_function_behavior(figure_result)
    accepted_contract, accepted = registry.classify_function_behavior(
        image_with_auxiliary_result
    )

    assert rejected_contract is None
    assert rejected is False
    assert accepted_contract is not None
    assert accepted is True


def test_registry_cache_miss_is_prepared_out_of_process(monkeypatch) -> None:
    """A cold caller never performs runtime behavior probes in its own thread."""

    from openhcs.processing.backends.lib_registry.registry_service import (
        RegistryService,
    )

    prepared: list[bool] = []
    cached_catalog = {"numpy:identity": object()}
    cache_reads = iter((None, cached_catalog))
    monkeypatch.setattr(RegistryService, "_metadata_cache", None)
    monkeypatch.setattr(
        RegistryService,
        "_load_valid_persistent_catalog",
        classmethod(lambda cls: next(cache_reads)),
    )
    monkeypatch.setattr(
        RegistryService,
        "_prepare_persistent_catalog",
        classmethod(lambda cls: prepared.append(True)),
    )

    assert RegistryService.get_all_functions_with_metadata() is cached_catalog
    assert prepared == [True]
    assert RegistryService._metadata_cache is cached_catalog


def test_registry_preparation_uses_background_process_policy(monkeypatch) -> None:
    """The dedicated preparation interpreter remains console-free on Windows."""

    from openhcs.processing.backends.lib_registry import registry_service

    calls: list[tuple[tuple[str, ...], dict[str, object]]] = []
    policy = SimpleNamespace(
        python_executable=lambda executable: "pythonw.exe",
        popen_arguments=lambda: {"creationflags": 73},
    )
    monkeypatch.setattr(
        registry_service.BackgroundProcessLaunchPolicy,
        "current",
        classmethod(lambda cls, *, detached=False: policy),
    )

    def run(command, **kwargs):
        calls.append((tuple(command), kwargs))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(registry_service.subprocess, "run", run)

    registry_service.RegistryService._prepare_persistent_catalog()

    assert calls[0][0][:4] == (
        "pythonw.exe",
        "-m",
        "openhcs.runtime.zmq_execution_server_launcher",
        "--prepare-capabilities",
    )
    assert calls[0][1]["creationflags"] == 73


def test_library_registry_discovery_is_stable_across_fresh_worker_processes(
    tmp_path: Path,
) -> None:
    """Cold and cached fresh interpreters project the same exact declarations."""

    repository_root = Path(__file__).parents[2]
    environment = os.environ.copy()
    environment.update(
        {
            "OPENHCS_CPU_ONLY": "true",
            "XDG_CACHE_HOME": str(tmp_path / "cache"),
            "XDG_DATA_HOME": str(tmp_path / "data"),
        }
    )
    script = textwrap.dedent(
        """
        import importlib
        import json

        from openhcs.processing.backends.lib_registry.unified_registry import (
            LibraryRegistryBase,
        )

        declarations = {}
        for key, declaration in LibraryRegistryBase.__registry__.items():
            module = importlib.import_module(declaration.__module__)
            resolved = module
            for owner_name in declaration.__qualname__.split("."):
                resolved = getattr(resolved, owner_name)
            assert resolved is declaration
            declarations[key] = (
                declaration.__module__,
                declaration.__qualname__,
            )

        config = LibraryRegistryBase.__registry__._config
        assert config.discovery_package == "openhcs.processing.backends.lib_registry"
        assert config.discovery_recursive is False
        print(json.dumps(declarations, sort_keys=True))
        """
    )

    projections = []
    for _ in range(2):
        completed = subprocess.run(
            (sys.executable, "-c", script),
            cwd=repository_root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert completed.returncode == 0, completed.stderr
        projections.append(json.loads(completed.stdout))

    assert projections[0] == projections[1]
    assert {"openhcs", "skimage"}.issubset(projections[0])
    assert tuple((tmp_path / "cache" / "metaclass-registry").glob("*.json"))


def test_cold_execution_server_catalog_request_discovers_library_roots(
    tmp_path: Path,
) -> None:
    """The first live catalog request initializes the server-owned library registry."""

    repository_root = Path(__file__).parents[2]
    environment = os.environ.copy()
    environment.update(
        {
            "OPENHCS_CPU_ONLY": "true",
            "XDG_CACHE_HOME": str(tmp_path / "cache"),
            "XDG_DATA_HOME": str(tmp_path / "data"),
        }
    )
    script = textwrap.dedent(
        """
        import importlib
        import json
        import socket
        import threading
        import time

        from openhcs.agent.dto.functions import FunctionCatalogControlRequest
        from openhcs.processing.backends.lib_registry.unified_registry import (
            LibraryRegistryBase,
        )
        from openhcs.runtime.zmq_execution_client import ZMQExecutionClient
        from openhcs.runtime.zmq_execution_server import ZMQExecutionServer

        with socket.socket() as candidate:
            candidate.bind(("127.0.0.1", 0))
            port = candidate.getsockname()[1]

        server = ZMQExecutionServer(port=port, host="127.0.0.1")
        startup_started = time.perf_counter()
        server.start()
        startup_seconds = time.perf_counter() - startup_started

        def pump_server():
            while server.is_running():
                server.process_messages()
                time.sleep(0.01)

        server_thread = threading.Thread(target=pump_server, daemon=True)
        server_thread.start()
        client = ZMQExecutionClient(
            port=port,
            host="127.0.0.1",
            persistent=True,
        )
        try:
            assert client.connect(timeout=5.0)
            catalog_request_started = time.perf_counter()
            catalog = client.get_function_catalog(
                FunctionCatalogControlRequest(compact_signatures=True)
            )
            catalog_request_seconds = time.perf_counter() - catalog_request_started
        finally:
            client.disconnect()
            server.stop()
            server_thread.join(timeout=2.0)

        assert catalog.items
        assert LibraryRegistryBase.__registry__._config.discovery_package == (
            "openhcs.processing.backends.lib_registry"
        )
        assert {"openhcs", "skimage"}.issubset(LibraryRegistryBase.__registry__)
        for declaration in LibraryRegistryBase.__registry__.values():
            resolved = importlib.import_module(declaration.__module__)
            for owner_name in declaration.__qualname__.split("."):
                resolved = getattr(resolved, owner_name)
            assert resolved is declaration

        print(
            json.dumps(
                {
                    "catalog_size": len(catalog.items),
                    "catalog_request_seconds": catalog_request_seconds,
                    "library_roots": sorted(LibraryRegistryBase.__registry__),
                    "startup_seconds": startup_seconds,
                },
                sort_keys=True,
            )
        )
        """
    )

    completed = subprocess.run(
        (sys.executable, "-c", script),
        cwd=repository_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=90,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["catalog_size"] > 0
    # Cold discovery is deliberately isolated from the request thread.  The typed
    # pending protocol keeps the endpoint responsive until preparation completes;
    # the unit-level protocol tests assert that polling contract directly.  Server
    # startup itself must remain independent of discovery latency.
    assert result["startup_seconds"] < 5.0
    assert {"openhcs", "skimage"}.issubset(result["library_roots"])
