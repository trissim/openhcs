from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap


def test_ui_bridge_nominal_result_registries_round_trip_cache_keys(
    tmp_path: Path,
) -> None:
    repository_root = Path(__file__).parents[3]
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join(
                (
                    str(repository_root),
                    str(repository_root / "external" / "metaclass-registry" / "src"),
                )
            ),
            "QT_QPA_PLATFORM": "offscreen",
            "XDG_CACHE_HOME": str(tmp_path / "cache"),
        }
    )
    script = textwrap.dedent(
        """
        import json

        from metaclass_registry.cache import (
            CacheKeyCodec,
            RegistryCacheManager,
            deserialize_plugin_class,
            serialize_plugin_class,
        )
        from openhcs.pyqt_gui.services.ui_agent_bridge import (
            UiBridgeAcceptedMutationResultBuilder,
            UiBridgeMutationOutcomeProjector,
        )

        registry_keys = {}
        for root in (
            UiBridgeMutationOutcomeProjector,
            UiBridgeAcceptedMutationResultBuilder,
        ):
            registry = dict(root.__registry__)
            assert registry
            for key, strategy_type in registry.items():
                assert isinstance(key, str)
                assert key == strategy_type.value_type_label
                assert CacheKeyCodec.decode(CacheKeyCodec.encode(key)) == key
                assert (
                    type(root.for_result_type(strategy_type.value_type))
                    is strategy_type
                )

            cache = RegistryCacheManager(
                cache_name=f"test_{root.__name__}",
                version_getter=lambda: "test",
                serializer=serialize_plugin_class,
                deserializer=deserialize_plugin_class,
            )
            cache.save_cache(registry)
            assert cache.load_cache() == registry
            registry_keys[root.__name__] = tuple(registry)

        print(json.dumps(registry_keys, sort_keys=True))
        """
    )

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
    assert "Unsupported registry cache key type" not in completed.stderr
    registry_keys = json.loads(completed.stdout)
    assert all(registry_keys.values())
    assert all(isinstance(key, str) for keys in registry_keys.values() for key in keys)
