from __future__ import annotations

from dataclasses import dataclass

import dill as pickle
import pytest

from openhcs.core.config_cache import (
    ConfigCacheSpec,
    load_config_sync,
    save_config_sync,
)


@dataclass(frozen=True)
class _CurrentNested:
    value: int = 0
    added: str = "default"


@dataclass(frozen=True)
class _CurrentRoot:
    nested: "_CurrentNested"
    added: int = 7


@dataclass(frozen=True)
class _OtherRoot:
    value: int = 1


def test_typed_cache_round_trip_and_load_hook(tmp_path) -> None:
    cache_file = tmp_path / "config.cache"
    loaded: list[_CurrentRoot] = []
    spec = ConfigCacheSpec(
        config_type=_CurrentRoot,
        cache_file=cache_file,
        on_loaded=loaded.append,
    )
    config = _CurrentRoot(nested=_CurrentNested(value=4), added=9)

    assert save_config_sync(config, spec) is True
    restored = load_config_sync(spec)

    assert restored == config
    assert type(restored) is _CurrentRoot
    assert type(restored.nested) is _CurrentNested
    assert loaded == [restored]
    assert cache_file.read_text(encoding="utf-8").startswith(
        "# OpenHCS configuration"
    )


def test_cache_rejects_stale_root_type_without_compatibility_migration(
    tmp_path,
) -> None:
    cache_file = tmp_path / "config.cache"
    old_spec = ConfigCacheSpec(config_type=_OtherRoot, cache_file=cache_file)
    current_spec = ConfigCacheSpec(config_type=_CurrentRoot, cache_file=cache_file)
    assert save_config_sync(_OtherRoot(), old_spec) is True

    restored = load_config_sync(current_spec)

    assert restored is None


def test_cache_ignores_legacy_pickle_payload(
    tmp_path,
) -> None:
    cache_file = tmp_path / "config.cache"
    spec = ConfigCacheSpec(config_type=_CurrentRoot, cache_file=cache_file)
    with cache_file.open("wb") as stream:
        pickle.dump(_CurrentRoot(nested=_CurrentNested(value=4)), stream)

    restored = load_config_sync(spec)

    assert restored is None


def test_cache_rejects_wrong_root_type_before_writing(tmp_path) -> None:
    spec = ConfigCacheSpec(
        config_type=_CurrentRoot,
        cache_file=tmp_path / "config.cache",
    )

    with pytest.raises(TypeError, match="requires _CurrentRoot"):
        save_config_sync(_OtherRoot(), spec)

    assert not spec.cache_file.exists()


def test_failed_atomic_replace_preserves_previous_cache(
    tmp_path,
    monkeypatch,
) -> None:
    cache_file = tmp_path / "config.cache"
    spec = ConfigCacheSpec(config_type=_CurrentRoot, cache_file=cache_file)
    original = _CurrentRoot(nested=_CurrentNested(value=1))
    replacement = _CurrentRoot(nested=_CurrentNested(value=2))
    assert save_config_sync(original, spec) is True

    monkeypatch.setattr(
        "openhcs.core.config_cache.os.replace",
        lambda source, target: (_ for _ in ()).throw(
            OSError("replace failed")
        ),
    )

    assert save_config_sync(replacement, spec) is False
    assert load_config_sync(spec) == original
    assert tuple(tmp_path.glob(".config.cache.*.tmp")) == ()
