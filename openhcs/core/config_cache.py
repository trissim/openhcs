"""Typed persistence for immutable OpenHCS configuration roots."""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Callable,
    Generic,
    TypeVar,
)

from openhcs.core.config import GlobalPipelineConfig
from openhcs.core.config_document import ConfigDocumentAuthority

logger = logging.getLogger(__name__)

ConfigT = TypeVar("ConfigT")


@dataclass(frozen=True, slots=True)
class ConfigCacheSpec(Generic[ConfigT]):
    """Persistence identity and load hook for one authoritative config root."""

    config_type: type[ConfigT]
    cache_file: Path
    on_loaded: Callable[[ConfigT], None] | None = None


@dataclass(slots=True)
class ConfigCache(Generic[ConfigT]):
    """Typed cache bound to exactly one configuration declaration."""

    spec: ConfigCacheSpec[ConfigT]

    async def load(self) -> ConfigT | None:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, load_config_sync, self.spec)

    async def save(self, config: ConfigT) -> bool:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, save_config_sync, config, self.spec)

    async def clear(self) -> bool:
        try:
            self.spec.cache_file.unlink(missing_ok=True)
            logger.info("Cleared config cache: %s", self.spec.cache_file)
            return True
        except OSError as error:
            logger.error("Failed to clear config cache: %s", error)
            return False


def load_config_sync(spec: ConfigCacheSpec[ConfigT]) -> ConfigT | None:
    """Load one canonical config document for the exact declared root type."""

    try:
        if not spec.cache_file.exists():
            return None
        cached_config = ConfigDocumentAuthority.from_source(
            spec.cache_file.read_text(encoding="utf-8"),
            expected_config_type=spec.config_type,
        )
        if type(cached_config) is not spec.config_type:
            logger.warning(
                "Ignoring stale %s cache payload of type %s",
                spec.config_type.__name__,
                type(cached_config).__name__,
            )
            return None
        if spec.on_loaded is not None:
            spec.on_loaded(cached_config)
        logger.debug("Loaded %s from %s", spec.config_type.__name__, spec.cache_file)
        return cached_config
    except Exception as error:
        logger.warning("Failed to load config cache %s: %s", spec.cache_file, error)
        return None


def save_config_sync(config: ConfigT, spec: ConfigCacheSpec[ConfigT]) -> bool:
    """Persist one configuration after exact root-type validation."""

    if type(config) is not spec.config_type:
        raise TypeError(
            f"Cache {spec.cache_file.name} requires {spec.config_type.__name__}; "
            f"got {type(config).__name__}."
        )
    try:
        spec.cache_file.parent.mkdir(parents=True, exist_ok=True)
        payload = ConfigDocumentAuthority.render(
            config,
            expected_config_type=spec.config_type,
        ).encode("utf-8")
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{spec.cache_file.name}.",
            suffix=".tmp",
            dir=spec.cache_file.parent,
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, spec.cache_file)
            _fsync_directory(spec.cache_file.parent)
        finally:
            temporary_path.unlink(missing_ok=True)
        logger.debug("Saved %s to %s", spec.config_type.__name__, spec.cache_file)
        return True
    except Exception as error:
        logger.error("Failed to save config cache %s: %s", spec.cache_file, error)
        return False


def _fsync_directory(directory: Path) -> None:
    """Durably publish a completed atomic replace where the platform permits."""

    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _establish_global_pipeline_config(config: GlobalPipelineConfig) -> None:
    from objectstate.lazy_factory import ensure_global_config_context

    ensure_global_config_context(GlobalPipelineConfig, config)


def global_pipeline_config_cache_spec() -> ConfigCacheSpec[GlobalPipelineConfig]:
    from openhcs.core.xdg_paths import get_config_file_path

    return ConfigCacheSpec(
        config_type=GlobalPipelineConfig,
        cache_file=get_config_file_path("global_config.config"),
        on_loaded=_establish_global_pipeline_config,
    )


def global_pipeline_config_cache() -> ConfigCache[GlobalPipelineConfig]:
    return ConfigCache(
        spec=global_pipeline_config_cache_spec(),
    )


async def load_cached_global_config() -> GlobalPipelineConfig:
    cached_config = await global_pipeline_config_cache().load()
    if cached_config is not None:
        logger.info("Using cached global configuration")
        return cached_config
    logger.info("Using default global configuration")
    default_config = GlobalPipelineConfig()
    _establish_global_pipeline_config(default_config)
    return default_config


def load_cached_global_config_sync() -> GlobalPipelineConfig:
    cached_config = load_config_sync(global_pipeline_config_cache_spec())
    if cached_config is not None:
        logger.info("Using cached global configuration")
        return cached_config
    logger.info("Using default global configuration")
    default_config = GlobalPipelineConfig()
    _establish_global_pipeline_config(default_config)
    return default_config


def save_global_config_sync(config: GlobalPipelineConfig) -> bool:
    return save_config_sync(config, global_pipeline_config_cache_spec())
