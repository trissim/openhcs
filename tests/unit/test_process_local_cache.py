from openhcs.core.process_local_cache import (
    IdentityBoundProcessCache,
    RegisteredProcessLocalBoundedCache,
    clear_registered_process_local_caches,
)


def test_registered_process_local_cache_cleanup_clears_cache_instance():
    class TestRegisteredProcessLocalCleanupCache(
        RegisteredProcessLocalBoundedCache[str, str]
    ):
        max_entries = 2

    cache = TestRegisteredProcessLocalCleanupCache.process_cache()
    cache.store_value("source", "payload")

    assert cache.cached_value("source") == "payload"

    clear_registered_process_local_caches()

    assert cache.cached_value("source") is None


def test_identity_bound_process_cache_cleanup_clears_cache_instance():
    class TestIdentityBoundProcessLocalCleanupCache(IdentityBoundProcessCache):
        registry_key = "test_identity_bound_process_local_cleanup"

    owner = object()
    cache = TestIdentityBoundProcessLocalCleanupCache.process_cache()
    cache.put_bound(owner, "payload")

    assert cache.get_bound(owner) == "payload"

    clear_registered_process_local_caches()

    assert cache.get_bound(owner) is None
