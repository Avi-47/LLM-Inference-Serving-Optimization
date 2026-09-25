import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from prefix_caching import PrefixCache, invalidate_prefix_cache


def entry(target="target-v1", draft="draft-v1", value=None):
    return {"target_version": target, "draft_version": draft, "value": value}


def test_identical_prefix_hits_and_different_prefix_misses():
    cache = PrefixCache(max_entries=4)
    cache.put((1, 2, 3), entry())

    assert cache.get((1, 2, 3), target_version="target-v1") ["value"] is None
    assert cache.get((1, 2, 4), target_version="target-v1") is None
    assert cache.metrics()["cache_hits"] == 1
    assert cache.metrics()["cache_misses"] == 1


def test_version_mismatch_invalidates_and_recomputes():
    cache = PrefixCache(max_entries=4)
    cache.put((1, 2, 3), entry(target="old"))

    assert cache.get((1, 2, 3), target_version="new") is None
    cache.put((1, 2, 3), entry(target="new", value="recomputed"))
    assert cache.get((1, 2, 3), target_version="new")["value"] == "recomputed"


def test_lru_eviction_and_access_updates_recency():
    cache = PrefixCache(max_entries=2)
    cache.put((1,), entry(value=1))
    cache.put((2,), entry(value=2))
    assert cache.get((1,), target_version="target-v1")["value"] == 1
    cache.put((3,), entry(value=3))

    assert cache.get((2,), target_version="target-v1") is None
    assert cache.get((1,), target_version="target-v1")["value"] == 1
    assert cache.metrics()["evictions"] == 1


def test_clear_and_specific_invalidation_release_entries():
    cache = PrefixCache(max_entries=4)
    cache.put((1,), entry())
    cache.put((2,), entry())
    cache.invalidate_prefix_cache(prefix_ids=(1,))
    assert cache.get((1,), target_version="target-v1") is None
    assert len(cache) == 1

    cache.invalidate_prefix_cache(clear=True)
    assert len(cache) == 0


def test_cached_first_token_matches_baseline_value():
    cache = PrefixCache(max_entries=2)
    baseline_tokens = [41, 42, 43]
    cache.put((1, 2, 3), entry(value=baseline_tokens[0]))

    cached_entry = cache.get((1, 2, 3), target_version="target-v1")
    assert [cached_entry["value"]] == [baseline_tokens[0]]


def test_module_invalidation_can_remove_old_version_entries():
    import prefix_caching

    prefix_caching.prefix_cache_pool.invalidate_prefix_cache(clear=True)
    prefix_caching.prefix_cache_pool.put((9,), entry(target="old"))
    prefix_caching.prefix_cache_pool.put((10,), entry(target="current"))
    invalidate_prefix_cache(target_version="old")

    assert (9,) not in prefix_caching.prefix_cache_pool
    assert (10,) in prefix_caching.prefix_cache_pool
    invalidate_prefix_cache(clear=True)