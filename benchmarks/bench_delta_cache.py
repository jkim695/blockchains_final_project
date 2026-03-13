"""Benchmark delta-cache proof patching vs full proof regeneration.

For a tree of size N:
  1. Build the tree, generate a proof at version V.
  2. Apply D=10 more mutations while recording deltas.
  3. Compare:  (a) patch stale proof via AardvarkDeltaCache
               (b) regenerate the proof from scratch.
Patching should be significantly faster than full regeneration.
"""

import random

from src.trees.merkle_prefix_tree import MerklePrefixTree
from src.cache.aardvark_delta_cache import AardvarkDeltaCache


VALUE = b"\xab" * 8
NUM_EXTRA_MUTATIONS = 10


def _setup(n, seed=42):
    """Build tree of size *n*, generate a stale proof, apply extra mutations.

    Returns ``(cache, stale_proof, proof_version, current_version, tree, target_key)``.
    """
    rng = random.Random(seed)
    keys = rng.sample(range(1, n * 100), n)

    tree = MerklePrefixTree()
    cache = AardvarkDeltaCache()

    # Populate tree, recording each transition.
    for k in keys:
        tree.insert(k, VALUE)
        cache.record_transition(tree)

    # Pick a key and generate a proof at the current version.
    target_key = rng.choice(keys)
    stale_proof = tree.generate_proof(target_key)
    proof_version = cache.get_version()

    # Apply D more mutations (inserts of new keys).
    extra_keys = rng.sample(range(n * 100, n * 200), NUM_EXTRA_MUTATIONS)
    for k in extra_keys:
        tree.insert(k, VALUE)
        cache.record_transition(tree)

    current_version = cache.get_version()
    return cache, stale_proof, proof_version, current_version, tree, target_key


# ---------------------------------------------------------------------------
# (a) Patch stale proof via delta cache
# ---------------------------------------------------------------------------

def test_delta_cache_patch(benchmark, tree_size):
    """Patch a stale proof using AardvarkDeltaCache.patch_proof."""
    cache, stale_proof, proof_ver, cur_ver, _tree, _key = _setup(tree_size)

    def do_patch():
        cache.patch_proof(stale_proof, proof_ver, cur_ver)

    benchmark(do_patch)


# ---------------------------------------------------------------------------
# (b) Full proof regeneration from scratch
# ---------------------------------------------------------------------------

def test_full_proof_regeneration(benchmark, tree_size):
    """Regenerate the proof from scratch on the updated tree."""
    _cache, _stale_proof, _pv, _cv, tree, target_key = _setup(tree_size)

    benchmark(tree.generate_proof, target_key)
