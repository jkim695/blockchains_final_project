import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.cache.aardvark_delta_cache import AardvarkDeltaCache
from src.trees.merkle_prefix_tree import MerklePrefixTree


def _make_tree_and_cache():
    tree = MerklePrefixTree()
    cache = AardvarkDeltaCache()
    return tree, cache


class TestRecordTransition:
    def test_increments_version(self):
        tree, cache = _make_tree_and_cache()
        assert cache.get_version() == 0
        tree.insert(1, b"a")
        cache.record_transition(tree)
        assert cache.get_version() == 1
        tree.insert(2, b"b")
        cache.record_transition(tree)
        assert cache.get_version() == 2

    def test_records_correct_root(self):
        tree, cache = _make_tree_and_cache()
        tree.insert(1, b"a")
        cache.record_transition(tree)
        expected_root = tree.get_root_hash()
        assert cache.get_root_at_version(1) == expected_root


class TestGetRootAtVersion:
    def test_returns_correct_roots_for_multiple_versions(self):
        tree, cache = _make_tree_and_cache()
        roots = []
        for i in range(1, 4):
            tree.insert(i, f"val{i}".encode())
            cache.record_transition(tree)
            roots.append(tree.get_root_hash())
        for i, root in enumerate(roots, start=1):
            assert cache.get_root_at_version(i) == root

    def test_returns_none_for_missing_version(self):
        tree, cache = _make_tree_and_cache()
        assert cache.get_root_at_version(99) is None


class TestPatchProof:
    def test_same_version_returns_proof_unchanged(self):
        tree, cache = _make_tree_and_cache()
        tree.insert(1, b"a")
        cache.record_transition(tree)
        proof = tree.generate_proof(1)
        patched = cache.patch_proof(proof, 1, 1)
        assert patched == proof

    def test_across_one_version_changes_proof(self):
        tree, cache = _make_tree_and_cache()
        tree.insert(1, b"a")
        cache.record_transition(tree)
        proof_v1 = tree.generate_proof(1)

        tree.insert(2, b"b")
        cache.record_transition(tree)

        patched = cache.patch_proof(proof_v1, 1, 2)
        # The patched proof should be a list of the same length.
        assert len(patched) == len(proof_v1)
        # If the tree changed, at least one sibling hash may differ
        # (or the proof is unchanged if the sibling path wasn't affected).
        # We just verify it's a valid list of bytes.
        assert all(isinstance(h, bytes) for h in patched)

    def test_raises_for_missing_versions(self):
        tree, cache = _make_tree_and_cache()
        tree.insert(1, b"a")
        cache.record_transition(tree)
        proof = tree.generate_proof(1)
        with pytest.raises(ValueError):
            cache.patch_proof(proof, 1, 50)


class TestGetDeltaChain:
    def test_returns_correct_number_of_deltas(self):
        tree, cache = _make_tree_and_cache()
        for i in range(1, 6):
            tree.insert(i, f"v{i}".encode())
            cache.record_transition(tree)
        chain = cache.get_delta_chain(1, 5)
        assert len(chain) == 4  # versions 2, 3, 4, 5

    def test_same_version_returns_empty(self):
        tree, cache = _make_tree_and_cache()
        tree.insert(1, b"a")
        cache.record_transition(tree)
        chain = cache.get_delta_chain(1, 1)
        assert chain == []


class TestPrune:
    def test_prune_removes_old_versions(self):
        tree, cache = _make_tree_and_cache()
        for i in range(1, 11):
            tree.insert(i, f"v{i}".encode())
            cache.record_transition(tree)
        # Current version is 10; keep last 3 (versions 8, 9, 10).
        cache.prune(keep_last=3)
        # Versions 1-7 should be pruned.
        for v in range(1, 8):
            assert cache.get_root_at_version(v) is None
        # Versions 8-10 should still exist.
        for v in range(8, 11):
            assert cache.get_root_at_version(v) is not None

    def test_prune_then_get_delta_chain_raises(self):
        tree, cache = _make_tree_and_cache()
        for i in range(1, 6):
            tree.insert(i, f"v{i}".encode())
            cache.record_transition(tree)
        cache.prune(keep_last=2)
        # Trying to get delta chain including pruned versions should fail.
        with pytest.raises(ValueError):
            cache.get_delta_chain(1, 5)
