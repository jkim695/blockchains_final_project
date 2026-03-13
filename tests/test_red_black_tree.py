import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.trees.red_black_tree import RedBlackTree, EMPTY_HASH


class TestInsertAndSearch:
    def test_single_key(self):
        tree = RedBlackTree()
        tree.insert(1, b"value1")
        assert tree.search(1) == b"value1"

    def test_multiple_keys(self):
        tree = RedBlackTree()
        tree.insert(10, b"ten")
        tree.insert(20, b"twenty")
        tree.insert(30, b"thirty")
        assert tree.search(10) == b"ten"
        assert tree.search(20) == b"twenty"
        assert tree.search(30) == b"thirty"

    def test_overwrite(self):
        tree = RedBlackTree()
        tree.insert(5, b"old")
        tree.insert(5, b"new")
        assert tree.search(5) == b"new"

    def test_search_missing_key_returns_none(self):
        tree = RedBlackTree()
        tree.insert(1, b"value")
        assert tree.search(999) is None

    def test_search_empty_tree(self):
        tree = RedBlackTree()
        assert tree.search(1) is None


class TestDelete:
    def test_delete_existing_key(self):
        tree = RedBlackTree()
        tree.insert(42, b"data")
        assert tree.search(42) == b"data"
        tree.delete(42)
        assert tree.search(42) is None

    def test_delete_non_existing_key(self):
        tree = RedBlackTree()
        tree.insert(1, b"value")
        root_before = tree.get_root_hash()
        tree.delete(999)
        root_after = tree.get_root_hash()
        assert root_before == root_after

    def test_delete_then_reinsert(self):
        tree = RedBlackTree()
        tree.insert(7, b"first")
        tree.delete(7)
        tree.insert(7, b"second")
        assert tree.search(7) == b"second"

    def test_delete_multiple(self):
        tree = RedBlackTree()
        for i in range(10):
            tree.insert(i, f"val{i}".encode())
        for i in range(5):
            tree.delete(i)
        for i in range(5):
            assert tree.search(i) is None
        for i in range(5, 10):
            assert tree.search(i) == f"val{i}".encode()


class TestRootHash:
    def test_determinism_same_ops_same_root(self):
        tree1 = RedBlackTree()
        tree2 = RedBlackTree()
        for k in [10, 20, 30]:
            tree1.insert(k, b"val")
            tree2.insert(k, b"val")
        assert tree1.get_root_hash() == tree2.get_root_hash()

    def test_empty_tree_root(self):
        tree = RedBlackTree()
        assert tree.get_root_hash() == EMPTY_HASH

    def test_root_changes_on_insert(self):
        tree = RedBlackTree()
        root0 = tree.get_root_hash()
        tree.insert(1, b"a")
        root1 = tree.get_root_hash()
        assert root0 != root1

    def test_root_changes_on_delete(self):
        tree = RedBlackTree()
        tree.insert(1, b"a")
        tree.insert(2, b"b")
        root_before = tree.get_root_hash()
        tree.delete(1)
        root_after = tree.get_root_hash()
        assert root_before != root_after

    def test_root_returns_to_empty_after_all_deletes(self):
        tree = RedBlackTree()
        tree.insert(1, b"a")
        tree.delete(1)
        assert tree.get_root_hash() == EMPTY_HASH


class TestGenerateProof:
    def test_proof_non_empty_for_existing_key(self):
        tree = RedBlackTree()
        tree.insert(100, b"data")
        tree.insert(200, b"data2")
        proof = tree.generate_proof(100)
        assert len(proof) > 0

    def test_proof_empty_for_missing_key(self):
        tree = RedBlackTree()
        tree.insert(1, b"data")
        proof = tree.generate_proof(999)
        assert proof == []

    def test_proof_empty_on_empty_tree(self):
        tree = RedBlackTree()
        proof = tree.generate_proof(1)
        assert proof == []


class TestGetChangedNodes:
    def test_returns_changes_and_clears(self):
        tree = RedBlackTree()
        tree.insert(1, b"a")
        changes = tree.get_changed_nodes()
        assert len(changes) > 0
        # Second call should be empty since the log was cleared.
        assert tree.get_changed_nodes() == []

    def test_multiple_inserts_accumulate_changes(self):
        tree = RedBlackTree()
        tree.insert(1, b"a")
        tree.insert(2, b"b")
        changes = tree.get_changed_nodes()
        assert len(changes) >= 2
