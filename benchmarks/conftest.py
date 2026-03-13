"""Shared fixtures for benchmark suite."""

import os
import sys
import random

import pytest

# Ensure the project root is on sys.path so `src.*` imports resolve.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.trees.merkle_prefix_tree import MerklePrefixTree
from src.trees.red_black_tree import RedBlackTree, RBNode, Color


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALUE = b"\xab" * 8


def bulk_insert_rbt(tree, keys):
    """Insert *keys* into a RedBlackTree without triggering an O(N) Merkle
    rebuild after every key.  Calls ``get_root_hash()`` once at the end."""
    for k in keys:
        node = RBNode(key=k, value=VALUE, color=Color.RED)
        node.left = tree._nil
        node.right = tree._nil
        node.parent = tree._nil
        tree._bst_insert(node)
        tree._insert_fixup(node)
        tree._rehash_node(node)
        tree._size += 1
        tree._mark_dirty()
    tree.get_root_hash()


# ---------------------------------------------------------------------------
# Parametrised fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(params=[100, 1_000, 10_000], ids=["N=100", "N=1K", "N=10K"])
def tree_size(request):
    """Number of keys to populate a tree with."""
    return request.param


@pytest.fixture(
    params=["merkle_prefix", "red_black"],
    ids=["MerklePrefixTree", "RedBlackTree"],
)
def tree_factory(request):
    """Return a zero-argument factory that creates a fresh, empty tree."""
    factories = {
        "merkle_prefix": MerklePrefixTree,
        "red_black": RedBlackTree,
    }
    return factories[request.param]


@pytest.fixture()
def populated_trees(tree_size):
    """Create both tree types populated with *tree_size* random keys.

    Returns a dict:
        {
            "merkle_prefix": <MerklePrefixTree>,
            "red_black": <RedBlackTree>,
            "keys": [list of inserted keys],
            "size": tree_size,
        }
    """
    rng = random.Random(42)
    keys = rng.sample(range(1, tree_size * 100), tree_size)

    mpt = MerklePrefixTree()
    for k in keys:
        mpt.insert(k, VALUE)

    rbt = RedBlackTree()
    bulk_insert_rbt(rbt, keys)

    return {
        "merkle_prefix": mpt,
        "red_black": rbt,
        "keys": keys,
        "size": tree_size,
    }
