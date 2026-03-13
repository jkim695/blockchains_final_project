"""Benchmark proof generation for both tree types at various sizes.

This is the key benchmark: MerklePrefixTree.generate_proof is O(log N)
while RedBlackTree.generate_proof is O(N) due to full in-order traversal.
"""

import random

from src.trees.merkle_prefix_tree import MerklePrefixTree
from src.trees.red_black_tree import RedBlackTree, RBNode, Color


VALUE = b"\xab" * 8


def _build_tree(tree_cls, n, seed=42):
    """Build a tree of *tree_cls* populated with *n* random keys.

    For RedBlackTree, uses a bulk-insert strategy that defers the expensive
    O(N) Merkle rebuild until all keys are inserted, avoiding O(N^2) setup.

    Returns ``(tree, keys)`` where *keys* is the list of inserted keys.
    """
    rng = random.Random(seed)
    keys = rng.sample(range(1, n * 100), n)
    tree = tree_cls()

    if isinstance(tree, RedBlackTree):
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
        # Build the Merkle commitment once.
        tree.get_root_hash()
    else:
        for k in keys:
            tree.insert(k, VALUE)

    return tree, keys


# ---------------------------------------------------------------------------
# MerklePrefixTree proof generation  -- expected O(log N)
# ---------------------------------------------------------------------------

def test_proof_generation_merkle_prefix(benchmark, tree_size):
    """Generate a proof for a random existing key in a MerklePrefixTree."""
    tree, keys = _build_tree(MerklePrefixTree, tree_size)
    rng = random.Random(99)
    target_key = rng.choice(keys)

    benchmark(tree.generate_proof, target_key)


# ---------------------------------------------------------------------------
# RedBlackTree proof generation  -- expected O(N)
# ---------------------------------------------------------------------------

def test_proof_generation_red_black(benchmark, tree_size):
    """Generate a proof for a random existing key in a RedBlackTree."""
    tree, keys = _build_tree(RedBlackTree, tree_size)
    rng = random.Random(99)
    target_key = rng.choice(keys)

    benchmark(tree.generate_proof, target_key)
