"""Benchmark insert throughput for both tree types at various sizes.

For MerklePrefixTree, insert() is O(log N) per key (includes Merkle rehash).
For RedBlackTree, the structural insert is O(log N) but the full insert()
triggers an O(N) Merkle rebuild each time.  To keep the benchmark practical,
we measure the structural insert (BST insert + fixup + per-node rehash) for
RedBlackTree and a single final get_root_hash() separately.
"""

import random

from src.trees.merkle_prefix_tree import MerklePrefixTree
from src.trees.red_black_tree import RedBlackTree, RBNode, Color


VALUE = b"\xab" * 8


def _generate_keys(n, seed=42):
    """Return *n* unique random integer keys."""
    rng = random.Random(seed)
    return rng.sample(range(1, n * 100), n)


# ---------------------------------------------------------------------------
# MerklePrefixTree insert  -- O(log N) per key
# ---------------------------------------------------------------------------

def test_insert_merkle_prefix(benchmark, tree_size):
    """Insert *tree_size* keys into a fresh MerklePrefixTree."""
    keys = _generate_keys(tree_size)

    def do_inserts():
        tree = MerklePrefixTree()
        for k in keys:
            tree.insert(k, VALUE)

    benchmark(do_inserts)


# ---------------------------------------------------------------------------
# RedBlackTree structural insert  -- O(log N) per key
# ---------------------------------------------------------------------------

def test_insert_red_black(benchmark, tree_size):
    """Insert *tree_size* keys into a fresh RedBlackTree (structural cost).

    Measures the BST insert + RB fixup + per-node rehash without the O(N)
    full Merkle rebuild that ``insert()`` normally triggers.  A single
    ``get_root_hash()`` is called at the end to finalize the commitment.
    """
    keys = _generate_keys(tree_size)

    def do_inserts():
        tree = RedBlackTree()
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

    benchmark(do_inserts)
