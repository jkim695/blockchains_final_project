"""Memory usage measurements for both tree types at various sizes.

Uses tracemalloc to measure peak memory. These are regular pytest tests
(not pytest-benchmark tests) that print and assert memory measurements.
"""

import random
import tracemalloc

from src.trees.merkle_prefix_tree import MerklePrefixTree
from src.trees.red_black_tree import RedBlackTree


VALUE = b"\xab" * 8

SIZES = [100, 1_000, 10_000]


def _bulk_insert(tree, keys):
    """Insert keys into *tree*, deferring expensive root-hash rebuilds.

    For RedBlackTree, each insert() call triggers an O(N) Merkle rebuild.
    To keep the memory benchmark practical we suppress that by marking the
    tree as not-dirty before the root-hash call on all but the final insert.
    MerklePrefixTree is unaffected (its insert is already O(log N)).
    """
    if isinstance(tree, RedBlackTree):
        for k in keys:
            # Perform the structural insert + fixup + per-node rehash,
            # but skip the expensive full-Merkle rebuild each time.
            existing = tree._find_node(k)
            if existing is not tree._nil:
                existing.value = VALUE
                tree._rehash_node(existing)
                tree._mark_dirty()
                continue
            from src.trees.red_black_tree import RBNode, Color
            node = RBNode(key=k, value=VALUE, color=Color.RED)
            node.left = tree._nil
            node.right = tree._nil
            node.parent = tree._nil
            tree._bst_insert(node)
            tree._insert_fixup(node)
            tree._rehash_node(node)
            tree._size += 1
            tree._mark_dirty()
        # Compute the root hash once at the end.
        tree.get_root_hash()
    else:
        for k in keys:
            tree.insert(k, VALUE)


def _measure_peak_memory(tree_cls, n, seed=42):
    """Insert *n* keys into *tree_cls* and return peak memory in bytes."""
    rng = random.Random(seed)
    keys = rng.sample(range(1, n * 100), n)

    tracemalloc.start()
    tree = tree_cls()
    _bulk_insert(tree, keys)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


# ---------------------------------------------------------------------------
# MerklePrefixTree memory
# ---------------------------------------------------------------------------

def test_memory_merkle_prefix():
    """Measure peak memory for MerklePrefixTree at various sizes."""
    prev_peak = 0
    for n in SIZES:
        peak = _measure_peak_memory(MerklePrefixTree, n)
        peak_kb = peak / 1024
        print(f"  MerklePrefixTree  N={n:>6}  peak={peak_kb:>10.1f} KB")
        # Sanity: memory should grow with size.
        assert peak > prev_peak, (
            f"Expected memory to grow: N={n} peak={peak} <= prev={prev_peak}"
        )
        prev_peak = peak


# ---------------------------------------------------------------------------
# RedBlackTree memory
# ---------------------------------------------------------------------------

def test_memory_red_black():
    """Measure peak memory for RedBlackTree at various sizes."""
    prev_peak = 0
    for n in SIZES:
        peak = _measure_peak_memory(RedBlackTree, n)
        peak_kb = peak / 1024
        print(f"  RedBlackTree      N={n:>6}  peak={peak_kb:>10.1f} KB")
        assert peak > prev_peak, (
            f"Expected memory to grow: N={n} peak={peak} <= prev={prev_peak}"
        )
        prev_peak = peak


# ---------------------------------------------------------------------------
# Comparative summary
# ---------------------------------------------------------------------------

def test_memory_comparison():
    """Print a side-by-side comparison of peak memory for both tree types."""
    print()
    print(f"  {'N':>6}  {'MerklePrefixTree':>20}  {'RedBlackTree':>20}  {'Ratio (RBT/MPT)':>16}")
    print(f"  {'---':>6}  {'---':>20}  {'---':>20}  {'---':>16}")
    for n in SIZES:
        mpt_peak = _measure_peak_memory(MerklePrefixTree, n)
        rbt_peak = _measure_peak_memory(RedBlackTree, n)
        ratio = rbt_peak / mpt_peak if mpt_peak > 0 else float("inf")
        print(
            f"  {n:>6}  {mpt_peak / 1024:>18.1f} KB"
            f"  {rbt_peak / 1024:>18.1f} KB"
            f"  {ratio:>14.2f}x"
        )
