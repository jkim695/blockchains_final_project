"""Hash-augmented Red-Black Tree implementing the StateTree interface.

This implementation is designed for benchmarking against a Merkle-Prefix Tree.
The RBT maintains a standard red-black tree for O(log N) insert/delete/search,
but its root-hash computation and proof generation are intentionally O(N) because
the tree must serialize all leaves in-order and build a temporary Merkle tree
each time a commitment or proof is needed.
"""

from enum import Enum
from typing import List, Optional, Tuple

from src.trees.base import StateTree
from src.utils.hashing import hash_internal, hash_leaf


EMPTY_HASH = b"\x00" * 32


class Color(Enum):
    RED = 0
    BLACK = 1


class RBNode:
    """A node in the Red-Black Tree."""

    __slots__ = ("key", "value", "left", "right", "parent", "color", "hash")

    def __init__(
        self,
        key: int = 0,
        value: bytes = b"",
        color: Color = Color.BLACK,
    ) -> None:
        self.key: int = key
        self.value: bytes = value
        self.color: Color = color
        self.left: "RBNode" = self  # will be overwritten; sentinel points to itself
        self.right: "RBNode" = self
        self.parent: "RBNode" = self
        self.hash: bytes = EMPTY_HASH


class RedBlackTree(StateTree):
    """Red-Black Tree with hash augmentation.

    Structural operations (insert, delete, search) are O(log N).
    Root-hash computation and proof generation are O(N) — they build a
    temporary Merkle tree over the in-order serialization of all nodes.
    """

    def __init__(self) -> None:
        self._nil: RBNode = RBNode(color=Color.BLACK)
        self._nil.left = self._nil
        self._nil.right = self._nil
        self._nil.parent = self._nil
        self._nil.hash = EMPTY_HASH

        self._root: RBNode = self._nil
        self._size: int = 0

        # Cached Merkle tree state built from the in-order traversal.
        self._cached_root_hash: bytes = EMPTY_HASH
        self._cached_merkle_hashes: List[bytes] = []
        self._dirty: bool = False

        # Change tracking: (old_hash, new_hash) for Merkle-tree internal nodes.
        self._changed_nodes: List[Tuple[bytes, bytes]] = []

    # ------------------------------------------------------------------
    # StateTree interface
    # ------------------------------------------------------------------

    def insert(self, key: int, value: bytes) -> bytes:
        """Insert or update *key* with *value*. Returns the new root hash."""
        existing = self._find_node(key)
        if existing is not self._nil:
            existing.value = value
            self._rehash_node(existing)
            self._mark_dirty()
            return self.get_root_hash()

        node = RBNode(key=key, value=value, color=Color.RED)
        node.left = self._nil
        node.right = self._nil
        node.parent = self._nil
        self._bst_insert(node)
        self._insert_fixup(node)
        self._rehash_node(node)
        self._size += 1
        self._mark_dirty()
        return self.get_root_hash()

    def delete(self, key: int) -> bytes:
        """Delete *key*. Returns the new root hash."""
        node = self._find_node(key)
        if node is self._nil:
            return self.get_root_hash()
        self._delete_node(node)
        self._size -= 1
        self._mark_dirty()
        return self.get_root_hash()

    def search(self, key: int) -> Optional[bytes]:
        """Return the value associated with *key*, or ``None``."""
        node = self._find_node(key)
        if node is self._nil:
            return None
        return node.value

    def get_root_hash(self) -> bytes:
        """Return the Merkle root commitment.

        This is O(N): it does an in-order traversal, collects data hashes,
        and builds a temporary balanced Merkle tree.
        """
        if not self._dirty:
            return self._cached_root_hash

        old_merkle = self._cached_merkle_hashes
        leaf_hashes = self._collect_leaf_hashes()

        if not leaf_hashes:
            new_merkle: List[bytes] = []
            root_hash = EMPTY_HASH
        else:
            new_merkle, root_hash = self._build_merkle_tree(leaf_hashes)

        # Track changes between old and new Merkle internal nodes.
        self._record_merkle_changes(old_merkle, new_merkle)

        self._cached_merkle_hashes = new_merkle
        self._cached_root_hash = root_hash
        self._dirty = False
        return root_hash

    def generate_proof(self, key: int) -> List[bytes]:
        """Generate a Merkle inclusion proof for *key*.

        O(N): performs an in-order traversal, builds a temporary Merkle tree,
        locates the key's leaf, and collects sibling hashes along its path.
        """
        leaf_hashes = self._collect_leaf_hashes()
        if not leaf_hashes:
            return []

        # Find the position of the target key in the sorted order.
        target_hash: Optional[bytes] = None
        target_index: Optional[int] = None
        node = self._find_node(key)
        if node is self._nil:
            return []

        target_data_hash = hash_leaf(
            node.key.to_bytes(32, "big") + node.value
        )
        for i, lh in enumerate(leaf_hashes):
            if lh == target_data_hash:
                target_index = i
                target_hash = lh
                break

        if target_index is None:
            return []

        merkle_nodes, _ = self._build_merkle_tree(leaf_hashes)
        return self._extract_proof(merkle_nodes, len(leaf_hashes), target_index)

    def get_changed_nodes(self) -> List[Tuple[bytes, bytes]]:
        """Return and clear the list of ``(old_hash, new_hash)`` pairs."""
        # Ensure the Merkle tree is up to date so changes are captured.
        if self._dirty:
            self.get_root_hash()
        changes = self._changed_nodes
        self._changed_nodes = []
        return changes

    # ------------------------------------------------------------------
    # BST helpers
    # ------------------------------------------------------------------

    def _find_node(self, key: int) -> RBNode:
        node = self._root
        while node is not self._nil:
            if key == node.key:
                return node
            elif key < node.key:
                node = node.left
            else:
                node = node.right
        return self._nil

    def _bst_insert(self, z: RBNode) -> None:
        y = self._nil
        x = self._root
        while x is not self._nil:
            y = x
            if z.key < x.key:
                x = x.left
            else:
                x = x.right
        z.parent = y
        if y is self._nil:
            self._root = z
        elif z.key < y.key:
            y.left = z
        else:
            y.right = z

    # ------------------------------------------------------------------
    # Rotations
    # ------------------------------------------------------------------

    def _left_rotate(self, x: RBNode) -> None:
        y = x.right
        x.right = y.left
        if y.left is not self._nil:
            y.left.parent = x
        y.parent = x.parent
        if x.parent is self._nil:
            self._root = y
        elif x is x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.left = x
        x.parent = y
        # Rehash affected nodes bottom-up.
        self._rehash_node(x)
        self._rehash_node(y)

    def _right_rotate(self, y: RBNode) -> None:
        x = y.left
        y.left = x.right
        if x.right is not self._nil:
            x.right.parent = y
        x.parent = y.parent
        if y.parent is self._nil:
            self._root = x
        elif y is y.parent.right:
            y.parent.right = x
        else:
            y.parent.left = x
        x.right = y
        y.parent = x
        # Rehash affected nodes bottom-up.
        self._rehash_node(y)
        self._rehash_node(x)

    # ------------------------------------------------------------------
    # Insert fix-up
    # ------------------------------------------------------------------

    def _insert_fixup(self, z: RBNode) -> None:
        while z.parent.color == Color.RED:
            if z.parent is z.parent.parent.left:
                y = z.parent.parent.right
                if y.color == Color.RED:
                    z.parent.color = Color.BLACK
                    y.color = Color.BLACK
                    z.parent.parent.color = Color.RED
                    z = z.parent.parent
                else:
                    if z is z.parent.right:
                        z = z.parent
                        self._left_rotate(z)
                    z.parent.color = Color.BLACK
                    z.parent.parent.color = Color.RED
                    self._right_rotate(z.parent.parent)
            else:
                y = z.parent.parent.left
                if y.color == Color.RED:
                    z.parent.color = Color.BLACK
                    y.color = Color.BLACK
                    z.parent.parent.color = Color.RED
                    z = z.parent.parent
                else:
                    if z is z.parent.left:
                        z = z.parent
                        self._right_rotate(z)
                    z.parent.color = Color.BLACK
                    z.parent.parent.color = Color.RED
                    self._left_rotate(z.parent.parent)
        self._root.color = Color.BLACK

    # ------------------------------------------------------------------
    # Delete helpers
    # ------------------------------------------------------------------

    def _transplant(self, u: RBNode, v: RBNode) -> None:
        if u.parent is self._nil:
            self._root = v
        elif u is u.parent.left:
            u.parent.left = v
        else:
            u.parent.right = v
        v.parent = u.parent

    def _tree_minimum(self, x: RBNode) -> RBNode:
        while x.left is not self._nil:
            x = x.left
        return x

    def _delete_node(self, z: RBNode) -> None:
        y = z
        y_original_color = y.color

        if z.left is self._nil:
            x = z.right
            self._transplant(z, z.right)
            self._rehash_ancestors(x.parent)
        elif z.right is self._nil:
            x = z.left
            self._transplant(z, z.left)
            self._rehash_ancestors(x.parent)
        else:
            y = self._tree_minimum(z.right)
            y_original_color = y.color
            x = y.right
            if y.parent is z:
                x.parent = y  # important when x is sentinel
            else:
                self._transplant(y, y.right)
                y.right = z.right
                y.right.parent = y
            self._transplant(z, y)
            y.left = z.left
            y.left.parent = y
            y.color = z.color
            self._rehash_node(y)
            self._rehash_ancestors(y.parent)

        if y_original_color == Color.BLACK:
            self._delete_fixup(x)

    def _delete_fixup(self, x: RBNode) -> None:
        while x is not self._root and x.color == Color.BLACK:
            if x is x.parent.left:
                w = x.parent.right
                if w.color == Color.RED:
                    w.color = Color.BLACK
                    x.parent.color = Color.RED
                    self._left_rotate(x.parent)
                    w = x.parent.right
                if w.left.color == Color.BLACK and w.right.color == Color.BLACK:
                    w.color = Color.RED
                    x = x.parent
                else:
                    if w.right.color == Color.BLACK:
                        w.left.color = Color.BLACK
                        w.color = Color.RED
                        self._right_rotate(w)
                        w = x.parent.right
                    w.color = x.parent.color
                    x.parent.color = Color.BLACK
                    w.right.color = Color.BLACK
                    self._left_rotate(x.parent)
                    x = self._root
            else:
                w = x.parent.left
                if w.color == Color.RED:
                    w.color = Color.BLACK
                    x.parent.color = Color.RED
                    self._right_rotate(x.parent)
                    w = x.parent.left
                if w.right.color == Color.BLACK and w.left.color == Color.BLACK:
                    w.color = Color.RED
                    x = x.parent
                else:
                    if w.left.color == Color.BLACK:
                        w.right.color = Color.BLACK
                        w.color = Color.RED
                        self._left_rotate(w)
                        w = x.parent.left
                    w.color = x.parent.color
                    x.parent.color = Color.BLACK
                    w.left.color = Color.BLACK
                    self._right_rotate(x.parent)
                    x = self._root
        x.color = Color.BLACK

    # ------------------------------------------------------------------
    # Hash augmentation (per-node subtree hashes)
    # ------------------------------------------------------------------

    def _compute_node_hash(self, node: RBNode) -> bytes:
        """Compute the subtree hash for a single node."""
        if node is self._nil:
            return EMPTY_HASH
        data_hash = hash_leaf(node.key.to_bytes(32, "big") + node.value)
        return hash_internal(node.left.hash, data_hash + node.right.hash)

    def _rehash_node(self, node: RBNode) -> None:
        """Recompute the hash for *node* and propagate up to the root."""
        current = node
        while current is not self._nil:
            current.hash = self._compute_node_hash(current)
            current = current.parent

    def _rehash_ancestors(self, node: RBNode) -> None:
        """Rehash from *node* up to the root."""
        self._rehash_node(node)

    # ------------------------------------------------------------------
    # In-order traversal & temporary Merkle tree
    # ------------------------------------------------------------------

    def _collect_leaf_hashes(self) -> List[bytes]:
        """In-order traversal collecting each node's data hash."""
        result: List[bytes] = []
        self._inorder(self._root, result)
        return result

    def _inorder(self, node: RBNode, acc: List[bytes]) -> None:
        if node is self._nil:
            return
        self._inorder(node.left, acc)
        data_hash = hash_leaf(node.key.to_bytes(32, "big") + node.value)
        acc.append(data_hash)
        self._inorder(node.right, acc)

    @staticmethod
    def _build_merkle_tree(leaves: List[bytes]) -> Tuple[List[bytes], bytes]:
        """Build a balanced Merkle tree over *leaves*.

        Returns ``(all_node_hashes, root_hash)`` where *all_node_hashes* is a
        flat array representation of the tree (index 0 unused, root at index 1).
        """
        n = len(leaves)
        if n == 0:
            return [], EMPTY_HASH

        # Pad to the next power of two.
        size = 1
        while size < n:
            size <<= 1

        # Tree array: internal nodes at indices [1, size), leaves at [size, 2*size).
        tree: List[bytes] = [EMPTY_HASH] * (2 * size)
        for i in range(n):
            tree[size + i] = leaves[i]

        # Build bottom-up.
        for i in range(size - 1, 0, -1):
            tree[i] = hash_internal(tree[2 * i], tree[2 * i + 1])

        root_hash = tree[1]
        return tree, root_hash

    @staticmethod
    def _extract_proof(tree: List[bytes], num_leaves: int, index: int) -> List[bytes]:
        """Extract sibling hashes for the leaf at *index* in the Merkle tree."""
        # Determine the padded size.
        size = 1
        while size < num_leaves:
            size <<= 1

        proof: List[bytes] = []
        pos = size + index  # position in the flat array
        while pos > 1:
            sibling = pos ^ 1
            proof.append(tree[sibling])
            pos >>= 1
        return proof

    # ------------------------------------------------------------------
    # Change tracking
    # ------------------------------------------------------------------

    def _mark_dirty(self) -> None:
        """Mark the cached Merkle tree as stale."""
        self._dirty = True

    def _record_merkle_changes(
        self, old_tree: List[bytes], new_tree: List[bytes]
    ) -> None:
        """Compare old and new Merkle-tree arrays, recording changed internal nodes."""
        if not old_tree and not new_tree:
            return

        max_len = max(len(old_tree), len(new_tree))
        for i in range(1, max_len):
            old_h = old_tree[i] if i < len(old_tree) else EMPTY_HASH
            new_h = new_tree[i] if i < len(new_tree) else EMPTY_HASH
            if old_h != new_h:
                self._changed_nodes.append((old_h, new_h))
