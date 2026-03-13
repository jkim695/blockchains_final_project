"""Patricia-compressed binary Merkle trie for Layer-2 order-book state."""

from typing import Dict, List, Optional, Tuple

from src.trees.base import StateTree
from src.utils.hashing import hash_internal, hash_leaf

EMPTY_HASH: bytes = b"\x00" * 32
KEY_BITS: int = 64


def _key_to_bits(key: int) -> Tuple[int, ...]:
    """Convert an integer key to a fixed-length tuple of 64 bits (MSB first)."""
    return tuple((key >> (KEY_BITS - 1 - i)) & 1 for i in range(KEY_BITS))


def _key_to_bytes(key: int) -> bytes:
    """Convert an integer key to an 8-byte big-endian representation."""
    return key.to_bytes(8, byteorder="big")


class TrieNode:
    """A node in the Patricia-compressed binary Merkle trie.

    Leaves hold a value and have no children.
    Internal nodes have one or two children keyed by bit (0 or 1).
    The ``prefix`` stores the compressed path bits leading into this node.
    """

    __slots__ = ("children", "value", "hash", "prefix")

    def __init__(
        self,
        prefix: Tuple[int, ...] = (),
        value: Optional[bytes] = None,
        node_hash: bytes = EMPTY_HASH,
    ) -> None:
        self.children: Dict[int, "TrieNode"] = {}
        self.value: Optional[bytes] = value
        self.hash: bytes = node_hash
        self.prefix: Tuple[int, ...] = prefix

    @property
    def is_leaf(self) -> bool:
        return self.value is not None


class MerklePrefixTree(StateTree):
    """Patricia-compressed binary Merkle trie implementing ``StateTree``."""

    def __init__(self) -> None:
        self.root: Optional[TrieNode] = None
        self._changed_nodes: List[Tuple[bytes, bytes]] = []

    # ------------------------------------------------------------------
    # StateTree interface
    # ------------------------------------------------------------------

    def insert(self, key: int, value: bytes) -> bytes:
        bits = _key_to_bits(key)
        key_bytes = _key_to_bytes(key)
        self.root = self._insert(self.root, bits, 0, key_bytes, value)
        return self.get_root_hash()

    def delete(self, key: int) -> bytes:
        bits = _key_to_bits(key)
        self.root = self._delete(self.root, bits, 0)
        return self.get_root_hash()

    def search(self, key: int) -> Optional[bytes]:
        bits = _key_to_bits(key)
        return self._search(self.root, bits, 0)

    def get_root_hash(self) -> bytes:
        if self.root is None:
            return EMPTY_HASH
        return self.root.hash

    def generate_proof(self, key: int) -> List[bytes]:
        bits = _key_to_bits(key)
        proof: List[bytes] = []
        self._generate_proof(self.root, bits, 0, proof)
        return proof

    def get_changed_nodes(self) -> List[Tuple[bytes, bytes]]:
        changes = self._changed_nodes
        self._changed_nodes = []
        return changes

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _record_change(self, old_hash: bytes, new_hash: bytes) -> None:
        self._changed_nodes.append((old_hash, new_hash))

    @staticmethod
    def _compute_hash(node: TrieNode) -> bytes:
        """Recompute and cache the hash of *node*."""
        if node.is_leaf:
            return node.hash  # already set at creation / update time
        children = node.children
        if len(children) == 2:
            return hash_internal(children[0].hash, children[1].hash)
        if len(children) == 1:
            child_bit, child = next(iter(children.items()))
            # Single-child internal: hash with EMPTY_HASH for the missing side.
            if child_bit == 0:
                return hash_internal(child.hash, EMPTY_HASH)
            else:
                return hash_internal(EMPTY_HASH, child.hash)
        # No children and not a leaf — shouldn't happen in normal operation.
        return EMPTY_HASH  # pragma: no cover

    def _rehash(self, node: TrieNode) -> None:
        old_hash = node.hash
        node.hash = self._compute_hash(node)
        if old_hash != node.hash:
            self._record_change(old_hash, node.hash)

    # ------------------------------------------------------------------
    # Insert
    # ------------------------------------------------------------------

    def _insert(
        self,
        node: Optional[TrieNode],
        bits: Tuple[int, ...],
        depth: int,
        key_bytes: bytes,
        value: bytes,
    ) -> TrieNode:
        if node is None:
            # Create a new leaf whose prefix is the remaining bits.
            remaining = bits[depth:]
            leaf = TrieNode(
                prefix=remaining,
                value=value,
                node_hash=hash_leaf(key_bytes + value),
            )
            self._record_change(EMPTY_HASH, leaf.hash)
            return leaf

        prefix = node.prefix
        # Walk the shared prefix of this node.
        match_len = 0
        while (
            match_len < len(prefix)
            and depth + match_len < len(bits)
            and bits[depth + match_len] == prefix[match_len]
        ):
            match_len += 1

        if match_len < len(prefix):
            # The new key diverges inside this node's prefix — split.
            return self._split_and_insert(
                node, bits, depth, match_len, key_bytes, value
            )

        # Full prefix matched; advance depth.
        depth += match_len

        if node.is_leaf:
            if depth == len(bits):
                # Exact match — update existing leaf value.
                old_hash = node.hash
                node.value = value
                node.hash = hash_leaf(key_bytes + value)
                if old_hash != node.hash:
                    self._record_change(old_hash, node.hash)
                return node
            else:
                # Existing leaf needs to be pushed down.
                return self._expand_leaf_and_insert(
                    node, bits, depth, key_bytes, value
                )

        # Internal node — descend into the appropriate child.
        bit = bits[depth]
        child = node.children.get(bit)
        node.children[bit] = self._insert(child, bits, depth + 1, key_bytes, value)
        self._rehash(node)
        return node

    def _split_and_insert(
        self,
        node: TrieNode,
        bits: Tuple[int, ...],
        depth: int,
        match_len: int,
        key_bytes: bytes,
        value: bytes,
    ) -> TrieNode:
        """Split *node*'s prefix at *match_len* and insert the new key."""
        old_hash = node.hash
        prefix = node.prefix
        common = prefix[:match_len]
        existing_bit = prefix[match_len]

        # Shorten the existing node's prefix to the unmatched tail.
        node.prefix = prefix[match_len + 1:]

        # Create a new internal node with the common prefix.
        new_internal = TrieNode(prefix=common)

        # The existing subtree becomes a child under existing_bit.
        new_internal.children[existing_bit] = node

        # Create a leaf for the new key under the diverging bit.
        remaining = bits[depth + match_len + 1:]
        new_leaf = TrieNode(
            prefix=remaining,
            value=value,
            node_hash=hash_leaf(key_bytes + value),
        )
        diverging_bit = bits[depth + match_len]
        new_internal.children[diverging_bit] = new_leaf
        self._record_change(EMPTY_HASH, new_leaf.hash)

        # Hash the new internal node.
        new_internal.hash = self._compute_hash(new_internal)
        self._record_change(old_hash, new_internal.hash)

        return new_internal

    def _expand_leaf_and_insert(
        self,
        leaf: TrieNode,
        bits: Tuple[int, ...],
        depth: int,
        key_bytes: bytes,
        value: bytes,
    ) -> TrieNode:
        """An existing leaf has a shorter path than the new key — expand it."""
        # Reconstruct the full bit-path for the existing leaf.
        # The existing leaf sits at *depth* (its prefix already consumed).
        # We need to figure out the existing leaf's remaining bits.
        # Since the leaf has no further bits (its prefix was fully consumed
        # before calling this method), we need to re-derive from its data.
        #
        # Actually, the existing leaf's prefix was already fully consumed,
        # meaning the leaf occupied all bits up to ``depth``.  But the new
        # key still has bits remaining, so the new key is *longer* — which
        # can't happen with fixed-length keys.  This branch should never
        # trigger with a consistent 64-bit key space, but we handle it
        # defensively by treating the existing leaf as having an implicit
        # empty remaining path.
        #
        # We convert the leaf into an internal node and push both keys down.
        old_hash = leaf.hash
        old_value = leaf.value

        # Turn the leaf into an internal node.
        leaf.value = None

        # The existing value becomes a child leaf with no further prefix.
        existing_child = TrieNode(
            prefix=(), value=old_value, node_hash=old_hash
        )

        # New key child.
        new_bit = bits[depth]
        remaining = bits[depth + 1:]
        new_leaf = TrieNode(
            prefix=remaining,
            value=value,
            node_hash=hash_leaf(key_bytes + value),
        )
        self._record_change(EMPTY_HASH, new_leaf.hash)

        # If both land on the same bit we need to recurse, but with
        # fixed-length keys this means the existing leaf *also* has more
        # bits, which contradicts the premise.  Guard anyway:
        leaf.children[new_bit] = new_leaf
        if new_bit == 0:
            leaf.children[1] = existing_child
        else:
            leaf.children[0] = existing_child

        self._rehash(leaf)
        return leaf

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def _delete(
        self,
        node: Optional[TrieNode],
        bits: Tuple[int, ...],
        depth: int,
    ) -> Optional[TrieNode]:
        if node is None:
            return None

        prefix = node.prefix
        match_len = 0
        while (
            match_len < len(prefix)
            and depth + match_len < len(bits)
            and bits[depth + match_len] == prefix[match_len]
        ):
            match_len += 1

        if match_len < len(prefix):
            # Key not present (diverges inside this node's prefix).
            return node

        depth += match_len

        if node.is_leaf:
            if depth == len(bits):
                # Found the leaf to delete.
                self._record_change(node.hash, EMPTY_HASH)
                return None
            # Key not present.
            return node

        # Internal node — descend.
        bit = bits[depth]
        child = node.children.get(bit)
        if child is None:
            return node  # Key not present.

        result = self._delete(child, bits, depth + 1)
        old_hash = node.hash

        if result is None:
            del node.children[bit]
        else:
            node.children[bit] = result

        # Collapse if only one child remains (Patricia re-compression).
        if len(node.children) == 1:
            remaining_bit, remaining_child = next(iter(node.children.items()))
            # Merge prefixes: current prefix + the bit used to reach child + child's prefix.
            merged_prefix = node.prefix + (remaining_bit,) + remaining_child.prefix
            remaining_child.prefix = merged_prefix
            # Record the internal node disappearing.
            self._record_change(old_hash, remaining_child.hash)
            return remaining_child

        if len(node.children) == 0:
            # All children removed (shouldn't happen normally).
            self._record_change(old_hash, EMPTY_HASH)
            return None

        # Two children still present — just rehash.
        self._rehash(node)
        return node

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _search(
        self,
        node: Optional[TrieNode],
        bits: Tuple[int, ...],
        depth: int,
    ) -> Optional[bytes]:
        if node is None:
            return None

        prefix = node.prefix
        for i, p in enumerate(prefix):
            if depth + i >= len(bits) or bits[depth + i] != p:
                return None
        depth += len(prefix)

        if node.is_leaf:
            return node.value if depth == len(bits) else None

        bit = bits[depth]
        return self._search(node.children.get(bit), bits, depth + 1)

    # ------------------------------------------------------------------
    # Proof generation
    # ------------------------------------------------------------------

    def _generate_proof(
        self,
        node: Optional[TrieNode],
        bits: Tuple[int, ...],
        depth: int,
        proof: List[bytes],
    ) -> bool:
        """Walk toward the key, collecting sibling hashes. Returns True if key found."""
        if node is None:
            return False

        prefix = node.prefix
        for i, p in enumerate(prefix):
            if depth + i >= len(bits) or bits[depth + i] != p:
                return False
        depth += len(prefix)

        if node.is_leaf:
            return depth == len(bits)

        bit = bits[depth]
        sibling_bit = 1 - bit
        child = node.children.get(bit)

        found = self._generate_proof(child, bits, depth + 1, proof)
        if found:
            # Append sibling hash (or EMPTY_HASH if sibling absent).
            sibling = node.children.get(sibling_bit)
            sibling_hash = sibling.hash if sibling is not None else EMPTY_HASH
            proof.append(sibling_hash)
        return found
