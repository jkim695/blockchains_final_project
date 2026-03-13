"""Aardvark asynchronous versioning cache for patching stale Merkle proofs.

When concurrent state updates invalidate a previously-generated proof, the
delta cache can patch it by replaying recorded node-hash changes rather than
recomputing the full tree.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.trees.base import StateTree
from src.utils.hashing import hash_internal


@dataclass
class Delta:
    """A single state transition record."""

    version: int
    old_root: bytes
    new_root: bytes
    changed_nodes: List[Tuple[bytes, bytes]]  # (old_hash, new_hash)
    timestamp: float


class AardvarkDeltaCache:
    """Versioning cache that patches stale proofs via recorded delta chains.

    Instead of regenerating a Merkle proof from scratch after the tree has
    moved forward, callers can use *patch_proof* to surgically replace the
    sibling hashes that changed between the proof's version and the current
    version.
    """

    def __init__(self, max_retained_versions: int = 1000) -> None:
        self._deltas: Dict[int, Delta] = {}
        self._root_to_version: Dict[bytes, int] = {}
        self._current_version: int = 0
        self._max_retained_versions: int = max_retained_versions

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_transition(self, tree: StateTree) -> None:
        """Record a state transition after a tree mutation.

        Must be called *after* the mutation so that ``tree.get_root_hash()``
        returns the new root and ``tree.get_changed_nodes()`` returns the
        changes from that mutation.
        """
        new_root = tree.get_root_hash()
        changed_nodes = tree.get_changed_nodes()

        if self._current_version == 0:
            old_root = b"\x00" * 32
        else:
            old_root = self._deltas[self._current_version].new_root

        self._current_version += 1
        delta = Delta(
            version=self._current_version,
            old_root=old_root,
            new_root=new_root,
            changed_nodes=changed_nodes,
            timestamp=time.time(),
        )

        self._deltas[self._current_version] = delta
        self._root_to_version[new_root] = self._current_version

        # Auto-prune when exceeding the retention limit.
        if len(self._deltas) > self._max_retained_versions:
            self.prune(self._max_retained_versions)

    # ------------------------------------------------------------------
    # Proof patching
    # ------------------------------------------------------------------

    def patch_proof(
        self,
        proof: List[bytes],
        proof_version: int,
        target_version: int,
    ) -> List[bytes]:
        """Patch *proof* from *proof_version* forward to *target_version*.

        Complexity: O(D * P * C) where D = versions bridged, P = proof length,
        C = average changes per version.

        Raises ``ValueError`` if any intermediate version is missing from the
        cache.
        """
        if proof_version == target_version:
            return list(proof)

        chain = self.get_delta_chain(proof_version, target_version)
        patched = list(proof)

        for delta in chain:
            for old_hash, new_hash in delta.changed_nodes:
                for i, node in enumerate(patched):
                    if node == old_hash:
                        patched[i] = new_hash

        return patched

    def get_delta_chain(
        self, from_version: int, to_version: int
    ) -> List[Delta]:
        """Return the ordered list of deltas between two versions.

        Raises ``ValueError`` if any version in the range is missing.
        """
        if from_version == to_version:
            return []

        start = min(from_version, to_version) + 1
        end = max(from_version, to_version) + 1
        chain: List[Delta] = []

        for v in range(start, end):
            if v not in self._deltas:
                raise ValueError(
                    f"Version {v} not found in delta cache "
                    f"(range {from_version} -> {to_version})"
                )
            chain.append(self._deltas[v])

        return chain

    # ------------------------------------------------------------------
    # Garbage collection
    # ------------------------------------------------------------------

    def prune(self, keep_last: int) -> None:
        """Remove deltas older than ``current_version - keep_last``."""
        cutoff = self._current_version - keep_last
        versions_to_remove = [v for v in self._deltas if v <= cutoff]

        for v in versions_to_remove:
            delta = self._deltas.pop(v)
            # Clean up root -> version mapping if it still points here.
            if self._root_to_version.get(delta.new_root) == v:
                del self._root_to_version[delta.new_root]

    # ------------------------------------------------------------------
    # Proof verification
    # ------------------------------------------------------------------

    def verify_proof(
        self, proof: List[bytes], key: int, expected_root: bytes
    ) -> bool:
        """Verify an inclusion proof by rehashing from the leaf upward.

        *proof* is a list of sibling hashes ordered from the leaf level to the
        root.  At each level the bit of *key* determines whether the current
        hash is the left or right child.
        """
        current = proof[0]
        for i, sibling in enumerate(proof[1:]):
            if (key >> i) & 1 == 0:
                current = hash_internal(current, sibling)
            else:
                current = hash_internal(sibling, current)
        return current == expected_root

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_version(self) -> int:
        """Return the current version number."""
        return self._current_version

    def get_root_at_version(self, version: int) -> Optional[bytes]:
        """Return the root hash recorded at *version*, or ``None``."""
        delta = self._deltas.get(version)
        if delta is not None:
            return delta.new_root
        return None
