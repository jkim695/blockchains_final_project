"""Abstract base class for authenticated state trees."""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple


class StateTree(ABC):
    """Interface that both MerklePrefixTree and RedBlackTree must implement.

    Every mutation (insert/delete) must track which node hashes changed
    so the AardvarkDeltaCache can record transitions.
    """

    @abstractmethod
    def insert(self, key: int, value: bytes) -> bytes:
        """Insert or update a key-value pair. Returns the new root hash."""

    @abstractmethod
    def delete(self, key: int) -> bytes:
        """Delete a key. Returns the new root hash."""

    @abstractmethod
    def search(self, key: int) -> Optional[bytes]:
        """Look up a key. Returns the value or None if not found."""

    @abstractmethod
    def get_root_hash(self) -> bytes:
        """Return the current Merkle root commitment."""

    @abstractmethod
    def generate_proof(self, key: int) -> List[bytes]:
        """Generate an inclusion proof (list of sibling hashes) for a key."""

    @abstractmethod
    def get_changed_nodes(self) -> List[Tuple[bytes, bytes]]:
        """Return (old_hash, new_hash) pairs for nodes changed in the last mutation.

        Calling this clears the internal change log.
        """
