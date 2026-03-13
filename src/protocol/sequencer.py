import threading
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Tuple, Optional
from src.trees.base import StateTree
from src.cache.aardvark_delta_cache import AardvarkDeltaCache


@dataclass
class Transaction:
    """A single order book operation."""
    op: str          # "insert" or "delete"
    key: int         # price level
    value: bytes     # order data (empty for delete)


class Sequencer:
    """Thread-safe transaction sequencer.

    Multiple threads may call ``submit_transaction`` concurrently.
    ``process_batch`` drains the queue under a state lock so that tree
    mutations remain serialized while submissions continue to arrive.
    Read operations (``get_proof``, ``get_state_root``) acquire a shared
    read-lock so they never see a half-applied batch.
    """

    def __init__(self, tree: StateTree, cache: AardvarkDeltaCache):
        self._tree = tree
        self._cache = cache
        self._tx_queue: Deque[Transaction] = deque()
        self._queue_lock = threading.Lock()
        # RLock allows the same thread to re-enter (e.g. process_batch
        # calling get_state_root internally) and acts as a write-lock
        # for tree mutations.  Readers also acquire it briefly to get a
        # consistent snapshot, which is acceptable because reads are fast.
        self._state_lock = threading.RLock()

    def submit_transaction(self, tx: Transaction) -> None:
        """Enqueue a transaction for processing.  Thread-safe."""
        with self._queue_lock:
            self._tx_queue.append(tx)

    def submit_transactions(self, txs: List[Transaction]) -> None:
        """Enqueue multiple transactions atomically.  Thread-safe."""
        with self._queue_lock:
            self._tx_queue.extend(txs)

    def process_batch(self) -> bytes:
        """Drain the queue and apply all transactions to the tree.

        Acquires the state lock so that no reads see intermediate states.
        The queue lock is held only long enough to swap out the pending
        transactions, so new submissions are not blocked during execution.
        Returns the new state root.
        """
        # Atomically drain the queue into a local list.
        with self._queue_lock:
            batch = list(self._tx_queue)
            self._tx_queue.clear()

        # Apply under the state lock — serializes mutations and blocks
        # concurrent reads until the batch is fully committed.
        with self._state_lock:
            for tx in batch:
                if tx.op == "insert":
                    self._tree.insert(tx.key, tx.value)
                elif tx.op == "delete":
                    self._tree.delete(tx.key)
                else:
                    raise ValueError(f"Unknown operation: {tx.op}")
                self._cache.record_transition(self._tree)
            return self._tree.get_root_hash()

    def get_state_root(self) -> bytes:
        """Return the current committed root hash.  Thread-safe."""
        with self._state_lock:
            return self._tree.get_root_hash()

    def get_proof(self, key: int) -> Tuple[List[bytes], int]:
        """Generate an inclusion proof for *key*.

        Returns ``(proof, version)`` under the state lock so the proof
        and version are guaranteed to be consistent.  Thread-safe.
        """
        with self._state_lock:
            proof = self._tree.generate_proof(key)
            version = self._cache.get_version()
            return proof, version

    def get_version(self) -> int:
        """Return the current version number.  Thread-safe."""
        with self._state_lock:
            return self._cache.get_version()
