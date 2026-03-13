import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.trees.merkle_prefix_tree import MerklePrefixTree, EMPTY_HASH
from src.cache.aardvark_delta_cache import AardvarkDeltaCache
from src.protocol.sequencer import Sequencer, Transaction


def _make_sequencer():
    tree = MerklePrefixTree()
    cache = AardvarkDeltaCache()
    return Sequencer(tree, cache)


class TestSubmitAndProcessBatch:
    def test_process_batch_changes_state_root(self):
        seq = _make_sequencer()
        root_before = seq.get_state_root()
        assert root_before == EMPTY_HASH
        seq.submit_transaction(Transaction(op="insert", key=1, value=b"data"))
        new_root = seq.process_batch()
        assert new_root != EMPTY_HASH
        assert new_root == seq.get_state_root()

    def test_process_batch_with_delete(self):
        seq = _make_sequencer()
        seq.submit_transaction(Transaction(op="insert", key=1, value=b"data"))
        seq.process_batch()
        seq.submit_transaction(Transaction(op="delete", key=1, value=b""))
        root_after = seq.process_batch()
        assert root_after == EMPTY_HASH

    def test_empty_batch(self):
        seq = _make_sequencer()
        root = seq.process_batch()
        assert root == EMPTY_HASH

    def test_unknown_op_raises(self):
        seq = _make_sequencer()
        seq.submit_transaction(Transaction(op="update", key=1, value=b"x"))
        with pytest.raises(ValueError, match="Unknown operation"):
            seq.process_batch()


class TestGetProof:
    def test_get_proof_returns_proof_and_version(self):
        seq = _make_sequencer()
        seq.submit_transaction(Transaction(op="insert", key=10, value=b"val"))
        seq.submit_transaction(Transaction(op="insert", key=20, value=b"val2"))
        seq.process_batch()
        proof, version = seq.get_proof(10)
        assert isinstance(proof, list)
        assert len(proof) > 0
        assert version > 0

    def test_proof_version_matches_cache_version(self):
        seq = _make_sequencer()
        seq.submit_transaction(Transaction(op="insert", key=1, value=b"a"))
        seq.process_batch()
        _, version = seq.get_proof(1)
        assert version == seq.get_version()


class TestMultipleBatches:
    def test_multiple_batches_increment_version(self):
        seq = _make_sequencer()
        seq.submit_transaction(Transaction(op="insert", key=1, value=b"a"))
        seq.process_batch()
        assert seq.get_version() == 1

        seq.submit_transaction(Transaction(op="insert", key=2, value=b"b"))
        seq.process_batch()
        assert seq.get_version() == 2

        seq.submit_transaction(Transaction(op="insert", key=3, value=b"c"))
        seq.process_batch()
        assert seq.get_version() == 3

    def test_batch_with_multiple_transactions(self):
        seq = _make_sequencer()
        seq.submit_transaction(Transaction(op="insert", key=1, value=b"a"))
        seq.submit_transaction(Transaction(op="insert", key=2, value=b"b"))
        seq.submit_transaction(Transaction(op="insert", key=3, value=b"c"))
        seq.process_batch()
        # 3 transactions => 3 transitions recorded
        assert seq.get_version() == 3
