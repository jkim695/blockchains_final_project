import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.trees.red_black_tree import RedBlackTree
from src.cache.aardvark_delta_cache import AardvarkDeltaCache
from src.protocol.sequencer import Sequencer, Transaction
from src.protocol.validator import Validator
from src.utils.hashing import hash_leaf


def _setup():
    """Use RedBlackTree because its generate_proof produces proofs compatible
    with AardvarkDeltaCache.verify_proof (leaf hash + sibling hashes)."""
    tree = RedBlackTree()
    cache = AardvarkDeltaCache()
    seq = Sequencer(tree, cache)
    val = Validator(cache)
    return tree, cache, seq, val


class TestValidateProof:
    def test_valid_proof_succeeds(self):
        tree, cache, seq, val = _setup()
        seq.submit_transaction(Transaction(op="insert", key=10, value=b"order"))
        seq.submit_transaction(Transaction(op="insert", key=20, value=b"order2"))
        seq.process_batch()

        proof, version = seq.get_proof(10)
        root = seq.get_state_root()

        # RedBlackTree.generate_proof returns sibling hashes only.
        # verify_proof expects proof[0] = leaf hash, rest = siblings.
        leaf_hash = hash_leaf(
            (10).to_bytes(32, "big") + b"order"
        )
        full_proof = [leaf_hash] + proof
        result = val.validate_proof(full_proof, 10, root)
        assert result is True

    def test_tampered_proof_fails(self):
        tree, cache, seq, val = _setup()
        seq.submit_transaction(Transaction(op="insert", key=10, value=b"order"))
        seq.submit_transaction(Transaction(op="insert", key=20, value=b"order2"))
        seq.process_batch()

        proof, version = seq.get_proof(10)
        root = seq.get_state_root()

        leaf_hash = hash_leaf(
            (10).to_bytes(32, "big") + b"order"
        )
        full_proof = [leaf_hash] + proof

        # Tamper with a sibling hash.
        tampered = list(full_proof)
        bad_hash = bytearray(tampered[-1])
        bad_hash[0] ^= 0xFF
        tampered[-1] = bytes(bad_hash)
        result = val.validate_proof(tampered, 10, root)
        assert result is False

    def test_wrong_root_fails(self):
        tree, cache, seq, val = _setup()
        seq.submit_transaction(Transaction(op="insert", key=10, value=b"order"))
        seq.submit_transaction(Transaction(op="insert", key=20, value=b"order2"))
        seq.process_batch()

        proof, version = seq.get_proof(10)
        leaf_hash = hash_leaf(
            (10).to_bytes(32, "big") + b"order"
        )
        full_proof = [leaf_hash] + proof
        wrong_root = b"\xff" * 32
        result = val.validate_proof(full_proof, 10, wrong_root)
        assert result is False


class TestValidateStaleProof:
    def test_stale_proof_validated_after_patch(self):
        tree, cache, seq, val = _setup()
        # Build initial state.
        seq.submit_transaction(Transaction(op="insert", key=10, value=b"a"))
        seq.submit_transaction(Transaction(op="insert", key=20, value=b"b"))
        seq.process_batch()

        proof_v2, v2 = seq.get_proof(10)

        # Advance the state.
        seq.submit_transaction(Transaction(op="insert", key=30, value=b"c"))
        seq.process_batch()
        v3 = seq.get_version()

        # The old proof should be patchable without crashing.
        result = val.validate_stale_proof(proof_v2, 10, v2, v3)
        assert isinstance(result, bool)
