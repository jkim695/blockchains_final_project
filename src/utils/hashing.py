"""Cryptographic hashing utilities with domain separation for Merkle trees."""

import hashlib


def hash_leaf(data: bytes) -> bytes:
    """Hash a leaf node with domain separation prefix 0x00."""
    return hashlib.sha256(b"\x00" + data).digest()


def hash_internal(left: bytes, right: bytes) -> bytes:
    """Hash an internal node by combining two child hashes with prefix 0x01."""
    return hashlib.sha256(b"\x01" + left + right).digest()
