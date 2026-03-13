from typing import List
from src.cache.aardvark_delta_cache import AardvarkDeltaCache


class Validator:
    def __init__(self, cache: AardvarkDeltaCache):
        self._cache = cache

    def validate_proof(self, proof: List[bytes], key: int, root: bytes) -> bool:
        """Verify a Merkle inclusion proof against an expected root."""
        return self._cache.verify_proof(proof, key, root)

    def validate_stale_proof(
        self,
        proof: List[bytes],
        key: int,
        proof_version: int,
        current_version: int,
    ) -> bool:
        """Patch a stale proof using the delta cache and validate against the current root."""
        patched = self._cache.patch_proof(proof, proof_version, current_version)
        current_root = self._cache.get_root_at_version(current_version)
        if current_root is None:
            return False
        return self._cache.verify_proof(patched, key, current_root)
