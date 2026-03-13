# Verifiable L2 Matching Engine: Merkle-Prefix Tree vs Red-Black Tree

Benchmarks a **Hybrid Merkle-Prefix Order Book Tree** (Lighter Protocol) against a standard **Red-Black Tree** for Layer-2 matching engine proof generation. Integrates **Aardvark's delta caching** to solve the stale proof bottleneck.

## Key Thesis

- **MerklePrefixTree**: O(log N) proof generation via sibling hash collection along trie path
- **RedBlackTree**: O(N) proof generation requiring full in-order traversal + temporary Merkle tree
- **AardvarkDeltaCache**: Patches stale proofs in O(D*P) instead of regenerating from scratch

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Run Tests

```bash
pytest tests/ -v
```

## Run Benchmarks

```bash
pytest benchmarks/ -v
```

## Project Structure

```
src/
  trees/          # MerklePrefixTree (Patricia trie) and RedBlackTree
  cache/          # AardvarkDeltaCache (transaction versioning + delta tracking)
  protocol/       # Sequencer (tx ordering) and Validator (proof verification)
  utils/          # SHA-256 hashing with domain separation
tests/            # Unit tests for all components
benchmarks/       # Performance benchmarks (insert, proof gen, delta cache, memory)
```
