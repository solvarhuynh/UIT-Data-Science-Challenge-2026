# FULLDOC_TOP200_QWEN manifest provenance

Date: 2026-09-17

The historical report recorded manifest file SHA256
`404d5926eb42f4a3c84b77bd96782654e33854a8eb16f2d67234821d3d37d5ce`.
The exact prior bytes were not recoverable: repository search found only this
historical hash in the durability report, not an old manifest blob or backup.

The current manifest bytes are SHA256
`1f26dde0bb3a03c5482a0c2d03a1a1d47e40ca53ac2d8410bb9c6022811e5722`.
The frozen worklists are unchanged: all 32 worklist SHA256 values match their
current manifest entries, and their deterministic rebuild reproduced all 32
worklist files byte-for-byte. The recomputed universe SHA256 remains
`28ee4cc484eccd52e0013bd9b0bd512cc4911e675d7ee8aa84413af72f83b13d`.

The unchanged builder reproduced the same manifest semantics. After replacing
only temporary output path strings with the canonical frozen paths in memory,
its serialized manifest SHA is exactly `1f26dde0bb3a03c5482a0c2d03a1a1d47e40ca53ac2d8410bb9c6022811e5722`.
The only semantic differences in the isolated temporary rebuild are those
temporary path strings; parity and current-canary bytes also reproduce exactly.

Therefore the current `1f26...` file is authoritative as a byte/metadata
refreeze of the same frozen production universe. The old `404d...` hash is
preserved as historical evidence and is not silently rewritten.
