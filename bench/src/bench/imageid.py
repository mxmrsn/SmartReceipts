"""Deterministic image-id generation that matches the Swift labeler.

Mirrors `apps/macos-labeler/Sources/ReceiptLabeler/Models/ImageIDGenerator.swift`
bit-for-bit so the Python-side `bench prelabel` writes label files that the
Swift labeler will pick up under the same id.

The algorithm: SHA-256 of the filename stem (no extension), take the first
16 bytes, set RFC 4122 version=5 and variant bits.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from uuid import UUID


def uuid_for_filename(filename: str) -> UUID:
    """Stable UUID derived from the filename stem (extension stripped).

    Re-encoding `IMG_1234.heic` → `IMG_1234.jpg` keeps the same id, which
    matches the Swift labeler's behaviour and lets re-encoded copies share
    a single label file.
    """
    stem = os.path.splitext(filename)[0]
    digest = hashlib.sha256(stem.encode("utf-8")).digest()
    b = bytearray(digest[:16])
    b[6] = (b[6] & 0x0F) | 0x50  # version 5
    b[8] = (b[8] & 0x3F) | 0x80  # RFC 4122 variant
    return UUID(bytes=bytes(b))


def uuid_for_path(path: Path) -> UUID:
    return uuid_for_filename(path.name)
