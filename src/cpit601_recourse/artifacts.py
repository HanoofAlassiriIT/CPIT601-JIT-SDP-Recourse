"""Artifact integrity utilities for frozen research assets."""

from __future__ import annotations

import hashlib
from pathlib import Path


class ArtifactIntegrityError(RuntimeError):
    """Raised when a frozen artifact fails integrity verification."""


def sha256_file(
    path: str | Path,
) -> str:
    """Return SHA-256 for a file."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Artifact does not exist: {path}"
        )

    if not path.is_file():
        raise IsADirectoryError(
            f"Expected file artifact: {path}"
        )

    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def verify_sha256(
    path: str | Path,
    expected_sha256: str,
) -> bool:
    """Return True if a file matches the expected SHA-256."""

    actual = sha256_file(path)

    return actual == expected_sha256.lower()


def require_sha256(
    path: str | Path,
    expected_sha256: str,
) -> str:
    """Verify SHA-256 or raise ArtifactIntegrityError.

    Returns the verified actual SHA-256.
    """

    path = Path(path)

    actual = sha256_file(path)
    expected = expected_sha256.lower()

    if actual != expected:
        raise ArtifactIntegrityError(
            "Artifact integrity check failed. "
            f"path={path}; "
            f"expected_sha256={expected}; "
            f"actual_sha256={actual}"
        )

    return actual
