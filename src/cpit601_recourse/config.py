"""Frozen configuration loading and integrity utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .artifacts import (
    require_sha256,
    sha256_file,
    verify_sha256,
)


PACKAGE_ROOT = Path(__file__).resolve().parent / "_data"

FINAL_CONFIG_PATH = (
    PACKAGE_ROOT
    / "configs"
    / "final_config.json"
)

VALIDATION_CONFIG_PATH = (
    PACKAGE_ROOT
    / "configs"
    / "validation_config.json"
)


EXPECTED_FINAL_CONFIG_SHA256 = (
    "f1a96ed265f52b247b4ae33812f501a3"
    "be9007b9a1b2cc751dba75081d74368f"
)

EXPECTED_VALIDATION_CONFIG_SHA256 = (
    "3e263487177a45ec54b103635eeeb4ad"
    "39f534ff3a9199b2e8f1e6b4db3deaab"
)


def load_json_config(
    path: str | Path,
) -> dict[str, Any]:
    """Load a JSON config without modifying the source file."""

    path = Path(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as stream:
        data = json.load(stream)

    if not isinstance(data, dict):
        raise TypeError(
            f"Expected JSON object in {path}"
        )

    return data


def verify_final_config() -> bool:
    """Verify packaged FINAL_CONFIG identity."""

    return verify_sha256(
        FINAL_CONFIG_PATH,
        EXPECTED_FINAL_CONFIG_SHA256,
    )


def verify_validation_config() -> bool:
    """Verify packaged Validation config identity."""

    return verify_sha256(
        VALIDATION_CONFIG_PATH,
        EXPECTED_VALIDATION_CONFIG_SHA256,
    )


def load_final_config() -> dict[str, Any]:
    """Verify and load the frozen FINAL_CONFIG."""

    require_sha256(
        FINAL_CONFIG_PATH,
        EXPECTED_FINAL_CONFIG_SHA256,
    )

    return load_json_config(
        FINAL_CONFIG_PATH
    )


def load_validation_config() -> dict[str, Any]:
    """Verify and load the frozen Validation config."""

    require_sha256(
        VALIDATION_CONFIG_PATH,
        EXPECTED_VALIDATION_CONFIG_SHA256,
    )

    return load_json_config(
        VALIDATION_CONFIG_PATH
    )
