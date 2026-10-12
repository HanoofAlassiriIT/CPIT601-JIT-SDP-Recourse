"""Frozen predictor asset loading utilities.

This module loads already-fitted project bundles.
It performs no training, fitting, tuning, or prediction.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib

from .artifacts import require_sha256
from .config import (
    PACKAGE_ROOT,
    EXPECTED_FINAL_CONFIG_SHA256,
)


SUPPORTED_PROJECTS = (
    "OpenStack",
    "Qt",
)

ASSET_MANIFEST_PATH = (
    PACKAGE_ROOT
    / "assets"
    / "asset_manifest.json"
)

EXPECTED_ASSET_MANIFEST_SHA256 = (
    "32ca9553cf552ec28f84b2ab44fdf951"
    "e211c5895c5e16c179eeaa677f00720b"
)


RUNTIME_ASSET_DIR_ENV = "CPIT601_RUNTIME_ASSET_DIR"

EXPECTED_SOURCE_CHECKPOINT_SHA256 = (
    "5e776484a3cac4e4267fea11b2dc0c1f"
    "d9d4f7a1f954feb1ebb1772e9cb4a250"
)

EXPECTED_BUNDLE_KEYS = {
    "bundle_format_version",
    "project",
    "source_checkpoint_name",
    "source_checkpoint_sha256",
    "final_config_sha256",
    "predictor",
    "scaler",
    "X_clean",
    "nn_model",
}


def _resolve_asset_context() -> tuple[Path, Path, bool]:
    """Resolve frozen or locally reconstructed runtime assets."""

    runtime_dir_value = os.environ.get(
        RUNTIME_ASSET_DIR_ENV
    )

    if runtime_dir_value:
        runtime_dir = Path(
            runtime_dir_value
        ).expanduser().resolve()

        manifest_path = (
            runtime_dir
            / "reconstructed_asset_manifest.json"
        )

        if not manifest_path.exists():
            raise FileNotFoundError(
                "Runtime asset manifest not found: "
                f"{manifest_path}"
            )

        return manifest_path, runtime_dir, False

    return ASSET_MANIFEST_PATH, PACKAGE_ROOT, True


def load_asset_manifest() -> dict[str, Any]:
    """Verify and load the active asset manifest."""

    manifest_path, _, frozen_mode = _resolve_asset_context()

    if frozen_mode:
        require_sha256(
            manifest_path,
            EXPECTED_ASSET_MANIFEST_SHA256,
        )

    with manifest_path.open(
        "r",
        encoding="utf-8",
    ) as stream:
        manifest = json.load(stream)

    if not isinstance(manifest, dict):
        raise TypeError(
            "Asset manifest must be a JSON object."
        )

    return manifest


def validate_project(project: str) -> str:
    """Validate an exact supported project name."""

    if project not in SUPPORTED_PROJECTS:
        raise ValueError(
            "Unsupported project. "
            f"Expected one of {SUPPORTED_PROJECTS}; "
            f"received {project!r}."
        )

    return project


def load_project_bundle(
    project: str,
) -> dict[str, Any]:
    """Load a hash-verified frozen project bundle."""

    project = validate_project(project)

    manifest = load_asset_manifest()

    bundles = manifest.get("bundles")

    if not isinstance(bundles, dict):
        raise RuntimeError(
            "Invalid asset manifest: missing bundles mapping."
        )

    if project not in bundles:
        raise RuntimeError(
            f"Asset manifest has no entry for {project}."
        )

    bundle_info = bundles[project]

    relative_path = bundle_info["path"]
    expected_sha256 = bundle_info["sha256"]

    _, asset_root, frozen_mode = _resolve_asset_context()

    bundle_path = (
        asset_root
        / Path(relative_path)
    )

    # Verify bytes before deserialization.
    require_sha256(
        bundle_path,
        expected_sha256,
    )

    bundle = joblib.load(
        bundle_path
    )

    if not isinstance(bundle, dict):
        raise TypeError(
            f"{project} bundle must be a dictionary."
        )

    if set(bundle.keys()) != EXPECTED_BUNDLE_KEYS:
        raise RuntimeError(
            f"{project} bundle structure mismatch."
        )

    if bundle["project"] != project:
        raise RuntimeError(
            f"{project} bundle project metadata mismatch."
        )

    if (
        bundle["final_config_sha256"]
        != EXPECTED_FINAL_CONFIG_SHA256
    ):
        raise RuntimeError(
            f"{project} FINAL_CONFIG provenance mismatch."
        )

    if frozen_mode:
        expected_core_sha = (
            manifest["source_checkpoint"]["sha256"]
        )
    else:
        expected_core_sha = (
            EXPECTED_SOURCE_CHECKPOINT_SHA256
        )

    if (
        bundle["source_checkpoint_sha256"]
        != expected_core_sha
    ):
        raise RuntimeError(
            f"{project} checkpoint provenance mismatch."
        )

    return bundle



def predict_labels(
    project: str,
    X,
):
    """Predict frozen classifier labels after strict input validation."""

    from .data import validate_input_frame

    validated_X = validate_input_frame(
        project,
        X,
    )

    bundle = load_project_bundle(
        project
    )

    predictor = bundle["predictor"]

    predictions = predictor.predict(
        validated_X
    )

    return predictions



def predict_defect_probability(
    project: str,
    X,
):
    """Return frozen-model probability of the binary defect class."""

    from .config import load_final_config
    from .data import validate_input_frame

    validated_X = validate_input_frame(
        project,
        X,
    )

    bundle = load_project_bundle(
        project
    )

    predictor = bundle["predictor"]

    config = load_final_config()

    desired_clean_class = (
        config["predictor"]
        ["desired_recourse_class"]
    )

    classes = list(
        predictor.classes_
    )

    if len(classes) != 2:
        raise RuntimeError(
            "Expected a binary frozen predictor."
        )

    if desired_clean_class not in classes:
        raise RuntimeError(
            "Frozen desired CLEAN class is absent "
            "from predictor classes."
        )

    defect_classes = [
        label
        for label in classes
        if label != desired_clean_class
    ]

    if len(defect_classes) != 1:
        raise RuntimeError(
            "Could not identify a unique defect class."
        )

    defect_class = defect_classes[0]

    defect_index = classes.index(
        defect_class
    )

    probabilities = predictor.predict_proba(
        validated_X
    )[:, defect_index]

    return probabilities
