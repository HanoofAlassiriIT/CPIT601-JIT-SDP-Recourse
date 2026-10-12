"""Integrity tests for frozen configuration and packaged assets."""

from pathlib import Path

import json

from cpit601_recourse.artifacts import (
    sha256_file,
)

from cpit601_recourse.config import (
    verify_final_config,
    verify_validation_config,
)

from cpit601_recourse.predictor import (
    load_project_bundle,
)


EXPECTED_FINAL_CONFIG_SHA256 = (
    "f1a96ed265f52b247b4ae33812f501a3b"
    "e9007b9a1b2cc751dba75081d74368f"
)


EXPECTED_BUNDLE_SHA256 = {
    "OpenStack":
        "7469c712b9107cb7def93898674cb39e4"
        "afc55e2f0abf7cac602f717ba73c997",

    "Qt":
        "39fcdd305d65e9b8c5a1575363e5d056"
        "a4fbd32af8ce2eb3e8a4b6161a58ef0a",
}


def _package_root():
    return Path(
        __file__
    ).resolve().parents[
        1
    ]


def _load_asset_manifest():
    path = (
        _package_root()
        / "assets/asset_manifest.json"
    )

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def test_frozen_config_integrity():
    assert verify_final_config() is True

    assert verify_validation_config() is True


def test_asset_manifest_preserves_frozen_scientific_state():
    manifest = _load_asset_manifest()

    assert manifest[
        "final_config"
    ][
        "sha256"
    ] == EXPECTED_FINAL_CONFIG_SHA256

    assert manifest[
        "training_or_refit_performed"
    ] is False


def test_packaged_bundle_hashes_match_manifest_and_frozen_values():
    manifest = _load_asset_manifest()

    package_root = _package_root()

    for project, expected_sha256 in EXPECTED_BUNDLE_SHA256.items():
        metadata = manifest[
            "bundles"
        ][
            project
        ]

        assert metadata[
            "sha256"
        ] == expected_sha256

        bundle_path = (
            package_root
            / metadata[
                "path"
            ]
        )

        assert bundle_path.exists()

        assert sha256_file(
            bundle_path
        ) == expected_sha256


def test_loaded_bundles_reference_frozen_config():
    for project in [
        "OpenStack",
        "Qt",
    ]:
        bundle = load_project_bundle(
            project
        )

        assert bundle[
            "project"
        ] == project

        assert bundle[
            "final_config_sha256"
        ] == EXPECTED_FINAL_CONFIG_SHA256

        for required_key in [
            "predictor",
            "scaler",
            "X_clean",
            "nn_model",
        ]:
            assert required_key in bundle
