
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


RUN_NETWORK_TESTS = (
    os.environ.get(
        "CPIT601_RUN_NETWORK_TESTS"
    )
    == "1"
)


@pytest.mark.skipif(
    not RUN_NETWORK_TESTS,
    reason=(
        "Set CPIT601_RUN_NETWORK_TESTS=1 "
        "to run the pinned-source reconstruction test."
    ),
)
def test_pinned_source_reconstruction_and_runtime_loader(
    tmp_path,
    monkeypatch,
):
    project_root = Path(
        __file__
    ).resolve().parents[1]

    script_path = (
        project_root
        / "scripts"
        / "prepare_assets.py"
    )

    runtime_dir = (
        tmp_path
        / "runtime_assets"
    )

    result = subprocess.run(
        [
            sys.executable,
            str(
                script_path
            ),
            "--output-dir",
            str(
                runtime_dir
            ),
        ],
        cwd=str(
            project_root
        ),
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, (
        result.stdout
        + "\n"
        + result.stderr
    )

    manifest_path = (
        runtime_dir
        / "reconstructed_asset_manifest.json"
    )

    assert manifest_path.exists()

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    assert (
        manifest[
            "asset_type"
        ]
        == "locally_reconstructed"
    )

    assert (
        manifest[
            "scientific_parity_required"
        ]
        is True
    )

    assert (
        manifest[
            "byte_identical_to_original_required"
        ]
        is False
    )

    monkeypatch.setenv(
        "CPIT601_RUNTIME_ASSET_DIR",
        str(
            runtime_dir
        ),
    )

    from cpit601_recourse.predictor import (
        load_project_bundle,
    )

    expected_clean_rows = {
        "OpenStack": 6757,
        "Qt": 14124,
    }

    for project in [
        "OpenStack",
        "Qt",
    ]:
        bundle = load_project_bundle(
            project
        )

        assert (
            bundle[
                "project"
            ]
            == project
        )

        assert len(
            bundle[
                "X_clean"
            ]
        ) == expected_clean_rows[
            project
        ]

        assert len(
            bundle[
                "predictor"
            ].estimators_
        ) == 100

        assert (
            bundle[
                "final_config_sha256"
            ]
            == (
                "f1a96ed265f52b247b4ae33812f501a3"
                "be9007b9a1b2cc751dba75081d74368f"
            )
        )

        assert (
            bundle[
                "source_checkpoint_sha256"
            ]
            == (
                "5e776484a3cac4e4267fea11b2dc0c1f"
                "d9d4f7a1f954feb1ebb1772e9cb4a250"
            )
        )
