"""Prepare CPIT601 runtime assets from verified source datasets."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
import urllib.request

from pathlib import Path

import joblib
import pandas as pd
import numpy as np

from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

SPEC_PATH = (
    PROJECT_ROOT
    / "src"
    / "cpit601_recourse"
    / "_data"
    / "reconstruction_spec.json"
)


def sha256_bytes(
    data: bytes,
) -> str:
    return hashlib.sha256(
        data
    ).hexdigest()


def load_reconstruction_spec() -> dict:
    if not SPEC_PATH.exists():
        raise FileNotFoundError(
            f"Missing reconstruction specification: {SPEC_PATH}"
        )

    with SPEC_PATH.open(
        "r",
        encoding="utf-8",
    ) as stream:
        spec = json.load(
            stream
        )

    if spec.get(
        "scientific_parity_required"
    ) is not True:
        raise RuntimeError(
            "Reconstruction specification does not require scientific parity."
        )

    return spec


def read_verified_project_csv(
    source_zip: Path,
    project: str,
    spec: dict,
) -> tuple[pd.DataFrame, str]:
    project_spec = spec[
        "projects"
    ][
        project
    ]

    member = project_spec[
        "dataset_member"
    ]

    expected_sha256 = project_spec[
        "dataset_sha256"
    ]

    with zipfile.ZipFile(
        source_zip,
        "r",
    ) as archive:
        raw_bytes = archive.read(
            member
        )

    actual_sha256 = sha256_bytes(
        raw_bytes
    )

    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"{project} source SHA256 mismatch. "
            f"Expected {expected_sha256}, got {actual_sha256}."
        )

    frame = pd.read_csv(
        io.BytesIO(
            raw_bytes
        )
    )

    if frame.empty:
        raise RuntimeError(
            f"{project} source dataset is empty."
        )

    return frame, actual_sha256


def validate_source_zip(
    source_zip: Path,
) -> None:
    if not source_zip.exists():
        raise FileNotFoundError(
            f"Source ZIP not found: {source_zip}"
        )

    spec = load_reconstruction_spec()

    for project in [
        "OpenStack",
        "Qt",
    ]:
        frame, source_sha256 = read_verified_project_csv(
            source_zip=source_zip,
            project=project,
            spec=spec,
        )

        print(
            project
        )

        print(
            "  rows:",
            len(
                frame
            )
        )

        print(
            "  columns:",
            len(
                frame.columns
            )
        )

        print(
            "  source SHA256:",
            source_sha256
        )

    print()
    print(
        "Source validation: PASS"
    )




def read_verified_upstream_project_csv(
    project: str,
    spec: dict,
) -> tuple[pd.DataFrame, str]:
    project_spec = spec[
        "projects"
    ][
        project
    ]

    url = project_spec[
        "upstream_raw_url"
    ]

    expected_sha256 = project_spec[
        "dataset_sha256"
    ]

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "CPIT601-Reproducibility"
        },
    )

    with urllib.request.urlopen(
        request,
        timeout=60,
    ) as response:
        raw_bytes = response.read()

    actual_sha256 = sha256_bytes(
        raw_bytes
    )

    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"{project} upstream SHA256 mismatch. "
            f"Expected {expected_sha256}, got {actual_sha256}."
        )

    frame = pd.read_csv(
        io.BytesIO(
            raw_bytes
        )
    )

    if frame.empty:
        raise RuntimeError(
            f"{project} upstream dataset is empty."
        )

    return frame, actual_sha256


def validate_upstream_sources() -> None:
    spec = load_reconstruction_spec()

    for project in [
        "OpenStack",
        "Qt",
    ]:
        frame, source_sha256 = (
            read_verified_upstream_project_csv(
                project=project,
                spec=spec,
            )
        )

        print(
            project
        )

        print(
            "  rows:",
            len(
                frame
            )
        )

        print(
            "  columns:",
            len(
                frame.columns
            )
        )

        print(
            "  SHA256 exact:",
            source_sha256
            == spec[
                "projects"
            ][
                project
            ][
                "dataset_sha256"
            ]
        )


def reconstruct_project_bundle(
    source_zip: Path | None,
    project: str,
) -> tuple[dict, pd.DataFrame, float]:
    spec = load_reconstruction_spec()

    project_spec = spec[
        "projects"
    ][
        project
    ]

    settings = spec[
        "shared_settings"
    ]

    if source_zip is None:
        frame, _ = read_verified_upstream_project_csv(
            project=project,
            spec=spec,
        )
    else:
        frame, _ = read_verified_project_csv(
            source_zip=source_zip,
            project=project,
            spec=spec,
        )

    frame = frame[
        (
            frame["author_date"]
            >= project_spec["author_date_start"]
        )
        & (
            frame["author_date"]
            <= project_spec["author_date_end"]
        )
    ].copy()

    frame["defect"] = (
        frame["bugcount"] > 0
    ).astype(
        int
    )

    frame["author_datetime"] = pd.to_datetime(
        frame["author_date"],
        unit="s",
    )

    frame = frame.sort_values(
        "author_datetime"
    ).reset_index(
        drop=True
    )

    frame["self"] = frame["self"].apply(
        lambda value: 1 if value is True else 0
    )

    frame["rtime"] = (
        frame["rtime"] / 3600
    ) / 24

    frame["age"] = (
        frame["age"] / 3600
    ) / 24

    frame = frame.set_index(
        "commit_id"
    )

    y = frame[
        "defect"
    ].astype(
        int
    ).copy()

    drop_columns = [
        "author_date",
        "author_datetime",
        "bugcount",
        "fixcount",
        "revd",
        "tcmt",
        "oexp",
        "orexp",
        "osexp",
        "osawr",
        "defect",
    ]

    X_all = frame.drop(
        columns=drop_columns
    ).fillna(
        0
    )

    X = X_all[
        project_spec["features"]
    ].copy()

    train_end = int(
        len(X)
        * settings["official_train_fraction"]
    )

    X_train = X.iloc[
        :train_end
    ].copy()

    y_train = y.iloc[
        :train_end
    ].copy()

    development_end = int(
        len(X_train)
        * settings["development_fraction_of_train"]
    )

    X_dev = X_train.iloc[
        :development_end
    ].copy()

    y_dev = y_train.iloc[
        :development_end
    ].copy()

    smote = SMOTE(
        k_neighbors=settings["smote_k_neighbors"],
        random_state=settings["smote_random_state"],
    )

    X_dev_smote, y_dev_smote = smote.fit_resample(
        X_dev,
        y_dev,
    )

    predictor = RandomForestClassifier(
        n_estimators=settings["rf_n_estimators"],
        random_state=settings["rf_random_state"],
        n_jobs=-1,
    )

    predictor.fit(
        X_dev_smote,
        y_dev_smote,
    )

    X_clean = X_dev.loc[
        y_dev == settings["clean_reference_class"]
    ].copy()

    scaler = RobustScaler()

    X_clean_scaled = scaler.fit_transform(
        X_clean
    )

    loo_model = NearestNeighbors(
        n_neighbors=settings["knn_k"] + 1,
        metric=settings["knn_metric"],
    )

    loo_model.fit(
        X_clean_scaled
    )

    distances, _ = loo_model.kneighbors(
        X_clean_scaled
    )

    tau = float(
        np.percentile(
            distances[
                :,
                1:
            ].mean(
                axis=1
            ),
            settings["threshold_percentile"],
        )
    )

    nn_model = NearestNeighbors(
        n_neighbors=settings["knn_k"],
        metric=settings["knn_metric"],
    )

    nn_model.fit(
        X_clean_scaled
    )

    if len(X_clean) != project_spec[
        "expected_clean_development_rows"
    ]:
        raise RuntimeError(
            f"{project} clean Development row-count mismatch."
        )

    if not np.isclose(
        tau,
        project_spec["frozen_tau"],
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError(
            f"{project} frozen tau mismatch."
        )

    bundle = {
        "bundle_format_version": spec[
            "bundle_format_version"
        ],
        "project": project,
        "predictor": predictor,
        "X_clean": X_clean,
        "scaler": scaler,
        "nn_model": nn_model,
        "final_config_sha256": spec[
            "final_config_sha256"
        ],
        "source_checkpoint_name": spec[
            "source_checkpoint"
        ][
            "name"
        ],
        "source_checkpoint_sha256": spec[
            "source_checkpoint"
        ][
            "sha256"
        ],
    }

    return bundle, X, tau



def sha256_file(
    path: Path,
) -> str:
    hasher = hashlib.sha256()

    with path.open(
        "rb"
    ) as stream:
        while True:
            chunk = stream.read(
                1024 * 1024
            )

            if not chunk:
                break

            hasher.update(
                chunk
            )

    return hasher.hexdigest()


def write_reconstructed_assets(
    source_zip: Path | None,
    output_dir: Path,
) -> dict:
    spec = load_reconstruction_spec()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = {}

    filenames = {
        "OpenStack": "openstack.joblib",
        "Qt": "qt.joblib",
    }

    for project, filename in filenames.items():
        bundle, _, tau = reconstruct_project_bundle(
            source_zip=source_zip,
            project=project,
        )

        output_path = (
            output_dir
            / filename
        )

        joblib.dump(
            bundle,
            output_path,
            compress=3,
        )

        records[
            project
        ] = {
            "path": filename,
            "sha256": sha256_file(
                output_path
            ),
            "size_bytes": output_path.stat().st_size,
            "dataset_sha256": spec[
                "projects"
            ][
                project
            ][
                "dataset_sha256"
            ],
            "frozen_tau": tau,
            "original_bundle_sha256": spec[
                "projects"
            ][
                project
            ][
                "original_bundle_sha256"
            ],
        }

    manifest = {
        "manifest_version": "1.0",
        "asset_type": "locally_reconstructed",
        "scientific_parity_required": True,
        "byte_identical_to_original_required": False,
        "final_config_sha256": spec[
            "final_config_sha256"
        ],
        "reconstruction_spec_sha256": sha256_file(
            SPEC_PATH
        ),
        "bundles": records,
    }

    manifest_path = (
        output_dir
        / "reconstructed_asset_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare CPIT601 runtime assets from "
            "verified PyExplainer source datasets."
        )
    )

    parser.add_argument(
        "--source-zip",
        type=Path,
        required=False,
        default=None,
        help=(
            "Optional local PyExplainer replication-package ZIP. "
            "If omitted, verified datasets are downloaded from "
            "the pinned official upstream commit."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "Directory where reconstructed runtime assets "
            "will be written."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.source_zip is not None:
        validate_source_zip(
            source_zip=args.source_zip,
        )
    else:
        print(
            "Source mode: pinned official upstream commit"
        )

        validate_upstream_sources()

    output_dir = args.output_dir.expanduser().resolve()

    manifest = write_reconstructed_assets(
        source_zip=args.source_zip,
        output_dir=output_dir,
    )

    print()
    print(
        "Runtime assets written to:",
        output_dir,
    )

    print(
        "OpenStack SHA256:",
        manifest["bundles"]["OpenStack"]["sha256"],
    )

    print(
        "Qt SHA256:",
        manifest["bundles"]["Qt"]["sha256"],
    )

    print()
    print(
        "Set CPIT601_RUNTIME_ASSET_DIR to:"
    )

    print(
        output_dir
    )


if __name__ == "__main__":
    main()
