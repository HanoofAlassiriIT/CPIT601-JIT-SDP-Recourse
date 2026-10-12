"""Small end-to-end smoke test for Proposed recourse."""

import numpy as np
import pandas as pd

from cpit601_recourse import run_recourse
from cpit601_recourse.predictor import (
    load_project_bundle,
    predict_labels,
)


def _build_deterministic_smoke_query():
    project = "OpenStack"

    bundle = load_project_bundle(
        project
    )

    X_clean = bundle[
        "X_clean"
    ].reset_index(
        drop=True
    )

    rng = np.random.default_rng(
        42
    )

    n_synthetic = 2000

    synthetic = pd.DataFrame(
        index=range(
            n_synthetic
        ),
        columns=X_clean.columns,
        dtype=float,
    )

    for feature in X_clean.columns:
        sampled_positions = rng.integers(
            0,
            len(
                X_clean
            ),
            size=n_synthetic,
        )

        synthetic[
            feature
        ] = X_clean[
            feature
        ].to_numpy()[
            sampled_positions
        ]

    predictions = predict_labels(
        project,
        synthetic,
    )

    defective_positions = np.where(
        np.asarray(
            predictions
        ) == 1
    )[0]

    assert len(
        defective_positions
    ) > 0

    smoke_position = int(
        defective_positions[
            0
        ]
    )

    query = synthetic.iloc[
        [
            smoke_position
        ]
    ].copy()

    return query


def test_proposed_strict_end_to_end_smoke():
    query = _build_deterministic_smoke_query()

    result = run_recourse(
        query=query,
        project="OpenStack",
        method="Proposed",
        configuration="Strict",
        commit_id="unit-test-synthetic",
    )

    assert result["project"] == "OpenStack"

    assert result["method"] == "Proposed-Strict"

    assert result["configuration"] == "Strict"

    assert result["commit_id"] == "unit-test-synthetic"

    assert int(
        result["original_prediction"]
    ) == 1

    assert bool(
        result["recourse_success"]
    ) is True

    assert int(
        result["n_returned_counterfactuals"]
    ) == 3

    assert len(
        result["counterfactuals"]
    ) == 3

    for counterfactual in result[
        "counterfactuals"
    ]:
        assert bool(
            counterfactual["usable"]
        ) is True

        assert bool(
            counterfactual["actionable"]
        ) is True

        assert bool(
            counterfactual["semantic_valid"]
        ) is True

        assert float(
            counterfactual["plausibility_ratio"]
        ) <= 1.0
