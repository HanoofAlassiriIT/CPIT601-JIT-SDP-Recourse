"""Frozen clean-anchor counterfactual generator."""

from itertools import combinations
import time

import numpy as np
import pandas as pd

from .config import load_final_config
from .data import get_feature_order
from .data import validate_input_frame
from .evaluation import get_allowed_features
from .plausibility import is_empirically_plausible
from .plausibility import load_plausibility_context
from .plausibility import mean_knn_distance
from .plausibility import plausibility_ratio
from .plausibility import robust_l1_distance
from .predictor import predict_defect_probability
from .predictor import predict_labels


def generate_clean_anchor_cfs(
    query,
    project: str,
    configuration: str,
) -> dict[str, object]:
    """
    Generate frozen Proposed-method counterfactual recourse.

    Candidate values are copied from nearby real clean Development
    commits and only predefined allowed features may change.

    Scientific settings are loaded from FINAL_CONFIG rather than
    exposed as tunable runtime parameters.
    """

    start_time = time.perf_counter()

    config = load_final_config()

    generator_config = config.get(
        "generator"
    )

    if not isinstance(
        generator_config,
        dict,
    ):
        raise RuntimeError(
            "FINAL_CONFIG generator section is invalid."
        )

    expected_ranking = [
        "n_changed_features",
        "plausibility_ratio",
        "robust_l1",
        "defect_probability",
    ]

    if generator_config.get(
        "anchor_pool"
    ) != "real_clean_Development_commits":
        raise RuntimeError(
            "Frozen Proposed anchor-pool configuration changed."
        )

    if generator_config.get(
        "anchor_retrieval_space"
    ) != "full_predictor_feature_space":
        raise RuntimeError(
            "Frozen Proposed anchor-retrieval space changed."
        )

    ranking = generator_config.get(
        "ranking"
    )

    if ranking != expected_ranking:
        raise RuntimeError(
            "Frozen Proposed ranking changed."
        )

    if generator_config.get(
        "ranking_direction"
    ) != "ascending_lexicographic":
        raise RuntimeError(
            "Frozen Proposed ranking direction changed."
        )

    n_anchors = int(
        generator_config[
            "n_anchors"
        ]
    )

    max_changes = int(
        generator_config[
            "max_changes"
        ]
    )

    top_k = int(
        generator_config[
            "top_k"
        ]
    )

    if n_anchors != 20:
        raise RuntimeError(
            "Frozen Proposed n_anchors changed."
        )

    if max_changes != 3:
        raise RuntimeError(
            "Frozen Proposed max_changes changed."
        )

    if top_k != 3:
        raise RuntimeError(
            "Frozen Proposed top_k changed."
        )

    feature_order = get_feature_order(
        project
    )

    allowed_features = get_allowed_features(
        project,
        configuration,
    )

    if not isinstance(
        query,
        pd.DataFrame,
    ):
        raise TypeError(
            "query must be a pandas DataFrame."
        )

    if len(query) != 1:
        raise ValueError(
            "query must contain exactly one row."
        )

    missing_query_features = [
        feature
        for feature in feature_order
        if feature not in query.columns
    ]

    if missing_query_features:
        raise ValueError(
            "query is missing predictor features: "
            + str(
                missing_query_features
            )
        )

    query_features = (
        query[
            feature_order
        ]
        .copy()
    )

    validate_input_frame(
        project,
        query_features,
    )

    query_row = query_features.iloc[
        0
    ]

    context = load_plausibility_context(
        project
    )

    X_clean = context[
        "X_clean"
    ]

    scaler = context[
        "scaler"
    ]

    nn_model = context[
        "nn_model"
    ]

    if list(
        X_clean.columns
    ) != feature_order:
        raise RuntimeError(
            "Frozen clean Development reference schema mismatch."
        )

    query_scaled = scaler.transform(
        query_features
    )

    _, indices = nn_model.kneighbors(
        query_scaled,
        n_neighbors=min(
            n_anchors,
            len(X_clean),
        ),
    )

    anchor_indices = indices[
        0
    ]

    n_anchors_examined = len(
        anchor_indices
    )

    candidate_records = []

    for anchor_rank, anchor_idx in enumerate(
        anchor_indices,
        start=1,
    ):
        anchor = X_clean.iloc[
            anchor_idx
        ]

        anchor_commit_id = X_clean.index[
            anchor_idx
        ]

        differing_allowed = [
            feature
            for feature in allowed_features
            if not np.isclose(
                float(
                    query_row[
                        feature
                    ]
                ),
                float(
                    anchor[
                        feature
                    ]
                ),
                rtol=1e-6,
                atol=1e-8,
                equal_nan=False,
            )
        ]

        maximum_subset_size = min(
            max_changes,
            len(
                differing_allowed
            ),
        )

        for subset_size in range(
            1,
            maximum_subset_size + 1,
        ):
            for changed_features in combinations(
                differing_allowed,
                subset_size,
            ):
                candidate = query_row.copy()

                for feature in changed_features:
                    candidate[
                        feature
                    ] = anchor[
                        feature
                    ]

                candidate_records.append(
                    {
                        "anchor_rank": anchor_rank,
                        "anchor_commit_id": anchor_commit_id,
                        "changed_features_tuple": changed_features,
                        "candidate": candidate,
                    }
                )

    n_candidates_generated = len(
        candidate_records
    )

    if n_candidates_generated == 0:
        runtime_seconds = (
            time.perf_counter()
            - start_time
        )

        return {
            "counterfactuals": pd.DataFrame(),
            "diagnostics": {
                "n_anchors_examined": int(
                    n_anchors_examined
                ),
                "n_candidates_generated": 0,
                "n_unique_candidates": 0,
                "n_prediction_flips": 0,
                "n_plausible_all": 0,
                "n_valid_and_plausible": 0,
                "n_returned": 0,
                "runtime_seconds": float(
                    runtime_seconds
                ),
            },
        }

    candidates = pd.DataFrame(
        [
            record[
                "candidate"
            ]
            for record in candidate_records
        ]
    )

    candidates = (
        candidates[
            feature_order
        ]
        .reset_index(
            drop=True
        )
    )

    n_unique_candidates = int(
        len(
            candidates.drop_duplicates()
        )
    )

    predictions = np.asarray(
        predict_labels(
            project,
            candidates,
        ),
        dtype=int,
    )

    defect_probabilities = np.asarray(
        predict_defect_probability(
            project,
            candidates,
        ),
        dtype=float,
    )

    valid = (
        predictions == 0
    )

    knn_distances = np.asarray(
        mean_knn_distance(
            project,
            candidates,
        ),
        dtype=float,
    )

    plausibility_ratios = np.asarray(
        plausibility_ratio(
            project,
            candidates,
        ),
        dtype=float,
    )

    plausible = np.asarray(
        is_empirically_plausible(
            project,
            candidates,
        ),
        dtype=bool,
    )

    accepted = (
        valid
        &
        plausible
    )

    repeated_query = pd.concat(
        [
            query_features.reset_index(
                drop=True
            )
        ]
        * len(candidates),
        ignore_index=True,
    )

    robust_l1_all = np.asarray(
        robust_l1_distance(
            project,
            repeated_query,
            candidates,
        ),
        dtype=float,
    )

    evaluated = []

    accepted_indices = np.where(
        accepted
    )[0]

    for candidate_index in accepted_indices:
        record = candidate_records[
            candidate_index
        ]

        changed_features = record[
            "changed_features_tuple"
        ]

        candidate_row = candidates.iloc[
            candidate_index
        ]

        changes_text = " | ".join(
            [
                (
                    f"{feature}: "
                    f"{float(query_row[feature]):.4f}"
                    f" → "
                    f"{float(candidate_row[feature]):.4f}"
                )
                for feature in changed_features
            ]
        )

        evaluated.append(
            {
                "anchor_rank": record[
                    "anchor_rank"
                ],
                "anchor_commit_id": record[
                    "anchor_commit_id"
                ],
                "n_changed_features": len(
                    changed_features
                ),
                "changed_features": ", ".join(
                    changed_features
                ),
                "changes": changes_text,
                "defect_probability": float(
                    defect_probabilities[
                        candidate_index
                    ]
                ),
                "knn_distance": float(
                    knn_distances[
                        candidate_index
                    ]
                ),
                "plausibility_ratio": float(
                    plausibility_ratios[
                        candidate_index
                    ]
                ),
                "robust_l1": float(
                    robust_l1_all[
                        candidate_index
                    ]
                ),
                "candidate": candidate_row.copy(),
            }
        )

    if len(
        evaluated
    ) == 0:
        counterfactuals = pd.DataFrame()

    else:
        counterfactuals = (
            pd.DataFrame(
                evaluated
            )
            .sort_values(
                by=ranking,
                ascending=[
                    True,
                    True,
                    True,
                    True,
                ],
            )
            .reset_index(
                drop=True
            )
            .head(
                top_k
            )
            .reset_index(
                drop=True
            )
        )

    runtime_seconds = (
        time.perf_counter()
        - start_time
    )

    diagnostics = {
        "n_anchors_examined": int(
            n_anchors_examined
        ),
        "n_candidates_generated": int(
            n_candidates_generated
        ),
        "n_unique_candidates": int(
            n_unique_candidates
        ),
        "n_prediction_flips": int(
            valid.sum()
        ),
        "n_plausible_all": int(
            plausible.sum()
        ),
        "n_valid_and_plausible": int(
            accepted.sum()
        ),
        "n_returned": int(
            len(
                counterfactuals
            )
        ),
        "runtime_seconds": float(
            runtime_seconds
        ),
    }

    return {
        "counterfactuals": counterfactuals,
        "diagnostics": diagnostics,
    }
