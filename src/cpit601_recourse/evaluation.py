"""Unified counterfactual evaluation utilities.

This module will provide the shared post-generation evaluator used
for all recourse methods. This initial stage only exposes frozen
feature-mask loading.
"""

from __future__ import annotations

import numpy as np

from .config import load_final_config
from .data import get_feature_order


def get_allowed_features(
    project: str,
    configuration: str,
) -> list[str]:
    """Return the frozen evaluation mask for a configuration."""

    config = load_final_config()

    masks = config.get("masks")

    if not isinstance(masks, dict):
        raise RuntimeError(
            "FINAL_CONFIG masks section is invalid."
        )

    # Validate project through frozen feature schema.
    full_schema = get_feature_order(
        project
    )

    configuration_normalized = (
        str(configuration)
        .strip()
        .lower()
    )

    if configuration_normalized == "all":
        return list(full_schema)

    if configuration_normalized not in {
        "strict",
        "extended",
    }:
        raise ValueError(
            "Unsupported evaluation configuration. "
            "Expected 'strict', 'extended', or 'all'; "
            f"received {configuration!r}."
        )

    if project not in masks:
        raise RuntimeError(
            f"No frozen masks found for {project}."
        )

    project_masks = masks[project]

    mask = project_masks.get(
        configuration_normalized
    )

    if not isinstance(mask, list):
        raise RuntimeError(
            f"Missing frozen {configuration_normalized} "
            f"mask for {project}."
        )

    unknown = [
        feature
        for feature in mask
        if feature not in full_schema
    ]

    if unknown:
        raise RuntimeError(
            f"Frozen mask contains unknown features: {unknown}"
        )

    if len(mask) != len(set(mask)):
        raise RuntimeError(
            "Frozen mask contains duplicate features."
        )

    return list(mask)

def get_changed_features(
    query_row,
    candidate_row,
    project: str,
    rtol: float = 1e-6,
    atol: float = 1e-8,
) -> list[str]:
    """Return predictor features changed relative to the query row."""

    feature_order = get_feature_order(
        project
    )

    missing_query = [
        feature
        for feature in feature_order
        if feature not in query_row
    ]

    if missing_query:
        raise ValueError(
            "Query row is missing predictor features: "
            f"{missing_query}"
        )

    missing_candidate = [
        feature
        for feature in feature_order
        if feature not in candidate_row
    ]

    if missing_candidate:
        raise ValueError(
            "Candidate row is missing predictor features: "
            f"{missing_candidate}"
        )

    changed_features = []

    for feature in feature_order:
        query_value = float(
            query_row[feature]
        )

        candidate_value = float(
            candidate_row[feature]
        )

        if not np.isclose(
            query_value,
            candidate_value,
            rtol=rtol,
            atol=atol,
            equal_nan=False,
        ):
            changed_features.append(
                feature
            )

    return changed_features

def evaluate_actionability(
    query_row,
    candidate_row,
    project: str,
    configuration: str,
    rtol: float = 1e-6,
    atol: float = 1e-8,
) -> dict[str, object]:
    """Compute changed-feature and frozen-mask actionability diagnostics."""

    changed_features = get_changed_features(
        query_row,
        candidate_row,
        project,
        rtol=rtol,
        atol=atol,
    )

    allowed_features = get_allowed_features(
        project,
        configuration,
    )

    allowed_feature_set = set(
        allowed_features
    )

    noncontrollable_features = sorted(
        set(changed_features)
        - allowed_feature_set
    )

    n_changed_features = len(
        changed_features
    )

    actionable = (
        n_changed_features > 0
        and len(noncontrollable_features) == 0
    )

    return {
        "changed_features": changed_features,
        "n_changed_features": n_changed_features,
        "noncontrollable_features": noncontrollable_features,
        "actionable": actionable,
    }

def get_integer_like_features(
    project: str,
) -> list[str]:
    """Return frozen Development-derived integer-like features."""

    config = load_final_config()

    semantic_config = config.get(
        "semantic_validity"
    )

    if not isinstance(
        semantic_config,
        dict,
    ):
        raise RuntimeError(
            "FINAL_CONFIG semantic_validity section is invalid."
        )

    feature_order = get_feature_order(
        project
    )

    config_key = (
        f"{project}_integer_like"
    )

    integer_like_features = semantic_config.get(
        config_key
    )

    if not isinstance(
        integer_like_features,
        list,
    ):
        raise RuntimeError(
            f"Missing frozen integer-like features for {project}."
        )

    unknown = [
        feature
        for feature in integer_like_features
        if feature not in feature_order
    ]

    if unknown:
        raise RuntimeError(
            "Frozen integer-like feature list contains "
            f"unknown features: {unknown}"
        )

    if len(integer_like_features) != len(
        set(integer_like_features)
    ):
        raise RuntimeError(
            "Frozen integer-like feature list contains duplicates."
        )

    return list(
        integer_like_features
    )

def evaluate_semantic_validity(
    query_row,
    candidate_row,
    project: str,
    rtol: float = 1e-6,
    atol: float = 1e-8,
) -> dict[str, object]:
    """Evaluate frozen finite-value and integer-like semantic rules."""

    config = load_final_config()

    semantic_config = config.get(
        "semantic_validity"
    )

    if not isinstance(
        semantic_config,
        dict,
    ):
        raise RuntimeError(
            "FINAL_CONFIG semantic_validity section is invalid."
        )

    if semantic_config.get(
        "finite_values_required"
    ) is not True:
        raise RuntimeError(
            "Frozen semantic policy requires finite values."
        )

    if semantic_config.get(
        "changed_integer_like_metrics_must_be_integer"
    ) is not True:
        raise RuntimeError(
            "Frozen semantic policy requires changed "
            "integer-like metrics to remain integer-valued."
        )

    if semantic_config.get(
        "observed_min_max_bounds_used"
    ) is not False:
        raise RuntimeError(
            "Frozen semantic policy must not use "
            "observed min/max bounds."
        )

    feature_order = get_feature_order(
        project
    )

    missing_candidate = [
        feature
        for feature in feature_order
        if feature not in candidate_row
    ]

    if missing_candidate:
        raise ValueError(
            "Candidate row is missing predictor features: "
            f"{missing_candidate}"
        )

    changed_features = get_changed_features(
        query_row,
        candidate_row,
        project,
        rtol=rtol,
        atol=atol,
    )

    integer_like_features = set(
        get_integer_like_features(
            project
        )
    )

    semantic_violation_features = []

    finite_values_ok = True

    for feature in feature_order:
        value = float(
            candidate_row[feature]
        )

        if not np.isfinite(
            value
        ):
            finite_values_ok = False

            semantic_violation_features.append(
                feature
            )

    for feature in changed_features:
        if feature not in integer_like_features:
            continue

        value = float(
            candidate_row[feature]
        )

        if (
            not np.isfinite(
                value
            )
            or not np.isclose(
                value,
                round(value),
                rtol=0.0,
                atol=atol,
            )
        ):
            semantic_violation_features.append(
                feature
            )

    semantic_violation_features = sorted(
        set(
            semantic_violation_features
        )
    )

    discrete_semantics_ok = (
        len(semantic_violation_features) == 0
    )

    semantic_valid = (
        finite_values_ok
        and discrete_semantics_ok
    )

    return {
        "finite_values_ok": finite_values_ok,
        "discrete_semantics_ok": discrete_semantics_ok,
        "semantic_valid": semantic_valid,
        "semantic_violation_features": (
            semantic_violation_features
        ),
    }


def evaluate_counterfactual_set(
    query,
    candidate_features,
    project: str,
    method_name: str,
    configuration: str,
    commit_id=None,
    runtime_seconds=np.nan,
    candidate_count=np.nan,
    seed=np.nan,
    rtol: float = 1e-6,
    atol: float = 1e-8,
):
    """
    Independently evaluate generated counterfactual candidates.

    Returns
    -------
    cf_results : pandas.DataFrame
        One row per candidate with scientific evaluation fields.

    commit_result : pandas.DataFrame
        One-row commit-level summary.

    Notes
    -----
    usable_core =
        valid
        AND empirically_plausible
        AND actionable

    usable =
        usable_core
        AND semantic_valid
    """

    import pandas as pd

    from .config import load_final_config
    from .data import get_feature_order
    from .data import validate_input_frame
    from .plausibility import is_empirically_plausible
    from .plausibility import load_plausibility_context
    from .plausibility import mean_knn_distance
    from .plausibility import plausibility_ratio
    from .plausibility import robust_l1_distance
    from .predictor import predict_defect_probability
    from .predictor import predict_labels

    feature_order = get_feature_order(
        project
    )

    get_allowed_features(
        project,
        configuration,
    )

    config = load_final_config()

    desired_class = (
        config["predictor"]
        ["desired_recourse_class"]
    )

    if (
        not isinstance(
            query,
            pd.DataFrame,
        )
        or len(query) != 1
    ):
        raise ValueError(
            "query must be a one-row pandas DataFrame."
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

    if commit_id is None:
        commit_id = (
            query_features.index[0]
        )

    if (
        candidate_features is None
        or len(candidate_features) == 0
    ):
        empty_cf = pd.DataFrame()

        commit_result = pd.DataFrame(
            [
                {
                    "project": project,
                    "commit_id": commit_id,
                    "method": method_name,
                    "configuration": configuration,
                    "success": False,
                    "n_generated": 0,
                    "n_returned": 0,
                    "n_valid": 0,
                    "n_actionable": 0,
                    "n_plausible": 0,
                    "n_semantically_valid": 0,
                    "n_usable_core": 0,
                    "n_usable": 0,
                    "best_n_changes": np.nan,
                    "best_plausibility_ratio": np.nan,
                    "best_robust_l1": np.nan,
                    "best_defect_probability": np.nan,
                    "runtime_seconds": runtime_seconds,
                    "candidate_count": candidate_count,
                    "seed": seed,
                }
            ]
        )

        return (
            empty_cf,
            commit_result,
        )

    if not isinstance(
        candidate_features,
        pd.DataFrame,
    ):
        candidate_features = pd.DataFrame(
            candidate_features
        )

    missing_candidate_features = [
        feature
        for feature in feature_order
        if feature not in candidate_features.columns
    ]

    if missing_candidate_features:
        raise ValueError(
            "candidate_features is missing predictor features: "
            + str(
                missing_candidate_features
            )
        )

    candidates = (
        candidate_features[
            feature_order
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    numeric_candidates = candidates.apply(
        pd.to_numeric,
        errors="coerce",
    )

    candidate_array = numeric_candidates.to_numpy(
        dtype=float
    )

    finite_values_ok = np.isfinite(
        candidate_array
    ).all(
        axis=1
    )

    n_candidates = len(
        numeric_candidates
    )

    predictions = np.full(
        n_candidates,
        -1,
        dtype=int,
    )

    defect_probabilities = np.full(
        n_candidates,
        np.nan,
        dtype=float,
    )

    knn_distances = np.full(
        n_candidates,
        np.nan,
        dtype=float,
    )

    plausibility_ratios = np.full(
        n_candidates,
        np.nan,
        dtype=float,
    )

    empirical_plausible = np.zeros(
        n_candidates,
        dtype=bool,
    )

    robust_l1 = np.full(
        n_candidates,
        np.nan,
        dtype=float,
    )

    finite_positions = np.where(
        finite_values_ok
    )[0]

    if len(finite_positions) > 0:
        finite_candidates = (
            numeric_candidates.iloc[
                finite_positions
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        finite_predictions = predict_labels(
            project,
            finite_candidates,
        )

        finite_probabilities = predict_defect_probability(
            project,
            finite_candidates,
        )

        finite_knn_distances = mean_knn_distance(
            project,
            finite_candidates,
        )

        finite_ratios = plausibility_ratio(
            project,
            finite_candidates,
        )

        finite_plausible = is_empirically_plausible(
            project,
            finite_candidates,
        )

        repeated_query = pd.concat(
            [
                query_features.reset_index(
                    drop=True
                )
            ]
            * len(finite_candidates),
            ignore_index=True,
        )

        finite_robust_l1 = robust_l1_distance(
            project,
            repeated_query,
            finite_candidates,
        )

        predictions[
            finite_positions
        ] = np.asarray(
            finite_predictions,
            dtype=int,
        )

        defect_probabilities[
            finite_positions
        ] = np.asarray(
            finite_probabilities,
            dtype=float,
        )

        knn_distances[
            finite_positions
        ] = np.asarray(
            finite_knn_distances,
            dtype=float,
        )

        plausibility_ratios[
            finite_positions
        ] = np.asarray(
            finite_ratios,
            dtype=float,
        )

        empirical_plausible[
            finite_positions
        ] = np.asarray(
            finite_plausible,
            dtype=bool,
        )

        robust_l1[
            finite_positions
        ] = np.asarray(
            finite_robust_l1,
            dtype=float,
        )

    valid = (
        finite_values_ok
        &
        (
            predictions
            == desired_class
        )
    )

    query_row = query_features.iloc[
        0
    ]

    changed_feature_lists = []
    n_changed_features = []
    noncontrollable_feature_lists = []
    actionable = []

    discrete_semantics_ok = []
    semantic_valid = []
    semantic_violation_feature_lists = []

    for position in range(
        n_candidates
    ):
        candidate_row = numeric_candidates.iloc[
            position
        ]

        actionability_result = evaluate_actionability(
            query_row,
            candidate_row,
            project,
            configuration,
            rtol=rtol,
            atol=atol,
        )

        semantic_result = evaluate_semantic_validity(
            query_row,
            candidate_row,
            project,
            rtol=rtol,
            atol=atol,
        )

        changed_feature_lists.append(
            actionability_result[
                "changed_features"
            ]
        )

        n_changed_features.append(
            actionability_result[
                "n_changed_features"
            ]
        )

        noncontrollable_feature_lists.append(
            actionability_result[
                "noncontrollable_features"
            ]
        )

        actionable.append(
            actionability_result[
                "actionable"
            ]
        )

        discrete_semantics_ok.append(
            semantic_result[
                "discrete_semantics_ok"
            ]
        )

        semantic_valid.append(
            semantic_result[
                "semantic_valid"
            ]
        )

        semantic_violation_feature_lists.append(
            semantic_result[
                "semantic_violation_features"
            ]
        )

    n_changed_features = np.asarray(
        n_changed_features,
        dtype=int,
    )

    actionable = np.asarray(
        actionable,
        dtype=bool,
    )

    discrete_semantics_ok = np.asarray(
        discrete_semantics_ok,
        dtype=bool,
    )

    semantic_valid = np.asarray(
        semantic_valid,
        dtype=bool,
    )

    duplicate = (
        numeric_candidates
        .duplicated(
            keep=False
        )
        .to_numpy()
    )

    usable_core = (
        valid
        &
        empirical_plausible
        &
        actionable
    )

    usable = (
        usable_core
        &
        semantic_valid
    )

    plausibility_context = load_plausibility_context(
        project
    )

    threshold = float(
        plausibility_context[
            "threshold"
        ]
    )

    cf_rows = []

    for position in range(
        n_candidates
    ):
        cf_rows.append(
            {
                "project": project,
                "commit_id": commit_id,
                "method": method_name,
                "configuration": configuration,
                "cf_rank_input": position + 1,
                "prediction": int(
                    predictions[
                        position
                    ]
                ),
                "valid": bool(
                    valid[
                        position
                    ]
                ),
                "defect_probability": (
                    float(
                        defect_probabilities[
                            position
                        ]
                    )
                    if np.isfinite(
                        defect_probabilities[
                            position
                        ]
                    )
                    else np.nan
                ),
                "n_changed_features": int(
                    n_changed_features[
                        position
                    ]
                ),
                "changed_features": ", ".join(
                    changed_feature_lists[
                        position
                    ]
                ),
                "noncontrollable_features": ", ".join(
                    noncontrollable_feature_lists[
                        position
                    ]
                ),
                "actionable": bool(
                    actionable[
                        position
                    ]
                ),
                "knn_distance": (
                    float(
                        knn_distances[
                            position
                        ]
                    )
                    if np.isfinite(
                        knn_distances[
                            position
                        ]
                    )
                    else np.nan
                ),
                "plausibility_distance": (
                    float(
                        knn_distances[
                            position
                        ]
                    )
                    if np.isfinite(
                        knn_distances[
                            position
                        ]
                    )
                    else np.nan
                ),
                "plausibility_threshold": threshold,
                "plausibility_ratio": (
                    float(
                        plausibility_ratios[
                            position
                        ]
                    )
                    if np.isfinite(
                        plausibility_ratios[
                            position
                        ]
                    )
                    else np.nan
                ),
                "empirically_plausible": bool(
                    empirical_plausible[
                        position
                    ]
                ),
                "robust_l1": (
                    float(
                        robust_l1[
                            position
                        ]
                    )
                    if np.isfinite(
                        robust_l1[
                            position
                        ]
                    )
                    else np.nan
                ),
                "duplicate": bool(
                    duplicate[
                        position
                    ]
                ),
                "finite_values_ok": bool(
                    finite_values_ok[
                        position
                    ]
                ),
                "discrete_semantics_ok": bool(
                    discrete_semantics_ok[
                        position
                    ]
                ),
                "semantic_valid": bool(
                    semantic_valid[
                        position
                    ]
                ),
                "semantic_violation_features": ", ".join(
                    semantic_violation_feature_lists[
                        position
                    ]
                ),
                "usable_core": bool(
                    usable_core[
                        position
                    ]
                ),
                "usable": bool(
                    usable[
                        position
                    ]
                ),
                "runtime_seconds": runtime_seconds,
                "candidate_count": candidate_count,
                "seed": seed,
            }
        )

    cf_results = pd.DataFrame(
        cf_rows
    )

    usable_results = (
        cf_results[
            cf_results[
                "usable"
            ]
        ]
        .copy()
    )

    if len(usable_results) > 0:
        usable_results = usable_results.sort_values(
            by=[
                "n_changed_features",
                "plausibility_ratio",
                "robust_l1",
                "defect_probability",
            ],
            ascending=[
                True,
                True,
                True,
                True,
            ],
        )

        best = usable_results.iloc[
            0
        ]

        best_n_changes = int(
            best[
                "n_changed_features"
            ]
        )

        best_plausibility_ratio = float(
            best[
                "plausibility_ratio"
            ]
        )

        best_robust_l1 = float(
            best[
                "robust_l1"
            ]
        )

        best_defect_probability = float(
            best[
                "defect_probability"
            ]
        )

    else:
        best_n_changes = np.nan
        best_plausibility_ratio = np.nan
        best_robust_l1 = np.nan
        best_defect_probability = np.nan

    commit_result = pd.DataFrame(
        [
            {
                "project": project,
                "commit_id": commit_id,
                "method": method_name,
                "configuration": configuration,
                "success": bool(
                    usable.any()
                ),
                "n_generated": int(
                    n_candidates
                ),
                "n_returned": int(
                    n_candidates
                ),
                "n_valid": int(
                    valid.sum()
                ),
                "n_actionable": int(
                    actionable.sum()
                ),
                "n_plausible": int(
                    empirical_plausible.sum()
                ),
                "n_semantically_valid": int(
                    semantic_valid.sum()
                ),
                "n_usable_core": int(
                    usable_core.sum()
                ),
                "n_usable": int(
                    usable.sum()
                ),
                "best_n_changes": best_n_changes,
                "best_plausibility_ratio": best_plausibility_ratio,
                "best_robust_l1": best_robust_l1,
                "best_defect_probability": best_defect_probability,
                "runtime_seconds": runtime_seconds,
                "candidate_count": candidate_count,
                "seed": seed,
            }
        ]
    )

    return (
        cf_results,
        commit_result,
    )

