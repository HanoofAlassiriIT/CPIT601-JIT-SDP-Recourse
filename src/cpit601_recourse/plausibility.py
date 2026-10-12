"""Frozen empirical-plausibility asset loading.

This module loads already-fitted plausibility objects and frozen
configuration values. It performs no fitting, transformation,
nearest-neighbor query, distance calculation, or threshold estimation.
"""

from __future__ import annotations

from typing import Any

from .config import load_final_config
from .data import get_feature_order
from .predictor import (
    load_project_bundle,
    validate_project,
)


def load_plausibility_context(
    project: str,
) -> dict[str, Any]:
    """Load and verify the frozen plausibility context."""

    project = validate_project(project)

    bundle = load_project_bundle(
        project
    )

    config = load_final_config()

    plausibility_config = config.get(
        "plausibility"
    )

    if not isinstance(
        plausibility_config,
        dict,
    ):
        raise RuntimeError(
            "FINAL_CONFIG plausibility section is invalid."
        )

    thresholds = plausibility_config.get(
        "threshold"
    )

    if not isinstance(
        thresholds,
        dict,
    ):
        raise RuntimeError(
            "FINAL_CONFIG plausibility thresholds are invalid."
        )

    if project not in thresholds:
        raise RuntimeError(
            f"No frozen plausibility threshold for {project}."
        )

    k = plausibility_config.get("k")
    threshold = thresholds[project]

    feature_order = get_feature_order(
        project
    )

    scaler = bundle["scaler"]
    X_clean = bundle["X_clean"]
    nn_model = bundle["nn_model"]

    # --------------------------------------------------------
    # Metadata consistency only.
    # No scientific computation is performed here.
    # --------------------------------------------------------

    if list(X_clean.columns) != feature_order:
        raise RuntimeError(
            f"{project} clean-reference schema mismatch."
        )

    if scaler.n_features_in_ != len(feature_order):
        raise RuntimeError(
            f"{project} scaler feature-count mismatch."
        )

    if nn_model.n_features_in_ != len(feature_order):
        raise RuntimeError(
            f"{project} neighbor-model feature-count mismatch."
        )

    if nn_model.n_neighbors != k:
        raise RuntimeError(
            f"{project} frozen k mismatch."
        )

    if str(nn_model.metric).lower() != str(
        plausibility_config["distance"]
    ).lower():
        raise RuntimeError(
            f"{project} distance-metric mismatch."
        )

    return {
        "project": project,
        "feature_order": list(feature_order),
        "k": k,
        "threshold": float(threshold),
        "scaler": scaler,
        "X_clean": X_clean,
        "nn_model": nn_model,
    }



def transform_for_plausibility(
    project: str,
    X,
):
    """Transform validated inputs with the frozen clean-Development scaler."""

    import numpy as np

    from .data import validate_input_frame

    validated_X = validate_input_frame(
        project,
        X,
    )

    context = load_plausibility_context(
        project
    )

    transformed = context["scaler"].transform(
        validated_X
    )

    transformed = np.asarray(
        transformed,
        dtype=float,
    )

    expected_shape = (
        len(validated_X),
        len(context["feature_order"]),
    )

    if transformed.shape != expected_shape:
        raise RuntimeError(
            "Frozen scaler returned an unexpected shape."
        )

    if not np.isfinite(transformed).all():
        raise RuntimeError(
            "Frozen scaler returned non-finite values."
        )

    return transformed



def mean_knn_distance(
    project: str,
    X,
):
    """Return mean distance to frozen k nearest clean Development commits."""

    import numpy as np

    transformed = transform_for_plausibility(
        project,
        X,
    )

    context = load_plausibility_context(
        project
    )

    k = context["k"]

    distances, _ = context["nn_model"].kneighbors(
        transformed,
        n_neighbors=k,
        return_distance=True,
    )

    distances = np.asarray(
        distances,
        dtype=float,
    )

    if distances.shape != (
        len(X),
        k,
    ):
        raise RuntimeError(
            "Frozen neighbor query returned an unexpected shape."
        )

    if not np.isfinite(distances).all():
        raise RuntimeError(
            "Frozen neighbor query returned non-finite distances."
        )

    if (distances < 0).any():
        raise RuntimeError(
            "Frozen neighbor query returned negative distances."
        )

    mean_distance = distances.mean(
        axis=1
    )

    return mean_distance



def plausibility_ratio(
    project: str,
    X,
):
    """Return frozen empirical plausibility ratio d_k(x) / tau."""

    import numpy as np

    mean_distance = mean_knn_distance(
        project,
        X,
    )

    context = load_plausibility_context(
        project
    )

    threshold = float(
        context["threshold"]
    )

    if (
        not np.isfinite(threshold)
        or threshold <= 0
    ):
        raise RuntimeError(
            "Frozen plausibility threshold must be finite and positive."
        )

    ratio = (
        np.asarray(
            mean_distance,
            dtype=float,
        )
        / threshold
    )

    if not np.isfinite(ratio).all():
        raise RuntimeError(
            "Plausibility ratio contains non-finite values."
        )

    if (ratio < 0).any():
        raise RuntimeError(
            "Plausibility ratio contains negative values."
        )

    return ratio



def _ratio_is_accepted(
    ratio,
    cutoff: float,
):
    """Apply the frozen inclusive plausibility cutoff."""

    import numpy as np

    ratio = np.asarray(
        ratio,
        dtype=float,
    )

    if not np.isfinite(ratio).all():
        raise ValueError(
            "Plausibility ratio must contain only finite values."
        )

    if not np.isfinite(cutoff):
        raise RuntimeError(
            "Frozen plausibility cutoff must be finite."
        )

    return ratio <= cutoff


def is_empirically_plausible(
    project: str,
    X,
):
    """Return Boolean empirical-plausibility decisions."""

    from .config import load_final_config

    ratio = plausibility_ratio(
        project,
        X,
    )

    config = load_final_config()

    cutoff = float(
        config["plausibility"]
        ["accept_if_ratio_leq"]
    )

    return _ratio_is_accepted(
        ratio,
        cutoff,
    )



def robust_l1_distance(
    project: str,
    query_X,
    candidate_X,
):
    """Return robust L1 distance in frozen clean-scaled feature space."""

    import numpy as np

    if len(query_X) != len(candidate_X):
        raise ValueError(
            "query_X and candidate_X must contain the same number of rows."
        )

    query_scaled = transform_for_plausibility(
        project,
        query_X,
    )

    candidate_scaled = transform_for_plausibility(
        project,
        candidate_X,
    )

    if query_scaled.shape != candidate_scaled.shape:
        raise RuntimeError(
            "Scaled query and candidate shapes do not match."
        )

    distance = np.abs(
        candidate_scaled - query_scaled
    ).sum(axis=1)

    if not np.isfinite(distance).all():
        raise RuntimeError(
            "Robust L1 distance contains non-finite values."
        )

    if (distance < 0).any():
        raise RuntimeError(
            "Robust L1 distance contains negative values."
        )

    return distance
