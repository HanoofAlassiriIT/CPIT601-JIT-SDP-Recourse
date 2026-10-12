"""Supported-project feature schema utilities."""

from __future__ import annotations

from .config import load_final_config


def get_feature_order(
    project: str,
) -> list[str]:
    """Return the frozen predictor feature order for a project."""

    config = load_final_config()

    feature_order = config.get(
        "feature_order"
    )

    if not isinstance(feature_order, dict):
        raise RuntimeError(
            "FINAL_CONFIG feature_order section is invalid."
        )

    if project not in feature_order:
        supported = tuple(
            feature_order.keys()
        )

        raise ValueError(
            "Unsupported project. "
            f"Expected one of {supported}; "
            f"received {project!r}."
        )

    schema = feature_order[project]

    if not isinstance(schema, list):
        raise RuntimeError(
            f"{project} feature order must be a list."
        )

    if not all(
        isinstance(name, str)
        for name in schema
    ):
        raise RuntimeError(
            f"{project} feature order contains invalid entries."
        )

    if len(schema) != len(set(schema)):
        raise RuntimeError(
            f"{project} feature order contains duplicates."
        )

    return list(schema)



def validate_input_frame(
    project: str,
    X,
):
    """Validate supported-project predictor input without modifying it."""

    import numpy as np
    import pandas as pd
    from pandas.api.types import is_numeric_dtype

    expected = get_feature_order(project)

    if not isinstance(X, pd.DataFrame):
        raise TypeError(
            "Input X must be a pandas DataFrame."
        )

    if len(X) == 0:
        raise ValueError(
            "Input X must contain at least one row."
        )

    actual = list(X.columns)

    missing = [
        name
        for name in expected
        if name not in actual
    ]

    extra = [
        name
        for name in actual
        if name not in expected
    ]

    if missing:
        raise ValueError(
            f"Missing required features: {missing}"
        )

    if extra:
        raise ValueError(
            f"Unexpected features: {extra}"
        )

    if actual != expected:
        raise ValueError(
            "Feature order mismatch. "
            f"Expected {expected}; received {actual}."
        )

    non_numeric = [
        name
        for name in expected
        if not is_numeric_dtype(X[name])
    ]

    if non_numeric:
        raise TypeError(
            f"Non-numeric features: {non_numeric}"
        )

    values = X.to_numpy(
        dtype=float,
        copy=True,
    )

    if not np.isfinite(values).all():
        raise ValueError(
            "Input contains missing or non-finite values."
        )

    return X
