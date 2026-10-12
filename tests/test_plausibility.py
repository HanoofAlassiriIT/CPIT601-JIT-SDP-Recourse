"""Unit tests for frozen empirical-plausibility behavior."""

import numpy as np
import pytest

from cpit601_recourse.data import get_feature_order
from cpit601_recourse.plausibility import (
    is_empirically_plausible,
    load_plausibility_context,
    mean_knn_distance,
    plausibility_ratio,
    robust_l1_distance,
)


EXPECTED_THRESHOLDS = {
    "OpenStack": 7.947140456773377,
    "Qt": 22.183025376655454,
}


@pytest.mark.parametrize(
    "project",
    [
        "OpenStack",
        "Qt",
    ],
)
def test_frozen_plausibility_context(project):
    context = load_plausibility_context(
        project
    )

    assert context["project"] == project
    assert context["k"] == 5

    assert np.isclose(
        context["threshold"],
        EXPECTED_THRESHOLDS[project],
        rtol=0.0,
        atol=1e-12,
    )

    assert context["feature_order"] == get_feature_order(
        project
    )


@pytest.mark.parametrize(
    "project",
    [
        "OpenStack",
        "Qt",
    ],
)
def test_plausibility_metrics_are_consistent(project):
    context = load_plausibility_context(
        project
    )

    sample = context["X_clean"].iloc[
        [0]
    ].copy()

    distance = mean_knn_distance(
        project,
        sample,
    )

    ratio = plausibility_ratio(
        project,
        sample,
    )

    plausible = is_empirically_plausible(
        project,
        sample,
    )

    robust_l1 = robust_l1_distance(
        project,
        sample,
        sample,
    )

    assert distance.shape == (1,)
    assert ratio.shape == (1,)
    assert plausible.shape == (1,)
    assert robust_l1.shape == (1,)

    assert np.isclose(
        ratio[0],
        distance[0] / context["threshold"],
        rtol=1e-12,
        atol=1e-12,
    )

    assert bool(
        plausible[0]
    ) == bool(
        ratio[0] <= 1.0
    )

    assert np.isclose(
        robust_l1[0],
        0.0,
        rtol=0.0,
        atol=1e-12,
    )
