"""Invalid-input tests for the public recourse API."""

import numpy as np
import pytest

from cpit601_recourse import run_recourse

from cpit601_recourse.plausibility import (
    load_plausibility_context,
)


def _valid_openstack_query():
    context = load_plausibility_context(
        "OpenStack"
    )

    return context["X_clean"].iloc[
        [0]
    ].copy()


def test_rejects_unsupported_project():
    query = _valid_openstack_query()

    with pytest.raises(
        ValueError,
        match="Unsupported project",
    ):
        run_recourse(
            query=query,
            project="UnknownProject",
            method="Proposed",
            configuration="Strict",
        )


def test_rejects_unsupported_configuration():
    query = _valid_openstack_query()

    with pytest.raises(
        ValueError,
        match="configuration must be Strict, Extended, or All",
    ):
        run_recourse(
            query=query,
            project="OpenStack",
            method="Proposed",
            configuration="Unsupported",
        )


def test_rejects_missing_predictor_feature():
    query = _valid_openstack_query()

    query = query.drop(
        columns=[
            "la"
        ]
    )

    with pytest.raises(
        ValueError,
        match="query is missing predictor features",
    ):
        run_recourse(
            query=query,
            project="OpenStack",
            method="Proposed",
            configuration="Strict",
        )


def test_rejects_nonfinite_query_value():
    query = _valid_openstack_query()

    query.loc[
        query.index[
            0
        ],
        "la",
    ] = np.nan

    with pytest.raises(
        ValueError,
        match="missing or non-finite values",
    ):
        run_recourse(
            query=query,
            project="OpenStack",
            method="Proposed",
            configuration="Strict",
        )
