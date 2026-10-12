"""Unit tests for frozen counterfactual evaluation behavior."""

from cpit601_recourse.evaluation import (
    evaluate_actionability,
    evaluate_semantic_validity,
    get_changed_features,
)

from cpit601_recourse.plausibility import (
    load_plausibility_context,
)


def _openstack_clean_row():
    context = load_plausibility_context(
        "OpenStack"
    )

    return context["X_clean"].iloc[
        0
    ].copy()


def test_allowed_strict_change_is_actionable():
    query = _openstack_clean_row()

    candidate = query.copy()

    candidate["la"] = (
        candidate["la"]
        + 1
    )

    changed = get_changed_features(
        query,
        candidate,
        "OpenStack",
    )

    actionability = evaluate_actionability(
        query,
        candidate,
        "OpenStack",
        "Strict",
    )

    assert changed == [
        "la"
    ]

    assert actionability["changed_features"] == [
        "la"
    ]

    assert actionability["n_changed_features"] == 1

    assert actionability["noncontrollable_features"] == []

    assert actionability["actionable"] is True


def test_noncontrollable_strict_change_is_rejected():
    query = _openstack_clean_row()

    candidate = query.copy()

    candidate["nrev"] = (
        candidate["nrev"]
        + 1
    )

    actionability = evaluate_actionability(
        query,
        candidate,
        "OpenStack",
        "Strict",
    )

    assert actionability["changed_features"] == [
        "nrev"
    ]

    assert actionability["n_changed_features"] == 1

    assert actionability["noncontrollable_features"] == [
        "nrev"
    ]

    assert actionability["actionable"] is False


def test_fractional_integer_like_change_is_semantically_invalid():
    query = _openstack_clean_row()

    candidate = query.copy()

    candidate["la"] = (
        candidate["la"]
        + 0.5
    )

    semantic = evaluate_semantic_validity(
        query,
        candidate,
        "OpenStack",
    )

    assert semantic["finite_values_ok"] is True

    assert semantic["discrete_semantics_ok"] is False

    assert semantic["semantic_valid"] is False

    assert semantic["semantic_violation_features"] == [
        "la"
    ]


def test_unchanged_clean_row_is_semantically_valid():
    query = _openstack_clean_row()

    semantic = evaluate_semantic_validity(
        query,
        query.copy(),
        "OpenStack",
    )

    assert semantic["finite_values_ok"] is True

    assert semantic["discrete_semantics_ok"] is True

    assert semantic["semantic_valid"] is True

    assert semantic["semantic_violation_features"] == []



def test_unified_evaluator_empty_candidate_set_is_failure():
    import pandas as pd

    from cpit601_recourse.evaluation import (
        evaluate_counterfactual_set,
    )

    from cpit601_recourse.plausibility import (
        load_plausibility_context,
    )

    context = load_plausibility_context(
        "OpenStack"
    )

    query = context["X_clean"].iloc[
        [0]
    ].copy()

    empty_candidates = pd.DataFrame(
        columns=context["feature_order"]
    )

    cf_level, commit_level = evaluate_counterfactual_set(
        query=query,
        candidate_features=empty_candidates,
        project="OpenStack",
        method_name="UnitTest",
        configuration="Strict",
    )

    assert cf_level.shape == (
        0,
        0,
    )

    assert len(
        commit_level
    ) == 1

    row = commit_level.iloc[
        0
    ]

    assert bool(
        row["success"]
    ) is False

    assert int(
        row["n_generated"]
    ) == 0

    assert int(
        row["n_returned"]
    ) == 0

    assert int(
        row["n_usable"]
    ) == 0


def test_identical_clean_candidate_is_valid_but_not_usable():
    from cpit601_recourse.evaluation import (
        evaluate_counterfactual_set,
    )

    from cpit601_recourse.plausibility import (
        load_plausibility_context,
    )

    context = load_plausibility_context(
        "OpenStack"
    )

    query = context["X_clean"].iloc[
        [0]
    ].copy()

    candidate = query[
        context["feature_order"]
    ].copy()

    cf_level, commit_level = evaluate_counterfactual_set(
        query=query,
        candidate_features=candidate,
        project="OpenStack",
        method_name="UnitTest",
        configuration="Strict",
    )

    assert len(
        cf_level
    ) == 1

    cf_row = cf_level.iloc[
        0
    ]

    assert int(
        cf_row["prediction"]
    ) == 0

    assert bool(
        cf_row["valid"]
    ) is True

    assert bool(
        cf_row["empirically_plausible"]
    ) is True

    assert bool(
        cf_row["semantic_valid"]
    ) is True

    assert int(
        cf_row["n_changed_features"]
    ) == 0

    assert bool(
        cf_row["actionable"]
    ) is False

    assert bool(
        cf_row["usable_core"]
    ) is False

    assert bool(
        cf_row["usable"]
    ) is False

    commit_row = commit_level.iloc[
        0
    ]

    assert bool(
        commit_row["success"]
    ) is False

    assert int(
        commit_row["n_valid"]
    ) == 1

    assert int(
        commit_row["n_plausible"]
    ) == 1

    assert int(
        commit_row["n_semantically_valid"]
    ) == 1

    assert int(
        commit_row["n_actionable"]
    ) == 0

    assert int(
        commit_row["n_usable"]
    ) == 0
