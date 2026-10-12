"""Statistical utilities for frozen commit-level recourse results."""

import numpy as np
import pandas as pd

from scipy.stats import binomtest


def _paired_bootstrap_mean_ci(
    differences,
    seed=42,
    replicates=100000,
    confidence=0.95,
    batch_size=5000,
):
    """Paired bootstrap confidence interval for a mean difference."""

    values = np.asarray(
        differences,
        dtype=float,
    )

    if values.ndim != 1 or len(values) == 0:
        raise ValueError(
            "differences must be a non-empty one-dimensional array."
        )

    if replicates <= 0:
        raise ValueError(
            "replicates must be positive."
        )

    rng = np.random.default_rng(
        seed
    )

    bootstrap_means = np.empty(
        replicates,
        dtype=float,
    )

    completed = 0

    while completed < replicates:
        current_batch = min(
            batch_size,
            replicates - completed,
        )

        indices = rng.integers(
            0,
            len(values),
            size=(
                current_batch,
                len(values),
            ),
        )

        bootstrap_means[
            completed:completed + current_batch
        ] = values[
            indices
        ].mean(
            axis=1
        )

        completed += current_batch

    alpha = 1.0 - confidence

    lower = np.quantile(
        bootstrap_means,
        alpha / 2.0,
    )

    upper = np.quantile(
        bootstrap_means,
        1.0 - alpha / 2.0,
    )

    return float(
        lower
    ), float(
        upper
    )


def summarize_paired_binary_outcome(
    proposed_commit_results,
    dice_commit_results,
    id_column="commit_id",
    success_column="success",
    bootstrap_seed=42,
    bootstrap_replicates=100000,
):
    """Summarize paired usable-recourse success at commit level."""

    proposed = proposed_commit_results.copy()
    dice = dice_commit_results.copy()

    required_columns = {
        id_column,
        success_column,
    }

    if not required_columns.issubset(
        proposed.columns
    ):
        raise ValueError(
            "Proposed results are missing required columns."
        )

    if not required_columns.issubset(
        dice.columns
    ):
        raise ValueError(
            "DiCE results are missing required columns."
        )

    proposed[id_column] = proposed[
        id_column
    ].astype(
        str
    )

    dice[id_column] = dice[
        id_column
    ].astype(
        str
    )

    if not proposed[
        id_column
    ].is_unique:
        raise ValueError(
            "Proposed commit IDs must be unique."
        )

    if not dice[
        id_column
    ].is_unique:
        raise ValueError(
            "DiCE commit IDs must be unique."
        )

    if set(
        proposed[id_column]
    ) != set(
        dice[id_column]
    ):
        raise ValueError(
            "Paired methods must contain identical commit IDs."
        )

    paired = proposed[
        [
            id_column,
            success_column,
        ]
    ].rename(
        columns={
            success_column:
                "proposed_success",
        }
    ).merge(
        dice[
            [
                id_column,
                success_column,
            ]
        ].rename(
            columns={
                success_column:
                    "dice_success",
            }
        ),
        on=id_column,
        how="left",
        validate="one_to_one",
        sort=False,
    )

    proposed_success = paired[
        "proposed_success"
    ].astype(
        int
    ).to_numpy()

    dice_success = paired[
        "dice_success"
    ].astype(
        int
    ).to_numpy()

    if not np.isin(
        proposed_success,
        [
            0,
            1,
        ],
    ).all():
        raise ValueError(
            "Proposed success values must be binary."
        )

    if not np.isin(
        dice_success,
        [
            0,
            1,
        ],
    ).all():
        raise ValueError(
            "DiCE success values must be binary."
        )

    both_success = int(
        np.sum(
            (proposed_success == 1)
            & (dice_success == 1)
        )
    )

    proposed_only = int(
        np.sum(
            (proposed_success == 1)
            & (dice_success == 0)
        )
    )

    dice_only = int(
        np.sum(
            (proposed_success == 0)
            & (dice_success == 1)
        )
    )

    both_fail = int(
        np.sum(
            (proposed_success == 0)
            & (dice_success == 0)
        )
    )

    n_commits = len(
        paired
    )

    proposed_coverage = float(
        proposed_success.mean()
    )

    dice_coverage = float(
        dice_success.mean()
    )

    differences = (
        proposed_success
        - dice_success
    )

    delta_coverage = float(
        differences.mean()
    )

    ci_low, ci_high = (
        _paired_bootstrap_mean_ci(
            differences,
            seed=bootstrap_seed,
            replicates=bootstrap_replicates,
        )
    )

    discordant_pairs = (
        proposed_only
        + dice_only
    )

    if discordant_pairs == 0:
        mcnemar_p = 1.0

    else:
        mcnemar_p = float(
            binomtest(
                proposed_only,
                n=discordant_pairs,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )

    return {
        "n_commits": n_commits,
        "proposed_successes": int(
            proposed_success.sum()
        ),
        "dice_successes": int(
            dice_success.sum()
        ),
        "both_success": both_success,
        "proposed_only": proposed_only,
        "dice_only": dice_only,
        "both_fail": both_fail,
        "discordant_pairs": discordant_pairs,
        "proposed_coverage": proposed_coverage,
        "dice_coverage": dice_coverage,
        "delta_coverage": delta_coverage,
        "delta_coverage_pp": (
            100.0
            * delta_coverage
        ),
        "bootstrap_ci95_low": ci_low,
        "bootstrap_ci95_high": ci_high,
        "bootstrap_ci95_low_pp": (
            100.0
            * ci_low
        ),
        "bootstrap_ci95_high_pp": (
            100.0
            * ci_high
        ),
        "exact_mcnemar_two_sided_p": mcnemar_p,
        "bootstrap_seed": int(
            bootstrap_seed
        ),
        "bootstrap_replicates": int(
            bootstrap_replicates
        ),
    }



def build_both_success_quality_pairs(
    proposed_commit_results,
    dice_commit_results,
    id_column="commit_id",
    success_column="success",
):
    """Build one paired best-CF quality row per both-success commit."""

    proposed = proposed_commit_results.copy()
    dice = dice_commit_results.copy()

    metric_map = {
        "n_changed_features":
            "best_n_changes",
        "plausibility_ratio":
            "best_plausibility_ratio",
        "robust_l1":
            "best_robust_l1",
        "defect_probability":
            "best_defect_probability",
    }

    required_columns = {
        id_column,
        success_column,
        *metric_map.values(),
    }

    if not required_columns.issubset(
        proposed.columns
    ):
        raise ValueError(
            "Proposed results are missing required quality columns."
        )

    if not required_columns.issubset(
        dice.columns
    ):
        raise ValueError(
            "DiCE results are missing required quality columns."
        )

    proposed[id_column] = proposed[
        id_column
    ].astype(
        str
    )

    dice[id_column] = dice[
        id_column
    ].astype(
        str
    )

    if not proposed[
        id_column
    ].is_unique:
        raise ValueError(
            "Proposed commit IDs must be unique."
        )

    if not dice[
        id_column
    ].is_unique:
        raise ValueError(
            "DiCE commit IDs must be unique."
        )

    if set(
        proposed[id_column]
    ) != set(
        dice[id_column]
    ):
        raise ValueError(
            "Paired methods must contain identical commit IDs."
        )

    proposed_columns = [
        id_column,
        success_column,
        *metric_map.values(),
    ]

    dice_columns = [
        id_column,
        success_column,
        *metric_map.values(),
    ]

    paired = proposed[
        proposed_columns
    ].merge(
        dice[
            dice_columns
        ],
        on=id_column,
        how="left",
        suffixes=(
            "_proposed",
            "_dice",
        ),
        validate="one_to_one",
        sort=False,
    )

    both_success = paired[
        paired[
            f"{success_column}_proposed"
        ].astype(
            bool
        )
        & paired[
            f"{success_column}_dice"
        ].astype(
            bool
        )
    ].copy()

    output = pd.DataFrame()

    output[
        "commit_id"
    ] = both_success[
        id_column
    ].astype(
        str
    )

    for output_metric, source_metric in metric_map.items():
        proposed_column = (
            f"{source_metric}_proposed"
        )

        dice_column = (
            f"{source_metric}_dice"
        )

        output[
            f"proposed_{output_metric}"
        ] = pd.to_numeric(
            both_success[
                proposed_column
            ],
            errors="raise",
        )

        output[
            f"dice_{output_metric}"
        ] = pd.to_numeric(
            both_success[
                dice_column
            ],
            errors="raise",
        )

        output[
            f"diff_{output_metric}"
        ] = (
            output[
                f"proposed_{output_metric}"
            ]
            - output[
                f"dice_{output_metric}"
            ]
        )

    return output.reset_index(
        drop=True
    )



def _paired_bootstrap_median_ci(
    differences,
    seed=42,
    replicates=100000,
    batch_size=5000,
):
    """Paired-bootstrap CI for the median paired difference."""

    values = np.asarray(
        differences,
        dtype=float,
    )

    if values.ndim != 1 or len(values) == 0:
        raise ValueError(
            "differences must be a non-empty one-dimensional array."
        )

    if replicates <= 0:
        raise ValueError(
            "replicates must be positive."
        )

    rng = np.random.default_rng(
        seed
    )

    medians = np.empty(
        replicates,
        dtype=float,
    )

    completed = 0

    while completed < replicates:
        current_batch = min(
            batch_size,
            replicates - completed,
        )

        indices = rng.integers(
            0,
            len(values),
            size=(
                current_batch,
                len(values),
            ),
        )

        medians[
            completed:completed + current_batch
        ] = np.median(
            values[
                indices
            ],
            axis=1,
        )

        completed += current_batch

    lower, upper = np.quantile(
        medians,
        [
            0.025,
            0.975,
        ],
    )

    return float(
        lower
    ), float(
        upper
    )


def _matched_pairs_rank_biserial(
    differences,
):
    """Matched-pairs rank-biserial correlation."""

    diff = np.asarray(
        differences,
        dtype=float,
    )

    nonzero = diff[
        diff != 0
    ]

    if len(nonzero) == 0:
        return 0.0

    ranks = (
        pd.Series(
            np.abs(
                nonzero
            )
        )
        .rank(
            method="average"
        )
        .to_numpy()
    )

    positive_rank_sum = float(
        ranks[
            nonzero > 0
        ].sum()
    )

    negative_rank_sum = float(
        ranks[
            nonzero < 0
        ].sum()
    )

    denominator = (
        positive_rank_sum
        + negative_rank_sum
    )

    if denominator == 0:
        return 0.0

    return float(
        (
            positive_rank_sum
            - negative_rank_sum
        )
        / denominator
    )


def _holm_adjust(
    p_values,
):
    """Holm-adjust one family of p-values."""

    p = np.asarray(
        p_values,
        dtype=float,
    )

    order = np.argsort(
        p
    )

    adjusted_sorted = np.empty(
        len(p),
        dtype=float,
    )

    running_max = 0.0

    for rank_position, original_index in enumerate(
        order
    ):
        multiplier = (
            len(p)
            - rank_position
        )

        candidate = min(
            1.0,
            multiplier
            * p[
                original_index
            ],
        )

        running_max = max(
            running_max,
            candidate,
        )

        adjusted_sorted[
            rank_position
        ] = running_max

    adjusted = np.empty(
        len(p),
        dtype=float,
    )

    for rank_position, original_index in enumerate(
        order
    ):
        adjusted[
            original_index
        ] = adjusted_sorted[
            rank_position
        ]

    return adjusted


def summarize_paired_quality(
    project_pairs,
    bootstrap_seed=42,
    bootstrap_replicates=100000,
):
    """Summarize the frozen paired quality family across projects."""

    from scipy.stats import wilcoxon

    metrics = [
        "n_changed_features",
        "plausibility_ratio",
        "robust_l1",
        "defect_probability",
    ]

    rows = []

    for project, paired_df in project_pairs.items():
        for metric in metrics:
            proposed = pd.to_numeric(
                paired_df[
                    f"proposed_{metric}"
                ],
                errors="raise",
            ).to_numpy(
                dtype=float
            )

            dice = pd.to_numeric(
                paired_df[
                    f"dice_{metric}"
                ],
                errors="raise",
            ).to_numpy(
                dtype=float
            )

            if len(proposed) == 0:
                raise ValueError(
                    f"{project}: paired quality table is empty."
                )

            if len(proposed) != len(dice):
                raise ValueError(
                    f"{project}: paired quality lengths differ."
                )

            differences = (
                proposed
                - dice
            )

            if np.all(
                differences == 0
            ):
                statistic = 0.0
                p_value = 1.0

            else:
                result = wilcoxon(
                    proposed,
                    dice,
                    zero_method="wilcox",
                    correction=False,
                    alternative="two-sided",
                    method="auto",
                )

                statistic = float(
                    result.statistic
                )

                p_value = float(
                    result.pvalue
                )

            q25, q75 = np.quantile(
                differences,
                [
                    0.25,
                    0.75,
                ],
            )

            ci_low, ci_high = (
                _paired_bootstrap_median_ci(
                    differences,
                    seed=bootstrap_seed,
                    replicates=bootstrap_replicates,
                )
            )

            rows.append(
                {
                    "project":
                        project,

                    "metric":
                        metric,

                    "n_pairs":
                        int(
                            len(
                                differences
                            )
                        ),

                    "proposed_median":
                        float(
                            np.median(
                                proposed
                            )
                        ),

                    "dice_median":
                        float(
                            np.median(
                                dice
                            )
                        ),

                    "median_paired_difference":
                        float(
                            np.median(
                                differences
                            )
                        ),

                    "paired_difference_q25":
                        float(
                            q25
                        ),

                    "paired_difference_q75":
                        float(
                            q75
                        ),

                    "median_difference_ci95_low":
                        ci_low,

                    "median_difference_ci95_high":
                        ci_high,

                    "n_negative":
                        int(
                            np.sum(
                                differences < 0
                            )
                        ),

                    "n_zero":
                        int(
                            np.sum(
                                differences == 0
                            )
                        ),

                    "n_positive":
                        int(
                            np.sum(
                                differences > 0
                            )
                        ),

                    "wilcoxon_statistic":
                        statistic,

                    "wilcoxon_raw_p":
                        p_value,

                    "rank_biserial":
                        _matched_pairs_rank_biserial(
                            differences
                        ),
                }
            )

    results = pd.DataFrame(
        rows
    )

    results[
        "holm_p_quality_family"
    ] = _holm_adjust(
        results[
            "wilcoxon_raw_p"
        ].to_numpy(
            dtype=float
        )
    )

    results[
        "holm_significant_0_05"
    ] = (
        results[
            "holm_p_quality_family"
        ]
        < 0.05
    )

    return results



def build_runtime_pairs(
    proposed_commit_results,
    dice_commit_results,
    id_column="commit_id",
    runtime_column="runtime_seconds",
    expected_commit_ids=None,
):
    """Build exact paired per-query runtime observations."""

    proposed = proposed_commit_results.copy()
    dice = dice_commit_results.copy()

    required_columns = {
        id_column,
        runtime_column,
    }

    if not required_columns.issubset(
        proposed.columns
    ):
        raise ValueError(
            "Proposed results are missing required runtime columns."
        )

    if not required_columns.issubset(
        dice.columns
    ):
        raise ValueError(
            "DiCE results are missing required runtime columns."
        )

    proposed_ids = (
        proposed[
            id_column
        ]
        .astype(
            str
        )
        .tolist()
    )

    dice_ids = (
        dice[
            id_column
        ]
        .astype(
            str
        )
        .tolist()
    )

    if proposed_ids != dice_ids:
        raise ValueError(
            "Proposed and DiCE commit order must match exactly."
        )

    if expected_commit_ids is not None:
        expected_ids = [
            str(
                commit_id
            )
            for commit_id in expected_commit_ids
        ]

        if proposed_ids != expected_ids:
            raise ValueError(
                "Runtime commit order does not match the locked population."
            )

    proposed_runtime = pd.to_numeric(
        proposed[
            runtime_column
        ],
        errors="raise",
    ).to_numpy(
        dtype=float
    )

    dice_runtime = pd.to_numeric(
        dice[
            runtime_column
        ],
        errors="raise",
    ).to_numpy(
        dtype=float
    )

    if not np.isfinite(
        proposed_runtime
    ).all():
        raise ValueError(
            "Proposed runtimes must be finite."
        )

    if not np.isfinite(
        dice_runtime
    ).all():
        raise ValueError(
            "DiCE runtimes must be finite."
        )

    if not (
        proposed_runtime
        >= 0
    ).all():
        raise ValueError(
            "Proposed runtimes must be non-negative."
        )

    if not (
        dice_runtime
        >= 0
    ).all():
        raise ValueError(
            "DiCE runtimes must be non-negative."
        )

    if not (
        dice_runtime
        > 0
    ).all():
        raise ValueError(
            "DiCE runtimes must be positive for runtime ratios."
        )

    runtime_difference = (
        proposed_runtime
        - dice_runtime
    )

    runtime_ratio = (
        proposed_runtime
        / dice_runtime
    )

    return pd.DataFrame(
        {
            "commit_id":
                proposed_ids,

            "proposed_runtime_seconds":
                proposed_runtime,

            "dice_runtime_seconds":
                dice_runtime,

            "runtime_difference_seconds":
                runtime_difference,

            "runtime_ratio_proposed_over_dice":
                runtime_ratio,
        }
    )



def _summarize_runtime_project(
    project,
    paired_runtime,
    bootstrap_seed=42,
    bootstrap_replicates=100000,
):
    """Summarize frozen paired runtime outcomes for one project."""

    from scipy.stats import wilcoxon

    required_columns = {
        "proposed_runtime_seconds",
        "dice_runtime_seconds",
        "runtime_difference_seconds",
        "runtime_ratio_proposed_over_dice",
    }

    if not required_columns.issubset(
        paired_runtime.columns
    ):
        raise ValueError(
            "Paired runtime table is missing required columns."
        )

    proposed = pd.to_numeric(
        paired_runtime[
            "proposed_runtime_seconds"
        ],
        errors="raise",
    ).to_numpy(
        dtype=float
    )

    dice = pd.to_numeric(
        paired_runtime[
            "dice_runtime_seconds"
        ],
        errors="raise",
    ).to_numpy(
        dtype=float
    )

    differences = pd.to_numeric(
        paired_runtime[
            "runtime_difference_seconds"
        ],
        errors="raise",
    ).to_numpy(
        dtype=float
    )

    ratios = pd.to_numeric(
        paired_runtime[
            "runtime_ratio_proposed_over_dice"
        ],
        errors="raise",
    ).to_numpy(
        dtype=float
    )

    if len(
        proposed
    ) == 0:
        raise ValueError(
            f"{project}: paired runtime table is empty."
        )

    if not np.isfinite(
        proposed
    ).all():
        raise ValueError(
            f"{project}: Proposed runtimes must be finite."
        )

    if not np.isfinite(
        dice
    ).all():
        raise ValueError(
            f"{project}: DiCE runtimes must be finite."
        )

    if not np.isfinite(
        differences
    ).all():
        raise ValueError(
            f"{project}: runtime differences must be finite."
        )

    if not np.isfinite(
        ratios
    ).all():
        raise ValueError(
            f"{project}: runtime ratios must be finite."
        )

    if np.all(
        differences == 0
    ):
        statistic = 0.0
        raw_p = 1.0

    else:
        result = wilcoxon(
            proposed,
            dice,
            zero_method="wilcox",
            correction=False,
            alternative="two-sided",
            method="auto",
        )

        statistic = float(
            result.statistic
        )

        raw_p = float(
            result.pvalue
        )

    difference_q25, difference_q75 = np.quantile(
        differences,
        [
            0.25,
            0.75,
        ],
    )

    ratio_q25, ratio_q75 = np.quantile(
        ratios,
        [
            0.25,
            0.75,
        ],
    )

    ci_low, ci_high = _paired_bootstrap_median_ci(
        differences,
        seed=bootstrap_seed,
        replicates=bootstrap_replicates,
    )

    return {
        "project":
            project,

        "n_pairs":
            int(
                len(
                    paired_runtime
                )
            ),

        "proposed_median_runtime":
            float(
                np.median(
                    proposed
                )
            ),

        "proposed_q25_runtime":
            float(
                np.quantile(
                    proposed,
                    0.25,
                )
            ),

        "proposed_q75_runtime":
            float(
                np.quantile(
                    proposed,
                    0.75,
                )
            ),

        "proposed_mean_runtime":
            float(
                np.mean(
                    proposed
                )
            ),

        "dice_median_runtime":
            float(
                np.median(
                    dice
                )
            ),

        "dice_q25_runtime":
            float(
                np.quantile(
                    dice,
                    0.25,
                )
            ),

        "dice_q75_runtime":
            float(
                np.quantile(
                    dice,
                    0.75,
                )
            ),

        "dice_mean_runtime":
            float(
                np.mean(
                    dice
                )
            ),

        "median_paired_difference_seconds":
            float(
                np.median(
                    differences
                )
            ),

        "paired_difference_q25":
            float(
                difference_q25
            ),

        "paired_difference_q75":
            float(
                difference_q75
            ),

        "median_difference_ci95_low":
            ci_low,

        "median_difference_ci95_high":
            ci_high,

        "median_runtime_ratio_proposed_over_dice":
            float(
                np.median(
                    ratios
                )
            ),

        "runtime_ratio_q25":
            float(
                ratio_q25
            ),

        "runtime_ratio_q75":
            float(
                ratio_q75
            ),

        "n_proposed_faster":
            int(
                np.sum(
                    differences < 0
                )
            ),

        "n_equal_runtime":
            int(
                np.sum(
                    differences == 0
                )
            ),

        "n_proposed_slower":
            int(
                np.sum(
                    differences > 0
                )
            ),

        "wilcoxon_statistic":
            statistic,

        "wilcoxon_raw_p":
            raw_p,

        "rank_biserial":
            _matched_pairs_rank_biserial(
                differences
            ),

        "bootstrap_seed":
            int(
                bootstrap_seed
            ),

        "bootstrap_replicates":
            int(
                bootstrap_replicates
            ),
    }



def summarize_paired_runtime(
    project_pairs,
    bootstrap_seed=42,
    bootstrap_replicates=100000,
):
    """Summarize frozen paired runtime outcomes across projects."""

    if len(
        project_pairs
    ) == 0:
        raise ValueError(
            "project_pairs must contain at least one project."
        )

    rows = []

    for project, paired_runtime in project_pairs.items():
        rows.append(
            _summarize_runtime_project(
                project,
                paired_runtime,
                bootstrap_seed=bootstrap_seed,
                bootstrap_replicates=bootstrap_replicates,
            )
        )

    results = pd.DataFrame(
        rows
    )

    results[
        "holm_p_runtime_family"
    ] = _holm_adjust(
        results[
            "wilcoxon_raw_p"
        ].to_numpy(
            dtype=float
        )
    )

    results[
        "holm_significant_0_05"
    ] = (
        results[
            "holm_p_runtime_family"
        ]
        < 0.05
    )

    return results
