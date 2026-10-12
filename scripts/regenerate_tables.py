"""Regenerate manuscript tables from frozen raw result files."""

from pathlib import Path

import pandas as pd


def build_main_results_table(
    r_root,
):
    """Build the manuscript main-results table from commit-level files."""

    r_root = Path(
        r_root
    )

    inputs = [
        (
            "OpenStack",
            "Proposed-Strict",
            r_root
            / "R1_Proposed_Strict"
            / "openstack_proposed_strict_commit_level.csv",
        ),
        (
            "Qt",
            "Proposed-Strict",
            r_root
            / "R1_Proposed_Strict"
            / "qt_proposed_strict_commit_level.csv",
        ),
        (
            "OpenStack",
            "DiCE-Strict",
            r_root
            / "R2_DiCE_Strict"
            / "openstack_dice_strict_commit_level.csv",
        ),
        (
            "Qt",
            "DiCE-Strict",
            r_root
            / "R2_DiCE_Strict"
            / "qt_dice_strict_commit_level.csv",
        ),
        (
            "OpenStack",
            "DiCE-All",
            r_root
            / "R5_DiCE_All"
            / "openstack_dice_all_commit_level.csv",
        ),
        (
            "Qt",
            "DiCE-All",
            r_root
            / "R5_DiCE_All"
            / "qt_dice_all_commit_level.csv",
        ),
    ]

    rows = []

    for project, method, path in inputs:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing frozen raw result: {path}"
            )

        commits = pd.read_csv(
            path,
            float_precision="round_trip",
        )

        required_columns = {
            "project",
            "method",
            "success",
            "best_n_changes",
            "best_plausibility_ratio",
            "best_robust_l1",
            "best_defect_probability",
            "runtime_seconds",
        }

        missing_columns = (
            required_columns
            - set(
                commits.columns
            )
        )

        if missing_columns:
            raise ValueError(
                f"{path.name} is missing columns: "
                f"{sorted(missing_columns)}"
            )

        success = commits[
            "success"
        ].astype(
            bool
        )

        successful = commits[
            success
        ]

        n_commits = len(
            commits
        )

        successful_commits = int(
            success.sum()
        )

        rows.append(
            {
                "project":
                    project,

                "method":
                    method,

                "n_commits":
                    n_commits,

                "successful_commits":
                    successful_commits,

                "usable_commit_coverage_pct":
                    100.0
                    * successful_commits
                    / n_commits,

                "median_best_n_changes":
                    float(
                        successful[
                            "best_n_changes"
                        ].median()
                    ),

                "median_best_plausibility_ratio":
                    float(
                        successful[
                            "best_plausibility_ratio"
                        ].median()
                    ),

                "median_best_robust_l1":
                    float(
                        successful[
                            "best_robust_l1"
                        ].median()
                    ),

                "median_best_defect_probability":
                    float(
                        successful[
                            "best_defect_probability"
                        ].median()
                    ),

                "runtime_median_sec":
                    float(
                        commits[
                            "runtime_seconds"
                        ].median()
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )



def build_primary_paired_coverage_table(
    r_root,
):
    """Build Proposed-Strict vs DiCE-Strict paired coverage statistics."""

    from cpit601_recourse.statistics import (
        summarize_paired_binary_outcome,
    )

    r_root = Path(
        r_root
    )

    inputs = [
        (
            "OpenStack",
            r_root
            / "R1_Proposed_Strict"
            / "openstack_proposed_strict_commit_level.csv",
            r_root
            / "R2_DiCE_Strict"
            / "openstack_dice_strict_commit_level.csv",
        ),
        (
            "Qt",
            r_root
            / "R1_Proposed_Strict"
            / "qt_proposed_strict_commit_level.csv",
            r_root
            / "R2_DiCE_Strict"
            / "qt_dice_strict_commit_level.csv",
        ),
    ]

    rows = []

    for project, proposed_path, dice_path in inputs:
        proposed = pd.read_csv(
            proposed_path,
            float_precision="round_trip",
        )

        dice = pd.read_csv(
            dice_path,
            float_precision="round_trip",
        )

        summary = summarize_paired_binary_outcome(
            proposed,
            dice,
            bootstrap_seed=42,
            bootstrap_replicates=100000,
        )

        rows.append(
            {
                "project":
                    project,
                **summary,
            }
        )

    return pd.DataFrame(
        rows
    )





def build_primary_paired_quality_table(
    r_root,
):
    """Build Proposed-Strict vs DiCE-Strict paired quality statistics."""

    from cpit601_recourse.statistics import (
        build_both_success_quality_pairs,
        summarize_paired_quality,
    )

    r_root = Path(
        r_root
    )

    ranking = [
        "n_changed_features",
        "plausibility_ratio",
        "robust_l1",
        "defect_probability",
    ]

    metric_columns = {
        "best_n_changes":
            "n_changed_features",
        "best_plausibility_ratio":
            "plausibility_ratio",
        "best_robust_l1":
            "robust_l1",
        "best_defect_probability":
            "defect_probability",
    }

    def prepare_quality_commit_table(
        commit_path,
        cf_path,
    ):
        commits = pd.read_csv(
            commit_path,
            dtype={
                "commit_id": str,
            },
            float_precision="round_trip",
        )

        cfs = pd.read_csv(
            cf_path,
            dtype={
                "commit_id": str,
            },
        )

        usable = cfs[
            cfs[
                "usable"
            ].astype(
                bool
            )
        ].copy()

        best = (
            usable.sort_values(
                by=[
                    "commit_id",
                    *ranking,
                ],
                ascending=True,
                kind="mergesort",
            )
            .groupby(
                "commit_id",
                sort=False,
                as_index=False,
            )
            .head(
                1
            )
            .set_index(
                "commit_id"
            )
        )

        prepared = commits.copy()

        for target_column, source_column in metric_columns.items():
            prepared[
                target_column
            ] = prepared[
                "commit_id"
            ].map(
                best[
                    source_column
                ]
            )

        return prepared

    project_inputs = [
        (
            "OpenStack",
            r_root
            / "R1_Proposed_Strict"
            / "openstack_proposed_strict_commit_level.csv",
            r_root
            / "R1_Proposed_Strict"
            / "openstack_proposed_strict_cf_level.csv",
            r_root
            / "R2_DiCE_Strict"
            / "openstack_dice_strict_commit_level.csv",
            r_root
            / "R2_DiCE_Strict"
            / "openstack_dice_strict_cf_level.csv",
        ),
        (
            "Qt",
            r_root
            / "R1_Proposed_Strict"
            / "qt_proposed_strict_commit_level.csv",
            r_root
            / "R1_Proposed_Strict"
            / "qt_proposed_strict_cf_level.csv",
            r_root
            / "R2_DiCE_Strict"
            / "qt_dice_strict_commit_level.csv",
            r_root
            / "R2_DiCE_Strict"
            / "qt_dice_strict_cf_level.csv",
        ),
    ]

    project_pairs = {}

    for (
        project,
        proposed_commit_path,
        proposed_cf_path,
        dice_commit_path,
        dice_cf_path,
    ) in project_inputs:
        proposed = prepare_quality_commit_table(
            proposed_commit_path,
            proposed_cf_path,
        )

        dice = prepare_quality_commit_table(
            dice_commit_path,
            dice_cf_path,
        )

        project_pairs[
            project
        ] = build_both_success_quality_pairs(
            proposed,
            dice,
        )

    return summarize_paired_quality(
        project_pairs,
        bootstrap_seed=42,
        bootstrap_replicates=100000,
    )



def build_primary_paired_runtime_table(
    r_root,
):
    """Build Proposed-Strict vs DiCE-Strict paired runtime statistics."""

    from cpit601_recourse.statistics import (
        build_runtime_pairs,
        summarize_paired_runtime,
    )

    r_root = Path(
        r_root
    )

    project_inputs = [
        (
            "OpenStack",
            r_root
            / "R1_Proposed_Strict"
            / "openstack_proposed_strict_commit_level.csv",
            r_root
            / "R2_DiCE_Strict"
            / "openstack_dice_strict_commit_level.csv",
        ),
        (
            "Qt",
            r_root
            / "R1_Proposed_Strict"
            / "qt_proposed_strict_commit_level.csv",
            r_root
            / "R2_DiCE_Strict"
            / "qt_dice_strict_commit_level.csv",
        ),
    ]

    project_pairs = {}

    for project, proposed_path, dice_path in project_inputs:
        proposed = pd.read_csv(
            proposed_path,
            float_precision="round_trip",
        )

        dice = pd.read_csv(
            dice_path,
            float_precision="round_trip",
        )

        project_pairs[
            project
        ] = build_runtime_pairs(
            proposed,
            dice,
        )

    return summarize_paired_runtime(
        project_pairs,
        bootstrap_seed=42,
        bootstrap_replicates=100000,
    )



def _build_strict_to_extended_transition(
    strict_commit_path,
    extended_commit_path,
    extended_cf_path,
):
    """Build one Strict-to-Extended commit transition table."""

    strict = pd.read_csv(
        strict_commit_path,
        dtype={
            "commit_id": str,
        },
        float_precision="round_trip",
    )

    extended = pd.read_csv(
        extended_commit_path,
        dtype={
            "commit_id": str,
        },
        float_precision="round_trip",
    )

    cfs = pd.read_csv(
        extended_cf_path,
        dtype={
            "commit_id": str,
        },
        float_precision="round_trip",
    )

    assert strict[
        "commit_id"
    ].tolist() == extended[
        "commit_id"
    ].tolist()

    usable = cfs[
        cfs[
            "usable"
        ].astype(
            bool
        )
    ].copy()

    changed_text = usable[
        "changed_features"
    ].fillna(
        ""
    ).astype(
        str
    )

    usable[
        "_ent_changed"
    ] = changed_text.str.contains(
        r"(?:^|[^A-Za-z0-9_])ent(?:[^A-Za-z0-9_]|$)",
        regex=True,
    )

    ent_by_commit = usable.groupby(
        "commit_id"
    )[
        "_ent_changed"
    ].any()

    non_ent_by_commit = (
        usable.assign(
            _non_ent_cf=~usable[
                "_ent_changed"
            ]
        )
        .groupby(
            "commit_id"
        )[
            "_non_ent_cf"
        ]
        .any()
    )

    result = pd.DataFrame(
        {
            "commit_id":
                strict[
                    "commit_id"
                ],

            "strict_success":
                strict[
                    "success"
                ].astype(
                    bool
                ),

            "extended_success":
                extended[
                    "success"
                ].astype(
                    bool
                ),
        }
    )

    result[
        "transition"
    ] = "both_fail"

    result.loc[
        result[
            "strict_success"
        ]
        &
        result[
            "extended_success"
        ],
        "transition",
    ] = "both_success"

    result.loc[
        result[
            "strict_success"
        ]
        &
        ~result[
            "extended_success"
        ],
        "transition",
    ] = "strict_only"

    result.loc[
        ~result[
            "strict_success"
        ]
        &
        result[
            "extended_success"
        ],
        "transition",
    ] = "extended_only"

    result[
        "any_usable_ent_cf"
    ] = result[
        "commit_id"
    ].map(
        ent_by_commit
    ).eq(
        True
    )

    result[
        "any_usable_non_ent_cf"
    ] = result[
        "commit_id"
    ].map(
        non_ent_by_commit
    ).eq(
        True
    )

    extended_only = (
        result[
            "transition"
        ]
        == "extended_only"
    )

    result[
        "extended_only_with_ent"
    ] = (
        extended_only
        &
        result[
            "any_usable_ent_cf"
        ]
    )

    result[
        "extended_only_with_non_ent"
    ] = (
        extended_only
        &
        result[
            "any_usable_non_ent_cf"
        ]
    )

    result[
        "extended_only_ent_only_among_returned_usable"
    ] = (
        extended_only
        &
        result[
            "any_usable_ent_cf"
        ]
        &
        ~result[
            "any_usable_non_ent_cf"
        ]
    )

    return result



def build_strict_to_extended_tables(
    r_root,
):
    """Build secondary Strict-to-Extended transition and summary tables."""

    r_root = Path(
        r_root
    )

    cases = [
        (
            "OpenStack",
            "Proposed",
            "openstack_proposed_strict_to_extended_transitions.csv",
            r_root
            / "R1_Proposed_Strict"
            / "openstack_proposed_strict_commit_level.csv",
            r_root
            / "R7_Proposed_Extended"
            / "openstack_proposed_extended_commit_level.csv",
            r_root
            / "R7_Proposed_Extended"
            / "openstack_proposed_extended_cf_level.csv",
        ),
        (
            "Qt",
            "Proposed",
            "qt_proposed_strict_to_extended_transitions.csv",
            r_root
            / "R1_Proposed_Strict"
            / "qt_proposed_strict_commit_level.csv",
            r_root
            / "R7_Proposed_Extended"
            / "qt_proposed_extended_commit_level.csv",
            r_root
            / "R7_Proposed_Extended"
            / "qt_proposed_extended_cf_level.csv",
        ),
        (
            "OpenStack",
            "DiCE",
            "openstack_dice_strict_to_extended_transitions.csv",
            r_root
            / "R2_DiCE_Strict"
            / "openstack_dice_strict_commit_level.csv",
            r_root
            / "R6_DiCE_Extended"
            / "openstack_dice_extended_commit_level.csv",
            r_root
            / "R6_DiCE_Extended"
            / "openstack_dice_extended_cf_level.csv",
        ),
        (
            "Qt",
            "DiCE",
            "qt_dice_strict_to_extended_transitions.csv",
            r_root
            / "R2_DiCE_Strict"
            / "qt_dice_strict_commit_level.csv",
            r_root
            / "R6_DiCE_Extended"
            / "qt_dice_extended_commit_level.csv",
            r_root
            / "R6_DiCE_Extended"
            / "qt_dice_extended_cf_level.csv",
        ),
    ]

    transitions = {}
    summary_rows = []

    for (
        project,
        method,
        filename,
        strict_path,
        extended_path,
        extended_cf_path,
    ) in cases:
        transition = _build_strict_to_extended_transition(
            strict_path,
            extended_path,
            extended_cf_path,
        )

        transitions[
            filename
        ] = transition

        n_commits = len(
            transition
        )

        strict_success = int(
            transition[
                "strict_success"
            ].sum()
        )

        extended_success = int(
            transition[
                "extended_success"
            ].sum()
        )

        extended_only = (
            transition[
                "transition"
            ]
            == "extended_only"
        )

        summary_rows.append(
            {
                "project":
                    project,

                "method":
                    method,

                "n_commits":
                    n_commits,

                "strict_success":
                    strict_success,

                "strict_coverage_pct":
                    100.0
                    * strict_success
                    / n_commits,

                "extended_success":
                    extended_success,

                "extended_coverage_pct":
                    100.0
                    * extended_success
                    / n_commits,

                "delta_coverage_pp":
                    100.0
                    * (
                        extended_success
                        - strict_success
                    )
                    / n_commits,

                "both_success":
                    int(
                        (
                            transition[
                                "transition"
                            ]
                            == "both_success"
                        ).sum()
                    ),

                "strict_only":
                    int(
                        (
                            transition[
                                "transition"
                            ]
                            == "strict_only"
                        ).sum()
                    ),

                "extended_only":
                    int(
                        extended_only.sum()
                    ),

                "both_fail":
                    int(
                        (
                            transition[
                                "transition"
                            ]
                            == "both_fail"
                        ).sum()
                    ),

                "extended_success_with_any_usable_ent_cf":
                    int(
                        (
                            transition[
                                "extended_success"
                            ]
                            &
                            transition[
                                "any_usable_ent_cf"
                            ]
                        ).sum()
                    ),

                "extended_success_with_any_usable_non_ent_cf":
                    int(
                        (
                            transition[
                                "extended_success"
                            ]
                            &
                            transition[
                                "any_usable_non_ent_cf"
                            ]
                        ).sum()
                    ),

                "gained_commits_with_any_usable_ent_cf":
                    int(
                        transition[
                            "extended_only_with_ent"
                        ].sum()
                    ),

                "gained_commits_with_any_usable_non_ent_cf":
                    int(
                        transition[
                            "extended_only_with_non_ent"
                        ].sum()
                    ),

                "gained_commits_ent_only_among_returned_usable":
                    int(
                        transition[
                            "extended_only_ent_only_among_returned_usable"
                        ].sum()
                    ),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    return summary, transitions



def write_regenerated_tables(
    r_root,
    output_dir,
):
    """Regenerate and write validated manuscript tables."""

    r_root = Path(
        r_root
    )

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    main_results = build_main_results_table(
        r_root
    )

    paired_coverage = build_primary_paired_coverage_table(
        r_root
    )

    paired_quality = build_primary_paired_quality_table(
        r_root
    )

    paired_runtime = build_primary_paired_runtime_table(
        r_root
    )

    (
        extended_summary,
        transition_tables,
    ) = build_strict_to_extended_tables(
        r_root
    )

    outputs = {
        "main_results.csv":
            main_results,

        "primary_paired_coverage.csv":
            paired_coverage,

        "primary_paired_quality.csv":
            paired_quality,

        "primary_paired_runtime.csv":
            paired_runtime,

        "strict_to_extended_summary.csv":
            extended_summary,
    }

    outputs.update(
        transition_tables
    )

    written_paths = []

    for filename, table in outputs.items():
        path = (
            output_dir
            / filename
        )

        table.to_csv(
            path,
            index=False,
        )

        written_paths.append(
            path
        )

    return written_paths



def _parse_args():
    """Parse command-line arguments for table regeneration."""

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Regenerate manuscript tables from frozen raw result files."
        )
    )

    parser.add_argument(
        "--r-root",
        required=True,
        help="Path to the frozen R result directory.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for regenerated CSV tables.",
    )

    return parser.parse_args()


def main():
    """CLI entry point for deterministic table regeneration."""

    args = _parse_args()

    written_paths = write_regenerated_tables(
        args.r_root,
        args.output_dir,
    )

    print(
        f"Regenerated {len(written_paths)} tables."
    )

    for path in written_paths:
        print(
            path
        )


if __name__ == "__main__":
    main()
