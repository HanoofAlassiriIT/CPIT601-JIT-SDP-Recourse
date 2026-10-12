"""Stable public inference API for frozen JIT-SDP recourse."""

import math

import numpy as np
import pandas as pd

from .data import get_feature_order
from .data import validate_input_frame
from .dice_baseline import generate_dice_counterfactuals
from .evaluation import evaluate_counterfactual_set
from .predictor import predict_defect_probability
from .predictor import predict_labels
from .proposed import generate_clean_anchor_cfs


def run_recourse(
    query,
    project: str,
    method: str = "Proposed",
    configuration: str = "Strict",
    commit_id=None,
    query_position: int = 0,
    dice_explainer=None,
) -> dict[str, object]:
    """
    Run frozen prediction and counterfactual recourse for one commit.

    Parameters
    ----------
    query
        One-row pandas DataFrame containing the frozen predictor schema.
    project
        Supported project name.
    method
        "Proposed" or "DiCE".
    configuration
        "Strict", "Extended", or "All".
        Proposed supports Strict and Extended.
        DiCE supports Strict, Extended, and All.
    commit_id
        Optional identifier included in the structured output.
    query_position
        Zero-based position used only for the frozen DiCE seed rule.
    dice_explainer
        Required explicitly when method="DiCE".
        No hidden checkpoint path is used.
    """

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

    feature_order = get_feature_order(
        project
    )

    missing_features = [
        feature
        for feature in feature_order
        if feature not in query.columns
    ]

    if missing_features:
        raise ValueError(
            "query is missing predictor features: "
            + str(
                missing_features
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

    method_normalized = str(
        method
    ).strip().lower()

    configuration_normalized = str(
        configuration
    ).strip().lower()

    configuration_names = {
        "strict": "Strict",
        "extended": "Extended",
        "all": "All",
    }

    if (
        configuration_normalized
        not in configuration_names
    ):
        raise ValueError(
            "configuration must be Strict, Extended, or All."
        )

    configuration_name = configuration_names[
        configuration_normalized
    ]

    original_prediction = int(
        predict_labels(
            project,
            query_features,
        )[0]
    )

    original_defect_probability = float(
        predict_defect_probability(
            project,
            query_features,
        )[0]
    )

    if method_normalized == "proposed":
        if configuration_name == "All":
            raise ValueError(
                "Proposed supports Strict or Extended only."
            )

        generation = generate_clean_anchor_cfs(
            query=query_features,
            project=project,
            configuration=configuration_name,
        )

        generated = generation[
            "counterfactuals"
        ]

        diagnostics = generation[
            "diagnostics"
        ]

        if len(
            generated
        ) == 0:
            candidate_features = pd.DataFrame(
                columns=feature_order
            )

        else:
            candidate_features = pd.DataFrame(
                generated[
                    "candidate"
                ].tolist()
            )

            candidate_features = (
                candidate_features[
                    feature_order
                ]
                .astype(
                    float
                )
                .reset_index(
                    drop=True
                )
            )

        method_name = (
            "Proposed-"
            + configuration_name
        )

        runtime_seconds = float(
            diagnostics[
                "runtime_seconds"
            ]
        )

        candidate_count = int(
            diagnostics[
                "n_candidates_generated"
            ]
        )

        seed = np.nan

    elif method_normalized == "dice":
        if dice_explainer is None:
            raise ValueError(
                "dice_explainer is required explicitly "
                "when method='DiCE'."
            )

        generation = generate_dice_counterfactuals(
            query=query_features,
            project=project,
            configuration=configuration_name,
            query_position=query_position,
            dice_explainer=dice_explainer,
        )

        candidate_features = generation[
            "counterfactuals"
        ]

        diagnostics = generation[
            "diagnostics"
        ]

        method_name = (
            "DiCE-"
            + configuration_name
        )

        runtime_seconds = float(
            diagnostics[
                "runtime_seconds"
            ]
        )

        candidate_count = int(
            diagnostics[
                "raw_generated_cfs"
            ]
        )

        seed = int(
            diagnostics[
                "seed"
            ]
        )

    else:
        raise ValueError(
            "method must be Proposed or DiCE."
        )

    cf_evaluation, commit_evaluation = (
        evaluate_counterfactual_set(
            query=query_features,
            candidate_features=candidate_features,
            project=project,
            method_name=method_name,
            configuration=configuration_name,
            commit_id=commit_id,
            runtime_seconds=runtime_seconds,
            candidate_count=candidate_count,
            seed=seed,
        )
    )

    commit_row = commit_evaluation.iloc[
        0
    ]

    counterfactuals = []

    for row_index in range(
        len(
            cf_evaluation
        )
    ):
        evaluation_row = cf_evaluation.iloc[
            row_index
        ]

        candidate_row = candidate_features.iloc[
            row_index
        ]

        counterfactual_record = {
            "rank": int(
                row_index + 1
            ),
            "candidate": {
                feature: float(
                    candidate_row[
                        feature
                    ]
                )
                for feature in feature_order
            },
            "changed_features": str(
                evaluation_row[
                    "changed_features"
                ]
            ),
            "n_changed_features": int(
                evaluation_row[
                    "n_changed_features"
                ]
            ),
            "actionable": bool(
                evaluation_row[
                    "actionable"
                ]
            ),
            "empirically_plausible": bool(
                evaluation_row[
                    "empirically_plausible"
                ]
            ),
            "plausibility_ratio": float(
                evaluation_row[
                    "plausibility_ratio"
                ]
            ),
            "robust_l1": float(
                evaluation_row[
                    "robust_l1"
                ]
            ),
            "defect_probability": float(
                evaluation_row[
                    "defect_probability"
                ]
            ),
            "semantic_valid": bool(
                evaluation_row[
                    "semantic_valid"
                ]
            ),
            "usable_core": bool(
                evaluation_row[
                    "usable_core"
                ]
            ),
            "usable": bool(
                evaluation_row[
                    "usable"
                ]
            ),
        }

        if method_normalized == "proposed":
            generated_row = generated.iloc[
                row_index
            ]

            counterfactual_record[
                "anchor_rank"
            ] = int(
                generated_row[
                    "anchor_rank"
                ]
            )

            counterfactual_record[
                "anchor_commit_id"
            ] = str(
                generated_row[
                    "anchor_commit_id"
                ]
            )

        counterfactuals.append(
            counterfactual_record
        )

    structured_diagnostics = {}

    for key, value in diagnostics.items():
        if isinstance(
            value,
            (
                np.integer,
            ),
        ):
            structured_diagnostics[
                key
            ] = int(
                value
            )

        elif isinstance(
            value,
            (
                np.floating,
            ),
        ):
            structured_diagnostics[
                key
            ] = float(
                value
            )

        elif isinstance(
            value,
            (
                np.bool_,
            ),
        ):
            structured_diagnostics[
                key
            ] = bool(
                value
            )

        else:
            structured_diagnostics[
                key
            ] = value

    output = {
        "project": project,
        "commit_id": (
            None
            if commit_id is None
            else str(
                commit_id
            )
        ),
        "method": method_name,
        "configuration": configuration_name,
        "original_prediction": original_prediction,
        "original_defect_probability":
            original_defect_probability,
        "recourse_success": bool(
            commit_row[
                "success"
            ]
        ),
        "n_returned_counterfactuals": int(
            len(
                cf_evaluation
            )
        ),
        "runtime_seconds": runtime_seconds,
        "counterfactuals": counterfactuals,
        "diagnostics": structured_diagnostics,
    }

    if method_normalized == "dice":
        output[
            "seed"
        ] = seed

        output[
            "generation_error"
        ] = bool(
            diagnostics[
                "generation_error"
            ]
        )

        output[
            "no_cf_found"
        ] = bool(
            diagnostics[
                "no_cf_found"
            ]
        )

        output[
            "technical_error"
        ] = bool(
            diagnostics[
                "technical_error"
            ]
        )

    return output
