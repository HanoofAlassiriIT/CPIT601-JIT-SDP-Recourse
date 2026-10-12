"""Frozen seeded DiCE counterfactual-generation wrapper."""

import time

import pandas as pd

from .config import load_final_config
from .data import get_feature_order
from .data import validate_input_frame
from .evaluation import get_allowed_features


def generate_dice_counterfactuals(
    query,
    project: str,
    configuration: str,
    query_position: int,
    dice_explainer,
) -> dict[str, object]:
    """
    Generate DiCE counterfactuals under the frozen study protocol.

    Scientific DiCE settings are loaded from FINAL_CONFIG.
    The per-query seed is:
        global_seed + zero_based_query_position
    """

    config = load_final_config()

    dice_config = config.get(
        "dice"
    )

    if not isinstance(
        dice_config,
        dict,
    ):
        raise RuntimeError(
            "FINAL_CONFIG dice section is invalid."
        )

    if dice_config.get(
        "method"
    ) != "random":
        raise RuntimeError(
            "Frozen DiCE method changed."
        )

    if dice_config.get(
        "per_query_seed_rule"
    ) != "global_seed + zero_based_query_position":
        raise RuntimeError(
            "Frozen DiCE seed rule changed."
        )

    desired_class = int(
        dice_config[
            "desired_class"
        ]
    )

    global_seed = int(
        dice_config[
            "primary_global_seed"
        ]
    )

    sample_size = int(
        dice_config[
            "sample_size"
        ]
    )

    total_cfs = int(
        dice_config[
            "total_CFs"
        ]
    )

    if desired_class != 0:
        raise RuntimeError(
            "Frozen DiCE desired class changed."
        )

    if global_seed != 42:
        raise RuntimeError(
            "Frozen DiCE global seed changed."
        )

    if sample_size != 1000:
        raise RuntimeError(
            "Frozen DiCE sample_size changed."
        )

    if total_cfs != 3:
        raise RuntimeError(
            "Frozen DiCE total_CFs changed."
        )

    if not isinstance(
        query_position,
        int,
    ):
        raise TypeError(
            "query_position must be an integer."
        )

    if query_position < 0:
        raise ValueError(
            "query_position must be zero-based and non-negative."
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

    feature_order = get_feature_order(
        project
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

    configuration_normalized = str(
        configuration
    ).strip().lower()

    if configuration_normalized == "strict":
        features_to_vary = get_allowed_features(
            project,
            "Strict",
        )

    elif configuration_normalized == "extended":
        features_to_vary = get_allowed_features(
            project,
            "Extended",
        )

    elif configuration_normalized == "all":
        features_to_vary = "all"

    else:
        raise ValueError(
            "configuration must be Strict, Extended, or All."
        )

    seed = (
        global_seed
        + query_position
    )

    start_time = time.perf_counter()

    generation_error = False
    error_message = ""

    try:
        result = dice_explainer.generate_counterfactuals(
            query_features,
            total_CFs=total_cfs,
            desired_class=desired_class,
            features_to_vary=features_to_vary,
            random_seed=seed,
            sample_size=sample_size,
            verbose=False,
        )

        raw_cfs = (
            result
            .cf_examples_list[
                0
            ]
            .final_cfs_df
        )

    except Exception as error:
        generation_error = True
        error_message = str(
            error
        )
        raw_cfs = None

    runtime_seconds = (
        time.perf_counter()
        - start_time
    )

    if (
        raw_cfs is None
        or len(raw_cfs) == 0
    ):
        counterfactuals = pd.DataFrame(
            columns=feature_order
        )

    else:
        counterfactuals = (
            raw_cfs[
                feature_order
            ]
            .astype(
                float
            )
            .reset_index(
                drop=True
            )
        )

    no_cf_found = (
        generation_error
        and "No counterfactuals found"
        in error_message
    )

    technical_error = (
        generation_error
        and not no_cf_found
    )

    diagnostics = {
        "seed": int(
            seed
        ),
        "generation_error": bool(
            generation_error
        ),
        "error_message": error_message,
        "no_cf_found": bool(
            no_cf_found
        ),
        "technical_error": bool(
            technical_error
        ),
        "raw_generated_cfs": int(
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
