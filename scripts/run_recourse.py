"""Command-line interface for frozen JIT-SDP recourse."""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from cpit601_recourse.api import run_recourse


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run frozen JIT-SDP prediction and "
            "counterfactual recourse for one commit."
        )
    )

    parser.add_argument(
        "--input-csv",
        required=True,
        help="Path to a one-row CSV containing predictor features.",
    )

    parser.add_argument(
        "--project",
        required=True,
        choices=[
            "OpenStack",
            "Qt",
        ],
        help="Frozen project predictor to use.",
    )

    parser.add_argument(
        "--method",
        default="Proposed",
        choices=[
            "Proposed",
            "DiCE",
        ],
        help="Recourse method.",
    )

    parser.add_argument(
        "--configuration",
        default="Strict",
        choices=[
            "Strict",
            "Extended",
            "All",
        ],
        help="Recourse configuration.",
    )

    parser.add_argument(
        "--commit-id",
        default=None,
        help=(
            "Optional commit identifier. "
            "If omitted, a commit_id column in the CSV is used when present."
        ),
    )

    parser.add_argument(
        "--query-position",
        type=int,
        default=0,
        help=(
            "Zero-based query position used by the frozen "
            "DiCE seed rule."
        ),
    )

    parser.add_argument(
        "--dice-explainer",
        default=None,
        help=(
            "Path to a compatible stored DiCE explainer. "
            "Required when --method DiCE."
        ),
    )

    parser.add_argument(
        "--output-json",
        default=None,
        help=(
            "Optional output JSON path. "
            "If omitted, JSON is written to stdout."
        ),
    )

    args = parser.parse_args()

    input_path = Path(
        args.input_csv
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {input_path}"
        )

    query_table = pd.read_csv(
        input_path
    )

    if len(
        query_table
    ) != 1:
        raise ValueError(
            "Input CSV must contain exactly one row."
        )

    commit_id = args.commit_id

    if (
        commit_id is None
        and "commit_id" in query_table.columns
    ):
        commit_id = str(
            query_table.iloc[
                0
            ][
                "commit_id"
            ]
        )

    dice_explainer = None

    if args.method == "DiCE":
        if args.dice_explainer is None:
            raise ValueError(
                "--dice-explainer is required when --method DiCE."
            )

        dice_explainer_path = Path(
            args.dice_explainer
        )

        if not dice_explainer_path.exists():
            raise FileNotFoundError(
                "DiCE explainer not found: "
                + str(
                    dice_explainer_path
                )
            )

        dice_explainer = joblib.load(
            dice_explainer_path
        )

    result = run_recourse(
        query=query_table,
        project=args.project,
        method=args.method,
        configuration=args.configuration,
        commit_id=commit_id,
        query_position=args.query_position,
        dice_explainer=dice_explainer,
    )

    output_text = json.dumps(
        result,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    )

    if args.output_json is None:
        print(
            output_text
        )

    else:
        output_path = Path(
            args.output_json
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path.write_text(
            output_text + "\n",
            encoding="utf-8",
        )

        print(
            "Structured recourse output written to:",
            output_path,
        )


if __name__ == "__main__":
    main()
