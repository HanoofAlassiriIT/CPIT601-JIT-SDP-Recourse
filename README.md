# CPIT 601 Reusable Recourse Package

## Project

**Feasibility-Aware Counterfactual Recourse for
Just-in-Time Software Defect Prediction (JIT-SDP)**

## Current status

This repository is the reproducibility package for the **CPIT601 Release**.

The scientific method and experimental configuration are frozen, and the reusable implementation has passed the permanent regression and reconstruction tests.

## Quick Start

1. Install: `python -m pip install -e .`
2. Prepare runtime assets: `python scripts/prepare_assets.py --output-dir runtime_assets`
3. Set `CPIT601_RUNTIME_ASSET_DIR` to the generated `runtime_assets` directory.
4. Run `python -m pytest -q` to verify the installation.

## Frozen scientific configuration

The package contains byte-identical copies of:

- `configs/final_config.json`
- `configs/validation_config.json`

Their provenance and hashes are recorded in:

- `configs/config_provenance.json`

The frozen configuration must not be changed as a consequence of
held-out/Test outcomes.

## Scientific boundary

The final package will preserve the frozen research behavior:

- fixed JIT defect predictor;
- developer-controllable intervention masks;
- real clean Development commits as the empirical reference;
- empirical plausibility checking;
- semantic-validity checking;
- independent verification of the class flip;
- frozen candidate ranking;
- structured top-k recourse output.

Empirical plausibility is **not causal feasibility**.

The package does not claim that changing a metric guarantees a
specific real-world code edit or defect-prevention outcome.

## Current implementation status

Completed so far:

- package directory structure;
- frozen config copies;
- config provenance;
- environment snapshot;
- pinned requirements;
- `pyproject.toml`;
- package namespace;
- artifact-integrity utilities;
- frozen config loader.

Not yet completed:

- input/schema validation;
- frozen predictor loading;
- frozen plausibility assets;
- unified evaluator;
- Proposed recourse generator;
- DiCE baseline wrapper;
- stable inference API/CLI;
- unit tests;
- notebook/package parity tests;
- clean-environment installation;
- integration/smoke tests;
- manuscript-table regeneration.

## Reproducibility rule

Refactoring is allowed only when it does not alter scientific behavior.

Before this package is considered complete, it must demonstrate
package/notebook parity and clean-environment reproducibility.

## Installation status

A clean-environment installation has **not yet been tested**.

## Usage status

The stable public inference API and CLI are available.

Python API: from cpit601_recourse import run_recourse

The run_recourse(...) function accepts one validated commit row with explicit project, method, and configuration selection.

DiCE inference requires an explicitly supplied compatible dice_explainer. No hidden checkpoint path is used.

CLI: scripts/run_recourse.py

For DiCE, provide --dice-explainer <path>. The --query-position argument is the zero-based position used by the frozen per-query seed rule.

Clean-environment installation and smoke verification remain pending and are handled separately in the reproducibility closure stage.

## Data provenance

Runtime assets are reconstructed from the official PyExplainer repository pinned to commit:

`5df12399933d1faf2b28ba8f985cc7b3725d31c2`

The source CSV files are verified by SHA256 before reconstruction.

## Supervisor Demo

Open `notebooks/CPIT601_Demo.ipynb` in Google Colab and run all cells from top to bottom.

The demo automatically installs the package, reconstructs the verified runtime assets, and runs a real OpenStack counterfactual-recourse example.
