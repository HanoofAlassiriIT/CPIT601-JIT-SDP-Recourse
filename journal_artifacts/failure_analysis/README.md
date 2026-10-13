# Proposed-Strict Failure Analysis

These files summarize the frozen reserved-Test failure analysis used in the journal manuscript.

The analysis does not modify the predictor, intervention masks, plausibility thresholds, ranking rule, anchor count, max_changes, or random seeds.

Summary:
- OpenStack: 6 failures; all had no CLEAN prediction flip under the frozen Strict search.
- Qt: 9 failures; 7 had no CLEAN prediction flip and 2 produced CLEAN flips that did not satisfy the empirical-plausibility criterion.

The source evaluation remains the frozen CPIT601-v1.0 scientific configuration.
