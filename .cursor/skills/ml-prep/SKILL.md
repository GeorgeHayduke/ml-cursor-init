---
name: ml-prep
description: Engineer features, resolve leakage, and fit preprocessors on the train window only. Use for feature engineering, encoding, or when the user says /ml-prep.
disable-model-invocation: true
---

# /ml-prep — Feature Engineering (Step 4)

Run after `/ml-data`. Reads `data/processed/train.*` / `eval.*` / `test.*`
and `configs/sampling.yaml`; produces the feature set every model in
`/ml-model` trains on. Never touch the test window's *content* here beyond
applying already-fit transforms to it — no fitting anything on eval or test.

## Step 1 — Resolve leakage flags

`/ml-data` flagged any column only knowable after the outcome occurred, but
didn't decide what to do about it. Go through that list now and make an
explicit call per column: excluded (the normal outcome), or kept with a
documented reason it's actually safe (rare — be skeptical of these). Record
the decision in `PROJECT.md`, not just in code.

## Step 2 — Build the feature set

Propose derived features from the data dictionary — ratios, aggregates,
time-based features (natural here given the OOT sampling — recency,
rolling windows, trend features computed only from information available
as of each record's timestamp). Confirm the list with the user rather than
silently generating dozens of features; more isn't automatically better and
each one is something `/ml-explain` has to account for later.

Write the feature list — name, derivation, rationale — into
`configs/features.yaml` and into `templates/report_template.md` Section 4.
This file, along with `configs/sampling.yaml`, is what `/ml-model` reads to
know what it's training on.

## Step 3 — Preprocessing: fit on train only

Any imputation, scaling, or encoding gets **fit on the train window only**,
then applied unchanged to eval and test. This is the same discipline as the
sampling rule in `ml-lifecycle.mdc` — fitting anything on eval/test data
leaks information into what's supposed to be an honest holdout. Persist the
fitted transformers (e.g. under `models/preprocessing/`) so `/ml-model` and
later re-scoring both use the exact same transform, not a re-fit one.

Handle encoding per model family rather than forcing one shared
representation:

- **CatBoost** takes categoricals natively — pass them through unencoded,
  just tell CatBoost which columns are categorical.
- **Random Forest and XGBoost** need numeric input — encode categoricals
  (one-hot for low-cardinality, target/frequency encoding — computed on
  train only, standard k-fold-safe method if target encoding is used — for
  high-cardinality).

Keep both representations tied to the same underlying feature *definitions*
in `configs/features.yaml` so the three models are being compared on the
same information, just encoded differently — not on different feature
sets.

## Step 4 — Save & verify

Write the engineered features to `data/processed/features_train.*`,
`features_eval.*`, `features_test.*` (or add columns onto the existing
split files — whichever fits the project's data format). Before finishing,
re-check: no feature was derived using information not available at
prediction time, and no transform was fit using eval or test rows.

## Step 5 — Wrap up

Check off Step 4 in `PROJECT.md`. Tell the user the feature count, anything
excluded for leakage and why, and that `/ml-model` (Multi-Model Training,
step 5) is next.
