---
name: ml-model
description: Train the RF + XGBoost + CatBoost bake-off on train, tune on eval, log runs. Use for multi-model training, hyperparameter search, or when the user says /ml-model.
disable-model-invocation: true
---

# /ml-model — Multi-Model Training (Step 5)

Run after `/ml-prep`. Confirm `configs/sampling.yaml` and
`configs/features.yaml` both exist before starting — if either is missing,
stop and point back to `/ml-data` or `/ml-prep` rather than guessing at a
split or feature set.

This step trains on **train**, tunes and selects on **eval**. It does not
touch the **test** window at all — that stays untouched until `/ml-evaluate`
(step 6). A model comparison happens here for selection purposes, but it is
not the honest, reportable performance number; that only exists once it's
been measured against test.

## Step 1 — Confirm the roster

Default bake-off, per house convention: **Random Forest + XGBoost +
CatBoost** — minimum three models, always. Ask whether AutoGluon should be
added this time as a quick-and-dirty baseline or an advanced
auto-ensembling pass — it's situational, not automatic, so don't add it
without asking.

## Step 2 — Train baselines

Fit each model with reasonable default hyperparameters on train, score on
eval, and log the result before doing any tuning — this baseline pass is
what later tells you whether tuning actually helped.

## Step 3 — Hyperparameter tuning

Ask whether tuning runs locally or in the cloud — check `PROJECT.md` for
the cloud decision from `/ml-init`. If a provider was chosen and the search
is nontrivial (many trials, large data), default to that provider's tuning
service (e.g. Vertex AI hyperparameter tuning, SageMaker Automatic Model
Tuning, Azure ML HyperDrive) rather than running it locally — but confirm
before kicking off anything that spends meaningful cloud compute.

For each model, define a search space and a tuning method (random search,
Bayesian/Optuna, or the cloud service's native tuner), tune against the eval
window only, and **record the number of trials run for each model** — this
is required later in `templates/report_template.md` Section 5.1, which
explicitly asks whether enough trials were run to trust the result. Don't
tune one model for 200 trials and another for 10 without saying so.

## Step 4 — Log every run

Use whichever experiment-tracking approach `/ml-init` recorded (structured
local logs under `experiments/`, or MLflow). Each run record should capture
at minimum: model type, hyperparameters, train/eval metrics, which
`configs/sampling.yaml` and `configs/features.yaml` were in effect, and the
git commit. This is what makes "how did you choose hyperparameters"
answerable later instead of reconstructed from memory.

## Step 5 — Compare and persist

Produce a comparison table (model, key hyperparameters, eval-window
metrics) and write it into `templates/report_template.md` Section 5.
Persist each tuned model's artifact under `models/` (e.g.
`models/random_forest.pkl`, `models/xgboost.json`, `models/catboost.cbm`).
Don't yet declare a single "champion" based on eval performance alone —
that call belongs to `/ml-evaluate`, which tests against the untouched test
window and the business constraint pinned down in `/ml-define`.

## Step 6 — Wrap up

Check off Step 5 in `PROJECT.md`. Tell the user the roster trained, trial
counts per model, and the eval-window comparison, and that `/ml-evaluate`
(step 6) is next — where the real, reportable performance numbers get
produced against test.
