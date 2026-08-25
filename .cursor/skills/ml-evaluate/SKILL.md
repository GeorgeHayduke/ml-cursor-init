---
name: ml-evaluate
description: Score the untouched test window, pick a champion, and build the FPR operating table. Use for model evaluation, ROC/PR, operating points, or when the user says /ml-evaluate.
disable-model-invocation: true
---

# /ml-evaluate — Model Evaluation (Step 6)

Run after `/ml-model`. **This is the first skill that touches the test
window.** Everything before this used train/eval only — treat the test
window as a one-shot resource: score it once per candidate model, don't
iterate against it, and don't come back to re-tune based on what it shows.

## Step 1 — Confirm inputs

Check `configs/sampling.yaml`, `configs/features.yaml`, and the trained
model artifacts from `/ml-model` all exist. Confirm (from experiment logs)
that the test window hasn't already been touched by an earlier step — if
it has, flag that the resulting numbers are no longer an honest holdout
estimate.

## Step 2 — Score test, once

Score every candidate from `/ml-model`'s roster against the test window.
Build the performance table:

| Model | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|

Explain F1 in plain language alongside it (harmonic mean of precision and
recall — penalizes a model that trades one off heavily for the other).

## Step 3 — Select the champion

Pick the champion from test performance — not eval performance, which was
only for tuning/selection triage in `/ml-model`. State the primary metric
driving the choice and why (ties back to the success criteria pinned down
in `/ml-define`).

## Step 4 — ROC and PR charts

Build both, annotated at a chosen score threshold (0.5 unless the project
has a reason otherwise): state plainly where that threshold lands on
FPR/TPR (ROC) and precision/recall (PR) for the test set.

## Step 5 — Operating table

For the champion only: swept by **FPR from 0% to 5% in 0.25% increments**
(21 rows). For each FPR step, back out the score threshold, TPR, precision,
recall, F1.

| FPR | Threshold | TPR | Precision | Recall | F1 |
|---|---|---|---|---|---|

## Step 6 — Recommend an operating point

Pull the business constraint pinned down in `/ml-define` (Step 1) — the
FP/FN cost asymmetry or stated tolerance — and use it to pick a
recommended row from the operating table. State the recommendation and the
one-paragraph reasoning tying it back to that constraint; don't just pick
the row with the best F1.

## Step 7 — Document tuning sufficiency

Carry forward the trial counts `/ml-model` logged per model and state
plainly whether that was enough to trust the result, now that test
performance is known — if tuning barely moved the needle or the champion
is suspiciously close to its untuned baseline, say so.

## Step 8 — Write up and wrap

Run `python -m pytest tests/test_lifecycle_steps.py::TestStep06Evaluate`. Do
not wrap until it passes.

Write the performance table, ROC/PR charts, operating table, and
recommended operating point into `templates/report_template.md` Section 6.
Check off Step 6 in `PROJECT.md`, name the champion model explicitly (this
is what `/ml-explain` and `/ml-calibrate` operate on next), and point to
`/ml-explain` as next.
