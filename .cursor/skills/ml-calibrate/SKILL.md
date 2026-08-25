---
name: ml-calibrate
description: Calibration curve, 1-1000 score bands, then full-data final fit with a drift check. Use for calibration, score normalization, deployment refit, or when the user says /ml-calibrate.
disable-model-invocation: true
---

# /ml-calibrate — Model Calibration (Step 8 + Step 8.5)

Run after `/ml-explain`. This skill covers both Step 8 (Calibration) and
Step 8.5 (Final Fit for Deployment) — they're handled together here since
8.5's calibration-drift check needs the same champion/test context Step 8
just built, rather than reloading it in a separate skill.

## Step 8.1 — Calibration curve

Build a reliability diagram for the champion's raw test-window scores
(predicted probability vs. observed frequency, binned). State plainly
whether the model is well-calibrated, over-confident, or under-confident,
and where (e.g. "over-confident above 0.7").

## Step 8.2 — Score normalization

Fit a rank-based spline mapping the raw test-window score to a normalized
1-1000 scale, banded by population percentile. Default bands (adjust per
project, note here if they were changed):

| Normalized band | Population share |
|---|---|
| 1000-900 | ~0.25% (rarest) |
| 900-800 | ~0.25% |
| 800-500 | ~4.5% |
| 500-0 | ~95% |

Write both deliverables into `templates/report_template.md` Section 8.

## Step 8.5 — Final fit for deployment

Refit the champion — same architecture, same hyperparameters, no
re-tuning — on the **full dataset**: train + eval + test combined. This is
the model that actually deploys.

**Calibration-drift check** (don't assume it carries over, check it):

1. Re-score the test-window records with the refit model.
2. Compare that score distribution to the original (pre-refit) test-window
   scores from Step 8.1/8.2.
3. Close → carry the existing calibration mapping and Step 6 operating
   table forward unchanged.
4. Shifted → recalibrate using the refit model's test-window scores
   against actual outcomes, flagging that this is slightly optimistic
   since those records also trained the refit model — or, for a stricter
   check, note that a small most-recent slice should have been held back
   before the full refit specifically for this validation.

Persist the refit model as the deployment artifact (e.g.
`models/champion_final.*`, distinct from the pre-refit champion used for
evaluation). Write the refit confirmation and drift-check outcome into
`templates/report_template.md` Section 8.5.

## Step 9 — Wrap up

Check off Step 8 and Step 8.5 in `PROJECT.md`. Point to `/ml-document` to
assemble the finished report, and `/ml-integrate` (predict / explain /
label) after that.
