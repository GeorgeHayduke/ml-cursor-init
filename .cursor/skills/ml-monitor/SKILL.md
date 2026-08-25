---
name: ml-monitor
description: Watch production scores for input drift, score drift, and labeled performance decay; set alerts that can trigger /ml-retrain. Use after /ml-integrate, for drift, PSI, production QA, or when the user says /ml-monitor.
disable-model-invocation: true
---

# /ml-monitor — Model Monitoring (Step 11)

Run after `/ml-integrate`. This skill does **not** change the champion,
the threshold, or the calibration. It watches live (or shadow) scores
against the **train-window feature baseline** and the **test-window score
and performance baseline** from steps 6–8, then alerts. A breach
recommends `/ml-retrain`; it does not start a retrain by itself.

If `configs/integration.yaml` is missing, stop and point back to
`/ml-integrate`.

Steps 11 and 12 are ongoing after deployment — there is no single "done
date." The definition-of-done here is: config + job + first baseline
report exist, and `PROJECT.md` says monitoring is in place.

## Step 1 — Confirm baselines (do not rebuild from production)

- **Feature baseline:** train-window distributions for the features in
  `configs/features.yaml` (and at least the top SHAP features from
  `/ml-explain`).
- **Score baseline:** test-window `pred_1` and `score_norm` from the
  pre-refit champion used in evaluation — unless Step 8.5 recalibrated,
  in which case use the mapping that actually shipped.
- **Performance baseline:** test-window ROC-AUC plus precision/recall/F1
  and expected cost at the **locked** operating point (not a newly swept
  threshold).
- **Label lag:** from `/ml-define`'s target (e.g. 12-month default).
  Labeled performance **cannot** be computed until outcomes exist. Until
  then, watch only input + score drift, and say so plainly.

## Step 2 — Ask cadence and alert bars

Ask, don't guess:

1. How often scores land (should match `/ml-integrate`'s schedule).
2. How often to run the monitor job (default: every scoring run for
   drift; labeled performance only when a cohort's outcome window has
   closed).
3. Alert thresholds. If the user has none, use these defaults and record
   that they are defaults:
   - Feature PSI > 0.20 → investigate; > 0.30 → retrain candidate
   - Score-distribution shift (PSI on `pred_1` or `score_norm`) > 0.20
     → investigate; > 0.30 → retrain candidate
   - Once labels exist: ROC-AUC drop ≥ 0.03 vs test baseline, or expected
     cost at the locked point worse than the Step 6 recommendation by a
     margin the user names → retrain candidate
4. Who gets the alert (the stakeholders already in `PROJECT.md`, unless
   they name someone else).

## Step 3 — What to compute

Every monitor run writes a dated folder under `reports/monitoring/`:

1. **Input drift** — PSI (or equivalent) per monitored feature vs train
   baseline; call out the top SHAP features even if their PSI is small.
2. **Score drift** — PSI / histogram comparison of `pred_1` and
   `score_norm` vs the shipped baseline; share of rows with `decision = 1`
   vs the test-window rate at the same threshold (volume/alert-rate
   drift).
3. **Labeled performance** — only for rows whose outcome is knowable.
   Same table as Step 6 (ROC-AUC, precision, recall, F1) **at the locked
   threshold**, plus expected cost using the FP/FN costs from
   `/ml-define`. Do **not** sweep a new operating table here and quietly
   change production.
4. **Verdict** — `ok` / `investigate` / `retrain_candidate`, with the
   rule that fired.

Honor the cloud decision: if production scores live in GCS/S3/Azure,
read them there. Don't assume a local CSV once cloud is in play.

## Step 4 — Persist config and job

Write `configs/monitoring.yaml` (baselines, PSI thresholds, label lag,
cadence, output path). Implement `src/<project_slug>/monitor.py` that
reads a scored extract (the integration contract) plus optional labels
and writes the report. Log each run under `experiments/` like training
runs (params = thresholds, data version, git commit, verdict).

## Step 5 — Wrap up

Run `python -m pytest tests/test_lifecycle_steps.py::TestStep11Monitor`. Do
not wrap until it passes (a missing `reports/monitoring/` run skips, not
fails, until the first live/shadow scores exist).

Fill `templates/report_template.md` Section 11. Check off Step 11 in
`PROJECT.md` as "in place (ongoing)." Tell the user which defaults were
used, when labeled metrics will first be possible (lag), and that
`/ml-retrain` is the next skill when a run returns `retrain_candidate`
**or** when the calendar cadence in that skill is due — whichever comes
first. Do not run `/ml-retrain` from here unless the user asked to.
