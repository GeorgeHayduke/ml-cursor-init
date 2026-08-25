---
name: ml-retrain
description: Slide the OOT windows and refit the locked champion (or a confirmed bake-off), then promote only if the new test window beats the old model. Use when monitoring flags retrain_candidate, on a calendar cadence, or when the user says /ml-retrain.
disable-model-invocation: true
---

# /ml-retrain — Periodic Retraining (Step 12)

Run when `/ml-monitor` returns `retrain_candidate`, or on the calendar
cadence recorded here, or when the user explicitly asks. This skill
**loops back to steps 2–3** for new data and a new Out-of-Time split. It
does not skip evaluation: a retrained model is not production until it
beats the currently shipped model on a **new** test window.

This is ongoing — checking off Step 12 in `PROJECT.md` means "retrain
process defined (and this run documented)," not "we will never retrain
again."

If the user wants a retrain but `/ml-integrate` was never run, flag that
there is nothing to replace, then continue with a research retrain if
they still want one.

## Step 1 — Why this run, and what stays locked

Ask:

1. **Trigger:** monitoring verdict, calendar cadence, or ad hoc. Record
   the trigger and, if monitoring, the report path.
2. **Cadence (if not already in `configs/retrain.yaml`):** e.g. quarterly,
   or "when PSI > 0.30," or both. Write it so the next run doesn't
   re-ask.
3. **Hyperparameters:** default is **hold** the champion's locked HPs
   from the last `/ml-model` / 8.5 artifact. Re-tune only with an
   explicit yes.
4. **Roster:** default is **same champion family only** (the shipped
   architecture). Re-run the RF + XGBoost + CatBoost bake-off only with
   an explicit yes.
5. **Features:** default is **same definitions** in `configs/features.yaml`.
   Drop or replace a feature only if monitoring showed it died or leaked;
   that is a `/ml-prep` decision, not a silent change inside training.

Do not proceed until 1 is answered. Use defaults for 3–5 if the user
defers, and write those defaults into `configs/retrain.yaml`.

## Step 2 — New data and a new OOT split (back to steps 2–3)

Land new raw extracts under `data/raw/` (versioned, not overwritten).
Restructure into `data/interim/` as `/ml-data` would.

**Slide the windows** — do not randomly reshuffle:

- New **test** window = the most recent period, never overlapping the new
  eval, large enough to evaluate (flag rare-event volume).
- New **eval** = the period immediately before new test.
- New **train** = everything before new eval that you are willing to fit
  on (usually old train+eval+test plus new arrivals up to the new eval
  start). Record any rows dropped (policy, leakage, decayed history).

Reuse the same **seed** as `configs/sampling.yaml` unless the user
changes it on purpose. Write a new sampling file (e.g.
`configs/sampling.yaml` updated in place **and** a dated copy under
`experiments/`) so the previous split remains recoverable.

Explore **only the new train window**. New eval/test: structural sanity
check only — same rule as `/ml-data`.

Refit preprocessors on the **new train only**, persist a new
`models/preprocessing/` (do not overwrite the production copy until
promotion). Apply to new eval/test.

## Step 3 — Fit candidates

- Always score the **currently shipped** `champion_final` on the **new
  test window** (forward performance of the old model). That number is
  the bar.
- Fit the retrained candidate(s) on new train; if a bake-off was
  approved, tune on new eval only, never on new test. If HPs are held,
  skip tuning and log that.
- Log every run under `experiments/` with trigger, sampling file, feature
  file, git commit.

## Step 4 — Evaluate and promotion gate

On the **new test window only**, produce the Step 6 artifacts for the
candidate **and** for the old shipped model: performance table, ROC/PR at
the old locked threshold **and** a fresh operating table (FPR 0–5% by
0.25%) for the candidate.

**Promote** the candidate only if:

- It beats the old model on the project's primary success metric from
  `/ml-define`, **and**
- There exists an operating-table row that still satisfies the business
  constraint (FP/FN costs or volume cap).

If it fails, keep production as-is, write the failed bake-off into
`reports/`, and stop. Do not "ship it anyway" because the calendar said
retrain.

If it wins: pick a new operating point from **this** operating table
against the same business constraint (the old threshold may no longer
land at the same FPR). Then `/ml-explain` on the new champion / new
test (TreeSHAP, same TP/FP/FN sorts), then calibration (curve + 1–1000
bands) on the new test, then Step 8.5 full-data refit on the new
train+eval+test. Persist as a **new** `models/champion_final.*` (keep
the previous artifact for rollback). Repeat the 8.5 drift check.

## Step 5 — Cut over and wrap

Update `configs/integration.yaml` `model_version` to the new artifact.
Do not delete the previous version until a rollback window the user
names has passed. Point `/ml-monitor` baselines at the new train
features and the new test score/performance numbers.

Write `configs/retrain.yaml` (cadence, hold-HP default, last run id,
rollback path). Fill `templates/report_template.md` Section 12 and
regenerate the shareable report (`/ml-document` or the same assembly
rules). Check off Step 12 in `PROJECT.md` as "process in place; last run
<date>." Tell the user: promoted or not, old vs new test metrics,
new threshold if any, and that the next event is a monitor run or the
next cadence — back to step 2/3 when it fires.
