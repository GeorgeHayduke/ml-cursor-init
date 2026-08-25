---
name: ml-define
description: Write the ML problem charter into PROJECT.md — target, success criteria, business constraint, timeline. Use after /ml-init, for problem formulation, or when the user says /ml-define.
disable-model-invocation: true
---

# /ml-define — Problem Formulation (Step 1)

Run after `/ml-init`. This expands the one-line problem statement from init
into the actual charter — all of it lives inside `PROJECT.md`, there is no
separate charter file. Everything downstream (`/ml-data` onward) treats
`PROJECT.md` as the source of truth for what this project is trying to do.

## Step 1 — Ask

Ask for, and don't accept vague answers on:

1. **Problem statement:** confirm/refine the one-liner from `/ml-init` into
   a real paragraph — the decision this model drives, and why now.
2. **Target variable:** a precise, unambiguous definition (e.g. `default =
   1` if a loan charges off within 12 months of origination — not just
   "predict default"). If this can't be stated precisely yet, that's a
   signal the project isn't ready for `/ml-data`.
3. **Success criteria:** the metric(s) that decide whether the model is
   good enough, and the actual threshold for "good enough" — not just
   "high accuracy."
4. **Business constraint (not the operating point):** pin down the
   real-world cost asymmetry — what a false positive costs vs. what a
   false negative costs, or a stated tolerance ("we can't have more than
   roughly N false positives a month given review capacity"). This is
   *not* the same as choosing an FPR/threshold operating point — that
   happens in Step 6 once real model scores exist and can be swept against
   this constraint. Record the constraint here; the number gets chosen
   later.
5. **Stakeholders:** who needs to sign off, who consumes the output.
6. **Constraints:** latency, interpretability, regulatory, fairness, data
   availability — whatever actually shapes the approach.
7. **Baseline / current process:** optional. Many projects are exploratory
   and there's no real current process to beat — "none, this is
   exploratory" is a perfectly valid answer. If there *is* something being
   replaced (a rule, a manual review step, an older model), record what it
   is so evaluation has something concrete to compare against.
8. **Kickoff date:** needed for Step 2's timeline.

Don't proceed to Step 2 until 1-4 are answered — 6-7 can be thin for a
smaller or more exploratory project, but 1-4 are load-bearing for
everything downstream.

## Step 2 — Timeline (per step, not per week)

Build a per-step table, not a week-bucket table — every lifecycle step gets
its own row with a target start/end date computed from the kickoff date.
Default durations below sum to 5 weeks (25 business days) — this is the
default cadence from `ml-lifecycle.mdc`, adjust per project as needed, and
if one step's duration changes, recompute every date after it.

| Step | Default duration | Target start | Target end |
|---|---|---|---|
| 1. Problem Formulation | 2 business days | | |
| 2. Data Gathering & Structuring | 3 business days | | |
| 3. Data Exploration | 2 business days | | |
| 4. Feature Engineering | 3 business days | | |
| 5. Multi-Model Training | 4 business days | | |
| 6. Model Evaluation | 2 business days | | |
| 7. Model Interpretability | 2 business days | | |
| 8. Model Calibration | 1 business day | | |
| 8.5. Final Fit for Deployment | 1 business day | | |
| 9. Model Documentation | 2 business days | | |
| 10. Model Integration | 3 business days | | |

Steps 11 (Monitoring) and 12 (Periodic Retraining) start after step 10 and
run on an ongoing cadence, not a fixed date range — list them without target
dates, note "ongoing, post-deployment" instead.

Compute target start/end dates from the kickoff date in business days
(skip weekends).

## Step 3 — Update `PROJECT.md`

Replace the rough week-bucket table `/ml-init` seeded with this per-step
table. Write in everything from Step 1 (problem statement, target
definition, success criteria, business constraint, stakeholders,
constraints, baseline, kickoff date). Restate — don't re-ask — the cloud
and experiment-tracking decisions already recorded from `/ml-init`, so the
charter reads as one complete document.

## Step 4 — Wrap up

Check off Step 1 in the lifecycle checklist. Tell the user the charter and
timeline are in `PROJECT.md`, restate the business constraint and target
definition back to them for confirmation, and point to `/ml-data` as next,
unless it's already been run.
