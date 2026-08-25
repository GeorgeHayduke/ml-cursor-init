---
name: ml-data
description: Land data, lock the Out-of-Time sampling split, and explore the train window only. Use for data gathering, sampling strategy, EDA, or when the user says /ml-data.
disable-model-invocation: true
---

# /ml-data — Data Gathering, Structuring & Exploration (Steps 2-3)

Run after `/ml-init` (and ideally `/ml-define`, though this works from
whatever problem statement exists in `PROJECT.md` if `/ml-define` hasn't run
yet — flag that gap rather than blocking on it). This skill owns the
**sampling strategy** for the project — every later step reads the split it
produces here rather than re-deriving or re-asking for it.

## Step 1 — Gather & structure

Ask (or confirm from `PROJECT.md` if already stated): data source(s) — files,
warehouse table(s)/query, API — the grain (one row = ?), and the target
variable definition.

Check `PROJECT.md` for the cloud decision made in `/ml-init`:

- **No cloud:** source/land data locally as below.
- **Cloud chosen (GCP/AWS/Azure/other):** default data gathering and
  structuring to that provider's storage (a GCS bucket / S3 path / Azure
  Blob container, recorded in `configs/`), mirroring the same
  raw/interim/processed layout remotely. Keep a local `data/` folder as a
  thin cache or sample for interactive dev only — don't assume the full
  dataset lands on local disk once cloud is in play.

Land raw extracts under `data/raw/` untouched. Do any joining/reshaping into
`data/interim/`. While structuring, flag — don't silently include — any
column that's only knowable after the outcome occurred (an obvious leakage
risk); note it in `PROJECT.md` even though feature selection itself happens
in `/ml-prep`.

Write a short data dictionary (column, type, source, notes) to
`reports/data_dictionary.md`. Don't skip columns because they seem
self-explanatory.

## Step 2 — Sampling strategy (the split)

This locks in the Sampling Strategy section from `ml-lifecycle.mdc`. Ask,
don't assume:

- **Method:** default is **Out-of-Time (OOT)**. If the user wants
  random/stratified instead, get explicit confirmation and record the
  deviation with a reason — OOT is the house default, not the only option.
- **Seed:** pick and record one (e.g. 42, or user-specified) — this seed is
  reused by every later step, not re-rolled per skill.
- **Train window:** date range.
- **Eval window:** date range, immediately after train, no overlap.
- **Test window:** most recent date range, no overlap with eval, held out
  untouched until steps 6-8.

Validate before writing anything: windows are chronologically ordered and
non-overlapping, test is the most recent, and each window has enough volume
to be statistically meaningful (for a rare-event target, flag it if eval or
test looks like it has too few positive-class events to evaluate on — don't
proceed silently).

Write the finalized strategy to `configs/sampling.yaml` (method, seed,
window boundaries) — this file is the single source of truth every
downstream skill reads, not something re-asked at each step. Split
`data/interim/` into `data/processed/train.*`, `eval.*`, `test.*`
accordingly.

Update `PROJECT.md` and `templates/report_template.md` Section 2.1 with the
finalized sampling strategy.

## Step 3 — Data exploration (train window only)

Explore only `data/processed/train.*` in depth. Eval and test get, at most,
a structural sanity check — row counts, schema match, no obviously broken
values. Do **not** profile their distributions or their relationship to the
target — that's a hard rule from `ml-lifecycle.mdc`, not a suggestion, and
it's what keeps steps 6-8 honest later.

Within the train window:

- **Exploration relative to the target:** pick the 3-6 features that matter
  most and show how each relates to the target (a chart or crosstab per
  relationship, one line on why it matters) — not a generic univariate pass
  over every column.
- **Anomaly detection:** identify 5-10 anomalous records and explain each —
  what's anomalous, likely cause (data error vs. rare-but-real vs.
  something else), and what was done about it (kept / excluded /
  corrected).

Save generated charts to `reports/figures/` and write the findings straight
into `templates/report_template.md` Section 3 (3.1 and the anomaly table) —
this section should be close to final after this step, not a placeholder
someone fills in later from memory.

## Step 4 — Wrap up

Check off steps 2 and 3 in `PROJECT.md`. Tell the user what was decided —
sampling windows, seed, key EDA findings, anomaly count, any leakage risks
flagged — and that `/ml-prep` (Feature Engineering, step 4) is next.
