---
name: ml-integrate
description: Ship the champion as predict / explain / label — batch, API, or pipeline — with the locked operating point and calibration. Use after /ml-document, for production scoring, serving, or when the user says /ml-integrate.
disable-model-invocation: true
---

# /ml-integrate — Model Integration (Step 10)

Run after `/ml-document` (step 9) and after Step 8.5 has produced
`models/champion_final.*`. This skill does **not** retrain, retune, or
pick a new threshold. It wraps the locked champion, the Step 6 operating
point, and the Step 8 score mapping into something that can score new
records: **predict**, **explain** (TreeSHAP, for actioned rows), **label**
(the operating-point decision).

If `PROJECT.md` still has Step 8.5 unchecked, stop and point back to
`/ml-calibrate`. If there is no champion or no recommended operating
point, stop and point back to `/ml-evaluate`.

Start from `templates/integration.yaml` (or `assets/integration.yaml`
next to this `SKILL.md` after a global install). Fill it; don't invent a
second threshold.

## Step 1 — Confirm the locked artifacts

Read, don't re-derive:

- Champion path and family from `models/champion.json` /
  `models/champion_final.*`
- Operating point (threshold, FPR/TPR, precision/recall, F1) from the
  Step 6 recommendation
- Calibration / 1–1000 mapping from Step 8 (and 8.5 if it was redone)
- Preprocessors from `/ml-prep` (`models/preprocessing/`)
- Feature list from `configs/features.yaml`
- Cloud decision from `PROJECT.md` — honor it; do not introduce a
  provider that wasn't chosen at `/ml-init`

## Step 2 — Ask serving shape

Ask, don't guess:

1. **Mode:** batch job (default), HTTP API, or embed in an existing
   pipeline. If they already have a scoring job, wrap that — don't stand
   up a new service for its own sake.
2. **Grain and schedule:** one row = the same grain as training; how
   often new rows arrive (nightly origination file, streaming, etc.).
3. **Action:** what the operating-point **label** does in the real
   process (e.g. route to manual review). The label is the decision at
   the locked threshold, not a second model.
4. **Explanations:** TreeSHAP on the **actioned** cohort only (default),
   or on every row (confirm — cost). Same TreeSHAP path as `/ml-explain`;
   not LIME.
5. **Cutover:** shadow (score alongside, no action) vs replace the
   current process. Default to shadow first if a current process exists.

Do not proceed to Step 3 until 1–3 are answered.

## Step 3 — Scoring contract

Every scored record must emit at least:

| Field | Meaning |
|---|---|
| `record_id` | Same id used in training/reporting |
| `pred_1` | Raw positive-class score from the champion |
| `score_norm` | 1–1000 mapped score from the locked calibration |
| `decision` | 1 if `pred_1` ≥ locked threshold, else 0 |
| `model_version` | Git commit + artifact filename of `champion_final` |
| `scored_at` | Timestamp |

Write this contract into `configs/integration.yaml` together with mode,
schedule, threshold, champion path, preprocessor path, and whether
explanations run on actioned rows only.

Implement `src/<project_slug>/score.py` (name may match the project) that:

- Loads **only** `champion_final` and the persisted train-fit
  preprocessors — never refits on production data
- Applies the same CatBoost-native vs RF/XGB-encoded split as `/ml-prep`
- Refuses to score if required feature columns are missing
- Is idempotent on `record_id` + `model_version` (re-runs don't double-act)

If mode is API, keep the same function and add a thin wrapper; the
scoring logic must not fork. If cloud was chosen, put the job/service
definition under `cloud/<provider>/` and use that provider's batch/serving
primitive (Vertex / SageMaker / Azure ML, etc.) rather than a local cron
pretending to be production.

## Step 4 — Explain path

Add `src/<project_slug>/explain.py` (or a flag on the scorer) that
computes TreeSHAP for rows with `decision = 1` (or whatever the actioned
class is). Persist top contributing features with direction, same local
shape as `/ml-explain`. Do not block the batch on explanations if the
user chose async explain.

## Step 5 — Write up and wrap

Run `python -m pytest tests/test_lifecycle_steps.py::TestStep10Integrate`. Do
not wrap until it passes.

Fill `templates/report_template.md` Section 10 (and the project report
if it already exists). Check off Step 10 in `PROJECT.md`. Restate mode,
threshold, and champion artifact. Point to `/ml-monitor` as next —
monitoring is ongoing, but the first monitor config should be written
before the first unscored week of production.
