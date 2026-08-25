---
name: ml-document
description: Assemble the finished model report from PROJECT.md, configs, and step artifacts. Use for model documentation, HTML reports, document_model.py, or when the user says /ml-document.
disable-model-invocation: true
---

# /ml-document — Model Documentation (Step 9)

Run once steps 1-8.5 have produced real content to assemble — this skill
doesn't generate new analysis, it gathers what earlier steps already
produced into one finished, shareable document. It can be run partway
through a project to snapshot current status, but the output should say
plainly which sections are still pending rather than silently leaving old
placeholder text in place.

This is also the skill to reach for when someone wants the finished
report to hand to another project or another person — the output file is
self-contained and portable, unlike `PROJECT.md` (which is this repo's
working status doc) or `templates/report_template.md` (which is the blank
template, not a finished report).

## Kit files

Resolve the report template and `document_model.py` in this order:

1. `assets/report_template.md` and `scripts/document_model.py` next to this
   `SKILL.md` (global install)
2. Workspace `templates/report_template.md` and `templates/document_model.py`

If the user wants a mechanical HTML report from an already-trained model
(no full lifecycle), run `scripts/document_model.py` (or the workspace
copy) instead of assembling markdown by hand. The HTML template is
`assets/model_report_template.html` or `templates/model_report_template.html`.

## Step 1 — Check what's actually available

Look for, and note what's missing rather than guessing or inventing
content for it:

- `PROJECT.md` — charter from `/ml-define` (problem statement, target
  definition, success criteria, business constraint, stakeholders,
  constraints, baseline, timeline).
- `configs/sampling.yaml`, `configs/features.yaml` — from `/ml-data` and
  `/ml-prep`.
- Model comparison + run logs — from `/ml-model`.
- Evaluation artifacts (performance table, ROC/PR charts, operating table,
  HP-tuning notes) — from `/ml-evaluate`, once it exists.
- Interpretability artifacts (global importance, PDP/ALE/ICE, TP/FP/FN
  cohorts, bias probe) — from `/ml-explain`, once it exists.
- Calibration artifacts (reliability diagram, score-normalization bands) —
  from `/ml-calibrate`, once it exists.
- The Step 8.5 final-fit decision and its calibration-drift check.
- Integration / monitoring / retrain configs (`configs/integration.yaml`,
  `configs/monitoring.yaml`, `configs/retrain.yaml`) once those skills
  have run.

If a later-step skill hasn't been run yet in this project, mark that
section `[Pending — Step N not yet run]` in the output rather than
fabricating numbers or leaving the generic template prompt text in place.

## Step 2 — Assemble the report

Copy `templates/report_template.md` to `reports/<project_slug>_report.md`
and replace every section with real content from Step 1's sources — not
more placeholders. Strip the instructional blockquotes (`> Fill in: ...`)
as each section is populated; a finished report shouldn't still contain
authoring prompts. Keep the table structures (performance table, operating
table, cohort tables, score-normalization bands) exactly as specified in
the template — this is what makes reports comparable across projects.

## Step 3 — Add what the template doesn't already cover

- **Reproducibility appendix:** git commit hash, environment spec (lockfile
  / `pyproject.toml`), the exact `configs/sampling.yaml` and `configs/
  features.yaml` used, and library versions for whichever of RF / XGBoost /
  CatBoost / AutoGluon / SHAP were actually used.
- **Limitations & assumptions:** what this model should explicitly not be
  used for, known blind spots (e.g. thin coverage for a segment), and any
  staleness assumption (how long before this model needs retraining
  regardless of monitoring signals — ties to Step 12).
- **Status & sign-off:** draft vs. final, author, and reviewer if this
  project has one — carried at the top of the document per the template.

## Step 4 — Wrap up

Run `python -m pytest tests/test_lifecycle_steps.py::TestStep09Document`. Do
not wrap until it passes.

Check off Step 9 in `PROJECT.md`. Tell the user the path to the finished
report and which sections (if any) are still `[Pending]`. Note explicitly
that this report should be regenerated — not hand-edited and left stale —
whenever an upstream step changes materially (a recalibration after Step
8.5, a retrain in Step 12), and that `/ml-integrate` (step 10) is next once
the report is complete.
