---
name: ml-init
description: Scaffold a new ML project repo, lock cloud/tracking decisions, and write PROJECT.md. Use when starting an ML project, initializing a model repo, or when the user says /ml-init.
disable-model-invocation: true
---

# /ml-init — Initialize a new ML project repo

Run this once, at the very start of a new project. It scaffolds the repo and
locks in the handful of decisions that shouldn't be revisited mid-project
(cloud, tracking). Follow the standing conventions in
`.cursor/rules/ml-lifecycle.mdc` throughout — this skill is what puts those
conventions into an actual repo.

## Kit files

Resolve in this order (do not invent a template from scratch if a kit file exists):

1. `assets/` next to this `SKILL.md` (global install via `./install.sh`)
2. Current workspace, if this kit is open: `templates/` and `.cursor/rules/ml-lifecycle.mdc`
3. Ask where the kit clone lives if neither exists

Copy into the new project:

- `ml-lifecycle.mdc` → `<project>/.cursor/rules/ml-lifecycle.mdc`
- `report_template.md` → `<project>/templates/report_template.md`
- `document_model.py` and `model_report_template.html` → `<project>/templates/`
- `integration.yaml`, `monitoring.yaml`, `retrain.yaml` → `<project>/templates/`
  (filled later by `/ml-integrate` / `/ml-monitor` / `/ml-retrain`)
- This skill pack (the `ml-*` folders under `.cursor/skills/`) → `<project>/.cursor/skills/` so teammates without a global install still get `/ml-init` etc.

## Step 1 — Ask before scaffolding

Ask the user for (don't guess or default silently on any of these):

1. **Project name** and a one-sentence problem statement (this seeds step 1,
   Problem Formulation, and `/ml-define` will expand on it later).
2. **Problem type** — binary classification, multiclass, regression, other —
   just enough to pick sane defaults for later steps (e.g. ROC/PR only make
   sense for classification).
3. **Cloud decision** — is this project running in the cloud at all? If yes:
   GCP, AWS, Azure, or other (name it). This is a one-time, up-front
   decision per the rule file — do not ask again later, and do not infer it
   from data size.
4. **Experiment tracking start point** — confirm the default (structured
   local run logs under `experiments/`) or, if the user wants to start with
   MLflow immediately, note that instead.
5. **Environment tool** — default to `uv` for the Python environment unless
   the user names a preference (conda, poetry, plain venv).

Do not proceed to Step 2 until all five are answered.

## Step 2 — Scaffold the repo

Create this structure (adjust names only if the user's answers above require
it — e.g. add a cloud directory only if cloud was chosen):

```
<project>/
├── .cursor/
│   ├── rules/ml-lifecycle.mdc
│   └── skills/               # copy the ml-* skill pack from this kit
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── notebooks/
├── src/
│   └── <project_slug>/
│       └── __init__.py
├── experiments/              # structured local run logs (json/yaml per run)
├── models/                   # trained model artifacts (gitignored contents)
├── reports/
│   └── figures/
├── configs/
├── templates/                # report_template.md + document_model.py
├── cloud/                    # only if a cloud provider was chosen
│   └── <gcp|aws|azure|other>/
│       └── README.md         # stub — fill in as integration/monitoring
│                             #   steps (10/11) approach
├── PROJECT.md                # see Step 3
├── README.md
├── .gitignore
├── .pre-commit-config.yaml
└── pyproject.toml            # or environment.yml / poetry.lock per Step 1.5
```

- `data/`, `models/`, and `experiments/` outputs go in `.gitignore` — only
  the directory structure and any small reference/schema files are tracked.
- If MLflow was chosen in Step 1.4 instead of local logs, add an
  `mlruns/`-aware `.gitignore` entry and a one-line note in `PROJECT.md`
  instead of the `experiments/` folder description.

## Step 3 — Write `PROJECT.md`

Populate it with:

- The problem statement and problem type from Step 1.
- The cloud decision (or "local-only" if none was chosen), stated plainly so
  later steps don't have to re-ask.
- The experiment-tracking approach in use.
- A **rough placeholder timeline** only — a single line noting the default
  5-week cadence exists. Do not build out the full week-by-week or
  per-step table here: `/ml-define` (Step 1, Problem Formulation) owns the
  authoritative timeline, built as a per-step table with real target dates
  once a kickoff date is known. Building a detailed table at init time,
  before the kickoff date or scope is confirmed, just creates something
  `/ml-define` has to overwrite.

- A checklist of all 13 steps (1 through 12, plus 8.5 — Final Fit for
  Deployment, which sits between Calibration and Documentation), unchecked,
  so progress is visible at a glance in every project going forward.

## Step 4 — Git

`git init` if not already a repo, then a first commit containing the
scaffold (message: `chore: initialize ML project scaffold`). Do not commit
anything under `data/`, `models/`, or run-log outputs even if they happen to
exist already — respect `.gitignore` from the first commit.

## Step 5 — Hand off

Tell the user the repo is scaffolded, restate the cloud/tracking decisions
back to them so there's no ambiguity, and point them to `/ml-define` as the
next skill (step 1, Problem Formulation, in full — this skill only seeds it).
