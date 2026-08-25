# ml-cursor-init

Cursor skills for a 12-step ML model lifecycle. Install once, then run the
skills **in this order**. Do not skip. Do not start `/ml-model` until
`/ml-prep` is done.

The numbers below are the **process order** (what you type). The lifecycle
step numbers (1–12) are what `PROJECT.md` tracks.

```
you type          lifecycle
─────────────     ─────────────────────────────
/ml-init          scaffold (not a numbered step)
/ml-define        1. Problem Formulation
/ml-data          2. Data Gathering & Structuring
                  3. Data Exploration  (train window only)
/ml-prep          4. Feature Engineering
/ml-model         5. Multi-Model Training          ← train + eval only
/ml-evaluate      6. Model Evaluation              ← first time you touch test
/ml-explain       7. Model Interpretability
/ml-calibrate     8. Model Calibration
                  8.5 Final Fit for Deployment
/ml-document      9. Model Documentation
                  10–12 not in this pack yet
```

---

## Longer example — same order, loan-default project

This is the path `examples/demo_loan_default/` already followed. Use it as
the picture of “what happens at each step.” Your project will differ in
names and numbers; the **order** does not.

Open an empty folder (or the folder that should become the repo). Agent
chat, not Ask. Type `/` plus the skill name and send. Answer the questions;
then run the next skill.

### 0. `/ml-init` — scaffold

**When:** brand-new project. Once.

**You answer:**

- Name: `demo_loan_default`
- One-liner: predict 12-month consumer-loan default so high-risk apps go
  to manual review
- Type: binary classification
- Cloud: none (local)
- Tracking: local logs under `experiments/`
- Env: `uv`

**You get:** `data/`, `src/`, `configs/`, `experiments/`, `models/`,
`reports/`, `PROJECT.md`, git init. Cloud/tracking are locked here and
not re-asked later.

**Next:** `/ml-define`

### 1. `/ml-define` — problem formulation

**When:** immediately after init.

**You answer:**

- Target: `default = 1` if the loan defaults within 12 months of
  origination
- Success: ROC-AUC ≥ 0.85 on test, and expected review cost below the
  “review nothing” baseline
- Constraint (not the threshold yet): FP ≈ $50 (needless review), FN ≈
  $2,000 (missed default)
- Kickoff date (for the per-step timeline)

**You get:** a filled charter in `PROJECT.md`. The operating point is
**not** chosen here — that waits for step 6.

**Next:** `/ml-data`

### 2–3. `/ml-data` — gather, split, explore train only

**When:** after the target is precise enough to land data.

**You answer:** source, grain (one row = one origination), and the three
**date** windows. Default split is Out-of-Time, not a random shuffle.

Demo windows:

| Window | Dates | Role |
|---|---|---|
| Train | 2024-01-01 → 2025-03-11 | fit |
| Eval | 2025-03-11 → 2025-08-02 | tune / pick among models |
| Test | 2025-08-02 → 2025-12-30 | honest holdout — **hands off** until `/ml-evaluate` |

**You get:** `data/raw/` untouched, `data/processed/train|eval|test.*`,
`configs/sampling.yaml` (seed 42 in the demo), data dictionary, train-only
EDA + anomalies. Eval/test get a schema/row-count check only — no target
plots.

**Next:** `/ml-prep`

### 4. `/ml-prep` — features

**When:** split exists. Transforms fit on **train only**, then applied to
eval and test.

**You answer:** which derived features to keep; what to do with any
leakage-flagged columns (usually drop).

Demo: base origination fields plus `delinquency_rate` and
`utilization_x_dti`; categoricals native for CatBoost, one-hot for RF/XGB
from train categories only.

**You get:** `configs/features.yaml`, `data/processed/features_*.csv`,
preprocessors under `models/preprocessing/`.

**Next:** `/ml-model`

### 5. `/ml-model` — bake-off (still no test)

**When:** features exist. Trains on train, tunes on eval.

Default roster: **Random Forest + XGBoost + CatBoost**. AutoGluon only if
you ask.

**You get:** model files under `models/`, comparison table, run logs in
`experiments/`. No champion yet — that is a test-window call.

**Next:** `/ml-evaluate`

### 6. `/ml-evaluate` — test, once

**When:** the three models are trained. **First skill that scores test.**
Do not retune after you see these numbers.

**You get:**

- Performance table (ROC-AUC, precision, recall, F1)
- Champion (demo: CatBoost, test ROC-AUC 0.934)
- ROC and PR charts at a stated threshold
- Operating table: FPR 0% → 5% in 0.25% steps
- Recommended row from that table, tied to the FP/FN costs from
  `/ml-define` (demo: ~4.5% FPR, threshold 0.405)

**Next:** `/ml-explain`

### 7. `/ml-explain` — TreeSHAP

**When:** champion is named. SHAP on the **same** test window. Not LIME.

**You get:** global importance + PDP/ALE/ICE on top variables; local
cohorts in this exact sort:

- TP — actual default, `pred_1` descending
- FP — non-default but high score, `pred_1` descending
- FN — actual default but low score, `pred_1` ascending

Plus a bias probe across those cohorts.

**Next:** `/ml-calibrate`

### 8. `/ml-calibrate` — scores, then full-data fit

**When:** explanations are written. This skill does two lifecycle steps.

1. **Step 8** — reliability diagram + rank spline onto a 1–1000 scale
   (default bands: top 0.25% → 1000–900, next 0.25% → 900–800, next 4.5%
   → 800–500, rest → 500–0).
2. **Step 8.5** — refit the champion on train+eval+test (same
   hyperparameters, no re-tune). Re-score test; if the distribution
   shifted, recalibrate and flag that as slightly optimistic.

Demo: CatBoost refit on 6,000 rows; scores shifted, so calibration was
redone.

**Next:** `/ml-document`

### 9. `/ml-document` — one shareable report

**When:** you want something to hand someone. Does not invent new
analysis; it gathers `PROJECT.md`, configs, figures, and run logs into
`reports/<project>_report.md`. Missing steps are marked `[Pending]`, not
filled with fake numbers.

Demo output: `examples/demo_loan_default/reports/demo_loan_default_report.md`
(and `reports/report.html`).

**After that:** steps 10–12 (`/ml-integrate`, `/ml-monitor`, `/ml-retrain`)
are not in this pack. Stop, or design them by hand.

---

## How to invoke

In **Agent** chat:

| You type | What happens |
|---|---|
| `/ml-init` (or `/` then pick it) | Runs that step |
| `@ml-init` | Only attaches the instructions; still say what to do |

Use slash when you want the next step in the process to run.

---

## Install (once)

```bash
git clone git@github.com:GeorgeHayduke/ml-cursor-init.git
cd ml-cursor-init
chmod +x install.sh
./install.sh
```

Reload Cursor: Command Palette → **Developer: Reload Window**. Confirm
under **Customize → Skills**.

Skills go to `~/.cursor/skills/` (every workspace). The lifecycle rule
goes to `~/.cursor/rules/ml-lifecycle.mdc`. If a chat ignores the rule,
paste that file into **Cursor Settings → Rules → User Rules**.

To copy into one repo instead of installing globally:

```bash
mkdir -p .cursor/skills .cursor/rules templates
cp -R path/to/ml-cursor-init/.cursor/skills/. .cursor/skills/
cp path/to/ml-cursor-init/.cursor/rules/ml-lifecycle.mdc .cursor/rules/
cp -R path/to/ml-cursor-init/templates/. templates/
```

---

## Already have a trained model?

Skip the lifecycle and generate the HTML report:

```bash
python templates/document_model.py \
  --model catboost:models/catboost.cbm:catboost \
  --data data/test_labeled.csv \
  --target defaulted \
  --output-dir reports/
```

Or `/ml-document` and tell the agent to use that script.

Do not put skills in `~/.cursor/skills-cursor/` — that directory is
Cursor’s own.
