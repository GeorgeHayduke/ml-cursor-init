# demo_loan_default — Model Development Report

**Author:** Claude (Cowork) · **Date:** 2026-08-24 · **Repo:** `env_cursor_ml/examples/demo_loan_default` · **Status:** simulated worked example (synthetic data)

This is a fully worked, end-to-end run of the `env_cursor_ml` lifecycle —
`/ml-define` → `/ml-data` → `/ml-prep` → `/ml-model` → `/ml-evaluate` →
`/ml-explain` → `/ml-calibrate` — against synthetic data, produced to prove
out the rules/templates/commands built so far, not a real credit decision.
Steps 9 onward that don't have a command yet (10-12) are marked pending.

**One-line verdict:** the CatBoost champion (test ROC-AUC 0.934) is a
reasonable model on this synthetic problem; the interesting finding is
methodological, not predictive — the Step 8.5 full-data refit shifted
scores enough that the pre-refit calibration had to be thrown out, which is
exactly the scenario `ml-lifecycle.mdc`'s "check, don't assume" rule was
written for.

---

## 1. Problem Formulation

Predict whether a consumer loan defaults within 12 months of origination,
to route high-risk applications to manual underwriting review before
disbursement.

- **Target variable:** `default = 1` if the loan defaults within 12 months
  of origination (synthetic, ~12% positive rate).
- **Success criteria:** ROC-AUC ≥ 0.85 on test.
- **Business constraint:** false positive (unneeded review) ≈ $50; false
  negative (missed default) ≈ $2,000. Used in Section 6 to pick the
  operating point — not decided here.
- **Baseline:** none — exploratory, simulated example.
- **Timeline & cloud decision:** see `PROJECT.md` — local-only, no cloud.

---

## 2. Data Gathering & Structuring

Synthetic data via `sklearn.datasets.make_classification` (n=6,000, 12
features, ~12% positive rate), with a fabricated `origination_date` and a
neutral `region` categorical (A-D) added on top, standing in for a real
warehouse pull.

| Source | Rows | Date range | Notes |
|---|---|---|---|
| synthetic (`make_classification`, seed=42) | 6,000 | 2024-01-01 – 2025-12-30 | region A-D added independently; 8 records seeded as deliberate anomalies |

### 2.1 Sampling strategy

Method: **Out-of-Time (OOT)**, 60/20/20 split by `origination_date`.

| | Value |
|---|---|
| Seed | 42 |
| Train window | 2024-01-01 – 2025-03-11 (n=3,600, default rate 13.08%) |
| Eval window | 2025-03-11 – 2025-08-02 (n=1,200, default rate 11.83%) |
| Test window | 2025-08-02 – 2025-12-30 (n=1,200, default rate 12.67%) |

---

## 3. Data Exploration

*(Train window only, per house convention.)*

### 3.1 Exploration relative to the target

Correlation with `default`, train window:

| Feature | Correlation |
|---|---|
| debt_to_income | -0.191 |
| loan_to_value | -0.187 |
| payment_history_score | -0.168 |
| months_on_book | -0.159 |
| num_delinquencies_24m | +0.159 |
| credit_utilization | +0.150 |

The rest (`revolving_balance`, `employment_stability`, `income`,
`feat_misc_1`, `credit_score_proxy`, `num_inquiries_6m`) show near-zero
linear correlation individually — expected, since `make_classification`'s
signal is distributed non-linearly across features and interactions, which
is exactly what the tree models pick up in Section 7's SHAP importance.

### 3.2 Anomaly detection

7 anomalous records flagged (z-score > 4 on train window) out of 8
deliberately seeded:

| Record ID | Field | Value | Likely cause | Action |
|---|---|---|---|---|
| L102964 | income | 15.31 | seeded outlier (extreme income) | kept, flagged |
| L100859 | income | 16.05 | seeded outlier | kept, flagged |
| L101268 | income | -6.44 | negative income — data error | flagged for exclusion in a real project |
| L103800 | income | 8.96 | seeded outlier | kept, flagged |
| L100178 | credit_utilization | 13.10 | seeded outlier (extreme utilization) | kept, flagged |
| L100223 | credit_utilization | 12.33 | seeded outlier | kept, flagged |
| L102341 | months_on_book | 8.44 | mild outlier, plausible | kept |

(One seeded `months_on_book = -1` garbage record fell below the z-score
threshold and wasn't auto-flagged — a reminder that a single anomaly
detection method won't catch everything; `/ml-prep` clips negative
`months_on_book` regardless, see Section 4.)

---

## 4. Feature Engineering

Leakage review: no columns excluded — all fields are knowable at
origination time.

| Feature | Derivation | Rationale |
|---|---|---|
| delinquency_rate | `num_delinquencies_24m / (months_on_book_clipped + 1)` | recent delinquency intensity |
| utilization_x_dti | `credit_utilization * debt_to_income` | interaction of two individually predictive fields |
| region (CatBoost) | passed through natively | CatBoost handles categoricals directly |
| region_A..D (RF/XGBoost) | one-hot, categories fit on train only | RF/XGBoost need numeric input |

15 features for CatBoost, 18 for RF/XGBoost (same underlying definitions,
different categorical representation) — see `configs/features.yaml`.

---

## 5. Multi-Model Training

Default bake-off: Random Forest + XGBoost + CatBoost. AutoGluon not used
this run (not requested).

| Model | Eval ROC-AUC | Best hyperparameters | Trials |
|---|---|---|---|
| CatBoost | 0.9507 | depth=6, learning_rate=0.08, iterations=250 | 5 |
| XGBoost | 0.9411 | max_depth=5, learning_rate=0.05, n_estimators=250 | 5 |
| Random Forest | 0.9180 | max_depth=10, min_samples_leaf=5, n_estimators=400 | 5 |

### 5.1 Hyperparameter tuning

Manual grid, 5 trials per model, tuned against the eval window only (no
k-fold CV, per house convention — see `ml-lifecycle.mdc`). 5 trials per
model is thin for a real project — enough to demonstrate the mechanism
here, not enough to claim the search was exhaustive. A real project should
run more trials, especially for CatBoost/XGBoost where the eval gap between
trial 3 and the winner was under 0.01 AUC — plausibly noise at this trial
count.

---

## 6. Model Evaluation

*(Test window, scored once.)*

### 6.1 Performance table

| Model | ROC-AUC | Precision@0.5 | Recall@0.5 | F1@0.5 |
|---|---|---|---|---|
| CatBoost | 0.9340 | 0.7947 | 0.7895 | 0.7921 |
| XGBoost | 0.9247 | 0.7312 | 0.7697 | 0.7500 |
| Random Forest | 0.9007 | 0.6218 | 0.6382 | 0.6299 |

**F1, in one line:** the harmonic mean of precision and recall — it
penalizes a model that trades one off heavily for the other, so a high F1
means the model is reasonably balanced between catching positives and not
flooding results with false alarms.

**Champion: CatBoost** (highest test ROC-AUC — consistent with its eval
ranking in Section 5, so ranking didn't reshuffle when tested honestly).

### 6.2 / 6.3 ROC and Precision-Recall charts

See `reports/figures/step6_roc.png` and `step6_pr.png`. At threshold=0.5:
FPR=0.019, TPR=0.789 (ROC); precision=0.795, recall=0.789 (PR).

### 6.4 Operating table

Swept by FPR 0% → 5% in 0.25% increments (21 rows), champion only:

| FPR | Threshold | TPR | Precision | Recall | F1 | Expected cost |
|---|---|---|---|---|---|---|
| 0.00% | 0.913 | 0.388 | 1.000 | 0.388 | 0.559 | $185,987 |
| 1.00% | 0.780 | 0.592 | 0.900 | 0.592 | 0.714 | $124,526 |
| 2.00% | 0.605 | 0.757 | 0.871 | 0.757 | 0.810 | $75,042 |
| 3.00% | 0.519 | 0.789 | 0.800 | 0.789 | 0.795 | $65,564 |
| 4.00% | 0.461 | 0.803 | 0.767 | 0.803 | 0.785 | $62,106 |
| **4.50%** | **0.405** | **0.816** | **0.729** | **0.816** | **0.770** | **$58,355 (min)** |
| 5.00% | 0.405 | 0.816 | 0.729 | 0.816 | 0.770 | $58,617 |

(Full 21-row table in `reports/figures/step6_operating_table.csv`.)

**Recommended operating point: FPR ≈ 4.5%, threshold ≈ 0.405.** Against
the $50-FP / $2,000-FN constraint from Section 1, expected cost is
minimized around FPR=4.5% ($58,355 across the 1,200-record test window) —
below that, missed defaults dominate cost; above it, marginal reviews stop
paying for themselves. This is $181,632 cheaper than reviewing nothing
(FPR=0% catches only 38.8% of defaults) and meaningfully cheaper than the
naive threshold=0.5 point (FPR≈1.9%, ~$70-80k range by interpolation).

---

## 7. Model Interpretability

Champion: CatBoost, TreeSHAP via `get_feature_importance(type="ShapValues")`.

### 7.1 Global explanations

Top variables by mean |SHAP|: `months_on_book`, `feat_misc_1`,
`credit_utilization`, `credit_score_proxy`, `debt_to_income`. See
`step7_global_importance.png` and `step7_pdp.png`. Note: `feat_misc_1`
ranking highly is a quirk of the synthetic generator (it isn't guaranteed
to be pure noise) — in a real project every top-ranked variable needs a
business-meaning sanity check before being trusted, not just a SHAP score.

### 7.2 Local explanations

Full tables in `step7_cohort_{tp,fp,fn}.csv` (top 10 each, n=120/31/32
total in each full cohort). Sample:

**TP** (ordered by pred_1 descending) — top score 0.9985 (L101701) down to
0.9934 (L105623), all correctly caught at high confidence.

**FP** (ordered by pred_1 descending) — top false alarm 0.9121 (L100569):
scored as high-risk, didn't actually default. These sit just below the
true-positive score range, meaning the model isn't wildly wrong on them,
just on the wrong side of the line.

**FN** (ordered by pred_1 ascending) — worst miss 0.0029 (L105565): a
default the model was highly confident would *not* happen. These are the
cases most worth a manual look — a confidently-wrong miss is a different
problem than a borderline one.

### 7.3 Bias probe

Region share, full cohorts (not just the top-10 shown above — a 10-record
sample is too small to draw a conclusion from):

| Region | Overall | TP (n=120) | FP (n=31) | FN (n=32) |
|---|---|---|---|---|
| A | 37.7% | 29.2% | 29.0% | 43.8% |
| B | 28.4% | 30.0% | 41.9% | 25.0% |
| C | 19.8% | 25.0% | 22.6% | 12.5% |
| D | 14.2% | 15.8% | 6.5% | 18.8% |

Largest deviation from the overall distribution: 13.5 points (region B in
the FP cohort). Below the 15-point flag threshold used here — **no
notable skew found**. Worth noting for methodology: the top-10 display
tables in 7.2 looked much more skewed by region (up to 32 points off) —
that's a small-sample artifact, not a real signal, which is exactly why
the probe needs to run against the full cohort rather than the display
table.

---

## 8. Model Calibration

### 8.1 Calibration curve

Mean |observed − predicted| across 10 quantile bins: **0.045** —
reasonably well-calibrated pre-refit. See `step8_calibration_curve.png`.

### 8.2 Score normalization

Rank-based spline (PCHIP through the band boundaries), test window:

| Normalized band | Target share | Actual share | Raw score range |
|---|---|---|---|
| 1000-900 | 0.25% | 0.25% | [0.998, 0.998] |
| 900-800 | 0.25% | 0.25% | [0.997, 0.998] |
| 800-500 | 4.50% | 4.50% | [0.912, 0.996] |
| 500-0 | 95.00% | 95.00% | [0.000, 0.911] |

Bands hit their target population share exactly by construction (rank-based).

---

## 8.5 Final Fit for Deployment

CatBoost refit on the full dataset (n=6,000 = train+eval+test combined),
same hyperparameters (depth=6, learning_rate=0.08, iterations=250), no
re-tuning.

**Calibration-drift check:**

1. Re-scored the 1,200 test-window records with the refit model.
2. Compared to the pre-refit test-window scores: mean |post − pre| score
   difference = **0.058**, correlation = **0.908**.
3. **Decision: recalibrate.** A 0.058 mean absolute shift with correlation
   under 0.91 is a meaningful move, not noise — the pre-refit calibration
   curve and score bands in Section 8 should **not** be carried forward
   unchanged onto `models/champion_final.cbm`. (Flag: any recalibration
   using these same test-window records would be slightly optimistic,
   since they also trained the refit model — a real project should hold
   back one more small recent slice before the full refit specifically for
   this check, as `ml-lifecycle.mdc` recommends.)

This is the clearest finding of this simulated run: **"maybe not" was the
wrong instinct here** — the refit moved scores enough to matter. See
`step8_5_drift_check.png`.

---

## 9. Model Documentation

This report, generated by walking through the same assembly `/ml-document`
would perform. Reproducibility: seed=42 throughout;
`configs/sampling.yaml` and `configs/features.yaml` in this folder are the
exact configs used; library versions — scikit-learn 1.8.0, xgboost 3.2.0,
catboost 1.2.10, shap 0.51.0.

**Limitations & assumptions:** synthetic data — no real predictive
relationships, do not draw real underwriting conclusions from this report.
`feat_misc_1`'s ranking (Section 7.1) is a generator artifact, not a
business signal. The Section 8.5 finding (recalibration needed) is the
one result here likely to generalize to real projects using this
framework — treat it as a standing warning, not a one-off.

## 10. Model Integration — `[Pending — /ml-integrate not yet built]`

## 11. Model Monitoring — `[Pending — /ml-monitor not yet built]`

## 12. Periodic Retraining — `[Pending — /ml-retrain not yet built]`

---

*Generated as a worked example of `report_template.md` in `env_cursor_ml`,
via `src/01_data.py` through `src/06_calibrate.py`. Re-run those scripts
(seed=42) to reproduce every number above exactly.*
