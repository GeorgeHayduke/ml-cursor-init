<!--
report_template.md

Shared template referenced by `.cursor/commands/ml-document.md` (step 9,
Model Documentation) and by `.cursor/rules/ml-lifecycle.mdc`. One copy lives
here in env_cursor_ml; each project repo gets its own filled-in copy under
`reports/`.

How to use: work through the sections in order — most of them can be
drafted as soon as their corresponding lifecycle step is done, so this
doesn't have to be written all at once at the end. Every blockquote line
starting "Fill in:" is a prompt for what goes there, not final text — delete
the prompt once the section is written. Table skeletons show the exact
columns/sort order expected; keep them even if a row set ends up sparse.
-->

# [Project Name] — Model Development Report

**Author:** [name] · **Date:** [date] · **Repo:** [link] · **Status:** [draft / final]

> Fill in: one paragraph — what this model does, for whom, and the one-line
> verdict (ship it / needs more work / rejected), so a reader who stops here
> still gets the point.

---

## 1. Problem Formulation

> Fill in: the business problem, why it matters now, and what decision this
> model is meant to drive.

- **Target variable:** [name + definition — e.g. `default = 1` if a loan
  charges off within 12 months of origination]
- **Success criteria:** [the metric(s) that decide whether this model is
  good enough, and the threshold for "good enough"]
- **Constraints:** [latency, interpretability, regulatory, fairness, or data
  availability constraints that shaped the approach]
- **Timeline & cloud decision:** see `PROJECT.md` for the project-level
  timeline and the up-front cloud (none / GCP / AWS / Azure / other) call.

---

## 2. Data Gathering & Structuring

> Fill in: data sources, time window, grain (one row = ?), and how sources
> were joined/structured. Note any known gaps or exclusions and why.

| Source | Rows | Date range | Notes |
|---|---|---|---|
| [source] | [n] | [range] | [notes] |

### 2.1 Sampling strategy

Default method: **Out-of-Time (OOT)**.

| | Value |
|---|---|
| Seed | [seed] |
| Train window | [start – end] |
| Eval window | [start – end] |
| Test window | [start – end, most recent] |

> Fill in: why these window boundaries (e.g. a known regime change,
> seasonality, or simply "most recent N months held out"). Eval and test
> windows should not be explored beyond confirming they're structurally
> sane — see Section 3.

---

## 3. Data Exploration

This is the "Detailed Analysis" portion of the report — the point isn't
length, it's demonstrating you actually understand the data and the model,
not just that you ran the steps.

### 3.1 Exploration relative to the target

> Fill in: how key features actually relate to the target — not a generic
> univariate profile of every column. Show the 3-6 relationships that
> matter most (a chart or crosstab per relationship, one line on why it
> matters for the target).

### 3.2 Anomaly detection

> Fill in: identify and explain 5-10 anomalous records. "Explain" means
> *why* each is anomalous and what it implies (data quality issue? a
> legitimate but rare case? something to exclude before training?) — not
> just flagging them.

| Record ID | What's anomalous | Likely cause | Action taken |
|---|---|---|---|
| [id] | [description] | [data error / rare-but-real / other] | [kept / excluded / corrected] |

---

## 4. Feature Engineering

> Fill in: the feature set used, notable derived features, and any leakage
> checks performed (features that wouldn't be available at prediction time
> get flagged and removed here, not caught later).

| Feature | Derivation | Rationale |
|---|---|---|
| [feature] | [how it's built] | [why it should help] |

---

## 5. Multi-Model Training

Minimum three models trained and compared. Default bake-off per house
convention is **Random Forest + XGBoost + CatBoost**; note if AutoGluon was
added as a quick baseline or advanced ensembling pass, and why.

| Model | Library/version | Key hyperparameters | Training time |
|---|---|---|---|
| Random Forest | | | |
| XGBoost | | | |
| CatBoost | | | |
| [AutoGluon, if used] | | | |

### 5.1 Hyperparameter tuning

> Fill in: tuning method (grid / random / Bayesian / Optuna, etc.), search
> space, and — the part that's easy to skip — **how many trials were run,
> and whether that's actually enough to trust the result.** If tuning was
> skipped or minimal, say so plainly rather than implying it was thorough.

---

## 6. Model Evaluation

### 6.1 Performance table

| Model | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| Random Forest | | | | |
| XGBoost | | | | |
| CatBoost | | | | |

> F1, in one line: [the harmonic mean of precision and recall — it
> penalizes models that trade one off heavily for the other, so a high F1
> means the model is reasonably balanced between catching positives and not
> flooding results with false alarms.]

### 6.2 ROC chart

> Fill in / embed chart. Annotate the chosen score threshold on the curve
> and state plainly: at threshold = [x], this lands at FPR = [x], TPR = [x]
> on the test set.

### 6.3 Precision-Recall chart

> Fill in / embed chart. At the same threshold = [x]: precision = [x],
> recall = [x].

### 6.4 Operating table

Swept by **FPR from 0% to 5% in 0.25% increments** — this is the standard
operating-point artifact for this template, not a generic threshold table.

| FPR | Score threshold | TPR | Precision | Recall | F1 |
|---|---|---|---|---|---|
| 0.00% | | | | | |
| 0.25% | | | | | |
| 0.50% | | | | | |
| ... | | | | | |
| 5.00% | | | | | |

> **Recommended operating point:** [FPR / threshold] — [one paragraph on
> why this point, in terms of the business constraint from Section 1, not
> just "best F1"].

---

## 7. Model Interpretability

Best model: [name].

### 7.1 Global explanations

> Fill in / embed: variable importance (TreeSHAP), plus PDP/ALE/ICE plots
> for the top [n] variables, with one line per variable on what the shape
> of the relationship means in plain terms.

### 7.2 Local explanations

Three cohorts, top-N each, sorted exactly as below (do not reorder):

**True Positives** (actual positive, ordered by `pred_1` descending)

| Record ID | pred_1 | Actual | Top SHAP drivers |
|---|---|---|---|

**False Positives** (actual negative, high-scoring — ordered by `pred_1` descending)

| Record ID | pred_1 | Actual | Top SHAP drivers |
|---|---|---|---|

**False Negatives** (actual positive, low-scoring — ordered by `pred_1` ascending)

| Record ID | pred_1 | Actual | Top SHAP drivers |
|---|---|---|---|

### 7.3 Bias probe

> Fill in: look across the TP/FP/FN cohorts above for shared attributes —
> a demographic, a segment, a data-collection artifact. State plainly
> whether anything concerning turned up, even if the answer is "nothing
> notable found" — don't skip this because it's clean.

---

## 8. Model Calibration

### 8.1 Calibration curve

> Fill in / embed: reliability diagram (predicted probability vs. observed
> frequency). One line on whether the model is well-calibrated,
> over-confident, or under-confident, and where.

### 8.2 Score normalization

Rank-based spline mapping from raw model score to a normalized 1-1000
scale, banded by population percentile:

| Normalized band | Population share | Raw score range | Notes |
|---|---|---|---|
| 1000-900 | ~0.25% (rarest) | | |
| 900-800 | ~0.25% | | |
| 800-500 | ~4.5% | | |
| 500-0 | ~95% | | |

> These bands are a starting default, not fixed — adjust once the real
> score distribution is in hand and note here what changed and why.

---

## 8.5 Final Fit for Deployment

The champion model above, refit on the **full dataset** (train + eval +
test windows combined, same hyperparameters — no re-tuning here) for
production use.

> Fill in: confirm the refit was done, and report the calibration-drift
> check —
>
> 1. Re-scored the test-window records with the refit model.
> 2. Compared that score distribution to the original (pre-refit) test-
>    window scores: [close / shifted — describe].
> 3. Decision: [carried forward existing calibration + operating table /
>    recalibrated — if recalibrated, note the caveat that the check is
>    slightly optimistic since those records were also used to fit the
>    refit model, or note that a held-back slice was used instead].

---

## 9. Model Documentation

> Fill in: assumptions made, known limitations, what this model should
> *not* be used for, and reproducibility notes (data version / git commit /
> environment used to produce these results).

---

## 10. Model Integration

> Fill in: how this model reaches production — batch job, API, embedded in
> another pipeline — and, if a cloud provider was chosen up front, which
> services carry it (e.g. GCP Vertex, AWS SageMaker/Batch, Azure ML).

---

## 11. Model Monitoring

> Fill in: what gets watched in production (input drift, score
> distribution drift, performance decay against ground truth once it's
> available) and the alerting thresholds.

---

## 12. Periodic Retraining

> Fill in: retraining cadence or trigger conditions, and what changes each
> retrain (data window, features, hyperparameters re-tuned or held fixed).

---

*Generated from `report_template.md` in `env_cursor_ml`. See
`.cursor/rules/ml-lifecycle.mdc` for the conventions this template
encodes.*
