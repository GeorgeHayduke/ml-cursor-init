# /ml-explain — Model Interpretability (Step 7)

Run after `/ml-evaluate`, against the champion model it named. Everything
here is TreeSHAP-based per house convention — don't substitute LIME or
another explainability library.

## Step 1 — Global explanations

Compute SHAP values for the champion on the test window (the same set
`/ml-evaluate` scored — don't introduce a different sample here). Produce:

- Variable importance ranked by mean absolute SHAP value.
- PDP/ALE/ICE plots for the top 5-6 variables by that ranking, each with
  one line on what the shape of the relationship means in plain terms —
  not just "importance is high," but what direction and shape.

## Step 2 — Local explanations: three cohorts, exact sort order

Do not reorder or substitute a different sort — this is a fixed
convention, not a preference:

- **TP** — actual positive, ordered by `pred_1` **descending**.
- **FP** — actual negative but high-scoring, ordered by `pred_1`
  **descending**.
- **FN** — actual positive but low-scoring, ordered by `pred_1`
  **ascending**.

Top-N per cohort (10 unless the project specifies otherwise). For each
record: id, `pred_1`, actual label, and its top SHAP-driving features with
direction (which features pushed the score up, which pushed it down).

## Step 3 — Bias probe

Look across the three cohorts above for shared attributes — a segment, a
data-collection artifact, anything overrepresented in FP or FN relative to
the overall population. Compare each cohort's feature distributions
against the full test-window distribution, not just against each other.
State the result plainly either way: something concerning found, or
nothing notable — don't skip this because it came back clean.

## Step 4 — Write up and wrap

Write global and local sections into `report_template.md` Section 7. Check
off Step 7 in `PROJECT.md`. Point to `/ml-calibrate` (step 8, which also
covers step 8.5's final-fit) as next.
