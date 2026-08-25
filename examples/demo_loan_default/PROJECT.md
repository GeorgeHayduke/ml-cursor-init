# demo_loan_default — Project Charter & Status

**Status:** Simulated worked example — synthetic data, built to exercise the
`env_cursor_ml` Cursor rules/commands end to end, not a real project.

## 1. Problem Formulation

**Problem statement:** Predict whether a consumer loan will default within
12 months of origination, to route high-risk applications to manual
underwriting review before disbursement.

**Target variable:** `default = 1` if the loan defaults within 12 months of
origination; `0` otherwise. (Synthetic — generated via
`sklearn.datasets.make_classification`, ~12% positive rate.)

**Success criteria:** ROC-AUC ≥ 0.85 on the test window, and an operating
point that keeps expected cost (below) below the pre-model baseline cost of
reviewing nothing (~$240,000 across the test window, at $2,000 per missed
default).

**Business constraint (pinned down here, NOT the operating point):** a
false positive (unnecessary manual review) costs ≈$50; a false negative
(missed default) costs ≈$2,000. The actual operating point is chosen in
Step 6 by sweeping the operating table against this constraint.

**Stakeholders:** Credit risk team (consumer), underwriting ops (review
capacity).

**Constraints:** Interpretability required (SHAP-based explanations for
every review-routed application); no real-time latency constraint (batch
scoring at origination).

**Baseline / current process:** None — exploratory. This is a simulated
example project, not a real deployment replacing an existing process.

**Kickoff date:** 2026-08-24 (simulated).

## Cloud & tracking decisions

- **Cloud:** none — fully local, per this demo.
- **Experiment tracking:** structured local logs (`experiments/`).

## Timeline (per-step, simulated — not literally executed on this cadence)

| Step | Status |
|---|---|
| 1. Problem Formulation | ✅ done |
| 2. Data Gathering & Structuring | ✅ done |
| 3. Data Exploration | ✅ done |
| 4. Feature Engineering | ✅ done |
| 5. Multi-Model Training | ✅ done |
| 6. Model Evaluation | ✅ done |
| 7. Model Interpretability | ✅ done |
| 8. Model Calibration | ✅ done |
| 8.5. Final Fit for Deployment | ✅ done |
| 9. Model Documentation | ✅ done (this report) |
| 10. Model Integration | ⬜ not built — `/ml-integrate` doesn't exist yet |
| 11. Model Monitoring | ⬜ not built — `/ml-monitor` doesn't exist yet |
| 12. Periodic Retraining | ⬜ not built — `/ml-retrain` doesn't exist yet |

## Leakage review (Step 4)

No columns excluded — all fields are knowable at origination time. See
`configs/features.yaml`.
