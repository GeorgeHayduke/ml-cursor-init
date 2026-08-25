"""
Step 8 + 8.5 demo — Calibration and Final Fit for Deployment.
Mirrors /ml-calibrate (which folds 8.5 in with 8).
"""
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier
from catboost import CatBoostClassifier, Pool
from sklearn.ensemble import RandomForestClassifier

with open("models/champion.json") as f:
    champion_info = json.load(f)
champion_name = champion_info["champion"]

with open("experiments/run_log_step5.json") as f:
    run_log = json.load(f)
best_run = max((r for r in run_log if r["model"] == champion_name), key=lambda r: r["eval_auc"])
best_params = best_run["params"]
print(f"Champion: {champion_name}, best params from Step 5: {best_params}")

test_cb = pd.read_csv("data/processed/features_test.csv")
test_enc = pd.read_csv("data/processed/features_rf_xgb_test.csv")
cb_feature_cols = [c for c in test_cb.columns if c not in ("record_id", "default")]
enc_feature_cols = [c for c in test_enc.columns if c not in ("record_id", "default")]
y_test = test_cb["default"].values

# ---- Step 8.1: pre-refit champion scores on test (reuse Step 6/7 logic) ----
if champion_name == "catboost":
    model = CatBoostClassifier()
    model.load_model("models/catboost.cbm")
    pre_scores = model.predict_proba(Pool(test_cb[cb_feature_cols], cat_features=["region"]))[:, 1]
elif champion_name == "xgboost":
    model = XGBClassifier()
    model.load_model("models/xgboost.json")
    pre_scores = model.predict_proba(test_enc[enc_feature_cols])[:, 1]
else:
    with open("models/random_forest.pkl", "rb") as f:
        model = pickle.load(f)
    pre_scores = model.predict_proba(test_enc[enc_feature_cols])[:, 1]

# ---- Step 8.1: calibration curve ----
frac_pos, mean_pred = calibration_curve(y_test, pre_scores, n_bins=10, strategy="quantile")
fig, ax = plt.subplots(figsize=(5.5, 5))
ax.plot(mean_pred, frac_pos, marker="o", label=champion_name)
ax.plot([0, 1], [0, 1], "--", color="gray", label="perfectly calibrated")
ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Observed frequency")
ax.set_title(f"Calibration curve — {champion_name} (pre-refit, test window)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("reports/figures/step8_calibration_curve.png", dpi=130)
plt.close(fig)

calib_gap = float(np.mean(np.abs(frac_pos - mean_pred)))
print(f"Mean |observed - predicted| across bins: {calib_gap:.4f} "
      f"({'reasonably well-calibrated' if calib_gap < 0.05 else 'meaningful miscalibration — consider recalibrating'})")

# ---- Step 8.2: score normalization — rank-based spline, 1-1000 scale ----
# Anchors: (population percentile from the top, normalized score)
anchors_pct = [0.0, 0.0025, 0.005, 0.05, 1.0]
anchors_val = [1000, 900, 800, 500, 0]
spline = PchipInterpolator(anchors_pct, anchors_val)

order = np.argsort(-pre_scores)  # rank 1 = highest score = highest risk
pct_rank = np.empty_like(pre_scores)
pct_rank[order] = (np.arange(len(pre_scores)) + 0.5) / len(pre_scores)
normalized = spline(pct_rank)

norm_df = pd.DataFrame({"record_id": test_cb["record_id"], "raw_score": pre_scores,
                         "population_percentile": pct_rank, "normalized_score": normalized})
norm_df.to_csv("reports/figures/step8_score_normalization.csv", index=False)

band_summary = []
band_edges = [(1000, 900, 0.0, 0.0025), (900, 800, 0.0025, 0.005), (800, 500, 0.005, 0.05), (500, 0, 0.05, 1.0)]
for hi, lo, p_lo, p_hi in band_edges:
    in_band = (pct_rank > p_lo) & (pct_rank <= p_hi) if p_lo > 0 else (pct_rank <= p_hi)
    raw_in_band = pre_scores[in_band]
    band_summary.append({
        "normalized_band": f"{hi}-{lo}",
        "target_population_share": f"{(p_hi - p_lo) * 100:.2f}%",
        "actual_population_share": f"{in_band.mean() * 100:.2f}%",
        "raw_score_range": f"[{raw_in_band.min():.3f}, {raw_in_band.max():.3f}]" if in_band.any() else "n/a",
    })
band_df = pd.DataFrame(band_summary)
band_df.to_csv("reports/figures/step8_score_bands.csv", index=False)
print("\nScore normalization bands (test window):")
print(band_df.to_string(index=False))

# ---- Step 8.5: final fit on FULL data (train + eval + test) ----
train_cb = pd.read_csv("data/processed/features_train.csv")
eval_cb = pd.read_csv("data/processed/features_eval.csv")
train_enc = pd.read_csv("data/processed/features_rf_xgb_train.csv")
eval_enc = pd.read_csv("data/processed/features_rf_xgb_eval.csv")

full_cb = pd.concat([train_cb, eval_cb, test_cb], ignore_index=True)
full_enc = pd.concat([train_enc, eval_enc, test_enc], ignore_index=True)
y_full = full_cb["default"].values

print(f"\nRefitting {champion_name} on full data (n={len(full_cb)} = train+eval+test), same hyperparameters, no re-tuning.")
if champion_name == "catboost":
    final_model = CatBoostClassifier(random_seed=42, auto_class_weights="Balanced", verbose=False, **best_params)
    final_model.fit(Pool(full_cb[cb_feature_cols], y_full, cat_features=["region"]))
    final_model.save_model("models/champion_final.cbm")
    post_scores = final_model.predict_proba(Pool(test_cb[cb_feature_cols], cat_features=["region"]))[:, 1]
elif champion_name == "xgboost":
    scale_pos_weight = (y_full == 0).sum() / (y_full == 1).sum()
    final_model = XGBClassifier(random_state=42, eval_metric="auc", scale_pos_weight=scale_pos_weight, **best_params)
    final_model.fit(full_enc[enc_feature_cols], y_full)
    final_model.save_model("models/champion_final.json")
    post_scores = final_model.predict_proba(test_enc[enc_feature_cols])[:, 1]
else:
    final_model = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced", **best_params)
    final_model.fit(full_enc[enc_feature_cols], y_full)
    with open("models/champion_final.pkl", "wb") as f:
        pickle.dump(final_model, f)
    post_scores = final_model.predict_proba(test_enc[enc_feature_cols])[:, 1]

# ---- Calibration-drift check: compare pre- vs post-refit scores on the SAME test records ----
mean_abs_diff = float(np.mean(np.abs(post_scores - pre_scores)))
corr = float(np.corrcoef(pre_scores, post_scores)[0, 1])
drift_verdict = "close — carry forward existing calibration/operating table" if mean_abs_diff < 0.03 else \
    "shifted — recalibrate (flag: optimistic, these records also trained the refit model)"

fig, ax = plt.subplots(figsize=(5.5, 5))
ax.scatter(pre_scores, post_scores, s=8, alpha=0.4)
ax.plot([0, 1], [0, 1], "--", color="gray")
ax.set_xlabel("Pre-refit score (train-only champion)")
ax.set_ylabel("Post-refit score (full-data champion)")
ax.set_title("Step 8.5 — calibration-drift check (test-window records)")
fig.tight_layout()
fig.savefig("reports/figures/step8_5_drift_check.png", dpi=130)
plt.close(fig)

print(f"\nDrift check — mean |post - pre| score diff: {mean_abs_diff:.4f}, correlation: {corr:.4f}")
print(f"Verdict: {drift_verdict}")

with open("reports/figures/step8_5_final_fit_summary.json", "w") as f:
    json.dump({
        "champion": champion_name, "refit_on_n_records": len(full_cb),
        "mean_abs_score_diff": round(mean_abs_diff, 4), "correlation": round(corr, 4),
        "verdict": drift_verdict, "calibration_gap_pre_refit": round(calib_gap, 4),
    }, f, indent=2)
