"""
Materializes every number the HTML/JS report needs into one JSON file:
full ROC/PR curves, calibration bins, PDP curves, drift scatter (subsampled),
plus everything already computed in reports/figures/*.csv|json.
"""
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score
from sklearn.calibration import calibration_curve
from scipy.interpolate import PchipInterpolator
from xgboost import XGBClassifier
from catboost import CatBoostClassifier, Pool

test_cb = pd.read_csv("data/processed/features_test.csv")
test_enc = pd.read_csv("data/processed/features_rf_xgb_test.csv")
cb_feature_cols = [c for c in test_cb.columns if c not in ("record_id", "default")]
enc_feature_cols = [c for c in test_enc.columns if c not in ("record_id", "default")]
y_test = test_cb["default"].values

with open("models/random_forest.pkl", "rb") as f:
    rf = pickle.load(f)
xgb = XGBClassifier(); xgb.load_model("models/xgboost.json")
cat = CatBoostClassifier(); cat.load_model("models/catboost.cbm")
cat_pool = Pool(test_cb[cb_feature_cols], cat_features=["region"])

scores = {
    "random_forest": rf.predict_proba(test_enc[enc_feature_cols])[:, 1],
    "xgboost": xgb.predict_proba(test_enc[enc_feature_cols])[:, 1],
    "catboost": cat.predict_proba(cat_pool)[:, 1],
}

out = {}

# ---- ROC / PR curves for all three (for the model-comparison chart) ----
out["roc_curves"] = {}
out["pr_curves"] = {}
for name, s in scores.items():
    fpr, tpr, _ = roc_curve(y_test, s)
    idx = np.linspace(0, len(fpr) - 1, min(120, len(fpr))).astype(int)
    out["roc_curves"][name] = {"fpr": fpr[idx].round(4).tolist(), "tpr": tpr[idx].round(4).tolist(),
                                "auc": round(float(roc_auc_score(y_test, s)), 4)}
    prec, rec, _ = precision_recall_curve(y_test, s)
    idx2 = np.linspace(0, len(prec) - 1, min(120, len(prec))).astype(int)
    out["pr_curves"][name] = {"precision": prec[idx2].round(4).tolist(), "recall": rec[idx2].round(4).tolist()}

# ---- Calibration curve (champion, pre-refit) ----
champion_scores = scores["catboost"]
frac_pos, mean_pred = calibration_curve(y_test, champion_scores, n_bins=10, strategy="quantile")
out["calibration"] = {"mean_pred": mean_pred.round(4).tolist(), "frac_pos": frac_pos.round(4).tolist()}

# ---- Score normalization curve (smooth, 100 points) ----
anchors_pct = [0.0, 0.0025, 0.005, 0.05, 1.0]
anchors_val = [1000, 900, 800, 500, 0]
spline = PchipInterpolator(anchors_pct, anchors_val)
grid = np.linspace(0, 1, 200)
out["score_spline"] = {"percentile": grid.round(5).tolist(), "normalized": spline(grid).round(1).tolist()}

# ---- PDP curves for top 4 numeric variables ----
importance = pd.read_csv("reports/figures/step7_global_importance.csv")
top_vars = [v for v in importance["feature"].tolist() if not v.startswith("region")][:4]
pdp = {}
for var in top_vars:
    grid_vals = np.linspace(test_cb[var].quantile(0.02), test_cb[var].quantile(0.98), 20)
    preds = []
    for val in grid_vals:
        X_mod = test_cb[cb_feature_cols].copy()
        X_mod[var] = val
        preds.append(float(cat.predict_proba(Pool(X_mod, cat_features=["region"]))[:, 1].mean()))
    pdp[var] = {"grid": grid_vals.round(3).tolist(), "pred": [round(p, 4) for p in preds]}
out["pdp"] = pdp

# ---- Drift scatter (subsampled to 300 points for a light payload) ----
with open("models/champion.json") as f:
    champ = json.load(f)
final = CatBoostClassifier(); final.load_model("models/champion_final.cbm")
post_scores = final.predict_proba(cat_pool)[:, 1]
rng = np.random.default_rng(42)
sub_idx = rng.choice(len(champion_scores), size=min(300, len(champion_scores)), replace=False)
out["drift_scatter"] = {"pre": champion_scores[sub_idx].round(4).tolist(), "post": post_scores[sub_idx].round(4).tolist()}

# ---- Pull in the small pre-computed tables/summaries as-is ----
out["performance_table"] = pd.read_csv("reports/figures/step6_performance_table.csv").to_dict("records")
out["model_comparison"] = pd.read_csv("reports/figures/step5_model_comparison.csv").to_dict("records")
out["operating_table"] = pd.read_csv("reports/figures/step6_operating_table.csv").to_dict("records")
out["global_importance"] = importance.head(10).to_dict("records")
out["bias_probe"] = pd.read_csv("reports/figures/step7_bias_probe.csv").to_dict("records")
out["anomalies"] = pd.read_csv("reports/figures/step3_anomalies.csv").to_dict("records")
out["score_bands"] = pd.read_csv("reports/figures/step8_score_bands.csv").to_dict("records")
with open("reports/figures/step6_recommendation.json") as f:
    out["recommendation"] = json.load(f)
with open("reports/figures/step8_5_final_fit_summary.json") as f:
    out["final_fit"] = json.load(f)
with open("reports/figures/step7_bias_probe_summary.json") as f:
    out["bias_summary"] = json.load(f)

for cohort in ["tp", "fp", "fn"]:
    out[f"cohort_{cohort}"] = pd.read_csv(f"reports/figures/step7_cohort_{cohort}.csv").head(10).to_dict("records")

# ---- Step 7 extended: ICE, ALE, and full-cohort "common themes" ----
with open("reports/figures/step7_extended.json") as f:
    extended = json.load(f)
out["ice"] = extended["ice"]
out["ale"] = extended["ale"]
out["cohort_themes"] = extended["cohort_themes"]
out["interp_top_vars"] = extended["top_vars"]

with open("reports/report_data.json", "w") as f:
    json.dump(out, f)

print("Wrote reports/report_data.json —", len(json.dumps(out)), "bytes")
