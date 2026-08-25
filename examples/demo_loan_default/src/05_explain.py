"""
Step 7 demo — Model Interpretability. Mirrors /ml-explain.

Loads whichever model /ml-evaluate named champion, computes TreeSHAP global
importance + a simple PDP-style sweep for the top variables, builds the
three local-explanation cohorts in the exact required sort order, and runs
a bias probe across them using the 'region' field.
"""
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from catboost import CatBoostClassifier, Pool

with open("models/champion.json") as f:
    champion_info = json.load(f)
champion_name = champion_info["champion"]
print(f"Champion model: {champion_name}")

test_cb = pd.read_csv("data/processed/features_test.csv")
test_enc = pd.read_csv("data/processed/features_rf_xgb_test.csv")
cb_feature_cols = [c for c in test_cb.columns if c not in ("record_id", "default")]
enc_feature_cols = [c for c in test_enc.columns if c not in ("record_id", "default")]
y_test = test_cb["default"].values

if champion_name == "catboost":
    model = CatBoostClassifier()
    model.load_model("models/catboost.cbm")
    pool = Pool(test_cb[cb_feature_cols], cat_features=["region"])
    scores = model.predict_proba(pool)[:, 1]
    shap_full = model.get_feature_importance(pool, type="ShapValues")
    shap_values = shap_full[:, :-1]  # last column is the base/expected value
    feature_cols = cb_feature_cols
    X_display = test_cb[feature_cols].reset_index(drop=True)
else:
    import shap
    if champion_name == "xgboost":
        model = XGBClassifier()
        model.load_model("models/xgboost.json")
    else:
        with open("models/random_forest.pkl", "rb") as f:
            model = pickle.load(f)
    scores = model.predict_proba(test_enc[enc_feature_cols])[:, 1]
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(test_enc[enc_feature_cols])
    shap_values = sv[1] if isinstance(sv, list) else sv
    feature_cols = enc_feature_cols
    X_display = test_enc[feature_cols].reset_index(drop=True)

record_ids = test_cb["record_id"].reset_index(drop=True)

# ---- Global: variable importance by mean |SHAP| ----
mean_abs_shap = np.abs(shap_values).mean(axis=0)
importance = pd.DataFrame({"feature": feature_cols, "mean_abs_shap": mean_abs_shap}).sort_values(
    "mean_abs_shap", ascending=False
)
importance.to_csv("reports/figures/step7_global_importance.csv", index=False)

fig, ax = plt.subplots(figsize=(6, 5))
top10 = importance.head(10).iloc[::-1]
ax.barh(top10["feature"], top10["mean_abs_shap"])
ax.set_xlabel("mean |SHAP value|")
ax.set_title(f"Global variable importance — {champion_name}")
fig.tight_layout()
fig.savefig("reports/figures/step7_global_importance.png", dpi=130)
plt.close(fig)

top_vars = importance.head(5)["feature"].tolist()
print("Top 5 variables by mean |SHAP|:", top_vars)

# ---- PDP-style sweep for top numeric variables ----
numeric_top_vars = [v for v in top_vars if not v.startswith("region")][:4]
fig, axes = plt.subplots(1, len(numeric_top_vars), figsize=(4 * len(numeric_top_vars), 3.5))
if len(numeric_top_vars) == 1:
    axes = [axes]
for ax, var in zip(axes, numeric_top_vars):
    grid = np.linspace(X_display[var].quantile(0.02), X_display[var].quantile(0.98), 15)
    preds = []
    for val in grid:
        X_mod = X_display.copy()
        X_mod[var] = val
        if champion_name == "catboost":
            p = model.predict_proba(Pool(X_mod, cat_features=["region"]))[:, 1].mean()
        else:
            p = model.predict_proba(X_mod)[:, 1].mean()
        preds.append(p)
    ax.plot(grid, preds)
    ax.set_xlabel(var, fontsize=8)
    ax.set_ylabel("avg predicted P(default)", fontsize=8)
fig.suptitle(f"Partial dependence — top variables ({champion_name})")
fig.tight_layout()
fig.savefig("reports/figures/step7_pdp.png", dpi=130)
plt.close(fig)

# ---- Local explanations: TP / FP / FN, exact sort order ----
df = pd.DataFrame({"record_id": record_ids, "pred_1": scores, "actual": y_test})
pred_50 = (df["pred_1"] >= 0.5).astype(int)

shap_df = pd.DataFrame(shap_values, columns=feature_cols)
shap_df["record_id"] = record_ids


def top_drivers(row_idx, n=3):
    row = shap_df.iloc[row_idx][feature_cols]
    ranked = row.reindex(row.abs().sort_values(ascending=False).index).head(n)
    return "; ".join(f"{k}={v:+.3f}" for k, v in ranked.items())


tp_mask = (df["actual"] == 1) & (pred_50 == 1)
fp_mask = (df["actual"] == 0) & (pred_50 == 1)
fn_mask = (df["actual"] == 1) & (pred_50 == 0)

tp = df[tp_mask].sort_values("pred_1", ascending=False).head(10).copy()
fp = df[fp_mask].sort_values("pred_1", ascending=False).head(10).copy()
fn = df[fn_mask].sort_values("pred_1", ascending=True).head(10).copy()

for cohort_df, name in [(tp, "TP"), (fp, "FP"), (fn, "FN")]:
    cohort_df["top_shap_drivers"] = [top_drivers(i) for i in cohort_df.index]
    cohort_df.to_csv(f"reports/figures/step7_cohort_{name.lower()}.csv", index=False)

print(f"\nTP cohort (n={tp_mask.sum()} at threshold 0.5, top {len(tp)} shown):")
print(tp[["record_id", "pred_1", "actual"]].to_string(index=False))
print(f"\nFP cohort (n={fp_mask.sum()}, top {len(fp)} shown):")
print(fp[["record_id", "pred_1", "actual"]].to_string(index=False))
print(f"\nFN cohort (n={fn_mask.sum()}, top {len(fn)} shown):")
print(fn[["record_id", "pred_1", "actual"]].to_string(index=False))

# ---- Bias probe: region distribution across cohorts vs overall ----
# Uses the FULL TP/FP/FN cohorts (not just the top-10 shown above) — a
# top-10 sample is too small to draw a bias conclusion from; the full
# cohort (n=120/31/32 here) is what the probe should actually run against.
region_overall = test_cb["region"].value_counts(normalize=True).sort_index()
region_by_cohort = {}
cohort_sizes = {"TP": int(tp_mask.sum()), "FP": int(fp_mask.sum()), "FN": int(fn_mask.sum())}
for mask, name in [(tp_mask, "TP"), (fp_mask, "FP"), (fn_mask, "FN")]:
    sub = test_cb[mask.values]
    region_by_cohort[name] = sub["region"].value_counts(normalize=True).reindex(region_overall.index).fillna(0)

bias_table = pd.DataFrame({"overall": region_overall, **region_by_cohort}).round(3)
bias_table.to_csv("reports/figures/step7_bias_probe.csv")
print(f"\nBias probe — region share, overall vs. each FULL cohort (n={cohort_sizes}):")
print(bias_table.to_string())
max_dev = (bias_table[["TP", "FP", "FN"]].sub(bias_table["overall"], axis=0)).abs().max().max()
print(f"\nLargest single deviation from overall region distribution: {max_dev:.3f} "
      f"({'flag for review' if max_dev > 0.15 else 'no notable skew found'})")
with open("reports/figures/step7_bias_probe_summary.json", "w") as f:
    json.dump({"cohort_sizes": cohort_sizes, "max_deviation": round(float(max_dev), 3)}, f, indent=2)
