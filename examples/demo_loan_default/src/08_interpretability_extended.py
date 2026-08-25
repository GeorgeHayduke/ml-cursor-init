"""
Step 7 extension — ICE, ALE, and cohort "common themes" for the HTML/JS report.

05_explain.py already computes: global importance (mean |SHAP|), a plain PDP
sweep, and the three TP/FP/FN cohorts with per-record top SHAP drivers.

This script adds the three things asked for on top of that:
  1. ICE curves (individual conditional expectation) for the top variables,
     so the PDP average line can be shown against the individual curves it's
     averaging over — a PDP alone can hide heterogeneous/canceling effects.
  2. ALE curves (accumulated local effects) for the same variables — unlike
     PDP, ALE doesn't extrapolate into unrealistic (var, other-features)
     combinations when features are correlated, so it's the more defensible
     "how does this variable actually affect the prediction" plot when
     features aren't independent (which they never really are here).
  3. Cohort "common themes": the FULL TP/FP/FN cohort's top-3 SHAP drivers,
     tallied by how often each feature shows up and its average signed
     contribution — the same full-cohort discipline the bias probe uses
     (never just the top-N display rows), applied to "what typically drives
     this cohort" instead of "is one attribute over/under-represented".
"""
import json
import pickle
import numpy as np
import pandas as pd
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
record_ids = test_cb["record_id"].reset_index(drop=True)

if champion_name == "catboost":
    model = CatBoostClassifier()
    model.load_model("models/catboost.cbm")
    pool = Pool(test_cb[cb_feature_cols], cat_features=["region"])
    scores = model.predict_proba(pool)[:, 1]
    shap_full = model.get_feature_importance(pool, type="ShapValues")
    shap_values = shap_full[:, :-1]
    feature_cols = cb_feature_cols
    X_display = test_cb[feature_cols].reset_index(drop=True)

    def predict(X):
        return model.predict_proba(Pool(X, cat_features=["region"]))[:, 1]
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

    def predict(X):
        return model.predict_proba(X)[:, 1]

importance = pd.read_csv("reports/figures/step7_global_importance.csv")
top_vars = [v for v in importance["feature"].tolist() if not v.startswith("region")][:3]
print("Top 3 variables for ICE/ALE:", top_vars)

rng = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# 1. ICE + PDP — grid of 15 points per var, individual curves for a sample
#    of 40 records so the panel stays readable, PDP = the mean of ALL rows
#    (not just the sampled 40) at each grid point.
# ---------------------------------------------------------------------------
ice_out = {}
sample_idx = rng.choice(len(X_display), size=min(40, len(X_display)), replace=False)

for var in top_vars:
    grid = np.linspace(X_display[var].quantile(0.02), X_display[var].quantile(0.98), 15)
    ice_curves = []
    pdp_curve = []
    for val in grid:
        X_mod = X_display.copy()
        X_mod[var] = val
        preds_all = predict(X_mod)
        pdp_curve.append(float(preds_all.mean()))
        ice_curves.append(preds_all[sample_idx])
    ice_curves = np.array(ice_curves)  # (grid_len, n_sampled)
    ice_out[var] = {
        "grid": [round(float(g), 3) for g in grid],
        "pdp": [round(float(p), 4) for p in pdp_curve],
        # transpose to one array per sampled record, centered at its own
        # first grid point (the standard "centered ICE" view) so curves
        # starting from different baselines can be compared on shape alone
        "ice_centered": [
            [round(float(v - ice_curves[0, j]), 4) for v in ice_curves[:, j]]
            for j in range(ice_curves.shape[1])
        ],
        "ice_raw": [
            [round(float(v), 4) for v in ice_curves[:, j]]
            for j in range(ice_curves.shape[1])
        ],
    }
print("ICE done for:", list(ice_out.keys()))

# ---------------------------------------------------------------------------
# 2. ALE — 12 quantile bins per var. Standard first-order ALE:
#    within each bin, for every observation, replace the variable with the
#    bin's upper and lower edge, take predict(upper) - predict(lower),
#    average within the bin, then cumulative-sum across bins and center
#    the whole curve to have zero mean (weighted by bin occupancy).
# ---------------------------------------------------------------------------
ale_out = {}
K = 12
for var in top_vars:
    x = X_display[var].values
    edges = np.unique(np.quantile(x, np.linspace(0, 1, K + 1)))
    if len(edges) < 3:
        print(f"  skipping ALE for {var} — not enough distinct values for {K} bins")
        continue
    bin_idx = np.clip(np.digitize(x, edges[1:-1], right=True), 0, len(edges) - 2)

    bin_effect = []
    bin_count = []
    for b in range(len(edges) - 1):
        in_bin = bin_idx == b
        n_b = int(in_bin.sum())
        bin_count.append(n_b)
        if n_b == 0:
            bin_effect.append(0.0)
            continue
        X_lo = X_display[in_bin].copy()
        X_hi = X_display[in_bin].copy()
        X_lo[var] = edges[b]
        X_hi[var] = edges[b + 1]
        delta = predict(X_hi) - predict(X_lo)
        bin_effect.append(float(delta.mean()))

    accumulated = np.concatenate([[0.0], np.cumsum(bin_effect)])
    # value at each bin edge; center by the occupancy-weighted mean
    midpoint_vals = 0.5 * (accumulated[:-1] + accumulated[1:])
    weights = np.array(bin_count) / max(sum(bin_count), 1)
    centered_mean = float(np.sum(midpoint_vals * weights))
    ale_centered = accumulated - centered_mean

    ale_out[var] = {
        "edges": [round(float(e), 3) for e in edges],
        "ale": [round(float(a), 4) for a in ale_centered],
        "bin_counts": bin_count,
    }
print("ALE done for:", list(ale_out.keys()))

# ---------------------------------------------------------------------------
# 3. Cohort common themes — FULL cohort (not top-10 display), same masks
#    05_explain.py uses. Tally each record's top-3 |SHAP| drivers; report
#    per feature: how often it lands in the top 3 for that cohort, its
#    average signed SHAP value when it does, and the dominant direction.
# ---------------------------------------------------------------------------
shap_df = pd.DataFrame(shap_values, columns=feature_cols)
pred_50 = (scores >= 0.5).astype(int)
tp_mask = (y_test == 1) & (pred_50 == 1)
fp_mask = (y_test == 0) & (pred_50 == 1)
fn_mask = (y_test == 1) & (pred_50 == 0)

themes_out = {}
for mask, name in [(tp_mask, "TP"), (fp_mask, "FP"), (fn_mask, "FN")]:
    sub = shap_df[mask].reset_index(drop=True)
    n = len(sub)
    tally = {f: {"count": 0, "signed_sum": 0.0} for f in feature_cols}
    for i in range(n):
        row = sub.iloc[i]
        top3 = row.reindex(row.abs().sort_values(ascending=False).index).head(3)
        for feat, val in top3.items():
            tally[feat]["count"] += 1
            tally[feat]["signed_sum"] += val
    rows = []
    for feat, t in tally.items():
        if t["count"] == 0:
            continue
        rows.append({
            "feature": feat,
            "pct_in_top3": round(100 * t["count"] / n, 1),
            "avg_signed_shap": round(t["signed_sum"] / t["count"], 3),
            "direction": "raises risk" if t["signed_sum"] >= 0 else "lowers risk",
        })
    rows.sort(key=lambda r: -r["pct_in_top3"])
    themes_out[name] = {"cohort_n": int(n), "themes": rows[:6]}
    print(f"\n{name} cohort (n={n}) — top recurring drivers:")
    for r in rows[:6]:
        print(f"  {r['feature']:<22} in top-3 for {r['pct_in_top3']:>5.1f}% of cases, "
              f"avg SHAP {r['avg_signed_shap']:+.3f} ({r['direction']})")

# ---------------------------------------------------------------------------
out = {"ice": ice_out, "ale": ale_out, "cohort_themes": themes_out, "top_vars": top_vars}
with open("reports/figures/step7_extended.json", "w") as f:
    json.dump(out, f)
print("\nWrote reports/figures/step7_extended.json —", len(json.dumps(out)), "bytes")
