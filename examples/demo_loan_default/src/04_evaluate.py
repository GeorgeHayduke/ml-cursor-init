"""
Step 6 demo — Model Evaluation. Mirrors /ml-evaluate.

First script that touches the test window. Scores it once per model,
selects the champion off test (not eval) performance, builds ROC/PR charts
and the FPR-swept operating table, and recommends an operating point
against a fabricated business constraint (stand-in for what /ml-define
would have pinned down for a real project).
"""
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier, Pool

test_cb = pd.read_csv("data/processed/features_test.csv")
test_enc = pd.read_csv("data/processed/features_rf_xgb_test.csv")
cb_feature_cols = [c for c in test_cb.columns if c not in ("record_id", "default")]
enc_feature_cols = [c for c in test_enc.columns if c not in ("record_id", "default")]
y_test = test_cb["default"].values

with open("models/random_forest.pkl", "rb") as f:
    rf = pickle.load(f)
xgb = XGBClassifier()
xgb.load_model("models/xgboost.json")
cat = CatBoostClassifier()
cat.load_model("models/catboost.cbm")

scores = {
    "random_forest": rf.predict_proba(test_enc[enc_feature_cols])[:, 1],
    "xgboost": xgb.predict_proba(test_enc[enc_feature_cols])[:, 1],
    "catboost": cat.predict_proba(Pool(test_cb[cb_feature_cols], cat_features=["region"]))[:, 1],
}

# ---- Step 2: performance table (test, scored once) ----
rows = []
for name, s in scores.items():
    pred_50 = (s >= 0.5).astype(int)
    rows.append({
        "model": name,
        "roc_auc": round(roc_auc_score(y_test, s), 4),
        "precision_at_0.5": round(precision_score(y_test, pred_50), 4),
        "recall_at_0.5": round(recall_score(y_test, pred_50), 4),
        "f1_at_0.5": round(f1_score(y_test, pred_50), 4),
    })
perf_table = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
perf_table.to_csv("reports/figures/step6_performance_table.csv", index=False)
print("=== Step 6: Model Evaluation (test window, scored once) ===")
print(perf_table.to_string(index=False))

# ---- Step 3: select champion off TEST performance ----
champion_name = perf_table.iloc[0]["model"]
champion_scores = scores[champion_name]
print(f"\nChampion (by test ROC-AUC): {champion_name}")
with open("models/champion.json", "w") as f:
    json.dump({"champion": champion_name, "test_roc_auc": float(perf_table.iloc[0]["roc_auc"])}, f, indent=2)

# ---- Step 4: ROC + PR charts annotated at threshold 0.5 ----
fpr, tpr, roc_thresh = roc_curve(y_test, champion_scores)
prec, rec, pr_thresh = precision_recall_curve(y_test, champion_scores)

idx_50 = np.argmin(np.abs(roc_thresh - 0.5))
fpr_50, tpr_50 = fpr[idx_50], tpr[idx_50]
pred_50 = (champion_scores >= 0.5).astype(int)
prec_50 = precision_score(y_test, pred_50)
rec_50 = recall_score(y_test, pred_50)

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, label=f"{champion_name} (AUC={perf_table.iloc[0]['roc_auc']:.3f})")
ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
ax.scatter([fpr_50], [tpr_50], color="red", zorder=5, label=f"threshold=0.5 (FPR={fpr_50:.3f}, TPR={tpr_50:.3f})")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title(f"ROC — {champion_name} (test window)")
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout()
fig.savefig("reports/figures/step6_roc.png", dpi=130)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(rec, prec, label=champion_name)
ax.scatter([rec_50], [prec_50], color="red", zorder=5, label=f"threshold=0.5 (P={prec_50:.3f}, R={rec_50:.3f})")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title(f"Precision-Recall — {champion_name} (test window)")
ax.legend(loc="lower left", fontsize=8)
fig.tight_layout()
fig.savefig("reports/figures/step6_pr.png", dpi=130)
plt.close(fig)

# ---- Step 5: Operating table — FPR 0% to 5% in 0.25% increments ----
op_rows = []
for target_fpr in np.arange(0, 5.0001, 0.25) / 100:
    idx = np.searchsorted(fpr, target_fpr, side="right") - 1
    idx = max(idx, 0)
    thresh = roc_thresh[idx] if idx < len(roc_thresh) else 1.0
    pred = (champion_scores >= thresh).astype(int)
    op_rows.append({
        "fpr_target_pct": round(target_fpr * 100, 2),
        "threshold": round(float(thresh), 4),
        "tpr": round(float(tpr[idx]), 4),
        "precision": round(precision_score(y_test, pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, pred, zero_division=0), 4),
    })
operating_table = pd.DataFrame(op_rows)
operating_table.to_csv("reports/figures/step6_operating_table.csv", index=False)
print(f"\nOperating table ({len(operating_table)} rows, FPR 0-5% by 0.25%) -> reports/figures/step6_operating_table.csv")

# ---- Step 6: recommend operating point against a business constraint ----
# Fabricated business constraint (stand-in for /ml-define's pinned-down
# cost asymmetry): a false positive (unneeded manual review) costs ~$50;
# a false negative (missed default) costs ~$2,000. Minimize expected cost
# across the swept operating table.
FP_COST, FN_COST = 50, 2000
n_test = len(y_test)
n_pos = int(y_test.sum())
n_neg = n_test - n_pos

costs = []
for _, row in operating_table.iterrows():
    n_fp = row["fpr_target_pct"] / 100 * n_neg
    n_fn = (1 - row["tpr"]) * n_pos
    costs.append(n_fp * FP_COST + n_fn * FN_COST)
operating_table["expected_cost"] = [round(c, 0) for c in costs]
best_row = operating_table.loc[operating_table["expected_cost"].idxmin()]
operating_table.to_csv("reports/figures/step6_operating_table.csv", index=False)

print(f"\nBusiness constraint: FP≈${FP_COST} (manual review), FN≈${FN_COST} (missed default)")
print(f"Recommended operating point: FPR={best_row['fpr_target_pct']}%, threshold={best_row['threshold']}, "
      f"TPR={best_row['tpr']}, precision={best_row['precision']}, recall={best_row['recall']}, "
      f"expected_cost=${best_row['expected_cost']:.0f}")

with open("reports/figures/step6_recommendation.json", "w") as f:
    json.dump({
        "champion": champion_name,
        "fp_cost": FP_COST, "fn_cost": FN_COST,
        "recommended_operating_point": best_row.to_dict(),
    }, f, indent=2, default=str)
