"""
Step 5 demo — Multi-Model Training. Mirrors /ml-model.

Trains the default bake-off (RF + XGBoost + CatBoost) on train, tunes
against eval only (never touches test), logs every run as a structured
local record under experiments/, and persists model artifacts + a
comparison table. Does NOT declare a champion — that's /ml-evaluate's job.
"""
import json
import pickle
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier, Pool

SEED = 42
rng = np.random.default_rng(SEED)

train_cb = pd.read_csv("data/processed/features_train.csv")
eval_cb = pd.read_csv("data/processed/features_eval.csv")
train_enc = pd.read_csv("data/processed/features_rf_xgb_train.csv")
eval_enc = pd.read_csv("data/processed/features_rf_xgb_eval.csv")

cb_feature_cols = [c for c in train_cb.columns if c not in ("record_id", "default")]
enc_feature_cols = [c for c in train_enc.columns if c not in ("record_id", "default")]

y_train, y_eval = train_cb["default"].values, eval_cb["default"].values

run_log = []


def log_run(model_name, params, train_auc, eval_auc, trial_idx, n_trials, seconds):
    run_log.append({
        "model": model_name,
        "trial": trial_idx,
        "n_trials_planned": n_trials,
        "params": params,
        "train_auc": round(train_auc, 4),
        "eval_auc": round(eval_auc, 4),
        "seconds": round(seconds, 2),
        "sampling_config": "configs/sampling.yaml",
        "features_config": "configs/features.yaml",
    })


results = {}

# ---------------- Random Forest ----------------
rf_grid = [
    {"n_estimators": 200, "max_depth": 6, "min_samples_leaf": 20},
    {"n_estimators": 300, "max_depth": 8, "min_samples_leaf": 10},
    {"n_estimators": 400, "max_depth": 10, "min_samples_leaf": 5},
    {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 15},
    {"n_estimators": 500, "max_depth": 8, "min_samples_leaf": 8},
]
best_rf, best_rf_auc, best_rf_params = None, -1, None
for i, params in enumerate(rf_grid, 1):
    t0 = time.time()
    m = RandomForestClassifier(random_state=SEED, n_jobs=-1, class_weight="balanced", **params)
    m.fit(train_enc[enc_feature_cols], y_train)
    tr_auc = roc_auc_score(y_train, m.predict_proba(train_enc[enc_feature_cols])[:, 1])
    ev_auc = roc_auc_score(y_eval, m.predict_proba(eval_enc[enc_feature_cols])[:, 1])
    log_run("random_forest", params, tr_auc, ev_auc, i, len(rf_grid), time.time() - t0)
    if ev_auc > best_rf_auc:
        best_rf, best_rf_auc, best_rf_params = m, ev_auc, params
results["random_forest"] = {"model": best_rf, "eval_auc": best_rf_auc, "params": best_rf_params, "feature_cols": enc_feature_cols}

# ---------------- XGBoost ----------------
xgb_grid = [
    {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.1},
    {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05},
    {"n_estimators": 400, "max_depth": 4, "learning_rate": 0.03},
    {"n_estimators": 250, "max_depth": 5, "learning_rate": 0.05},
    {"n_estimators": 350, "max_depth": 3, "learning_rate": 0.07},
]
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
best_xgb, best_xgb_auc, best_xgb_params = None, -1, None
for i, params in enumerate(xgb_grid, 1):
    t0 = time.time()
    m = XGBClassifier(random_state=SEED, eval_metric="auc", scale_pos_weight=scale_pos_weight, **params)
    m.fit(train_enc[enc_feature_cols], y_train)
    tr_auc = roc_auc_score(y_train, m.predict_proba(train_enc[enc_feature_cols])[:, 1])
    ev_auc = roc_auc_score(y_eval, m.predict_proba(eval_enc[enc_feature_cols])[:, 1])
    log_run("xgboost", params, tr_auc, ev_auc, i, len(xgb_grid), time.time() - t0)
    if ev_auc > best_xgb_auc:
        best_xgb, best_xgb_auc, best_xgb_params = m, ev_auc, params
results["xgboost"] = {"model": best_xgb, "eval_auc": best_xgb_auc, "params": best_xgb_params, "feature_cols": enc_feature_cols}

# ---------------- CatBoost ----------------
cat_features = ["region"]
train_pool = Pool(train_cb[cb_feature_cols], y_train, cat_features=cat_features)
eval_pool_ds = Pool(eval_cb[cb_feature_cols], y_eval, cat_features=cat_features)

cat_grid = [
    {"depth": 4, "learning_rate": 0.1, "iterations": 200},
    {"depth": 6, "learning_rate": 0.05, "iterations": 300},
    {"depth": 5, "learning_rate": 0.03, "iterations": 400},
    {"depth": 6, "learning_rate": 0.08, "iterations": 250},
    {"depth": 4, "learning_rate": 0.05, "iterations": 350},
]
best_cat, best_cat_auc, best_cat_params = None, -1, None
for i, params in enumerate(cat_grid, 1):
    t0 = time.time()
    m = CatBoostClassifier(random_seed=SEED, auto_class_weights="Balanced", verbose=False, **params)
    m.fit(train_pool)
    tr_auc = roc_auc_score(y_train, m.predict_proba(train_cb[cb_feature_cols])[:, 1])
    ev_auc = roc_auc_score(y_eval, m.predict_proba(eval_cb[cb_feature_cols])[:, 1])
    log_run("catboost", params, tr_auc, ev_auc, i, len(cat_grid), time.time() - t0)
    if ev_auc > best_cat_auc:
        best_cat, best_cat_auc, best_cat_params = m, ev_auc, params
results["catboost"] = {"model": best_cat, "eval_auc": best_cat_auc, "params": best_cat_params, "feature_cols": cb_feature_cols}

# ---------------- Persist ----------------
with open("models/random_forest.pkl", "wb") as f:
    pickle.dump(results["random_forest"]["model"], f)
results["xgboost"]["model"].save_model("models/xgboost.json")
results["catboost"]["model"].save_model("models/catboost.cbm")

with open("experiments/run_log_step5.json", "w") as f:
    json.dump(run_log, f, indent=2)

comparison = pd.DataFrame([
    {"model": name, "eval_auc": r["eval_auc"], "best_params": r["params"], "n_trials": 5}
    for name, r in results.items()
])
comparison.to_csv("reports/figures/step5_model_comparison.csv", index=False)

print("=== Step 5: Multi-Model Training complete ===")
print(comparison.to_string(index=False))
print(f"\n{len(run_log)} total runs logged -> experiments/run_log_step5.json")
print("Model artifacts -> models/random_forest.pkl, models/xgboost.json, models/catboost.cbm")
print("\nNo champion declared yet — test window untouched, per /ml-model convention.")
