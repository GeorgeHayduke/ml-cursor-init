"""
Step 4 demo — Feature Engineering. Mirrors /ml-prep.

Leakage check: all fields are inputs known at origination time (or are the
synthetic target itself) — nothing here is post-outcome, so no columns are
excluded. In a real project this is where those flags would get resolved.

Derives a couple of ratio features, fits an encoder on TRAIN ONLY, and
produces two representations of the same feature *definitions*: one for
CatBoost (native categorical), one for RF/XGBoost (one-hot encoded).
"""
import json
import pandas as pd

train = pd.read_csv("data/processed/train.csv", parse_dates=["origination_date"])
eval_ = pd.read_csv("data/processed/eval.csv", parse_dates=["origination_date"])
test = pd.read_csv("data/processed/test.csv", parse_dates=["origination_date"])

BASE_FEATURES = [
    "credit_utilization", "income", "debt_to_income", "months_on_book",
    "num_delinquencies_24m", "credit_score_proxy", "loan_to_value",
    "num_inquiries_6m", "payment_history_score", "revolving_balance",
    "employment_stability", "feat_misc_1",
]
CATEGORICAL = ["region"]


def add_derived(df):
    df = df.copy()
    # months_on_book has a couple of garbage -1 values from Step 3's
    # anomaly set — clip rather than silently propagate.
    mob = df["months_on_book"].clip(lower=0)
    df["delinquency_rate"] = df["num_delinquencies_24m"] / (mob + 1)
    df["utilization_x_dti"] = df["credit_utilization"] * df["debt_to_income"]
    return df


DERIVED_FEATURES = ["delinquency_rate", "utilization_x_dti"]

train = add_derived(train)
eval_ = add_derived(eval_)
test = add_derived(test)

feature_cols = BASE_FEATURES + DERIVED_FEATURES + CATEGORICAL

# ---- CatBoost-ready representation: categoricals passed through raw ----
train[["record_id", "default"] + feature_cols].to_csv("data/processed/features_train.csv", index=False)
eval_[["record_id", "default"] + feature_cols].to_csv("data/processed/features_eval.csv", index=False)
test[["record_id", "default"] + feature_cols].to_csv("data/processed/features_test.csv", index=False)

# ---- RF/XGBoost-ready representation: one-hot region, fit categories on TRAIN only ----
train_categories = sorted(train["region"].unique().tolist())


def encode(df):
    df = df.copy()
    for cat in train_categories:
        df[f"region_{cat}"] = (df["region"] == cat).astype(int)
    return df.drop(columns=["region"])


train_enc = encode(train)
eval_enc = encode(eval_)
test_enc = encode(test)

encoded_feature_cols = BASE_FEATURES + DERIVED_FEATURES + [f"region_{c}" for c in train_categories]

train_enc[["record_id", "default"] + encoded_feature_cols].to_csv("data/processed/features_rf_xgb_train.csv", index=False)
eval_enc[["record_id", "default"] + encoded_feature_cols].to_csv("data/processed/features_rf_xgb_eval.csv", index=False)
test_enc[["record_id", "default"] + encoded_feature_cols].to_csv("data/processed/features_rf_xgb_test.csv", index=False)

features_config = {
    "leakage_review": "no columns excluded — all fields knowable at origination time",
    "base_features": BASE_FEATURES,
    "derived_features": {
        "delinquency_rate": "num_delinquencies_24m / (months_on_book_clipped + 1) — recent delinquency intensity",
        "utilization_x_dti": "credit_utilization * debt_to_income — interaction, both individually predictive",
    },
    "categorical_features": {
        "region": {
            "catboost": "passed through natively, flagged as categorical",
            "rf_xgboost": f"one-hot encoded, categories fit on train only: {train_categories}",
        }
    },
    "preprocessing_fit_on": "train window only",
}
with open("configs/features.yaml", "w") as f:
    f.write("# Feature definitions — Step 4 (Feature Engineering)\n")
    for k, v in features_config.items():
        f.write(f"{k}: {json.dumps(v)}\n")

print("configs/features.yaml written")
print(json.dumps(features_config, indent=2))
print(f"\nCatBoost feature set ({len(feature_cols)} cols): {feature_cols}")
print(f"RF/XGBoost feature set ({len(encoded_feature_cols)} cols): {encoded_feature_cols}")
