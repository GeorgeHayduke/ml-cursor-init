"""
Step 2/3 demo — Data Gathering & Structuring + Data Exploration.
Mirrors what /ml-data would do: generate/land raw data, define the OOT
sampling strategy, write configs/sampling.yaml, split into train/eval/test,
and run train-window-only EDA + anomaly detection.

This demo uses sklearn.datasets.make_classification as a stand-in for a
real data warehouse pull, since the point is to exercise the lifecycle
end-to-end, not to source real loan data.
"""
import json
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

SEED = 42
rng = np.random.default_rng(SEED)

N = 6000
N_FEATURES = 12
N_INFORMATIVE = 8
N_REDUNDANT = 2

X, y = make_classification(
    n_samples=N,
    n_features=N_FEATURES,
    n_informative=N_INFORMATIVE,
    n_redundant=N_REDUNDANT,
    n_clusters_per_class=2,
    weights=[0.88, 0.12],  # ~12% positive (default) rate
    flip_y=0.02,
    class_sep=1.1,
    random_state=SEED,
)

feature_names = [
    "credit_utilization", "income", "debt_to_income", "months_on_book",
    "num_delinquencies_24m", "credit_score_proxy", "loan_to_value",
    "num_inquiries_6m", "payment_history_score", "revolving_balance",
    "employment_stability", "feat_misc_1",
]
assert len(feature_names) == N_FEATURES

df = pd.DataFrame(X, columns=feature_names)
df["default"] = y

# Synthetic categorical feature — a neutral operational segment, not a
# protected attribute, used later to exercise CatBoost-native-categorical
# handling and the bias probe.
df["region"] = rng.choice(["A", "B", "C", "D"], size=N, p=[0.35, 0.3, 0.2, 0.15])

# Synthetic origination_date so the sampling strategy has something to be
# Out-of-Time *about* — 24 months, Jan 2024 through Dec 2025.
start = pd.Timestamp("2024-01-01")
day_offsets = rng.integers(0, 730, size=N)
df["origination_date"] = start + pd.to_timedelta(day_offsets, unit="D")

df["record_id"] = [f"L{100000 + i}" for i in range(N)]

# A handful of deliberately anomalous records for Step 3's anomaly
# detection to find: extreme income with extreme utilization, and a few
# negative/garbage values that a real ingestion pipeline would produce.
anomaly_idx = rng.choice(N, size=8, replace=False)
df.loc[anomaly_idx[:3], "income"] = df["income"].max() + rng.uniform(5, 8, size=3)
df.loc[anomaly_idx[3:6], "credit_utilization"] = df["credit_utilization"].max() + rng.uniform(4, 6, size=3)
df.loc[anomaly_idx[6:8], "months_on_book"] = -1  # garbage value from a real-world join gap

df.to_csv("data/raw/loans_raw.csv", index=False)

# ---- Sampling strategy: Out-of-Time, 60/20/20 by origination_date ----
df_sorted = df.sort_values("origination_date").reset_index(drop=True)
n = len(df_sorted)
train_end = int(n * 0.60)
eval_end = int(n * 0.80)

train_df = df_sorted.iloc[:train_end]
eval_df = df_sorted.iloc[train_end:eval_end]
test_df = df_sorted.iloc[eval_end:]

train_df.to_csv("data/processed/train.csv", index=False)
eval_df.to_csv("data/processed/eval.csv", index=False)
test_df.to_csv("data/processed/test.csv", index=False)

sampling_config = {
    "method": "out_of_time",
    "seed": SEED,
    "train_window": [str(train_df["origination_date"].min().date()), str(train_df["origination_date"].max().date())],
    "eval_window": [str(eval_df["origination_date"].min().date()), str(eval_df["origination_date"].max().date())],
    "test_window": [str(test_df["origination_date"].min().date()), str(test_df["origination_date"].max().date())],
    "counts": {"train": len(train_df), "eval": len(eval_df), "test": len(test_df)},
    "target_rate": {
        "train": round(float(train_df["default"].mean()), 4),
        "eval": round(float(eval_df["default"].mean()), 4),
        "test": round(float(test_df["default"].mean()), 4),
    },
}
with open("configs/sampling.yaml", "w") as f:
    f.write("# Sampling Strategy — Step 2 (Data Gathering & Structuring)\n")
    f.write("# Default method per ml-lifecycle.mdc: Out-of-Time (OOT)\n")
    for k, v in sampling_config.items():
        f.write(f"{k}: {json.dumps(v)}\n")

print("Sampling config:")
print(json.dumps(sampling_config, indent=2))

# ---- Step 3: Data Exploration — TRAIN WINDOW ONLY ----
print("\n--- Step 3: EDA relative to target (train window only) ---")
target_corr = train_df[feature_names].corrwith(train_df["default"]).sort_values(key=abs, ascending=False)
print("Correlation with target (train only):")
print(target_corr.round(3))

# Anomaly detection: simple z-score flag on train window, explained
anomalies = []
for col in ["income", "credit_utilization", "months_on_book"]:
    z = (train_df[col] - train_df[col].mean()) / train_df[col].std()
    flagged = train_df.loc[z.abs() > 4, ["record_id", col]]
    for _, row in flagged.iterrows():
        anomalies.append({"record_id": row["record_id"], "field": col, "value": round(float(row[col]), 2)})

anomalies_df = pd.DataFrame(anomalies).drop_duplicates(subset="record_id")
anomalies_df.to_csv("reports/figures/step3_anomalies.csv", index=False)
print(f"\nFlagged {len(anomalies_df)} anomalous records (train window) -> reports/figures/step3_anomalies.csv")
print(anomalies_df.to_string(index=False))
