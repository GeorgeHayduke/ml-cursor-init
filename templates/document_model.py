#!/usr/bin/env python3
"""
document_model.py — turn an existing scikit-learn / XGBoost / CatBoost model
plus a labeled dataset into an interactive HTML model report, with no Cursor
or AI step required to run it.

Covers what a model that already exists (not one being built from scratch
through the full 12-step lifecycle) actually needs documented: Evaluation
(ROC/PR, operating table), Interpretability (global importance, PDP, ICE,
ALE, TP/FP/FN examples with reasons, cohort common themes, optional bias
probe), and Calibration (reliability diagram + rank-based score
normalization). All narrative sentences are generated mechanically from the
computed numbers — nothing here requires a human or an AI in the loop to
produce a complete, honest report; an AI assistant (e.g. running this from a
Cursor command) can still improve the prose afterward, but the tool doesn't
depend on it.

Usage (single model):
    python document_model.py \
        --model catboost:models/catboost.cbm:catboost \
        --data data/test_labeled.csv \
        --target defaulted --id-col loan_id \
        --categorical-cols region,segment \
        --fp-cost 50 --fn-cost 2000 \
        --project-name "Loan Default Risk Model" \
        --output-dir reports/

Usage (compare several models, e.g. a bake-off):
    python document_model.py \
        --model random_forest:models/rf.pkl:sklearn \
        --model xgboost:models/xgb.json:xgboost \
        --model catboost:models/cat.cbm:catboost \
        --data data/test_labeled.csv --target defaulted \
        --project-name "Bake-off"

Only needs: a model file per --model, and one dataframe (CSV or parquet)
containing the feature columns plus the target column (and optionally an id
column). No configs/*.yaml, no PROJECT.md, no upstream lifecycle scaffolding
required — those are used automatically if present (see --sampling-note),
but the tool works from just these two inputs.
"""
import argparse
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Fixed defaults — adjust here if a project needs different ones. These
# mirror ml-lifecycle.mdc's stated defaults; they're defaults, not laws.
# ---------------------------------------------------------------------------
OPERATING_FPR_MAX_PCT = 5.0
OPERATING_FPR_STEP_PCT = 0.25
SCORE_BAND_ANCHORS_PCT = [0.0, 0.0025, 0.005, 0.05, 1.0]
SCORE_BAND_ANCHORS_VAL = [1000, 900, 800, 500, 0]
SCORE_BAND_EDGES = [(1000, 900, 0.0, 0.0025), (900, 800, 0.0025, 0.005),
                     (800, 500, 0.005, 0.05), (500, 0, 0.05, 1.0)]
CALIBRATION_GAP_OK_THRESHOLD = 0.05
BIAS_PROBE_FLAG_THRESHOLD = 0.15  # percentage-point deviation from overall

# A small fixed categorical palette (from the dataviz reference palette),
# cycled if more models are compared than it has slots for.
MODEL_PALETTE = [
    ("#2a78d6", "#3987e5"),  # blue
    ("#eb6834", "#d95926"),  # orange
    ("#1baf7a", "#199e70"),  # aqua
    ("#eda100", "#c98500"),  # yellow
    ("#e87ba4", "#d55181"),  # magenta
]


# ---------------------------------------------------------------------------
# Model adapter — unifies sklearn / XGBoost / CatBoost load + score + SHAP
# ---------------------------------------------------------------------------
class ModelAdapter:
    def __init__(self, name, path, kind, cat_features=None):
        self.name = name
        self.kind = kind
        self.cat_features = cat_features or []
        self.model = self._load(path, kind)

    def _load(self, path, kind):
        if kind == "catboost":
            from catboost import CatBoostClassifier
            m = CatBoostClassifier()
            m.load_model(path)
            return m
        elif kind == "xgboost":
            from xgboost import XGBClassifier
            m = XGBClassifier()
            m.load_model(path)
            return m
        elif kind == "sklearn":
            try:
                with open(path, "rb") as f:
                    return pickle.load(f)
            except Exception:
                import joblib
                return joblib.load(path)
        else:
            raise ValueError(f"Unknown model kind '{kind}' for model '{self.name}' "
                              f"— use one of: sklearn, xgboost, catboost")

    def _pool(self, X):
        from catboost import Pool
        cat = [c for c in self.cat_features if c in X.columns]
        return Pool(X, cat_features=cat) if cat else Pool(X)

    def predict_proba(self, X):
        if self.kind == "catboost":
            return self.model.predict_proba(self._pool(X))[:, 1]
        try:
            return self.model.predict_proba(X)[:, 1]
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Model '{self.name}' ({self.kind}) couldn't score the given features — "
                f"{self.kind} models need fully numeric, pre-encoded input. If this "
                f"dataset has string/categorical columns, either encode them first "
                f"(matching however this model was trained) or, if this model natively "
                f"handles categoricals, pass --model {self.name}:{{path}}:catboost instead. "
                f"Original error: {e}"
            ) from e

    def shap_values(self, X):
        if self.kind == "catboost":
            shap_full = self.model.get_feature_importance(self._pool(X), type="ShapValues")
            return shap_full[:, :-1]
        import shap
        explainer = shap.TreeExplainer(self.model)
        sv = explainer.shap_values(X)
        return sv[1] if isinstance(sv, list) else sv


def parse_model_arg(spec):
    """'name:path:kind' -> (name, path, kind)"""
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError(f"--model expects 'name:path:kind', got: {spec!r}")
    name, path, kind = parts
    if kind not in ("sklearn", "xgboost", "catboost"):
        raise ValueError(f"--model kind must be sklearn/xgboost/catboost, got: {kind!r} in {spec!r}")
    return name, path, kind


def load_dataframe(path):
    path = Path(path)
    if path.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Core numeric computations
# ---------------------------------------------------------------------------
def compute_performance_and_curves(y, scores, threshold):
    from sklearn.metrics import (roc_auc_score, roc_curve, precision_recall_curve,
                                  precision_score, recall_score, f1_score)
    pred = (scores >= threshold).astype(int)
    perf = {
        "roc_auc": round(float(roc_auc_score(y, scores)), 4),
        f"precision_at_{threshold}": round(float(precision_score(y, pred, zero_division=0)), 4),
        f"recall_at_{threshold}": round(float(recall_score(y, pred, zero_division=0)), 4),
        f"f1_at_{threshold}": round(float(f1_score(y, pred, zero_division=0)), 4),
    }
    fpr, tpr, roc_thresh = roc_curve(y, scores)
    prec, rec, _ = precision_recall_curve(y, scores)
    idx = np.linspace(0, len(fpr) - 1, min(120, len(fpr))).astype(int)
    idx2 = np.linspace(0, len(prec) - 1, min(120, len(prec))).astype(int)
    roc_curve_out = {"fpr": fpr[idx].round(4).tolist(), "tpr": tpr[idx].round(4).tolist(),
                      "auc": perf["roc_auc"]}
    pr_curve_out = {"precision": prec[idx2].round(4).tolist(), "recall": rec[idx2].round(4).tolist()}
    return perf, roc_curve_out, pr_curve_out, fpr, tpr, roc_thresh


def compute_operating_table(y, scores, fpr, tpr, roc_thresh, fp_cost=None, fn_cost=None):
    from sklearn.metrics import precision_score, recall_score, f1_score
    n = len(y)
    n_pos = int(y.sum())
    n_neg = n - n_pos
    rows = []
    for target_fpr in np.arange(0, OPERATING_FPR_MAX_PCT + 1e-6, OPERATING_FPR_STEP_PCT) / 100:
        idx = max(np.searchsorted(fpr, target_fpr, side="right") - 1, 0)
        thresh = float(roc_thresh[idx]) if idx < len(roc_thresh) else 1.0
        pred = (scores >= thresh).astype(int)
        row = {
            "fpr_target_pct": round(target_fpr * 100, 2),
            "threshold": round(thresh, 4),
            "tpr": round(float(tpr[idx]), 4),
            "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        }
        if fp_cost is not None and fn_cost is not None:
            n_fp = row["fpr_target_pct"] / 100 * n_neg
            n_fn = (1 - row["tpr"]) * n_pos
            row["expected_cost"] = round(n_fp * fp_cost + n_fn * fn_cost, 0)
        rows.append(row)
    return rows


def compute_global_importance(shap_values, feature_cols):
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(-mean_abs)
    return [{"feature": feature_cols[i], "mean_abs_shap": round(float(mean_abs[i]), 4)} for i in order]


def compute_ice_pdp(adapter, X, top_vars, sample_n=40, grid_n=15, seed=42):
    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(len(X), size=min(sample_n, len(X)), replace=False)
    out = {}
    for var in top_vars:
        grid = np.linspace(X[var].quantile(0.02), X[var].quantile(0.98), grid_n)
        ice_curves, pdp_curve = [], []
        for val in grid:
            X_mod = X.copy()
            X_mod[var] = val
            preds = adapter.predict_proba(X_mod)
            pdp_curve.append(float(preds.mean()))
            ice_curves.append(preds[sample_idx])
        ice_arr = np.array(ice_curves)
        out[var] = {
            "grid": [round(float(g), 3) for g in grid],
            "pdp": [round(float(p), 4) for p in pdp_curve],
            "ice_raw": [[round(float(v), 4) for v in ice_arr[:, j]] for j in range(ice_arr.shape[1])],
        }
    return out


def compute_ale(adapter, X, top_vars, bins=12):
    out = {}
    for var in top_vars:
        x = X[var].values
        edges = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
        if len(edges) < 3:
            continue
        bin_idx = np.clip(np.digitize(x, edges[1:-1], right=True), 0, len(edges) - 2)
        bin_effect, bin_count = [], []
        for b in range(len(edges) - 1):
            in_bin = bin_idx == b
            n_b = int(in_bin.sum())
            bin_count.append(n_b)
            if n_b == 0:
                bin_effect.append(0.0)
                continue
            X_lo, X_hi = X[in_bin].copy(), X[in_bin].copy()
            X_lo[var] = edges[b]
            X_hi[var] = edges[b + 1]
            delta = adapter.predict_proba(X_hi) - adapter.predict_proba(X_lo)
            bin_effect.append(float(delta.mean()))
        accumulated = np.concatenate([[0.0], np.cumsum(bin_effect)])
        midpoints = 0.5 * (accumulated[:-1] + accumulated[1:])
        weights = np.array(bin_count) / max(sum(bin_count), 1)
        centered = accumulated - float(np.sum(midpoints * weights))
        out[var] = {"edges": [round(float(e), 3) for e in edges],
                     "ale": [round(float(a), 4) for a in centered],
                     "bin_counts": bin_count}
    return out


def compute_cohorts_and_themes(y, scores, ids, shap_values, feature_cols, threshold, top_n):
    shap_df = pd.DataFrame(shap_values, columns=feature_cols)
    pred = (scores >= threshold).astype(int)
    tp_mask = (y == 1) & (pred == 1)
    fp_mask = (y == 0) & (pred == 1)
    fn_mask = (y == 1) & (pred == 0)

    def top_drivers(row_idx, n=3):
        row = shap_df.iloc[row_idx][feature_cols]
        ranked = row.reindex(row.abs().sort_values(ascending=False).index).head(n)
        return "; ".join(f"{k}={v:+.3f}" for k, v in ranked.items())

    df = pd.DataFrame({"record_id": ids, "pred_1": scores, "actual": y})
    cohorts = {}
    for mask, name, ascending in [(tp_mask, "tp", False), (fp_mask, "fp", False), (fn_mask, "fn", True)]:
        sub = df[mask].sort_values("pred_1", ascending=ascending).head(top_n).copy()
        sub["top_shap_drivers"] = [top_drivers(i) for i in sub.index]
        cohorts[name] = sub.to_dict("records")

    themes = {}
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
            rows.append({"feature": feat, "pct_in_top3": round(100 * t["count"] / max(n, 1), 1),
                         "avg_signed_shap": round(t["signed_sum"] / t["count"], 3),
                         "direction": "raises risk" if t["signed_sum"] >= 0 else "lowers risk"})
        rows.sort(key=lambda r: -r["pct_in_top3"])
        themes[name] = {"cohort_n": int(n), "themes": rows[:6]}

    return cohorts, themes, {"tp": int(tp_mask.sum()), "fp": int(fp_mask.sum()), "fn": int(fn_mask.sum())}


def compute_bias_probe(df_raw, tp_mask, fp_mask, fn_mask, categorical_cols):
    out = {}
    for col in categorical_cols:
        if col not in df_raw.columns:
            continue
        overall = df_raw[col].value_counts(normalize=True).sort_index()
        by_cohort = {}
        for mask, name in [(tp_mask, "TP"), (fp_mask, "FP"), (fn_mask, "FN")]:
            sub = df_raw[mask]
            by_cohort[name] = sub[col].value_counts(normalize=True).reindex(overall.index).fillna(0)
        table = pd.DataFrame({"overall": overall, **by_cohort}).round(4)
        table.index.name = col
        rows = table.reset_index().to_dict("records")
        max_dev = float((table[["TP", "FP", "FN"]].sub(table["overall"], axis=0)).abs().max().max())
        out[col] = {"table": rows, "max_deviation": round(max_dev, 4)}
    return out


def compute_calibration_and_spline(y, scores, ids):
    from sklearn.calibration import calibration_curve
    frac_pos, mean_pred = calibration_curve(y, scores, n_bins=10, strategy="quantile")
    calib_gap = float(np.mean(np.abs(frac_pos - mean_pred)))

    from scipy.interpolate import PchipInterpolator
    spline = PchipInterpolator(SCORE_BAND_ANCHORS_PCT, SCORE_BAND_ANCHORS_VAL)
    grid = np.linspace(0, 1, 200)
    spline_out = {"percentile": grid.round(5).tolist(), "normalized": spline(grid).round(1).tolist()}

    order = np.argsort(-scores)
    pct_rank = np.empty_like(scores)
    pct_rank[order] = (np.arange(len(scores)) + 0.5) / len(scores)

    bands = []
    for hi, lo, p_lo, p_hi in SCORE_BAND_EDGES:
        in_band = (pct_rank > p_lo) & (pct_rank <= p_hi) if p_lo > 0 else (pct_rank <= p_hi)
        raw_in_band = scores[in_band]
        bands.append({
            "normalized_band": f"{hi}-{lo}",
            "target_population_share": f"{(p_hi - p_lo) * 100:.2f}%",
            "actual_population_share": f"{in_band.mean() * 100:.2f}%",
            "raw_score_range": f"[{raw_in_band.min():.3f}, {raw_in_band.max():.3f}]" if in_band.any() else "n/a",
        })
    return ({"mean_pred": mean_pred.round(4).tolist(), "frac_pos": frac_pos.round(4).tolist()},
            spline_out, bands, calib_gap)


# ---------------------------------------------------------------------------
# Narrative generation — every sentence below is computed from real numbers,
# not a fixed template string; each function degrades gracefully (returns a
# plain, honest sentence) when the underlying pattern isn't found.
# ---------------------------------------------------------------------------
def narrative_verdict(label, auc, op_row, n):
    base = f"{label} scores ROC-AUC {auc:.3f} on the provided data (n={n:,})."
    if op_row:
        base += (f" At the recommended operating point (FPR≈{op_row['fpr_target_pct']:.2f}%, "
                 f"threshold≈{op_row['threshold']:.3f}), it catches {op_row['tpr']:.1%} of positives "
                 f"at {op_row['precision']:.1%} precision.")
    return base


def narrative_cost_curve(operating_table, fp_cost, fn_cost):
    zero_fpr_row = min(operating_table, key=lambda r: r["fpr_target_pct"])
    best = min(operating_table, key=lambda r: r["expected_cost"])
    missed_pct = (1 - zero_fpr_row["tpr"]) * 100
    ratio = fn_cost / fp_cost if fp_cost else float("inf")
    return (f"Cost is high at low FPR because a low FPR forces a high threshold, and a high threshold "
            f"misses positives: at FPR={zero_fpr_row['fpr_target_pct']:.2f}% this model's TPR is only "
            f"{zero_fpr_row['tpr']:.1%}, so {missed_pct:.1f}% of actual positives in this data are false "
            f"negatives — and at ${fn_cost:,.0f} per missed positive vs ${fp_cost:,.0f} per false positive "
            f"({ratio:.0f}×), even a modest miss rate outweighs the near-zero false-positive cost. "
            f"Cost falls as the threshold drops and TPR climbs, until FP volume catches up around "
            f"FPR≈{best['fpr_target_pct']:.2f}%, which is where the minimum sits.")


def narrative_ice_pdp(ice_data):
    if not ice_data:
        return ""
    fracs = []
    for var, d in ice_data.items():
        pdp_trend = d["pdp"][-1] - d["pdp"][0]
        agree = sum(1 for line in d["ice_raw"] if (line[-1] - line[0] >= 0) == (pdp_trend >= 0))
        fracs.append((var, agree / max(len(d["ice_raw"]), 1)))
    worst_var, worst_frac = min(fracs, key=lambda x: x[1])
    if worst_frac < 0.7:
        return (f"PDP alone can hide effects that cancel out across records. That shows up here: for "
                f"{worst_var}, only {worst_frac:.0%} of the sampled individual ICE curves move in the same "
                f"direction as the PDP average, meaning the population-level trend is masking real "
                f"heterogeneity underneath — worth checking which records disagree before trusting this "
                f"variable's effect at face value.")
    floor = min(f for _, f in fracs)
    return (f"PDP alone can hide effects that cancel out across records — e.g. the variable raising risk "
            f"for some records while lowering it for others. Showing the individual ICE curves underneath "
            f"is how you'd catch that; here at least {floor:.0%} of sampled records move in the same "
            f"direction as the PDP average for every top variable, so no such heterogeneity shows up.")


def narrative_ale_pdp(pdp_data, ale_data):
    if not ale_data:
        return ""
    corrs = []
    for var, ad in ale_data.items():
        pd_ = pdp_data.get(var)
        if not pd_:
            continue
        p = np.array(pd_["pdp"]); p = p - p.mean()
        a = np.array(ad["ale"]); a = a - a.mean()
        pg, ag = np.array(pd_["grid"]), np.array(ad["edges"])
        p_on_a = np.interp(ag, pg, p)
        c = float(np.corrcoef(p_on_a, a)[0, 1]) if np.std(p_on_a) > 0 and np.std(a) > 0 else 1.0
        corrs.append((var, c))
    if not corrs:
        return ""
    worst_var, worst_c = min(corrs, key=lambda x: x[1])
    if worst_c < 0.7:
        return (f"PDP can extrapolate into unrealistic feature combinations when inputs are correlated. "
                f"That looks like it matters here — for {worst_var}, the ALE and PDP shapes only correlate "
                f"at {worst_c:.2f}, meaning they tell meaningfully different stories about this variable's "
                f"effect; trust the ALE panel over the PDP one for it.")
    return ("PDP can extrapolate into unrealistic feature combinations when inputs are correlated — ALE "
            "only ever averages over the local neighborhood a value actually appears in, so it's the more "
            "defensible read when features are correlated. Here the ALE and PDP shapes broadly agree for "
            "every top variable, so these features aren't correlated enough to matter for their apparent "
            "effects.")


def narrative_cohort_themes(themes):
    tp_themes = themes.get("TP", {}).get("themes", [])
    fn_themes = themes.get("FN", {}).get("themes", [])
    tp_feats = {t["feature"] for t in tp_themes[:3]}
    fn_feats = {t["feature"] for t in fn_themes[:3]}
    shared = tp_feats & fn_feats
    if shared and fn_themes:
        fn_avg = np.mean([abs(t["avg_signed_shap"]) for t in fn_themes if t["feature"] in shared])
        tp_avg = np.mean([abs(t["avg_signed_shap"]) for t in tp_themes if t["feature"] in shared])
        ratio = tp_avg / fn_avg if fn_avg > 0 else float("inf")
        shared_str = ", ".join(sorted(shared))
        return (f"False negatives share top drivers with true positives ({shared_str}) but at "
                f"{ratio:.1f}× smaller average magnitude — these are cases the model saw the right "
                f"signal in, just too weakly to cross the threshold, which is a more concrete \"why it "
                f"missed\" story than the row list alone gives you.")
    return ("False negatives and true positives don't share top drivers here, which suggests the model is "
            "missing these cases for a structurally different reason than a merely weak signal — worth a "
            "closer look at what's actually driving the false negatives.")


def narrative_bias_probe(bias_out, threshold=BIAS_PROBE_FLAG_THRESHOLD):
    notes = {}
    for col, info in bias_out.items():
        max_dev = info["max_deviation"]
        flagged = max_dev > threshold
        verdict = "flag for review" if flagged else "no notable skew found"
        notes[col] = (f"Largest deviation on '{col}': {max_dev*100:.1f} points — "
                       f"{'above' if flagged else 'below'} the {threshold*100:.0f}-point flag threshold. "
                       f"{verdict.capitalize()}.")
    return notes


def narrative_calibration(calib_gap, threshold=CALIBRATION_GAP_OK_THRESHOLD):
    verdict = ("reasonably well-calibrated" if calib_gap < threshold else
               "meaningful miscalibration — consider recalibrating, or lead with the normalized score "
               "below rather than the raw probability")
    return f"Mean |observed − predicted| across bins: {calib_gap:.3f} — {verdict}."


# ---------------------------------------------------------------------------
# Simple conditional-block template engine — no external dependency.
# <!--IF:KEY--> ... <!--ENDIF:KEY--> is kept iff bool(flags.get(KEY)) is True.
# {{TOKEN}} is replaced with str(tokens[TOKEN]).
# ---------------------------------------------------------------------------
def render_template(template_text, flags, tokens):
    def strip_blocks(text):
        pattern = re.compile(r"<!--IF:(\w+)-->(.*?)<!--ENDIF:\1-->", re.S)

        def repl(m):
            key, body = m.group(1), m.group(2)
            return body if flags.get(key) else ""

        prev = None
        while prev != text:
            prev = text
            text = pattern.sub(repl, text)
        return text

    text = strip_blocks(template_text)
    for key, val in tokens.items():
        text = text.replace("{{" + key + "}}", str(val))
    return text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True, metavar="NAME:PATH:KIND",
                     help="Repeatable. kind is sklearn|xgboost|catboost. First one given is the "
                          "primary/champion model unless --champion is set.")
    ap.add_argument("--champion", help="Model NAME to treat as champion (default: highest ROC-AUC)")
    ap.add_argument("--data", required=True, help="CSV or parquet with features + target (+ optional id col)")
    ap.add_argument("--target", required=True, help="Name of the target/label column")
    ap.add_argument("--id-col", default=None, help="Name of a record-id column (auto-generated if omitted)")
    ap.add_argument("--categorical-cols", default="", help="Comma-separated categorical column names — used "
                     "for CatBoost's native categorical handling and as bias-probe candidates")
    ap.add_argument("--fp-cost", type=float, default=None)
    ap.add_argument("--fn-cost", type=float, default=None)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--top-n-vars", type=int, default=3, help="How many top variables get PDP/ICE/ALE")
    ap.add_argument("--top-n-cohort", type=int, default=10, help="Rows shown per TP/FP/FN example table")
    ap.add_argument("--ice-sample", type=int, default=40)
    ap.add_argument("--project-name", default=None)
    ap.add_argument("--author", default="")
    ap.add_argument("--date", default="")
    ap.add_argument("--output-dir", default="reports")
    ap.add_argument("--template", default=None, help="Path to model_report_template.html "
                     "(default: alongside this script)")
    args = ap.parse_args()

    categorical_cols = [c.strip() for c in args.categorical_cols.split(",") if c.strip()]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {args.data} ...")
    df = load_dataframe(args.data)
    if args.target not in df.columns:
        sys.exit(f"--target '{args.target}' not found in {args.data}. Columns: {list(df.columns)}")
    y = df[args.target].values
    if args.id_col and args.id_col in df.columns:
        ids = df[args.id_col].astype(str).values
    else:
        ids = np.array([f"row_{i:06d}" for i in range(len(df))])
    exclude = {args.target, args.id_col} - {None}
    feature_cols = [c for c in df.columns if c not in exclude]
    X = df[feature_cols].reset_index(drop=True)
    n = len(df)
    print(f"  {n:,} rows, {len(feature_cols)} feature columns, target='{args.target}' "
          f"(positive rate {y.mean():.2%})")

    models = {}
    for spec in args.model:
        name, path, kind = parse_model_arg(spec)
        print(f"Loading model '{name}' ({kind}) from {path} ...")
        models[name] = ModelAdapter(name, path, kind, cat_features=categorical_cols)

    print("Scoring ...")
    scores = {name: adapter.predict_proba(X) for name, adapter in models.items()}

    performance_table = []
    roc_curves, pr_curves = {}, {}
    per_model_curves = {}
    for name, s in scores.items():
        perf, roc_c, pr_c, fpr, tpr, roc_thresh = compute_performance_and_curves(y, s, args.threshold)
        performance_table.append({"model": name, **perf})
        roc_curves[name] = roc_c
        pr_curves[name] = pr_c
        per_model_curves[name] = (fpr, tpr, roc_thresh)

    champion_name = args.champion or max(performance_table, key=lambda r: r["roc_auc"])["model"]
    print(f"Champion: {champion_name} (ROC-AUC {next(r['roc_auc'] for r in performance_table if r['model']==champion_name):.4f})")
    champ_scores = scores[champion_name]
    champ_adapter = models[champion_name]
    fpr, tpr, roc_thresh = per_model_curves[champion_name]

    print("Computing operating table ...")
    operating_table = compute_operating_table(y, champ_scores, fpr, tpr, roc_thresh,
                                               args.fp_cost, args.fn_cost)
    recommendation = None
    if args.fp_cost is not None and args.fn_cost is not None:
        recommendation = min(operating_table, key=lambda r: r["expected_cost"])

    print("Computing SHAP importance, PDP, ICE, ALE ...")
    shap_values = champ_adapter.shap_values(X)
    global_importance = compute_global_importance(shap_values, feature_cols)
    top_vars = [v["feature"] for v in global_importance if v["feature"] not in categorical_cols][:args.top_n_vars]
    ice_data = compute_ice_pdp(champ_adapter, X, top_vars, sample_n=args.ice_sample)
    ale_data = compute_ale(champ_adapter, X, top_vars)
    pdp_data = {v: {"grid": ice_data[v]["grid"], "pdp": ice_data[v]["pdp"]} for v in ice_data}

    print("Building TP/FP/FN cohorts and common themes ...")
    cohorts, themes, cohort_sizes = compute_cohorts_and_themes(
        y, champ_scores, ids, shap_values, feature_cols, args.threshold, args.top_n_cohort)

    bias_probe = {}
    if categorical_cols:
        print(f"Running bias probe on: {categorical_cols} ...")
        pred = (champ_scores >= args.threshold).astype(int)
        tp_mask = (y == 1) & (pred == 1)
        fp_mask = (y == 0) & (pred == 1)
        fn_mask = (y == 1) & (pred == 0)
        bias_probe = compute_bias_probe(df, tp_mask, fp_mask, fn_mask, categorical_cols)

    print("Computing calibration curve and score-normalization spline ...")
    calibration, score_spline, score_bands, calib_gap = compute_calibration_and_spline(y, champ_scores, ids)

    report_data = {
        "models": [{"key": k, "label": k} for k in models],
        "champion": champion_name,
        "performance_table": performance_table,
        "roc_curves": roc_curves,
        "pr_curves": pr_curves,
        "operating_table": operating_table,
        "global_importance": global_importance,
        "interp_top_vars": top_vars,
        "pdp": pdp_data,
        "ice": ice_data,
        "ale": ale_data,
        "cohort_tp": cohorts["tp"],
        "cohort_fp": cohorts["fp"],
        "cohort_fn": cohorts["fn"],
        "cohort_themes": themes,
        "cohort_sizes": cohort_sizes,
        "calibration": calibration,
        "score_spline": score_spline,
        "score_bands": score_bands,
    }
    if recommendation is not None:
        report_data["recommendation"] = recommendation
        report_data["fp_cost"] = args.fp_cost
        report_data["fn_cost"] = args.fn_cost
    if bias_probe:
        report_data["bias_probe"] = bias_probe

    narrative = {
        "verdict": narrative_verdict(champion_name, next(r["roc_auc"] for r in performance_table
                                                           if r["model"] == champion_name),
                                      recommendation, n),
        "cost_curve": narrative_cost_curve(operating_table, args.fp_cost, args.fn_cost)
                      if recommendation is not None else "",
        "ice_pdp": narrative_ice_pdp(ice_data),
        "ale_pdp": narrative_ale_pdp(pdp_data, ale_data),
        "cohort_themes": narrative_cohort_themes(themes),
        "calibration": narrative_calibration(calib_gap),
        "bias_probe": narrative_bias_probe(bias_probe) if bias_probe else {},
    }
    report_data["narrative"] = narrative

    project_name = args.project_name or f"{champion_name} model report"
    slug = re.sub(r"[^a-z0-9]+", "_", project_name.lower()).strip("_") or "model"
    data_path = out_dir / f"{slug}_report_data.json"
    with open(data_path, "w") as f:
        json.dump(report_data, f)
    print(f"Wrote {data_path} ({data_path.stat().st_size:,} bytes)")

    template_path = Path(args.template) if args.template else Path(__file__).with_name("model_report_template.html")
    if not template_path.exists():
        sys.exit(f"Template not found at {template_path} — pass --template explicitly.")
    template_text = template_path.read_text(encoding="utf-8")

    flags = {
        "MULTI_MODEL": len(models) > 1,
        "HAS_COST": recommendation is not None,
        "HAS_BIAS_PROBE": bool(bias_probe),
    }
    data_json = json.dumps(report_data).replace("</script>", "<\\/script>")
    tokens = {
        "PROJECT_NAME": project_name,
        "AUTHOR": args.author or "unspecified",
        "DATE": args.date or "unspecified",
        "N_ROWS": f"{n:,}",
        "TARGET_COL": args.target,
        "CHAMPION": champion_name,
        "VERDICT_TEXT": narrative["verdict"],
        "DATA_JSON": data_json,
    }
    html = render_template(template_text, flags, tokens)
    html_path = out_dir / f"{slug}_report.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"Wrote {html_path} ({html_path.stat().st_size:,} bytes)")
    print(f"\nDone. Open {html_path} in a browser to view the report.")


if __name__ == "__main__":
    main()
