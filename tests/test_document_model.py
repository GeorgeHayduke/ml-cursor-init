import numpy as np
import pytest


def test_parse_model_arg(document_model):
    assert document_model.parse_model_arg("cat:models/m.cbm:catboost") == (
        "cat",
        "models/m.cbm",
        "catboost",
    )
    with pytest.raises(ValueError, match="name:path:kind"):
        document_model.parse_model_arg("only-two:parts")
    with pytest.raises(ValueError, match="kind must be"):
        document_model.parse_model_arg("m:path:lightgbm")


def test_operating_table_fpr_sweep(document_model):
    rng = np.random.default_rng(42)
    y = np.array([0] * 80 + [1] * 20)
    scores = np.concatenate([rng.random(80) * 0.4, 0.5 + rng.random(20) * 0.5])
    perm = rng.permutation(len(y))
    y, scores = y[perm], scores[perm]
    from sklearn.metrics import roc_curve

    fpr, tpr, thresh = roc_curve(y, scores)
    rows = document_model.compute_operating_table(y, scores, fpr, tpr, thresh, fp_cost=50, fn_cost=2000)
    assert len(rows) == 21
    assert rows[0]["fpr_target_pct"] == 0.0
    assert rows[-1]["fpr_target_pct"] == 5.0
    assert all("threshold" in r and "tpr" in r and "f1" in r for r in rows)
    assert all("expected_cost" in r for r in rows)


def test_cohort_sort_order(document_model):
    y = np.array([1, 1, 0, 0, 1, 0])
    scores = np.array([0.9, 0.2, 0.8, 0.1, 0.05, 0.7])
    ids = np.array(["a", "b", "c", "d", "e", "f"])
    shap = np.zeros((6, 2))
    shap[:, 0] = scores
    cohorts, _, counts = document_model.compute_cohorts_and_themes(
        y, scores, ids, shap, ["f1", "f2"], threshold=0.5, top_n=10
    )
    tp_scores = [r["pred_1"] for r in cohorts["tp"]]
    fp_scores = [r["pred_1"] for r in cohorts["fp"]]
    fn_scores = [r["pred_1"] for r in cohorts["fn"]]
    assert tp_scores == sorted(tp_scores, reverse=True)
    assert fp_scores == sorted(fp_scores, reverse=True)
    assert fn_scores == sorted(fn_scores)
    assert counts == {"tp": 1, "fp": 2, "fn": 2}


def test_score_bands_cover_population(document_model):
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=400)
    scores = rng.random(400)
    ids = np.arange(400)
    _, _, bands, gap = document_model.compute_calibration_and_spline(y, scores, ids)
    assert [b["normalized_band"] for b in bands] == ["1000-900", "900-800", "800-500", "500-0"]
    shares = [float(b["actual_population_share"].rstrip("%")) for b in bands]
    assert pytest.approx(sum(shares), abs=0.2) == 100.0
    assert gap >= 0


def test_global_importance_ranks_by_mean_abs_shap(document_model):
    shap = np.array([[1.0, -4.0], [1.0, 4.0], [0.0, 0.0]])
    ranked = document_model.compute_global_importance(shap, ["weak", "strong"])
    assert ranked[0]["feature"] == "strong"
    assert ranked[1]["feature"] == "weak"
