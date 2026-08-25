"""One test class per lifecycle step. Missing artifacts skip — run pytest after every skill."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest

from lifecycle_checks import (
    assert_cohort_sort,
    assert_oot_windows,
    assert_operating_table,
    load_sampling,
    require,
)


@pytest.fixture
def root() -> Path:
    env = os.environ.get("ML_PROJECT_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve().parent
    if (here.parent / "PROJECT.md").exists() and here.parent.name != "templates":
        return here.parent
    demo = here.parents[1] / "examples" / "demo_loan_default"
    if (demo / "PROJECT.md").exists():
        return demo
    pytest.skip("no ML project root (PROJECT.md)")


class TestStep00Init:
    def test_scaffold(self, root: Path):
        assert (root / "PROJECT.md").is_file()
        for folder in ("data", "src", "configs", "reports", "experiments", "models"):
            assert (root / folder).exists(), folder


class TestStep01Define:
    def test_charter_fields(self, root: Path):
        text = require(root / "PROJECT.md").read_text().lower()
        for needle in ("target variable", "success criteria", "business constraint"):
            assert needle in text, f"PROJECT.md missing {needle}"


class TestStep02Data:
    def test_oot_split_and_disjoint_ids(self, root: Path):
        sampling = load_sampling(require(root / "configs" / "sampling.yaml"))
        assert sampling["method"] in ("out_of_time", "oot")
        assert sampling["seed"] is not None
        assert_oot_windows(sampling)
        frames = []
        for split in ("train", "eval", "test"):
            path = require(root / "data" / "processed" / f"{split}.csv")
            df = pd.read_csv(path)
            assert "record_id" in df.columns
            frames.append(df)
        ids = [set(df["record_id"]) for df in frames]
        assert ids[0].isdisjoint(ids[1])
        assert ids[0].isdisjoint(ids[2])
        assert ids[1].isdisjoint(ids[2])
        for df in frames:
            if "origination_date" in df.columns:
                assert df["origination_date"].notna().all()


class TestStep04Prep:
    def test_features_aligned_and_train_only_cats(self, root: Path):
        require(root / "configs" / "features.yaml")
        train = pd.read_csv(require(root / "data" / "processed" / "features_train.csv"))
        eval_ = pd.read_csv(require(root / "data" / "processed" / "features_eval.csv"))
        test = pd.read_csv(require(root / "data" / "processed" / "features_test.csv"))
        assert list(train.columns) == list(eval_.columns) == list(test.columns)
        assert "default" in train.columns or "target" in train.columns
        if "region" in train.columns:
            train_cats = set(train["region"].dropna().unique())
            eval_extra = set(eval_["region"].dropna().unique()) - train_cats
            test_extra = set(test["region"].dropna().unique()) - train_cats
            assert not eval_extra, f"eval region values not seen in train: {eval_extra}"
            assert not test_extra, f"test region values not seen in train: {test_extra}"


class TestStep05Model:
    def test_bakeoff_three_models_eval_only(self, root: Path):
        log_path = require(root / "experiments" / "run_log_step5.json")
        log = json.loads(log_path.read_text())
        models = {row["model"] for row in log}
        assert {"random_forest", "xgboost", "catboost"} <= models
        assert all("eval_auc" in row for row in log)
        cmp_path = root / "reports" / "figures" / "step5_model_comparison.csv"
        if cmp_path.exists():
            cmp = pd.read_csv(cmp_path)
            assert set(cmp["model"]) >= {"random_forest", "xgboost", "catboost"}


class TestStep06Evaluate:
    def test_performance_and_operating_table(self, root: Path):
        perf = pd.read_csv(require(root / "reports" / "figures" / "step6_performance_table.csv"))
        assert "roc_auc" in perf.columns
        assert {"precision_at_0.5", "recall_at_0.5", "f1_at_0.5"} <= set(perf.columns) or (
            "precision" in perf.columns and "recall" in perf.columns and "f1" in perf.columns
        )
        op = pd.read_csv(require(root / "reports" / "figures" / "step6_operating_table.csv"))
        assert_operating_table(op)
        rec = json.loads(require(root / "reports" / "figures" / "step6_recommendation.json").read_text())
        assert rec.get("champion")
        champ = json.loads(require(root / "models" / "champion.json").read_text())
        assert champ.get("champion") == rec["champion"]


class TestStep07Explain:
    def test_cohort_sort_and_importance(self, root: Path):
        figures = root / "reports" / "figures"
        for kind in ("tp", "fp", "fn"):
            df = pd.read_csv(require(figures / f"step7_cohort_{kind}.csv"))
            assert_cohort_sort(df, kind)
        imp = pd.read_csv(require(figures / "step7_global_importance.csv"))
        assert "mean_abs_shap" in imp.columns or "importance" in imp.columns
        require(figures / "step7_bias_probe.csv")


class TestStep08Calibrate:
    def test_bands_and_final_fit_note(self, root: Path):
        bands = pd.read_csv(require(root / "reports" / "figures" / "step8_score_bands.csv"))
        assert list(bands["normalized_band"]) == ["1000-900", "900-800", "800-500", "500-0"]
        summary = json.loads(
            require(root / "reports" / "figures" / "step8_5_final_fit_summary.json").read_text()
        )
        assert "verdict" in summary or "champion" in summary


class TestStep09Document:
    def test_report_has_no_authoring_prompts(self, root: Path):
        reports = list((root / "reports").glob("*_report.md"))
        if not reports:
            pytest.skip("no reports/*_report.md yet")
        text = reports[0].read_text()
        assert "> Fill in:" not in text


class TestStep10Integrate:
    def test_scoring_contract_config(self, root: Path):
        path = root / "configs" / "integration.yaml"
        require(path)
        text = path.read_text()
        for key in ("champion_artifact", "operating_point", "output_fields"):
            assert key in text
        assert "pred_1" in text and "score_norm" in text and "decision" in text
        assert "null" not in text.split("threshold:")[1].splitlines()[0], (
            "operating-point threshold still unset"
        )


class TestStep11Monitor:
    def test_monitoring_config_and_verdicts(self, root: Path):
        text = require(root / "configs" / "monitoring.yaml").read_text()
        assert "retrain_candidate" in text
        assert "feature_psi" in text
        mon_dir = root / "reports" / "monitoring"
        if not mon_dir.exists() or not any(mon_dir.iterdir()):
            pytest.skip("no reports/monitoring run yet")


class TestStep12Retrain:
    def test_retrain_config_has_promotion_gate(self, root: Path):
        text = require(root / "configs" / "retrain.yaml").read_text()
        assert "hold_hyperparameters" in text
        assert "require_beat_old_model_on_new_test" in text
        if "last_run: null" in text or "last_run: " not in text:
            pytest.skip("no retrain run recorded yet")
