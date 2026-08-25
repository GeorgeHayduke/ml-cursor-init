from pathlib import Path


def test_report_template_has_all_lifecycle_sections(repo_root: Path):
    text = (repo_root / "templates" / "report_template.md").read_text()
    for n, title in [
        (1, "Problem Formulation"),
        (2, "Data Gathering"),
        (6, "operating table"),
        (7, "Interpretability"),
        (8, "Calibration"),
        (8.5, "Final Fit"),
        (10, "Model Integration"),
        (11, "Model Monitoring"),
        (12, "Periodic Retraining"),
    ]:
        assert title.lower() in text.lower(), title


def test_config_templates_have_required_keys(repo_root: Path):
    integration = (repo_root / "templates" / "integration.yaml").read_text()
    for key in ("champion_artifact", "operating_point", "output_fields", "explain"):
        assert key in integration
    monitoring = (repo_root / "templates" / "monitoring.yaml").read_text()
    for key in ("feature_psi", "score_psi", "label_lag", "retrain_candidate"):
        assert key in monitoring
    retrain = (repo_root / "templates" / "retrain.yaml").read_text()
    for key in ("hold_hyperparameters", "rebake_off", "require_beat_old_model_on_new_test"):
        assert key in retrain


def test_demo_operating_table_is_21_fpr_rows(repo_root: Path):
    path = (
        repo_root
        / "examples"
        / "demo_loan_default"
        / "reports"
        / "figures"
        / "step6_operating_table.csv"
    )
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    assert len(lines) == 22  # header + 21 FPR steps
    fprs = [float(ln.split(",")[0]) for ln in lines[1:]]
    assert fprs[0] == 0.0
    assert fprs[-1] == 5.0
    steps = [round(b - a, 2) for a, b in zip(fprs, fprs[1:])]
    assert set(steps) == {0.25}
