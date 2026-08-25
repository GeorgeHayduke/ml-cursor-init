"""Assertions shared by per-step lifecycle tests. Copied into each project."""
from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

import pandas as pd


def require(path: Path):
    if not path.exists():
        import pytest
        pytest.skip(f"not produced yet: {path}")
    return path


def parse_window(raw: str) -> tuple[date, date]:
    parts = [p.strip().strip('"').strip("'") for p in raw.split(",")]
    return date.fromisoformat(parts[0]), date.fromisoformat(parts[1])


def load_sampling(path: Path) -> dict:
    text = path.read_text()
    method = re.search(r'^method:\s*"?([^"\n]+)"?', text, re.M)
    seed = re.search(r"^seed:\s*(\d+)", text, re.M)
    windows = {}
    for key in ("train_window", "eval_window", "test_window"):
        m = re.search(rf"^{key}:\s*\[([^\]]+)\]", text, re.M)
        if not m:
            raise AssertionError(f"missing {key} in {path}")
        windows[key] = parse_window(m.group(1))
    return {
        "method": method.group(1) if method else None,
        "seed": int(seed.group(1)) if seed else None,
        **windows,
    }


def assert_oot_windows(sampling: dict):
    train_s, train_e = sampling["train_window"]
    eval_s, eval_e = sampling["eval_window"]
    test_s, test_e = sampling["test_window"]
    assert train_s < train_e
    assert eval_s < eval_e
    assert test_s < test_e
    assert train_e <= eval_s, "eval must start at or after train ends (no overlap)"
    assert eval_e <= test_s, "test must start at or after eval ends (no overlap)"
    assert test_e >= test_s


def assert_operating_table(df: pd.DataFrame):
    assert list(df.columns)[:6] == [
        "fpr_target_pct",
        "threshold",
        "tpr",
        "precision",
        "recall",
        "f1",
    ]
    assert len(df) == 21
    fprs = df["fpr_target_pct"].astype(float).tolist()
    assert fprs[0] == 0.0
    assert fprs[-1] == 5.0
    steps = [round(b - a, 2) for a, b in zip(fprs, fprs[1:])]
    assert set(steps) == {0.25}


def assert_cohort_sort(df: pd.DataFrame, kind: str):
    scores = df["pred_1"].astype(float).tolist()
    if kind in ("tp", "fp"):
        assert scores == sorted(scores, reverse=True), f"{kind} must be pred_1 descending"
        if kind == "tp":
            assert set(df["actual"].astype(int)) <= {1}
        else:
            assert set(df["actual"].astype(int)) <= {0}
    elif kind == "fn":
        assert scores == sorted(scores), "fn must be pred_1 ascending"
        assert set(df["actual"].astype(int)) <= {1}
    else:
        raise ValueError(kind)


def load_json(path: Path):
    return json.loads(require(path).read_text())
