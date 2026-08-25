from pathlib import Path
import importlib.util

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def document_model():
    path = ROOT / "templates" / "document_model.py"
    spec = importlib.util.spec_from_file_location("document_model", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
