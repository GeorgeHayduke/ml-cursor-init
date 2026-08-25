import os
import stat
import subprocess
from pathlib import Path


def test_install_script_copies_skills_and_rule(repo_root: Path, tmp_path: Path):
    script = repo_root / "install.sh"
    assert script.stat().st_mode & stat.S_IXUSR
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    result = subprocess.run(
        ["bash", str(script)],
        cwd=repo_root,
        env={**os.environ, "HOME": str(fake_home)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    skills = fake_home / ".cursor" / "skills"
    for name in ("ml-init", "ml-integrate", "ml-monitor", "ml-retrain", "ml-document"):
        assert (skills / name / "SKILL.md").is_file()
    assert (skills / "ml-init" / "assets" / "report_template.md").is_file()
    assert (skills / "ml-init" / "assets" / "integration.yaml").is_file()
    assert (skills / "ml-integrate" / "assets" / "integration.yaml").is_file()
    assert (skills / "ml-document" / "scripts" / "document_model.py").is_file()
    assert (fake_home / ".cursor" / "rules" / "ml-lifecycle.mdc").is_file()
