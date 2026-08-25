from pathlib import Path
import re

import pytest

REQUIRED_SKILLS = [
    "ml-init",
    "ml-define",
    "ml-data",
    "ml-prep",
    "ml-model",
    "ml-evaluate",
    "ml-explain",
    "ml-calibrate",
    "ml-document",
    "ml-integrate",
    "ml-monitor",
    "ml-retrain",
]


def _frontmatter(text: str) -> dict:
    match = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    assert match, "SKILL.md must start with YAML frontmatter"
    fields = {}
    for line in match.group(1).splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        fields[key.strip()] = value.strip()
    return fields


def test_skill_folders_match_required_set(repo_root: Path):
    skills_dir = repo_root / ".cursor" / "skills"
    found = sorted(p.name for p in skills_dir.iterdir() if p.is_dir())
    assert found == sorted(REQUIRED_SKILLS)


@pytest.mark.parametrize("name", REQUIRED_SKILLS)
def test_skill_frontmatter(repo_root: Path, name: str):
    skill_md = repo_root / ".cursor" / "skills" / name / "SKILL.md"
    text = skill_md.read_text()
    meta = _frontmatter(text)
    assert meta["name"] == name
    assert meta["description"]
    assert len(meta["description"]) <= 1024
    assert meta.get("disable-model-invocation") == "true"
    assert f"# /{name}" in text
    assert len(text.splitlines()) < 500


def test_lifecycle_rule_always_applies(repo_root: Path):
    rule = (repo_root / ".cursor" / "rules" / "ml-lifecycle.mdc").read_text()
    assert "alwaysApply: true" in rule
    for heading in (
        "Step 6 definition-of-done",
        "Step 7 definition-of-done",
        "Step 8 definition-of-done",
        "Step 10 definition-of-done",
        "Step 11 definition-of-done",
        "Step 12 definition-of-done",
    ):
        assert heading in rule


def test_readme_lists_skills_in_process_order(repo_root: Path):
    readme = (repo_root / "README.md").read_text()
    fence = re.search(r"```\n(you type.*?)\n```", readme, re.DOTALL)
    assert fence, "README must contain the process-order code block"
    block = fence.group(1)
    found = re.findall(r"^/(ml-[a-z-]+)", block, re.MULTILINE)
    assert found == REQUIRED_SKILLS
