# ml-cursor-init

Cursor skills + standing rules for a 12-step ML model lifecycle. Open a new
project, type `/ml-init`, and the agent scaffolds the repo and walks the
rest of the steps as `/ml-define`, `/ml-data`, `/ml-prep`, `/ml-model`,
`/ml-evaluate`, `/ml-explain`, `/ml-calibrate`, `/ml-document`.

## Install skills (global — every Cursor project)

This is the default. Skills land in `~/.cursor/skills/` so `/ml-init` works
no matter which folder you have open.

```bash
git clone git@github.com:GeorgeHayduke/ml-cursor-init.git
cd ml-cursor-init
chmod +x install.sh
./install.sh
```

Then **reload Cursor** (`Developer: Reload Window`) so Agent rediscovers
skills.

Confirm they loaded: **Customize → Skills** in the sidebar. You should see
`ml-init`, `ml-define`, `ml-data`, and the rest.

### Use them

In Agent chat:

- **Slash:** type `/` and pick `ml-init` (or type `/ml-init` and send)
- **Attach as context:** type `@` and select the skill (`@ml-init`)

The lifecycle rule (`ml-lifecycle.mdc`) is copied to `~/.cursor/rules/` so
the 12-step conventions stay in context even when you are not running a
skill. If a new chat does not pick it up, paste that file into
**Cursor Settings → Rules → User Rules**.

## Install into one project instead

Copy the pack into a repo if teammates should get the skills from git
without a global install:

```bash
mkdir -p .cursor/skills .cursor/rules templates
cp -R path/to/ml-cursor-init/.cursor/skills/. .cursor/skills/
cp path/to/ml-cursor-init/.cursor/rules/ml-lifecycle.mdc .cursor/rules/
cp -R path/to/ml-cursor-init/templates/. templates/
```

Or let `/ml-init` do that copy when it scaffolds a new ML repo.

## What you get

| Kind | Path | Role |
|---|---|---|
| Skills | `.cursor/skills/ml-*/SKILL.md` | Step-by-step workflows, invoked with `/` or `@` |
| Rule | `.cursor/rules/ml-lifecycle.mdc` | Always-on conventions (OOT split, RF+XGB+CatBoost, TreeSHAP, operating table) |
| Templates | `templates/` | Report markdown, HTML report, `document_model.py` |

`/ml-integrate`, `/ml-monitor`, and `/ml-retrain` are not in this pack yet
(steps 10–12).

Worked example: `examples/demo_loan_default/`.

## Cursor layout this kit follows

Cursor loads skills from, in order of how you usually install them:

| Location | Scope |
|---|---|
| `~/.cursor/skills/` | You, every workspace (**this kit's default**) |
| `.cursor/skills/` | This repo only, shared via git |

Each skill is a folder whose name matches the `name:` in `SKILL.md`. Do not
put skills in `~/.cursor/skills-cursor/` — that directory is Cursor's own.
