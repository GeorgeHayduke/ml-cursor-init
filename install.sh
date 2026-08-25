#!/usr/bin/env bash
# Install this kit's Cursor skills globally so /ml-init etc. work in every project.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
DEST_SKILLS="${HOME}/.cursor/skills"
DEST_RULES="${HOME}/.cursor/rules"

mkdir -p "$DEST_SKILLS" "$DEST_RULES"

for skill_dir in "$ROOT"/.cursor/skills/*/; do
  name="$(basename "$skill_dir")"
  dest="$DEST_SKILLS/$name"
  rm -rf "$dest"
  mkdir -p "$dest"
  cp -R "$skill_dir"/. "$dest/"
done

# Bundle templates next to the skills that need them (global install has no repo root).
mkdir -p "$DEST_SKILLS/ml-init/assets" \
         "$DEST_SKILLS/ml-document/assets" \
         "$DEST_SKILLS/ml-document/scripts" \
         "$DEST_SKILLS/ml-integrate/assets" \
         "$DEST_SKILLS/ml-monitor/assets" \
         "$DEST_SKILLS/ml-retrain/assets"

cp "$ROOT/.cursor/rules/ml-lifecycle.mdc" "$DEST_SKILLS/ml-init/assets/"
cp "$ROOT/templates/report_template.md" "$DEST_SKILLS/ml-init/assets/"
cp "$ROOT/templates/document_model.py" "$DEST_SKILLS/ml-init/assets/"
cp "$ROOT/templates/model_report_template.html" "$DEST_SKILLS/ml-init/assets/"
cp "$ROOT/templates/integration.yaml" "$DEST_SKILLS/ml-init/assets/"
cp "$ROOT/templates/monitoring.yaml" "$DEST_SKILLS/ml-init/assets/"
cp "$ROOT/templates/retrain.yaml" "$DEST_SKILLS/ml-init/assets/"
cp "$ROOT/templates/integration.yaml" "$DEST_SKILLS/ml-integrate/assets/"
cp "$ROOT/templates/monitoring.yaml" "$DEST_SKILLS/ml-monitor/assets/"
cp "$ROOT/templates/retrain.yaml" "$DEST_SKILLS/ml-retrain/assets/"

cp "$ROOT/templates/document_model.py" "$DEST_SKILLS/ml-document/scripts/"
cp "$ROOT/templates/model_report_template.html" "$DEST_SKILLS/ml-document/assets/"
cp "$ROOT/templates/report_template.md" "$DEST_SKILLS/ml-document/assets/"

cp "$ROOT/.cursor/rules/ml-lifecycle.mdc" "$DEST_RULES/ml-lifecycle.mdc"

echo "Installed skills:"
ls -1 "$DEST_SKILLS" | sed 's/^/  /'
echo
echo "Installed always-on rule:"
echo "  $DEST_RULES/ml-lifecycle.mdc"
echo
echo "Reload Cursor (Developer: Reload Window), then in Agent chat type /ml-init"
echo "or attach a skill with @ml-init"
