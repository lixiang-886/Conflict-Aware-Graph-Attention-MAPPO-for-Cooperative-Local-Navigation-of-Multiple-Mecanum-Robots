#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

export MPLBACKEND=Agg
PYTHON_BIN="${PYTHON_BIN:-python3}"

generators=(
  generate_system_architecture.py
  generate_mecanum_geometry.py
  generate_conflict_coordination.py
  generate_main_efficiency.py
  generate_trajectory_clearance.py
  generate_ablation_summary.py
  generate_intervention_summary.py
)

for generator in "${generators[@]}"; do
  "${PYTHON_BIN}" "scripts/figures/${generator}"
done

mkdir -p paper/mdpi/figures_vector
for svg in paper/mdpi/figures_editable/*.svg; do
  name="$(basename "${svg}" .svg)"
  inkscape "${svg}" \
    --export-area-page \
    --export-type=pdf \
    --export-filename="paper/mdpi/figures_vector/${name}.pdf"
  inkscape "${svg}" \
    --export-area-page \
    --export-type=png \
    --export-dpi=600 \
    --export-background=white \
    --export-background-opacity=255 \
    --export-filename="paper/mdpi/figures_vector/${name}.png"
done

find paper/mdpi/figures_editable -name '*.svg' -print0 |
  xargs -0 -n1 xmllint --noout
