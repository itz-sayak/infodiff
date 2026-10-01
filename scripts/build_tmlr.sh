#!/usr/bin/env bash
# Build the TMLR version: share generated tables, figures and number macros with manuscript/.
set -e
cd "$(dirname "$0")/.."
cp manuscript/tables/*.tex manuscript_tmlr/tables/
cp manuscript/numbers.tex manuscript_tmlr/numbers.tex
cp manuscript/figures/*.pdf manuscript_tmlr/figures/
cd manuscript_tmlr && tectonic -X compile main.tex
