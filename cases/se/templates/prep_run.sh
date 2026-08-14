#!/bin/bash
# prep_run.sh -- create the output directory tree a run's casefile expects.
# Filters do not create parent directories (FieldIO's create_directory is
# non-recursive; HistoryPoints/ModalEnergy streams fail silently), so every
# OutputFile parent must exist before launch.
#
# Usage: ./prep_run.sh <run_dir>    (run_dir must contain casefile.xml)
set -eu

run_dir="${1:?usage: $0 <run_dir>}"
casefile="$run_dir/casefile.xml"
[ -f "$casefile" ] || { echo "error: no casefile.xml in $run_dir" >&2; exit 1; }

grep -oP '<PARAM\s+NAME="OutputFile">\s*\K[^<]+' "$casefile" | while read -r f; do
    d=$(dirname "$f")
    [ "$d" = "." ] && continue
    mkdir -p "$run_dir/$d"
    echo "created: $run_dir/$d"
done
