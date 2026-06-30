#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
WORKDIR="$ROOT_DIR/solvers/IncNavierStokesSolver/Tests"
OUTCSV="${1:-$ROOT_DIR/py_utils/tests/results/do_validation_matrix.csv}"

mkdir -p "$(dirname "$OUTCSV")"

cat > "$OUTCSV" <<CSV
case,S,NP,MR,OUVar,L2u,L2v,L2p,Linfu,Linfv,Linfp,InitMethodUsed,InitRank,OrthoErr,CovSymErr,CovCondProxy
CSV

shopt -s nullglob
logs=("$WORKDIR"/DO_tmp_*.log)
if [[ ${#logs[@]} -eq 0 ]]; then
  echo "No DO_tmp_*.log files found in $WORKDIR" >&2
  exit 1
fi

for LOG in "${logs[@]}"; do
  CASE="$(basename "${LOG%.log}")"
  FLD="$WORKDIR/${CASE}.fld"
  if [[ ! -f "$FLD" ]]; then
    continue
  fi

  if [[ "$CASE" =~ _S([0-9]+)_NP([0-9]+)_MR([0-9]+)_OU([-0-9eE.+]+)$ ]]; then
    S="${BASH_REMATCH[1]}"
    NP="${BASH_REMATCH[2]}"
    MR="${BASH_REMATCH[3]}"
    OUVAR="${BASH_REMATCH[4]}"
  else
    S=""
    NP=""
    MR=""
    OUVAR=""
  fi

  L2U=$(awk '/L 2 error \(variable u\)/{print $NF}' "$LOG" | tail -n1)
  L2V=$(awk '/L 2 error \(variable v\)/{print $NF}' "$LOG" | tail -n1)
  L2P=$(awk '/L 2 error \(variable p\)/{print $NF}' "$LOG" | tail -n1)
  LIU=$(awk '/L inf error \(variable u\)/{print $NF}' "$LOG" | tail -n1)
  LIV=$(awk '/L inf error \(variable v\)/{print $NF}' "$LOG" | tail -n1)
  LIP=$(awk '/L inf error \(variable p\)/{print $NF}' "$LOG" | tail -n1)

  INITM=$(awk -F'[<>]' '/DO_InitMethodUsed/{print $3}' "$FLD" | tail -n1)
  INITR=$(awk -F'[<>]' '/DO_InitRank/{print $3}' "$FLD" | tail -n1)
  ORTHO=$(awk -F'[<>]' '/DO_OrthoErr/{print $3}' "$FLD" | tail -n1)
  COVSYM=$(awk -F'[<>]' '/DO_CovSymErr/{print $3}' "$FLD" | tail -n1)
  COND=$(awk -F'[<>]' '/DO_CovCondProxy/{print $3}' "$FLD" | tail -n1)

  echo "$CASE,$S,$NP,$MR,$OUVAR,$L2U,$L2V,$L2P,$LIU,$LIV,$LIP,$INITM,$INITR,$ORTHO,$COVSYM,$COND" >> "$OUTCSV"
done

echo "Wrote $OUTCSV"
