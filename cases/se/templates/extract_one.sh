#!/bin/bash
# extract_one.sh -- one FieldConvert interpolation + one reduction.
#
#   extract_one.sh dns RUN CHKFILE   OUTDIR PHI KEEP
#   extract_one.sh do  RUN ARCHIVE   OUTDIR PHI KEEP
#
# RUN      run directory (holds geometry.xml)
# CHKFILE  output/chks/casefile_N.chk   (relative to RUN)
# ARCHIVE  output/do/casefile.do_NNNNNN.fld (relative to RUN; a directory
#          for MPI runs, a single file for serial runs)
# OUTDIR   where <index>.npz goes (index = N for chks, NNNNNN for archives)
# PHI      phi npz from make_phi.sh, or "none"
# KEEP     "grid" to keep the full plane of mean and modes (DO only), else "-"
#
# Environment: FIELDCONVERT = FieldConvert executable (or a wrapper), EXTRACT_LIB =
# path of extract_lib.py. The interpolated .dat is deleted after the
# reduction. Exit code 0 only if the .npz exists at the end.
set -u
MODE=$1; RUN=$2; FILE=$3; OUTDIR=$4; PHI=$5; KEEP=$6
: "${FIELDCONVERT:?set FIELDCONVERT}" "${EXTRACT_LIB:?set EXTRACT_LIB}"

base=$(basename "$FILE")
case "$MODE" in
    dns) idx=$(echo "$base" | sed -n 's/^casefile_\([0-9]*\)\.chk$/\1/p') ;;
    do)  idx=$(echo "$base" | sed -n 's/^casefile\.do_\([0-9]*\)\.fld$/\1/p') ;;
    *)   echo "extract_one.sh: MODE must be dns or do" >&2; exit 2 ;;
esac
[ -n "$idx" ] || { echo "extract_one.sh: cannot parse index from $base" >&2; exit 2; }

out=$OUTDIR/$idx.npz
[ -s "$out" ] && exit 0                      # already done (restartable)
dat=$OUTDIR/tmp_$idx.dat

cd "$RUN" || exit 2
box="box=901,121,1,-5,40,-1.5,1.5,0,0"
"$FIELDCONVERT" -f -m "interppoints:fromxml=geometry.xml:fromfld=$FILE:$box" "$dat" > "$dat.log" 2>&1
if [ ! -s "$dat" ]; then
    # one write per failure so parallel workers do not interleave lines
    msg=$(printf 'FAILED FieldConvert: %s\n%s' "$FILE" "$(tail -n 5 "$dat.log")")
    echo "$msg" >&2; rm -f "$dat" "$dat.log"; exit 1
fi
rm -f "$dat.log"

phiarg=(); [ "$PHI" != "none" ] && phiarg=(--phi "$PHI")
if [ "$MODE" = dns ]; then
    python3 "$EXTRACT_LIB" reduce-dns "$dat" "$out" "${phiarg[@]}"
else
    keeparg=(); [ "$KEEP" = grid ] && keeparg=(--keep-grid)
    python3 "$EXTRACT_LIB" reduce-do "$dat" "$out" --info "$FILE" "${phiarg[@]}" "${keeparg[@]}"
fi
rc=$?
rm -f "$dat"
if [ $rc -ne 0 ] || [ ! -s "$out" ]; then echo "FAILED reduce: $FILE" >&2; rm -f "$out"; exit 1; fi
exit 0
