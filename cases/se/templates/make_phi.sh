#!/bin/bash
# make_phi.sh -- build the reference direction phi of the measure
#     A(t) = <u(t), phi>,  phi = antisym(u_fp) / ||antisym(u_fp)||,
# from one checkpoint u_fp of the unforced asymmetric fixed point at the Re
# of interest (Re 100: ic.chk of any run; other Re: the last checkpoint of
# an unforced run started from ic.chk once it is steady).
#
#   make_phi.sh RUN CHKFILE OUT.npz
#
# RUN      a run directory holding geometry.xml (the mesh of CHKFILE)
# CHKFILE  the fixed-point checkpoint, relative to RUN (e.g. ic.chk)
# OUT.npz  where phi goes (e.g. $HOME/nektar/cases/se/templates/phi_Re100.npz)
#
# Environment: FIELDCONVERT = FieldConvert executable (or a wrapper), EXTRACT_LIB =
# path of extract_lib.py. On JED, first:
#   module load gcc openmpi openblas arpack-ng; source ~/venvs/postproc/bin/activate
#   export FIELDCONVERT=$HOME/nektar/build/utilities/FieldConvert/FieldConvert
#   export EXTRACT_LIB=$HOME/nektar/cases/se/templates/extract_lib.py
set -eu
RUN=${1:?run dir}; CHK=${2:?checkpoint}; OUT=${3:?output npz}
: "${FIELDCONVERT:?set FIELDCONVERT}" "${EXTRACT_LIB:?set EXTRACT_LIB}"
dat=$(mktemp --suffix=.dat)
cd "$RUN"
"$FIELDCONVERT" -f -m "interppoints:fromxml=geometry.xml:fromfld=$CHK:box=901,121,1,-5,40,-1.5,1.5,0,0" "$dat" > /dev/null 2>&1
[ -s "$dat" ] || { echo "FieldConvert failed on $CHK" >&2; rm -f "$dat"; exit 1; }
python3 "$EXTRACT_LIB" make-phi "$dat" "$OUT"
rm -f "$dat"
