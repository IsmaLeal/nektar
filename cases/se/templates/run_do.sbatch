#!/bin/bash
#SBATCH -N 1
#SBATCH -n 18
#SBATCH --time 7-00:00:00
#SBATCH --output /scratch/lealaske/se_runs/logs/do_%A_%a.out
#
# run_do.sbatch -- one DO run per array task, seeded by POD from the DNS at
# its own (Re, sigma). Row SLURM_ARRAY_TASK_ID+1 of do_sweep.tsv
# (Re, sigma, S, Np, stride, seed, T_END, eps[, fseed]) fills casefile_do.xml;
# the run lives in
# /scratch/lealaske/se_runs/Re<Re>_sig<sigma>/do_Re<Re>_sig<sigma>_S<S>_Np<Np>_seed<seed>[_eps<eps>][_frc<fseed>]/
# (one directory per (Re, sigma) pair holds its DNS and DO runs; never
# reused: an existing run dir aborts the task). The snapshot set is
# ../dns_Re<Re>_sig<sigma>_seed<seed>/output/chks/, every <stride>-th chk
# (stride 1 = all, 2 = every 8 t.u., 5 = every 20 t.u.). <seed> names the DNS
# that seeds the POD; the optional 9th column <fseed> sets DOForcingSeed on
# its own (default: fseed = seed), for repeats that change only the noise.
#
# Launch from the templates dir, N = number of rows in do_sweep.tsv:
#   mkdir -p /scratch/lealaske/se_runs/logs
#   sbatch --array=0-$((N-1)) run_do.sbatch
set -eu

TPL=$HOME/nektar/cases/se/templates
ROOT=/scratch/lealaske/se_runs
SOLVER=$HOME/nektar/build/solvers/IncNavierStokesSolver/IncNavierStokesSolver

# --- row of the sweep table -------------------------------------------------
ROW=$((SLURM_ARRAY_TASK_ID + 1))
read -r RE SIGMA S NP STRIDE SEED T_END EPS FSEED < <(sed -n "${ROW}p" "$TPL/do_sweep.tsv")
[ -n "${EPS:-}" ] || { echo "error: row $ROW of do_sweep.tsv is missing or short" >&2; exit 1; }
FSEED=${FSEED:-$SEED}                          # forcing seed; the DNS seed unless the 9th column says otherwise

PAIR=$ROOT/Re${RE}_sig${SIGMA}                 # the (Re, sigma) directory shared by the DNS and its DO runs
RUN=$PAIR/do_Re${RE}_sig${SIGMA}_S${S}_Np${NP}_seed${SEED}
[ "$EPS" = "0.01" ] || RUN=${RUN}_eps${EPS}   # default eps (1e-2) runs carry no suffix
[ "$FSEED" = "$SEED" ] || RUN=${RUN}_frc${FSEED}   # noise-only repeats carry the forcing seed
DNS=$PAIR/dns_Re${RE}_sig${SIGMA}_seed${SEED}
[ -e "$RUN" ] && { echo "error: $RUN already exists, not overwriting" >&2; exit 1; }
[ -d "$DNS/output/chks" ] || { echo "error: no DNS snapshots at $DNS/output/chks" >&2; exit 1; }
echo "task $SLURM_ARRAY_TASK_ID: Re=$RE sigma=$SIGMA S=$S Np=$NP stride=$STRIDE seed=$SEED fseed=$FSEED T_END=$T_END eps=$EPS -> $RUN"

# --- snapshot pattern (relative to the run dir; ';' joins several globs) -----
REL=../dns_Re${RE}_sig${SIGMA}_seed${SEED}/output/chks/casefile_
case "$STRIDE" in
    1) PATTERN="${REL}*.chk" ;;
    2) PATTERN="${REL}*0.chk;${REL}*2.chk;${REL}*4.chk;${REL}*6.chk;${REL}*8.chk" ;;
    5) PATTERN="${REL}*0.chk;${REL}*5.chk" ;;
    *) echo "error: stride $STRIDE not supported (1, 2 or 5)" >&2; exit 1 ;;
esac

# --- run dir: inputs + output tree ------------------------------------------
mkdir -p "$RUN/output/chks" "$RUN/output/do" "$RUN/output/asymmetry" "$RUN/output/energy" "$RUN/output/measures"
cp "$TPL/geometry.opt" "$TPL/ic.chk" "$TPL/qdag_channel.fld" "$RUN/"
cp "$TPL/geometry_m03.xml" "$RUN/geometry.xml"   # quad mesh (meshes/m03); templates/geometry.xml is the tri mesh
sed -e "s|@DT@|0.01|" -e "s|@T_END@|$T_END|" -e "s|@RE@|$RE|" \
    -e "s|@DO_MODES@|$S|" -e "s|@DO_PARTICLES@|$NP|" \
    -e "s|@YI_SEED@|1234|" -e "s|@YI_SIGMA@|0.01|" \
    -e "s|@FORCING_TAU@|25|" -e "s|@FORCING_SIGMA@|$SIGMA|" -e "s|@FORCING_SEED@|$FSEED|" \
    -e "s|@SNAPSHOT_PATTERN@|$PATTERN|" -e "s|@CHK_STEPS@|2500|" -e "s|@INFO_STEPS@|100|" -e "s|@COV_EPS@|$EPS|" \
    "$TPL/casefile_do.xml" > "$RUN/casefile.xml"
if grep -q '@' "$RUN/casefile.xml"; then
    echo "error: unsubstituted placeholder in $RUN/casefile.xml" >&2; exit 1
fi

# --- solve ------------------------------------------------------------------
module load gcc openmpi openblas arpack-ng
cd "$RUN"
srun "$SOLVER" geometry.xml casefile.xml > run.log 2>&1
