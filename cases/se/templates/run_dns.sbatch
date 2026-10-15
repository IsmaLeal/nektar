#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time 60:00:00
#SBATCH --output /scratch/lealaske/se_runs/logs/dns_%A_%a.out
#
# run_dns.sbatch -- one forced DNS per array task. Row SLURM_ARRAY_TASK_ID+1
# of dns_sweep.tsv (Re, sigma, seed, T_END[, NP]) fills casefile_dns.xml; the run lives in
# /scratch/lealaske/se_runs/Re<Re>_sig<sigma>/dns_Re<Re>_sig<sigma>_seed<seed>/ (one
# directory per (Re, sigma) pair holds its DNS and DO runs; never reused: an
# existing run dir aborts the task). NP = DOParticles (default 1); a row with
# NP != 1 gets the suffix _Np<NP> (its forcing = particle 0 of the DO run with
# that seed and Np). An optional 6th column TAG is appended to the name as is
# (- for none); an optional 7th column IC names the initial condition file in
# the templates dir (default ic.chk), e.g. the start of one DO particle written
# by make_particle_ic.py.
#
# Launch from the templates dir, N = number of rows in dns_sweep.tsv:
#   mkdir -p /scratch/lealaske/se_runs/logs
#   sbatch --array=0-$((N-1)) run_dns.sbatch
set -eu

TPL=$HOME/nektar/cases/se/templates
ROOT=/scratch/lealaske/se_runs
SOLVER=$HOME/nektar/build/solvers/IncNavierStokesSolver/IncNavierStokesSolver

# --- row of the sweep table -------------------------------------------------
ROW=$((SLURM_ARRAY_TASK_ID + 1))
read -r RE SIGMA SEED T_END NP TAG IC < <(sed -n "${ROW}p" "$TPL/dns_sweep.tsv")
NP=${NP:-1}; TAG=${TAG:-}; [ "$TAG" = - ] && TAG=; IC=${IC:-ic.chk}
[ -f "$TPL/$IC" ] || { echo "error: no initial condition $TPL/$IC" >&2; exit 1; }
[ -n "${T_END:-}" ] || { echo "error: row $ROW of dns_sweep.tsv is missing or short" >&2; exit 1; }

RUN=$ROOT/Re${RE}_sig${SIGMA}/dns_Re${RE}_sig${SIGMA}_seed${SEED}   # the (Re, sigma) directory shared with the DO runs
if [ "$NP" != 1 ]; then RUN=${RUN}_Np${NP}; fi
RUN=${RUN}${TAG}
[ -e "$RUN" ] && { echo "error: $RUN already exists, not overwriting" >&2; exit 1; }
echo "task $SLURM_ARRAY_TASK_ID: Re=$RE sigma=$SIGMA seed=$SEED T_END=$T_END NP=$NP IC=$IC -> $RUN"

# --- run dir: inputs + output tree ------------------------------------------
mkdir -p "$RUN/output/chks" "$RUN/output/asymmetry" "$RUN/output/energy"
cp "$TPL/geometry.opt" "$TPL/qdag_channel.fld" "$RUN/"
cp "$TPL/$IC" "$RUN/ic.chk"
cp "$TPL/geometry_m03.xml" "$RUN/geometry.xml"   # quad mesh (meshes/m03); templates/geometry.xml is the tri mesh
sed -e "s/@RE@/$RE/" -e "s/@SIGMA@/$SIGMA/" -e "s/@SEED@/$SEED/" -e "s/@T_END@/$T_END/" -e "s/@NP@/$NP/" \
    "$TPL/casefile_dns.xml" > "$RUN/casefile.xml"
if grep -q '@' "$RUN/casefile.xml"; then
    echo "error: unsubstituted placeholder in $RUN/casefile.xml" >&2; exit 1
fi

# --- solve ------------------------------------------------------------------
module load gcc openmpi openblas arpack-ng
cd "$RUN"
srun "$SOLVER" geometry.xml casefile.xml > run.log 2>&1
