#!/bin/bash
#SBATCH --job-name=extract
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/lealaske/se_runs/logs/extract_%j.out
#
# extract.sbatch -- interpolate every checkpoint (dns) or every archive
# (do) of one run on the 901 x 121 plane, keeping only the asymmetry
# measures' inputs, and merge into one HDF5 in RUN/output/extract/.
#
#   sbatch extract.sbatch dns RUN PHI
#   sbatch extract.sbatch do  RUN PHI [KEEP_EVERY]
#
# RUN        absolute run directory (/scratch/lealaske/se_runs/...)
# PHI        phi npz from make_phi.sh, or "none" (then A is not computed;
#            for a DNS it can be computed later from grid.h5)
# KEEP_EVERY DO only: keep the full plane of mean and modes for archives
#            whose step is a multiple of KEEP_EVERY*100 (i.e. every
#            KEEP_EVERY time units). Default 1000. 0 = none.
#
# Output: RUN/output/extract/grid.h5 (dns) or measures.h5 (do); the per-file
# .npz stay in RUN/output/extract/npz until the h5 is verified, then are
# removed. Restartable: files already reduced are skipped.

set -u  # raise error if there is an undefined variable
# save the arguments
MODE=${1:?dns|do}; RUN=${2:?run dir}; PHI=${3:?phi npz or none}; KEEP_EVERY=${4:-1000}

TPL=${TPL:-$HOME/nektar/cases/se/templates}     # overridable for a local dry run
export FIELDCONVERT=${FIELDCONVERT:-$HOME/nektar/build/utilities/FieldConvert/FieldConvert}
export EXTRACT_LIB=$TPL/extract_lib.py          # used by extract_one.sh to get the useful data (Yi, a_i, G, A, axis and walls) into an .npz 
module load gcc openmpi openblas arpack-ng
[ -f ~/venvs/postproc/bin/activate ] && source ~/venvs/postproc/bin/activate  # check venv exists and activate it

[ -d "$RUN" ] || { echo "no run dir $RUN" >&2; exit 1; }	# || runs the 2nd command if the 1st fails
[ "$PHI" = none ] || [ -s "$PHI" ] || { echo "no phi file $PHI" >&2; exit 1; }
OUT=$RUN/output/extract; NPZ=$OUT/npz; mkdir -p "$NPZ"		# create output dirs
NP=${SLURM_CPUS_ON_NODE:-$(nproc)}
echo "$(date) $MODE $RUN phi=$PHI keep_every=$KEEP_EVERY workers=$NP"

cd "$RUN"
if [ "$MODE" = dns ]; then
    ls output/chks/casefile_*.chk | sed 's|^|-|' > "$OUT/list.txt"	# list all casefiles in list.txt
else
    ls -d output/do/casefile.do_*.fld | sed 's|^|-|' > "$OUT/list.txt"
fi
sed -i 's|^-||' "$OUT/list.txt"		# sed -i overwrites the first regex (the beginning of the line with - and removes -) into list.txt
N=$(wc -l < "$OUT/list.txt"); echo "$N files"	# wc -l counts lines in list.txt (with < it receives the contents, outputting the #lines without the filename)

# KEEP flag per file (DO only): "grid" when step % (KEEP_EVERY*100) == 0
keep_of() {
    [ "$MODE" = do ] && [ "$KEEP_EVERY" -gt 0 ] || { echo -; return; }
    s=$(basename "$1" | sed -n 's/^casefile\.do_\([0-9]*\)\.fld$/\1/p'); s=$((10#$s))
    [ $((s % (KEEP_EVERY * 100))) -eq 0 ] && echo grid || echo -
}
export -f keep_of; export MODE KEEP_EVERY
: > "$OUT/args.txt"
while read -r f; do echo "$f $(keep_of "$f")" >> "$OUT/args.txt"; done < "$OUT/list.txt"

# extract one by one
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1   # one BLAS thread per worker: NP workers x 64 threads hits RLIMIT_NPROC
xargs -P "$NP" -n 2 sh -c "$TPL/extract_one.sh $MODE $RUN \$0 $NPZ $PHI \$1" < "$OUT/args.txt" 2> "$OUT/failed.txt"
NF=$(grep -c FAILED "$OUT/failed.txt" || true)
NOK=$(ls "$NPZ"/*.npz 2>/dev/null | wc -l)
echo "$(date) reduced $NOK / $N, failed $NF"
[ "$NF" -eq 0 ] || { echo "failures listed in $OUT/failed.txt; rerun the same command to retry them" >&2; exit 1; }

# assemble the HDF5
if [ "$MODE" = dns ]; then
    python3 "$TPL/assemble.py" dns "$RUN" "$NPZ" "$OUT/grid.h5" --chk-steps 400 || exit 1
else
    python3 "$TPL/assemble.py" do "$RUN" "$NPZ" "$OUT/measures.h5" || exit 1
fi
rm -rf "$NPZ"
echo "$(date) done: $(ls -la "$OUT"/*.h5)"
