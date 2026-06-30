# py_utils

Reproducible utilities for DO runs.

Use `runs/<...>/<run_id>/` as the working directory for real experiments.
Keep `cases/<...>/` as templates/assets.

## IC generation (DO ensemble)

Create a profile template:

```bash
python3 py_utils/workflow.py ic-profile \
  --out runs/<run_id>/ic/my_ic_profile.py \
  --case-xml runs/<run_id>/casefile.xml
```

Generate `.fld` IC samples + manifest:

```bash
python3 py_utils/workflow.py ic-generate \
  --profile runs/<run_id>/ic/my_ic_profile.py \
  --mesh-pts runs/<run_id>/ic/mesh_pts.pts \
  --case-xml runs/<run_id>/casefile.xml \
  --n-samples 500 \
  --seed 1 \
  --samples-dir runs/<run_id>/ic/samples \
  --manifest runs/<run_id>/ic/do_init_samples.txt
```

Notes:
- Works for 2D and 3D mesh point files (`DIM="2"` or `DIM="3"`).
- Manifest is one `.fld` path per line; used by `DOInitSampleFilesManifest`.
- The generated default profile is geometry-agnostic (smooth random fields),
  with no built-in vortex assumption.
- `ic-profile` infers `FIELDS` from case XML variables; use `--fields` to override.
- Profile API supports:
  - `sample_parameters(rng, sample_index)`
  - `sample_parameters(rng, sample_index, coords_meta)`

## Post-processing pipeline

```bash
python3 py_utils/workflow.py run-case --case-dir runs/<run_id>
```

Default stages:
- `extract`
- `suite`
- `reg-derive-stats`

Default outputs:
- `runs/<run_id>/output/out.h5`
- `runs/<run_id>/output/py_utils_results/`

Larger pipeline:

```bash
python3 py_utils/workflow.py run-case \
  --case-dir runs/<run_id> \
  --stages extract,derive,suite,check,video,panels,reg-derive-stats
```

## Mesh plotting

```bash
python3 py_utils/workflow.py plot-mesh --xml /path/to/geometry.xml
```
