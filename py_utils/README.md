# py_utils

Python tools around the Nektar++ DO solver (DOVelocityCorrectionScheme). Two independent parts:

- `sudden_exp_postproc/`: the sudden-expansion campaign (`cases/se`). Its basic tools are in one
  module, `se.py`; the rest are scripts that use it.
- everything else in this folder: the generic DO tool chain of June 2026 (any case: vortex,
  cylinder), with `workflow.py` as its entry point.

Interpreter: `/usr/bin/python3` (Python 3.12, numpy 1.26, h5py, matplotlib, scipy, pandas). Every
script except the GUI runs with it, and `cases/se/figures/Makefile` calls it. The GUI needs PyQt6
and qdarkstyle, which are not installed (`dovelocitycorrectionscheme_gui/README.md`).

## se.py: the basic tools of the sudden-expansion campaign

From a console or any script:

```python
import sys; sys.path.insert(0, '/home/isma/nektar_src_full/py_utils/sudden_exp_postproc')
import se
```

In PyCharm: right click on `py_utils/sudden_exp_postproc`, Mark Directory as, Sources Root; then
`import se` works in its Python console and in every file.

| section | names | what they do |
|---|---|---|
| plane | `x, y, X, Y, DX, DY, INSIDE, W, JLOW, JUP, XPOS, XWALL` | the 901 x 121 extraction plane, x in [-5, 40], y in [-1.5, 1.5]; INSIDE is False on the inlet walls |
| plane measures | `ip, m_grid, reattachment, mirror, antisym, read_dat` | inner product, M, bubble end, mirror y -> -y; defined in `cases/se/templates/extract_lib.py` (the module JED runs), re-exported |
| runs | `SWEEP, params, h5, find, pair_dir, a_fp` | runs of the JED sweep: what a run name says, its grid.h5 / measures.h5 |
| load | `load(f)` | every dataset of a grid.h5 or measures.h5 in a dict; u, v and yi stay on disk until sliced |
| series | `dns, do_all, do_one, series` | A, M, dL, Y1 against time, for a DNS, every DO particle, or one particle |
| particle | `particle_plane(h, p, k)` | u, v of DO particle p on the plane at the k-th instant of grid/t |
| derivatives | `wall_dudn, dL_shear, vorticity` | du/dn on the first row off each wall, the bubble length difference on it, the vorticity |
| crossings | `crossings(t, X, h)` | the hysteresis rule of every transition count and escape time |
| raw runs | `FIELDCONVERT, fc_env, interp, casefile_param, load_dat, read_his, axis_M, his_M, yi_from_fld, yi_series, mp_series, mp_series_chain` | runs read without the JED extraction: FieldConvert on the archives, HistoryPoints files |

Examples:

```python
import h5py, matplotlib.pyplot as plt, se

run = se.SWEEP / 'Re100_sig0.0144/dns_Re100_sig0.0144_seed45'
f = h5py.File(se.h5(run), 'r')
d = se.load(f)                          # d['t'], d['A'], d['M_grid'], d['u'] ...
tau = se.wall_dudn(d['u'][1000])        # du/dn at |y| = 1.475, lower tau[0], upper tau[1]
plt.plot(se.x, tau[0]); plt.plot(se.x, tau[1]); plt.axhline(0)

g = h5py.File(se.h5(se.SWEEP / 'Re100_sig0.0144/do_Re100_sig0.0144_S8_Np1000_seed45_eps0.0001'), 'r')
u, v = se.particle_plane(g, p=6, k=3)   # particle 6 at g['grid/t'][3]

t, X = se.series(run, 'A')              # X of shape (T, 1)
print(len(se.crossings(t, X[:, 0], 0.5 * se.a_fp(100))), 'crossings of +-0.5 A_fp')
```

## sudden_exp_postproc/: the scripts

Figures of `cases/se/figures` (`make fields hists stills traces rates` there):

- `vary_hists.py`: every histogram where one thing varies: finds the lines of DO runs (same pair, one of
  S, Np, eps varied; the forcing-seed repeats _frc pooled into their run) and draws M, A and the
  escape times of each, with the DNS (`--list` prints the lines).
- `panel.py`: one figure out of panels; each panel is a plot kind drawn by `draw_hist.py`
  (histograms), `draw_cross.py` (A(t) with crossings), `draw_trace.py` (DNS twin against a DO
  particle) or `draw_qdag.py` (q^dagger on the DO modes).
- `plot_plane.py`: fields on the extraction plane (phi, modes, q^dagger); `plane_from_fld.py`
  puts any .fld on that plane as an .npz.
- `sweep_rates.py`: transition rates of the forced DNS sweep (table and figure).
- `do_rates.py`: per-particle transition rates of the DO runs at Re 100, sigma 0.0144 (table).
- `jed_sweep_freemode.py`: free-mode reader of the sweep and the 3 x 3 histogram grid of M.
- `figsave.py`: saves a figure and, next to it, the .cmd with the command line that made it.

The folders of `cases/se/figures` are written down at the top of its Makefile: `Re<Re>/`,
`Re<Re>_sig<sigma>/` with one folder per run and per line of DO runs (`vary_<p>_<fixed>/`),
`sweep/`, `candidacy/`.

Figures in the manim house style (classic_isma): the data with `/usr/bin/python3`, the drawing with
`~/todisimo/despojos/divulg/.venv/bin/python` (manim, no h5py):

- `dudy_wall_data.py` + `dudy_wall_movie.py`: movie of du/dy at the two walls against x, with the u
  field between them, along a DNS.
- `do_fields.py` (+ `do_fields_still.py`, the drawing): still of phi and mode 1 of every DO run, with
  <u_K, phi>^2 and <u_K - phi, u_K - phi>^2 along the run; one command, `python3 do_fields.py`
  (or the runs named).

Console scripts, called by nothing:

- `qdag_projection.py`: projection of q^dagger on the DO modes (the original of `draw_qdag.py`).
- `phi_vs_u1.py`: the distance between mode 1 and phi at the archived plane instants of one DO run.
- `asym_measures.py`: the workflow of 2026-09-01 on the out_grid.h5 of the 2026-08-16 run
  (M1, the centreline M, dL, dL from du/dy, deficit barycentre).

Raw runs (read from the archives and HistoryPoints files, before the JED extraction):

- `plot_dns_M_v.py`: M(t) and the centreline sensor of a run; for a DO run, M of every particle.
- `do_stats_plots.py`: transition statistics of a DO run (trajectories, pdf of M, escape times,
  occupation of Y_i), with DNS overlays.
- `mbar_from_his.py`: M(t) of a HistoryPoints axis file, final value and tail mean.
- `phase_space_video.py`, `yi_timeseries.py`: Y_1 against Y_2 as a video; Y_i against time.
- `render_do_panels.py`: the do_panels video of a DO run from scratch (plane interpolation with
  FieldConvert, then `assemble_do_h5.py`, then `reconstruction/animation/animate_do_panels.py`).

Validation checks of July 2026 (research plan, section 3.3):

- `convergence_orders.py`: temporal orders of the coupled DO scheme from a dt-refinement family.
- `rankinv_check.py`: invariance of the ensemble statistics under the MPI rank count.
- `vortex_mc_check.py`: matched-ensemble Monte Carlo check of the DO closure (vortex case).
- `forcing_gain_check.py`: white-noise injection audit of the DO forcing path.

`candidacy_figs/`: one script per figure of the candidacy plan and slides; the figures are kept
in `cases/se/figures/candidacy/` (`make candidacy` copies them to the candidacy folders).
`fig_mowlavi_comparison.py`: the vortex validation figure against the published DO results.

## Generic DO tool chain (top level)

- `workflow.py`: one entry point for the stages below (`python3 py_utils/workflow.py --help`).
- `load_chk.py`: checkpoints and DO archives of a run into one out.h5.
- `reconstruction/`: realisations from mean, modes and coefficients; videos of a DO run
  (`animate_do_panels.py`) and of one realisation (`animate_single.py`).
- `tests/`: the DO validation suite (`run_do_test_suite.py`) and its tests; `regression/`
  derives the DO statistics CSV and plots the mode variances.
- `ic_sampler/`: initial-condition samples for the DO initialisation.
- `mesh/plot_geometry_xml.py`: plot of a 2D mesh.
- `dovelocitycorrectionscheme_gui/`: desktop GUI (`python3 py_utils/workflow.py gui`).
- `common/`: helpers of this chain (npz/h5 loading, divergence, norms).
- Standalone: `check_convergence.py` (settling of a run), `compare_do_runs.py` (two or more DO
  runs side by side), `do_subspace_convergence.py` (convergence diagnostics between two DO runs), `do_compose_mc_ics.py` (per-particle
  initial conditions from a DO archive), `mpi_scaling.py` (MPI scaling study),
  `plot_decay_rate.py` (|v(x, 0, t)| at several x), `plot_forcing_channels.py` (the forcing
  fields of a case), `plot_frame.py` (last frame of an out.h5), `vorticity_video.py`.

### IC generation (DO ensemble)

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

### Post-processing pipeline

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

### Mesh plotting

```bash
python3 py_utils/workflow.py plot-mesh --xml /path/to/geometry.xml
```
