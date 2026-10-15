# py_utils

The tools of the sudden-expansion campaign (`cases/se`): one module, `se.py`, holds the basic tools, the
rest are scripts that use it. Everything is in `sudden_exp_postproc/`.

Interpreters: `/usr/bin/python3` (Python 3.12, numpy 1.26, h5py, matplotlib, scipy, pandas) for every
script and for `cases/se/figures/Makefile`; the manim drawings (`*_still.py`, `dudy_wall_movie.py`,
`channel.py`) run in the divulg venv, `~/todisimo/despojos/divulg/.venv/bin/python` (manim + classic_isma,
no h5py), called by the script that prepares their data.

Installed once in each (done on this laptop, 2026-10-15), so `import se` works in any directory, in a
console, a script or PyCharm, and every script runs from anywhere:

```bash
/usr/bin/python3 -m pip install --user --no-build-isolation --break-system-packages -e py_utils
~/todisimo/despojos/divulg/.venv/bin/python -m pip install -e py_utils
```

Every script says at its top what it does and how to call it: `python3 <script>.py --help`.

## se.py: the basic tools

```python
import se
```

| section | names | what they do |
|---|---|---|
| plane | `x, y, X, Y, DX, DY, INSIDE, W, JLOW, JUP, XPOS, XWALL` | the 901 x 121 extraction plane, x in [-5, 40], y in [-1.5, 1.5]; INSIDE is False on the inlet walls |
| plane measures | `ip, m_grid, reattachment, mirror, antisym, read_dat` | inner product, M, bubble end, mirror y -> -y; defined in `cases/se/templates/extract_lib.py` (the module JED runs), re-exported |
| runs | `SWEEP, params, h5, is_filter, find, runs, pair_dir, eps_str, short_eps, disk_eps` | runs of the JED sweep: what a run name says, its measures file; eps is typed and printed as 1e-4 (the run directories spell it eps0.0001, find accepts both) |
| rules | `C, M_WELL, DL_FP, a_fp(Re)` | the thresholds of every transition count: a crossing of +-C x well level (crossings) |
| load | `load(f)` | every dataset of a measures file in a dict; u, v and yi stay on disk until sliced |
| series | `dns, do_all, do_one, series` | A, M, dL, Y1 against time, for a DNS, every DO particle, or one particle |
| particle | `particle_plane(h, p, k)` | u, v of DO particle p on the plane at the k-th instant of grid/t (python extraction only) |
| derivatives | `wall_dudn, dL_shear, vorticity` | du/dn on the first row off each wall, the bubble length difference on it, the vorticity |
| crossings | `crossings(t, X, h)` | the hysteresis rule of every transition count and escape time |
| raw runs | `FIELDCONVERT, fc_env, interp, casefile_param, load_dat, yi_from_fld, yi_series` | the archives themselves: FieldConvert on them, the Yi in their metadata |

The measures file of a run, `se.h5(run)`, is either of two: the python extraction JED ran on the
archives (`output/extract/grid.h5` for a DNS, `measures.h5` for a DO run; `cases/se/templates/extract.sbatch`),
or `output/measures/measures.h5`, written by the solver's DOMeasures filter (exact quadrature on the
expansion, format DO_MEASURES_V1, `casefile_do.xml`). The series functions read either; the filter is the
format of every new run, the extraction stays for the 2026_09_13 sweep. The filter keeps no planes, so
`particle_plane`, `do_fields.py` and `draw_qdag.py` need the extraction or the archives.

Examples:

```python
import h5py, matplotlib.pyplot as plt, se

run = se.SWEEP / 'Re100_sig0.0144/dns_Re100_sig0.0144_seed45'
d = se.load(se.h5(run))                 # d['t'], d['A'], d['M_grid'], d['u'] ...
tau = se.wall_dudn(d['u'][1000])        # du/dn at |y| = 1.475, lower tau[0], upper tau[1]
plt.plot(se.x, tau[0]); plt.plot(se.x, tau[1]); plt.axhline(0)

t, X = se.series(run, 'A')              # X of shape (T, 1)
print(len(se.crossings(t, X[:, 0], 0.5 * se.a_fp(100))), 'crossings of +-0.5 A_fp')

pair = se.pair_dir('Re100_sig0.0144')
runs, p = se.find(pair, 'S6_Np1000_seed45_eps1e-4@269')      # one DO run, particle 269
t, X = se.series(runs[0], 'M', p)
```

## The scripts

Figures of `cases/se/figures` (`make fields hists stills traces rates` there; the folder names are at the top
of its Makefile; every figure gets a `.cmd` next to it with the command line that made it, `figsave.py`):

- `panel.py`: one figure out of panels; each panel is a plot kind drawn by `draw_hist.py` (histograms),
  `draw_cross.py` (A(t) with crossings), `draw_trace.py` (DNS twin against a DO particle) or
  `draw_qdag.py` (q^dagger on the DO modes).
- `vary_hists.py`: every histogram where one of S, Np, eps varies, one figure per line and quantity, with
  the DNS (`--list` prints the lines). `grid_hists.py`: the S x Np grid of one pair, one DO curve per eps
  in each cell, drawn in manim (`grid_hists_still.py`). `sweep_hists.py`: one panel per (Re, sigma) pair of
  the whole sweep. All three count their histograms with `draw_hist.py`.
- `sweep_rates.py`: transition rates of the forced DNS sweep (table and figure). `do_rates.py`:
  per-particle rates of the DO runs at Re 100, sigma 0.0144 (table).
- `plot_plane.py`: fields on the extraction plane (phi, modes, q^dagger); `plane_from_fld.py` puts any .fld
  on that plane as an .npz.
- `do_fields.py` (+ `do_fields_still.py`): still of phi and mode 1 of every DO run, with <u_K, phi>^2 and
  <u_K - phi, u_K - phi>^2 along the run. `dudy_wall_data.py` + `dudy_wall_movie.py`: movie of du/dy at
  the two walls against x, with the u field between them, along a DNS. `channel.py`: the channel they both
  draw (outline, inlet solid, field colours, colour bar).

Videos of a DO run: `render_do_panels.py` (FieldConvert on the archives, then `assemble_do_h5.py`, then
`animate_do_panels.py`).

Validation checks of July 2026 (research plan, section 3.3): `convergence_orders.py` (temporal orders of
the coupled DO scheme), `rankinv_check.py` (invariance under the MPI rank count), `vortex_mc_check.py`
(matched-ensemble Monte Carlo check of the DO closure), `forcing_gain_check.py` (white-noise injection
audit). `compare_measures.py`: the DOMeasures filter of a run against the python extraction of the same run.

Console scripts, called by nothing: `qdag_projection.py` (q^dagger on the DO modes, the original of
`draw_qdag.py`), `phi_vs_u1.py` (mode 1 against phi at the archived plane instants), `mean_check.py`
(is the DO mean the long-time DNS mean), `asym_measures.py` (the workflow of 2026-09-01 on an out_grid.h5).
