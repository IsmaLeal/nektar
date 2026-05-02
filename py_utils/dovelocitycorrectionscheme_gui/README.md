# DOVelocityCorrectionScheme GUI

Desktop front-end for the DOVelocityCorrectionScheme DO-VCS solver and the `py_utils` post-processing pipeline.

## Install dependencies

```bash
pip install -r py_utils/dovelocitycorrectionscheme_gui/requirements.txt
```

## Launch

```bash
python3 py_utils/workflow.py gui
# or
python3 -m py_utils.dovelocitycorrectionscheme_gui
```

## Two tabs

### Run

1. Pick a template `casefile.xml` (e.g. `cases/vortex/runs/04_29_firstchecks/casefile.xml`).
2. Edit time, physics, DO, stochastic-forcing, checkpoint cadence.
3. Choose a run directory; click **Save XML** (writes `casefile.xml`) or **Run**.
4. The status panel shows live ETA: each `*.chk` drop is fed into a rolling
   linear regression of wall-time vs. simulation time, and the bar/ETA update
   on every drop.

The solver binary is auto-detected at:

- `$DOMINE_SOLVER` (env var override)
- `build/solvers/IncNavierStokesSolver/IncNavierStokesSolver`
- `build/dist/bin/IncNavierStokesSolver`
- `build-debug/solvers/IncNavierStokesSolver/IncNavierStokesSolver-g`
- `which IncNavierStokesSolver`

### Visualize

Pick a run directory; XML, `out.h5` and a default output dir are auto-detected.
Choose a stage:

- **Vorticity video** — wraps `py_utils/vorticity_video.py`.
- **DO multi-panel animation** — wraps `py_utils/reconstruction/animation/animate_do_panels.py`.
- **Convergence diagnostics** — wraps `py_utils/check_convergence.py`.
- **Mesh geometry** — wraps `py_utils/mesh/plot_geometry_xml.py`.

Hit **Render**. Logs stream into the console; static plots are shown inline,
videos play in an embedded `QMediaPlayer`. The output file path appears under
the preview.

## Notes

- XML editing is round-trip safe via `lxml`: only the named `<P>` parameters,
  the `SolverType` `<I>` entry, the `ForcingChannels` block, and the
  `Checkpoint`/`DOArchive` filters are touched. Anything else (boundary
  conditions, geometry, expansion table, comments) is preserved.
- ETA needs at least two checkpoint drops to fit a slope; until then the field
  shows `—` and the bar fills based on the most recent observation only.
- The visualization tab parses `frame i/N` log lines (where present) to drive a
  determinate progress bar; otherwise it shows an indeterminate spinner.
