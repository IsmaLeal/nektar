# Cases Overview

- Structure: each case sits in its own folder (e.g., `cases/LambOseen2D/`) with a session file (`*.xml`), optional parameters (`*.opt`), restart/initial conditions (`*.fld`), and `*.tst` files when run under the `Tester` harness. Keep outputs inside an `output/` subfolder to stay clear of git.
- Running manually: from your build tree, use the solver executable that matches the session, e.g. `./build/dist/bin/IncNavierStokesSolver cases/LambOseen2D/KovaFlow_m8.xml`. Redirect outputs to `cases/.../output/` to avoid clutter.
- Using Tester: run `ctest -R Tester` from `build/` or call `./build/dist/bin/Tester cases/LambOseen2D/KovaFlow_m8.tst` for a single case. Add new `.tst` files next to their sessions and register them in the relevant `CMakeLists.txt` if needed.
- Naming: prefer descriptive folder names and variants (e.g., `<FlowType>_<Re>_<BC>`). Store small, reproducible inputs in git; keep heavy result files (`*.fld`, `*.chk`, time histories) in `output/` and out of commits.
