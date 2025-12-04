# IncNavierStokesSolver Tests

- Purpose: regression/acceptance tests for the incompressible Navier-Stokes solver. Keep tests small so they run quickly under `ctest`.
- Running: from `build/`, run all tests with `ctest -R IncNavierStokesSolver` or a single test with `ctest -R <name>`. You can also call `./dist/bin/Tester <path/to/test>.tst` for ad-hoc runs.
- Adding tests: place the session (`*.xml`), options (`*.opt`), and expected output references alongside the `.tst` file. Register the new test in this solver’s `CMakeLists.txt` so CTest picks it up.
- Outputs: direct solver outputs to an `output/` subfolder (gitignored) when experimenting locally. Only commit input/reference files that are required for reproducibility and are small enough to live in the repo.
- Naming: follow clear, physics-focused names (e.g., `KovaFlow_Re<Re>_<BC>`), and keep variants consistent with existing cases.
