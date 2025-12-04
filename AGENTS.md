# Repository Guidelines
Always answer in English. I am a PhD student at EPFL, Lausanne, in the department of Mechanical Engineering at the Laboratory of Fluid Mechanics and Instabilities.

## Project Structure & Module Organization
- Core libraries live in `library/` (framework) and `ThirdParty/` (vendored deps). Pre-built PDE solvers are under `solvers/`, with reusable utilities in `utilities/` and reusable templates in `templates/`. Cases and sample inputs sit in `cases/` and `tests/Examples/`. Packaging scripts reside in `pkg/`, CMake helpers in `cmake/`, and docs in `docs/`. Keep build artefacts in `build/` and avoid committing generated files.

## Build, Test, and Development Commands
- Configure & build: `mkdir -p build && cd build && cmake .. && make -j$(nproc)`. Use `ccmake ..` for interactive options (MPI, GPU flags, install prefix).
- Install locally from the build dir: `make install` (default to `build/dist`).
- Run automated tests after a build: `ctest -j$(nproc)`; filter with `ctest -R <pattern>`. Solver example suites can also be driven via the `Tester` executable for `.tst` files in `tests/Examples/`.
- Developer tip: clean config with `rm -rf build/*` when changing toolchains or major options.

## Coding Style & Naming Conventions
- C++17 code is formatted with clang-format 16 using the repo `.clang-format` (4-space indent, braces on new lines for control blocks/classes). Run `git clang-format` on touched files before committing.
- Prefer descriptive names mirroring solver physics; keep consistency with surrounding code (e.g., class names in `CamelCase`, local variables `snake_case`).
- Add Doxygen comments for public APIs and keep in-source comments concise and purpose-driven.

## Testing Guidelines
- Add regression or unit tests alongside the code: solver-specific tests go under the corresponding `solvers/<Solver>/Tests` or `tests/Examples` entry, library tests under the matching `library/.../Tests`. Register new tests in the relevant `CMakeLists.txt`.
- Run `ctest` (or `ctest -C Release` if you built with multi-config) before pushing; note any expected skips. Attach small input files only—avoid large data.

## Commit & Pull Request Guidelines
- Branch names follow `feature/<name>`, `fix/<name>`, `ticket/<id>-<name>`, or `tidy/<name>`. Keep commits focused; subjects should be short, imperative, and reference tickets when applicable.
- Before opening an MR to `master`, ensure tests pass, code is clang-formatted, docs are updated, and a CHANGELOG entry is added when user-facing changes occur.
- Fill out the MR template with a clear problem statement, solution summary, test evidence (`ctest`/`Tester` output), and any screenshots for UI/docs changes. Prefer small, reviewable MRs and respond promptly to review comments.

