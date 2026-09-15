# Contributing to numerical-core

numerical-core is a library of numerical primitives for dense computations on
Kotlin/JVM with reliability control of the results. The rules below govern
all changes to the code and documentation.

## Language

KDoc, comments, commit messages and all documentation are written in English; the only
exception is `docs/ABSTRACT.md`, the software-registration abstract, which is kept in
Russian. Identifiers in code are in English. The style
is descriptive: no capitalized emphasis, no jargon, no references to external projects or
version history.

## Public API

- `explicitApi()` mode is enabled: every new element is declared `internal`. An element
  becomes public only once it has KDoc and a test covering its contract.
- An unreliable value is never returned as a plain number: `ForwardError`,
  `Measured`, `ConditionEstimate.isReliable` or `null` are used instead.
- For non-finite input (`NaN`, `±Inf`) the `solve` and `cholesky` methods throw an exception;
  the remaining operations propagate it according to the rules of floating-point arithmetic,
  like BLAS.
- The behavior of a method (checks, exceptions, result shape) does not depend on the
  BLAS/LAPACK implementation.

## Numerical methods

- Every method is accompanied by an accuracy test (comparison with an independent oracle or
  with an analytic answer) and an entry in `docs/SOURCES.md` citing the source of the formulas.
- Thresholds (`SINGULARITY_RELATIVE_TOLERANCE`, `INVERSION_RESIDUAL_TOLERANCE`,
  `MACHINE_NOISE_THRESHOLD`) are changed only with a justification in `docs/ACCURACY.md`.
- A change to a formula is verified by the golden reference tests and by jqwik properties on
  both implementations (`-Dnumerics.backend=java` and `native`). Golden references are
  regenerated with `./gradlew :numerical-core:regenerateGolden` only on a deliberate behavior
  change, with the reason recorded in `core/src/test/resources/golden/README.md`.

## Commands

    ./gradlew build                                              # tests, Kover coverage, both modules
    ./gradlew :numerical-core:test -Dnumerics.backend=java       # portable Java implementation
    ./gradlew :numerical-core:test -Dnumerics.backend=native     # system implementation
    ./gradlew :numerical-core:benchmark -Pbench.args="256 1024"  # performance measurements
    ./gradlew :numerical-core:dokkaHtml                          # API documentation

The `check` task includes the Kover coverage threshold; a change that drops coverage below
the threshold fails the build. All commands are run with the `--offline` flag.

Releasing a version: a `vX.Y.Z` tag on `main` triggers publication to GitHub Packages from CI.

## Documentation

After an API change, update `README.md` (the contents table, the example) and the
corresponding document in `docs/`. The numbers in `docs/PERFORMANCE.md` are obtained with
the `benchmark` command, stating the machine and the BLAS/LAPACK implementation.

## Commits

Commit messages are written in English with an area prefix: `api:`, `algo:`,
`test:`, `docs:`, `build:`. One change corresponds to one commit; the message describes
what changed and why.
