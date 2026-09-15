# Golden behavior references of numerical-core 0.1.0

Purpose: to record the actual behavior of the public API of version 0.1.0 (linear solves,
matrix operations, Cholesky factorization, conditioning, eigenvalues, Gauss–Legendre
quadrature, the pure functions `Measured`/`ForwardError`/`orders`) so that regressions
introduced while reworking the library are detected by comparison with the golden reference,
not only by property tests. The inputs are deterministic (`GoldenInputs`, `java.util.Random`
with fixed seeds).

Format: JSON; every `Double` value is written as a 16-character hexadecimal string
(`toRawBits()`), i.e. without loss of precision. The `generatedWith` field stores the version,
the name of the BLAS/LAPACK implementation and the generation date. Reading and writing are
done by the `GoldenIo` class without external dependencies.

The references are regenerated with `./gradlew :numerical-core:regenerateGolden` (tag
`golden-generate`; not part of the regular `test` and `fastTest` tasks). This is allowed only
on a deliberate behavior change, with the reason described in the commit message.

Check tolerances (`Golden*Test`, tag `fast`): `1e-12` relative in the norm
`‖got−exp‖∞ / max(‖exp‖∞, 1)`; `1e-6` for Hilbert matrices (`hilbert-*` in `solve.json` and
`conditioning.json`), because at n=8 the condition number is about 1e10 and different
correct LU implementations diverge at that level; `pure.json` is compared bit for bit
(`assertBits`), `null == null`.

The `inversionResidual` field is compared by order of magnitude (or both values lie below the
noise level `100·ε·condInf`) and by the `isReliable` flag rather than relatively: on
ill-conditioned matrices this quantity is rounding noise that depends on the LAPACK
implementation and the order of operations.

The `orders` reference for the input containing an infinity was updated in 1.0.0: an infinite
error yields `NaN`, as a zero one does.
