package numerics

import numerics.backend.Backends
import numerics.backend.LinAlgBackend

/**
 * Computation settings: the linear algebra backend, whether parallel assembly is allowed, and the
 * degree of parallelism. An immutable value; passed to every object of one problem so that the
 * results are consistent.
 *
 * Compared by value (`data class`): two independently created contexts with the same
 * parameters describe the same configuration and are considered compatible.
 *
 * @param backend linear algebra backend
 * @param parallel whether parallel assembly is allowed ([ParallelAssembly])
 * @param parallelism number of threads for parallel assembly (at least 1)
 */
public data class NumericsContext(
    val backend: LinAlgBackend = Backends.default(),
    val parallel: Boolean = true,
    val parallelism: Int = Runtime.getRuntime().availableProcessors(),
) {
    init {
        require(parallelism >= 1) { "Parallelism must be at least 1, got $parallelism" }
    }

    /** Default context and context compatibility check. */
    public companion object {
        /**
         * Shared default context instance: one per process rather than one per call
         * with a default parameter. `lazy` because [Backends.default] may throw,
         * and an exception from a class initializer degenerates into a
         * `NoClassDefFoundError` without the cause message on subsequent accesses.
         */
        private val shared: Lazy<NumericsContext> = lazy { NumericsContext() }

        /** Default context: the default linear algebra backend with parallel assembly enabled. */
        public fun default(): NumericsContext = shared.value

        /**
         * Requires that a dependency uses the same context as its owner. A context mismatch
         * means that parts of one problem would be computed by different linear algebra
         * backends, which silently shifts the low-order bits of the result.
         *
         * @param owner name of the owning class for the error message
         * @param expected the owner's context
         * @param dependency name of the dependency parameter
         * @param actual the dependency's context
         * @throws IllegalArgumentException if the contexts differ
         */
        public fun requireSame(
            owner: String,
            expected: NumericsContext,
            dependency: String,
            actual: NumericsContext,
        ) {
            require(expected == actual) {
                "Numerics context of '$dependency' (${actual.describe()}) does not match the context " +
                    "of '$owner' (${expected.describe()}). Pass the same NumericsContext " +
                    "to all related objects"
            }
        }
    }

    /** Short description for error messages. */
    public fun describe(): String = "backend=${backend.name}, parallel=$parallel, parallelism=$parallelism"
}
