package numerics.backend

import numerics.ReferenceLinearAlgebra

/**
 * Эталонный бэкенд линейной алгебры на чистом JVM (без нативных зависимостей).
 *
 * Делегирует тяжёлые операции «оракулу корректности»
 * [ReferenceLinearAlgebra]. Это полноценный второй бэкенд: он доказывает
 * работоспособность SPI и служит CPU-only fallback'ом, когда нативный
 * OpenBLAS недоступен. Доступен всегда.
 */
object ReferenceBackend : LinAlgBackend {

    override val name: String = "reference (pure-JVM)"

    /** Чистый JVM без нативных зависимостей — доступен на любой машине. */
    override fun isAvailable(): Boolean = true

    override fun matVec(a: Array<DoubleArray>, x: DoubleArray): DoubleArray =
        ReferenceLinearAlgebra.matVec(a, x)

    override fun matTransVec(a: Array<DoubleArray>, y: DoubleArray): DoubleArray =
        ReferenceLinearAlgebra.matTransVec(a, y)

    override fun matMat(a: Array<DoubleArray>, b: Array<DoubleArray>): Array<DoubleArray> =
        ReferenceLinearAlgebra.matMat(a, b)

    override fun atWa(a: Array<DoubleArray>, w: DoubleArray): Array<DoubleArray> =
        ReferenceLinearAlgebra.atWa(a, w)

    override fun addScaled(a: Array<DoubleArray>, b: Array<DoubleArray>, s: Double): Array<DoubleArray> =
        ReferenceLinearAlgebra.addScaled(a, b, s)

    /**
     * Решение СЛАУ методом LU с частичным выбором ведущего элемента.
     * @throws IllegalStateException при вырожденности.
     */
    override fun solve(a: Array<DoubleArray>, b: DoubleArray): DoubleArray =
        ReferenceLinearAlgebra.solve(a, b)
}
