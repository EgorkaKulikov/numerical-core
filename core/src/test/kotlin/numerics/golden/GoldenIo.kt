package numerics.golden

import java.io.File
import kotlin.math.abs
import kotlin.math.max
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlin.test.fail

/**
 * Reads/writes golden references without external dependencies.
 *
 * The format is JSON; every `Double` is written as a 16-character hex string (`toRawBits()`),
 * so values are restored bit for bit. Supported values: object (`Map<String, Any?>`),
 * array (`List<Any?>`), string, `Int`, `Boolean`, `null`.
 */
object GoldenIo {

    // ---- Number encoding --------------------------------------------------------

    fun dbl(v: Double): String = "%016x".format(v.toRawBits())

    fun toDbl(s: Any?): Double = Double.fromBits(java.lang.Long.parseUnsignedLong(s as String, 16))

    fun vec(v: DoubleArray): List<String> = v.map { dbl(it) }

    fun toVec(x: Any?): DoubleArray = (x as List<*>).map { toDbl(it) }.toDoubleArray()

    fun mat(a: Array<DoubleArray>): List<List<String>> = a.map { vec(it) }

    fun toMat(x: Any?): Array<DoubleArray> = (x as List<*>).map { toVec(it) }.toTypedArray()

    /** `null` stays `null`; otherwise a hex string. */
    fun dblOrNull(v: Double?): String? = v?.let { dbl(it) }

    fun toDblOrNull(x: Any?): Double? = if (x == null) null else toDbl(x)

    @Suppress("UNCHECKED_CAST")
    fun obj(x: Any?): Map<String, Any?> = x as Map<String, Any?>

    /** Extracts a nested object by key, failing with a clear message if the key is absent. */
    fun section(root: Map<String, Any?>, key: String): Map<String, Any?> =
        obj(root[key] ?: fail("Golden reference has no section '$key'"))

    fun entry(section: Map<String, Any?>, key: String): Any? {
        if (!section.containsKey(key)) fail("Golden reference has no case '$key'; run ./gradlew regenerateGolden")
        return section[key]
    }

    // ---- Files ------------------------------------------------------------------

    fun writeGolden(dir: File, name: String, root: Map<String, Any?>) {
        dir.mkdirs()
        val sb = StringBuilder()
        writeValue(sb, root, 0)
        sb.append('\n')
        File(dir, name).writeText(sb.toString())
    }

    fun readGolden(name: String): Map<String, Any?> {
        val stream = GoldenIo::class.java.getResourceAsStream("/golden/$name")
            ?: fail("Golden reference /golden/$name not found in resources; run ./gradlew regenerateGolden")
        val text = stream.bufferedReader(Charsets.UTF_8).use { it.readText() }
        return obj(Parser(text).parseDocument())
    }

    // ---- Comparisons ------------------------------------------------------------

    /**
     * `‖got − expected‖∞ / max(‖expected‖∞, 1) ≤ tol`. Bitwise-identical elements
     * (including ±Inf and NaN) count as zero discrepancy.
     */
    fun assertClose(expected: DoubleArray, got: DoubleArray, tol: Double, label: String) {
        assertEquals(expected.size, got.size, "$label: size")
        val rel = relativeDiff(expected, got, label)
        assertTrue(rel <= tol, "$label: relative discrepancy rel=$rel > tol=$tol")
    }

    /** Relative discrepancy in the same norm, without asserting (for diagnostics). */
    fun relativeDiff(expected: DoubleArray, got: DoubleArray, label: String): Double {
        var maxDiff = 0.0
        var maxExp = 0.0
        for (i in expected.indices) {
            val e = expected[i]
            val g = got[i]
            if (e.isFinite()) maxExp = max(maxExp, abs(e))
            if (e.toBits() == g.toBits()) continue
            val d = abs(g - e)
            if (d.isNaN()) fail("$label[$i]: expected $e, got $g")
            maxDiff = max(maxDiff, d)
        }
        return maxDiff / max(maxExp, 1.0)
    }

    fun assertClose(expected: Array<DoubleArray>, got: Array<DoubleArray>, tol: Double, label: String) {
        assertEquals(expected.size, got.size, "$label: row count")
        for (i in expected.indices) {
            assertEquals(expected[i].size, got[i].size, "$label: length of row $i")
        }
        val e = expected.flatMap { it.asList() }.toDoubleArray()
        val g = got.flatMap { it.asList() }.toDoubleArray()
        assertClose(e, g, tol, label)
    }

    fun assertClose(expected: Double, got: Double, tol: Double, label: String) =
        assertClose(doubleArrayOf(expected), doubleArrayOf(got), tol, label)

    /**
     * Compares the inversion residual as rounding noise rather than as a measurement:
     * accepted if both values are at most 1e-12, or agree in order of magnitude
     * (`max <= 10 * max(min, 1e-300)`), or both stay below the noise level `100 * eps * condInf`
     * for a matrix with the given condition number. A relative comparison is meaningless here:
     * the value depends on the operation order of the particular LAPACK implementation.
     */
    fun assertNoiseLevel(expected: Double, got: Double, condInf: Double, label: String) {
        val hi = maxOf(expected, got)
        val lo = minOf(expected, got)
        val noise = 100.0 * Math.ulp(1.0) * condInf
        val ok = hi <= 1e-12 || hi <= 10.0 * maxOf(lo, 1e-300) || hi <= noise
        assertTrue(
            ok,
            "$label: discrepancy exceeds an order of magnitude and the noise level $noise: expected $expected, got $got",
        )
    }

    /** Bitwise equality (`toBits()`: all NaNs are considered equal to each other). */
    fun assertBits(expected: Double, got: Double, label: String) {
        assertTrue(
            expected.toBits() == got.toBits(),
            "$label: expected $expected (${dbl(expected)}), got $got (${dbl(got)})",
        )
    }

    fun assertBitsOrNull(expected: Double?, got: Double?, label: String) {
        if (expected == null || got == null) {
            assertEquals(expected, got, "$label: nullness")
        } else {
            assertBits(expected, got, label)
        }
    }

    // ---- Writer ----------------------------------------------------------------

    private fun indent(sb: StringBuilder, level: Int) {
        repeat(level) { sb.append("  ") }
    }

    private fun writeString(sb: StringBuilder, s: String) {
        sb.append('"')
        for (c in s) {
            when (c) {
                '"' -> sb.append("\\\"")
                '\\' -> sb.append("\\\\")
                '\n' -> sb.append("\\n")
                '\r' -> sb.append("\\r")
                '\t' -> sb.append("\\t")
                else -> if (c < ' ') sb.append("\\u%04x".format(c.code)) else sb.append(c)
            }
        }
        sb.append('"')
    }

    private fun writeValue(sb: StringBuilder, v: Any?, level: Int) {
        when (v) {
            null -> sb.append("null")
            is Boolean -> sb.append(v.toString())
            is Int -> sb.append(v.toString())
            is Long -> sb.append(v.toString())
            is String -> writeString(sb, v)
            is Map<*, *> -> {
                if (v.isEmpty()) {
                    sb.append("{}")
                    return
                }
                sb.append("{\n")
                var first = true
                for ((k, value) in v) {
                    if (!first) sb.append(",\n")
                    first = false
                    indent(sb, level + 1)
                    writeString(sb, k as String)
                    sb.append(": ")
                    writeValue(sb, value, level + 1)
                }
                sb.append('\n')
                indent(sb, level)
                sb.append('}')
            }
            is List<*> -> {
                if (v.isEmpty()) {
                    sb.append("[]")
                    return
                }
                // A list of scalars goes on one line, a list of containers one element per line.
                val scalar = v.all { it == null || it is String || it is Int || it is Long || it is Boolean }
                if (scalar) {
                    sb.append('[')
                    v.forEachIndexed { i, e ->
                        if (i > 0) sb.append(", ")
                        writeValue(sb, e, level + 1)
                    }
                    sb.append(']')
                } else {
                    sb.append("[\n")
                    v.forEachIndexed { i, e ->
                        if (i > 0) sb.append(",\n")
                        indent(sb, level + 1)
                        writeValue(sb, e, level + 1)
                    }
                    sb.append('\n')
                    indent(sb, level)
                    sb.append(']')
                }
            }
            is Double -> error("Double must be encoded via dbl(): $v")
            else -> error("Unsupported type in golden JSON: ${v::class}")
        }
    }

    // ---- Parser (recursive descent) -------------------------------------------

    private class Parser(private val s: String) {
        private var i = 0

        fun parseDocument(): Any? {
            val v = parseValue()
            skipWs()
            if (i != s.length) error("Trailing characters at position $i")
            return v
        }

        private fun skipWs() {
            while (i < s.length && s[i].isWhitespace()) i++
        }

        private fun expect(c: Char) {
            skipWs()
            if (i >= s.length || s[i] != c) error("Expected '$c' at position $i")
            i++
        }

        private fun parseValue(): Any? {
            skipWs()
            if (i >= s.length) error("Unexpected end of document")
            return when (val c = s[i]) {
                '{' -> parseObject()
                '[' -> parseArray()
                '"' -> parseString()
                't' -> literal("true", true)
                'f' -> literal("false", false)
                'n' -> literal("null", null)
                else -> if (c == '-' || c.isDigit()) parseNumber() else error("Unexpected character '$c' at position $i")
            }
        }

        private fun literal(word: String, value: Any?): Any? {
            if (!s.startsWith(word, i)) error("Expected '$word' at position $i")
            i += word.length
            return value
        }

        private fun parseNumber(): Any {
            val start = i
            if (s[i] == '-') i++
            while (i < s.length && s[i].isDigit()) i++
            val text = s.substring(start, i)
            return text.toIntOrNull() ?: text.toLong()
        }

        private fun parseString(): String {
            expect('"')
            val sb = StringBuilder()
            while (true) {
                if (i >= s.length) error("Unterminated string")
                val c = s[i++]
                when (c) {
                    '"' -> return sb.toString()
                    '\\' -> {
                        val e = s[i++]
                        when (e) {
                            '"' -> sb.append('"')
                            '\\' -> sb.append('\\')
                            '/' -> sb.append('/')
                            'n' -> sb.append('\n')
                            'r' -> sb.append('\r')
                            't' -> sb.append('\t')
                            'b' -> sb.append('\b')
                            'f' -> sb.append('\u000C')
                            'u' -> {
                                sb.append(s.substring(i, i + 4).toInt(16).toChar())
                                i += 4
                            }
                            else -> error("Unknown escape sequence \\$e")
                        }
                    }
                    else -> sb.append(c)
                }
            }
        }

        private fun parseArray(): List<Any?> {
            expect('[')
            val out = ArrayList<Any?>()
            skipWs()
            if (s[i] == ']') {
                i++
                return out
            }
            while (true) {
                out.add(parseValue())
                skipWs()
                when (s[i]) {
                    ',' -> i++
                    ']' -> {
                        i++
                        return out
                    }
                    else -> error("Expected ',' or ']' at position $i")
                }
            }
        }

        private fun parseObject(): Map<String, Any?> {
            expect('{')
            val out = LinkedHashMap<String, Any?>()
            skipWs()
            if (s[i] == '}') {
                i++
                return out
            }
            while (true) {
                skipWs()
                val key = parseString()
                expect(':')
                out[key] = parseValue()
                skipWs()
                when (s[i]) {
                    ',' -> i++
                    '}' -> {
                        i++
                        return out
                    }
                    else -> error("Expected ',' or '}' at position $i")
                }
            }
        }
    }
}
