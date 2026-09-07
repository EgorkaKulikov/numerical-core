# Источники численных методов: numerical-core

Документ фиксирует происхождение каждого алгоритма библиотеки. Все методы здесь —
классические, их первичные источники — учебники численного анализа; ссылки даны на
издания, по которым сверялись формулы. Специфические для минимальных сплайнов и
интегральных уравнений источники ведутся в соседних репозиториях (см. конец файла).

## Условные обозначения статуса

- **Классика** — метод описан в учебной литературе; реализация сверена с формулами
  источника и тестами модуля.
- **Собственное решение** — инженерное решение проекта без внешнего источника;
  обоснование — в KDoc указанного файла.

## 1. Квадратура

| Элемент | Реализация | Источник | Статус |
|---|---|---|---|
| Узлы и веса Гаусса–Лежандра на `[-1, 1]`: нули многочлена Лежандра `P_m` методом Ньютона от начального приближения `cos(pi (i + 3/4)/(m + 1/2))`, веса `2 / ((1 - x^2) P_m'(x)^2)` | `GaussLegendre.gaussLegendreReference` (`numerics/Quadrature.kt`) | [Davis, Rabinowitz 1984], гл. 2; [Stoer, Bulirsch 2002], §3.6 | Классика |
| Составная квадратура по разбиению отрезка с аффинным переносом узлов на каждый подынтервал | `GaussLegendre.integrate` | [Davis, Rabinowitz 1984], гл. 2 | Классика |
| Точность на многочленах степени `2m - 1` | проверяется `QuadratureTest.exactOnPolynomialsUpToDegree2mMinus1` | [Stoer, Bulirsch 2002], теорема 3.6.12 | Классика |

## 2. Плотная линейная алгебра

| Элемент | Реализация | Источник | Статус |
|---|---|---|---|
| LU-разложение с частичным выбором ведущего элемента, прямая и обратная подстановка | `ReferenceLinearAlgebra.solve` (`numerics/ReferenceLinearAlgebra.kt`) | [Golub, Van Loan 2013], §3.4 | Классика |
| Решение СЛАУ нативным LAPACK через multik/OpenBLAS | `MultikCpuBackend.solve` (`numerics/backend/MultikCpuBackend.kt`) | документация multik / LAPACK `dgesv` | Классика |
| Постпроверка решения по относительной невязке `‖Ax − b‖∞ / max(‖A‖∞‖x‖∞, ‖b‖∞)` и порог `1e-10` | `LinearAlgebra.SINGULARITY_RELATIVE_TOLERANCE` (`numerics/LinearAlgebra.kt`) | обратная устойчивость LU: [Higham 2002], гл. 9; выбор порога — KDoc константы | Собственное решение |
| Относительный порог вырожденности ведущего элемента `1e-14 · ‖A‖` (`PIVOT_RELATIVE_TOLERANCE`) в эталонной реализации | `ReferenceLinearAlgebra.solve` | KDoc метода; тест `LinearAlgebraRegressionTest.defect4_*` | Собственное решение |
| Разложение Холецкого (проверка положительной определённости) | `LinearAlgebra.cholesky` (`numerics/DenseOps.kt`) | [Golub, Van Loan 2013], §4.2 | Классика |

## 3. Обусловленность и оценка ошибки

| Элемент | Реализация | Источник | Статус |
|---|---|---|---|
| Обратная ошибка `ω = ‖Ax − b‖∞ / max(‖A‖∞‖x‖∞, ‖b‖∞)` и граница прямой ошибки `‖x − x*‖/‖x*‖ <= cond∞(A) · ω` | `Conditioning.relativeBackwardError`, `Conditioning.forwardError` (`numerics/Conditioning.kt`) | [Higham 2002], гл. 7 (нормовая обратная ошибка Ригала–Гаше) | Классика |
| `cond∞(A) = ‖A‖∞ ‖A⁻¹‖∞` через явное обращение; невязка обращения `‖A A⁻¹ − I‖∞` как признак достоверности | `Conditioning.conditionInf`, `Conditioning.inverse`, `Conditioning.inversionResidual` | [Golub, Van Loan 2013], §2.6; критерий достоверности — собственное решение (KDoc `ConditionEstimate`) | Классика / собственное решение |
| Собственные значения симметричной матрицы методом циклических вращений Якоби; спектральное число обусловленности | `Conditioning.symmetricEigenvalues`, `Conditioning.conditionSymmetric` | [Golub, Van Loan 2013], §8.5 | Классика |
| Закрытый тип `ForwardError` с режимами `Bounded` / `NoFiniteBound` / `Unreliable` | `numerics/Conditioning.kt` | — | Собственное решение: недостоверная оценка не может быть напечатана как число |

## 4. Инфраструктура

| Элемент | Реализация | Источник | Статус |
|---|---|---|---|
| Порог достоверности величин `1e-13` и запрет арифметики на шуме | `numerics/Reliability.kt` | — | Собственное решение (обоснование в KDoc `MACHINE_NOISE_THRESHOLD`) |
| Порядок сходимости `log2(E_h / E_{h/2})` | `orders` (`numerics/ConvergenceRates.kt`) | стандартное определение наблюдаемого порядка | Классика |
| Параллельная сборка матриц по строкам через `IntStream.parallel()` | `numerics/ParallelAssembly.kt` | — | Собственное решение |
| SPI бэкендов линейной алгебры с автоматическим откатом | `numerics/backend/*` | — | Собственное решение (см. `docs/HPC.md`) |

## Список литературы

1. **[Davis, Rabinowitz 1984]** Davis P. J., Rabinowitz P. *Methods of Numerical
   Integration.* — 2nd ed. — Academic Press, 1984.

2. **[Stoer, Bulirsch 2002]** Stoer J., Bulirsch R. *Introduction to Numerical
   Analysis.* — 3rd ed. — Springer, 2002. (Texts in Applied Mathematics, Vol. 12.)

3. **[Golub, Van Loan 2013]** Golub G. H., Van Loan C. F. *Matrix Computations.* —
   4th ed. — Johns Hopkins University Press, 2013.

4. **[Higham 2002]** Higham N. J. *Accuracy and Stability of Numerical Algorithms.* —
   2nd ed. — SIAM, 2002.

## Источники соседних репозиториев


- Минимальные сплайны, порождающие системы, аппроксимационные функционалы:
  `minimal-splines/docs/REFERENCES.md`
  (<https://github.com/EgorkaKulikov/minimal-splines/blob/main/docs/REFERENCES.md>).
- Схемы решения интегральных уравнений, регуляризация, сверка с публикациями:
  `integral-equations/docs/REFERENCES.md`
  (<https://github.com/EgorkaKulikov/integral-equations/blob/main/docs/REFERENCES.md>).
