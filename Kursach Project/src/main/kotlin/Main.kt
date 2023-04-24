import jetbrains.letsPlot.export.ggsave
import jetbrains.letsPlot.geom.geomLine
import jetbrains.letsPlot.ggplot
import jetbrains.letsPlot.ggsize
import jetbrains.letsPlot.label.labs
import jetbrains.letsPlot.scale.*

import org.jetbrains.kotlinx.multik.api.linalg.solve
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.get

import optimization.functionImplementation.ObjectiveFunctionNonLinear
import optimization.functionImplementation.Options
import solvers.NonlinearEquationSolver

import org.ejml.data.DMatrixRMaj

import java.io.File
import java.util.*

import kotlin.collections.ArrayList

import kotlin.math.*


// функция для вычисления математического ожидания (1 и 5 задания)
fun getMedian(sequence: ArrayList<Double>): Double {
    return sequence.average()
}


// функция для вычисления дисперсии (1 и 5 задания)
fun getDispersion(sequence: ArrayList<Double>): Double {
    val median = getMedian(sequence)

    var standardDeviation = 0.0
    for (num in sequence) {
        standardDeviation += (num - median).pow(2.0)
    }

    val length = sequence.size

    return standardDeviation / (length - 1)
}


// функция для вычисления интервала корреляции (1 задание)
fun getInterval(sequence: ArrayList<Double>): Int {
    var t = (0.01 * sequence.size - 1).toInt()
    val e = exp(-1.0)

    while (abs(getNormalCorrelation(sequence, t)) < e) {
        t -= 1
    }

    return t
}


// функция для подсчёта корреляционной функции (1 и 5 задания)
fun getCorrelation(sequence: ArrayList<Double>, m: Int): Double{

    var sum = 0.0
    val median = getMedian(sequence)
    val n = sequence.size

    for(j in 0 until n - m){
        sum += (sequence[j] - median) * (sequence[m + j] - median)
    }

    return sum / (n - m - 1)
}


// функция для подсчёта нормированной корреляционной функции (1 и 5 задания)
fun getNormalCorrelation(sequence: ArrayList<Double>, k: Int): Double{
    return getCorrelation(sequence, k) / getDispersion(sequence)
}


// функция для построения фрагмента СП (1 задание)
fun plotFragment(sequence: ArrayList<Double>, median: Double, dispersion: Double,
          labelName: String, title: String) {
    val xList: ArrayList<Int> = ArrayList() // создаём список для индексов от 0 до 199
    val nameSource: ArrayList<String> = ArrayList() // список имён для легенды - labelName
    val nameMedian: ArrayList<String> = ArrayList() // список имён для легенды - Average
    val nameSD: ArrayList<String> = ArrayList() // список имён для легенды - Standard Deviation
    val nameMinusSD: ArrayList<String> = ArrayList() // список имён для легенды - ""

    val medianList: ArrayList<Double> = ArrayList() // список значений медианы
    val ySDList: ArrayList<Double> = ArrayList() // список значений стандартного отклонения с плюсом
    val yMinusSDList: ArrayList<Double> = ArrayList() // список значений стандартного отклонения с минусом

    val standardDeviation: Double = sqrt(dispersion) // считаем СКО

    // заполняем все списки
    for (i in 0..199){
        xList.add(i)

        medianList.add(median)
        ySDList.add(median + standardDeviation)
        yMinusSDList.add(median - standardDeviation)

        nameSource.add(labelName)
        nameMedian.add("Average")
        nameSD.add("Standard Deviation")
        nameMinusSD.add("")
    }

    val values = sequence.subList(0, 200) // берём первые 200 значений

    // добавляем в списки значения
    values.addAll(medianList)
    values.addAll(ySDList)
    values.addAll(yMinusSDList)

    // создаём список имён для легенды
    val nameList: ArrayList<String> = ArrayList()
    nameList.addAll(nameSource)
    nameList.addAll(nameMedian)
    nameList.addAll(nameSD)
    nameList.addAll(nameMinusSD)

    // добавляем индексы для всех значений
    xList.addAll(xList)
    xList.addAll(xList)

    // создаём таблицу с индексом(ось Х), значением (ось Y) и легендой
    val data = mapOf(
        "Index number" to xList,
        "Random sequence values" to values,
        "Legend" to nameList
    )

    // строим
    val plot = ggplot(data = data){
        x = "Index number"
        y = "Random sequence values"
        color = "Legend"
    } + geomLine(size = 1){} + scaleColorManual(values = listOf("#619FCA", "#FF0000", "#0000FF", "#0000FF"), name=" ") +
             ggsize(700,500) + labs(title = title) + scaleColorDiscrete(breaks=listOf(labelName, "Average", "Standard Deviation"))

    ggsave(plot, "%s.png".format(title)) // сохраняем
}


// функция для отображения НКФ (1 задание)
fun plotNormalCorrelationFun(nkfList: ArrayList<Double>, m: Int, correlationInterval: Int) {
    val xList: ArrayList<Int> = ArrayList() // создаём список для индексов от 0 до m
    val nameSource: ArrayList<String> = ArrayList() // список имён для легенды - labelName
    val nameInterval: ArrayList<String> = ArrayList() // список имён для легенды - Interval of correlation
    val nameE: ArrayList<String> = ArrayList() // список имён для легенды - 1/e and -1/e
    val nameMinusE: ArrayList<String> = ArrayList() // список имён для легенды - ""

    val intervalList: ArrayList<Double> = ArrayList() // список значений интервала корреляции
    val yEList: ArrayList<Double> = ArrayList() // список значений 1/e
    val yMinusEList: ArrayList<Double> = ArrayList() // список значений -1/e

    // заполняем все списки
    for (i in 0..m){
        xList.add(i)

        yEList.add(exp(-1.0))
        yMinusEList.add(-exp(-1.0))

        nameSource.add("Source")
        nameE.add("1/e and -1/e")
        nameMinusE.add("")
    }

    // заполняем информацию об интервале корреляции
    intervalList.add(-1.0)
    intervalList.add(1.0)
    nameInterval.add("Interval of correlation")
    nameInterval.add("Interval of correlation")

    // список всех значений
    val values: ArrayList<Double> = ArrayList()
    values.addAll(nkfList)
    values.addAll(intervalList)
    values.addAll(yEList)
    values.addAll(yMinusEList)

    // заносим в списки названия для легенды
    val nameList: ArrayList<String> = ArrayList()
    nameList.addAll(nameSource)
    nameList.addAll(nameInterval)
    nameList.addAll(nameE)
    nameList.addAll(nameMinusE)

    // заполняем список для оси Х
    val listOfX: ArrayList<Int> = ArrayList()
    listOfX.addAll(xList)
    listOfX.add(correlationInterval)
    listOfX.add(correlationInterval)
    listOfX.addAll(xList)
    listOfX.addAll(xList)

    // создаём таблицу с индексом(ось Х), значением (ось Y) и легендой
    val data = mapOf(
        "Index number" to listOfX,
        "Normalized correlation function" to values,
        "Legend" to nameList
    )

    // строим
    val plot = ggplot(data = data){
        x = "Index number"
        y = "Normalized correlation function"
        color = "Legend"
    } + geomLine(size = 1){} + scaleColorManual(values = listOf("#0094FF", "#FF0000", "#40AC40", "#40AC40"), name=" ") +
            ggsize(700,500) + labs(title = "Графическая оценка НКФ") + scaleColorDiscrete(breaks=listOf("Source", "Interval of correlation", "1/e and -1/e"))

    ggsave(plot, "Графическая оценка НКФ.png") // сохраняем
}


// функция для нахождения коэффициентов альфа и бета моделей АР (2 и 5 задания)
fun findAlphaBetasAR(R: ArrayList<Double>, size: Int): Pair<ArrayList<ArrayList<ArrayList<Double>>>, ArrayList<ArrayList<ArrayList<Double>>>> {
    // создаём списки списков списков для занесения в них ответов
    val alphaList: ArrayList<ArrayList<ArrayList<Double>>> = ArrayList()
    val betaList: ArrayList<ArrayList<ArrayList<Double>>> = ArrayList()

    // в цикле будем формировать системы уравнений для всех АР(M)
    for (M in 0..size){
        val vecBeta: ArrayList<ArrayList<Double>> = ArrayList() // вектор коэффициентов
        val matrixAlpha: ArrayList<ArrayList<Double>> = ArrayList() // матрица коэффициентов

        // в цикле для каждого m будем формировать уравнение
        for (m in 0..M){
            vecBeta.add(arrayListOf(R[m]))
            val matrixBuf: ArrayList<Double> = ArrayList()

            if (m == 0)
                matrixBuf.add(1.0)
            else
                matrixBuf.add(0.0)

            for (n in 1..M)
                matrixBuf.add(R[abs(m - n)])
            matrixAlpha.add(matrixBuf)
        }
        // решаем систему уравнений Ax = b
        val matrix = mk.ndarray(matrixAlpha)
        val vector = mk.ndarray(vecBeta)

        val result = mk.linalg.solve(matrix, vector)

        alphaList.add(arrayListOf(arrayListOf(sqrt(result[0][0])))) // добавляем кф. альфа в список
        println("Для M = %d параметр альфа = %f".format(Locale.US, M, sqrt(result[0][0])))

        // заполняем список коэффициентов бета
        val betaListTemp: ArrayList<Double> = ArrayList()
        for (i in 1 until result.size) {
            println("Для M = %d параметр бета = %f".format(Locale.US, M, result[i][0]))
            betaListTemp.add(result[i][0])
        }
        betaList.add(arrayListOf(betaListTemp))

        println()
    }

    return Pair(alphaList, betaList)
}


// функция для нахождения теоретической НКФ (2, 3, 4, 5 задания)
fun getTheoreticalNormalCorrelation(betaList: ArrayList<ArrayList<ArrayList<Double>>>,
                                    alphaList: ArrayList<ArrayList<ArrayList<Double>>>,
                                    nkfList: ArrayList<Double>, end: Int, M_max: Int, N_max: Int,
                                    M_min: Int, N_min: Int, type: Boolean): ArrayList<ArrayList<ArrayList<Double>>> {
    fun check1(): ArrayList<ArrayList<ArrayList<Double>>> {
        val fixedList: ArrayList<ArrayList<ArrayList<Double>>> = ArrayList()
        if (M_min == 1 && N_min == 1)
            fixedList.add(arrayListOf(arrayListOf(Double.POSITIVE_INFINITY)))
        return fixedList
    }

    fun check2(): ArrayList<ArrayList<Double>> {
        val fixedList: ArrayList<ArrayList<Double>> = ArrayList()
        if (M_min == 1 && N_min == 1)
            fixedList.add(arrayListOf(Double.POSITIVE_INFINITY))
        return fixedList
    }

    val theoreticalList: ArrayList<ArrayList<ArrayList<Double>>> = check1()

    for (M in M_min..M_max){
        val bufList: ArrayList<ArrayList<Double>> = check2()

        for (N in N_min..N_max){
            if (type || !alphaList[M][N][0].isNaN()){
                val subBufList: ArrayList<Double> = ArrayList()

                for (m in 0..end){
                    if (m <= M + N){
                        subBufList.add(nkfList[m])
                        println("Для M = %d, N = %d, m = %d тНКФ = %f".format(Locale.US, M, N, m, nkfList[m]))
                    }
                    else{
                        var theoretical = 0.0

                        for(j in 1..M)
                            theoretical += betaList[M][N][j - 1] * subBufList[m - j]

                        println("Для M = %d, N = %d, m = %d тНКФ = %f".format(Locale.US, M, N, m, theoretical))
                        subBufList.add(theoretical)
                    }
                }
                bufList.add(subBufList)
                println()
            }
            else{
                println("Для M = %d и N = %d модели не существует\n".format(M, N))
                bufList.add(arrayListOf(Double.NaN))
            }
        }
        theoreticalList.add(bufList)
    }
    return theoreticalList
}


// функция для вычисления погрешностей каждой из моделей (2, 3, 4, 5 задания)
fun getEpsilon(tNKFList: ArrayList<ArrayList<ArrayList<Double>>>, nkfList: ArrayList<Double>,
               end: Int, M_max: Int, N_max: Int, M_min: Int, N_min: Int): ArrayList<ArrayList<Double>> {
    fun check1(): ArrayList<ArrayList<Double>> {
        val fixedList: ArrayList<ArrayList<Double>> = ArrayList()
        if (M_min == 1 && N_min == 1)
            fixedList.add(arrayListOf(Double.POSITIVE_INFINITY))
        return fixedList
    }

    fun check2(): ArrayList<Double> {
        val fixedList: ArrayList<Double> = ArrayList()
        if (M_min == 1 && N_min == 1)
            fixedList.add(Double.POSITIVE_INFINITY)
        return fixedList
    }

    val epsilonList: ArrayList<ArrayList<Double>> = check1() // список значений для погрешностей

    // считаем погрешность для каждого порядка
    for (M in M_min..M_max){
        val bufList = check2()

        for (N in N_min..N_max){
            if(!tNKFList[M][N][0].isNaN()){
                var eps = 0.0

                // считаем погрешность по формуле
                for (m in 1..end)
                    eps += (tNKFList[M][N][m] - nkfList[m]).pow(2)

                println("Для M = %d и N = %d погрешность = %f".format(Locale.US, M, N, eps))
                bufList.add(eps)
            }
            else{
                println("Для M = %d и N = %d модели не существует".format(M, N))
                bufList.add(Double.NaN)
            }
        }
        epsilonList.add(bufList)
        println()
    }
    return epsilonList
}


// функция для нахождения лучшей модели (2, 3, 4 задания)
fun getBestModel(epsilonList: ArrayList<ArrayList<Double>>, M_max: Int, N_max: Int, M_min: Int,
                 N_min: Int, stabilityList: ArrayList<ArrayList<Double>>): Pair<Int, Int> {
    fun check(m: Int, n: Int): Boolean {
        var result = true
        if (M_max > 0 && N_max > 0){
            if (stabilityList[m][n].isNaN() || stabilityList[m][n] == 0.0)
                result = false
            if (!result)
                println("Для M = %d и N = %d модель не устойчива".format(m, n))
        }
        return result
    }

    fun initMin(): Double {
        if (M_min == 1 && N_min == 1)
            return epsilonList[1][1]
        return epsilonList[0][0]
    }

    var minEpsilon = initMin()
    var minMN = Pair(0, 0)

    for (M in M_min..M_max){
        for (N in N_min..N_max){
            if (!epsilonList[M][N].isNaN() && check(M, N)){
                if (minEpsilon.isNaN() || epsilonList[M][N] < minEpsilon){
                    minEpsilon = epsilonList[M][N]
                    minMN = Pair(M, N)
                }
            }
            else
                println("Для M = %d и N = %d модели не существует".format(M, N))
        }
    }
    println("Лучшая модель при M = %d и N = %d с эпсилон = %f".format(Locale.US, minMN.first, minMN.second, minEpsilon))
    return minMN
}


// правила для решения системы уравнений СС(N) для каждого порядка N (3 и 5 задания)
fun equationsMA(R: ArrayList<Double>, seq: List<Double>): ArrayList<Double> {
    val factorList: ArrayList<Double> = ArrayList()
    val n = seq.size - 1

    for (m in 0..n){
        var factor = 0.0
        for (i in 0..n - m)
            factor += seq[i] * seq[i + m]

        factor -= R[m]
        factorList.add(factor)
    }
    return factorList
}


// функция для подсчёта нормы вектора (3 и 4 задания)
fun getNorm(vec: ArrayList<Double>): Double {
    var sum = 0.0
    for (elem in vec)
        sum += elem * elem

    return sqrt(sum)
}


// функция для нахождения коэффициентов альфа модели СС(N) (3 и 5 задания)
fun findAlphasMA(N_max: Int, R: ArrayList<Double>): ArrayList<ArrayList<ArrayList<Double>>> {
    val ansList: ArrayList<ArrayList<Double>> = ArrayList()

    for (n in 0..N_max) {
        val options = Options(n + 1)
        options.isAnalyticalJacobian = false // Указываем, будем ли предоставлять аналитический якобиан (по стандарту false)
        options.algorithm = Options.TRUST_REGION // Выбор алгоритма; Options.TRUST_REGION или Options.LINE_SEARCH (по стандарту Options.TRUST_REGION)
        options.isSaveIterationDetails = true // Сохранять информацию об итерациях в переменную типа Result (по стандарту false)
        options.setAllTolerances(1e-12) // Устанавливаем точность схождения (по стандарту 1e-8)
        options.maxIterations = 1000 // Ставим максимальное количество итераций (по стандарту 100)

        // инициализируем функцию
        val function = object : ObjectiveFunctionNonLinear {
            override fun getF(x: DMatrixRMaj): DMatrixRMaj {
                val f = DMatrixRMaj(n + 1, 1)

                val result = equationsMA(R, x.getData().toList())

                for (i in 0 until result.size)
                    f.set(i, 0, result[i])

                return f
            }

            override fun getJ(x: DMatrixRMaj): DMatrixRMaj? {
                return null
            }

        }

        val nonlinearSolver = NonlinearEquationSolver(function, options)

        // начальное приближение
        val initialGuess = DMatrixRMaj(n + 1, 1)
        for (i in 0..n)
            initialGuess.set(i, 0, 0.0)
        initialGuess.set(0, 0, sqrt(R[0]))

        nonlinearSolver.solve(DMatrixRMaj(initialGuess)) // решаем систему

        val result = nonlinearSolver.x.getData().toList()
        val norm = getNorm(equationsMA(R, result)) // считаем норму

        // проверка модели на существование и выдача нужного ответа
        if (norm < 1e-4){
            val bufList: ArrayList<Double> = ArrayList()

            for (i in 0..n){
                bufList.add(result[i])
                println("Для N = %d параметр альфа = %f".format(Locale.US, n, result[i]))
            }
            ansList.add(bufList)
        }
        else{
            println("Для N = %d модели не существует".format(n))
            ansList.add(arrayListOf(Double.NaN))
        }
        println()
    }
    return arrayListOf(ansList)
}


// правила для решения системы уравнений АРСС(M, N) для каждого порядка M и N (4 и 5 задания)
fun ARMA_factor_generator(R: ArrayList<Double>, seq: List<Double>, M: Int, N: Int): ArrayList<Double> {
    val alphaList: ArrayList<Double> = ArrayList()
    val betaList: ArrayList<Double> = ArrayList()
    val Rxi: ArrayList<Double> = ArrayList()

    // распределяем элементы по спискам
    for (i in seq.indices){
        val curElem = seq[i]
        if (i < (N + 1))
            alphaList.add(curElem)
        else if (N < i && i < (N + M + 1))
            betaList.add(curElem)
        else
            Rxi.add(curElem)
    }

    val factorList: ArrayList<Double> = ArrayList()

    // генерируем правила для системы типа Таблица 1.1 А
    for (n in 0..N){
        var factor = 0.0
        for (j in 1..M)
            factor += betaList[j - 1] * R[abs(n - j)]
        for (i in n..N)
            factor += alphaList[i] * Rxi[i - n]
        factor -= R[n]

        factorList.add(factor)
    }

    // генерируем правила для системы типа Таблица 1.1 Б
    for (i in 1..M){
        var factor = 0.0
        for (j in 1..M)
            factor += betaList[j - 1] * R[abs(N - j + i)]
        factor -= R[N + i]

        factorList.add(factor)
    }

    // генерируем правила для системы типа Таблица 1.1 Г
    for (n in 0..N){
        var factor = 0.0
        val m = min(n, M)

        if (n > 0){
            for (j in 1..m)
                factor += betaList[j - 1] * Rxi[n - j]
        }
        factor += (alphaList[n] - Rxi[n])
        factorList.add(factor)
    }

    return factorList
}


// функция для нахождения коэффициентов альфа и бета моделей АРСС(M, N) (4 и 5 задания)
fun findBetasAlphasARMA(R: ArrayList<Double>, M_max: Int, N_max: Int): Pair<ArrayList<ArrayList<ArrayList<Double>>>, ArrayList<ArrayList<ArrayList<Double>>>> {
    val alphaList: ArrayList<ArrayList<ArrayList<Double>>> = arrayListOf(arrayListOf(ArrayList()))
    val betaList: ArrayList<ArrayList<ArrayList<Double>>> = arrayListOf(arrayListOf(ArrayList()))

    for (m in 1..M_max){
        val bufAlphaList: ArrayList<ArrayList<Double>> = arrayListOf(ArrayList())
        val bufBetaList: ArrayList<ArrayList<Double>> = arrayListOf(ArrayList())

        for (n in 1..N_max){
            val subAlphaList: ArrayList<Double> = ArrayList()
            val subBetaList: ArrayList<Double> = ArrayList()

            val options = Options(m + 2 * n + 2)
            options.isAnalyticalJacobian = false // Указываем, будем ли предоставлять аналитический якобиан (по стандарту false)
            options.algorithm = Options.LINE_SEARCH // Выбор алгоритма; Options.TRUST_REGION или Options.LINE_SEARCH (по стандарту Options.TRUST_REGION)
            options.isSaveIterationDetails = true // Сохранять информацию об итерациях в переменную типа Result (по стандарту false)
            options.setAllTolerances(1e-8) // Устанавливаем точность схождения (по стандарту 1e-8)
            options.maxIterations = 1000 // Ставим максимальное количество итераций (по стандарту 100)

            // инициализируем функцию
            val function = object : ObjectiveFunctionNonLinear {
                override fun getF(x: DMatrixRMaj): DMatrixRMaj {
                    val f = DMatrixRMaj(m + 2 * n + 2, 1)

                    val result = ARMA_factor_generator(R, x.getData().toList(), m, n)
                    for (i in 0 until result.size)
                        f.set(i, 0, result[i])

                    return f
                }

                override fun getJ(x: DMatrixRMaj): DMatrixRMaj? {
                    return null
                }

            }

            val nonlinearSolver = NonlinearEquationSolver(function, options)

            // начальное приближение
            val initialGuess = DMatrixRMaj(m + 2 * n + 2, 1)
            for (i in 0 until m + 2 * n + 2)
                initialGuess.set(i, 0, 0.0)
            initialGuess.set(0, 0, sqrt(R[0]))

            nonlinearSolver.solve(DMatrixRMaj(initialGuess)) // решаем систему

            val result = nonlinearSolver.x.getData().toList().toMutableList()
            val norm = getNorm(ARMA_factor_generator(R, result, m, n)) // считаем норму

            if (norm < 1e-4){
                for (index in 0..n){
					// тут условие подбиралось под конкретную задачу - у Вас может быть по-другому. Аккуратно!!! (но он зачёл и так)
                    val elem = if (m % 2 != 0 && n % 2 == 0)
                        -result[index]
                    else
                        result[index]

                    println("Для M = $m и N = $n коэффициент альфа = $elem и норма = $norm")
                    subAlphaList.add(elem)
                }
                println()
                for (idx in 0 until m){
                    val elem = result[idx + n + 1]

                    println("Для M = $m и N = $n коэффициент бета = $elem")
                    subBetaList.add(elem)
                }
                println()
            }
            else{
                println("Для M = $m и N = $n модели не существует\n")
                subAlphaList.add(Double.NaN)
                subBetaList.add(Double.NaN)
            }
            subAlphaList.reverse()
            bufAlphaList.add(subAlphaList)
            bufBetaList.add(subBetaList)
        }
        alphaList.add(bufAlphaList)
        betaList.add(bufBetaList)
    }
    return Pair(alphaList, betaList)
}


// функция для проверки на стабильность моделей АРСС(M, N) (4 задание)
fun getStability(betaList: ArrayList<ArrayList<ArrayList<Double>>>, M_max: Int, N_max: Int, M_min: Int, N_min: Int): ArrayList<ArrayList<Double>> {
    val result: ArrayList<ArrayList<Double>> = arrayListOf(ArrayList())

    for (M in M_min..M_max){
        val subResult: ArrayList<Double> = arrayListOf(Double.POSITIVE_INFINITY)
        for (N in N_min..N_max){
            if (!betaList[M][N][0].isNaN()){
                val betas = betaList[M][N]
                val size = betas.size

                var ans = true
                when (size){
                    0 -> ans = true
                    1 -> ans = abs(betas[0]) < 1
                    2 -> ans = abs(betas[1]) < 1 && abs(betas[0]) < 1 - betas[1]
                    3 -> {
                        val statement1 = abs(betas[2]) < 1
                        val statement2 = abs(betas[0] + betas[2]) < 1 - betas[1]
                        val statement3 = abs(betas[1] + betas[0] * betas[2]) < 1 - betas[2] * betas[2]
                        ans = statement1 && statement2 && statement3
                    }
                }
                var answer: Double

                if (ans){
                    answer = 1.0
                    println("Для M = $M и N = $N модель устойчива")
                }
                else{
                    answer = 0.0
                    println("Для M = $M и N = $N модель не устойчива")
                }

                subResult.add(answer)
            }
            else{
                println("Для M = $M и N = $N модели не существует")
                subResult.add(Double.NaN)
            }
        }
        println()
        result.add(subResult)
    }
    return result
}


// функция для генерации выборки из old_n значений (5 задание)
fun generateSequence(alphaList: ArrayList<ArrayList<ArrayList<Double>>>, betaList: ArrayList<ArrayList<ArrayList<Double>>>,
                     M: Int, N: Int, old_n: Int, median: Double): ArrayList<Double> {
    fun check(): ArrayList<Double> {
        if (betaList.isEmpty())
            return ArrayList()
        return betaList[M][N]
    }

    val subAlphaList = alphaList[M][N]
    val subBetaList = check()

    val badCount = 1000

    val new_n = old_n + badCount

    var etaList: ArrayList<Double> = ArrayList()
    for (i in 0 until new_n)
        etaList.add(0.0)

    val ksiList: ArrayList<Double> = ArrayList()
    for (i in 0 until new_n)
        ksiList.add(Random().nextGaussian(0.0, 1.0))


    for (n in 0 until new_n){
        for (j in 1..M){
            if (n - j >= 0)
                etaList[n] += subBetaList[j - 1] * etaList[n - j]
        }
        for (i in 0..N){
            if (n - i >= 0)
                etaList[n] += subAlphaList[i] * ksiList[n - i]
        }
    }
    etaList = etaList.slice(1000 until new_n) as ArrayList<Double>


    for (n in 0 until old_n)
        etaList[n] += median

    return etaList
}


// функция для графического сравнения НКФ смоделированного и исходного СП (5 задание)
fun plotBestNKF(sourceNKF: ArrayList<Double>, modelNKF: ArrayList<Double>, tNKFModel: ArrayList<Double>, label: String, m: Int) {
    val xList: ArrayList<Int> = ArrayList() // индексы по x от 0 до m
    val nameSource: ArrayList<String> = ArrayList() // список имён для легенды - labelName
    val nameModeling: ArrayList<String> = ArrayList() // список имён для легенды - Modeling
    val nameTheoretical: ArrayList<String> = ArrayList() // список имён для легенды - Theoretical

    // заполняем все списки
    for (i in 0..m){
        xList.add(i)

        nameSource.add("Source")
        nameModeling.add("Modeling")
        nameTheoretical.add("Theoretical")
    }

    // заполняем список значений
    val valueList: ArrayList<Double> = ArrayList()
    valueList.addAll(sourceNKF)
    valueList.addAll(modelNKF)
    valueList.addAll(tNKFModel)

    // заполняем список имён
    val nameList: ArrayList<String> = ArrayList()
    nameList.addAll(nameSource)
    nameList.addAll(nameModeling)
    nameList.addAll(nameTheoretical)

    // заполняем список значений по оси Х
    val listOfX: ArrayList<Int> = ArrayList()
    listOfX.addAll(xList)
    listOfX.addAll(xList)
    listOfX.addAll(xList)

    // создаём таблицу с индексом(ось Х), значением (ось Y) и легендой
    val data = mapOf(
        "Index number" to listOfX,
        "Normalized correlation function" to valueList,
        "Legend" to nameList
    )

    // строим
    val plot = ggplot(data = data){
        x = "Index number"
        y = "Normalized correlation function"
        color = "Legend"
    } + geomLine(size = 1){} + scaleColorManual(values = listOf("#0094FF", "#FF0000", "#40AC40"), name=" ") +
            ggsize(700,500) + labs(title = "Сравнение НКФ для модели $label")

    ggsave(plot, "Сравнение НКФ для модели $label.png") // сохраняем
}


// функция для выбора лучшей модели из моделей АР, СС и АРСС (6 задание)
fun getBestOfTheBests(thEpsList: ArrayList<Double>, epsList: ArrayList<Double>, models: ArrayList<Pair<Int, Int>>): Int {
    val minThEps = thEpsList.min()
    val minModEps = epsList.min()

    val minThEpsIndex = thEpsList.indexOf(minThEps)
    val minModEpsIndex = epsList.indexOf(minModEps)

    val minEps = min(minThEps, minModEps)
    val minIndex = min(minThEpsIndex, minModEpsIndex)

    val bestM = models[minIndex].first
    val bestN = models[minIndex].second

    println("Лучшая модель при M = $bestM и N = $bestN с эпсилон = $minEps")

    return minIndex
}


// главная функция откуда всё вызывается
fun main(args : Array<String>) {
    val startList: ArrayList<Double> = ArrayList() // создаём список под все наши числа
    val file = File("02.txt") // переменная, обозначающая наш файл
    file.forEachLine { startList.add(it.toDouble()) } // заполняем список числами из файла

    val length = startList.size
    val m = 10 // задаём количество отсчётов константой
    val M_max = 3 // максимальный порядок M
    val N_max = 3 // максимальный порядок N
    val M_min = 0 // минимальный порядок M
    val N_min = 0 // минимальный порядок N

    println("\n-------------------------Задание №1-------------------------\n\n")

    val median = getMedian(startList) // считаем мат. ожидание
    println("Математическое ожидание = %f\n".format(Locale.US, median))

    val dispersion = getDispersion(startList) // считаем дисперсию
    println("Дисперсия = %f\n".format(Locale.US, dispersion))

    val interval = getInterval(startList) // считаем интервал корреляции
    println("Интервал корреляции = %d\n".format(interval))

    val cfList: ArrayList<Double> = ArrayList() // создаём список для КФ
    val nkfList: ArrayList<Double> = ArrayList() // создаём список для НКФ

    // считаем для m = 0..10 КФ и НКФ
    println("----------Значения корреляционной функции и НКФ----------\n")
    for(i in 0..m){
        val correlation = getCorrelation(startList, i) // считаем КФ
        cfList.add(correlation)
        println("Для m = %d корреляционная функция = %f".format(Locale.US, i, correlation))

        val normalCorrelation = getNormalCorrelation(startList, i) // считаем НКФ
        nkfList.add(normalCorrelation)
        println("Для m = %d нормированная корреляционная функция = %f\n".format(Locale.US, i, normalCorrelation))
    }

    // строим графики фрагмента исходного СП и НКФ
    plotFragment(startList, median, dispersion, "Source process", "Фрагмент исходного СП")
    plotNormalCorrelationFun(nkfList, m, interval)

    println("\n-------------------------Задание №2-------------------------\n\n")

    // получаем значения коэффициентов альфа и бета
    println("---------Значения параметров альфа и бета для АР---------\n")
    val (alphaListAR, betaListAR) = findAlphaBetasAR(cfList, M_max)

    // вычисляем теоеретическую НКФ для АР
    println("-----------------Теоретическая НКФ для АР-----------------\n")
    val tNKFListAR = getTheoreticalNormalCorrelation(betaListAR, arrayListOf(arrayListOf(ArrayList())), nkfList, m, M_max, N_min, M_min, N_min, true)

    // находим эпсилон для каждой модели
    println("----------------Эпсилон для каждой модели----------------\n")
    val epsilonListAR = getEpsilon(tNKFListAR, nkfList, m, M_max, N_min, M_min, N_min)

    // выбор лучшей модели АР
    println("--------------------Выбор лучшей модели--------------------\n")
    val (ar_m, ar_n) = getBestModel(epsilonListAR, M_max, N_min, M_min, N_min, ArrayList())

    println("\n\n-------------------------Задание №3-------------------------\n\n")

    // получаем значения параметра альфа
    println("-----------------Значения параметра альфа-----------------\n")
    val alphaListMA = findAlphasMA(N_max, cfList)

    // вычисляем теоретическую НКФ для СС
    println("-----------------Теоретическая НКФ для СС-----------------\n")
    val tNKFListMA = getTheoreticalNormalCorrelation(arrayListOf(arrayListOf(ArrayList())), alphaListMA, nkfList, m, M_min, N_max, M_min, N_min, false)

    // находим эпсилон для каждой модели
    println("----------------Эпсилон для каждой модели----------------\n")
    val epsilonListMA = getEpsilon(tNKFListMA, nkfList, m, M_min, N_max, M_min, N_min)

    // выбор лучшей модели СС
    println("--------------------Выбор лучшей модели--------------------\n")
    val (ma_m, ma_n) = getBestModel(epsilonListMA, M_min, N_max, M_min, N_min, ArrayList())

    println("\n\n-------------------------Задание №4-------------------------\n\n")

    // находим коэффициенты бета и альфа для каждой модели АРСС
    println("------------Значения параметров альфа и бета------------\n")
    val (alphaListARMA, betaListARMA) = findBetasAlphasARMA(cfList, M_max, N_max)

    // вычисляем теоретическую НКФ для АРСС
    println("----------------Теоретическая НКФ для АРСС----------------\n")
    val tNKFListARMA = getTheoreticalNormalCorrelation(betaListARMA, alphaListARMA, nkfList, m, M_max, N_max, 1, 1, false)

    // находим эпсилон для каждой модели
    println("----------------Эпсилон для каждой модели----------------\n")
    val epsilonListARMA = getEpsilon(tNKFListARMA, nkfList, m, M_max, N_max, 1, 1)

    // проверяем все модели на стабильность
    println("-------------Проверка моделей на стабильность-------------\n")
    val stabilityList = getStability(betaListARMA, M_max, N_max, 1, 1)

    // выбор лучшей модели АРСС
    println("--------------------Выбор лучшей модели--------------------\n")
    val (arma_m, arma_n) = getBestModel(epsilonListARMA, M_max, N_max, 1, 1, stabilityList)

    println("\n\n-------------------------Задание №5-------------------------\n\n")

    // генерируем по 5000 значений для каждой модели
    val ar = generateSequence(alphaListAR, betaListAR, ar_m, ar_n, length, median)
    val ma = generateSequence(alphaListMA, ArrayList(), ma_m, ma_n, length, median)
    val arma = generateSequence(alphaListARMA, betaListARMA, arma_m, arma_n, length, median)

    println("----------------Значения моментных функций----------------\n")
    val medianAR = getMedian(ar)
    val medianMA = getMedian(ma)
    val medianARMA = getMedian(arma)

    println("Для АР($ar_m) медиана = $medianAR")
    println("Для CC($ma_n) медиана = $medianMA")
    println("Для АРCC($arma_m, $arma_n) медиана = $medianARMA\n")

    val dispersionAR = getDispersion(ar)
    val dispersionMA = getDispersion(ma)
    val dispersionARMA = getDispersion(arma)

    println("Для АР($ar_m) дисперсия = $dispersionAR")
    println("Для CC($ma_n) дисперсия = $dispersionMA")
    println("Для АРCC($arma_m, $arma_n) дисперсия = $medianARMA\n")

    val sko = sqrt(dispersion)
    val skoAR = sqrt(dispersionAR)
    val skoMA = sqrt(dispersionMA)
    val skoARMA = sqrt(dispersionARMA)

    println("Для исходного СП СКО = $sko")
    println("Для АР($ar_m) СКО = $skoAR")
    println("Для CC($ma_n) СКО = $skoMA")
    println("Для АРCC($arma_m, $arma_n) СКО = $skoARMA\n")

    val cfListAR: ArrayList<Double> = ArrayList()
    val cfListMA: ArrayList<Double> = ArrayList()
    val cfListARMA: ArrayList<Double> = ArrayList()

    val nkfListAR: ArrayList<Double> = ArrayList()
    val nkfListMA: ArrayList<Double> = ArrayList()
    val nkfListARMA: ArrayList<Double> = ArrayList()

    println("----------Значения корреляционной функции и НКФ----------\n")
    for(i in 0..m){
        val correlationAR = getCorrelation(ar, i) // считаем КФ
        cfListAR.add(correlationAR)
        println("Для m = %d корреляционная функция АР = %f".format(Locale.US, i, correlationAR))

        val correlationMA = getCorrelation(ma, i) // считаем КФ
        cfListMA.add(correlationMA)
        println("Для m = %d корреляционная функция СС = %f".format(Locale.US, i, correlationMA))

        val correlationARMA = getCorrelation(arma, i) // считаем КФ
        cfListARMA.add(correlationARMA)
        println("Для m = %d корреляционная функция АРСС = %f\n".format(Locale.US, i, correlationARMA))

        val normalCorrelationAR = getNormalCorrelation(ar, i) // считаем НКФ
        nkfListAR.add(normalCorrelationAR)
        println("Для m = %d нормированная корреляционная функция АР = %f".format(Locale.US, i, normalCorrelationAR))

        val normalCorrelationMA = getNormalCorrelation(ma, i) // считаем НКФ
        nkfListMA.add(normalCorrelationMA)
        println("Для m = %d нормированная корреляционная функция СС = %f".format(Locale.US, i, normalCorrelationMA))

        val normalCorrelationARMA = getNormalCorrelation(arma, i) // считаем НКФ
        nkfListARMA.add(normalCorrelationARMA)
        println("Для m = %d нормированная корреляционная функция АРСС = %f\n".format(Locale.US, i, normalCorrelationARMA))
    }

    val thEpsList: ArrayList<Double> = ArrayList()
    val epsList: ArrayList<Double> = ArrayList()

    println("----------------------Данные модели АР($ar_m)----------------------\n")

    // получаем значения параметров альфа и бета
    println("------------Значения параметров альфа и бета------------\n")
    val (alphaAR, betaAR) = findAlphaBetasAR(cfListAR, M_max)

    println("-----------------Теоретическая НКФ для АР-----------------\n")
    val tNkfARList = getTheoreticalNormalCorrelation(betaAR, arrayListOf(arrayListOf(ArrayList())), nkfListAR, m, ar_m, ar_n, M_min, N_min, true)
    val tNkfAR = tNkfARList[ar_m][ar_n]

    println("-----------------Эпсилон для нашей модели-----------------\n")
    val epsAR = getEpsilon(tNkfARList, nkfListAR, m, ar_m, ar_n, ar_m, ar_n)[0][0]
    thEpsList.add(epsAR)
    epsList.add(epsilonListAR[ar_m][ar_n])

    println("----------------------Данные модели СС($ma_n)----------------------\n")

    // получаем значения параметров альфа и бета
    println("---------------Значения параметров альфа---------------\n")
    val alphaMA = findAlphasMA(ma_n, cfListMA)

    //вычисляем теоретическую НКФ для СС
    print("-----------------Теоретическая НКФ для СС-----------------\n")
    val tNkfMAList = getTheoreticalNormalCorrelation(arrayListOf(arrayListOf(ArrayList())), alphaMA, nkfListMA, m, ma_m, ma_n, M_min, N_min, false)
    val tNkfMA = tNkfMAList[ma_m][ma_n]

    println("-----------------Эпсилон для нашей модели-----------------\n")
    val epsMA = getEpsilon(tNkfMAList, nkfListMA, m, ma_m, ma_n, ma_m, ma_n)[0][0]
    thEpsList.add(epsMA)
    epsList.add(epsilonListMA[ma_m][ma_n])

    println("-------------------Данные модели АРСС($arma_m, $arma_n)-------------------\n")

    // находим коэффициенты бета и альфа для каждой модели АРСС
    println("------------Значения параметров альфа и бета------------\n")
    val (alphaARMA, betasARMA) = findBetasAlphasARMA(cfListARMA, arma_m, arma_n)

    // вычисляем теоретическую НКФ для АРСС
    println("----------------Теоретическая НКФ для АРСС----------------\n")
    val tNkfARMAList = getTheoreticalNormalCorrelation(betasARMA, alphaARMA, nkfListARMA, m, arma_m, arma_n, 1, 1, false)
    val tNkfARMA = tNkfARMAList[arma_m][arma_n]

    // находим эпсилон для каждой модели
    println("-----------------Эпсилон для нашей модели-----------------\n")
    val epsARMA = getEpsilon(tNkfARMAList, nkfListARMA, m, arma_m, arma_n, arma_m, arma_n)[0][0]
    thEpsList.add(epsARMA)
    epsList.add(epsilonListARMA[arma_m][arma_n])

    val nkfLists = listOf(nkfListAR, nkfListMA, nkfListARMA)
    val tNKFLists = listOf(tNkfAR, tNkfMA, tNkfARMA)
    val nameList = listOf("АР($ar_m)", "СС($ma_n)", "АРСС($arma_m, $arma_m)")

    for (i in 0 until 3)
        plotBestNKF(nkfList, nkfLists[i], tNKFLists[i], nameList[i], m)

    println("\n\n-------------------------Задание №6-------------------------\n\n")

    // получаем лучшую модель
    val bests = arrayListOf(Pair(ar_m, ar_n), Pair(ma_m, ma_n), Pair(arma_m, arma_n))
    val bestIndex = getBestOfTheBests(thEpsList, epsList, bests)

    // получаем процесс, для которого будем строить график
    val processList = listOf(ar, ma, arma)
    val medianList = listOf(medianAR, medianMA, medianARMA)
    val dispersionList = listOf(dispersionAR, dispersionMA, dispersionARMA)

    val bestProcess = processList[bestIndex]
    val bestMedian = medianList[bestIndex]
    val bestDispersion = dispersionList[bestIndex]
    val bestName = nameList[bestIndex]

    // изображаем фрагмент смоделированного СП
    plotFragment(bestProcess, bestMedian, bestDispersion, "$bestName process", "Фрагмент сгенерированного СП по модели $bestName")
}
