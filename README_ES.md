# Estrategia de Pairs Trading con Filtros NLMS
### Estimación Adaptativa del Hedge Ratio con Filtros VS-NLMS de Paso Variable y Predicción de μ con Machine Learning

**Autor:** Ignacio Blasi Sanchiz

---

## Resumen

Este proyecto implementa un **sistema de arbitraje estadístico por pares** utilizando filtros adaptativos de procesamiento digital de señales para estimar un hedge ratio variable en el tiempo. La innovación principal es la aplicación del filtro **VS-NLMS de paso variable** (Variable Step-Size NLMS) — una clase de algoritmos de la teoría de filtros adaptativos — al problema de estimación dinámica de cointegración en series temporales financieras.

El sistema se valida con una **optimización walk-forward** sobre 16 años de datos diarios (2010–2026), empleando una cascada de tres filtros previos al trading (cointegración de Johansen, half-life, umbral de Sharpe en train) para garantizar que solo se operan periodos con base estructural sólida.

Una línea de investigación secundaria estudia si se puede sustituir la regla heurística de adaptación de μ por una **red neuronal entrenada sobre una competición de filtros en paralelo**. Se han evaluado dos arquitecturas: `ML_VSNLMSFilter` (MLP, sin memoria temporal) y `GRU_VSNLMSFilter` (GRU con BPTT en numpy puro, con estado oculto h[t]). El MLP supera al baseline en pares con señal moderada; el GRU añade valor en pares con dependencias temporales (DBS/UOB, GS/MS).

---

## Índice

1. [Formulación del Problema](#1-formulación-del-problema)
2. [Filtro de Cointegración en Tres Etapas](#2-filtro-de-cointegración-en-tres-etapas)
3. [Teoría de Filtros Adaptativos](#3-teoría-de-filtros-adaptativos)
4. [El Filtro VS-NLMS en Detalle](#4-el-filtro-vs-nlms-en-detalle)
5. [Estrategia de Trading](#5-estrategia-de-trading)
6. [Metodología de Validación Walk-Forward](#6-metodología-de-validación-walk-forward)
7. [Resultados del Escaneo del Universo](#7-resultados-del-escaneo-del-universo)
8. [Mejores Pares — Resultados Detallados](#8-mejores-pares--resultados-detallados)
9. [Universo Macro — FX, Bonos, Materias Primas e Índices](#9-universo-macro--fx-bonos-materias-primas-e-índices)
10. [Experimentos de Machine Learning](#10-experimentos-de-machine-learning) — MLP · GRU · Flags de ejecución
11. [Cómo Usar la Estrategia en la Práctica](#11-cómo-usar-la-estrategia-en-la-práctica)
12. [Estructura del Proyecto](#12-estructura-del-proyecto)
13. [Instalación y Uso](#13-instalación-y-uso)
14. [Construcción de Portfolio y Resultados](#14-construcción-de-portfolio-y-resultados)
15. [Referencias](#15-referencias)

---

## 1. Formulación del Problema

El pairs trading explota una idea fundamental: dos empresas del mismo sector comparten los mismos factores de riesgo macroeconómico. Cuando ambas acciones están impulsadas por los mismos fundamentos económicos, sus precios deberían moverse juntos a largo plazo. Las divergencias temporales de ese equilibrio representan ineficiencias de mercado explotables.

Formalmente, modelamos la relación como:

```
precio_Y[t] = β[t] · precio_X[t] + e[t]
```

donde:
- `β[t]` es el **hedge ratio** — cuántos dólares de X necesitamos para cubrir un dólar de Y
- `e[t]` es el **spread** — el residuo que debería ser estacionario si el par está cointegrado
- El subíndice `[t]` en β indica que permitimos que el ratio **cambie con el tiempo**

El reto principal es estimar `β[t]` en tiempo real sin sesgo de anticipación (lookahead bias). Aquí es donde entran los filtros adaptativos.

**¿Por qué no usar mínimos cuadrados ordinarios (OLS)?** OLS calcula un único β fijo sobre toda la muestra. Pero las relaciones de cointegración derivan con el tiempo. Un β estático produce un spread que solo es localmente estacionario, generando señales falsas cuando β ha cambiado.

---

## 2. Filtro de Cointegración en Tres Etapas

Antes de operar cualquier fold, deben superarse tres filtros secuenciales. Si alguno falla, el fold se salta por completo.

### Etapa 1 — Test de Cointegración de Johansen (95% de confianza)

Usamos el **test de máxima verosimilitud de Johansen (1988)** en lugar del procedimiento de dos pasos de Engle-Granger. Ventajas:

- **Simetría**: no requiere elegir cuál serie es la "dependiente"
- **Mayor potencia estadística** en muestras finitas
- Estima directamente el número de vectores de cointegración

El estadístico de la traza contrasta H₀: r = 0 (ningún vector de cointegración) frente a H₁: r ≥ 1. Rechazamos H₀ al 5% de significancia si:

```
λ_traza = −T · Σᵢ log(1 − λ̂ᵢ)  >  valor_crítico(95%)  [= 15.495]
```

**¿Por qué Johansen y no Engle-Granger?** El test de Engle-Granger requiere elegir qué serie es Y y cuál es X, y da resultados distintos según la elección. Johansen trata ambas simétricamente y tiene mayor potencia, lo que es importante cuando se trabaja con ventanas de 756 días.

### Etapa 2 — Filtro de Half-Life [3, 60] días

Aunque la cointegración esté confirmada estadísticamente, el spread puede revertir demasiado lento (> 60 días: poco rentable, demasiado tiempo en drawdown) o demasiado rápido (< 3 días: los costes de ejecución dominan). Estimamos el half-life usando el coeficiente AR(1) del spread:

```
Δspread[t] = ρ · spread[t−1] + ε[t]
half_life   = −log(2) / log(1 + ρ)
```

Solo se operan spreads con `3 ≤ half_life ≤ 60` días.

### Etapa 3 — Umbral de Sharpe en Train (≥ 0.30)

Incluso con cointegración válida y buen half-life, el Sharpe del periodo de entrenamiento debe ser ≥ 0.30 tras la optimización de parámetros. Esto garantiza que existe alfa económico real en el periodo de entrenamiento, no solo cointegración estadística. Pares que están cointegrados pero no son rentables (por ejemplo, AT&T/Verizon — declive estructural de AT&T) quedan correctamente eliminados.

---

## 3. Teoría de Filtros Adaptativos

### 3.0 Modelo Lineal Compartido

Todas las implementaciones de filtros comparten el mismo **modelo de predicción lineal**:

```
ŷ[t]  = w[t]ᵀ · x[t]              (predicción en el instante t)
e[t]  = y[t] − ŷ[t]               (error de predicción = spread)
w[t+1] = w[t] + f(e[t], x[t], ·)  (regla de actualización del peso)
```

Con `n_taps = 1`, `w[t]` es un escalar — el hedge ratio `β[t]`.

---

### 3.1 El Filtro NLMS — Base Teórica

El filtro **NLMS (Normalized Least Mean Squares)** normaliza la actualización del gradiente LMS por la potencia instantánea de la entrada:

```
w[t+1] = w[t] + μ · e[t] · x[t] / (‖x[t]‖² + ε)
```

El término `‖x[t]‖² + ε` hace que el paso efectivo sea independiente de la amplitud de la entrada. Esta normalización tiene una interpretación geométrica limpia: la actualización proyecta el vector de pesos sobre el hiperplano `{w : wᵀx[t] = y[t]}`, avanzando una fracción μ hacia él.

**El dilema fundamental del NLMS**: un μ grande converge rápido pero rastrea el ruido; un μ pequeño es estable pero lento para adaptarse. Un μ fijo no puede ser óptimo para todos los regímenes de mercado.

---

### 3.2 VS-NLMS — Paso Variable (Filtro en Producción)

El **NLMS de Paso Variable** (Kwong & Johnston, 1992) adapta μ en línea usando la **correlación de signos** entre errores de predicción consecutivos:

```
corr[t]  =  sign(e[t]) · sign(e[t-1])     ∈ {−1, 0, +1}
μ[t]     =  clip( α · μ[t-1]  +  γ · corr[t],   μ_min,  μ_max )
w[t+1]   =  w[t]  +  μ[t] · e[t] · x[t] / (‖x[t]‖² + ε)
```

| `corr[t]` | Significado | Respuesta |
|---|---|---|
| `+1` | Errores del mismo signo → el filtro se equivoca sistemáticamente | Aumentar μ: adaptarse más rápido |
| `−1` | Errores alternos → el filtro oscila alrededor de la solución | Disminuir μ: estabilizarse |
| `0` | Uno de los errores es cero | Sin cambio |

**¿Por qué usar `sign()` en lugar del producto directo?** La función `sign()` hace la regla invariante a la escala (independiente del nivel de precios) y robusta a valores atípicos.

**Parámetros optimizados**:

| Parámetro | Valor | Significado |
|---|---|---|
| `mu_init` | 0.01–0.20 | μ inicial (búsqueda en grid por fold) |
| `mu_min` | 0.001 | Suelo |
| `mu_max` | 0.50 | Techo: previene inestabilidad |
| `alpha` | 0.990 | Momento de μ |
| `gamma` | 0.050 | Escala del ajuste por paso |

---

## 4. El Filtro VS-NLMS en Detalle

### 4.1 β[t] ES el Peso del Filtro

```
Peso del filtro w[t]   ←→   Hedge ratio β[t]
Entrada del filtro x[t] ←→   precio_X[t]
Objetivo del filtro y[t] ←→   precio_Y[t]
Error del filtro e[t]  ←→   Spread: precio_Y[t] − β[t]·precio_X[t]
```

El filtro ES el estimador del hedge ratio. Los errores de predicción SON el spread.

### 4.2 Por qué μ ≈ 0.43 Emerge como Óptimo

En aplicaciones clásicas de DSP, el NLMS normalmente usa μ ∈ [0.01, 0.2]. Para la mayoría de pares en este universo, la búsqueda en grid y la adaptación del VS-NLMS convergen hacia μ ≈ 0.43 — mucho más alto de lo que la intuición sugiere.

La razón: los hedge ratios han estado en **tendencias persistentes de varios meses**. Cuando β deriva al alza durante semanas, los errores consecutivos del spread tienen consistentemente el mismo signo. La regla del VS-NLMS identifica correctamente esto como subajuste crónico (`corr = +1` repetidamente) y empuja μ hacia arriba para rastrear la deriva agresivamente.

### 4.3 ML vs Baseline — Cuándo Gana Cada Uno

| Calidad de la señal | Baseline (heurístico) | ML (MLP) | ML (GRU) | Por qué |
|---|---|---|---|---|
| Muy limpia (Nestle/Unilever) | **Gana** | Pierde | Cerca | El spread es tan estable que Kwong-Johnston ya es óptimo |
| Moderada (V/MA, MCO/SPGI) | Pierde | **Gana** | Bueno | Cambios de régimen; el ML detecta cuándo adaptarse más rápido |
| Con dependencia temporal (DBS, GS/MS) | Pierde | Pierde | **Gana** | El GRU detecta patrones en secuencia de errores que el MLP ignora |
| Ruidosa (pocos folds, Sharpe bajo) | Malos | Malos | Malos | No hay señal consistente que aprender |

---

## 5. Estrategia de Trading

### 5.1 Cálculo del Z-Score

El spread bruto `e[t]` se normaliza con un z-score rolling:

```
μ_s[t]  = media(e[t−L : t])
σ_s[t]  = std(e[t−L : t])
z[t]    = (e[t] − μ_s[t]) / σ_s[t]
```

El lookback `L` se optimiza por fold (30–120 días en el grid).

### 5.2 Generación de Señales

```
z[t] < −entry_z  →  señal[t+1] = +1   (spread por debajo: long Y, short X)
z[t] > +entry_z  →  señal[t+1] = −1   (spread por encima: short Y, long X)
|z[t]| < exit_z  →  señal[t+1] =  0   (spread ha revertido: cerrar posición)
```

Las señales usan `t+1` para evitar lookahead: observamos el z-score al cierre del día t y actuamos en la apertura del día t+1.

**Umbrales optimizados**: `entry_z = 1.5σ`, `exit_z = 0.25σ – 0.75σ` (varía por fold).

### 5.3 Sizing Proporcional al Z-Score

El tamaño de la posición escala proporcionalmente a la magnitud del z-score:

```
tamaño_posición[t] = dirección_señal[t] · |z[t]| / entry_z
```

Un spread en 3σ recibe una posición 2× mayor que uno en 1.5σ. Es un sizing tipo Kelly suave que naturalmente aumenta la exposición cuando la convicción es mayor.

### 5.4 Cálculo del Retorno

```
retorno_spread[t]    = ΔY[t]/Y[t-1] − β[t-1] · ΔX[t]/X[t-1]
retorno_estrategia[t] = señal[t-1] · retorno_spread[t]
```

Donde `β[t-1]` es el hedge ratio del día anterior (sin lookahead).

---

## 6. Metodología de Validación Walk-Forward

### 6.1 Por qué Walk-Forward

La optimización in-sample simple captura las idiosincrasias del periodo histórico concreto — esto es **sesgo de selección**. El walk-forward garantiza que cada punto de test se evalúa después de que el modelo fue entrenado exclusivamente con datos del pasado relativo a ese punto.

### 6.2 Implementación

```
Ventana de train:  756 días (~3 años)
Ventana de test:   252 días (~1 año)
Paso:              252 días (periodos de test sin solapamiento)
Folds totales:     13 (para 16 años de datos)

Fold 1:  Train [2010 → 2013]  →  Test [2013 → 2014]
Fold 2:  Train [2011 → 2014]  →  Test [2014 → 2015]
...
Fold 13: Train [2022 → 2025]  →  Test [2025 → 2026]
```

Las ventanas de train se solapan. El invariante crítico: **las ventanas de test nunca se solapan**, y ninguna información futura se filtra al entrenamiento.

### 6.3 Qué se Optimiza por Fold

| Parámetro | Grid | Notas |
|---|---|---|
| `mu_init` | [0.01, 0.05, 0.10, 0.20] | μ inicial para VS-NLMS |
| `lookback` | [30, 60, 90, 120] días | Ventana rolling del z-score |
| `entry_z` | [1.5] | Fijo (mejor en la mayoría de folds) |
| `exit_z` | [0.25, 0.50, 0.75] | Umbral de salida |

**Total**: 48 combinaciones por fold. **Objetivo**: `score = Sharpe + 0.5 × MaxDrawdown`.

### 6.4 Warm-Up del Filtro

El filtro VS-NLMS se ejecuta sobre la ventana completa (train + test) durante la evaluación, pero solo se registra el periodo de test. Este calentamiento garantiza que el filtro ha convergido a un hedge ratio estable antes de que comience el periodo de test.

### 6.5 Detección de Sobreajuste

```
degradación = 1 − (sharpe_medio_test / sharpe_medio_train)
```

Una degradación por debajo del 30% es aceptable. Por encima del 50% es una señal de advertencia de sobreajuste.

---

## 7. Resultados del Escaneo del Universo

Se han testado más de 70 pares en renta variable (EE.UU., ADRs europeos y asiáticos), divisas y bonos. El **filtro de tres etapas** (Johansen + half-life + Sharpe en train) elimina la gran mayoría.

### 7.1 Tasa de Cointegración por Par — Resumen

| Sector | Mejor par | Folds coint. | % | Resultado |
|--------|-----------|-------------|---|-----------|
| FMCG Global | Nestle/Unilever (NSRGY/UNLYF) | 8/12 | 67% | ✓ EXCELENTE |
| Bancos SG | DBS/UOB (DBSDY/UOVEY) | 8/13 | 62% | ✓ BUENO |
| Pagos EE.UU. | Visa/Mastercard (V/MA) | 5/14 | 36% | ✓ BUENO |
| Ratings | Moody's/S&P Global (MCO/SPGI) | 4/13 | 31% | ✓ BUENO |
| Lujo | LVMH/Kering (LVMUY/PPRUY) | 3/11 | 27% | ~ OK |
| Bebidas | Diageo/Pernod (DEO/PRNDY) | 5/13 | 38% | ~ OK |
| Farmacéutica | Roche/AstraZeneca (RHHBY/AZN) | 4/13 | 31% | ~ OK |
| Energía | BP/Shell (BP/SHEL) | 3/13 | 23% | ~ marginal |
| Banca EE.UU. | Goldman/Morgan Stanley (GS/MS) | 1/13 | 8% | ~ marginal |

### 7.2 Pares que Fallaron (muestra representativa)

Los siguientes pares mostraron < 15% de cointegración en todos los folds, lo que los hace no operables con esta metodología:

- **Tecnología EE.UU.**: AMD/Intel, MSFT/Oracle, QCOM/Broadcom, ASML/AMAT — las tendencias alcistas del mercado rompen la cointegración
- **Financials EE.UU.**: JPM/BAC, BLK/SCHW, USB/PNC, STT/BK — diferencias estructurales post-regulación 2008
- **Salud EE.UU.**: JNJ/ABT, PFE/MRK, UNH/CVS — actividad de M&A y divergencia de pipelines
- **Industriales EE.UU.**: CAT/DE, HON/3M, UPS/FDX, BA/LMT — modelos de negocio demasiado diferentes
- **Energía EE.UU.**: XOM/CVX, SLB/HAL — los ciclos de materias primas dominan sobre la mean-reversion
- **Telecom**: T/VZ — el declive estructural de AT&T (deuda de TimeWarner) rompió el par permanentemente
- **Mercados asiáticos**: Toyota/Honda, Alibaba/JD, TSMC/UMC, Infosys/Wipro — riesgo regulatorio y cambios de régimen
- **Banca japonesa**: Nomura/Mizuho, Sumitomo/Mizuho — máximo 23%, distorsiones del Banco de Japón
- **Bancos australianos**: CBA/NAB — Sharpe 0.11, señal insuficiente

**Causa raíz**: el mercado alcista de EE.UU. 2010–2026 produce alta correlación entre valores pero no cointegración. Los spreads derivan en lugar de revertir. Solo los pares con economías verdaderamente idénticas (duopolios de pagos, gigantes de FMCG, oligopolios de datos financieros) mantienen el equilibrio a largo plazo.

---

## 8. Mejores Pares — Resultados Detallados

Todos los resultados usan `--no-background` (cada fold entrenado solo con sus propios datos, sin leakage del pool) y la cascada de tres filtros. El mejor resultado por fila aparece en **negrita**.

| Par | Tickers | Sector | Folds | Sharpe Base | Sharpe MLP | Sharpe GRU | Return (anual) | DD máx | Operable hoy |
|-----|---------|--------|-------|-------------|------------|------------|----------------|--------|--------------|
| Nestle / Unilever | NSRGY / UNLYF | FMCG Global | 8/12 | **2.64** | 2.36 | 2.58 | 41.6% | −13% | ✗ |
| Visa / Mastercard | V / MA | Pagos EE.UU. | 5/14 | 1.07 | **1.30** | 0.81 | 14.9% | −15% | ✗ |
| LVMH / Kering | LVMUY / PPRUY | Lujo | 3/11 | 0.61 | **0.92** | 0.85 | 13.1% | −12% | ✗ |
| Moody's / S&P Global | MCO / SPGI | Ratings | 4/13 | 0.85 | **0.91** | 0.89 | 9.4% | −7% | **✓** |
| Diageo / Pernod | DEO / PRNDY | Bebidas | 5/13 | **0.93** | 0.62 | 0.86 | 6.2% | −14% | ✗ |
| DBS / UOB | DBSDY / UOVEY | Bancos SG | 8/13 | 0.91 | 0.77 | **0.93** | 8.6% | −14% | ✗ |
| Roche / AstraZeneca | RHHBY / AZN | Farmacéutica | 4/13 | **0.68** | 0.57 | 0.52 | 16.4% | −28% | ✗ |
| Goldman / Morgan Stanley | GS / MS | Banca EE.UU. | 1/13 | 0.58 | 0.70 | **0.87** | 8.3% | −10% | ✗ |
| BP / Shell | BP / SHEL | Energía | 3/13 | 0.53 | 0.59 | 0.58 | 13.4% | −16% | ✗ |

**Notas:**
- **Operable hoy**: el fold más reciente (train 2022–2025, test 2025–2026) superó los tres filtros
- **Baseline gana** cuando la señal es muy limpia (Nestle, Diageo) — Kwong-Johnston ya es óptimo
- **MLP gana** en pares con cambios de régimen moderados (V/MA, MCO/SPGI, LVMH/Kering)
- **GRU gana** en pares con dependencias temporales en el spread (DBS/UOB, GS/MS)
- MCO/SPGI es el **único par actualmente operable** en renta variable con Sharpe > 0.9 y DD < −8%
- Roche/AZN tiene el mayor retorno anualizado (16.4%) pero también el mayor drawdown (−28%)

---

## 9. Universo Macro — FX, Bonos, Materias Primas e Índices

### 9.1 Motivación

El pairs trading en renta variable se basa en la exposición sectorial compartida. El pairs trading macro explota una clase distinta de relaciones: **vínculos económicos estructuralmente anclados** — bloques de divisas ligados a la misma materia prima, curvas de tipos dentro del mismo soberano, ciclos de mercados desarrollados frente a emergentes.

Estas relaciones difieren fundamentalmente de los pares sectoriales:
- Están impulsadas por regímenes macro globales (risk-on/risk-off, superciclos de materias primas, divergencia de política de bancos centrales) en lugar de por fundamentos a nivel empresa
- Son más estables a largo plazo, pero están sujetas a **rupturas de régimen duras** (p. ej., eliminación del ancla del SNB en 2015, shock de tipos de 2022)
- Requieren fuentes de datos distintas: FX vía yfinance (expresado como USD por unidad de divisa extranjera), yields de bonos vía FRED (convertidos a precios de bono cupón cero), índices vía ETFs

### 9.2 Fuentes de Datos

| Clase de activo | Fuente | Formato del ticker | Notas |
|---|---|---|---|
| Pares de FX | yfinance | `XXXUSD=X` | USD por unidad de divisa extranjera |
| Yields de bonos | FRED | `DGS2`, `DGS10` | Convertidos a precio: P = exp(−y·T) |
| Índices de renta variable | yfinance | Tickers de ETF | EFA (MSCI EAFE), EEM (MSCI EM) |
| Materias primas | yfinance | Spot/ETF | GLD, SLV, USO, BNO... |

Los precios de FX se expresan como USD por unidad de divisa extranjera: NOKUSD=X subiendo significa corona noruega fortaleciéndose. Largo NOK / corto CAD = apuesta a que la corona noruega supera al dólar canadiense.

### 9.3 El Proyecto Paralelo nlms-macro

Los activos macro se gestionan en el proyecto paralelo `nlms-macro/`, que usa el mismo motor walk-forward, cascada de tres filtros y la infraestructura VS-NLMS/ML — aplicados a series temporales no financieras de renta variable.

```
nlms-macro/
├── walk_forward_ml_mu.py   # Mismo motor que nlms-zsizing
├── fetch_yields.py         # Descarga DGS2/5/10/30 de FRED
├── fetch_fx.py             # Descarga pares FX de yfinance
└── data/
    ├── nok_cad.csv         # Par FX NOK/CAD (mejor resultado FX)
    ├── aud_cad.csv         # AUD/CAD (divisas de materias primas)
    ├── eur_chf.csv         # EUR/CHF (anclado por el SNB)
    ├── efa_eem.csv         # EFA/EEM (índices DM vs EM)
    └── 2y10y.csv           # Spread 2Y/10Y
```

### 9.4 Pares Macro Testados — Resultados Completos

Más de 16 clases de activos macro testadas. Se aplica el mismo filtro de tres etapas (Johansen + half-life + Sharpe de train).

**Pares operables encontrados:**

Todos los resultados usan `--no-background`. El mejor resultado por fila en **negrita**.

| Par | Tickers | Clase | Folds | Sharpe Base | Sharpe MLP | Sharpe GRU | Retorno (anual) | DD máx | Operable hoy |
|-----|---------|-------|-------|-------------|------------|------------|-----------------|--------|--------------|
| EFA / EEM | EFA / EEM | Índices | 4/13 | 0.83 | **1.32** | 1.04 | 9.3% | −7.2% | **✓** |
| NOK / CAD | NOKUSD=X / CADUSD=X | FX | 6/13 | **0.72** | 0.72 | 0.71 | 13.3% | −10.9% | ✗ |
| AUD / CAD | AUDUSD=X / CADUSD=X | FX | 3/13 | 0.50 | **0.61** | — | 3.9% | −5.1% | ✗ |
| EUR / CHF | EURUSD=X / CHFUSD=X | FX | 5/13 | **0.51** | 0.51 | — | 7.1% | −8.2% | ✗ |
| WTI / Brent (futuros) | CL=F / BZ=F | Materias primas | 3/13 | **0.51** | −1.26 | 0.30 | 4.1% | −12.3% | ✗ |
| Spread 2Y / 10Y | DGS2 / DGS10 | Bonos | 1/13 | 0.40 | **0.58** | — | 2.1% | −2.3% | ✗ ⚠ |

⚠ Solo 1 fold operable — resultado estadísticamente poco fiable.

**Notas sobre WTI/Brent**: Los ETFs (USO/BNO) fallan por costes de roll acumulados que generan deriva estructural. Los futuros continuos (CL=F/BZ=F) mantienen 3 folds operables con Sharpe de baseline 0.51. El **MLP destruye valor** en este par (−1.26) mientras el **GRU es más conservador** (0.30). El par no pasa el test de cointegración en el fold más reciente (2025), por lo que no es operable actualmente.

**Pares que fallaron (0 folds operables o drawdown inasumible):**

| Par | Motivo del fallo |
|---|---|
| Oro / Plata (GLD/SLV) | Solo 15% cointegración — la demanda industrial de la plata diverge en crisis |
| WTI / Brent (ETFs USO/BNO) | Costes de roll acumulados generan deriva estructural (usar futuros CL=F/BZ=F) |
| Gas Natural / Petróleo | Cointegrado pero drawdown explosivo (−83%) — rupturas de régimen del gas |
| LQD / HYG (IG vs HY) | 0% cointegración — el spread de crédito no es estacionario en 16 años |
| SPY / TLT (renta variable vs bonos) | 0% cointegración — los cambios de régimen risk-on/off son rupturas permanentes |
| SPY / IWM, SPY / QQQ | 0% cointegración — los factores divergen estructuralmente |
| JPY / CHF | Cointegrado pero 0 folds operables — el control de curva del Banco de Japón |
| Agrícolas (maíz/trigo, soja/maíz) | 0% cointegración — costes de almacenamiento y estacionalidad dominan |

### 9.5 Conclusiones Clave

**Por qué EFA/EEM funciona mejor**: Los índices de mercados desarrollados y emergentes comparten exposición al crecimiento global pero divergen en ciclos de aversión al riesgo. El spread captura el **ciclo de prima DM/EM**, con regímenes de varios años que el filtro ML detecta mejor que la heurística — lo que explica la gran brecha ML vs. baseline (1.32 vs. 0.83).

**Por qué fallan los pares de materias primas**: Los ETFs de futuros acumulan **costes de roll** que producen deriva estructural en el spread. El ratio oro/plata es un concepto conocido pero la plata falla porque su componente industrial genera divergencias que no revierten en un horizonte de 3 años.

**Por qué los pares FX funcionan moderadamente**: Los pares FX comparten vínculos de commodities/comercio, pero sufren **rupturas de régimen** por intervenciones de bancos centrales y cambios de ancla. NOK/CAD es el más estable porque ambas divisas flotan libremente y están ancladas al Brent y al WTI — dos referencias altamente correlacionadas con mínimo riesgo de base estructural.

**Por qué los pares de bonos fallan mayoritariamente**: Los ETFs de bonos tienen duración que varía con los tipos, haciendo que el valor relativo sea no estacionario. El único resultado marginal (2Y/10Y) tiene apenas 1 fold operable — insuficiente para despliegue real.

---

## 10. Experimentos de Machine Learning

### 10.1 ML_VSNLMSFilter — Sustituyendo la Heurística de μ

**Motivación**: La regla de Kwong-Johnston es una heurística manual. ¿Puede una red neuronal aprender un mapeo superior de la historia de errores al μ óptimo?

**Implementación**: `src/ml_nlms.py` — `train_mu_predictor()` y `ML_VSNLMSFilter`.

#### Fase de Entrenamiento (por fold, solo con datos de train)

**Paso 1 — Competición de filtros en paralelo**: Ejecutar 9 filtros NLMS simultáneamente, cada uno con un μ fijo distinto de `[0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]`.

**Paso 2 — Generación del objetivo (criterio de mean-reversion)**:

```
autocorr_i[t] = corr(e_i[t−20:t−1], e_i[t−19:t])   para cada filtro i
ganador[t]    = argmin_i  autocorr_i[t]              (más mean-reverting)
Y[t]          = μ_candidatos[ganador[t]]              (μ objetivo para el MLP)
```

**Paso 3 — Extracción de features** (12 features por instante):

| Feature | Descripción |
|---|---|
| z-score rolling | Z-score actual del spread (ventana de 5 días) |
| Autocorrelación | Autocorr lag-1 del spread (5 días) |
| Volatilidad del spread | Std rolling del spread |
| Correlación de signos | `sign(e[t])·sign(e[t-1])` |
| μ actual | Paso actual del VS-NLMS |
| β actual | Estimación actual del hedge ratio |
| + 6 features desfasadas | Features en t-1, t-2, t-5 |

**Paso 4**: Entrenar `MLPRegressor(hidden_layer_sizes=(64,32), max_iter=500)` para predecir el μ óptimo a partir de estos features.

### 10.2 GRU_VSNLMSFilter — Memoria Temporal

**Motivación**: El MLP trata cada timestep como independiente — `μ[t] = MLP(features[t])`. Un GRU mantiene un estado oculto `h[t]` que depende de todos los pasos anteriores, permitiéndole detectar patrones como *"el spread lleva 10 días con errores del mismo signo → subir μ progresivamente"*.

**Implementación**: `src/gru_nlms.py` — GRU completo en numpy puro (sin PyTorch), con BPTT y Adam.

```
Ecuaciones GRU:
    z[t] = σ(Wz·[x[t], h[t-1]] + bz)          # update gate
    r[t] = σ(Wr·[x[t], h[t-1]] + br)          # reset gate
    g[t] = tanh(Wg·[x[t], r[t]⊙h[t-1]] + bg)  # candidato
    h[t] = (1−z[t])⊙h[t-1] + z[t]⊙g[t]        # nuevo hidden state
    μ[t] = Wo·h[t] + bo                         # predicción
```

**Arquitectura**: 1 capa GRU, `hidden_size=32`, Adam con gradient clipping. Entrenamiento: BPTT completo + checkpoint de menor pérdida de validación (10% temporal).

**Cuándo usar GRU vs MLP**:
- **MLP**: mejor en pares con señal limpia y moderada (V/MA, EFA/EEM)
- **GRU**: mejor en pares donde el spread tiene dependencias temporales fuertes (DBS/UOB, GS/MS)
- El GRU puede sobreajustar con pocos folds de entrenamiento (< 4) — en esos casos usar MLP o baseline

### 10.3 Flags de Ejecución

```bash
# MLP (por defecto) — sin contaminación temporal del pool
python walk_forward_ml_mu.py --data=data/visa_mastercard.csv --pair="V/MA" --no-background

# GRU — misma interfaz, diferente predictor de μ
python walk_forward_ml_mu.py --data=data/visa_mastercard.csv --pair="V/MA" --gru --no-background

# Sin --no-background: el pool crece entre ejecuciones → Sharpe inflado (no usar para comparar)
python walk_forward_ml_mu.py --data=data/visa_mastercard.csv --pair="V/MA"
```

---

## 11. Cómo Usar la Estrategia en la Práctica

Esta sección explica **cómo ejecutar la estrategia realmente** — cuándo entrenar, cuándo testear, cuándo operar y cuándo parar.

### 11.1 Flujo General

```
Cada año (o cuando expira un fold):
  1. Descargar datos frescos (3 años atrás + año actual)
  2. Ejecutar los tres filtros sobre la nueva ventana de train
  3. Si todos los filtros pasan: grid search, obtener mejores parámetros
  4. Desplegar los hiperparámetros estructurales (lookback, entry_z, exit_z) y el modelo MLP congelado para los próximos 252 días. Importante: el MLP predice un μ[t] nuevo cada día según las features de mercado actuales — el hedge ratio β[t] se adapta de forma continua. Lo que queda fijo es la arquitectura y los pesos del modelo, no el step size.
  5. Monitorizar el spread diariamente: generar señales de entrada/salida
  6. Después de 252 días: volver al paso 1 (nuevo fold)
```

### 11.2 Paso a Paso: Entrenar un Nuevo Fold

**Cuándo entrenar**: una vez al año, al final del periodo de test anterior. Si tu ventana de test actual expira el 31 de diciembre, reentrena el fin de semana anterior al 1 de enero.

**Datos necesarios**: 3 años de precios de cierre ajustados diarios para ambos tickers. Usar `yfinance`:

```python
import yfinance as yf
df = yf.download(["MCO", "SPGI"], start="2022-01-01", end="2025-01-01",
                 auto_adjust=True)["Close"]
```

**Etapa 1 — Test de Johansen**: ejecutar sobre la ventana completa de 3 años:
```python
from src.cointegration import johansen_cointegration
result = johansen_cointegration(df["MCO"].values, df["SPGI"].values)
if not result["cointegrated"]:
    print("SALTAR — el par no está cointegrado este año")
    # NO operar este fold
```

**Etapa 2 — Half-life**:
```python
from src.cointegration import compute_halflife
hl = compute_halflife(df["MCO"].values, df["SPGI"].values)
if not hl["in_range"]:  # debe estar en [3, 60] días
    print(f"SALTAR — half-life {hl['halflife']:.0f}d fuera del rango operable")
```

**Etapa 3 — Grid search + Sharpe en train**:
```python
# Ejecutar walk_forward_ml_mu.py sobre el periodo de entrenamiento
# Los mejores parámetros se guardan en results/walk_forward_ml_mu_folds.csv
# Comprobar que best_train_sharpe >= 0.30
```

Si los tres filtros pasan, extraer los mejores parámetros (lookback, entry_z, exit_z, mu_init) de la optimización del entrenamiento.

### 11.3 Paso a Paso: Generar Señales Diarias

**Cada día de trading** (tras el cierre del mercado, antes de la apertura del día siguiente):

1. **Descargar los precios de cierre de hoy** para ambos tickers
2. **Actualizar el filtro VS-NLMS** con el precio de hoy:
   ```python
   beta_hoy = filtro.step(precio_x_hoy, precio_y_hoy)
   spread_hoy = precio_y_hoy - beta_hoy * precio_x_hoy
   ```
3. **Actualizar el z-score rolling** (usando el lookback del entrenamiento):
   ```python
   z_hoy = (spread_hoy - media_rolling_spread) / std_rolling_spread
   ```
4. **Comprobar la señal**:
   ```python
   if z_hoy < -1.5 and posicion_actual == 0:
       # ABRIR LARGO: comprar Y, vender X
       # Tamaño: |z_hoy| / 1.5 × nocional
   elif z_hoy > +1.5 and posicion_actual == 0:
       # ABRIR CORTO: vender Y, comprar X
       # Tamaño: |z_hoy| / 1.5 × nocional
   elif abs(z_hoy) < exit_z and posicion_actual != 0:
       # CERRAR: deshacer ambas patas
   ```
5. **Ejecutar en la apertura del día siguiente** (nunca al cierre de hoy — eso sería lookahead)

### 11.4 Sizing de la Posición

La estrategia es **dollar-neutral**: por cada 1€/$ de Y, ponerse corto en `β[t]` dólares de X.

```
nocional_Y = capital_total × fracción_posición × |z[t]| / entry_z
nocional_X = nocional_Y × β[t]

acciones_Y = redondear(nocional_Y / precio_Y)
acciones_X = redondear(nocional_X / precio_X)
```

**Fracción máxima de posición recomendada**: 10–20% del capital por par. Si se operan varios pares simultáneamente, asignar proporcionalmente al Sharpe (por ejemplo, MCO/SPGI con 0.91 recibe una fracción mayor que BP/Shell con 0.59).

### 11.5 Cuándo Dejar de Operar un Fold

Parar (cerrar todas las posiciones y esperar al próximo retrain anual) si ocurre cualquiera de lo siguiente:

| Condición | Acción |
|---|---|
| El periodo de test del fold expira (252 días) | Reentrenar el siguiente fold |
| El drawdown en OOS supera 2× el DD histórico máximo | Parar y reentrenar inmediatamente |
| El spread lleva > 60 días consecutivos fuera de las bandas | El par puede haberse roto — parar |
| Evento corporativo relevante (M&A, spin-off, quiebra) en cualquiera de los tickers | Parar inmediatamente, reentrenar tras el evento |

### 11.6 Ejecutar los Scripts Disponibles

```bash
# 1. Descargar datos para un sector específico
python scan_universe.py --sector financials

# 2. Walk-forward completo para un par específico (recomendado: --no-background)
python walk_forward_ml_mu.py --data=data/scan/mco_spgi.csv \
                              --pair="MCO / SPGI" \
                              --no-background

# 3. Escanear el universo completo (70+ pares, ~30 minutos)
python scan_universe.py

# 4. Comprobar operabilidad actual de todos los mejores pares
# Ver el fold más reciente en results/walk_forward_ml_mu_folds.csv
# Si el último fold superó los tres filtros → par operable ahora
```

### 11.7 Pares Actualmente Operables (marzo 2026)

| Par | Sharpe ML | Acción |
|-----|-----------|--------|
| **MCO / SPGI** | **0.91** | ✓ Operar ahora — último fold superó todos los filtros |
| V / MA | 1.09 | ✗ Último fold en el límite (Sharpe train 0.50) — monitorizar |
| DBS / UOB | 0.91 | ✗ Último fold falló Johansen (traza=13.8) — monitorizar |

**Monitorizar** significa: ejecutar el test de Johansen semanalmente sobre los últimos 756 días. Si traza > 15.5 durante dos semanas consecutivas → considerar entrar.

---

## 12. Estructura del Proyecto

```
nlms-zsizing/
│
├── walk_forward_ml_mu.py   # Punto de entrada principal — walk-forward baseline vs MLP
├── scan_universe.py        # Escáner del universo — testa 70+ pares automáticamente
├── portfolio_sharpe.py     # Sharpe del portfolio a partir de 11 series OOS
├── walk_forward.py         # Walk-forward simplificado (solo baseline VS-NLMS)
├── scan_pairs.py           # Escáner ligero de pares
├── analyze_pairs.py        # Análisis post-escaneo
│
├── src/
│   ├── nlms.py             # VSNLMSFilter — filtro adaptativo principal (Kwong-Johnston)
│   ├── ml_nlms.py          # ML_VSNLMSFilter — predictor de μ con MLP + pipeline de entrenamiento
│   ├── gru_nlms.py         # GRU_VSNLMSFilter — predictor de μ con GRU (flag --gru)
│   ├── cointegration.py    # Test de Johansen + estimador de half-life
│   ├── signals.py          # Cálculo del z-score + máquina de estados de señales
│   ├── optimizer.py        # Grid search + run_filter_pipeline
│   ├── backtest.py         # Simulador de P&L
│   ├── model_store.py      # Pool persistente de datos de entrenamiento MLP (models/)
│   ├── strategy.py         # Wrapper de alto nivel de la estrategia
│   ├── plots.py            # Curva de equity + visualización del spread
│   └── data_generator.py   # Generador de pares cointegrados sintéticos (para tests)
│
├── data/                   # CSVs — NO se suben (se descargan con yfinance)
│   ├── visa_mastercard.csv # Par principal de desarrollo
│   ├── scan/               # 70+ pares escaneados del universo
│   └── .gitkeep
│
├── models/                 # Datos de entrenamiento acumulados del MLP — NO se suben
│   └── .gitkeep
│
├── results/                # Resultados del walk-forward — NO se suben
│   └── .gitkeep
│
├── ibex35_statistical_arbitrage.ipynb  # Notebook de referencia (terceros)
├── requirements.txt
├── README.md               # Documentación en inglés
└── README_ES.md            # Documentación en castellano
```

---

## 13. Instalación y Uso

```bash
# 1. Clonar e instalar
git clone https://github.com/<tu-usuario>/nlms-pairs-trading.git
cd nlms-pairs-trading/nlms-zsizing
pip install -r requirements.txt

# 2. Descargar datos de un par (ejemplo)
python - <<'EOF'
import yfinance as yf, pandas as pd
d1 = yf.download("V",  start="2010-01-01", end="2026-01-01", auto_adjust=True)["Close"].squeeze()
d2 = yf.download("MA", start="2010-01-01", end="2026-01-01", auto_adjust=True)["Close"].squeeze()
df = pd.DataFrame({"price_x": d1, "price_y": d2}).dropna().reset_index()
df.columns = ["date","price_x","price_y"]
df.to_csv("data/visa_mastercard.csv", index=False)
EOF

# 3. Ejecutar el walk-forward — baseline VS-NLMS vs MLP (recomendado: --no-background)
python walk_forward_ml_mu.py --data=data/visa_mastercard.csv --no-background

# Usar GRU en lugar de MLP
python walk_forward_ml_mu.py --data=data/visa_mastercard.csv --no-background --gru

# 4. Escanear el universo completo de renta variable (descarga + testa 70+ pares)
python scan_universe.py

# 5. Calcular el Sharpe del portfolio sobre todos los pares activos
python portfolio_sharpe.py
```

### Flags disponibles

| Flag | Descripción |
|------|-------------|
| `--data=ruta` | CSV con columnas `date, price_x, price_y` |
| `--pair="X/Y"` | Nombre del par mostrado en los resultados |
| `--no-background` | Entrena solo con datos del fold actual — sin leakage entre pares (recomendado) |
| `--gru` | Usa GRU_VSNLMSFilter en lugar de ML_VSNLMSFilter |
| `--train=N` | Ventana de entrenamiento en días (por defecto: 756) |
| `--test=N` | Ventana de test en días (por defecto: 252) |

---

## 14. Construcción de Portfolio y Resultados

### 14.1 Metodología

Los Sharpes individuales no se pueden promediar para obtener el Sharpe del portfolio. El método correcto es:

1. **Extraer la serie diaria de retornos OOS** de cada par del walk-forward MLP (`--no-background`). Los días donde el fold está inactivo (falla cointegración, half-life o umbral de Sharpe en train) se mantienen como cero — el capital está aparcado.
2. **Combinar todas las series por fecha** en un único DataFrame, rellenando `NaN` con 0 en los días donde un par no tiene fold activo.
3. **Media de igual peso** entre los N pares: `portfolio[t] = media(retornos[t])`.
4. **Calcular el Sharpe sobre la serie combinada** — esto penaliza correctamente los días inactivos.

El beneficio de diversificación surge porque los retornos del spread de pares estructuralmente distintos (sectores, geografías, clases de activo diferentes) son casi incorrelados. Cuando las correlaciones son cercanas a cero, la volatilidad del portfolio cae proporcionalmente a `1/√N` mientras el retorno esperado se mantiene constante, mejorando el Sharpe.

### 14.2 Universo

11 pares incluidos (9 renta variable + 2 macro). WTI/Brent excluido (proxies ETF con costes de roll estructurales).

| Par | Clase de activo | Folds activos | Sharpe (individual) |
|-----|-----------------|---------------|---------------------|
| V / MA | Pagos | 4/13 | 1.30 |
| NSRGY / UNLYF | Consumer staples | 8/13 | 2.36 |
| DEO / PRNDY | Bebidas | 5/13 | 0.62 |
| DBS / UOB | Bancos de Singapur | 8/13 | 0.77 |
| MCO / SPGI | Agencias de rating | 4/13 | 0.91 |
| LVMUY / PPRUY | Lujo | 3/13 | 0.92 |
| RHHBY / AZN | Farmacéuticas | 4/13 | 0.57 |
| GS / MS | Banca de inversión EE.UU. | 1/13 | 0.70 |
| BP / SHEL | Energía | 3/13 | 0.59 |
| EFA / EEM | Índices DM/EM | 4/13 | 1.32 |
| NOK / CAD | FX | 6/13 | 0.72 |

### 14.3 Resultados del Portfolio (Igual Peso, MLP, --no-background)

| Métrica | Valor |
|---------|-------|
| **Sharpe ratio** | **1.94** |
| Retorno anualizado | 5.5% |
| Volatilidad anualizada | 2.8% |
| Max Drawdown | −3.6% |
| Días con posición | 1.972 / 3.375 |
| Correlación media entre pares | −0.003 |
| Ganancia por diversificación vs media individual | **+98%** |

Retornos anuales:

| Año | Retorno |
|-----|---------|
| 2013 | +4.9% |
| 2014 | +4.0% |
| 2015 | +4.7% |
| 2016 | +9.9% |
| 2017 | +3.7% |
| 2018 | +5.1% |
| 2019 | +0.6% |
| 2020 | +9.7% |
| 2021 | +20.8% |
| 2022 | +2.4% |
| 2023 | +3.2% |
| 2024 | +4.5% |

Ningún año negativo en 12 años de datos OOS.

### 14.4 Por Qué el Sharpe del Portfolio Supera los Individuales

Sharpe medio individual (calculado sobre días activos dentro del período de cada fold): 0.98. Sharpe del portfolio: 1.94 — una **ganancia por diversificación del +98%**.

La correlación media casi nula entre pares (−0.003) confirma que los retornos del spread en distintos sectores y clases de activo son esencialmente independientes. Esto es teóricamente esperado: el spread de V/MA captura el riesgo relativo Visa-Mastercard; el spread de EFA/EEM captura el riesgo relativo desarrollados-emergentes. Estas fuentes de riesgo no tienen relación estructural entre sí.

**Limitación principal**: los resultados excluyen costes de transacción (bid-ask spread, coste de préstamo para la pata corta, comisiones). Para pares líquidos (V/MA, MCO/SPGI, EFA/EEM) el impacto sería modesto. Para ADRs OTC (NSRGY/UNLYF, DBSDY/UOVEY), el slippage puede ser significativo.

---

## 15. Referencias

1. **Kwong, C.P. & Johnston, E.W. (1992)**. "A variable step size LMS algorithm." *IEEE Transactions on Signal Processing*, 40(7), 1633–1642. — Base del filtro VS-NLMS utilizado en todo el proyecto.

2. **Johansen, S. (1988)**. "Statistical analysis of cointegration vectors." *Journal of Economic Dynamics and Control*, 12(2-3), 231–254. — El test de cointegración usado en la Etapa 1.

3. **Engle, R.F. & Granger, C.W.J. (1987)**. "Co-integration and error correction: representation, estimation, and testing." *Econometrica*, 55(2), 251–276. — Teoría fundamental de cointegración.

4. **Vidyamurthy, G. (2004)**. *Pairs Trading: Quantitative Methods and Analysis*. Wiley Finance. — Referencia estándar para la metodología de pairs trading.

5. **Chan, E. (2013)**. *Algorithmic Trading: Winning Strategies and Their Rationale*. Wiley. — Guía práctica de implementación de pairs trading algorítmico.

6. **Haykin, S. (2002)**. *Adaptive Filter Theory* (4ª ed.). Prentice Hall. — Referencia completa para NLMS, RLS y teoría de filtros adaptativos.
