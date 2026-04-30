"""
nlms.py — Filtros adaptativos para estimación del hedge ratio.

Este módulo contiene cuatro implementaciones de filtros adaptativos:

    NLMSFilter      — NLMS estándar (baseline)
    RLSFilter       — Recursive Least Squares con forgetting factor λ
    LeakyNLMSFilter — NLMS con regularización L2 implícita (leakage)
    VSNLMSFilter    — NLMS con step size variable (Kwong-Johnston)

Todos tienen la misma interfaz: __init__, update(x, y), run(X, y).
Esto permite intercambiarlos sin tocar el resto del pipeline.

Modelo común:
    y[t] = w[t]ᵀ · x[t] + e[t]

En nuestro caso (n_taps=1):
    price_y[t] = β[t] · price_x[t] + e[t]

donde β[t] es el hedge ratio adaptativo y e[t] es el spread.

──────────────────────────────────────────────────────────────────
¿Por qué mejorar el filtro?

El NLMS estándar tiene tres limitaciones para datos financieros:

1. μ FIJO: el mismo step size durante toda la serie.
   En un régimen estable queremos μ pequeño (no sobre-reaccionar al ruido).
   Tras un cambio de régimen queremos μ grande (adaptarse rápido).
   Un único μ es siempre un compromiso subóptimo.

2. NORMALIZACIÓN INSTANTÁNEA: ‖x[t]‖² usa solo el precio de hoy.
   Un pico puntual de precios "frena" injustamente la actualización.

3. SIN REGULARIZACIÓN: los pesos pueden derivar sin límite durante
   periodos tranquilos, generando hedge ratios inestables.

Los tres filtros nuevos atacan exactamente estos tres problemas.
──────────────────────────────────────────────────────────────────
"""

import numpy as np


class NLMSFilter:
    """
    Filtro adaptativo NLMS estándar (baseline).

    Ecuación de actualización:
        w[t+1] = w[t] + μ · e[t] · x[t] / (‖x[t]‖² + ε)

    Parámetros
    ----------
    n_taps : int
        Número de coeficientes (1 para un solo hedge ratio).
    mu : float
        Step size en (0, 1]. Alto = adapta rápido pero ruidoso.
    eps : float
        Regularización anti-división-por-cero.
    """

    def __init__(self, n_taps: int = 1, mu: float = 0.1, eps: float = 1e-8):
        if not 0 < mu <= 1:
            raise ValueError("mu must be in (0, 1]")
        self.n_taps = n_taps
        self.mu = mu
        self.eps = eps
        self.weights = np.zeros(n_taps)

    def update(self, x: np.ndarray, y: float) -> tuple[float, float]:
        x = np.asarray(x, dtype=float)
        y_hat = np.dot(self.weights, x)
        error = y - y_hat
        norm_factor = np.dot(x, x) + self.eps
        self.weights = self.weights + self.mu * error * x / norm_factor
        return y_hat, error

    def run(self, X: np.ndarray, y: np.ndarray) -> dict:
        n_samples = len(y)
        y_hat = np.zeros(n_samples)
        errors = np.zeros(n_samples)
        weights_history = np.zeros((n_samples, self.n_taps))
        for t in range(n_samples):
            y_hat[t], errors[t] = self.update(X[t], y[t])
            weights_history[t] = self.weights.copy()
        return {"y_hat": y_hat, "errors": errors, "weights_history": weights_history}


class RLSFilter:
    """
    Recursive Least Squares (RLS) con forgetting factor λ.

    RLS resuelve el problema de mínimos cuadrados EXACTAMENTE en cada paso,
    manteniendo una estimación recursiva de la matriz de covarianza inversa P.

    ¿Por qué es mejor que NLMS?
    - NLMS hace un paso de gradiente aproximado. RLS hace el paso óptimo.
    - NLMS converge en O(1/μ) iteraciones. RLS converge en ~n_taps iteraciones.
    - Con n_taps=1, el RLS converge prácticamente en el primer paso.
    - El forgetting factor λ controla cuánta "memoria" tiene el filtro:
        λ = 1.00 → memoria infinita (como OLS sobre todos los datos)
        λ = 0.999 → ventana efectiva ≈ 1/(1-0.999) = 1000 días
        λ = 0.995 → ventana efectiva ≈ 200 días
        λ = 0.990 → ventana efectiva ≈ 100 días

    Ecuaciones (derivadas del filtro de Kalman):

        Px[t] = P[t-1] · x[t]                         (pre-cálculo)
        K[t]  = Px[t] / (λ + x[t]ᵀ · Px[t])          (Kalman gain)
        e[t]  = y[t] - w[t-1]ᵀ · x[t]                (error a priori)
        w[t]  = w[t-1] + K[t] · e[t]                  (actualización pesos)
        P[t]  = (P[t-1] - K[t] · Px[t]ᵀ) / λ         (actualización covarianza)

    P se inicializa como δ·I. Un δ grande = alta incertidumbre inicial = el filtro
    aprende muy rápido al principio y luego se estabiliza. Para precios ~100,
    δ=1000 funciona bien.

    Parámetros
    ----------
    n_taps : int
        Número de coeficientes.
    lam : float
        Forgetting factor en (0, 1]. Controla la "ventana efectiva".
    delta : float
        Inicialización de P = δ·I. Un δ grande acelera la convergencia inicial.
    """

    def __init__(self, n_taps: int = 1, lam: float = 0.99, delta: float = 1000.0):
        if not 0 < lam <= 1:
            raise ValueError("lam must be in (0, 1]")
        self.n_taps = n_taps
        self.lam = lam
        self.delta = delta
        self.weights = np.zeros(n_taps)
        # P = δ·I: alta incertidumbre inicial → aprendizaje rápido al principio
        self.P = delta * np.eye(n_taps)

    def update(self, x: np.ndarray, y: float) -> tuple[float, float]:
        x = np.asarray(x, dtype=float)

        # Error a priori: usando pesos ANTES de la actualización
        y_hat = np.dot(self.weights, x)
        error = y - y_hat

        # Kalman gain: K = P·x / (λ + xᵀ·P·x)
        Px = self.P @ x
        denom = self.lam + float(x @ Px)
        K = Px / denom

        # Actualizar pesos: w = w + K·e
        self.weights = self.weights + K * error

        # Actualizar covarianza: P = (P - K·Pxᵀ) / λ
        # Nota: K·Pxᵀ = outer(K, Px) es la corrección de rango 1
        # que reduce la incertidumbre en la dirección de x.
        self.P = (self.P - np.outer(K, Px)) / self.lam

        return y_hat, error

    def run(self, X: np.ndarray, y: np.ndarray) -> dict:
        n_samples = len(y)
        y_hat = np.zeros(n_samples)
        errors = np.zeros(n_samples)
        weights_history = np.zeros((n_samples, self.n_taps))
        for t in range(n_samples):
            y_hat[t], errors[t] = self.update(X[t], y[t])
            weights_history[t] = self.weights.copy()
        return {"y_hat": y_hat, "errors": errors, "weights_history": weights_history}


class LeakyNLMSFilter:
    """
    NLMS con término de "leakage" (regularización L2 implícita).

    Ecuación de actualización:
        w[t+1] = (1 - ρ) · w[t] + μ · e[t] · x[t] / (‖x[t]‖² + ε)

    ¿Por qué añadir leakage para pairs trading?

    El NLMS estándar puede dejar que el hedge ratio β derive lentamente
    hacia valores extremos durante periodos tranquilos (pocas señales, errores
    pequeños). Si la relación V/MA es β≈1.2, el filtro no tiene "ancla" que
    lo mantenga ahí si los precios son estables.

    El término (1-ρ) actúa como un decay exponencial: en cada paso, el peso
    se "olvida" en una fracción ρ. Esto equivale a resolver el problema:

        min_w  E[e²(t)] + (ρ/μ) · ‖w‖²     (penalización L2)

    Para ρ pequeño (0.0001-0.001):
    - El filtro se comporta casi exactamente como NLMS estándar
    - Pero los pesos no pueden derivar sin límite
    - En ausencia de señal, los pesos decaen suavemente hacia cero
      → el hedge ratio se estabiliza cerca de sus valores históricos

    Parámetros
    ----------
    n_taps : int
        Número de coeficientes.
    mu : float
        Step size en (0, 1].
    rho : float
        Leakage factor en [0, 1). 0 = NLMS estándar. 0.001 = leakage moderado.
        Un rho demasiado alto hace que los pesos decaigan a cero → mal hedge ratio.
    eps : float
        Regularización anti-división-por-cero.
    """

    def __init__(self, n_taps: int = 1, mu: float = 0.1, rho: float = 0.0001,
                 eps: float = 1e-8):
        if not 0 < mu <= 1:
            raise ValueError("mu must be in (0, 1]")
        if not 0 <= rho < 1:
            raise ValueError("rho must be in [0, 1)")
        self.n_taps = n_taps
        self.mu = mu
        self.rho = rho
        self.eps = eps
        self.weights = np.zeros(n_taps)

    def update(self, x: np.ndarray, y: float) -> tuple[float, float]:
        x = np.asarray(x, dtype=float)
        y_hat = np.dot(self.weights, x)
        error = y - y_hat
        norm_factor = np.dot(x, x) + self.eps
        # (1-ρ) decay + actualización NLMS estándar
        self.weights = (1 - self.rho) * self.weights + self.mu * error * x / norm_factor
        return y_hat, error

    def run(self, X: np.ndarray, y: np.ndarray) -> dict:
        n_samples = len(y)
        y_hat = np.zeros(n_samples)
        errors = np.zeros(n_samples)
        weights_history = np.zeros((n_samples, self.n_taps))
        for t in range(n_samples):
            y_hat[t], errors[t] = self.update(X[t], y[t])
            weights_history[t] = self.weights.copy()
        return {"y_hat": y_hat, "errors": errors, "weights_history": weights_history}


class VSNLMSFilter:
    """
    Variable Step-Size NLMS (inspirado en Kwong & Johnston, 1992).

    La limitación más fundamental del NLMS estándar es el μ fijo.
    Este filtro adapta μ automáticamente en función de la correlación
    entre errores consecutivos:

        - Si e[t] y e[t-1] tienen el MISMO signo:
          → el filtro está sistemáticamente mal (underfitting)
          → subir μ para adaptarse más rápido

        - Si e[t] y e[t-1] tienen SIGNOS OPUESTOS:
          → el filtro está persiguiendo ruido (overfitting)
          → bajar μ para estabilizarse

    Ecuaciones:

        corr[t] = sign(e[t]) · sign(e[t-1])      (correlación de signo: ±1)
        μ[t] = clip(α · μ[t-1] + γ · corr[t], μ_min, μ_max)
        w[t+1] = w[t] + μ[t] · e[t] · x[t] / (‖x[t]‖² + ε)

    Ventajas de usar sign() en lugar del producto directo e[t]·e[t-1]:
    - Es completamente independiente de la escala de los precios
    - Es robusto a outliers (un día con error enorme no dispara μ)
    - El parámetro γ tiene una interpretación clara: incremento máximo de μ por paso

    Parámetros
    ----------
    n_taps : int
        Número de coeficientes.
    mu_init : float
        Step size inicial.
    mu_min : float
        Step size mínimo (nunca aprenderá más lento que esto).
    mu_max : float
        Step size máximo (cap para evitar inestabilidad).
    alpha : float
        Factor de olvido del step size (momentum). Cercano a 1.
    gamma : float
        Magnitud del ajuste de μ por paso. Pequeño para cambios suaves.
    eps : float
        Regularización anti-división-por-cero.
    """

    def __init__(self, n_taps: int = 1, mu_init: float = 0.05,
                 mu_min: float = 0.001, mu_max: float = 0.5,
                 alpha: float = 0.999, gamma: float = 0.05,
                 eps: float = 1e-8):
        self.n_taps = n_taps
        self.mu = mu_init
        self.mu_min = mu_min
        self.mu_max = mu_max
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.weights = np.zeros(n_taps)
        self._prev_error = 0.0  # e[t-1] para calcular la correlación

    def update(self, x: np.ndarray, y: float) -> tuple[float, float]:
        x = np.asarray(x, dtype=float)
        y_hat = np.dot(self.weights, x)
        error = y - y_hat

        # Correlación de signo entre error actual y anterior: ±1
        # +1 → mismo signo → underfitting → subir μ
        # -1 → signo opuesto → ruido → bajar μ
        corr = np.sign(error) * np.sign(self._prev_error)

        # Actualizar step size
        self.mu = float(np.clip(
            self.alpha * self.mu + self.gamma * corr,
            self.mu_min, self.mu_max
        ))

        # Actualizar pesos con el μ recién calculado
        norm_factor = np.dot(x, x) + self.eps
        self.weights = self.weights + self.mu * error * x / norm_factor

        self._prev_error = error
        return y_hat, error

    def run(self, X: np.ndarray, y: np.ndarray) -> dict:
        n_samples = len(y)
        y_hat = np.zeros(n_samples)
        errors = np.zeros(n_samples)
        weights_history = np.zeros((n_samples, self.n_taps))
        mu_history = np.zeros(n_samples)
        for t in range(n_samples):
            y_hat[t], errors[t] = self.update(X[t], y[t])
            weights_history[t] = self.weights.copy()
            mu_history[t] = self.mu
        return {"y_hat": y_hat, "errors": errors,
                "weights_history": weights_history,
                "mu_history": mu_history}
