"""
cointegration.py — Tests estadísticos de cointegración y mean-reversion.

Funciones:
    johansen_cointegration()  — test de Johansen al nivel de confianza dado
    compute_halflife()        — half-life del spread vía AR(1)
"""

import numpy as np
from statsmodels.tsa.vector_ar.vecm import coint_johansen


def johansen_cointegration(price_x: np.ndarray, price_y: np.ndarray,
                           significance: float = 0.05) -> dict:
    """
    Test de cointegración de Johansen sobre un par de series de precios.

    Evalúa si price_x y price_y están cointegradas usando el test de la traza
    (trace statistic) de Johansen. Si al menos un vector de cointegración es
    significativo, el par pasa el filtro.

    Parámetros
    ----------
    price_x, price_y : np.ndarray
        Series de precios. Deben ser I(1) (no estacionarias individualmente).
    significance : float
        Nivel de significancia. 0.05 → 95% confianza, 0.01 → 99% confianza.

    Retorna
    -------
    dict con:
        cointegrated : bool   — True si el par está cointegrado al nivel dado
        trace_stat   : float  — estadístico de la traza (r=0)
        crit_value   : float  — valor crítico al nivel de significancia
        confidence   : str    — nivel de confianza ("90%", "95%", "99%")
    """
    data = np.column_stack([price_x, price_y])

    # det_order=0: constante restringida al espacio de cointegración.
    # k_ar_diff=1: un lag de diferencias (suficiente para precios diarios).
    result = coint_johansen(data, det_order=0, k_ar_diff=1)

    # statsmodels devuelve valores críticos en columnas 0,1,2
    # correspondientes a significancias 0.10, 0.05, 0.01
    sig_map = {0.10: 0, 0.05: 1, 0.01: 2}
    col = sig_map.get(significance, 1)

    trace_stat   = float(result.lr1[0])
    crit_value   = float(result.cvt[0, col])
    cointegrated = trace_stat > crit_value

    sig_labels = {0: "90%", 1: "95%", 2: "99%"}

    return {
        "cointegrated": cointegrated,
        "trace_stat":   trace_stat,
        "crit_value":   crit_value,
        "confidence":   sig_labels[col],
    }


def compute_halflife(price_x: np.ndarray, price_y: np.ndarray,
                     min_hl: float = 3.0, max_hl: float = 60.0) -> dict:
    """
    Calcula el half-life de mean-reversion del spread entre dos series de precios.

    Método:
        1. Estimar el spread via OLS: spread = price_y - β·price_x
        2. Ajustar AR(1): Δspread[t] = ρ·spread[t-1] + ε
        3. Half-life = -log(2) / log(1 + ρ)

    Interpretación:
        ρ < 0 → spread mean-revierte (necesario para trading)
        ρ ≥ 0 → spread tiene tendencia o random walk → no tradeable

        half_life < min_hl → demasiado rápido, ruido puro
        half_life > max_hl → demasiado lento, el régimen cambia antes de revertir

    Retorna
    -------
    dict con:
        halflife : float — half-life en días (inf si ρ ≥ 0)
        rho      : float — coeficiente AR(1) del spread
        in_range : bool  — True si el half-life está en [min_hl, max_hl]
        min_hl   : float
        max_hl   : float
    """
    beta   = np.cov(price_x, price_y)[0, 1] / (np.var(price_x) + 1e-10)
    spread = price_y - beta * price_x
    spread = spread - spread.mean()

    y_ar = np.diff(spread)
    x_ar = spread[:-1]
    rho  = np.cov(x_ar, y_ar)[0, 1] / (np.var(x_ar) + 1e-10)

    halflife = float("inf") if rho >= 0 else float(-np.log(2) / np.log(1 + rho))
    in_range = min_hl <= halflife <= max_hl

    return {
        "halflife": halflife,
        "rho":      rho,
        "in_range": in_range,
        "min_hl":   min_hl,
        "max_hl":   max_hl,
    }
