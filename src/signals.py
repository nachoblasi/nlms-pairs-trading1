"""
signals.py — Generación de señales de trading a partir del spread adaptativo.

Funciones:
    compute_zscore()    — normaliza el spread con una ventana rolling
    generate_signals()  — máquina de estados: flat / long / short
"""

import numpy as np
import pandas as pd


def compute_zscore(spread: np.ndarray, lookback: int = 50) -> np.ndarray:
    """
    Calcula el z-score rolling del spread.

    Fórmula: z[t] = (spread[t] - mean(spread[t-L:t])) / std(spread[t-L:t])

    Las primeras (lookback-1) posiciones son NaN: no hay suficientes datos
    para calcular la ventana completa, así que no se generan señales.

    Parámetros
    ----------
    spread   : np.ndarray — spread adaptativo (errores del NLMS)
    lookback : int        — tamaño de la ventana rolling (días)

    Retorna
    -------
    np.ndarray — z-scores (primeros lookback-1 valores son NaN)
    """
    s    = pd.Series(spread)
    mean = s.rolling(lookback, min_periods=lookback).mean()
    std  = s.rolling(lookback, min_periods=lookback).std()
    return ((s - mean) / std).values


def generate_signals(
    zscore: np.ndarray,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5,
    zscore_sizing: bool = False,
    max_leverage: float = 3.0,
    stop_loss_z: float = None,
) -> np.ndarray:
    """
    Genera señales de trading a partir del z-score.

    Máquina de estados con tres estados:
        flat  ( 0): sin posición
        long  (+1): largo en el spread (comprar Y, vender β·X)
        short (-1): corto en el spread (vender Y, comprar β·X)

    Lógica:
        Desde flat:  z < -entry → long  |  z > +entry → short
        Desde long:  z > -exit          → cerrar
        Desde short: z < +exit          → cerrar

    Con zscore_sizing=True la posición escala con |z| / entry_threshold
    (posiciones más extremas tienen mayor peso).

    Parámetros
    ----------
    zscore          : np.ndarray
    entry_threshold : float — umbral de entrada (valor absoluto)
    exit_threshold  : float — umbral de salida (valor absoluto)
    zscore_sizing   : bool  — escalar posición por magnitud del z-score
    max_leverage    : float — cap al multiplicador de posición
    stop_loss_z     : float — stop-loss (opcional, en unidades de z-score)

    Retorna
    -------
    np.ndarray — señales: +1 / -1 / 0 (o fraccionales si zscore_sizing=True)
    """
    n        = len(zscore)
    signals  = np.zeros(n)
    position = 0

    for t in range(n):
        if np.isnan(zscore[t]):
            signals[t] = 0
            continue

        if position == 0:
            if zscore[t] < -entry_threshold:
                position = 1
            elif zscore[t] > entry_threshold:
                position = -1

        elif position == 1:
            if zscore[t] > -exit_threshold:
                position = 0
            elif stop_loss_z is not None and zscore[t] < -stop_loss_z:
                position = 0

        elif position == -1:
            if zscore[t] < exit_threshold:
                position = 0
            elif stop_loss_z is not None and zscore[t] > stop_loss_z:
                position = 0

        if zscore_sizing and position != 0:
            size       = min(abs(zscore[t]) / entry_threshold, max_leverage)
            signals[t] = position * size
        else:
            signals[t] = position

    return signals
