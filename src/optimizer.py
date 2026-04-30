"""
optimizer.py — Grid search y pipeline de ejecución de filtros.

Funciones:
    build_vsnlms()          — factory para VSNLMSFilter con params fijos del proyecto
    build_ml_filter()       — factory para ML_VSNLMSFilter con modelo entrenado
    run_filter_pipeline()   — filtro → z-score → señales → backtest (en una sola llamada)
    evaluate_on_window()    — evalúa una combinación de parámetros en una ventana
    optimize_on_train()     — grid search completo sobre la ventana de train
"""

import itertools

import numpy as np
import pandas as pd

from src.nlms import VSNLMSFilter
from src.ml_nlms import ML_VSNLMSFilter, DEFAULT_FEATURE_WINDOW
from src.signals import compute_zscore, generate_signals
from src.backtest import backtest


# ── Configuración del grid (misma que walk_forward_ml_mu.py) ──────────────────

PARAM_GRID = {
    "filter_param": [0.01, 0.05, 0.1, 0.2],   # mu_init para VS-NLMS
    "lookback":     [30, 60, 90, 120],
    "entry_z":      [1.5],
    "exit_z":       [0.25, 0.5, 0.75],
}

VSNLMS_FIXED = {
    "mu_min": 0.001,
    "mu_max": 0.5,
    "alpha":  0.990,
    "gamma":  0.05,
}


# ── Factories ─────────────────────────────────────────────────────────────────

def build_vsnlms(mu_init: float) -> VSNLMSFilter:
    """Crea un VSNLMSFilter con los parámetros fijos del proyecto."""
    return VSNLMSFilter(
        n_taps  = 1,
        mu_init = mu_init,
        mu_min  = VSNLMS_FIXED["mu_min"],
        mu_max  = VSNLMS_FIXED["mu_max"],
        alpha   = VSNLMS_FIXED["alpha"],
        gamma   = VSNLMS_FIXED["gamma"],
    )


def build_ml_filter(model, scaler, mu_fallback: float) -> ML_VSNLMSFilter:
    """Crea un ML_VSNLMSFilter con el modelo entrenado."""
    return ML_VSNLMSFilter(
        n_taps         = 1,
        model          = model,
        scaler         = scaler,
        mu_min         = VSNLMS_FIXED["mu_min"],
        mu_max         = VSNLMS_FIXED["mu_max"],
        feature_window = DEFAULT_FEATURE_WINDOW,
        mu_fallback    = mu_fallback,
    )


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_filter_pipeline(
    df_window: pd.DataFrame,
    filt,
    lookback: int,
    entry_z: float,
    exit_z: float,
) -> dict:
    """
    Ejecuta el pipeline completo: filtro → z-score → señales → backtest.

    Funciona con cualquier filtro que implemente run(X, y).
    Usado tanto para el baseline (VSNLMSFilter) como para el ML (ML_VSNLMSFilter).

    Retorna
    -------
    dict con: bt, hedge_ratios, mu_history, spread, zscore, signals
    """
    X = df_window["price_x"].values.reshape(-1, 1)
    y = df_window["price_y"].values

    result       = filt.run(X, y)
    hedge_ratios = result["weights_history"][:, 0]
    spread       = result["errors"]
    mu_history   = result["mu_history"]

    zscore  = compute_zscore(spread, lookback=lookback)
    signals = generate_signals(zscore, entry_threshold=entry_z,
                               exit_threshold=exit_z, zscore_sizing=True)
    bt      = backtest(df_window, signals, hedge_ratios)

    return {
        "bt":           bt,
        "hedge_ratios": hedge_ratios,
        "mu_history":   mu_history,
        "spread":       spread,
        "zscore":       zscore,
        "signals":      signals,
    }


# ── Grid search ───────────────────────────────────────────────────────────────

def evaluate_on_window(
    df_window: pd.DataFrame,
    filter_param: float,
    lookback: int,
    entry_z: float,
    exit_z: float,
) -> dict:
    """
    Evalúa el VS-NLMS baseline en una ventana con una combinación de parámetros.

    Retorna dict con 'sharpe', 'max_dd' y 'score' = Sharpe + 0.5·MaxDD.
    """
    filt     = build_vsnlms(filter_param)
    pipeline = run_filter_pipeline(df_window, filt, lookback, entry_z, exit_z)

    returns = pipeline["bt"]["strategy_return"].values
    returns = returns[~np.isnan(returns)]

    if len(returns) == 0 or np.std(returns) == 0:
        return {"sharpe": -999, "max_dd": -1.0, "score": -999}

    total   = (1 + returns).prod() - 1
    ann_ret = (1 + total) ** (252 / len(returns)) - 1
    ann_vol = np.std(returns) * np.sqrt(252)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else 0

    cum    = np.cumprod(1 + returns)
    peak   = np.maximum.accumulate(cum)
    max_dd = float(np.min((cum - peak) / peak))

    return {"sharpe": sharpe, "max_dd": max_dd, "score": sharpe + 0.5 * max_dd}


def optimize_on_train(df_train: pd.DataFrame) -> dict | None:
    """
    Grid search sobre PARAM_GRID usando únicamente datos de train.

    Los parámetros óptimos (lookback, entry_z, exit_z) se aplican también
    al ML filter en el test → comparación justa sobre la misma configuración.

    Retorna
    -------
    dict con los mejores parámetros y métricas, o None si ninguno es válido.
    """
    best_score  = -999
    best_params = None

    combos = list(itertools.product(
        PARAM_GRID["filter_param"],
        PARAM_GRID["lookback"],
        PARAM_GRID["entry_z"],
        PARAM_GRID["exit_z"],
    ))

    for filter_param, lookback, entry_z, exit_z in combos:
        if exit_z >= entry_z:
            continue

        metrics = evaluate_on_window(df_train, filter_param, lookback, entry_z, exit_z)

        if metrics["score"] > best_score:
            best_score  = metrics["score"]
            best_params = {
                "filter_param": filter_param,
                "lookback":     lookback,
                "entry_z":      entry_z,
                "exit_z":       exit_z,
                "train_sharpe": metrics["sharpe"],
                "train_max_dd": metrics["max_dd"],
            }

    return best_params
