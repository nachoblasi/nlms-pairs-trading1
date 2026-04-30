"""
walk_forward.py — Walk-Forward Optimization del NLMS Pairs Trading.

Este es el test más honesto que podemos hacer. En vez de optimizar
sobre TODOS los datos (que es hacer trampa porque usas el futuro),
hacemos esto:

    1. Tomas una ventana de entrenamiento (ej: 3 años)
    2. Optimizas los parámetros SOLO con esos datos
    3. Testeas con los datos SIGUIENTES (ej: 1 año) que el optimizador
       nunca ha visto
    4. Avanzas la ventana y repites

    Ejemplo con datos 2010-2026:

    Ventana 1: Train [2010─2012]  →  Test [2013]
    Ventana 2: Train [2011─2013]  →  Test [2014]
    Ventana 3: Train [2012─2014]  →  Test [2015]
    ...
    Ventana N: Train [2022─2024]  →  Test [2025]

    Los retornos de TODOS los periodos de test se concatenan.
    Ese retorno concatenado es "out-of-sample": el modelo nunca vio
    esos datos cuando eligió los parámetros.

Si el walk-forward da buenos resultados, la señal es robusta.
Si da malos resultados, el backtest original era overfitting.

Filtros disponibles (cambia FILTER_TYPE abajo):
    "nlms"   — NLMS estándar (baseline, μ fijo)
    "rls"    — Recursive Least Squares (forgetting factor λ, convergencia óptima)
    "leaky"  — Leaky NLMS (regularización L2, previene drift del hedge ratio)
    "vsnlms" — Variable Step-Size NLMS (μ adaptativo, resuelve el μ-fijo problem)

Requisitos: pip install yfinance statsmodels
Ejecutar:   python walk_forward.py
"""

import numpy as np
import pandas as pd
import itertools
import time

from src.nlms import NLMSFilter, RLSFilter, LeakyNLMSFilter, VSNLMSFilter
from src.strategy import compute_zscore, generate_signals, backtest


# ══════════════════════════════════════════════════════════════
# CONFIGURACIÓN — cambia aquí para probar diferentes filtros
# ══════════════════════════════════════════════════════════════

# Tipo de filtro a usar en esta ejecución:
#   "nlms"   → μ fijo (baseline)
#   "rls"    → Recursive Least Squares con forgetting factor λ
#   "leaky"  → NLMS con leakage (regularización)
#   "vsnlms" → NLMS con step size variable
FILTER_TYPE = "vsnlms"

# Ventana de entrenamiento: cuántos días usa el optimizador para buscar parámetros.
# 756 días ≈ 3 años de trading.
TRAIN_DAYS = 756

# Ventana de test: cuántos días se testea con los parámetros elegidos.
# 252 días ≈ 1 año de trading.
TEST_DAYS = 252

# ── Grids de parámetros por tipo de filtro ──
#
# Cada filtro tiene su propio "filter_param" que se optimiza:
#   nlms   → mu (step size)
#   rls    → lam (forgetting factor)
#   leaky  → mu (step size, rho fijo en 0.0001)
#   vsnlms → mu_init (step size inicial, alpha/gamma fijos)
#
# Los parámetros de estrategia (lookback, entry_z, exit_z) son comunes.

PARAM_GRIDS = {
    "nlms": {
        "filter_param": [0.05, 0.1, 0.15, 0.2],   # mu
        "lookback":     [30, 60, 90, 120],
        "entry_z":      [1.5, 2.0, 2.5, 3.0],
        "exit_z":       [0.25, 0.5, 0.75],
    },
    "rls": {
        # λ controla la ventana efectiva: 1/(1-λ) días
        # λ=0.990 → ~100 días, λ=0.995 → ~200 días, λ=0.999 → ~1000 días
        "filter_param": [0.990, 0.995, 0.998, 0.999],
        "lookback":     [30, 60, 90, 120],
        "entry_z":      [1.5, 2.0, 2.5, 3.0],
        "exit_z":       [0.25, 0.5, 0.75],
    },
    "leaky": {
        "filter_param": [0.05, 0.1, 0.15, 0.2],   # mu
        "lookback":     [30, 60, 90, 120],
        "entry_z":      [1.5, 2.0, 2.5, 3.0],
        "exit_z":       [0.25, 0.5, 0.75],
    },
    "vsnlms": {
        "filter_param": [0.01, 0.05, 0.1, 0.2],   # mu_init
        "lookback":     [30, 60, 90, 120],
        # entry_z fijado en 1.5: ganó en 9/13 folds del walk-forward anterior.
        # exit_z fijado en 0.25: ganó en 7/13 folds (el más frecuente).
        # Fijar ambos reduce el grid de 192 → 16 combos/fold → menos overfitting.
        "entry_z":      [1.5],
        "exit_z":       [0.25, 0.5, 0.75],
    },
}

PARAM_GRID = PARAM_GRIDS[FILTER_TYPE]


def build_filter(filter_type: str, filter_param: float) -> object:
    """
    Crea una instancia del filtro adecuado con el parámetro principal.

    Parámetros secundarios de cada filtro:
    - RLS:    delta=1000 (inicialización de P, alta incertidumbre → aprende rápido)
    - Leaky:  rho=0.0001 (leakage mínimo pero efectivo para prevenir drift)
    - VSNLMS: mu_min=0.001, mu_max=0.5, alpha=0.990, gamma=0.05  ← optimized
    """
    if filter_type == "nlms":
        return NLMSFilter(n_taps=1, mu=filter_param)
    elif filter_type == "rls":
        return RLSFilter(n_taps=1, lam=filter_param, delta=1000.0)
    elif filter_type == "leaky":
        return LeakyNLMSFilter(n_taps=1, mu=filter_param, rho=0.0001)
    elif filter_type == "vsnlms":
        return VSNLMSFilter(n_taps=1, mu_init=filter_param,
                            mu_min=0.001, mu_max=0.5,
                            alpha=0.990, gamma=0.05)
    else:
        raise ValueError(f"Unknown filter_type: {filter_type}")


def evaluate_on_window(df_window: pd.DataFrame, filter_param: float,
                       lookback: int, entry_z: float, exit_z: float) -> dict:
    """
    Ejecuta el pipeline filtro + estrategia sobre una ventana de datos
    y devuelve métricas numéricas.

    Es similar a evaluate_params() de optimize.py pero opera sobre
    un subconjunto de datos (una ventana temporal).
    """
    filt = build_filter(FILTER_TYPE, filter_param)
    X = df_window["price_x"].values.reshape(-1, 1)
    y = df_window["price_y"].values
    result = filt.run(X, y)

    hedge_ratios = result["weights_history"][:, 0]
    adaptive_spread = result["errors"]

    zscore = compute_zscore(adaptive_spread, lookback=lookback)
    signals = generate_signals(zscore, entry_threshold=entry_z, exit_threshold=exit_z,
                               zscore_sizing=True)

    bt = backtest(df_window, signals, hedge_ratios)
    returns = bt["strategy_return"].values
    returns = returns[~np.isnan(returns)]

    if len(returns) == 0 or np.std(returns) == 0:
        return {"sharpe": -999, "max_dd": -1.0, "score": -999}

    total_return = (1 + returns).prod() - 1
    ann_return = (1 + total_return) ** (252 / len(returns)) - 1
    ann_vol = np.std(returns) * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0

    cum = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cum)
    drawdown = (cum - peak) / peak
    max_dd = np.min(drawdown)

    score = sharpe + 0.5 * max_dd

    return {"sharpe": sharpe, "max_dd": max_dd, "score": score}


def optimize_on_train(df_train: pd.DataFrame) -> dict:
    """
    Encuentra los mejores parámetros usando SOLO los datos de entrenamiento.

    Recorre el grid de parámetros y devuelve la combinación con mejor score.
    El periodo de test NUNCA se usa aquí — esa es la clave de walk-forward.
    """
    best_score = -999
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
            best_score = metrics["score"]
            best_params = {
                "filter_param": filter_param,
                "lookback": lookback,
                "entry_z": entry_z,
                "exit_z": exit_z,
                "train_score": best_score,
                "train_sharpe": metrics["sharpe"],
                "train_max_dd": metrics["max_dd"],
            }

    return best_params


def test_with_params(df_full_window: pd.DataFrame, params: dict,
                     train_end_idx: int) -> pd.DataFrame:
    """
    Aplica el filtro sobre la ventana COMPLETA (train + test),
    pero solo devuelve los retornos del periodo de TEST.

    ¿Por qué correr el filtro sobre train + test juntos?
    Porque el filtro necesita warm-up. Si lo arrancáramos solo en el
    periodo de test, los primeros meses serían basura (el filtro
    estaría convergiendo). Al correrlo desde el inicio del train,
    cuando llega al test ya está "caliente" y estimando bien.

    Pero las SEÑALES y RETORNOS solo los contamos desde el test.
    Los parámetros se eligieron SOLO con el train → no hay trampa.
    """
    filter_param = params["filter_param"]
    lookback = params["lookback"]
    entry_z = params["entry_z"]
    exit_z = params["exit_z"]

    filt = build_filter(FILTER_TYPE, filter_param)
    X = df_full_window["price_x"].values.reshape(-1, 1)
    y = df_full_window["price_y"].values
    result = filt.run(X, y)

    hedge_ratios = result["weights_history"][:, 0]
    adaptive_spread = result["errors"]

    zscore = compute_zscore(adaptive_spread, lookback=lookback)
    signals = generate_signals(zscore, entry_threshold=entry_z, exit_threshold=exit_z,
                               zscore_sizing=True)

    bt = backtest(df_full_window, signals, hedge_ratios)

    # Solo devolvemos el periodo de TEST
    test_results = bt.iloc[train_end_idx:].copy()
    return test_results


def main():
    print("=" * 65)
    print(f"WALK-FORWARD OPTIMIZATION — NLMS PAIRS TRADING (V / MA)")
    print(f"Filter: {FILTER_TYPE.upper()}")
    print("=" * 65)

    # ── Cargar datos ──
    print("\n[1] Loading data...")
    try:
        df = pd.read_csv("data/visa_mastercard.csv", parse_dates=["date"])
        print(f"    Loaded {len(df)} rows")
    except FileNotFoundError:
        print("    ERROR: Run 'python run_real_data.py' first to download data.")
        return

    total_days = len(df)
    n_windows = (total_days - TRAIN_DAYS) // TEST_DAYS

    grid_size = (len(PARAM_GRID["filter_param"]) * len(PARAM_GRID["lookback"]) *
                 len(PARAM_GRID["entry_z"]) * len(PARAM_GRID["exit_z"]))

    print(f"\n[2] Walk-forward configuration:")
    print(f"    Total days:      {total_days}")
    print(f"    Train window:    {TRAIN_DAYS} days (~{TRAIN_DAYS//252} years)")
    print(f"    Test window:     {TEST_DAYS} days (~{TEST_DAYS//252} year)")
    print(f"    Number of folds: {n_windows}")
    print(f"    Filter:          {FILTER_TYPE.upper()}")
    if FILTER_TYPE == "rls":
        print(f"    Filter param:    λ (forgetting factor) {PARAM_GRID['filter_param']}")
    elif FILTER_TYPE == "vsnlms":
        print(f"    Filter param:    mu_init {PARAM_GRID['filter_param']}")
    else:
        print(f"    Filter param:    mu {PARAM_GRID['filter_param']}")
    print(f"    Grid size:       {len(PARAM_GRID['filter_param'])} × "
          f"{len(PARAM_GRID['lookback'])} × "
          f"{len(PARAM_GRID['entry_z'])} × "
          f"{len(PARAM_GRID['exit_z'])} ≈ {grid_size} combos/fold")

    # ══════════════════════════════════════════════════════════
    # WALK-FORWARD LOOP
    # ══════════════════════════════════════════════════════════
    print(f"\n[3] Running walk-forward...\n")

    all_test_results = []
    window_summaries = []

    start_time = time.time()

    for fold in range(n_windows):
        train_start = fold * TEST_DAYS
        train_end = train_start + TRAIN_DAYS
        test_end = train_end + TEST_DAYS

        if test_end > total_days:
            break

        df_train = df.iloc[train_start:train_end].reset_index(drop=True)
        df_full = df.iloc[train_start:test_end].reset_index(drop=True)

        train_dates = (f"{df.iloc[train_start]['date'].strftime('%Y-%m-%d')} → "
                       f"{df.iloc[train_end-1]['date'].strftime('%Y-%m-%d')}")
        test_dates = (f"{df.iloc[train_end]['date'].strftime('%Y-%m-%d')} → "
                      f"{df.iloc[test_end-1]['date'].strftime('%Y-%m-%d')}")

        best_params = optimize_on_train(df_train)

        if best_params is None:
            print(f"    Fold {fold+1}: no valid parameters found, skipping.")
            continue

        test_results = test_with_params(df_full, best_params, TRAIN_DAYS)
        all_test_results.append(test_results)

        test_returns = test_results["strategy_return"].values
        test_returns = test_returns[~np.isnan(test_returns)]
        test_total = (1 + test_returns).prod() - 1 if len(test_returns) > 0 else 0
        test_vol = np.std(test_returns) * np.sqrt(252) if len(test_returns) > 0 else 0
        test_ann_ret = (1 + test_total) ** (252 / len(test_returns)) - 1 if len(test_returns) > 0 else 0
        test_sharpe = test_ann_ret / test_vol if test_vol > 0 else 0

        # Nombre legible del parámetro del filtro
        if FILTER_TYPE == "rls":
            param_str = f"λ={best_params['filter_param']:.3f}"
        elif FILTER_TYPE == "vsnlms":
            param_str = f"μ0={best_params['filter_param']:.2f}"
        else:
            param_str = f"μ={best_params['filter_param']:.2f}"

        summary = {
            "fold": fold + 1,
            "train_period": train_dates,
            "test_period": test_dates,
            "filter_param": best_params["filter_param"],
            "lookback": best_params["lookback"],
            "entry_z": best_params["entry_z"],
            "exit_z": best_params["exit_z"],
            "train_sharpe": best_params["train_sharpe"],
            "test_return": test_total,
            "test_sharpe": test_sharpe,
        }
        window_summaries.append(summary)

        print(f"    Fold {fold+1:2d}/{n_windows}  "
              f"Train: {train_dates}  Test: {test_dates}  "
              f"{param_str} lb={best_params['lookback']:3d} "
              f"ez={best_params['entry_z']:.1f} xz={best_params['exit_z']:.2f}  │  "
              f"Train Sharpe: {best_params['train_sharpe']:+.2f}  "
              f"Test Return: {test_total:+.1%}  "
              f"Test Sharpe: {test_sharpe:+.2f}")

    elapsed = time.time() - start_time
    print(f"\n    Completed in {elapsed:.0f}s")

    # ══════════════════════════════════════════════════════════
    # RESULTADOS AGREGADOS OUT-OF-SAMPLE
    # ══════════════════════════════════════════════════════════
    print(f"\n[4] Aggregated OUT-OF-SAMPLE results ({FILTER_TYPE.upper()}):")

    if not all_test_results:
        print("    No test results to aggregate.")
        return

    combined = pd.concat(all_test_results, ignore_index=True)
    oos_returns = combined["strategy_return"].values
    oos_returns = oos_returns[~np.isnan(oos_returns)]

    oos_total = (1 + oos_returns).prod() - 1
    oos_ann_ret = (1 + oos_total) ** (252 / len(oos_returns)) - 1
    oos_vol = np.std(oos_returns) * np.sqrt(252)
    oos_sharpe = oos_ann_ret / oos_vol if oos_vol > 0 else 0

    cum = np.cumprod(1 + oos_returns)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    oos_max_dd = np.min(dd)

    active = oos_returns[oos_returns != 0]
    oos_win_rate = np.sum(active > 0) / len(active) if len(active) > 0 else 0

    n_trades = np.sum(np.diff(combined["signal"].values) != 0)

    summary_df = pd.DataFrame(window_summaries)
    n_positive = (summary_df["test_return"] > 0).sum()
    n_negative = (summary_df["test_return"] <= 0).sum()

    print(f"\n    ┌─── Walk-Forward Performance (OUT-OF-SAMPLE) ────┐")
    print(f"    │  Filter:                  {FILTER_TYPE.upper():<20s}   │")
    print(f"    │  Total OOS return         {oos_total:>10.2%}           │")
    print(f"    │  Annualized return        {oos_ann_ret:>10.2%}           │")
    print(f"    │  Annualized volatility    {oos_vol:>10.2%}           │")
    print(f"    │  Sharpe ratio             {oos_sharpe:>10.2f}           │")
    print(f"    │  Max drawdown             {oos_max_dd:>10.2%}           │")
    print(f"    │  Win rate                 {oos_win_rate:>10.2%}           │")
    print(f"    │  Total trades             {n_trades:>10d}           │")
    print(f"    │  Profitable folds         {n_positive:>4d}/{n_positive+n_negative:<4d}            │")
    print(f"    └─────────────────────────────────────────────────┘")

    avg_train_sharpe = summary_df["train_sharpe"].mean()
    avg_test_sharpe = summary_df["test_sharpe"].mean()
    print(f"\n    Overfitting check:")
    print(f"    Avg Train Sharpe: {avg_train_sharpe:+.3f}")
    print(f"    Avg Test Sharpe:  {avg_test_sharpe:+.3f}")
    degradation = 1 - (avg_test_sharpe / avg_train_sharpe) if avg_train_sharpe != 0 else 0
    print(f"    Degradation:      {degradation:.0%}")

    if degradation < 0.3:
        print(f"    ✓ Low degradation — strategy appears ROBUST")
    elif degradation < 0.6:
        print(f"    ⚠ Moderate degradation — some overfitting likely")
    else:
        print(f"    ✗ High degradation — significant overfitting detected")

    summary_df.to_csv(f"results/walk_forward_{FILTER_TYPE}_folds.csv", index=False)
    combined.to_csv(f"results/walk_forward_{FILTER_TYPE}_oos.csv", index=False)

    print(f"\n    Fold details saved to results/walk_forward_{FILTER_TYPE}_folds.csv")
    print(f"    OOS results saved to results/walk_forward_{FILTER_TYPE}_oos.csv")
    print(f"\n✓ Done!")


if __name__ == "__main__":
    main()
