"""
optimize_vsnlms.py — Optimización de hiperparámetros internos del VSNLMSFilter.

Los hiperparámetros del VSNLMS (alpha, gamma, mu_min, mu_max) están fijos
en walk_forward.py pero nunca se han optimizado. Este script los busca.

¿Por qué es válido optimizarlos por separado?
    - Los parámetros de ESTRATEGIA (lookback, entry_z, exit_z) se optimizan
      dentro del walk-forward en cada fold → riesgo de overfitting por fold.
    - Los hiperparámetros del FILTRO son globales: afectan cómo el filtro
      aprende, no qué umbrales usa la estrategia. Son más estables entre
      periodos y menos propensos a overfitting si se optimizan sobre el
      OOS agregado de todos los folds.

Metodología:
    Para cada combinación (alpha, gamma, mu_min, mu_max):
        1. Corre el walk-forward completo (13 folds, OOS honest)
        2. Recoge el Sharpe OOS agregado
    Reporta las mejores combinaciones.

Ejecutar: python optimize_vsnlms.py
"""

import numpy as np
import pandas as pd
import itertools
import time

from src.nlms import VSNLMSFilter
from src.strategy import compute_zscore, generate_signals, backtest


# ══════════════════════════════════════════════════════════════
# CONFIGURACIÓN (misma que walk_forward.py)
# ══════════════════════════════════════════════════════════════

TRAIN_DAYS = 756
TEST_DAYS = 252

# Parámetros de estrategia (entry_z fijo en 1.5, igual que walk_forward)
STRATEGY_GRID = {
    "mu_init":  [0.01, 0.05, 0.1, 0.2],
    "lookback": [30, 60, 90, 120],
    "entry_z":  [1.5],
    "exit_z":   [0.25, 0.5, 0.75],
}

# ── Grid de hiperparámetros del VSNLMS a explorar ──
#
# alpha: factor de olvido del step size. Más alto = μ cambia más lentamente.
#   0.99   → μ puede cambiar bastante cada día
#   0.999  → μ cambia suavemente (actual)
#   0.9999 → μ casi no cambia → similar a NLMS estándar
#
# gamma: magnitud del ajuste de μ por paso (en unidades de μ).
#   0.01 → cambios muy suaves
#   0.05 → moderado (actual)
#   0.1  → adaptación agresiva
#   0.2  → muy agresivo
#
# mu_min: suelo del step size. Nunca aprende más lento que esto.
# mu_max: techo del step size. Cap para evitar inestabilidad.

HYPERPARAM_GRID = {
    "alpha":  [0.990, 0.995, 0.999, 0.9999],
    "gamma":  [0.01, 0.02, 0.05, 0.1, 0.2],
    "mu_min": [0.001, 0.005, 0.01],
    "mu_max": [0.3, 0.5, 0.8],
}


def run_walkforward(alpha: float, gamma: float, mu_min: float, mu_max: float,
                    df: pd.DataFrame) -> dict:
    """
    Ejecuta el walk-forward completo con los hiperparámetros dados.
    Devuelve las métricas OOS agregadas.
    """
    total_days = len(df)
    n_windows = (total_days - TRAIN_DAYS) // TEST_DAYS

    all_test_results = []

    for fold in range(n_windows):
        train_start = fold * TEST_DAYS
        train_end = train_start + TRAIN_DAYS
        test_end = train_end + TEST_DAYS

        if test_end > total_days:
            break

        df_train = df.iloc[train_start:train_end].reset_index(drop=True)
        df_full = df.iloc[train_start:test_end].reset_index(drop=True)

        # ── Optimizar en train ──
        best_score = -999
        best_params = None

        for mu_init, lookback, entry_z, exit_z in itertools.product(
            STRATEGY_GRID["mu_init"],
            STRATEGY_GRID["lookback"],
            STRATEGY_GRID["entry_z"],
            STRATEGY_GRID["exit_z"],
        ):
            if exit_z >= entry_z:
                continue

            filt = VSNLMSFilter(n_taps=1, mu_init=mu_init,
                                mu_min=mu_min, mu_max=mu_max,
                                alpha=alpha, gamma=gamma)
            X = df_train["price_x"].values.reshape(-1, 1)
            y = df_train["price_y"].values
            result = filt.run(X, y)

            zscore = compute_zscore(result["errors"], lookback=lookback)
            signals = generate_signals(zscore, entry_threshold=entry_z,
                                       exit_threshold=exit_z, zscore_sizing=True)
            bt = backtest(df_train, signals, result["weights_history"][:, 0])
            rets = bt["strategy_return"].dropna().values

            if len(rets) == 0 or np.std(rets) == 0:
                continue

            tot = (1 + rets).prod() - 1
            ann_r = (1 + tot) ** (252 / len(rets)) - 1
            ann_v = np.std(rets) * np.sqrt(252)
            sharpe = ann_r / ann_v if ann_v > 0 else 0
            cum = np.cumprod(1 + rets)
            max_dd = np.min((cum - np.maximum.accumulate(cum)) / np.maximum.accumulate(cum))
            score = sharpe + 0.5 * max_dd

            if score > best_score:
                best_score = score
                best_params = {"mu_init": mu_init, "lookback": lookback,
                               "entry_z": entry_z, "exit_z": exit_z}

        if best_params is None:
            continue

        # ── Testear en test (out-of-sample) ──
        filt = VSNLMSFilter(n_taps=1, mu_init=best_params["mu_init"],
                            mu_min=mu_min, mu_max=mu_max,
                            alpha=alpha, gamma=gamma)
        X = df_full["price_x"].values.reshape(-1, 1)
        y = df_full["price_y"].values
        result = filt.run(X, y)

        zscore = compute_zscore(result["errors"], lookback=best_params["lookback"])
        signals = generate_signals(zscore, entry_threshold=best_params["entry_z"],
                                   exit_threshold=best_params["exit_z"],
                                   zscore_sizing=True)
        bt = backtest(df_full, signals, result["weights_history"][:, 0])
        all_test_results.append(bt.iloc[TRAIN_DAYS:].copy())

    if not all_test_results:
        return {"sharpe": -999, "total_return": -1, "max_dd": -1, "degradation": 1}

    combined = pd.concat(all_test_results, ignore_index=True)
    oos_rets = combined["strategy_return"].dropna().values

    if len(oos_rets) == 0 or np.std(oos_rets) == 0:
        return {"sharpe": -999, "total_return": -1, "max_dd": -1, "degradation": 1}

    oos_total = (1 + oos_rets).prod() - 1
    oos_ann = (1 + oos_total) ** (252 / len(oos_rets)) - 1
    oos_vol = np.std(oos_rets) * np.sqrt(252)
    oos_sharpe = oos_ann / oos_vol if oos_vol > 0 else 0
    cum = np.cumprod(1 + oos_rets)
    max_dd = np.min((cum - np.maximum.accumulate(cum)) / np.maximum.accumulate(cum))

    return {"sharpe": oos_sharpe, "total_return": oos_total,
            "ann_return": oos_ann, "max_dd": max_dd}


def main():
    print("=" * 65)
    print("VSNLMS HYPERPARAMETER OPTIMIZATION")
    print("=" * 65)

    print("\n[1] Loading data...")
    try:
        df = pd.read_csv("data/visa_mastercard.csv", parse_dates=["date"])
        print(f"    Loaded {len(df)} rows")
    except FileNotFoundError:
        print("    ERROR: Run 'python run_real_data.py' first.")
        return

    combos = list(itertools.product(
        HYPERPARAM_GRID["alpha"],
        HYPERPARAM_GRID["gamma"],
        HYPERPARAM_GRID["mu_min"],
        HYPERPARAM_GRID["mu_max"],
    ))
    print(f"\n[2] Grid: {len(HYPERPARAM_GRID['alpha'])} alpha × "
          f"{len(HYPERPARAM_GRID['gamma'])} gamma × "
          f"{len(HYPERPARAM_GRID['mu_min'])} mu_min × "
          f"{len(HYPERPARAM_GRID['mu_max'])} mu_max = {len(combos)} combos")
    print(f"    Baseline (actual): alpha=0.999, gamma=0.05, mu_min=0.001, mu_max=0.5\n")

    results = []
    start = time.time()

    for i, (alpha, gamma, mu_min, mu_max) in enumerate(combos):
        metrics = run_walkforward(alpha, gamma, mu_min, mu_max, df)
        results.append({
            "alpha": alpha, "gamma": gamma,
            "mu_min": mu_min, "mu_max": mu_max,
            **metrics
        })

        elapsed = time.time() - start
        eta = elapsed / (i + 1) * (len(combos) - i - 1)
        print(f"  [{i+1:3d}/{len(combos)}] α={alpha} γ={gamma:.2f} "
              f"μ_min={mu_min} μ_max={mu_max}  │  "
              f"Sharpe={metrics['sharpe']:+.3f}  "
              f"Return={metrics['total_return']:+.1%}  "
              f"DD={metrics['max_dd']:.1%}  "
              f"ETA: {eta:.0f}s")

    # ── Resultados ──
    df_res = pd.DataFrame(results).sort_values("sharpe", ascending=False)

    print(f"\n[3] TOP 10 combinaciones por Sharpe OOS:")
    print(f"\n  {'alpha':>7} {'gamma':>6} {'mu_min':>7} {'mu_max':>7} │ "
          f"{'Sharpe':>7} {'Return':>8} {'MaxDD':>8}")
    print("  " + "─" * 65)
    for _, row in df_res.head(10).iterrows():
        marker = " ← BEST" if _ == df_res.index[0] else ""
        print(f"  {row['alpha']:>7.4f} {row['gamma']:>6.3f} {row['mu_min']:>7.4f} "
              f"{row['mu_max']:>7.2f} │ "
              f"{row['sharpe']:>7.3f} {row['total_return']:>8.1%} "
              f"{row['max_dd']:>8.1%}{marker}")

    best = df_res.iloc[0]
    print(f"\n[4] Mejor combinación encontrada:")
    print(f"    alpha  = {best['alpha']}")
    print(f"    gamma  = {best['gamma']}")
    print(f"    mu_min = {best['mu_min']}")
    print(f"    mu_max = {best['mu_max']}")
    print(f"    → Sharpe OOS: {best['sharpe']:+.3f}")
    print(f"    → Return OOS: {best['total_return']:+.1%}")
    print(f"    → Max DD:     {best['max_dd']:.1%}")

    baseline = df_res[(df_res["alpha"] == 0.999) & (df_res["gamma"] == 0.05) &
                      (df_res["mu_min"] == 0.001) & (df_res["mu_max"] == 0.5)]
    if not baseline.empty:
        b = baseline.iloc[0]
        print(f"\n    Baseline:  Sharpe={b['sharpe']:+.3f}  Return={b['total_return']:+.1%}")
        print(f"    Mejora:    ΔSharpe={best['sharpe']-b['sharpe']:+.3f}")

    df_res.to_csv("results/vsnlms_hyperparam_search.csv", index=False)
    print(f"\n    Resultados completos en results/vsnlms_hyperparam_search.csv")
    print(f"\n✓ Done! ({time.time()-start:.0f}s total)")


if __name__ == "__main__":
    main()
