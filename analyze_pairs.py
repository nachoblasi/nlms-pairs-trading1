"""
analyze_pairs.py — Test de cointegración + Walk-Forward para múltiples pares.

Para cada par:
  1. Descarga datos desde yfinance (2010-actualidad)
  2. Test de cointegración Johansen (trace stat > crit 95%)
  3. Si cointegrado → walk-forward optimization con VSNLMS
  4. Imprime resultados

Parámetros FIJOS (no se optimizan, igual que en walk_forward.py):
  - mu_min=0.001, mu_max=0.5, alpha=0.990, gamma=0.05  (VSNLMS)

Parámetros OPTIMIZADOS por walk-forward (dependen del par):
  - filter_param (mu_init), lookback, entry_z, exit_z
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import itertools
import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from src.nlms import VSNLMSFilter
from src.strategy import compute_zscore, generate_signals, backtest

# ── Configuración ─────────────────────────────────────────────────
PAIRS = [
    ("V", "MA", "Visa / Mastercard"),
]

# Ventanas a probar por orden (de más larga a más corta).
# Se usa la primera donde el par sea cointegrado.
CANDIDATE_STARTS = ["2010-01-01", "2015-01-01", "2018-01-01", "2020-01-01"]
END_DATE    = "2025-12-31"
TRAIN_DAYS  = 756   # ~3 años
TEST_DAYS   = 252   # ~1 año

# Parámetros FIJOS del VSNLMS (no cambian entre pares)
VSNLMS_FIXED = dict(mu_min=0.001, mu_max=0.5, alpha=0.990, gamma=0.05)

# Grid de parámetros a optimizar (dependen del par)
PARAM_GRID = {
    "filter_param": [0.01, 0.05, 0.1, 0.2],
    "lookback":     [30, 60, 90, 120],
    "entry_z":      [1.5, 2.0, 2.5, 3.0],
    "exit_z":       [0.25, 0.5, 0.75],
}


# ── Utilidades ────────────────────────────────────────────────────

def download_pair(ticker_x: str, ticker_y: str, start_date: str) -> pd.DataFrame | None:
    """Descarga precios ajustados y devuelve DataFrame con columnas date, price_x, price_y."""
    data = yf.download([ticker_x, ticker_y], start=start_date, end=END_DATE,
                       auto_adjust=True, progress=False)
    if data.empty:
        return None
    close = data["Close"]
    df = pd.DataFrame({
        "date":    close.index,
        "price_x": close[ticker_x].values,
        "price_y": close[ticker_y].values,
    })
    df = df.dropna().reset_index(drop=True)
    return df if len(df) > TRAIN_DAYS + TEST_DAYS else None


def test_cointegration(df: pd.DataFrame) -> dict:
    """
    Johansen sobre LOG-precios (estándar académico).
    Prueba det_order = -1, 0, 1 y reporta el más favorable.
    Cointegrado si CUALQUIERA de las tres especificaciones supera crit 95%.
    """
    log_x = np.log(df["price_x"].values)
    log_y = np.log(df["price_y"].values)
    endog = np.column_stack([log_x, log_y])

    best = {"cointegrated": False, "trace_stat": 0.0, "crit_95": 999.0, "det_order": None}
    for det in [-1, 0, 1]:
        try:
            result     = coint_johansen(endog, det_order=det, k_ar_diff=1)
            trace_stat = result.lr1[0]
            crit_95    = result.cvt[0, 1]
            if trace_stat > crit_95:
                # Es cointegrado — guardamos esta especificación y paramos
                return {"cointegrated": True, "trace_stat": trace_stat,
                        "crit_95": crit_95, "det_order": det}
            # No cointegrado pero guardamos el mejor (más cercano al umbral)
            if (crit_95 - trace_stat) < (best["crit_95"] - best["trace_stat"]):
                best = {"cointegrated": False, "trace_stat": trace_stat,
                        "crit_95": crit_95, "det_order": det}
        except Exception:
            continue
    return best


def build_filter(mu_init: float) -> VSNLMSFilter:
    return VSNLMSFilter(n_taps=1, mu_init=mu_init, **VSNLMS_FIXED)


def evaluate_on_window(df_window: pd.DataFrame, filter_param: float,
                       lookback: int, entry_z: float, exit_z: float) -> dict:
    filt = build_filter(filter_param)
    X = df_window["price_x"].values.reshape(-1, 1)
    y = df_window["price_y"].values
    result = filt.run(X, y)

    hedge_ratios   = result["weights_history"][:, 0]
    adaptive_spread = result["errors"]

    zscore  = compute_zscore(adaptive_spread, lookback=lookback)
    signals = generate_signals(zscore, entry_threshold=entry_z,
                               exit_threshold=exit_z, zscore_sizing=True)
    bt      = backtest(df_window, signals, hedge_ratios)
    returns = bt["strategy_return"].values
    returns = returns[~np.isnan(returns)]

    if len(returns) == 0 or np.std(returns) == 0:
        return {"sharpe": -999, "max_dd": -1.0, "score": -999}

    total_return = (1 + returns).prod() - 1
    ann_return   = (1 + total_return) ** (252 / len(returns)) - 1
    ann_vol      = np.std(returns) * np.sqrt(252)
    sharpe       = ann_return / ann_vol if ann_vol > 0 else 0

    cum      = np.cumprod(1 + returns)
    peak     = np.maximum.accumulate(cum)
    drawdown = (cum - peak) / peak
    max_dd   = np.min(drawdown)

    score = sharpe + 0.5 * max_dd
    return {"sharpe": sharpe, "max_dd": max_dd, "score": score}


def optimize_on_train(df_train: pd.DataFrame) -> dict | None:
    best_score  = -999
    best_params = None
    combos = list(itertools.product(
        PARAM_GRID["filter_param"], PARAM_GRID["lookback"],
        PARAM_GRID["entry_z"],      PARAM_GRID["exit_z"],
    ))
    for fp, lb, ez, xz in combos:
        if xz >= ez:
            continue
        metrics = evaluate_on_window(df_train, fp, lb, ez, xz)
        if metrics["score"] > best_score:
            best_score  = metrics["score"]
            best_params = {"filter_param": fp, "lookback": lb,
                           "entry_z": ez, "exit_z": xz,
                           "train_sharpe": metrics["sharpe"],
                           "train_score":  best_score}
    return best_params


def test_with_params(df_full: pd.DataFrame, params: dict) -> pd.DataFrame:
    filt = build_filter(params["filter_param"])
    X    = df_full["price_x"].values.reshape(-1, 1)
    y    = df_full["price_y"].values
    result = filt.run(X, y)

    hedge_ratios    = result["weights_history"][:, 0]
    adaptive_spread = result["errors"]

    zscore  = compute_zscore(adaptive_spread, lookback=params["lookback"])
    signals = generate_signals(zscore, entry_threshold=params["entry_z"],
                               exit_threshold=params["exit_z"], zscore_sizing=True)
    bt = backtest(df_full, signals, hedge_ratios)
    return bt.iloc[TRAIN_DAYS:].copy()


def run_walk_forward(df: pd.DataFrame, label: str) -> dict | None:
    total_days = len(df)
    n_windows  = (total_days - TRAIN_DAYS) // TEST_DAYS

    all_test_results = []
    window_summaries = []

    for fold in range(n_windows):
        train_start = fold * TEST_DAYS
        train_end   = train_start + TRAIN_DAYS
        test_end    = train_end   + TEST_DAYS
        if test_end > total_days:
            break

        df_train = df.iloc[train_start:train_end].reset_index(drop=True)
        df_full  = df.iloc[train_start:test_end].reset_index(drop=True)

        best_params = optimize_on_train(df_train)
        if best_params is None:
            continue

        test_results = test_with_params(df_full, best_params)
        all_test_results.append(test_results)

        test_returns  = test_results["strategy_return"].values
        test_returns  = test_returns[~np.isnan(test_returns)]
        test_total    = (1 + test_returns).prod() - 1 if len(test_returns) > 0 else 0
        test_vol      = np.std(test_returns) * np.sqrt(252) if len(test_returns) > 0 else 0
        test_ann_ret  = (1 + test_total) ** (252 / len(test_returns)) - 1 if len(test_returns) > 0 else 0
        test_sharpe   = test_ann_ret / test_vol if test_vol > 0 else 0

        window_summaries.append({
            "fold":          fold + 1,
            "filter_param":  best_params["filter_param"],
            "lookback":      best_params["lookback"],
            "entry_z":       best_params["entry_z"],
            "exit_z":        best_params["exit_z"],
            "train_sharpe":  best_params["train_sharpe"],
            "test_return":   test_total,
            "test_sharpe":   test_sharpe,
        })

        print(f"      Fold {fold+1:2d}/{n_windows}  "
              f"μ0={best_params['filter_param']:.2f}  "
              f"lb={best_params['lookback']:3d}  "
              f"ez={best_params['entry_z']:.1f}  xz={best_params['exit_z']:.2f}  │  "
              f"Train Sharpe: {best_params['train_sharpe']:+.2f}  "
              f"Test Return: {test_total:+.1%}  "
              f"Test Sharpe: {test_sharpe:+.2f}")

    if not all_test_results:
        return None

    combined    = pd.concat(all_test_results, ignore_index=True)
    oos_returns = combined["strategy_return"].values
    oos_returns = oos_returns[~np.isnan(oos_returns)]

    oos_total   = (1 + oos_returns).prod() - 1
    oos_ann_ret = (1 + oos_total) ** (252 / len(oos_returns)) - 1
    oos_vol     = np.std(oos_returns) * np.sqrt(252)
    oos_sharpe  = oos_ann_ret / oos_vol if oos_vol > 0 else 0

    cum      = np.cumprod(1 + oos_returns)
    peak     = np.maximum.accumulate(cum)
    dd       = (cum - peak) / peak
    oos_max_dd = np.min(dd)

    active    = oos_returns[oos_returns != 0]
    win_rate  = np.sum(active > 0) / len(active) if len(active) > 0 else 0
    n_trades  = int(np.sum(np.diff(combined["signal"].values) != 0))

    summary_df      = pd.DataFrame(window_summaries)
    n_positive      = int((summary_df["test_return"] > 0).sum())
    n_folds         = len(summary_df)
    avg_train_sharpe = summary_df["train_sharpe"].mean()
    avg_test_sharpe  = summary_df["test_sharpe"].mean()
    degradation     = 1 - (avg_test_sharpe / avg_train_sharpe) if avg_train_sharpe != 0 else 0

    # Parámetros más frecuentes (moda del walk-forward)
    best_fp  = summary_df["filter_param"].mode()[0]
    best_lb  = summary_df["lookback"].mode()[0]
    best_ez  = summary_df["entry_z"].mode()[0]
    best_xz  = summary_df["exit_z"].mode()[0]

    return {
        "label":           label,
        "oos_total":       oos_total,
        "oos_ann_ret":     oos_ann_ret,
        "oos_vol":         oos_vol,
        "oos_sharpe":      oos_sharpe,
        "oos_max_dd":      oos_max_dd,
        "win_rate":        win_rate,
        "n_trades":        n_trades,
        "n_positive_folds": n_positive,
        "n_folds":         n_folds,
        "avg_train_sharpe": avg_train_sharpe,
        "avg_test_sharpe":  avg_test_sharpe,
        "degradation":     degradation,
        "best_fp":         best_fp,
        "best_lb":         best_lb,
        "best_ez":         best_ez,
        "best_xz":         best_xz,
        "summary_df":      summary_df,
    }


# ── Main ──────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("ANÁLISIS DE PARES — Cointegración + Walk-Forward (VSNLMS)")
    print(f"Parámetros FIJOS: mu_min={VSNLMS_FIXED['mu_min']}  "
          f"mu_max={VSNLMS_FIXED['mu_max']}  "
          f"alpha={VSNLMS_FIXED['alpha']}  "
          f"gamma={VSNLMS_FIXED['gamma']}")
    print("=" * 70)

    results_all = []

    for ticker_x, ticker_y, label in PAIRS:
        print(f"\n{'─'*70}")
        print(f"PAR: {label}  ({ticker_x} / {ticker_y})")

        # 1+2. Buscar la ventana más larga donde el par sea cointegrado
        df = None
        coint_result = None
        used_start = None
        for start in CANDIDATE_STARTS:
            print(f"  [1] Descargando datos ({start} → {END_DATE})...")
            df_candidate = download_pair(ticker_x, ticker_y, start)
            if df_candidate is None:
                print(f"      Insuficientes datos para esta ventana.")
                continue
            print(f"      {len(df_candidate)} días descargados.")
            cr = test_cointegration(df_candidate)
            coint_str = f"Johansen(det={cr['det_order']}) log-prices  trace={cr['trace_stat']:.3f}  crit95={cr['crit_95']:.3f}"
            if cr["cointegrated"]:
                print(f"  [2] ✓ COINTEGRADO  ({coint_str})")
                df, coint_result, used_start = df_candidate, cr, start
                break
            else:
                print(f"  [2] ✗ NO cointegrado  ({coint_str})  → probando ventana más corta...")

        if df is None or coint_result is None:
            print(f"  ✗ No cointegrado en ninguna ventana. Saltando.")
            continue

        # 3. Walk-forward optimization
        print(f"  [3] Walk-forward ({len(df)} días, "
              f"{(len(df)-TRAIN_DAYS)//TEST_DAYS} folds)...")
        res = run_walk_forward(df, label)

        if res is None:
            print(f"  ✗ Walk-forward sin resultados.")
            continue

        # 4. Imprimir resumen
        deg_label = ("✓ ROBUSTO" if res["degradation"] < 0.3
                     else ("⚠ MODERADO" if res["degradation"] < 0.6
                           else "✗ OVERFITTING"))

        print(f"\n  [4] Resultados OUT-OF-SAMPLE — {label}")
        print(f"  ┌─────────────────────────────────────────────────────┐")
        print(f"  │  OOS Return total       {res['oos_total']:>10.2%}              │")
        print(f"  │  OOS Return anualizado  {res['oos_ann_ret']:>10.2%}              │")
        print(f"  │  Volatilidad anualizada {res['oos_vol']:>10.2%}              │")
        print(f"  │  Sharpe ratio           {res['oos_sharpe']:>10.2f}              │")
        print(f"  │  Max Drawdown           {res['oos_max_dd']:>10.2%}              │")
        print(f"  │  Win rate               {res['win_rate']:>10.2%}              │")
        print(f"  │  Trades totales         {res['n_trades']:>10d}              │")
        print(f"  │  Folds rentables        {res['n_positive_folds']:>4d}/{res['n_folds']:<4d}               │")
        print(f"  └─────────────────────────────────────────────────────┘")
        print(f"  Overfitting: Train Sharpe={res['avg_train_sharpe']:+.3f}  "
              f"Test Sharpe={res['avg_test_sharpe']:+.3f}  "
              f"Degradación={res['degradation']:.0%}  {deg_label}")
        print(f"  Parámetros más frecuentes (moda walk-forward):")
        print(f"    mu_init={res['best_fp']}  lookback={res['best_lb']}  "
              f"entry_z={res['best_ez']}  exit_z={res['best_xz']}")

        results_all.append(res)

    # Resumen global
    print(f"\n{'='*70}")
    print(f"RESUMEN GLOBAL")
    print(f"{'='*70}")
    if not results_all:
        print("  Ningún par pasó el test de cointegración.")
    else:
        header = f"  {'Par':<30} {'Sharpe':>7} {'OOS Ret':>9} {'MaxDD':>8} {'Degrad':>8} {'Estado'}"
        print(header)
        print(f"  {'-'*70}")
        for r in results_all:
            deg_sym = ("✓" if r["degradation"] < 0.3
                       else ("⚠" if r["degradation"] < 0.6 else "✗"))
            print(f"  {r['label']:<30} {r['oos_sharpe']:>7.2f} "
                  f"{r['oos_ann_ret']:>9.2%} {r['oos_max_dd']:>8.2%} "
                  f"{r['degradation']:>8.0%} {deg_sym}")


if __name__ == "__main__":
    main()
