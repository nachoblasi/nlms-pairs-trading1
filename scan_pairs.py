"""
scan_pairs.py — Escaneo de múltiples pares cointegrados.

Testea varios pares candidatos con el pipeline completo:
    1. Descarga datos
    2. Test de cointegración
    3. Walk-forward (ventana fija, sin filtro — el que mejor funciona)
    4. Ranking por Sharpe out-of-sample

Así encontramos qué par funciona mejor con nuestra estrategia NLMS.

Requisitos: pip install yfinance statsmodels
Ejecutar:   python scan_pairs.py
"""

import numpy as np
import pandas as pd
import itertools
import time
import yfinance as yf
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from src.nlms import NLMSFilter
from src.strategy import compute_zscore, generate_signals, backtest


# ══════════════════════════════════════════════════════════════
# PARES CANDIDATOS
# ══════════════════════════════════════════════════════════════
# Pares clásicos de sectores donde la cointegración es probable:
#   - Pagos: V/MA (ya testeado)
#   - Bebidas: KO/PEP
#   - Petróleo: XOM/CVX
#   - Bancos: JPM/BAC
#   - Tech/Cloud: GOOG/MSFT
#   - Retail: HD/LOW
#   - Pharma: JNJ/PFE
#   - Telecom: T/VZ

PAIRS = [
    ("V",    "MA",   "Visa / Mastercard"),
    ("KO",   "PEP",  "Coca-Cola / PepsiCo"),
    ("XOM",  "CVX",  "ExxonMobil / Chevron"),
    ("JPM",  "BAC",  "JPMorgan / Bank of America"),
    ("GOOG", "MSFT", "Google / Microsoft"),
    ("HD",   "LOW",  "Home Depot / Lowe's"),
    ("JNJ",  "PFE",  "Johnson&Johnson / Pfizer"),
    ("T",    "VZ",   "AT&T / Verizon"),
]

# Walk-forward config
TRAIN_DAYS = 756
TEST_DAYS = 252

# Grid de parámetros (reducido para velocidad — escaneamos muchos pares)
PARAM_GRID = {
    "mu":       [0.05, 0.1, 0.15, 0.2],
    "lookback": [30, 60, 90, 120],
    "entry_z":  [1.5, 2.0, 2.5],
    "exit_z":   [0.25, 0.5, 0.75],
}


# ══════════════════════════════════════════════════════════════
# FUNCIONES
# ══════════════════════════════════════════════════════════════

def download_pair(ticker_x, ticker_y, start="2010-01-01", end="2026-03-15"):
    """Descarga precios de cierre de dos activos."""
    try:
        data = yf.download([ticker_x, ticker_y], start=start, end=end,
                           progress=False)["Close"]
        data = data.dropna()

        if len(data) < TRAIN_DAYS + TEST_DAYS * 2:
            return None  # no hay suficientes datos

        df = pd.DataFrame({
            "date": data.index,
            "price_x": data[ticker_x].values,
            "price_y": data[ticker_y].values,
        }).reset_index(drop=True)

        return df
    except Exception as e:
        print(f"      Error downloading: {e}")
        return None


def test_cointegration(price_x, price_y):
    """Test de Johansen. Retorna dict con cointegrated y trace_stat."""
    try:
        endog = np.column_stack([price_x, price_y])
        result = coint_johansen(endog, det_order=0, k_ar_diff=1)
        trace_stat = result.lr1[0]
        crit_95    = result.cvt[0, 1]
        return {"cointegrated": trace_stat > crit_95,
                "trace_stat": trace_stat, "crit_95": crit_95}
    except Exception:
        return {"cointegrated": False, "trace_stat": 0.0, "crit_95": 999.0}


def evaluate_params(df_window, mu, lookback, entry_z, exit_z):
    """Evalúa parámetros sobre una ventana."""
    nlms = NLMSFilter(n_taps=1, mu=mu)
    X = df_window["price_x"].values.reshape(-1, 1)
    y = df_window["price_y"].values
    result = nlms.run(X, y)

    hedge_ratios = result["weights_history"][:, 0]
    spread = result["errors"]

    zscore = compute_zscore(spread, lookback=lookback)
    signals = generate_signals(zscore, entry_threshold=entry_z, exit_threshold=exit_z)

    bt = backtest(df_window, signals, hedge_ratios)
    returns = bt["strategy_return"].values
    returns = returns[~np.isnan(returns)]

    if len(returns) == 0 or np.std(returns) == 0:
        return -999

    total = (1 + returns).prod() - 1
    ann_ret = (1 + total) ** (252 / len(returns)) - 1
    vol = np.std(returns) * np.sqrt(252)
    sharpe = ann_ret / vol if vol > 0 else 0

    cum = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    max_dd = np.min(dd)

    return sharpe + 0.5 * max_dd


def optimize_on_train(df_train):
    """Grid search sobre el train."""
    best_score = -999
    best_params = None

    for mu, lookback, entry_z, exit_z in itertools.product(
        PARAM_GRID["mu"], PARAM_GRID["lookback"],
        PARAM_GRID["entry_z"], PARAM_GRID["exit_z"],
    ):
        if exit_z >= entry_z:
            continue

        score = evaluate_params(df_train, mu, lookback, entry_z, exit_z)

        if score > best_score:
            best_score = score
            best_params = {"mu": mu, "lookback": lookback,
                           "entry_z": entry_z, "exit_z": exit_z}

    return best_params


def walk_forward_test(df):
    """
    Ejecuta walk-forward completo sobre un par.
    Retorna métricas out-of-sample y detalle por fold.
    """
    total_days = len(df)
    n_folds = (total_days - TRAIN_DAYS) // TEST_DAYS

    all_test_results = []
    fold_details = []

    for fold in range(n_folds):
        train_start = fold * TEST_DAYS
        train_end = train_start + TRAIN_DAYS
        test_end = train_end + TEST_DAYS

        if test_end > total_days:
            break

        df_train = df.iloc[train_start:train_end].reset_index(drop=True)
        df_full = df.iloc[train_start:test_end].reset_index(drop=True)

        # Optimizar
        params = optimize_on_train(df_train)
        if params is None:
            continue

        # Test
        nlms = NLMSFilter(n_taps=1, mu=params["mu"])
        X = df_full["price_x"].values.reshape(-1, 1)
        y = df_full["price_y"].values
        result = nlms.run(X, y)

        hedge_ratios = result["weights_history"][:, 0]
        spread = result["errors"]

        zscore = compute_zscore(spread, lookback=params["lookback"])
        signals = generate_signals(zscore, entry_threshold=params["entry_z"],
                                    exit_threshold=params["exit_z"])

        bt = backtest(df_full, signals, hedge_ratios)
        test_bt = bt.iloc[TRAIN_DAYS:].copy()
        all_test_results.append(test_bt)

        # Métricas del fold
        ret = test_bt["strategy_return"].values
        ret = ret[~np.isnan(ret)]
        fold_total = (1 + ret).prod() - 1 if len(ret) > 0 else 0
        fold_details.append(fold_total)

    if not all_test_results:
        return None

    # Métricas agregadas
    combined = pd.concat(all_test_results, ignore_index=True)
    returns = combined["strategy_return"].values
    returns = returns[~np.isnan(returns)]

    if len(returns) == 0 or np.std(returns) == 0:
        return None

    total = (1 + returns).prod() - 1
    ann_ret = (1 + total) ** (252 / len(returns)) - 1
    vol = np.std(returns) * np.sqrt(252)
    sharpe = ann_ret / vol if vol > 0 else 0

    cum = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    max_dd = np.min(dd)

    active = returns[returns != 0]
    win_rate = np.sum(active > 0) / len(active) if len(active) > 0 else 0
    n_trades = np.sum(np.diff(combined["signal"].values) != 0)

    n_positive = sum(1 for r in fold_details if r > 0)

    return {
        "total": total,
        "ann_ret": ann_ret,
        "vol": vol,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "win_rate": win_rate,
        "n_trades": int(n_trades),
        "n_folds": len(fold_details),
        "n_positive": n_positive,
    }


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("PAIR SCANNER — NLMS PAIRS TRADING")
    print("=" * 60)

    results = []

    for i, (ticker_x, ticker_y, name) in enumerate(PAIRS):
        print(f"\n{'─'*60}")
        print(f"  [{i+1}/{len(PAIRS)}] {name} ({ticker_x} / {ticker_y})")
        print(f"{'─'*60}")

        # ── 1. Descargar ──
        print(f"    Downloading...")
        df = download_pair(ticker_x, ticker_y)

        if df is None:
            print(f"    ✗ Insufficient data. Skipping.")
            results.append({
                "pair": name, "ticker_x": ticker_x, "ticker_y": ticker_y,
                "status": "no_data", "coint_pvalue": None, "sharpe": None,
            })
            continue

        print(f"    {len(df)} trading days")

        # ── 2. Cointegración ──
        print(f"    Testing cointegration (Johansen)...")
        coint_result = test_cointegration(df["price_x"].values, df["price_y"].values)
        is_coint = coint_result["cointegrated"]

        print(f"    Trace stat: {coint_result['trace_stat']:.4f}  "
              f"(crit 95%: {coint_result['crit_95']:.4f})  "
              f"→  {'✓ COINTEGRATED' if is_coint else '✗ NOT cointegrated'}")

        # Guardamos datos del par para uso posterior
        df.to_csv(f"data/{ticker_x}_{ticker_y}.csv", index=False)

        if not is_coint:
            print(f"    Skipping walk-forward (not cointegrated).")
            results.append({
                "pair": name, "ticker_x": ticker_x, "ticker_y": ticker_y,
                "status": "not_cointegrated", "coint_trace_stat": coint_result["trace_stat"],
                "sharpe": None,
            })
            continue

        # ── 3. Walk-forward ──
        print(f"    Running walk-forward...")
        start_time = time.time()
        wf_metrics = walk_forward_test(df)
        elapsed = time.time() - start_time

        if wf_metrics is None:
            print(f"    ✗ Walk-forward failed.")
            results.append({
                "pair": name, "ticker_x": ticker_x, "ticker_y": ticker_y,
                "status": "wf_failed", "coint_pvalue": p_value, "sharpe": None,
            })
            continue

        print(f"    Done in {elapsed:.0f}s")
        print(f"    Sharpe: {wf_metrics['sharpe']:.3f}  |  "
              f"MaxDD: {wf_metrics['max_dd']:.2%}  |  "
              f"AnnRet: {wf_metrics['ann_ret']:.2%}  |  "
              f"WinRate: {wf_metrics['win_rate']:.2%}  |  "
              f"Folds: {wf_metrics['n_positive']}/{wf_metrics['n_folds']} profitable")

        results.append({
            "pair": name,
            "ticker_x": ticker_x,
            "ticker_y": ticker_y,
            "status": "ok",
            "coint_trace_stat": coint_result["trace_stat"],
            "days": len(df),
            "sharpe": wf_metrics["sharpe"],
            "ann_ret": wf_metrics["ann_ret"],
            "max_dd": wf_metrics["max_dd"],
            "total_ret": wf_metrics["total"],
            "win_rate": wf_metrics["win_rate"],
            "n_trades": wf_metrics["n_trades"],
            "n_folds": wf_metrics["n_folds"],
            "n_positive": wf_metrics["n_positive"],
        })

    # ══════════════════════════════════════════════════════════
    # RANKING
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  FINAL RANKING (by OOS Sharpe)")
    print(f"{'='*60}\n")

    # Separar pares válidos e inválidos
    valid = [r for r in results if r["status"] == "ok"]
    invalid = [r for r in results if r["status"] != "ok"]

    # Ordenar por Sharpe
    valid.sort(key=lambda x: x["sharpe"], reverse=True)

    print(f"    {'Rank':<5s} {'Pair':<30s} {'Sharpe':>7s} {'AnnRet':>8s} "
          f"{'MaxDD':>8s} {'WinRate':>8s} {'Folds':>7s}")
    print(f"    {'─'*75}")

    for rank, r in enumerate(valid, 1):
        folds_str = f"{r['n_positive']}/{r['n_folds']}"
        print(f"    {rank:<5d} {r['pair']:<30s} {r['sharpe']:>7.3f} "
              f"{r['ann_ret']:>8.2%} {r['max_dd']:>8.2%} "
              f"{r['win_rate']:>8.2%} {folds_str:>7s}")

    if invalid:
        print(f"\n    Excluded pairs:")
        for r in invalid:
            reason = "no data" if r["status"] == "no_data" else "not cointegrated"
            pval_str = f" (trace={r['coint_trace_stat']:.3f})" if r.get('coint_trace_stat') is not None else ""
            print(f"      {r['pair']:<30s} → {reason}{pval_str}")

    # Guardar
    results_df = pd.DataFrame(results)
    results_df.to_csv("results/pair_scan_results.csv", index=False)

    if valid:
        best = valid[0]
        print(f"\n    Best pair: {best['pair']} ({best['ticker_x']}/{best['ticker_y']})")
        print(f"    OOS Sharpe: {best['sharpe']:.3f}")
        print(f"    OOS Annual Return: {best['ann_ret']:.2%}")
        print(f"    Max Drawdown: {best['max_dd']:.2%}")

    print(f"\n    Full results saved to results/pair_scan_results.csv")
    print(f"\n✓ Done!")


if __name__ == "__main__":
    main()
