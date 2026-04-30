"""
portfolio_sharpe.py — Sharpe del portfolio combinando todos los pares (MLP, --no-background).

Metodología:
    1. Para cada par, ejecuta el walk-forward MLP y extrae la serie de retornos OOS diarios.
    2. Cuando un par no tiene fold activo en una fecha (cointegración fallida, etc.) → retorno = 0.
    3. Combina todas las series por fecha con igual peso (1/N por par).
    4. Calcula métricas sobre el portfolio resultante.

El Sharpe del portfolio puede superar al promedio individual gracias a la
diversificación: si las correlaciones entre spreads son bajas, la volatilidad
del portfolio cae más que el retorno esperado.
"""

import sys
import numpy as np
import pandas as pd

from src.ml_nlms import (train_mu_predictor, DEFAULT_MU_CANDIDATES,
                         DEFAULT_FEATURE_WINDOW, DEFAULT_TARGET_WINDOW)
from src.cointegration import johansen_cointegration, compute_halflife
from src.optimizer import (build_ml_filter, run_filter_pipeline,
                           optimize_on_train)

# ── Configuración ─────────────────────────────────────────────────────────────

TRAIN_DAYS    = 756
TEST_DAYS     = 252
MIN_SHARPE    = 0.30
ML_FALLBACK   = float(np.median(DEFAULT_MU_CANDIDATES))

# Pares a incluir (ruta al CSV, etiqueta). WTI/Brent excluido.
PAIRS = [
    # Renta variable
    ("data/visa_mastercard.csv",          "V / MA"),
    ("data/scan/nsrgy_unlyf.csv",         "NSRGY / UNLYF"),
    ("data/scan/deo_prndy.csv",           "DEO / PRNDY"),
    ("data/scan/dbsdy_uovey.csv",         "DBS / UOB"),
    ("data/scan/mco_spgi.csv",            "MCO / SPGI"),
    ("data/scan/lvmuy_ppruy.csv",         "LVMUY / PPRUY"),
    ("data/scan/rhhby_azn.csv",           "RHHBY / AZN"),
    ("data/scan/gs_ms.csv",               "GS / MS"),
    ("data/scan/bp_shel.csv",             "BP / SHEL"),
    # Macro
    ("../nlms-macro/data/efa_eem.csv",    "EFA / EEM"),
    ("../nlms-macro/data/nok_cad.csv",    "NOK / CAD"),
]


# ── Core: extraer retornos OOS de un par ──────────────────────────────────────

def get_pair_oos_returns(data_file: str, pair_name: str) -> pd.Series:
    """
    Ejecuta el walk-forward MLP (--no-background) para un par y devuelve
    la serie diaria de retornos OOS, indexada por fecha.

    Los días sin fold activo NO aparecen en la serie (se tratan como 0
    al hacer el merge del portfolio).
    """
    try:
        df = pd.read_csv(data_file, parse_dates=["date"])
    except FileNotFoundError:
        print(f"  [{pair_name}] ARCHIVO NO ENCONTRADO: {data_file}")
        return pd.Series(dtype=float, name=pair_name)

    total_days = len(df)
    n_folds    = (total_days - TRAIN_DAYS) // TEST_DAYS
    segments   = []

    for fold in range(n_folds):
        train_start = fold * TEST_DAYS
        train_end   = train_start + TRAIN_DAYS
        test_end    = min(train_end + TEST_DAYS, total_days)

        df_train = df.iloc[train_start:train_end]
        df_full  = df.iloc[train_start:test_end]

        # ── Filtro 1: Johansen ────────────────────────────────────────────────
        coint = johansen_cointegration(df_train["price_x"].values,
                                       df_train["price_y"].values)
        if not coint["cointegrated"]:
            continue

        # ── Filtro 2: Half-life ───────────────────────────────────────────────
        hl = compute_halflife(df_train["price_x"].values,
                              df_train["price_y"].values)
        if not hl["in_range"]:
            continue

        # ── Filtro 3: Grid search + Sharpe en train ───────────────────────────
        best_params = optimize_on_train(df_train)
        if best_params is None or best_params["train_sharpe"] < MIN_SHARPE:
            continue

        # ── Entrenar MLP ─────────────────────────────────────────────────────
        ml_model, ml_scaler, _, _ = train_mu_predictor(
            price_x        = df_train["price_x"].values,
            price_y        = df_train["price_y"].values,
            mu_candidates  = DEFAULT_MU_CANDIDATES,
            feature_window = DEFAULT_FEATURE_WINDOW,
            target_window  = DEFAULT_TARGET_WINDOW,
            X_extra        = None,
            Y_extra        = None,
        )

        # ── Evaluar sobre ventana completa (warm-up en train, test OOS) ───────
        filt_ml = build_ml_filter(ml_model, ml_scaler, ML_FALLBACK)
        pipe_ml = run_filter_pipeline(
            df_full, filt_ml,
            lookback = best_params["lookback"],
            entry_z  = best_params["entry_z"],
            exit_z   = best_params["exit_z"],
        )

        bt_test = pipe_ml["bt"].iloc[TRAIN_DAYS:].copy()
        dates   = df_full["date"].iloc[TRAIN_DAYS:].values
        bt_test.index = pd.DatetimeIndex(dates)

        rets = bt_test["strategy_return"]
        if len(rets) > 0:
            segments.append(rets)

    if not segments:
        print(f"  [{pair_name}] Sin folds activos")
        return pd.Series(dtype=float, name=pair_name)

    series = pd.concat(segments).sort_index()
    series.name = pair_name
    return series


# ── Métricas ──────────────────────────────────────────────────────────────────

def compute_metrics(returns: pd.Series) -> dict:
    r = returns.fillna(0).values
    r = r[~np.isnan(r)]
    if len(r) == 0 or np.std(r) == 0:
        return {"sharpe": 0, "ann_ret": 0, "ann_vol": 0, "max_dd": 0, "n_days": 0}

    total   = (1 + r).prod() - 1
    ann_ret = (1 + total) ** (252 / len(r)) - 1
    ann_vol = np.std(r) * np.sqrt(252)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else 0

    cum    = np.cumprod(1 + r)
    peak   = np.maximum.accumulate(cum)
    max_dd = float(np.min((cum - peak) / peak))

    return {"sharpe": sharpe, "ann_ret": ann_ret, "ann_vol": ann_vol,
            "max_dd": max_dd, "n_days": len(r)}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("PORTFOLIO SHARPE — MLP, equal-weight, --no-background")
    print("=" * 65)

    # Recoger series de retornos por par
    all_series = {}
    for data_file, pair_name in PAIRS:
        print(f"\n→ {pair_name} ...")
        s = get_pair_oos_returns(data_file, pair_name)
        if len(s) > 0:
            all_series[pair_name] = s
            m = compute_metrics(s)
            print(f"  Folds activos: {len(s)} días  |  Sharpe individual: {m['sharpe']:.2f}")

    if not all_series:
        print("Sin datos.")
        return

    # Merge por fecha, fill 0 (par flat cuando no hay fold activo)
    rets_df = pd.DataFrame(all_series)
    rets_df = rets_df.sort_index().fillna(0)

    # Portfolio igual peso
    n_pairs      = len(all_series)
    portfolio    = rets_df.mean(axis=1)          # 1/N por par
    port_metrics = compute_metrics(portfolio)

    # Correlaciones entre pares (solo días con al menos dos activos)
    active = (rets_df != 0)
    corr   = rets_df[active.sum(axis=1) >= 2].corr()

    # ── Resultados individuales ───────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("RESULTADOS INDIVIDUALES (retornos OOS activos)")
    print("-" * 65)
    print(f"{'Par':<22} {'Sharpe':>7} {'Ret.anual':>10} {'Vol.anual':>10} {'MaxDD':>8} {'Días':>6}")
    print("-" * 65)
    for name, s in all_series.items():
        m = compute_metrics(s)
        print(f"{name:<22} {m['sharpe']:>7.2f} {m['ann_ret']:>9.1%} {m['ann_vol']:>9.1%} "
              f"{m['max_dd']:>7.1%} {m['n_days']:>6}")

    # ── Portfolio ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"PORTFOLIO IGUAL PESO ({n_pairs} pares)")
    print("-" * 65)
    print(f"  Sharpe ratio:       {port_metrics['sharpe']:.3f}")
    print(f"  Retorno anualizado: {port_metrics['ann_ret']:.1%}")
    print(f"  Volatilidad anual:  {port_metrics['ann_vol']:.1%}")
    print(f"  Max Drawdown:       {port_metrics['max_dd']:.1%}")
    print(f"  Días con posición:  {(portfolio != 0).sum()} / {len(portfolio)}")

    # Gain from diversification
    avg_individual_sharpe = np.mean([compute_metrics(s)["sharpe"]
                                     for s in all_series.values()])
    print(f"\n  Sharpe promedio individual: {avg_individual_sharpe:.3f}")
    print(f"  Sharpe portfolio:           {port_metrics['sharpe']:.3f}")
    gain = port_metrics['sharpe'] / avg_individual_sharpe - 1
    print(f"  Ganancia por diversificación: {gain:+.1%}")

    # ── Retornos anuales del portfolio ────────────────────────────────────────
    print("\n" + "=" * 65)
    print("RETORNOS ANUALES DEL PORTFOLIO")
    print("-" * 65)
    print(f"{'Año':>6} {'Retorno':>10} {'Días activos':>14}")
    print("-" * 65)
    for year, grp in portfolio.groupby(portfolio.index.year):
        r = grp.fillna(0).values
        total_yr = (1 + r).prod() - 1
        active   = int((grp != 0).sum())
        print(f"{year:>6} {total_yr:>10.1%} {active:>14}")

    # ── Correlaciones ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("CORRELACIONES ENTRE SPREADS (solo días con ≥2 pares activos)")
    print("-" * 65)
    avg_corr = corr.values[np.triu_indices_from(corr.values, k=1)].mean()
    print(f"  Correlación media entre pares: {avg_corr:.3f}")
    print(f"  (0 = perfectamente diversificado, 1 = sin diversificación)")
    print()
    print(corr.round(2).to_string())


if __name__ == "__main__":
    main()
