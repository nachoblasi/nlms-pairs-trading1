"""
run_real_data.py — Ejecutar la estrategia NLMS con datos reales.

Este script:
1. Descarga datos de Yahoo Finance (V y MA)
2. Testea si están cointegradas (Engle-Granger)
3. Aplica el filtro NLMS
4. Genera señales, backtest y gráficos

Requisitos adicionales (instalar antes):
    pip install yfinance statsmodels

Ejecutar: python run_real_data.py
"""

import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from src.nlms import NLMSFilter
from src.strategy import compute_zscore, generate_signals, backtest, performance_metrics
from src.plots import plot_all


def download_pair(ticker_x: str, ticker_y: str,
                  start: str, end: str) -> pd.DataFrame:
    """
    Descarga precios de cierre ajustados de dos activos desde Yahoo Finance.

    Parámetros
    ----------
    ticker_x : str
        Ticker del activo X (ej: "V" para Visa).
    ticker_y : str
        Ticker del activo Y (ej: "MA" para Mastercard).
    start : str
        Fecha de inicio (formato "YYYY-MM-DD").
    end : str
        Fecha de fin (formato "YYYY-MM-DD").

    Retorna
    -------
    pd.DataFrame con columnas: date, price_x, price_y
    """
    print(f"    Downloading {ticker_x} and {ticker_y} from {start} to {end}...")

    # Descargamos ambos tickers a la vez.
    # "Close" nos da el precio de cierre ajustado por splits y dividendos,
    # que es lo correcto para análisis histórico.
    data = yf.download([ticker_x, ticker_y], start=start, end=end)["Close"]

    # Eliminamos días con datos faltantes (festivos donde solo cotiza uno)
    data = data.dropna()

    df = pd.DataFrame({
        "date": data.index,
        "price_x": data[ticker_x].values,
        "price_y": data[ticker_y].values,
    })
    df = df.reset_index(drop=True)

    print(f"    Downloaded {len(df)} trading days.")
    return df


def test_cointegration(price_x: np.ndarray, price_y: np.ndarray,
                       significance: float = 0.05) -> dict:
    """
    Testea cointegración entre dos series de precios usando el
    método de Johansen (1988).

    Ventajas sobre Engle-Granger:
      - No asume una dirección de causalidad (X→Y o Y→X)
      - Mayor potencia estadística en muestras finitas
      - Detecta la relación aunque la normalización sea ambigua

    El test traza evalúa H0: r=0 (ningún vector de cointegración).
    Si el estadístico traza supera el valor crítico al 95%, rechazamos
    H0 → las series están cointegradas.

    Parámetros
    ----------
    price_x, price_y : np.ndarray
        Series de precios.
    significance : float
        Nivel de significancia: 0.05 usa el valor crítico al 95%.

    Retorna
    -------
    dict con:
        cointegrated : bool
        trace_stat   : float  — estadístico traza para H0: r=0
        critical_values : dict  — valores críticos al 90%, 95%, 99%
    """
    endog = np.column_stack([price_x, price_y])
    result = coint_johansen(endog, det_order=0, k_ar_diff=1)

    trace_stat = result.lr1[0]          # H0: r=0
    crit_90    = result.cvt[0, 0]
    crit_95    = result.cvt[0, 1]
    crit_99    = result.cvt[0, 2]

    cointegrated = trace_stat > crit_95

    return {
        "cointegrated": cointegrated,
        "trace_stat": trace_stat,
        "critical_values": {
            "90%": crit_90,
            "95%": crit_95,
            "99%": crit_99,
        },
    }


def test_stationarity(spread: np.ndarray) -> dict:
    """
    Test Augmented Dickey-Fuller (ADF) sobre el spread.

    Hipótesis:
        H0: la serie tiene raíz unitaria (NO es estacionaria)
        H1: la serie es estacionaria

    Si p_value < 0.05 → rechazamos H0 → el spread ES estacionario → bien.

    Parámetros
    ----------
    spread : np.ndarray
        El spread (errores del NLMS o residuos OLS).

    Retorna
    -------
    dict con resultados del test.
    """
    # adfuller retorna: (test_stat, p_value, lags_used, nobs, critical_values, icbest)
    result = adfuller(spread, autolag="AIC")

    return {
        "stationary": result[1] < 0.05,
        "test_statistic": result[0],
        "p_value": result[1],
        "lags_used": result[2],
        "critical_values": result[4],
    }


def main():
    print("=" * 60)
    print("NLMS PAIRS TRADING — REAL DATA (V / MA)")
    print("=" * 60)

    # ══════════════════════════════════════════════════════════
    # PASO 1: DESCARGAR DATOS
    # ══════════════════════════════════════════════════════════
    print("\n[1] Downloading data...")

    # Visa = X, Mastercard = Y
    # Elegimos X e Y de forma que Y ≈ β·X con β > 0.
    # En general no importa cuál es X y cuál es Y, pero por convención
    # ponemos como Y el activo con precio más alto.
    df = download_pair(
        ticker_x="V",
        ticker_y="MA",
        start="2010-01-01",
        end="2026-03-15",
    )

    # Guardamos los datos descargados
    df.to_csv("data/visa_mastercard.csv", index=False)
    print(f"    Date range: {df['date'].iloc[0]} → {df['date'].iloc[-1]}")

    # ══════════════════════════════════════════════════════════
    # PASO 2: TEST DE COINTEGRACIÓN
    # ══════════════════════════════════════════════════════════
    print("\n[2] Testing cointegration (Johansen)...")

    coint_result = test_cointegration(df["price_x"].values, df["price_y"].values)

    print(f"    Trace statistic: {coint_result['trace_stat']:.4f}")
    print(f"    Critical values: {coint_result['critical_values']}")

    if coint_result["cointegrated"]:
        print("    ✓ COINTEGRATED at 5% significance level. Proceeding.")
    else:
        print("    ✗ NOT cointegrated at 5% significance level.")
        print("    ⚠ The NLMS spread may not be stationary. Proceed with caution.")
        print("    (Continuing anyway for educational purposes...)")

    # ══════════════════════════════════════════════════════════
    # PASO 3: FILTRO NLMS
    # ══════════════════════════════════════════════════════════
    print("\n[3] Running NLMS adaptive filter...")

    # Con datos reales, mu más bajo puede ser mejor porque hay más ruido.
    # mu=0.05 es más conservador que 0.1: se adapta más lento pero
    # es más estable frente al ruido del mercado real.
    # antes el mu era 0,05 y lo he cambiado a 0,2
    mu = 0.2
    nlms = NLMSFilter(n_taps=1, mu=mu)

    X = df["price_x"].values.reshape(-1, 1)
    y = df["price_y"].values

    result = nlms.run(X, y)

    hedge_ratios = result["weights_history"][:, 0]
    adaptive_spread = result["errors"]

    print(f"    mu = {mu}")
    print(f"    Final hedge ratio: {hedge_ratios[-1]:.4f}")
    print(f"    Spread std: {np.std(adaptive_spread[200:]):.4f}")
    # Nota: descartamos los primeros 200 datos del std para ignorar el warm-up

    # ── Test ADF sobre el spread NLMS ──
    print("\n    Testing stationarity of NLMS spread (ADF)...")
    # Usamos el spread después del warm-up (primeros 200 días)
    adf_result = test_stationarity(adaptive_spread[200:])
    print(f"    ADF statistic: {adf_result['test_statistic']:.4f}")
    print(f"    P-value:       {adf_result['p_value']:.6f}")
    if adf_result["stationary"]:
        print("    ✓ Spread is STATIONARY. Good for mean-reversion.")
    else:
        print("    ✗ Spread is NOT stationary. Signals may be unreliable.")

    # ══════════════════════════════════════════════════════════
    # PASO 4: Z-SCORE Y SEÑALES
    # ══════════════════════════════════════════════════════════
    print("\n[4] Computing z-score and generating signals...")

    lookback = 120  # 60 días ≈ 3 meses (un poco más que con datos sintéticos). Lo cambio a 120 que va mejor
    zscore = compute_zscore(adaptive_spread, lookback=lookback)

    entry_z = 1.5
    exit_z = 0.75
    signals = generate_signals(zscore, entry_threshold=entry_z, exit_threshold=exit_z)

    n_long = np.sum(signals == 1)
    n_short = np.sum(signals == -1)
    n_flat = np.sum(signals == 0)
    print(f"    Lookback: {lookback}  |  Entry: ±{entry_z}σ  |  Exit: ±{exit_z}σ")
    print(f"    Days long: {n_long}  |  Days short: {n_short}  |  Days flat: {n_flat}")

    # ══════════════════════════════════════════════════════════
    # PASO 5: BACKTEST
    # ══════════════════════════════════════════════════════════
    print("\n[5] Running backtest...")

    results = backtest(df, signals, hedge_ratios)
    results.to_csv("results/backtest_real.csv", index=False)

    metrics = performance_metrics(results)
    print("\n    ┌─── Performance Summary (V / MA) ───┐")
    for key, val in metrics.items():
        print(f"    │  {key:<25s} {str(val):>10s} │")
    print("    └────────────────────────────────────┘")

    # ══════════════════════════════════════════════════════════
    # PASO 6: GRÁFICOS
    # ══════════════════════════════════════════════════════════
    print("\n[6] Generating report plot...")
    plot_all(df, result, zscore, signals, results,
             save_path="results/real_data_report.png")

    print("\n✓ Done! Check results/ folder.")


if __name__ == "__main__":
    main()
