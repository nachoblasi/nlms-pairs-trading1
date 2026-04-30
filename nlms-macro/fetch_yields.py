"""
fetch_yields.py — Descarga yields del Tesoro US desde FRED (sin API key).

Series descargadas:
    DGS2   — 2-Year Treasury Constant Maturity Rate
    DGS5   — 5-Year Treasury Constant Maturity Rate
    DGS10  — 10-Year Treasury Constant Maturity Rate
    DGS30  — 30-Year Treasury Constant Maturity Rate

Pares generados (price_x = yield corto, price_y = yield largo):
    DGS2  / DGS10  — spread clásico 2Y-10Y (el más seguido por el mercado)
    DGS5  / DGS30  — spread 5Y-30Y (más volatilidad, más señal)
    DGS2  / DGS5   — tramo corto de la curva
    DGS10 / DGS30  — tramo largo de la curva

Nota sobre el modelo: los yields individuales son I(1) (no estacionarios),
pero su spread es cointegrado por fundamentos económicos. El NLMS estima
el β óptimo (no necesariamente 1) que produce el spread más estacionario.

Uso:
    python fetch_yields.py
    python fetch_yields.py --start 2000-01-01
"""

import argparse
import urllib.request
import io
from pathlib import Path

import pandas as pd

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"

# Series y su vencimiento en años (para convertir yield → precio de bono)
SERIES = {
    "DGS2":  2,
    "DGS5":  5,
    "DGS10": 10,
    "DGS30": 30,
}

PAIRS = [
    ("DGS2",  "DGS10", "2y10y"),   # el spread más famoso del mundo
    ("DGS5",  "DGS30", "5y30y"),   # spread largo, más volatilidad
    ("DGS2",  "DGS5",  "2y5y"),    # tramo corto de la curva
    ("DGS10", "DGS30", "10y30y"),  # tramo largo de la curva
]

DEFAULT_START = "1990-01-01"


def yield_to_price(yield_pct: pd.Series, maturity: int) -> pd.Series:
    """
    Convierte yields (en %) a precios de bono de cupón cero.

    Fórmula: P = 100 / (1 + y/100)^T

    Esto es necesario porque el pipeline de backtest usa pct_change() para
    calcular retornos, lo que asume datos de precio. Con yields directos,
    pct_change() daría % de cambio del yield (sin sentido económico).

    Con precios de bono: pct_change() ≈ -Duration × Δyield, que sí es el
    retorno real de un bono ante un movimiento de tipos.

    Ejemplos:
        y=4.0%  T=10 → P = 100/(1.04)^10 = 67.56
        y=4.1%  T=10 → P = 100/(1.041)^10 = 67.00 → retorno ≈ -0.83%
        (duración de un bono cupón cero de 10Y ≈ 10 → Δretorno ≈ -10 × 0.001 = -1.0%)
    """
    return 100.0 / (1.0 + yield_pct / 100.0) ** maturity


def fetch_series(series_id: str) -> pd.Series:
    """Descarga una serie de FRED y la devuelve como pd.Series con índice DatetimeIndex."""
    url = FRED_URL.format(series=series_id)
    print(f"    Descargando {series_id} de FRED...")
    with urllib.request.urlopen(url) as resp:
        content = resp.read().decode("utf-8")
    df = pd.read_csv(io.StringIO(content), parse_dates=["observation_date"])
    df = df.set_index("observation_date")
    s = pd.to_numeric(df[series_id], errors="coerce")   # "." → NaN
    s.name = series_id
    return s


def main():
    parser = argparse.ArgumentParser(description="Descarga yields FRED para nlms-macro")
    parser.add_argument("--start", default=DEFAULT_START)
    args = parser.parse_args()

    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)

    # ── Descarga todas las series ──────────────────────────────────────────────
    print("\n[1] Descargando series de FRED...")
    yields_data = {}
    prices_data = {}
    for sid, maturity in SERIES.items():
        s = fetch_series(sid)
        s = s[s.index >= args.start].dropna()
        yields_data[sid] = s
        prices_data[sid] = yield_to_price(s, maturity)
        print(f"    {sid} (T={maturity}Y): {len(s)} obs  {s.index[0].date()} → {s.index[-1].date()}"
              f"  yield={s.mean():.2f}%  precio={prices_data[sid].mean():.1f}")

    # ── Construir pares en precios de bono ────────────────────────────────────
    print("\n[2] Construyendo pares (yield → precio bono cupón cero)...")
    for x_id, y_id, fname in PAIRS:
        px = prices_data[x_id]
        py = prices_data[y_id]

        # Alinear por fecha
        df = pd.concat([px, py], axis=1).dropna()
        df.index.name = "date"
        df = df.reset_index()
        df.columns = ["date", "price_x", "price_y"]
        df["date"] = df["date"].dt.date

        out_path = out_dir / f"{fname}.csv"
        df.to_csv(out_path, index=False)

        # Mostrar spread en yields para referencia
        spread_y = (yields_data[y_id] - yields_data[x_id]).dropna()
        print(f"    {x_id}/{y_id} → {out_path.name}"
              f"  ({len(df)} obs, {df['date'].iloc[0]} → {df['date'].iloc[-1]})"
              f"  spread_yield medio={spread_y.mean()*100:.0f}bps  std={spread_y.std()*100:.0f}bps")

    print("\nListo.")


if __name__ == "__main__":
    main()
