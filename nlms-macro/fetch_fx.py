"""
fetch_fx.py — Descarga pares de divisas FX para nlms-macro.

Todos los pares se expresan como (price_x, price_y) en USD por unidad de
divisa extranjera → misma escala, pct_change() = retorno en USD correcto.

Pares sin USD directo (cruces):
    AUD/NZD  — Oceánicas. Mismos socios comerciales (China), corr ~0.95
    EUR/CHF  — Europeas. SNB ancla históricamente el cruce
    NOK/SEK  — Nórdicas. Economías casi idénticas, corr ~0.97
    EUR/GBP  — Europeas. Ciclos sincronizados

Pares commodity currencies vs USD:
    AUD/CAD  — Ambas ligadas a materias primas (China vs oil sands)
    NOK/CAD  — Divisas del petróleo (Brent vs WTI)

Safe havens vs USD:
    JPY/CHF  — Ambas aprecian en risk-off, divergen en ciclos distintos

Majors vs USD (USD como contraparte directa):
    EUR/GBP  — ya incluido arriba
    AUD/NZD  — ya incluido arriba

Uso:
    python fetch_fx.py
    python fetch_fx.py --start 2005-01-01
"""

import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf

FX_PAIRS = {
    # ── Cruces (sin USD directo) ───────────────────────────────────────────────
    "AUD/NZD": {
        "x": "AUDUSD=X", "y": "NZDUSD=X",
        "fname": "aud_nzd",
        "desc": "AUD vs NZD — oceánicas, mismos socios comerciales",
    },
    "EUR/CHF": {
        "x": "CHFUSD=X", "y": "EURUSD=X",
        "fname": "eur_chf",
        "desc": "EUR vs CHF — europeas, SNB históricamente ancla el cruce",
    },
    "NOK/SEK": {
        "x": "NOKUSD=X", "y": "SEKUSD=X",
        "fname": "nok_sek",
        "desc": "NOK vs SEK — coronas nórdicas, economías casi idénticas",
    },
    "EUR/GBP": {
        "x": "GBPUSD=X", "y": "EURUSD=X",
        "fname": "eur_gbp",
        "desc": "EUR vs GBP — europeas, ciclos sincronizados",
    },
    # ── Commodity currencies (USD como referencia) ────────────────────────────
    "AUD/CAD": {
        "x": "CADUSD=X", "y": "AUDUSD=X",
        "fname": "aud_cad",
        "desc": "AUD vs CAD — commodity currencies, China vs oil sands",
    },
    "NOK/CAD": {
        "x": "CADUSD=X", "y": "NOKUSD=X",
        "fname": "nok_cad",
        "desc": "NOK vs CAD — divisas del petróleo, Brent vs WTI",
    },
    # ── Safe havens (USD como referencia) ─────────────────────────────────────
    "JPY/CHF": {
        "x": "CHFUSD=X", "y": "JPYUSD=X",
        "fname": "jpy_chf",
        "desc": "JPY vs CHF — safe havens, divergen en ciclos distintos",
    },
    # ── Majors vs USD ─────────────────────────────────────────────────────────
    "EUR/USD vs GBP/USD": {
        "x": "GBPUSD=X", "y": "EURUSD=X",
        "fname": "eur_gbp_usd",
        "desc": "EUR/USD vs GBP/USD — majors europeos cotizados en USD",
    },
    "AUD/USD vs NZD/USD": {
        "x": "NZDUSD=X", "y": "AUDUSD=X",
        "fname": "aud_nzd_usd",
        "desc": "AUD/USD vs NZD/USD — oceánicas cotizadas en USD (mismo par diferente base)",
    },
    "NOK/USD vs SEK/USD": {
        "x": "SEKUSD=X", "y": "NOKUSD=X",
        "fname": "nok_sek_usd",
        "desc": "NOK/USD vs SEK/USD — nórdicas cotizadas en USD",
    },
}

DEFAULT_START = "2005-01-01"
DEFAULT_END   = "2026-01-01"


def download_fx(ticker: str, start: str, end: str) -> pd.Series:
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    return df["Close"].squeeze().rename(ticker)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end",   default=DEFAULT_END)
    args = parser.parse_args()

    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)

    for name, info in FX_PAIRS.items():
        print(f"\n[{name}] {info['desc']}")
        try:
            sx = download_fx(info["x"], args.start, args.end)
            sy = download_fx(info["y"], args.start, args.end)

            # Algunos pares requieren invertir el tipo (ej: USDCHF → CHFUSD)
            if info.get("invert_x"):
                sx = 1.0 / sx
                sx.name = info["x"] + "_inv"

            df = pd.concat([sx, sy], axis=1).dropna()
            df.index.name = "date"
            df = df.reset_index()
            df.columns = ["date", "price_x", "price_y"]
            df["date"] = pd.to_datetime(df["date"]).dt.date

            out_path = out_dir / f"{info['fname']}.csv"
            df.to_csv(out_path, index=False)

            corr = df["price_x"].corr(df["price_y"])
            print(f"    Guardado: {out_path}  ({len(df)} filas, "
                  f"{df['date'].iloc[0]} → {df['date'].iloc[-1]}, corr={corr:.3f})")
        except Exception as e:
            print(f"    ERROR: {e}")

    print("\nListo.")


if __name__ == "__main__":
    main()
