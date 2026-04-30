"""
fetch_data.py — Descarga datos de ETFs de bonos del Tesoro US y pares macro.

Pares disponibles:
    IEI / TLT  — curva 3-7Y vs 20+Y (el spread de curva clásico)
    IEF / TLT  — curva 7-10Y vs 20+Y
    SHY / TLT  — curva 1-3Y vs 20+Y (máxima pendiente)
    IEI / IEF  — segmento corto-medio de la curva

Uso:
    python fetch_data.py                    # descarga todos los pares
    python fetch_data.py --pair IEI/TLT     # solo un par
    python fetch_data.py --start 2005-01-01 # desde fecha concreta
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

# ── Pares definidos ────────────────────────────────────────────────────────────
PAIRS = {
    "IEI/TLT": {
        "x": "IEI",   # 3-7 Year Treasury
        "y": "TLT",   # 20+ Year Treasury
        "desc": "Curva 3-7Y vs 20+Y (spread clásico)",
    },
    "IEF/TLT": {
        "x": "IEF",   # 7-10 Year Treasury
        "y": "TLT",   # 20+ Year Treasury
        "desc": "Curva 7-10Y vs 20+Y",
    },
    "SHY/TLT": {
        "x": "SHY",   # 1-3 Year Treasury
        "y": "TLT",   # 20+ Year Treasury
        "desc": "Curva corta 1-3Y vs 20+Y (máxima pendiente)",
    },
    "IEI/IEF": {
        "x": "IEI",   # 3-7 Year Treasury
        "y": "IEF",   # 7-10 Year Treasury
        "desc": "Segmento medio de la curva",
    },
}

DEFAULT_START = "2004-01-01"   # IEI lanzado en 2007, TLT en 2002, IEF en 2002
DEFAULT_END   = "2026-01-01"


def fetch_pair(ticker_x: str, ticker_y: str, start: str, end: str) -> pd.DataFrame:
    """
    Descarga precios ajustados de dos ETFs y los alinea en un DataFrame.

    Retorna DataFrame con columnas: date, price_x, price_y
    Solo incluye días donde ambos ETFs tienen datos (inner join).
    """
    print(f"    Descargando {ticker_x}...")
    data_x = yf.download(ticker_x, start=start, end=end, auto_adjust=True, progress=False)
    print(f"    Descargando {ticker_y}...")
    data_y = yf.download(ticker_y, start=start, end=end, auto_adjust=True, progress=False)

    # Extraer columna Close (yfinance ≥1.0 devuelve MultiIndex (col, ticker))
    close_x = data_x["Close"].squeeze().rename(ticker_x)
    close_y = data_y["Close"].squeeze().rename(ticker_y)

    # Alinear por fecha (inner join: solo días con ambos precios)
    df = pd.concat([close_x, close_y], axis=1).dropna()
    df.index.name = "date"
    df = df.reset_index()
    df.columns = ["date", "price_x", "price_y"]
    df["date"] = pd.to_datetime(df["date"]).dt.date

    return df


def main():
    parser = argparse.ArgumentParser(description="Descarga datos macro para nlms-macro")
    parser.add_argument("--pair",  default=None, help="Par a descargar (ej: IEI/TLT). Si no se especifica, descarga todos.")
    parser.add_argument("--start", default=DEFAULT_START, help=f"Fecha inicio (default: {DEFAULT_START})")
    parser.add_argument("--end",   default=DEFAULT_END,   help=f"Fecha fin   (default: {DEFAULT_END})")
    args = parser.parse_args()

    pairs_to_fetch = PAIRS if args.pair is None else {args.pair: PAIRS[args.pair]}

    if args.pair is not None and args.pair not in PAIRS:
        print(f"Par '{args.pair}' no reconocido. Opciones: {list(PAIRS.keys())}")
        sys.exit(1)

    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)

    for name, info in pairs_to_fetch.items():
        print(f"\n[{name}] {info['desc']}")
        try:
            df = fetch_pair(info["x"], info["y"], args.start, args.end)
            fname = name.replace("/", "_").lower() + ".csv"
            out_path = out_dir / fname
            df.to_csv(out_path, index=False)
            print(f"    Guardado: {out_path}  ({len(df)} filas, {df['date'].iloc[0]} → {df['date'].iloc[-1]})")
        except Exception as e:
            print(f"    ERROR: {e}")

    print("\nListo.")


if __name__ == "__main__":
    main()
